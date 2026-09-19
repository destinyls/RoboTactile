"""Opt-in bounded observations of actual N0 tensors, never causal A/B evidence."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import random
import re
import sys
import tempfile
import weakref
from collections.abc import Mapping
from functools import wraps
from numbers import Integral
from pathlib import Path
from typing import Any, cast

import numpy as np

PROBE_ID = "n0_actual_input_trace_v1"
_MARKER = "_robotactile_actual_input_trace_v1"


def _tensor_summary(value: Any, array_path: Path | None = None) -> dict[str, Any]:
    if value is None:
        return {"present": False}
    dtype = str(value.dtype)
    if isinstance(value, np.ndarray):
        array = value
    else:
        cpu = value.detach().cpu()
        try:
            array = cpu.numpy()
        except TypeError:
            # NumPy has no native torch.bfloat16 representation. Conversion is
            # exact for bfloat16 values and explicitly recorded in the hash view.
            array = cpu.float().numpy()
    array = np.ascontiguousarray(array)
    finite = np.isfinite(array)
    values = array[finite].astype(np.float64)
    header = json.dumps(
        {"shape": list(array.shape), "dtype": array.dtype.str}, sort_keys=True
    ).encode()
    digest = hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()
    summary: dict[str, Any] = {
        "present": True,
        "shape": list(array.shape),
        "dtype": dtype,
        "hash_dtype": array.dtype.str,
        "sha256": digest,
        "finite_count": int(finite.sum()),
        "element_count": int(array.size),
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
        "mean": float(values.mean()) if values.size else None,
        "l2": float(np.linalg.norm(values)) if values.size else None,
    }
    if array_path is not None:
        with array_path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        summary["array_file"] = {
            "relative_path": array_path.name,
            "file_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "source_dtype": dtype,
            "conversion": "bfloat16_to_float32_exact"
            if dtype in ("torch.bfloat16", "bfloat16")
            else "none",
        }
    return summary


def _diagnostic_context(obs: Mapping[str, Any]) -> dict[str, str] | None:
    context = obs.get("_robotactile_diagnostic_context")
    if context is None:
        return None
    if not isinstance(context, Mapping):
        raise ValueError("diagnostic context must be a mapping")
    clean = {}
    for name in ("source_sha256", "protocol_sha256", "condition", "branch"):
        if name not in context:
            continue
        value = context[name]
        pattern = (
            r"[0-9a-fA-F]{64}"
            if name.endswith("sha256")
            else r"[A-Za-z0-9_.:/+\-]{1,128}"
        )
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
            raise ValueError(f"invalid diagnostic context field: {name}")
        clean[name] = value
    return clean


def _rng_fingerprint() -> dict[str, Any]:
    def digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    numpy_state = cast(tuple[Any, ...], np.random.get_state())
    domains: dict[str, Any] = {
        "python": digest(random.getstate()),
        "numpy": digest([numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]),
        "torch_cpu": None,
        "torch_cuda": None,
    }
    missing: list[str] = []
    # An active N0 server already imports torch. Do not import it here, since
    # importing a new runtime could itself alter process state.
    torch = sys.modules.get("torch")
    if torch is None:
        missing.extend(("torch_cpu_unavailable", "torch_cuda_unavailable"))
    else:
        try:
            domains["torch_cpu"] = _tensor_summary(torch.get_rng_state())["sha256"]
        except (AttributeError, RuntimeError):
            missing.append("torch_cpu_unavailable")
        try:
            if torch.cuda.is_initialized():
                domains["torch_cuda"] = [
                    _tensor_summary(state)["sha256"]
                    for state in torch.cuda.get_rng_state_all()
                ]
            else:
                missing.append("torch_cuda_not_initialized")
        except (AttributeError, RuntimeError):
            missing.append("torch_cuda_unavailable")
    return {
        "rng_after_reset_sha256": digest(domains) if not missing else None,
        "rng_after_reset_domains": domains,
        "rng_after_reset_missing": missing,
    }


def install_n0_input_probe(
    root: Path | str,
    limit: int = 32,
    *,
    server_type: type | None = None,
    capture_arrays: bool = False,
) -> str:
    """Wrap actual calls without changing return objects, model caches, or RNG.

    File writes and device-to-CPU reads add latency. These traces are diagnostic
    closed-loop observations and cannot establish isolated causal action effects.
    The caller must reject this overlay for source-bound paper serving.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 32:
        raise ValueError("diagnostic input trace limit must be in [1, 32]")
    if not isinstance(capture_arrays, bool):
        raise ValueError("capture_arrays must be a boolean")
    if server_type is None:
        module = importlib.import_module("n0_twam.n0_twam_server")
        server_type = module.TWAM_Server
    if getattr(server_type, _MARKER, False):
        raise ValueError("N0 input probe is already installed")
    names = ("infer", "_build_tactile_tensor", "_encode_tactile_obs")
    if any(not callable(getattr(server_type, name, None)) for name in names):
        raise ValueError("N0 server lacks required input probe hooks")
    original_infer, original_build, original_encode = (
        getattr(server_type, name) for name in names
    )
    trace_root = Path(root)
    trace_root.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="n0-input-probe-", dir=trace_root))
    states: weakref.WeakKeyDictionary[Any, dict[str, Any]] = weakref.WeakKeyDictionary()
    instance_ids = itertools.count()

    def state_for(server: Any) -> dict[str, Any]:
        # Store only diagnostic scalars; never retain tensors or model objects.
        if server not in states:
            states[server] = {
                "instance": next(instance_ids),
                "episode": 0,
                "seed": None,
                "count": 0,
                "infer_call": 0,
                "mode": "direct_encode",
                "pending": None,
                "diagnostic_context": None,
                "rng_after_reset": None,
            }
        return states[server]

    @wraps(original_infer)
    def infer(server: Any, obs: Mapping[str, Any]) -> Any:
        context = _diagnostic_context(obs) if obs.get("reset", False) else None
        state = state_for(server)
        state["infer_call"] += 1
        state["mode"] = (
            "reset"
            if obs.get("reset", False)
            else "compute_kv_cache"
            if obs.get("compute_kv_cache", False)
            else "infer"
        )
        result = original_infer(server, obs)
        if obs.get("reset", False):
            seed = obs.get("seed")
            state.update(
                episode=state["episode"] + 1,
                seed=int(seed)
                if isinstance(seed, Integral) and not isinstance(seed, bool)
                else None,
                count=0,
                infer_call=0,
                pending=None,
                diagnostic_context=context,
                rng_after_reset=_rng_fingerprint()
                if capture_arrays and context is not None
                else None,
            )
        return result

    @wraps(original_build)
    def build(server: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_build(server, *args, **kwargs)
        state = state_for(server)
        pending = state["pending"]
        if pending is not None:
            pending["build_call_count"] += 1
            # Pinned encode calls build once. Keep a bounded sample if a future
            # implementation calls it repeatedly, with the total count explicit.
            if len(pending["tactile_tensor_outputs"]) < 4:
                index = len(pending["tactile_tensor_outputs"])
                pending["tactile_tensor_outputs"].append(
                    _tensor_summary(
                        result,
                        array_path(state, state["count"] - 1, f"build-{index}")
                        if capture_arrays
                        else None,
                    )
                )
        return result

    def array_path(state: dict[str, Any], call: int, name: str) -> Path:
        return run_root / (
            f"server-{state['instance']:04d}-episode-{state['episode']:06d}"
            f"-encode-{call:02d}-{name}.npy"
        )

    @wraps(original_encode)
    def encode(server: Any, *args: Any, **kwargs: Any) -> Any:
        state = state_for(server)
        if state["count"] >= limit:
            return original_encode(server, *args, **kwargs)
        call = state["count"]
        state["count"] += 1
        pending: dict[str, Any] = {"build_call_count": 0, "tactile_tensor_outputs": []}
        state["pending"] = pending
        try:
            result = original_encode(server, *args, **kwargs)
        finally:
            state["pending"] = None
        config = server.job_config
        record = {
            "schema_id": PROBE_ID,
            "evidence_scope": "closed_loop_actual_tensor_trace_not_isolated_causal_effect",
            "episode_index": state["episode"],
            "episode_seed": state["seed"],
            "encode_call_index": call,
            "infer_call_index": state["infer_call"],
            "call_mode": state["mode"],
            "frame_st_id": int(server.frame_st_id)
            if isinstance(getattr(server, "frame_st_id", None), Integral)
            else None,
            "local_tactile_mode": str(getattr(config, "local_tactile_mode", "current")),
            "tactile_global_zero": bool(getattr(config, "tactile_global_zero", False)),
            **pending,
            "encoded_present": result is not None,
            "latents": {
                name: _tensor_summary(
                    result.get(name),
                    array_path(state, call, name) if capture_arrays else None,
                )
                for name in ("tactile_global_latent", "tactile_local_latent")
            }
            if isinstance(result, Mapping)
            else {},
        }
        if state["diagnostic_context"] is not None:
            record["diagnostic_context"] = dict(state["diagnostic_context"])
        if state["rng_after_reset"] is not None:
            record.update(state["rng_after_reset"])
        path = (
            run_root
            / f"server-{state['instance']:04d}-episode-{state['episode']:06d}-encode-{call:02d}.json"
        )
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return result

    server_class = cast(Any, server_type)
    server_class.infer = infer
    server_class._build_tactile_tensor = build
    server_class._encode_tactile_obs = encode
    setattr(server_type, _MARKER, True)
    return PROBE_ID
