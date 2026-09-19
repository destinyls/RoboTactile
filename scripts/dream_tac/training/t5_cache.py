"""Generate the real eight-prompt T5 cache through pinned Dream-Tac code."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from scripts.n0_twam.hpu_training.data.contracts import (
    SOURCE_MANIFEST_NAME,
    TASK_PROMPTS,
    TASKS,
    sha256_file,
)
from scripts.n0_twam.hpu_training.data.manifest_io import (
    atomic_json_no_clobber,
    load_json_object,
    receipt_sha256,
    signed_payload,
)

from .contracts import (
    DATASET_DIR_NAME,
    DREAM_TAC_TRAINING_PROTOCOL,
    DREAM_TAC_UPSTREAM_COMMIT,
    PROMPT_MANIFEST_NAME,
    T5_ENCODER_RUNTIME_PROTOCOL,
    T5_REQUEST_NAME,
    build_t5_cache_request,
    validate_prompt_manifest,
)

T5_RECEIPT_NAME = "t5_cache_receipt.json"
T5_RECEIPT_SCHEMA_VERSION = 2


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_contracts(output_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    source = load_json_object(output_root / SOURCE_MANIFEST_NAME)
    source_sha = source.get("manifest_sha256")
    if not isinstance(source_sha, str):
        raise ValueError("source split manifest has no SHA256")
    prompts = load_json_object(output_root / PROMPT_MANIFEST_NAME)
    validate_prompt_manifest(prompts, source_manifest_sha256=source_sha)
    request = load_json_object(output_root / T5_REQUEST_NAME)
    expected = build_t5_cache_request(prompt_manifest=prompts, output_root=output_root)
    if request != expected:
        raise ValueError("Dream-Tac T5 request identity mismatch")
    return prompts, request


def _validate_embeddings(
    embeddings: Mapping[str, object], expected_keys: Sequence[str]
) -> list[dict[str, object]]:
    if set(embeddings) != set(expected_keys):
        raise ValueError("Dream-Tac T5 cache keys do not match all eight prompts")
    inventory: list[dict[str, object]] = []
    for prompt in expected_keys:
        tensor = embeddings[prompt]
        shape = tuple(getattr(tensor, "shape", ()))
        dtype = str(getattr(tensor, "dtype", ""))
        device = getattr(getattr(tensor, "device", None), "type", None)
        if shape != (1, 512, 1024) or dtype != "torch.bfloat16" or device != "cpu":
            raise ValueError("Dream-Tac T5 tensors must be CPU bfloat16 [1,512,1024]")
        tensor_api = cast(Any, tensor)
        try:
            finite = bool(tensor_api.isfinite().all().item())
            nan_count = int(tensor_api.isnan().sum().item())
            inf_count = int(tensor_api.isinf().sum().item())
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "Dream-Tac T5 tensors must expose finite checks"
            ) from error
        if not finite or nan_count != 0 or inf_count != 0:
            raise ValueError(f"Dream-Tac T5 tensor is non-finite for prompt: {prompt}")
        inventory.append(
            {
                "prompt": prompt,
                "shape": list(shape),
                "dtype": dtype,
                "device": device,
                "finite": finite,
                "nan_count": nan_count,
                "inf_count": inf_count,
            }
        )
    return inventory


def _expected_inventory(expected_keys: Sequence[str]) -> list[dict[str, object]]:
    return [
        {
            "prompt": prompt,
            "shape": [1, 512, 1024],
            "dtype": "torch.bfloat16",
            "device": "cpu",
            "finite": True,
            "nan_count": 0,
            "inf_count": 0,
        }
        for prompt in expected_keys
    ]


def _expected_encoder_runtime() -> dict[str, object]:
    return {
        "protocol_id": T5_ENCODER_RUNTIME_PROTOCOL,
        "device": "cpu",
        "model_compute_dtype": "torch.float32",
        "active_token_batching": True,
        "max_output_tokens": 512,
        "output_dtype": "torch.bfloat16",
        "padding_value": 0.0,
    }


def _generate_cpu_fp32_embeddings(
    expected_keys: Sequence[str],
) -> dict[str, object]:
    """Encode active prompt tokens in one CPU FP32 batch and zero-pad to 512."""

    torch_module = cast(Any, importlib.import_module("torch"))
    encoder_module = importlib.import_module(
        "cosmos_policy._src.predict2.inference.get_t5_emb"
    )
    encoder_class = cast(Any, encoder_module.CosmosT5TextEncoder)
    encoder = encoder_class(device="cpu", local_files_only=True)
    encoder.text_encoder.to(device="cpu", dtype=torch_module.float32)
    parameter = next(encoder.text_encoder.parameters())
    if parameter.device.type != "cpu" or parameter.dtype != torch_module.float32:
        raise RuntimeError("Dream-Tac T5 encoder must run as CPU float32")
    token_lengths = [
        len(
            encoder.tokenizer.encode(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=512,
            )
        )
        for prompt in expected_keys
    ]
    active_tokens = max(token_lengths, default=0)
    if active_tokens <= 0 or active_tokens > 512:
        raise ValueError("Dream-Tac T5 active token length is outside [1,512]")
    encoded_result = encoder.encode_prompts(
        list(expected_keys),
        max_length=active_tokens,
        return_mask=True,
    )
    if not isinstance(encoded_result, tuple) or len(encoded_result) != 2:
        raise RuntimeError("Dream-Tac T5 encoder did not return embeddings and mask")
    encoded, attention_mask = encoded_result
    if encoded.dtype != torch_module.float32 or encoded.device.type != "cpu":
        raise RuntimeError("Dream-Tac T5 activations must remain CPU float32")
    embeddings: dict[str, object] = {}
    for index, prompt in enumerate(expected_keys):
        valid_tokens = int(attention_mask[index].sum().item())
        if valid_tokens != token_lengths[index]:
            raise RuntimeError("Dream-Tac T5 tokenizer length drifted during encoding")
        padded = torch_module.zeros(
            (1, 512, 1024),
            dtype=torch_module.bfloat16,
            device="cpu",
        )
        padded[0, :valid_tokens].copy_(
            encoded[index, :valid_tokens].to(dtype=torch_module.bfloat16)
        )
        embeddings[prompt] = padded.contiguous()
    return embeddings


def _atomic_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite T5 cache: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            pickle.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_t5_cache(output_root: Path) -> dict[str, object]:
    """Verify cache bytes from the signed receipt without unsafe unpickling."""

    prompts, request = _load_contracts(output_root)
    receipt = load_json_object(output_root / T5_RECEIPT_NAME)
    receipt_sha256(receipt, field="t5_cache_receipt_sha256")
    cache_path = output_root / DATASET_DIR_NAME / "t5_embeddings.pkl"
    expected = {
        "schema_version": T5_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "protocol_id": DREAM_TAC_TRAINING_PROTOCOL,
        "dream_tac_commit": DREAM_TAC_UPSTREAM_COMMIT,
        "prompt_manifest_sha256": prompts["prompt_manifest_sha256"],
        "t5_request_sha256": request["t5_request_sha256"],
        "cache_sha256": sha256_file(cache_path),
        "cache_keys": [TASK_PROMPTS[task] for task in TASKS],
        "encoder_runtime": _expected_encoder_runtime(),
        "tensor_inventory": _expected_inventory([TASK_PROMPTS[task] for task in TASKS]),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Dream-Tac T5 cache receipt mismatch")
    return receipt


def generate_t5_cache(*, dream_tac_root: Path, output_root: Path) -> dict[str, object]:
    """Call the pinned upstream encoder and atomically publish its real tensors."""

    prompts, request = _load_contracts(output_root)
    receipt_path = output_root / T5_RECEIPT_NAME
    cache_path = output_root / DATASET_DIR_NAME / "t5_embeddings.pkl"
    if receipt_path.exists() or cache_path.exists():
        return validate_t5_cache(output_root)
    root = dream_tac_root.resolve(strict=True)
    if _git_head(root) != DREAM_TAC_UPSTREAM_COMMIT:
        raise ValueError("Dream-Tac checkout is not at the pinned commit")
    sys.path.insert(0, str(root))
    expected_keys = [TASK_PROMPTS[task] for task in TASKS]
    if request.get("encoder_runtime") != _expected_encoder_runtime():
        raise ValueError("Dream-Tac T5 encoder runtime contract mismatch")
    embeddings = _generate_cpu_fp32_embeddings(expected_keys)
    inventory = _validate_embeddings(embeddings, expected_keys)
    _atomic_pickle(cache_path, embeddings)
    unsigned: dict[str, object] = {
        "schema_version": T5_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "protocol_id": DREAM_TAC_TRAINING_PROTOCOL,
        "dream_tac_commit": DREAM_TAC_UPSTREAM_COMMIT,
        "prompt_manifest_sha256": prompts["prompt_manifest_sha256"],
        "t5_request_sha256": request["t5_request_sha256"],
        "cache_relative_path": f"{DATASET_DIR_NAME}/t5_embeddings.pkl",
        "cache_sha256": sha256_file(cache_path),
        "cache_keys": expected_keys,
        "encoder_runtime": _expected_encoder_runtime(),
        "tensor_inventory": inventory,
    }
    receipt = signed_payload(unsigned, field="t5_cache_receipt_sha256")
    atomic_json_no_clobber(receipt_path, receipt)
    return validate_t5_cache(output_root)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dream-tac-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify:
        result = validate_t5_cache(args.output_root)
    else:
        if args.dream_tac_root is None:
            raise ValueError("--dream-tac-root is required unless --verify is used")
        result = generate_t5_cache(
            dream_tac_root=args.dream_tac_root, output_root=args.output_root
        )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["generate_t5_cache", "main", "validate_t5_cache"]
