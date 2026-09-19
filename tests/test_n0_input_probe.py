"""CPU-only probes verify passive tracing, not model inference success."""

import hashlib
import importlib.util
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from robotactile_benchmark.integrations.n0_twam.input_probe import (
    install_n0_input_probe,
)


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)
        self.dtype = self.data.dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


def fake_server_type():
    class Server:
        def __init__(self):
            self.job_config = types.SimpleNamespace(
                local_tactile_mode="current", tactile_global_zero=False
            )
            self.frame_st_id = 7
            self.build_calls = self.encode_calls = self.infer_calls = 0
            self.tensor = FakeTensor(np.arange(24).reshape(2, 3, 1, 2, 2))
            self.global_latent = FakeTensor([1, 2, 3])
            self.local_latent = FakeTensor([-1, 0, 1])
            self.output = {
                "tactile_global_latent": self.global_latent,
                "tactile_local_latent": self.local_latent,
            }
            self.reset_output = {}
            self.cache = []

        def _build_tactile_tensor(self, obs):
            self.build_calls += 1
            return self.tensor

        def _encode_tactile_obs(self, obs):
            self.encode_calls += 1
            assert self._build_tactile_tensor(obs) is self.tensor
            self.cache.append(self.encode_calls)
            return self.output

        def infer(self, obs):
            self.infer_calls += 1
            if obs.get("reset"):
                self.cache.clear()
                return self.reset_output
            return self._encode_tactile_obs(obs)

    return Server


def test_trace_preserves_objects_counts_rng_cache_and_limits(tmp_path):
    server_type = fake_server_type()
    install_n0_input_probe(tmp_path, limit=2, server_type=server_type)
    server = server_type()
    random_state, numpy_state = random.getstate(), np.random.get_state()
    assert (
        server.infer({"reset": True, "seed": 71, "prompt": "SECRET_TEXT"})
        is server.reset_output
    )
    for _ in range(5):
        assert (
            server.infer({"compute_kv_cache": True, "secret": "SECRET_VALUE"})
            is server.output
        )
    assert server.build_calls == server.encode_calls == 5
    assert server.cache == [1, 2, 3, 4, 5]
    assert server._build_tactile_tensor({}) is server.tensor
    assert random.getstate() == random_state
    actual_state = np.random.get_state()
    assert actual_state[0] == numpy_state[0]
    assert np.array_equal(actual_state[1], numpy_state[1])
    assert actual_state[2:] == numpy_state[2:]
    files = list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.npy"))
    assert len(files) == 2
    data = json.loads(files[0].read_text())
    assert data["episode_seed"] == 71
    assert data["call_mode"] == "compute_kv_cache"
    assert data["build_call_count"] == 1
    assert data["tactile_tensor_outputs"][0]["shape"] == [2, 3, 1, 2, 2]
    assert data["tactile_tensor_outputs"][0]["max"] == 23
    assert data["latents"]["tactile_local_latent"]["min"] == -1
    assert data["local_tactile_mode"] == "current"
    assert data["tactile_global_zero"] is False
    assert "diagnostic_context" not in data
    assert "SECRET" not in "".join(path.read_text() for path in files)
    hashes = {path: path.read_bytes() for path in files}
    server.infer({"reset": True, "seed": 72})
    server.infer({})
    assert len(list(tmp_path.rglob("*.json"))) == 3
    assert all(path.read_bytes() == payload for path, payload in hashes.items())


def test_separate_installations_do_not_clobber(tmp_path):
    for _ in range(2):
        server_type = fake_server_type()
        install_n0_input_probe(tmp_path, server_type=server_type)
        server_type().infer({})
        with pytest.raises(ValueError, match="already installed"):
            install_n0_input_probe(tmp_path, server_type=server_type)
    assert len(list(tmp_path.rglob("*.json"))) == 2
    for bad in (0, 33, True, 1.5):
        with pytest.raises(ValueError, match="limit"):
            install_n0_input_probe(tmp_path, bad, server_type=fake_server_type())


def load_launcher(monkeypatch):
    calls = []
    config = types.ModuleType("n0_twam.configs")
    config.TWAM_CONFIGS = {
        "multitask_server": types.SimpleNamespace(enable_offload=False)
    }
    server = types.ModuleType("n0_twam.n0_twam_server")
    server.init_logger = lambda: None
    server.run = calls.append
    monkeypatch.setitem(sys.modules, "n0_twam.configs", config)
    monkeypatch.setitem(sys.modules, "n0_twam.n0_twam_server", server)
    path = Path(__file__).parents[1] / "scripts/n0_twam/serve_official.py"
    spec = importlib.util.spec_from_file_location("input_probe_test_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def test_launcher_defaults_off_and_rejects_source_bound(monkeypatch, tmp_path):
    module, calls = load_launcher(monkeypatch)
    args = ["--port", "9000", "--save-root", str(tmp_path)]
    assert module.main(args) == 0
    assert len(calls) == 1
    assert not list(tmp_path.iterdir())
    for option in module._SOURCE_BOUND_ARGUMENTS:
        args.extend(["--" + option.replace("_", "-"), "stub"])
    with pytest.raises(ValueError, match="forbids diagnostic"):
        module.main(args + ["--diagnostic-input-trace-root", str(tmp_path)])
    with pytest.raises(ValueError, match="forbids diagnostic"):
        module.main(args + ["--diagnostic-input-capture-arrays"])
    assert len(calls) == 1


def test_arrays_are_exact_safe_loadable_bounded_and_passive(tmp_path):
    server_type = fake_server_type()
    original_encode = server_type._encode_tactile_obs

    def repeated_builds(self, obs):
        for _ in range(6):
            self._build_tactile_tensor(obs)
        return original_encode(self, obs)

    server_type._encode_tactile_obs = repeated_builds
    install_n0_input_probe(tmp_path, server_type=server_type, capture_arrays=True)
    server = server_type()
    context = {
        "source_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "condition": "clean",
        "branch": "replicate-0",
        "prompt": "SECRET_PROMPT",
        "api_key": "SECRET_API_KEY",
    }
    obs = {"reset": True, "seed": 71, "_robotactile_diagnostic_context": context}
    assert server.infer(obs) is server.reset_output
    random_state, numpy_state = random.getstate(), np.random.get_state()
    for _ in range(35):
        assert server.infer({}) is server.output
    assert server.encode_calls == 35
    assert server.build_calls == 35 * 7
    assert server.cache == list(range(1, 36))
    assert random.getstate() == random_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    traces = sorted(tmp_path.rglob("*.json"))
    assert len(traces) == 32
    assert len(list(tmp_path.rglob("*.npy"))) == 32 * 6
    for trace in traces:
        data = json.loads(trace.read_text())
        assert data["diagnostic_context"] == {
            name: context[name]
            for name in ("source_sha256", "protocol_sha256", "condition", "branch")
        }
        assert "SECRET" not in trace.read_text()
        assert data["build_call_count"] == 7
        summaries = [
            *((item, server.tensor.data) for item in data["tactile_tensor_outputs"]),
            (data["latents"]["tactile_global_latent"], server.global_latent.data),
            (data["latents"]["tactile_local_latent"], server.local_latent.data),
        ]
        for summary, expected in summaries:
            descriptor = summary["array_file"]
            relative = Path(descriptor["relative_path"])
            assert not relative.is_absolute() and relative.parent == Path(".")
            path = trace.parent / relative
            actual = np.load(path, allow_pickle=False)
            np.testing.assert_array_equal(actual, expected)
            assert (
                descriptor["file_sha256"]
                == hashlib.sha256(path.read_bytes()).hexdigest()
            )
            assert descriptor["shape"] == list(actual.shape)
            assert descriptor["dtype"] == actual.dtype.str
            assert descriptor["conversion"] == "none"
    server.infer({"reset": True})
    server.infer({})
    last = next(tmp_path.rglob("*episode-000002-encode-00.json"))
    assert "diagnostic_context" not in json.loads(last.read_text())
    assert obs["_robotactile_diagnostic_context"] is context


@pytest.mark.parametrize(
    "context",
    [
        [],
        {"condition": 1},
        {"branch": "x" * 129},
        {"source_sha256": "no"},
        {"protocol_sha256": False},
        {"branch": "line\nbreak"},
    ],
)
def test_invalid_diagnostic_context_is_rejected_before_model_call(tmp_path, context):
    server_type = fake_server_type()
    install_n0_input_probe(tmp_path, server_type=server_type)
    server = server_type()
    with pytest.raises(ValueError, match="diagnostic context"):
        server.infer({"reset": True, "_robotactile_diagnostic_context": context})
    assert server.infer_calls == 0
    assert not list(tmp_path.rglob("*.json"))


def test_bfloat16_is_saved_as_exact_float32(tmp_path):
    class BFloatTensor(FakeTensor):
        def __init__(self, data):
            super().__init__(data)
            self.dtype = "torch.bfloat16"

        def numpy(self):
            raise TypeError("unsupported ScalarType BFloat16")

        def float(self):
            return FakeTensor(self.data)

    server_type = fake_server_type()
    install_n0_input_probe(tmp_path, server_type=server_type, capture_arrays=True)
    server = server_type()
    server.tensor = BFloatTensor([0.5, 1.0, -2.0])
    server.infer({})
    trace = next(tmp_path.rglob("*.json"))
    descriptor = json.loads(trace.read_text())["tactile_tensor_outputs"][0][
        "array_file"
    ]
    assert descriptor["conversion"] == "bfloat16_to_float32_exact"
    assert descriptor["source_dtype"] == "torch.bfloat16"
    assert descriptor["dtype"] == "<f4"
    np.testing.assert_array_equal(
        np.load(trace.parent / descriptor["relative_path"], allow_pickle=False),
        server.tensor.data,
    )


def test_launcher_capture_arrays_requires_trace_and_passes_flag(monkeypatch, tmp_path):
    module, calls = load_launcher(monkeypatch)
    args = ["--port", "9000", "--save-root", str(tmp_path)]
    with pytest.raises(ValueError, match="require.*trace-root"):
        module.main(args + ["--diagnostic-input-capture-arrays"])
    import robotactile_benchmark.integrations.n0_twam.input_probe as probe

    installed = []
    monkeypatch.setattr(
        probe, "install_n0_input_probe", lambda *a, **kw: installed.append((a, kw))
    )
    module.main(
        args
        + [
            "--diagnostic-input-trace-root",
            str(tmp_path),
            "--diagnostic-input-capture-arrays",
        ]
    )
    assert installed == [((tmp_path, 32), {"capture_arrays": True})]
    assert len(calls) == 1


@pytest.mark.parametrize("torch_present", [False, True])
def test_reset_rng_fingerprint_is_read_only_and_marks_missing(
    tmp_path, monkeypatch, torch_present
):
    fake_torch = types.SimpleNamespace(
        get_rng_state=lambda: FakeTensor([1, 2]),
        cuda=types.SimpleNamespace(
            is_initialized=lambda: True,
            get_rng_state_all=lambda: [FakeTensor([3, 4])],
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch if torch_present else None)
    server_type = fake_server_type()
    install_n0_input_probe(tmp_path, server_type=server_type, capture_arrays=True)
    server = server_type()
    before = random.getstate(), np.random.get_state()
    server.infer({"reset": True, "_robotactile_diagnostic_context": {"branch": "a"}})
    server.infer({})
    first = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert bool(first["rng_after_reset_sha256"]) is torch_present
    assert bool(first["rng_after_reset_missing"]) is not torch_present
    assert first["rng_after_reset_domains"]["python"]
    assert first["rng_after_reset_domains"]["numpy"]
    assert before[0] == random.getstate()
    assert np.array_equal(before[1][1], np.random.get_state()[1])
    server.infer({"reset": True, "_robotactile_diagnostic_context": {"branch": "b"}})
    server.infer({})
    second = json.loads(
        next(tmp_path.rglob("*episode-000002-encode-00.json")).read_text()
    )
    assert first["rng_after_reset_domains"] == second["rng_after_reset_domains"]
