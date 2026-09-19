from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/ftp1_policy/serve_official.py"


def _load_server() -> Any:
    name = "robotactile_ftp1_server_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


server = _load_server()


class FakeWrapper:
    def __init__(self) -> None:
        self.model_config = SimpleNamespace(use_tactile_input=True)
        self.calls: list[dict[str, object]] = []

    @staticmethod
    def get_state_dim() -> int:
        return 120

    @staticmethod
    def get_action_dim() -> int:
        return 120

    @staticmethod
    def get_action_horizon() -> int:
        return 32

    def infer(self, **kwargs: object) -> np.ndarray:
        call_index = len(self.calls)
        self.calls.append(dict(kwargs))
        chunk = np.zeros((32, 120), dtype=np.float32)
        for index in range(32):
            chunk[index, 9:16] = call_index * 100 + index
            chunk[index, 44] = np.float32(0.2 + call_index * 0.5)
        return chunk


def _engine(wrapper: FakeWrapper, task_id: str = "insert_hole") -> Any:
    return server.FTP1PolicyEngine(
        wrapper,
        task_id=task_id,
        source_commit=server.PINNED_SOURCE_COMMIT,
        checkpoint_sha256="a" * 64,
        serve_bundle_sha256="b" * 64,
    )


def _image(value: int) -> np.ndarray:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[..., 0] = value
    image[..., 1] = value + 1
    image[..., 2] = value + 2
    return image


def _request(
    task_id: str,
    qpos8: np.ndarray,
    *,
    include_wrist: bool = False,
) -> dict[str, object]:
    contract = server.TASK_CONTRACTS[task_id]
    request: dict[str, object] = {
        "cmd": "predict",
        "task_id": task_id,
        "prompt": contract.prompt,
        "top": server.encode_ndarray(_image(10)),
        "left": server.encode_ndarray(_image(20)),
        "right": server.encode_ndarray(_image(30)),
        "qpos8": server.encode_ndarray(qpos8),
    }
    if include_wrist:
        request["wrist"] = server.encode_ndarray(_image(40))
    return request


def test_binary_ndarray_codec_is_exact_and_rejects_malformed_payloads() -> None:
    original = np.arange(8, dtype=np.float32)
    encoded = server.encode_ndarray(original)
    assert set(encoded) == {"__ndarray__", "dtype", "shape", "data"}
    assert isinstance(encoded["data"], bytes)

    decoded = server.decode_ndarray(
        encoded,
        name="qpos8",
        dtype=np.dtype(np.float32),
        shape=(8,),
    )
    np.testing.assert_array_equal(decoded, original)
    assert decoded.flags.writeable is False

    malformed = dict(encoded)
    malformed["data"] = bytes(encoded["data"])[:-1]
    with pytest.raises(ValueError, match="payload length"):
        server.decode_ndarray(
            malformed,
            name="qpos8",
            dtype=np.dtype(np.float32),
            shape=(8,),
        )


def test_engine_reproduces_state_tactile_camera_and_passthrough_contract() -> None:
    wrapper = FakeWrapper()
    engine = _engine(wrapper)
    qpos8 = np.linspace(0.0, 0.7, 8, dtype=np.float32)

    action = engine.predict(_request("insert_hole", qpos8))

    np.testing.assert_allclose(action[:7], qpos8[:7] + 1.0)
    assert action[7] == pytest.approx(0.2)
    call = wrapper.calls[0]
    images = call["images"]
    assert isinstance(images, dict)
    assert set(images) == {"camera_ego_rgb_0"}
    np.testing.assert_array_equal(images["camera_ego_rgb_0"][0, 0], [10, 11, 12])
    state = call["state"]
    assert isinstance(state, np.ndarray)
    assert state.shape == (1, 120)
    np.testing.assert_array_equal(state[0, 9:16], qpos8[:7])
    assert state[0, 44] == qpos8[7]
    tactile = call["tactiles"]
    assert isinstance(tactile, dict)
    assert tactile[server.TACTILE_KEY].shape == (1, 2, 224, 224, 3)
    np.testing.assert_array_equal(tactile[server.TACTILE_KEY][0, 0, 0, 0], [20, 21, 22])
    np.testing.assert_array_equal(tactile[server.TACTILE_KEY][0, 1, 0, 0], [30, 31, 32])


def test_engine_routes_wrist_only_for_official_all_camera_tasks() -> None:
    wrapper = FakeWrapper()
    engine = _engine(wrapper, task_id="lift_can")
    qpos8 = np.zeros(8, dtype=np.float32)

    engine.predict(_request("lift_can", qpos8, include_wrist=True))

    images = wrapper.calls[0]["images"]
    assert isinstance(images, dict)
    assert set(images) == {"camera_ego_rgb_0", "right_wrist_camera_rgb_0"}
    np.testing.assert_array_equal(
        images["right_wrist_camera_rgb_0"][0, 0], [40, 41, 42]
    )


def test_temporal_ensemble_skips_chunk_zero_and_applies_mix_actions() -> None:
    wrapper = FakeWrapper()
    engine = _engine(wrapper)
    first_qpos = np.arange(8, dtype=np.float32)
    second_qpos = first_qpos + 10.0

    first = engine.predict(_request("insert_hole", first_qpos))
    second = engine.predict(_request("insert_hole", second_qpos))

    np.testing.assert_allclose(first[:7], first_qpos[:7] + 1.0)
    assert first[7] == pytest.approx(0.2)
    old_prediction = first_qpos[:7] + 2.0
    new_prediction = second_qpos[:7] + 101.0
    weights = np.exp(-0.01 * np.arange(2, dtype=np.float32))
    weights /= weights.sum()
    np.testing.assert_allclose(
        second[:7],
        old_prediction * weights[0] + new_prediction * weights[1],
        rtol=1e-6,
    )
    assert second[7] == pytest.approx(0.2 * weights[0] + 0.7 * weights[1])


def test_handler_requires_seeded_reset_and_returns_exact_contract_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = FakeWrapper()
    engine = _engine(wrapper)
    handler = server.FTP1RequestHandler(engine)
    contract = server.TASK_CONTRACTS["insert_hole"]
    identity = {
        "task_id": "insert_hole",
        "prompt": contract.prompt,
    }

    health = handler.dispatch({"cmd": "health"})
    assert set(health) == {"status", "metadata"}
    with pytest.raises(RuntimeError, match="requires reset"):
        handler.dispatch(_request("insert_hole", np.zeros(8, dtype=np.float32)))
    seeded: list[int] = []
    monkeypatch.setattr(server, "_seed_policy_randomness", seeded.append)
    reset = handler.dispatch({"cmd": "reset", "seed": 17, **identity})
    assert set(reset) == {"status", "metadata", "seed"}
    assert reset["seed"] == 17
    assert seeded == [17]
    response = handler.dispatch(_request("insert_hole", np.zeros(8, dtype=np.float32)))
    assert set(response) == {"status", "action", "metadata"}
    assert response["status"] == "ok"
    metadata = response["metadata"]
    assert isinstance(metadata, dict)
    assert metadata == {
        "protocol_version": "robotactile-ftp1-zmq-v2",
        "randomness_contract": "episode_exogenous_seed_v1",
        "model": "ftp1_policy",
        "source_commit": server.PINNED_SOURCE_COMMIT,
        "checkpoint_sha256": "a" * 64,
        "serve_bundle_sha256": "b" * 64,
        "task_id": "insert_hole",
        "prompt": contract.prompt,
        "camera_keys": ["camera_ego_rgb_0"],
        "action_rep": "absolute",
        "output_action_dim": 8,
        "model_action_dim": 120,
        "action_horizon": 32,
        "chunk_index_offset": 1,
        "chunk_first_n": 20,
        "temporal_ensemble_k": 0.01,
        "use_tactile": True,
        "color_contract": "upstream_passthrough_v1",
    }
    action = server.decode_ndarray(
        response["action"],
        name="action",
        dtype=np.dtype(np.float32),
        shape=(8,),
    )
    assert action.shape == (8,)


def test_policy_randomness_seed_covers_python_numpy_and_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int]] = []
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: events.append(("torch", seed)),
        cuda=SimpleNamespace(
            manual_seed_all=lambda seed: events.append(("cuda", seed))
        ),
    )
    monkeypatch.setattr(
        server.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else importlib.import_module(name),
    )
    monkeypatch.setattr(
        server.random,
        "seed",
        lambda seed: events.append(("python", seed)),
    )
    monkeypatch.setattr(
        server.np.random,
        "seed",
        lambda seed: events.append(("numpy", seed)),
    )

    server._seed_policy_randomness((1 << 32) + 7)

    assert events == [
        ("python", (1 << 32) + 7),
        ("numpy", 7),
        ("torch", (1 << 32) + 7),
        ("cuda", (1 << 32) + 7),
    ]


def test_engine_rejects_non_tactile_or_wrong_shape_checkpoint() -> None:
    no_tactile = FakeWrapper()
    no_tactile.model_config.use_tactile_input = False
    with pytest.raises(ValueError, match="requires tactile"):
        _engine(no_tactile)

    wrong_horizon = FakeWrapper()
    wrong_horizon.get_action_horizon = lambda: 31  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="horizon"):
        _engine(wrong_horizon)


def test_server_exposes_only_the_six_published_univtac_checkpoints() -> None:
    assert set(server.TASK_CONTRACTS) == {
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
        "pull_out_key",
        "put_bottle_in_shelf",
    }


def test_installer_is_pinned_and_confined_to_its_own_runtime() -> None:
    installer = (ROOT / "scripts/ftp1_policy/install_official_runtime.sh").read_text(
        encoding="utf-8"
    )
    requirements = (ROOT / "scripts/ftp1_policy/requirements-runtime.txt").read_text(
        encoding="utf-8"
    )

    assert 'RUNTIME_ROOT="$DEPLOY_ROOT/runtime/ftp1-policy"' in installer
    assert '"python=3.11.11"' in installer
    assert server.PINNED_SOURCE_COMMIT in installer
    assert '"system_python_modified=false"' in installer
    assert '"system_cuda_modified=false"' in installer
    assert '"isaac_runtime_modified=false"' in installer
    assert '"n0_runtime_modified=false"' in installer
    assert "GIT_LFS_SKIP_SMUDGE=1" in installer
    assert (
        "ROBOTACTILE_FTP1_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple"
    ) in installer
    assert "https://*) ;;" in installer
    assert "unset PIP_EXTRA_INDEX_URL" in installer
    assert installer.count("--no-compile") == 3
    assert 'PYTORCH_CU128_INDEX_URL="https://download.pytorch.org/whl/cu128"' in (
        installer
    )
    assert '"cu128_bundle_package_count=17"' in installer
    assert '--index-url "$PYTORCH_CU128_INDEX_URL"' in installer
    assert "--no-deps" in installer
    assert "torch.version.cuda; arch=sorted(torch.cuda.get_arch_list())" in installer
    assert '"sm_120" in arch or "compute_120" in arch' in installer
    assert 'torch.ones((1,),device="cuda")' in installer
    assert '"system_cuda_modified=false"' in installer
    assert "provision_openpi_tokenizer.py" in installer
    assert 'OPENPI_DATA_HOME="$DEPLOY_ROOT/artifacts/openpi-data/ftp1-policy"' in (
        installer
    )
    assert (
        'TOKENIZER_SHA256="8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"'
        in installer
    )
    cu128_bundle = (
        "torch==2.7.1+cu128",
        "torchvision==0.22.1+cu128",
        "triton==3.3.1",
        "nvidia-cuda-nvrtc-cu12==12.8.61",
        "nvidia-cuda-runtime-cu12==12.8.57",
        "nvidia-cuda-cupti-cu12==12.8.57",
        "nvidia-cudnn-cu12==9.7.1.26",
        "nvidia-cublas-cu12==12.8.3.14",
        "nvidia-cufft-cu12==11.3.3.41",
        "nvidia-curand-cu12==10.3.9.55",
        "nvidia-cusolver-cu12==11.7.2.55",
        "nvidia-cusparse-cu12==12.5.7.53",
        "nvidia-cusparselt-cu12==0.6.3",
        "nvidia-nccl-cu12==2.26.2",
        "nvidia-nvtx-cu12==12.8.55",
        "nvidia-nvjitlink-cu12==12.8.61",
        "nvidia-cufile-cu12==1.13.0.11",
    )
    for frozen in cu128_bundle:
        assert frozen in installer
        assert frozen in requirements
    for frozen in (
        "av==16.0.1",
        "gcsfs==0.8.0",
        "ml-dtypes==0.4.1",
        "msgpack==1.1.2",
        "pytest==9.0.3",
        "pyzmq==27.1.0",
        "tensorstore==0.1.74",
    ):
        assert frozen in requirements
    assert '"pytest==9.0.3"' in installer
    assert "--resume-existing-runtime" in installer
    assert '[ ! -L "$RUNTIME_ROOT" ]' in installer
    assert "RESUMING_RUNTIME=true" in installer
    assert 'ENVIRONMENT_BACKEND="resumed_existing"' in installer


def _resume_installer_fixture(
    tmp_path: Path,
    *,
    bootstrap_identity: str = "25.3|80.9.0|0.45.1",
) -> tuple[Path, Path, dict[str, str]]:
    fixture_root = tmp_path / "repository"
    script_dir = fixture_root / "scripts/ftp1_policy"
    common_dir = fixture_root / "scripts/live_univtac"
    script_dir.mkdir(parents=True)
    common_dir.mkdir(parents=True)
    installer = script_dir / "install_official_runtime.sh"
    installer.write_text(
        (ROOT / "scripts/ftp1_policy/install_official_runtime.sh").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    installer.chmod(0o755)
    (script_dir / "requirements-runtime.txt").write_text("", encoding="utf-8")
    (script_dir / "provision_openpi_tokenizer.py").write_text(
        """from __future__ import annotations
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True, type=Path)
root = parser.parse_args().root
data_home = root / "artifacts/openpi-data/ftp1-policy"
target = data_home / "big_vision/paligemma_tokenizer.model"
receipt = root / "artifacts/deployment/ftp1_policy_openpi_tokenizer_provision.json"
target.parent.mkdir(parents=True, exist_ok=True)
receipt.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(b"fixture")
receipt.write_text(json.dumps({"status": "provisioned"}), encoding="utf-8")
print(data_home)
""",
        encoding="utf-8",
    )
    (common_dir / "common.sh").write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
ROBOTACTILE_REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
ROBOTACTILE_DEPLOY_ROOT=""
die() { printf 'ERROR: %s\\n' "$*" >&2; exit 2; }
info() { printf 'INFO: %s\\n' "$*" >&2; }
default_deployment_root() { printf '%s\\n' "$ROBOTACTILE_REPOSITORY_ROOT/deployment"; }
initialize_layout() {
  ROBOTACTILE_DEPLOY_ROOT="$1"; export ROBOTACTILE_DEPLOY_ROOT
  mkdir -p "$1/sources" "$1/artifacts/deployment" "$1/logs" "$1/runtime/locks"
}
require_command() { :; }
acquire_lock() { :; }
release_lock() { :; }
resolve_external_pin() {
  if [ "$2" = commit_sha ]; then
    printf '%s\\n' 89fa681d6c014cce28300946b7526db808e0b1c1
  else
    printf '%s\\n' https://example.invalid/ftp1-policy.git
  fi
}
ensure_pinned_git_source() {
  mkdir -p "$(dirname "$3/$4")" "$3/packages/openpi-client" \
    "$3/src/openpi/models_pytorch/transformers_replace"
  : > "$3/$4"
  : > "$3/src/openpi/models_pytorch/transformers_replace/overlay.py"
}
receipt_matches() { return 1; }
new_log_path() { local path="$ROBOTACTILE_DEPLOY_ROOT/logs/install.log"; : > "$path"; printf '%s\\n' "$path"; }
run_logged() { shift; "$@"; }
sha256_file() { printf '%064d\\n' 0; }
relative_to_deploy_root() { printf '%s\\n' "${1#"$ROBOTACTILE_DEPLOY_ROOT"/}"; }
write_receipt() { mkdir -p "$(dirname "$1")"; : > "$1"; }
""",
        encoding="utf-8",
    )

    deployment = tmp_path / "deployment"
    runtime = deployment / "runtime/ftp1-policy"
    runtime.mkdir(parents=True)
    python = runtime / "bin/python"
    python.parent.mkdir()
    python.write_text(
        """#!/usr/bin/env bash
set -u
if [ "${1:-}" = -c ]; then
  case "${2:-}" in
    *"import importlib.metadata as m,torch"*) printf '%s\\n' 'openpi|2.7.1+cu128|12.8|1.1.2|27.1.0' ;;
    *"import importlib.metadata as m; print"*) printf '%s\\n' "$FAKE_BOOTSTRAP_IDENTITY" ;;
    *"nvidia-cuda-nvrtc-cu12"*) printf '%s\\n' "$FAKE_CU128_BUNDLE_IDENTITY" ;;
    *"torch.cuda.get_arch_list"*) printf '%s\\n' '12.8|sm_120|torch.ones(cuda):ok' ;;
    *"import pathlib,transformers"*) printf '%s\\n' "$FAKE_TRANSFORMERS_ROOT" ;;
    *) exit 3 ;;
  esac
  exit 0
fi
if [ "${1:-}" = -m ] && [ "${2:-}" = pip ] && [ "${3:-}" = install ]; then
  printf '%s\\n' "$*" >> "$FAKE_PIP_LOG"
  [ "${FAKE_PIP_FAIL:-false}" != true ] || exit 17
  exit 0
fi
exit 3
""",
        encoding="utf-8",
    )
    python.chmod(0o755)
    transformers_root = tmp_path / "transformers"
    transformers_root.mkdir()
    pip_log = tmp_path / "pip.log"
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_BOOTSTRAP_IDENTITY": bootstrap_identity,
            "FAKE_CU128_BUNDLE_IDENTITY": (
                "torch==2.7.1+cu128|torchvision==0.22.1+cu128|triton==3.3.1|"
                "nvidia-cuda-nvrtc-cu12==12.8.61|"
                "nvidia-cuda-runtime-cu12==12.8.57|"
                "nvidia-cuda-cupti-cu12==12.8.57|"
                "nvidia-cudnn-cu12==9.7.1.26|"
                "nvidia-cublas-cu12==12.8.3.14|"
                "nvidia-cufft-cu12==11.3.3.41|"
                "nvidia-curand-cu12==10.3.9.55|"
                "nvidia-cusolver-cu12==11.7.2.55|"
                "nvidia-cusparse-cu12==12.5.7.53|"
                "nvidia-cusparselt-cu12==0.6.3|"
                "nvidia-nccl-cu12==2.26.2|"
                "nvidia-nvtx-cu12==12.8.55|"
                "nvidia-nvjitlink-cu12==12.8.61|"
                "nvidia-cufile-cu12==1.13.0.11"
            ),
            "FAKE_PIP_LOG": str(pip_log),
            "FAKE_TRANSFORMERS_ROOT": str(transformers_root),
        }
    )
    return installer, deployment, environment


def test_resume_existing_runtime_skips_bootstrap_and_finishes_install(
    tmp_path: Path,
) -> None:
    installer, deployment, environment = _resume_installer_fixture(tmp_path)

    completed = subprocess.run(
        (
            "bash",
            str(installer),
            "--root",
            str(deployment),
            "--resume-existing-runtime",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    installs = (tmp_path / "pip.log").read_text(encoding="utf-8").splitlines()
    assert len(installs) == 3
    assert all("--no-compile" in command for command in installs)
    assert all("--upgrade" not in command for command in installs)
    assert "--no-deps" in installs[0]
    assert "https://download.pytorch.org/whl/cu128" in installs[0]
    assert "torch==2.7.1+cu128" in installs[0]
    assert "torchvision==0.22.1+cu128" in installs[0]
    assert "nvidia-cufile-cu12==1.13.0.11" in installs[0]
    assert (deployment / "runtime/ftp1-policy/bin/python").exists()
    assert (
        deployment / "artifacts/deployment/ftp1_policy_runtime_install.json"
    ).exists()


def test_resume_existing_runtime_preserves_it_after_pip_failure(tmp_path: Path) -> None:
    installer, deployment, environment = _resume_installer_fixture(tmp_path)
    environment["FAKE_PIP_FAIL"] = "true"

    completed = subprocess.run(
        (
            "bash",
            str(installer),
            "--root",
            str(deployment),
            "--resume-existing-runtime",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "PyTorch cu128 bundle installation failed" in completed.stderr
    assert (deployment / "runtime/ftp1-policy/bin/python").exists()
    assert not (
        deployment / "artifacts/deployment/ftp1_policy_runtime_install.json"
    ).exists()


def test_resume_existing_runtime_rejects_incompatible_bootstrap(tmp_path: Path) -> None:
    installer, deployment, environment = _resume_installer_fixture(
        tmp_path,
        bootstrap_identity="25.2|80.9.0|0.45.1",
    )

    completed = subprocess.run(
        (
            "bash",
            str(installer),
            "--root",
            str(deployment),
            "--resume-existing-runtime",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "bootstrap identity is incompatible" in completed.stderr
    assert (deployment / "runtime/ftp1-policy/bin/python").exists()
    assert not (tmp_path / "pip.log").exists()


def test_resume_existing_runtime_rejects_an_existing_receipt(tmp_path: Path) -> None:
    installer, deployment, environment = _resume_installer_fixture(tmp_path)
    receipt = deployment / "artifacts/deployment/ftp1_policy_runtime_install.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")

    completed = subprocess.run(
        (
            "bash",
            str(installer),
            "--root",
            str(deployment),
            "--resume-existing-runtime",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "runtime/receipt is incomplete or incompatible" in completed.stderr
    assert (deployment / "runtime/ftp1-policy/bin/python").exists()
    assert not (tmp_path / "pip.log").exists()


def test_resume_existing_runtime_rejects_a_symlink_runtime_root(tmp_path: Path) -> None:
    installer, deployment, environment = _resume_installer_fixture(tmp_path)
    runtime = deployment / "runtime/ftp1-policy"
    target = tmp_path / "existing-runtime"
    runtime.rename(target)
    runtime.symlink_to(target, target_is_directory=True)

    completed = subprocess.run(
        (
            "bash",
            str(installer),
            "--root",
            str(deployment),
            "--resume-existing-runtime",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "refuses a symlink runtime root" in completed.stderr
    assert (target / "bin/python").exists()
    assert not (tmp_path / "pip.log").exists()


def test_existing_runtime_still_requires_explicit_resume_flag(tmp_path: Path) -> None:
    installer, deployment, environment = _resume_installer_fixture(tmp_path)

    completed = subprocess.run(
        ("bash", str(installer), "--root", str(deployment)),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "runtime/receipt is incomplete or incompatible" in completed.stderr
    assert (deployment / "runtime/ftp1-policy/bin/python").exists()
    assert not (tmp_path / "pip.log").exists()


def test_official_runtime_installer_help_documents_narrow_resume_scope() -> None:
    completed = subprocess.run(
        (
            "bash",
            str(ROOT / "scripts/ftp1_policy/install_official_runtime.sh"),
            "--help",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--resume-existing-runtime" in completed.stdout
    assert "interrupted one-time install" in completed.stdout
    assert "does not accept an arbitrary runtime path" in completed.stdout


def test_isaac_client_installer_is_pinned_and_keeps_n0_receipt_independent() -> None:
    installer = (ROOT / "scripts/ftp1_policy/install_isaac_client.sh").read_text(
        encoding="utf-8"
    )

    assert 'ISAAC_PYTHON="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0/python.sh"' in installer
    assert (
        'RECEIPT="$DEPLOY_ROOT/artifacts/deployment/'
        'ftp1_policy_isaac_client_install.json"'
    ) in installer
    assert '"msgpack==1.1.1" "pyzmq==27.1.0"' in installer
    assert "n0_twam_isaac_client_install.json" not in installer
    assert '"n0_runtime_modified=false"' in installer
    assert '"ftp1_server_runtime_modified=false"' in installer
    assert '"system_python_modified=false"' in installer
    assert '"system_cuda_modified=false"' in installer


def test_isaac_client_installer_help_needs_no_runtime_or_network() -> None:
    completed = subprocess.run(
        (
            "bash",
            str(ROOT / "scripts/ftp1_policy/install_isaac_client.sh"),
            "--help",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Usage: install_isaac_client.sh [--root PATH]" in completed.stdout
