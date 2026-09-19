"""Static contracts for native-build and qualification entry points."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC, QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)

ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts" / "live_univtac"
VCPKG_COMMIT = "ce613c41372b23b1f51333815feb3edd87ef8a8b"
VCPKG_BASELINE = "b2cb0da531c2f1f740045bfe7c4dac59f0b2b69c"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_script(name: str) -> ModuleType:
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tacex_uipc_installer_is_pinned_and_base_environment_isolated() -> None:
    """Catches floating native tools and accidental base-env installation."""
    core_installer = _text(SCRIPTS / "install_tacex_univtac.sh")
    uipc_installer = _text(SCRIPTS / "install_tacex_uipc_univtac.sh")
    curobo_installer = _text(SCRIPTS / "install_curobo_v0_7_7.sh")
    common = _text(SCRIPTS / "common.sh")
    lock = _text(ROOT / "requirements/uipc-toolchain-linux-64.lock.txt")

    assert VCPKG_COMMIT in uipc_installer
    assert VCPKG_BASELINE in uipc_installer
    assert 'MICROMAMBA_VERSION="2.9.0"' in uipc_installer
    assert "MICROMAMBA_SHA256=" in uipc_installer
    assert "--no-rc" in uipc_installer
    assert "-u CONDA_PREFIX" in uipc_installer
    assert 'TMPDIR="$DEPLOY_ROOT/runtime/tmp"' in uipc_installer
    assert 'CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURE"' in uipc_installer
    assert "CMAKE_CUDA_ARCHITECTURES=86" not in uipc_installer
    assert "VCPKG_FORCE_SYSTEM_BINARIES=1" in uipc_installer
    assert 'VCPKG_OVERLAY_PORTS="$VCPKG_OVERLAY_ROOT"' in uipc_installer
    assert '"version": "0.8.3"' in uipc_installer
    assert "cpptrace_overlay_version=0.8.3" in uipc_installer
    assert "tinygltf_overlay_version=2.9.3" in uipc_installer
    assert "TINYGLTF_ARCHIVE_SHA512=" in uipc_installer
    assert "--tinygltf-archive" in uipc_installer
    assert 'sha512_file "$TINYGLTF_ARCHIVE"' in uipc_installer
    assert 'sha512_file "$TINYGLTF_CACHE_PATH"' in uipc_installer
    assert 'TORCH_CUDA_ARCH_LIST="$CUDA_COMPUTE_CAPABILITY"' in core_installer
    assert "TORCH_CUDA_ARCH_LIST=8.6" not in core_installer
    assert "--no-cache-dir" in core_installer
    assert (
        'TORCH_LIBRARY_PATH="$ISAAC_SIM_PATH/kit/python/lib/'
        'python3.10/site-packages/torch/lib"' in curobo_installer
    )
    assert (
        'LD_LIBRARY_PATH="$TORCH_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"' in curobo_installer
    )
    assert (
        '"torch_library_path=$(relative_to_deploy_root '
        '"$TORCH_LIBRARY_PATH")"' in curobo_installer
    )
    assert "ROBOTACTILE_EXPECTED_CUROBO_SOURCE" in curobo_installer
    assert 'm.version("nvidia-curobo") == "0.7.7"' in curobo_installer
    assert "native_extensions_reused=$NATIVE_EXTENSIONS_REUSED" in curobo_installer
    assert "CMAKE_CUDA_COMPILER:FILEPATH=$CUDA_ROOT/bin/nvcc" in uipc_installer
    assert "--force-reinstall" in uipc_installer
    for installer in (core_installer, uipc_installer, curobo_installer):
        assert "--cuda-root" in installer
        assert "--cuda-architecture" in installer
        assert "--gpu" in installer
        assert "cuda_toolkit_path=$CUDA_ROOT" in installer
        assert "cuda_toolkit_version=$CUDA_TOOLKIT_VERSION" in installer
        assert "cuda_nvcc_identity=$CUDA_NVCC_IDENTITY" in installer
        assert "cuda_architecture=sm_$CUDA_ARCHITECTURE" in installer
        assert "cuda_compute_capability=$CUDA_COMPUTE_CAPABILITY" in installer
        assert "\nsudo " not in installer
    assert "get_device_capability(0)" in common
    assert "runtime/cuda-toolkit-12.8/bin/nvcc" in common
    assert "runtime/cuda-toolkit-12.4/bin/nvcc" in common
    assert "/usr/local/cuda-12.8/bin/nvcc" in common
    assert "/usr/local/cuda-12.4/bin/nvcc" in common
    assert "100|101|120" in common
    assert 'ROBOTACTILE_CUDA_TOOLKIT_VERSION" = "12.8"' in common
    tinygltf_overlay = ROOT / "integrations/vcpkg-overlay-ports/tinygltf"
    assert '"version": "2.9.3"' in _text(tinygltf_overlay / "vcpkg.json")
    assert "6dbcff3ea602d0aa" in _text(tinygltf_overlay / "portfile.cmake")
    cpptrace_overlay = tinygltf_overlay.parent / "cpptrace"
    assert '"version": "0.8.3"' in _text(cpptrace_overlay / "vcpkg.json")
    assert "e74dae5142362129" in _text(cpptrace_overlay / "portfile.cmake")
    assert lock.startswith("# RoboTactile UIPC toolchain")
    assert "cmake-3.26.4" in lock
    assert "gcc_impl_linux-64-11.4.0" in lock


def test_blackwell_attestation_requires_real_sm120_code_objects() -> None:
    wrapper = _text(SCRIPTS / "attest_cuda_native_extensions.sh")
    implementation = _text(SCRIPTS / "attest_cuda_native_extensions.py")
    cuda_12_8 = _text(SCRIPTS / "install_cuda_toolkit_12_8.sh")

    assert 'ROBOTACTILE_CUDA_TOOLKIT_PROFILE="12.8"' in cuda_12_8
    assert "--cuda-architecture" in wrapper
    assert "cuda_native_extensions_sm_${CUDA_ARCHITECTURE}.json" in wrapper
    assert '"--list-elf"' in implementation
    assert '"--list-ptx"' in implementation
    assert "lacks required sm_" in implementation
    assert '"_version_cuda.so"' in implementation
    assert '"cuda_version_metadata_stub"' in implementation
    assert '"_segment_csr_cuda.so"' in implementation
    assert "expected five cuRobo CUDA extensions" in implementation
    assert "refusing to overwrite CUDA attestation" in implementation


def test_blackwell_bootstrap_is_isolated_n0_only_and_sm120_bound() -> None:
    bootstrap = _text(SCRIPTS / "bootstrap_blackwell_sm120.sh")

    assert "deployment-sm120" in bootstrap
    assert "refusing to use the legacy deployment root" in bootstrap
    assert 'GPU_CAPABILITY" = "12.0"' in bootstrap
    assert "install_cuda_toolkit_12_8.sh" in bootstrap
    assert "--cuda-architecture 120" in bootstrap
    assert "attest_cuda_native_extensions.sh" in bootstrap
    assert "install_official_runtime.sh" in bootstrap
    assert "install_robotactile_client.sh" in bootstrap
    assert "n0_twam_enabled=true" in bootstrap
    assert "act_enabled=false" in bootstrap
    assert "install_act" not in bootstrap
    assert "system_cuda_modified=false" in bootstrap


def test_univtac_task_import_qualification_has_explicit_evidence_boundary() -> None:
    qualifier = _text(SCRIPTS / "qualify_univtac_task_import.sh")
    smoke = _text(SCRIPTS / "smoke_univtac_task_isaac.py")

    assert "--task" in qualifier
    assert "univtac_task_import_${TASK_ID}_v3.json" in qualifier
    assert "REGISTRY_RESOURCE_SHA256" in qualifier
    assert "TASK_SOURCE_SHA256" in qualifier
    assert "cpptrace_overlay_version=0.8.3" in qualifier
    assert "task_instantiated=false" in qualifier
    assert "simulator_steps_executed=0" in qualifier
    assert "_task_contract(task_id)" in smoke
    assert "config_class()" in smoke
    assert '"task_source_sha256"' in smoke
    assert 'parser.add_argument("--task"' in smoke
    assert "ROBOTACTILE_TASK_SMOKE_RESULT" in smoke
    assert 'extension_ids = ("omni.ui",)' in smoke
    assert "set_extension_enabled_immediate" in smoke
    assert "isaac_headless_task_config_import_v3" in qualifier
    assert "task_class(" not in smoke
    assert "AppLauncher(headless=True)" in smoke


def test_univtac_task_reset_qualification_uses_production_backend_contract() -> None:
    qualifier = _text(SCRIPTS / "qualify_univtac_task_reset.sh")
    smoke = _text(SCRIPTS / "smoke_univtac_task_reset_isaac.py")

    assert "univtac_task_import_${TASK_ID}_v3.json" in qualifier
    assert "univtac_task_reset_${CONTRACT_RUN_ID}.json" in qualifier
    assert "run_id=$CONTRACT_RUN_ID" in qualifier
    assert '--action-spec "$ACTION_SPEC"' in qualifier
    assert "task_source_sha256=$TASK_SOURCE_SHA256" in qualifier
    assert (
        "robotactile_isaac_install-${CURRENT_SOURCE_MANIFEST_SHA256:0:16}.json"
        in qualifier
    )
    assert "runtime_source_manifest_sha256" in qualifier
    assert "require_command timeout" in qualifier
    assert 'CUDA_VISIBLE_DEVICES="$GPU_INDEX"' in qualifier
    assert '"${TIMEOUT_SECONDS}s"' in qualifier
    assert "document.get(key) != value" in qualifier
    assert '"task_instantiated": True' in qualifier
    assert '"reset_completed": True' in qualifier
    assert '"observation_captured": True' in qualifier
    assert '"runtime_close_requested": True' in qualifier
    assert '"policy_loaded": False' in qualifier
    assert '"action_commands_executed": 0' in qualifier
    assert '"closed_loop_control_cycles": 0' in qualifier
    assert '"reset_diagnostics": reset_diagnostics' in smoke
    assert '"placement_reset_assessment"' in smoke
    assert "univtac_task_reset_witness_${CONTRACT_RUN_ID}.json" in qualifier
    assert '"reset_viable"' in qualifier
    assert "placement_failure_phase=${RESULT_FIELDS[13]}" in qualifier
    assert "isaac_headless_task_reset_observation_v2" in qualifier
    assert "status=$QUALIFICATION_STATUS" in qualifier
    assert '"construction_seed": initial_seed' in qualifier
    assert "construction_seed=$INITIAL_SEED" in qualifier
    assert '"torch_deterministic_algorithms": True' in qualifier
    assert "cublas_workspace_config=:4096:8" in qualifier
    assert '"top": {"dtype": "uint8", "shape": [270, 480, 3]}' in qualifier
    assert '"wrist_l": {"dtype": "uint8", "shape": [270, 480, 3]}' in qualifier
    assert "task_success_evaluated=false" in qualifier
    assert "launch_univtac_runtime" in smoke
    assert "build_univtac_backend_config(task_id, action_spec=action_spec)" in smoke
    assert "UniVTACIsaacBackend" in smoke
    assert "PolicyEpisodeContext" in smoke
    assert "backend.reset(context)" in smoke
    assert "backend.observe()" in smoke
    assert "_publish_result(result_path, payload)" in smoke
    assert smoke.index("_publish_result(result_path, payload)") < smoke.index(
        "backend.close()"
    )
    assert "backend.execute(" not in smoke
    assert "check_success(" not in smoke


def test_robotactile_isaac_installer_is_manifest_bound_and_isolated() -> None:
    installer = _text(SCRIPTS / "install_robotactile_isaac.sh")

    assert "update_source_manifest.py" in installer
    assert "source_manifest_sha256=" in installer
    assert "wheel_sha256=" in installer
    assert "--force-reinstall" in installer
    assert "--no-deps" in installer
    assert "--wheel" in installer
    assert "py3-none-any.whl" in installer
    assert "robotactile_isaac_install-${SOURCE_MANIFEST_SHA256:0:16}.json" in installer
    assert (
        'WHEEL_MANIFEST_MEMBER="robotactile_benchmark/source_manifest.sha256"'
        in installer
    )
    assert "wheel_source_manifest_sha256" in installer
    assert "zipfile.ZipFile" in installer
    assert installer.index('if [ -n "$WHEEL_INPUT" ]; then') < installer.index(
        "update_source_manifest.py"
    )
    assert '"$ISAAC_SIM_PATH/python.sh"' in installer
    assert "-u CONDA_PREFIX" in installer
    assert "\nsudo " not in installer


def test_univtac_pairing_qualifier_is_live_bounded_and_fail_closed() -> None:
    qualifier = _text(SCRIPTS / "qualify_univtac_task_pairing.sh")
    smoke = _text(SCRIPTS / "smoke_univtac_task_pairing_isaac.py")

    assert "ensure_pinned_git_source" in qualifier
    assert "univtac_task_pairing_${CONTRACT_RUN_ID}.json" in qualifier
    assert "run_id=$CONTRACT_RUN_ID" in qualifier
    assert '--action-spec "$ACTION_SPEC"' in qualifier
    assert "task_source_sha256=$TASK_SOURCE_SHA256" in qualifier
    assert "require_command timeout" in qualifier
    assert 'CUDA_VISIBLE_DEVICES="$GPU_INDEX"' in qualifier
    assert "in_process_snapshot_replay_equivalence_v1" in qualifier
    assert "simulator_qualification_claimed=false" in qualifier
    assert "task_success_evaluated=false" in qualifier
    assert "UniVTACPairedBackendSession" in smoke
    assert "_divergence_action(" in smoke
    assert "initial_model_visible_qpos8" in smoke
    assert "canonical_record.observation.proprio" in smoke
    assert "qualification_action" not in smoke
    assert "np.array(initial_action8" in smoke
    assert "session.new_backend()" in smoke
    assert '"canonical_reset"' in smoke
    assert '"snapshot_replay"' in smoke
    assert '"policy_loaded": False' in smoke
    assert '"simulator_qualification_claimed": False' in smoke
    assert 'payload["paired_reset_receipt"]' in smoke
    assert smoke.index("_publish(result_path, payload)") < smoke.index(
        "session.close()"
    )


def test_qualification_registry_covers_all_eight_frozen_tasks() -> None:
    module = _load_script("smoke_univtac_task_isaac.py")
    task_ids = (
        "grasp_classify",
        "insert_HDMI",
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
        "pull_out_key",
        "put_bottle_in_shelf",
    )
    for task_id in task_ids:
        module_name, source_sha256 = module._task_contract(task_id)
        assert module_name == f"envs.{task_id}"
        for action_spec in (QPOS8_ACTION_SPEC, EE8_ACTION_SPEC):
            config = build_univtac_backend_config(task_id, action_spec=action_spec)
            assert source_sha256 == config.task.task_source_sha256
            assert config.action_spec == action_spec


def test_reset_and_pairing_parsers_preserve_defaults_and_accept_ee8() -> None:
    for script_name in (
        "smoke_univtac_task_reset_isaac.py",
        "smoke_univtac_task_pairing_isaac.py",
    ):
        module = _load_script(script_name)
        defaults = module._argument_parser().parse_args([])
        assert defaults.task == "pull_out_key"
        assert defaults.action_spec == QPOS8_ACTION_SPEC
        selected = module._argument_parser().parse_args(
            ["--task", "lift_can", "--action-spec", EE8_ACTION_SPEC]
        )
        assert selected.task == "lift_can"
        assert selected.action_spec == EE8_ACTION_SPEC


def test_pairing_divergence_action_supports_qpos8_and_ee8() -> None:
    module = _load_script("smoke_univtac_task_pairing_isaac.py")
    qpos_config = build_univtac_backend_config(
        "insert_HDMI", action_spec=QPOS8_ACTION_SPEC
    )
    qpos = np.asarray(
        [
            (lower + upper) / 2.0
            for lower, upper in zip(
                qpos_config.action_lower_bounds,
                qpos_config.action_upper_bounds,
            )
        ],
        dtype=np.float32,
    )
    qpos_action = module._divergence_action(qpos, qpos_config)
    assert qpos_action.dtype == np.float32
    assert qpos_action.shape == (1, 8)
    assert qpos_action.flags.c_contiguous
    assert not np.shares_memory(qpos_action, qpos)
    assert np.count_nonzero(qpos_action[0] != qpos) == 1

    ee_config = build_univtac_backend_config("insert_HDMI", action_spec=EE8_ACTION_SPEC)
    ee = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.02], dtype=np.float32)
    ee_action = module._divergence_action(ee, ee_config)
    assert ee_action.dtype == np.float32
    assert ee_action.shape == (1, 8)
    assert not np.shares_memory(ee_action, ee)
    np.testing.assert_array_equal(ee_action[0, 3:], ee[3:])
    assert np.count_nonzero(ee_action[0, :3] != ee[:3]) == 1


def test_qualification_shells_reject_unknown_task_and_action_contracts() -> None:
    environment = dict(os.environ, LC_ALL="C", LANG="C")
    for name in (
        "qualify_univtac_task_import.sh",
        "qualify_univtac_task_reset.sh",
        "qualify_univtac_task_pairing.sh",
    ):
        completed = subprocess.run(
            ["bash", str(SCRIPTS / name), "--task", "not_registered"],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )
        assert completed.returncode == 2
        assert "frozen UniVTAC registry" in completed.stderr
    for name in (
        "qualify_univtac_task_reset.sh",
        "qualify_univtac_task_pairing.sh",
    ):
        completed = subprocess.run(
            ["bash", str(SCRIPTS / name), "--action-spec", "unknown"],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )
        assert completed.returncode == 2
        assert "action spec must be" in completed.stderr
