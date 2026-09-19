import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from robotactile_benchmark.clean_baseline import CleanCampaignProtocol
from robotactile_benchmark.deployment.layout import DeploymentLayout

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/n0_twam/run_clean_campaign_all_tasks.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("n0_clean_all_task_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_task_runner_has_qualification_execution_reporting_and_publish_gates() -> (
    None
):
    source = (ROOT / "scripts/n0_twam/run_clean_campaign_all_tasks.py").read_text(
        encoding="utf-8"
    )
    for required in (
        "verify_all_task_qualification",
        'qualification.action_spec != "ee8_absolute"',
        "run_clean_task_shard.py",
        "clean-campaign-report",
        "clean-campaign-publish",
        "campaign_manifest_sha256",
        "qualification_sha256",
        "QUALIFICATION_V3_SEMANTIC_VERSION",
        "--qualification-sha256",
        "receipt_sha256",
        "--max-new-trials-per-task",
        "--continue-on-infrastructure-failure",
        "--integration-config-label",
        "--action-execution-contract",
        "action_execution_contract",
        "--worker-contract",
        "--reset-equivalence-dir",
        "--reset-equivalence-receipt",
        "--reset-equivalence-receipt-sha256",
        "reset_equivalence_receipts",
        "--capture-profile",
        "--allow-compact-capture",
    ):
        assert required in source


def test_all_task_runner_defaults_to_training_60hz_and_checks_qualification() -> None:
    module = _module()
    args = module._parser().parse_args(
        [
            "--manifest",
            "campaign_manifest.json",
            "--gpus",
            "0",
            "--run-id",
            "run-v1",
        ]
    )
    production = "robotactile_n0_training_60hz_ee_v1"
    assert args.action_execution_contract == production
    assert args.worker_contract == "fresh_process_v1"
    assert args.reset_equivalence_dir is None
    assert args.qualification is None
    assert args.execution_profile is None
    assert args.capture_profile is None

    source = SimpleNamespace(
        action_execution_contract=production,
        native_step_contract="fixed_physics_steps_per_action_v1",
    )
    qualification = SimpleNamespace(
        source_bound=True,
        tasks=("insert_hole",),
        task_source_bindings=(source,),
    )
    module._require_qualification_execution_contract(
        qualification,
        ("insert_hole",),
        production,
    )
    with pytest.raises(ValueError, match="action execution contract mismatch"):
        module._require_qualification_execution_contract(
            qualification,
            ("insert_hole",),
            "univtac_stock_ee_v1",
        )


def test_execution_profiles_skip_diagnostic_qualification_and_gate_claims() -> None:
    module = _module()
    one_per_task = SimpleNamespace(
        protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
        sampling=None,
        trials=(
            SimpleNamespace(task="insert_hole"),
            SimpleNamespace(task="insert_tube"),
        ),
    )
    quick = module._resolve_execution_profile("quick", one_per_task)
    assert quick is module.ExecutionProfile.QUICK
    assert module._resolve_capture_profile(None, quick).value == "metrics_only_v1"
    assert (
        module._validate_execution_profile(
            profile=quick,
            manifest=one_per_task,
            worker_contract="fresh_process_v1",
            qualification_path=None,
            max_new_trials_per_task=None,
            publish_paper=False,
        )
        == 1
    )
    assert (
        module._resolve_execution_profile(None, one_per_task)
        is module.ExecutionProfile.DIAGNOSTIC
    )
    assert (
        module._resolve_capture_profile(None, module.ExecutionProfile.DIAGNOSTIC).value
        == "preview_v1"
    )
    assert (
        module._resolve_capture_profile("paper_full_v1", quick).value == "paper_full_v1"
    )

    repeated_task = SimpleNamespace(
        protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
        sampling=None,
        trials=(
            SimpleNamespace(task="insert_hole"),
            SimpleNamespace(task="insert_hole"),
        ),
    )
    with pytest.raises(ValueError, match="one target trial"):
        module._validate_execution_profile(
            profile=quick,
            manifest=repeated_task,
            worker_contract="fresh_process_v1",
            qualification_path=None,
            max_new_trials_per_task=None,
            publish_paper=False,
        )
    with pytest.raises(ValueError, match="fresh_process_v1"):
        module._validate_execution_profile(
            profile=quick,
            manifest=one_per_task,
            worker_contract="same_task_worker_v1",
            qualification_path=None,
            max_new_trials_per_task=1,
            publish_paper=False,
        )

    pilot = SimpleNamespace(
        protocol_id=CleanCampaignProtocol.PILOT,
        sampling=None,
        trials=(SimpleNamespace(task="insert_hole"),) * 10,
    )
    assert (
        module._resolve_execution_profile(None, pilot) is module.ExecutionProfile.CLAIM
    )
    claim_capture = module._resolve_capture_profile(None, module.ExecutionProfile.CLAIM)
    assert claim_capture.value == "paper_full_v1"
    with pytest.raises(ValueError, match="requires --qualification"):
        module._validate_execution_profile(
            profile=module.ExecutionProfile.CLAIM,
            manifest=pilot,
            worker_contract="fresh_process_v1",
            qualification_path=None,
            max_new_trials_per_task=None,
            publish_paper=False,
        )
    with pytest.raises(ValueError, match="paper_v1"):
        module._validate_execution_profile(
            profile=module.ExecutionProfile.CLAIM,
            manifest=pilot,
            worker_contract="fresh_process_v1",
            qualification_path=Path("qualification.json"),
            max_new_trials_per_task=None,
            publish_paper=True,
        )


def test_capture_profiles_fail_before_claim_or_persistent_light_execution() -> None:
    module = _module()
    preview = module.LiveCaptureProfile.PREVIEW
    full = module.LiveCaptureProfile.PAPER_FULL

    with pytest.raises(ValueError, match="claim execution requires paper_full"):
        module._validate_capture_profile(
            execution_profile=module.ExecutionProfile.CLAIM,
            capture_profile=preview,
            worker_contract="fresh_process_v1",
            publish_paper=False,
        )
    with pytest.raises(ValueError, match="publish-paper requires paper_full"):
        module._validate_capture_profile(
            execution_profile=module.ExecutionProfile.DIAGNOSTIC,
            capture_profile=preview,
            worker_contract="fresh_process_v1",
            publish_paper=True,
        )
    with pytest.raises(ValueError, match="do not support same_task_worker"):
        module._validate_capture_profile(
            execution_profile=module.ExecutionProfile.DIAGNOSTIC,
            capture_profile=preview,
            worker_contract="same_task_worker_v1",
            publish_paper=False,
        )
    module._validate_capture_profile(
        execution_profile=module.ExecutionProfile.CLAIM,
        capture_profile=full,
        worker_contract="fresh_process_v1",
        publish_paper=True,
    )


def test_persistent_worker_requires_task_local_reset_equivalence_receipts(
    tmp_path: Path,
) -> None:
    module = _module()
    root = tmp_path / "deployment"
    root.mkdir()
    proof_dir = root / "artifacts/deployment/persistent-reset-equivalence"
    proof_dir.mkdir(parents=True)
    proof = proof_dir / "lift_bottle.json"
    proof.write_text(json.dumps({"status": "passed"}), encoding="utf-8")

    selected, receipts = module._reset_equivalence_receipts(
        deployment_root=root,
        tasks=("lift_bottle",),
        protocol=CleanCampaignProtocol.PILOT,
        worker_contract="same_task_worker_v1",
        directory=proof_dir,
    )

    assert selected == proof_dir
    assert receipts["lift_bottle"][0] == proof
    assert len(receipts["lift_bottle"][1]) == 64
    selected, receipts = module._reset_equivalence_receipts(
        deployment_root=root,
        tasks=("lift_bottle",),
        protocol=CleanCampaignProtocol.DIAGNOSTIC,
        worker_contract="same_task_worker_v1",
        directory=None,
    )
    assert selected is None
    assert receipts == {}
    with pytest.raises(ValueError, match="pilot/paper persistent worker"):
        module._reset_equivalence_receipts(
            deployment_root=root,
            tasks=("lift_bottle",),
            protocol=CleanCampaignProtocol.PAPER,
            worker_contract="same_task_worker_v1",
            directory=None,
        )
    with pytest.raises(ValueError, match="requires the persistent"):
        module._reset_equivalence_receipts(
            deployment_root=root,
            tasks=("lift_bottle",),
            protocol=CleanCampaignProtocol.DIAGNOSTIC,
            worker_contract="fresh_process_v1",
            directory=proof_dir,
        )


def test_all_task_runner_resolves_labeled_configs_without_fallback(
    tmp_path: Path,
) -> None:
    resolve = _module()._labeled_integration_config
    layout = DeploymentLayout(tmp_path / "deployment")
    config = (
        layout.model_artifacts
        / "n0_twam/configs/lift_can-source-c43a216/integration_config.json"
    )
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")

    assert resolve(layout, "lift_can", "source-c43a216") == config.absolute()
    assert resolve(layout, "lift_can", None) is None
    with pytest.raises(ValueError, match="regular file"):
        resolve(layout, "pull_out_key", "source-c43a216")


def test_one_shot_runner_runs_quick_and_only_waits_for_optional_qualification() -> None:
    source = (ROOT / "scripts/n0_twam/run_clean_all_tasks_once.sh").read_text(
        encoding="utf-8"
    )
    for required in (
        'if [ -n "$QUALIFICATION" ]',
        'until [ -f "$QUALIFICATION" ]',
        "--protocol diagnostic_v1",
        "--trials-per-task 1",
        "run_clean_campaign_all_tasks.py",
        "--execution-profile quick",
        "--max-new-trials-per-task 1",
    ):
        assert required in source


def test_task_shard_isolates_n0_subprocesses_from_isaac_python() -> None:
    source = (ROOT / "scripts/n0_twam/run_clean_task_shard.py").read_text(
        encoding="utf-8"
    )
    for required in (
        '"LD_LIBRARY_PATH"',
        '"PYTHONHOME"',
        '"PYTHONPATH"',
        '"PYTHONUSERBASE"',
        '"VIRTUAL_ENV"',
        "n0_environment = _n0_subprocess_environment(layout.root)",
        '"ROBOTACTILE_N0_DIGEST_CACHE_DIR"',
        "env=n0_environment",
        "environment=n0_environment",
    ):
        assert required in source
