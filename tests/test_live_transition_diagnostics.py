from __future__ import annotations

import json
from pathlib import Path

from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.closed_loop.runner import (
    run_closed_loop_trial_with_evidence,
)
from robotactile_benchmark.execution import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
    load_live_univtac_artifact,
    load_live_univtac_run,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.trials import Condition


def _artifact(tmp_path: Path) -> Path:
    request = LiveUniVTACRunRequest(
        task_id="insert_HDMI",
        condition=Condition.CLEAN,
        policy_kind=LivePolicyKind.ACT,
        base_system_id="transition-diagnostics-test",
        dataset_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        base_system_manifest_sha256=None,
        initial_seed=10,
        exogenous_seed=20,
        max_control_cycles=2,
        max_observation_steps=3,
        execute_action_steps=1,
        wall_timeout_s=5.0,
        upstream_root=tmp_path / "upstream",
        runtime_dir=tmp_path / "runtime",
        output_dir=None,
        fault_manifest_path=None,
        rest_references_path=None,
        restoration_index=None,
        restoration_mode=None,
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name="cpu",
        simulator_device=None,
        launcher_args=production_univtac_launcher_args(),
    )
    loaded = load_live_univtac_run(request)
    evidence = run_closed_loop_trial_with_evidence(
        loaded.trial,
        loaded.run_spec,
        DeterministicFakeBackend(
            make_synthetic_episode(length=10),
            success_predicate_id=loaded.run_spec.success_predicate_id,
        ),
        DeterministicFakePolicy.for_trial(loaded.trial),
    )
    output = tmp_path / "bundle"
    write_live_univtac_artifact(output, loaded, evidence)
    return output


def test_transition_diagnostics_are_pinned_and_legacy_v1_remains_loadable(
    tmp_path: Path,
) -> None:
    output = _artifact(tmp_path)
    loaded = load_live_univtac_artifact(output)
    assert len(loaded.evidence.transition_entries) == 2
    assert "fake_state" in loaded.evidence.transition_entries[0].diagnostics

    receipt_path = output / "root_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["semantic_version"] = "1.0"
    receipt["members"] = [
        item for item in receipt["members"] if item["path"] != "transition_trace.json"
    ]
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    )
    (output / "transition_trace.json").unlink()

    legacy = load_live_univtac_artifact(output)
    assert legacy.root_receipt.semantic_version == "1.0"
    assert legacy.evidence.transition_entries == ()
