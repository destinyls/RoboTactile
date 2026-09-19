"""Complement coverage, paired denominators, and explicit ACT zero filling."""

from dataclasses import replace
from pathlib import Path

import pytest
from test_official_act_live_cli import _request

from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.execution.live_univtac import _make_live_policy
from robotactile_benchmark.execution.loading import load_live_univtac_run
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.policies.tactile_availability import (
    TactileAvailabilityMode,
    ZeroFillTactilePolicy,
)
from robotactile_benchmark.trials import Condition
from scripts.decision_stress.remaining import (
    AVAILABILITY,
    NATIVE,
    PREVIOUS,
    make_fault,
    write,
)
from scripts.decision_stress.report import aggregate


def test_complement_is_exact_and_disjoint() -> None:
    old = set(PREVIOUS) - {"clean"}
    assert len(NATIVE) == 8 and len(AVAILABILITY) == 2
    assert not old.intersection((*NATIVE, *AVAILABILITY))
    assert old | set(NATIVE) | set(AVAILABILITY) == set(CORE_OPERATOR_IDS)


@pytest.mark.parametrize("operator", (*NATIVE, *AVAILABILITY))
@pytest.mark.parametrize(
    "task,seed", [("grasp_classify", 5), ("lift_can", 3), ("lift_can", 7)]
)
def test_complement_fault_roundtrip(operator: str, task: str, seed: int) -> None:
    fault = make_fault(
        operator, task=task, seed=seed, pair_key="a" * 64, rest_sha="b" * 64
    )
    assert fault == FaultManifest.from_dict(fault.to_dict())
    assert 1 <= fault.start_index <= 8 and fault.stop_index == 301
    assert fault.severity_level == 5
    assert (
        fault.sha256
        == make_fault(
            operator, task=task, seed=seed, pair_key="a" * 64, rest_sha="b" * 64
        ).sha256
    )
    if operator == "T1_fixed_source_delay":
        assert fault.parameters["temporal_schedule"] == "window_to_end_v1"
    if operator == "A2_frame_erasure":
        assert fault.parameters["a2_end_policy"] == "episode_censored_v1"


def test_act_zero_fill_factory_is_opt_in(tmp_path: Path) -> None:
    clean = _request(tmp_path, Condition.CLEAN)
    assert not load_live_univtac_run(clean).policy_identity.supports_structural_absence
    adapted = replace(
        clean,
        tactile_availability_mode=TactileAvailabilityMode.ZERO_FILL,
        tactile_zero_shape=(6, 8, 3),
    )
    loaded = load_live_univtac_run(adapted)

    class Inner:
        def __init__(self, identity):
            self.identity = identity

    policy = _make_live_policy(loaded, lambda item: Inner(item.policy_identity), None)
    assert isinstance(policy, ZeroFillTactilePolicy)
    assert policy.identity == loaded.policy_identity
    assert loaded.content_sha256 != load_live_univtac_run(clean).content_sha256
    assert adapted.execute_action_steps == clean.execute_action_steps == 1


def test_invalid_pending_and_unmatched_excluded_from_pair_drop() -> None:
    base = dict(model="act", task="lift_can", condition="F2", protocol="native", seed=3)
    valid = dict(
        base,
        status="terminal",
        validation_passed=True,
        score_eligible=True,
        score_success=False,
        paired_clean_success=True,
        initial_state_match=True,
        valid_pair=True,
    )
    invalid = dict(valid, validation_passed=None, valid_pair=False)
    pending = dict(base, status="pending")
    unmatched = dict(
        valid, initial_state_match=False, valid_pair=False, score_success=True
    )
    stat = aggregate([valid, invalid, pending, unmatched])[0]
    assert stat["pending"] == 1 and stat["invalid_or_ineligible"] == 1
    assert stat["success_count"] == 1 and stat["success_rate"] == 0.5
    assert stat["valid_pair_count"] == 1 and stat["paired_sr_drop"] == 1.0
    assert stat["paired_clean_success_to_failure"] == 1


def test_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write(path, {"success": False})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write(path, {"success": True})
    assert path.read_bytes() == before
