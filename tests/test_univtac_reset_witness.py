from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from robotactile_benchmark.adapters.univtac import ContactPhaseState
from robotactile_benchmark.backends.univtac_conversion import (
    ConvertedUniVTACObservation,
)
from robotactile_benchmark.backends.univtac_reset_witness import (
    STRICT_RESET_QUALIFICATION_PROFILE,
    TRAJECTORY_RESET_QUALIFICATION_PROFILE,
    UNIVTAC_RESET_REFERENCE_SCHEMA,
    UNIVTAC_RESET_SEMANTIC_VERSION,
    UNIVTAC_RESET_WITNESS_SCHEMA,
    UniVTACResetReference,
    UniVTACResetReferenceError,
    build_univtac_reset_witness,
)
from robotactile_benchmark.closed_loop.contracts import PolicyEpisodeContext
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.fixtures import make_synthetic_episode


def _context(**overrides: object) -> PolicyEpisodeContext:
    values: dict[str, object] = {
        "episode_id": "reset-episode-001",
        "task": "insert_HDMI",
        "initial_seed": 1001,
        "exogenous_seed": 2002,
        "instruction": "Insert the HDMI plug.",
        "action_spec": "qpos8_next_step",
    }
    values.update(overrides)
    return PolicyEpisodeContext(**values)  # type: ignore[arg-type]


def _converted(
    *, native_step: int = 240, qpos_offset: float = 0.0
) -> ConvertedUniVTACObservation:
    source = make_synthetic_episode()[0]
    qpos8 = np.linspace(-0.4, 0.3, 8, dtype=np.float32)
    qpos8 += np.float32(qpos_offset)
    observation = replace(
        source.observation,
        episode_id="reset-episode-001",
        seed=1001,
        proprio=qpos8,
    )
    record = source.with_observation(observation)
    joint9 = np.concatenate((qpos8[:7], qpos8[-1:], qpos8[-1:])).astype(np.float32)
    return ConvertedUniVTACObservation(
        record=record,
        native_step_id=native_step,
        simulator_state_sha256="a" * 64,
        joint_reorder_witness_sha256="b" * 64,
        canonical_joint9=joint9,
        model_visible_qpos8=qpos8,
        left_phase_state=ContactPhaseState(False),
        right_phase_state=ContactPhaseState(False),
    )


def _reference(**overrides: object) -> UniVTACResetReference:
    values: dict[str, object] = {
        "task_id": "insert_HDMI",
        "initial_seed": 1001,
        "exogenous_seed": 2002,
        "pair_key": "1" * 64,
        "dataset_sha256": "2" * 64,
        "checkpoint_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "source_artifact_root_sha256": "6" * 64,
        "source_result_sha256": "5" * 64,
        "source_run_content_sha256": "7" * 64,
        "expected_simulator_state_sha256": "a" * 64,
        "expected_native_step": 240,
        "expected_qpos8": tuple(float(item) for item in np.linspace(-0.4, 0.3, 8)),
        "qpos_atol": 1e-5,
    }
    values.update(overrides)
    return UniVTACResetReference(**values)  # type: ignore[arg-type]


def test_reference_is_strict_json_roundtrippable_and_content_addressed() -> None:
    reference = _reference()
    document = reference.to_dict()
    loaded = UniVTACResetReference.from_dict(json.loads(json.dumps(document)))

    assert loaded == reference
    assert loaded.sha256 == reference.sha256 == canonical_hash(document)
    assert isinstance(document["expected_qpos8"], list)
    assert reference.schema == UNIVTAC_RESET_REFERENCE_SCHEMA
    assert reference.semantic_version == UNIVTAC_RESET_SEMANTIC_VERSION


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"initial_seed": True}, "initial_seed"),
        ({"pair_key": "not-a-sha"}, "pair_key"),
        ({"expected_native_step": -1}, "expected_native_step"),
        ({"expected_qpos8": (0.0,) * 7}, "length 8"),
        ({"expected_qpos8": (0.0,) * 7 + (float("nan"),)}, "finite"),
        ({"qpos_atol": 0.0}, "finite and positive"),
        ({"schema": "wrong"}, "schema mismatch"),
        ({"semantic_version": "2.0"}, "semantic version mismatch"),
    ],
)
def test_reference_rejects_invalid_contract_fields(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(UniVTACResetReferenceError, match=message):
        _reference(**overrides)


def test_reference_from_dict_rejects_noncanonical_shape() -> None:
    document = _reference().to_dict()
    document["extra"] = True
    with pytest.raises(UniVTACResetReferenceError, match="fields mismatch"):
        UniVTACResetReference.from_dict(document)

    document = _reference().to_dict()
    document["expected_qpos8"] = tuple(document["expected_qpos8"])  # type: ignore[arg-type]
    with pytest.raises(UniVTACResetReferenceError, match="JSON array"):
        UniVTACResetReference.from_dict(document)


def test_witness_without_reference_records_actual_state_without_claim() -> None:
    converted = _converted()
    witness = build_univtac_reset_witness(_context(), converted)

    assert witness["schema"] == UNIVTAC_RESET_WITNESS_SCHEMA
    assert witness["semantic_version"] == UNIVTAC_RESET_SEMANTIC_VERSION
    assert witness["native_step"] == 240
    assert witness["qpos8"] == [float(item) for item in converted.model_visible_qpos8]
    assert witness["canonical_joint9"] == [
        float(item) for item in converted.canonical_joint9
    ]
    assert witness["qpos8_sha256"] == canonical_hash(converted.model_visible_qpos8)
    assert witness["model_visible_proprio8"] == witness["qpos8"]
    assert witness["qpos8_is_model_visible_proprio"] is True
    assert witness["canonical_joint9_sha256"] == canonical_hash(
        converted.canonical_joint9
    )
    assert witness["reference_checks"] == {"reference_provided": False}
    assert "reset_viable" not in witness
    json.dumps(witness, allow_nan=False)


def test_ee_witness_records_qpos_and_model_visible_proprio_separately() -> None:
    converted = _converted()
    ee_proprio = np.linspace(1.0, 1.7, 8, dtype=np.float32)
    observation = replace(converted.record.observation, proprio=ee_proprio)
    converted = replace(
        converted,
        record=converted.record.with_observation(observation),
    )

    witness = build_univtac_reset_witness(
        _context(action_spec="ee8_absolute"),
        converted,
    )

    assert witness["qpos8_is_model_visible_proprio"] is False
    assert witness["model_visible_proprio8"] == [float(item) for item in ee_proprio]
    assert "reset_viable" not in witness


def test_matching_reference_declares_reset_viable() -> None:
    reference = _reference()
    witness = build_univtac_reset_witness(_context(), _converted(), reference)
    checks = witness["reference_checks"]

    assert isinstance(checks, dict)
    assert witness["reset_viable"] is True
    assert checks["reference_sha256"] == reference.sha256
    assert checks["simulator_state_match"] is True
    assert checks["simulator_state_match_required"] is True
    assert checks["qualification_profile"] == STRICT_RESET_QUALIFICATION_PROFILE
    assert checks["native_step_match"] is True
    assert checks["qpos8_atol_match"] is True
    assert checks["qpos8_max_abs_error"] < reference.qpos_atol


@pytest.mark.parametrize(
    ("native_step", "qpos_offset", "simulator_state", "failed_check"),
    [
        (241, 0.0, "a" * 64, "native_step_match"),
        (240, 0.01, "a" * 64, "qpos8_atol_match"),
        (240, 0.0, "c" * 64, "simulator_state_match"),
    ],
)
def test_runtime_drift_produces_false_reset_viability(
    native_step: int,
    qpos_offset: float,
    simulator_state: str,
    failed_check: str,
) -> None:
    converted = replace(
        _converted(native_step=native_step, qpos_offset=qpos_offset),
        simulator_state_sha256=simulator_state,
    )
    witness = build_univtac_reset_witness(
        _context(),
        converted,
        _reference(),
    )
    checks = witness["reference_checks"]

    assert isinstance(checks, dict)
    assert witness["reset_viable"] is False
    assert checks[failed_check] is False


def test_trajectory_profile_keeps_render_hash_diagnostic_only() -> None:
    converted = replace(_converted(), simulator_state_sha256="c" * 64)

    witness = build_univtac_reset_witness(
        _context(),
        converted,
        _reference(),
        require_simulator_state_match=False,
    )
    checks = witness["reference_checks"]

    assert isinstance(checks, dict)
    assert witness["reset_viable"] is True
    assert checks["simulator_state_match"] is False
    assert checks["simulator_state_match_required"] is False
    assert checks["native_step_match"] is True
    assert checks["qpos8_atol_match"] is True
    assert checks["qualification_profile"] == TRAJECTORY_RESET_QUALIFICATION_PROFILE


@pytest.mark.parametrize(
    ("native_step", "qpos_offset", "failed_check"),
    [
        (241, 0.0, "native_step_match"),
        (240, 0.01, "qpos8_atol_match"),
    ],
)
def test_trajectory_profile_keeps_physical_endpoint_checks_hard(
    native_step: int,
    qpos_offset: float,
    failed_check: str,
) -> None:
    converted = replace(
        _converted(native_step=native_step, qpos_offset=qpos_offset),
        simulator_state_sha256="c" * 64,
    )

    witness = build_univtac_reset_witness(
        _context(),
        converted,
        _reference(),
        require_simulator_state_match=False,
    )
    checks = witness["reference_checks"]

    assert isinstance(checks, dict)
    assert witness["reset_viable"] is False
    assert checks[failed_check] is False


def test_relaxed_profile_requires_reference_and_strict_boolean() -> None:
    with pytest.raises(ValueError, match="only be relaxed"):
        build_univtac_reset_witness(
            _context(),
            _converted(),
            require_simulator_state_match=False,
        )
    with pytest.raises(TypeError, match="must be bool"):
        build_univtac_reset_witness(
            _context(),
            _converted(),
            _reference(),
            require_simulator_state_match=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "reference",
    [
        _reference(task_id="lift_bottle"),
        _reference(initial_seed=999),
        _reference(exogenous_seed=999),
    ],
)
def test_reference_identity_mismatch_fails_closed(
    reference: UniVTACResetReference,
) -> None:
    with pytest.raises(UniVTACResetReferenceError, match="does not match context"):
        build_univtac_reset_witness(_context(), _converted(), reference)


def test_context_and_converted_observation_mismatch_fails_closed() -> None:
    with pytest.raises(UniVTACResetReferenceError, match="converted task"):
        build_univtac_reset_witness(
            _context(task="lift_bottle"),
            _converted(),
        )
