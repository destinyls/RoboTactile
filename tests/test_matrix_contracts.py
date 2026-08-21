from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    REST_REFERENCE_OPERATOR_IDS,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.matrix import (
    MatrixGridKind,
    MatrixGridPoint,
    MatrixManifest,
    build_focused_restoration_manifest,
    build_primary_matrix_manifest,
)
from robotactile_benchmark.severity import severity_value
from robotactile_benchmark.trials import (
    Condition,
    RestorationMode,
    TrialManifest,
    system_manifest_hash,
)


def _clean_trial() -> TrialManifest:
    return TrialManifest(
        task="pull_out_key",
        initial_seed=11,
        exogenous_seed=29,
        condition=Condition.CLEAN,
        base_system_id="univtac-act-tactile",
        executed_system_id="univtac-act-tactile",
        dataset_sha256="a" * 64,
        base_system_manifest_sha256=system_manifest_hash(
            "univtac-act-tactile", "b" * 64, "c" * 64, "qpos8_next_step"
        ),
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        action_spec="qpos8_next_step",
        fault_manifest_sha256=None,
        matched_no_touch_system_id=None,
        restoration_index=None,
        restoration_mode=None,
    )


def _fault(operator_id: str, severity_level: int, *, seed: int = 700) -> FaultManifest:
    parameters = {}
    if operator_id in REST_REFERENCE_OPERATOR_IDS:
        parameters["rest_reference_sha256"] = "d" * 64
    if operator_id == "C2_frame_misregistration":
        parameters["realization"] = "registered_pixels"
    return FaultManifest(
        operator_id=operator_id,
        severity_level=severity_level,
        operator_seed=seed + severity_level,
        start_index=20,
        stop_index=100,
        sensor_slots=("left", "right")
        if operator_id == "C1_sensor_identity_misrouting"
        else ("left",),
        observability=Observability.BLIND,
        parameters=parameters,
    )


def _primary_faults() -> tuple[FaultManifest, ...]:
    return tuple(
        _fault(operator_id, severity)
        for operator_id in sorted(CORE_OPERATOR_IDS)
        for severity in range(1, 6)
    )


def _primary_manifest() -> MatrixManifest:
    return build_primary_matrix_manifest(
        matrix_id="pull-key-primary-v1",
        clean=_clean_trial(),
        fault_manifests=_primary_faults(),
        restoration_index=60,
        restoration_mode=RestorationMode.VALID_STREAM_RESUME,
        no_touch_system_id="univtac-act-vision-only",
        no_touch_checkpoint_sha256="e" * 64,
        no_touch_config_sha256="f" * 64,
    )


def test_primary_manifest_materializes_70_rows_but_only_two_shared_baselines() -> None:
    manifest = _primary_manifest()

    assert manifest.kind is MatrixGridKind.PRIMARY
    assert len(manifest.comparisons) == 70
    assert len(manifest.cells) == 142
    assert {cell.trial.pair_key for cell in manifest.cells} == {manifest.pair_key}
    clean_cells = [
        cell for cell in manifest.cells if cell.trial.condition is Condition.CLEAN
    ]
    no_touch_cells = [
        cell for cell in manifest.cells if cell.trial.condition is Condition.NO_TOUCH
    ]
    assert len(clean_cells) == len(no_touch_cells) == 1
    assert {comparison.clean_cell_sha256 for comparison in manifest.comparisons} == {
        clean_cells[0].sha256
    }
    assert {comparison.no_touch_cell_sha256 for comparison in manifest.comparisons} == {
        no_touch_cells[0].sha256
    }
    assert {
        (comparison.operator_id, comparison.severity_level)
        for comparison in manifest.comparisons
    } == {
        (operator_id, severity)
        for operator_id in CORE_OPERATOR_IDS
        for severity in range(1, 6)
    }
    assert all(
        comparison.native_dose
        == severity_value(comparison.operator_id, comparison.severity_level)
        for comparison in manifest.comparisons
    )


def test_manifest_is_immutable_strictly_round_trippable_and_content_addressed() -> None:
    manifest = _primary_manifest()
    restored = MatrixManifest.from_dict(manifest.to_dict())

    assert restored == manifest
    assert restored.sha256 == manifest.sha256
    assert tuple(cell.sha256 for cell in restored.cells) == tuple(
        cell.sha256 for cell in manifest.cells
    )
    with pytest.raises(FrozenInstanceError):
        manifest.matrix_id = "changed"  # type: ignore[misc]
    malformed = manifest.to_dict()
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        MatrixManifest.from_dict(malformed)


def test_primary_manifest_rejects_incomplete_or_declared_grid() -> None:
    with pytest.raises(ValueError, match="14 x 5"):
        build_primary_matrix_manifest(
            matrix_id="missing-cell",
            clean=_clean_trial(),
            fault_manifests=_primary_faults()[:-1],
            restoration_index=60,
            restoration_mode=RestorationMode.VALID_STREAM_RESUME,
            no_touch_system_id="vision-only",
            no_touch_checkpoint_sha256="e" * 64,
            no_touch_config_sha256="f" * 64,
        )

    declared = _fault("T2_held_last_freeze", 1)
    declared = FaultManifest.from_dict(
        {**declared.to_dict(), "observability": Observability.DECLARED.value}
    )
    faults = list(_primary_faults())
    index = next(
        i
        for i, item in enumerate(faults)
        if item.operator_id == declared.operator_id and item.severity_level == 1
    )
    faults[index] = declared
    with pytest.raises(ValueError, match="blind"):
        build_primary_matrix_manifest(
            matrix_id="declared-cell",
            clean=_clean_trial(),
            fault_manifests=tuple(faults),
            restoration_index=60,
            restoration_mode=RestorationMode.VALID_STREAM_RESUME,
            no_touch_system_id="vision-only",
            no_touch_checkpoint_sha256="e" * 64,
            no_touch_config_sha256="f" * 64,
        )


def test_focused_restoration_grid_deduplicates_the_shared_faulted_execution() -> None:
    fault = _fault("F7_high_load_saturation", 3)
    manifest = build_focused_restoration_manifest(
        matrix_id="restoration-sweep-v1",
        clean=_clean_trial(),
        grid_points=(
            MatrixGridPoint("restore-40", fault, restoration_index=40),
            MatrixGridPoint("restore-60", fault, restoration_index=60),
        ),
        restoration_mode=RestorationMode.VALID_STREAM_RESUME,
        no_touch_system_id="univtac-act-vision-only",
        no_touch_checkpoint_sha256="e" * 64,
        no_touch_config_sha256="f" * 64,
    )

    assert manifest.kind is MatrixGridKind.FOCUSED_RESTORATION
    assert len(manifest.comparisons) == 2
    assert len(manifest.cells) == 5
    assert (
        len({comparison.faulted_cell_sha256 for comparison in manifest.comparisons})
        == 1
    )
    assert (
        len({comparison.restored_cell_sha256 for comparison in manifest.comparisons})
        == 2
    )


def test_comparison_rejects_boolean_native_dose_alias_for_integer_one() -> None:
    manifest = _primary_manifest()
    comparison = next(
        row
        for row in manifest.comparisons
        if row.operator_id == "F3_persistent_surface_artifact"
        and row.severity_level == 1
    )

    with pytest.raises(ValueError, match="native dose"):
        replace(comparison, native_dose=True)


def test_manifest_rejects_unreferenced_cells_and_crossed_operator_instances() -> None:
    first_fault = _fault("F7_high_load_saturation", 3, seed=700)
    second_fault = _fault("F7_high_load_saturation", 3, seed=900)
    manifest = build_focused_restoration_manifest(
        matrix_id="operator-instance-links-v1",
        clean=_clean_trial(),
        grid_points=(
            MatrixGridPoint("instance-a", first_fault, restoration_index=60),
            MatrixGridPoint("instance-b", second_fault, restoration_index=60),
        ),
        restoration_mode=RestorationMode.VALID_STREAM_RESUME,
        no_touch_system_id="univtac-act-vision-only",
        no_touch_checkpoint_sha256="e" * 64,
        no_touch_config_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="unreferenced"):
        replace(manifest, comparisons=manifest.comparisons[:1])

    crossed = replace(
        manifest.comparisons[0],
        faulted_cell_sha256=manifest.comparisons[1].faulted_cell_sha256,
    )
    with pytest.raises(ValueError, match="operator instance"):
        replace(manifest, comparisons=(crossed, manifest.comparisons[1]))
