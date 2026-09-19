"""Bounded pair-runner tests for N0 fault campaigns."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.n0_fault_campaign import runner
from robotactile_benchmark.n0_fault_campaign.contracts import (
    N0FaultCampaignCellSpec,
    N0FaultCampaignError,
    N0FaultCellDisposition,
)
from robotactile_benchmark.trials import Condition

PAIR = "a" * 64


def _cell(
    ordinal: int,
    condition: Condition,
    operator_id: str | None = None,
    severity_level: int | None = None,
) -> N0FaultCampaignCellSpec:
    live = condition is Condition.CLEAN or operator_id != "A1_stream_absence"
    label = "clean" if condition is Condition.CLEAN else f"{operator_id}/s3"
    return N0FaultCampaignCellSpec(
        ordinal=ordinal,
        task="insert_tube",
        initial_seed=7,
        exogenous_seed=7,
        pair_key=PAIR,
        condition=condition,
        disposition=(
            N0FaultCellDisposition.LIVE_REQUEST
            if live
            else N0FaultCellDisposition.UNSUPPORTED_CONTRACT
        ),
        operator_id=operator_id,
        severity_level=severity_level,
        operator_template_seed=None if operator_id is None else 29,
        request_relpath=(f"requests/insert_tube/{PAIR}/{label}.json" if live else None),
        artifact_relpath=f"artifacts/insert_tube/{PAIR}/{label}" if live else None,
        request_file_sha256="b" * 64 if live else None,
        trial_manifest_sha256="c" * 64,
        fault_manifest_sha256=None if operator_id is None else "d" * 64,
        fault_manifest_relpath=(
            None if operator_id is None else f"fault_manifests/{PAIR}/{'d' * 64}.json"
        ),
        unsupported_receipt_relpath=(
            None
            if live
            else f"unsupported_contracts/insert_tube/{PAIR}/{operator_id}/s3.json"
        ),
        unsupported_receipt_sha256=None if live else "e" * 64,
    )


def test_select_live_pair_cells_orders_clean_then_supported_faults() -> None:
    cells = (
        _cell(0, Condition.CLEAN),
        _cell(1, Condition.FAULTED, "T1_fixed_source_delay", 3),
        _cell(2, Condition.FAULTED, "F2_spatial_sensitivity_loss", 3),
        _cell(3, Condition.FAULTED, "A1_stream_absence", 3),
    )

    selected = runner.select_live_pair_cells(cells, None)

    assert selected[0].condition is Condition.CLEAN
    assert [cell.operator_id for cell in selected[1:]] == [
        "F2_spatial_sensitivity_loss",
        "T1_fixed_source_delay",
    ]
    assert all(
        cell.disposition is N0FaultCellDisposition.LIVE_REQUEST for cell in selected
    )


def test_multi_pair_selection_requires_explicit_pair_key() -> None:
    other = "f" * 64
    cells = (_cell(0, Condition.CLEAN),)
    cells += (
        N0FaultCampaignCellSpec.from_dict(
            {**cells[0].to_dict(), "ordinal": 1, "pair_key": other}
        ),
    )

    with pytest.raises(N0FaultCampaignError, match="explicit pair_key"):
        runner.select_live_pair_cells(cells, None)


def test_run_pair_propagates_capture_and_training_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells = (
        _cell(0, Condition.CLEAN),
        _cell(1, Condition.FAULTED, "T1_fixed_source_delay", 3),
    )
    loaded = SimpleNamespace(
        root=tmp_path,
        manifest=SimpleNamespace(campaign_id="n0-pilot", sha256="1" * 64, cells=cells),
        receipt_file_sha256="2" * 64,
    )
    requests = {
        str(cell.request_relpath): SimpleNamespace(
            output_dir=tmp_path / str(cell.artifact_relpath)
        )
        for cell in cells
    }
    captured: dict[str, object] = {}

    def execute(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        result = SimpleNamespace(
            paired=SimpleNamespace(group_content_sha256="3" * 64),
            artifacts=(
                SimpleNamespace(external_root_sha256="4" * 64),
                SimpleNamespace(external_root_sha256="5" * 64),
            ),
        )
        kwargs["pre_close_publisher"](result)
        return result

    monkeypatch.setattr(runner, "load_n0_fault_campaign_bundle", lambda _root: loaded)
    monkeypatch.setattr(
        runner,
        "load_live_univtac_request",
        lambda path: requests[str(Path(path).relative_to(tmp_path))],
    )
    monkeypatch.setattr(
        runner,
        "resolve_n0_runtime_artifacts",
        lambda _path: SimpleNamespace(manifest=SimpleNamespace(task_id="insert_tube")),
    )
    monkeypatch.setattr(runner, "execute_official_n0_paired_live_runs", execute)
    monkeypatch.setattr(
        runner,
        "write_paired_execution_receipt",
        lambda *_args: SimpleNamespace(file_sha256="6" * 64),
    )

    receipt = runner.run_n0_fault_pair(
        tmp_path,
        integration_config=tmp_path / "config.json",
        n0_source_root=tmp_path / "N0-TWAM",
        host="127.0.0.1",
        port=29601,
        capture_profile=LiveCaptureProfile.PREVIEW,
    )

    assert receipt.capture_profile == "preview_v1"
    assert receipt.action_execution_contract == ("robotactile_n0_training_60hz_ee_v1")
    assert captured["capture_profile"] is LiveCaptureProfile.PREVIEW
    assert (
        tmp_path / "executions/insert_tube" / PAIR / "pair_run_receipt.json"
    ).is_file()
