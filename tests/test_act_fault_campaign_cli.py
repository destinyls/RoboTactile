"""Public CLI tests for ACT Clean/Faulted campaigns."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robotactile_benchmark import cli
from robotactile_benchmark.act_fault_campaign import cli as campaign_cli


def test_generate_cli_selects_all_14_contract_operators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def generate(output: Path, spec: object) -> tuple[str, object]:
        captured.update(output=output, spec=spec)
        return (
            "created",
            SimpleNamespace(
                root=output,
                receipt_file_sha256="a" * 64,
                manifest=SimpleNamespace(
                    campaign_id="act-pilot",
                    sha256="b" * 64,
                    pair_count=1,
                    cell_count=15,
                    live_request_count=13,
                    unsupported_contract_count=2,
                ),
            ),
        )

    monkeypatch.setattr(campaign_cli, "generate_act_fault_campaign_bundle", generate)
    args = cli._parser().parse_args(  # noqa: SLF001
        [
            "generate-act-fault-campaign",
            "--output",
            str(tmp_path / "campaign"),
            "--campaign-id",
            "act-pilot",
            "--base-clean-request",
            str(tmp_path / "clean.json"),
            "--artifact-manifest",
            f"grasp_classify={tmp_path / 'act-manifest.json'}",
            "--deployment-root",
            str(tmp_path / "deployment"),
            "--severity",
            "5",
            "--operator-seed-master",
            "20260830",
            "--fault-start-index",
            "0",
            "--fault-stop-index",
            "300",
            "--rest-reference",
            f"grasp_classify={tmp_path / 'rest'}",
            "--reset-reference",
            str(tmp_path / "reset-reference.json"),
            "--reset-trajectory",
            str(tmp_path / "reset-trajectory.json"),
        ]
    )

    payload = campaign_cli.handle_act_fault_campaign_command(args)

    assert payload is not None
    spec = captured["spec"]
    assert len(spec.operator_ids) == 14
    assert spec.severity_levels == (5,)
    assert spec.fault_start_index == 0
    assert spec.fault_stop_index == 300
    assert spec.reset_reference_artifact_paths == (
        (tmp_path / "reset-reference.json").absolute(),
    )
    assert spec.reset_trajectory_artifact_paths == (
        (tmp_path / "reset-trajectory.json").absolute(),
    )
    assert payload["live_request_count"] == 13
    assert payload["unsupported_contract_count"] == 2


def test_run_cli_defaults_to_paper_full_capture(tmp_path: Path) -> None:
    args = cli._parser().parse_args(  # noqa: SLF001
        [
            "run-act-fault-campaign",
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--integration-config",
            str(tmp_path / "integration.json"),
        ]
    )

    assert args.capture_profile == "paper_full_v1"
    assert args.task is None
    assert args.pair_key is None


def test_report_cli_has_one_no_clobber_output(tmp_path: Path) -> None:
    args = cli._parser().parse_args(  # noqa: SLF001
        [
            "report-act-fault-campaign",
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--output",
            str(tmp_path / "report"),
        ]
    )

    assert args.output == tmp_path / "report"


def test_duplicate_task_mapping_is_rejected(tmp_path: Path) -> None:
    value = f"grasp_classify={tmp_path / 'artifact'}"
    with pytest.raises(ValueError, match="duplicate"):
        campaign_cli._task_path_mapping(  # noqa: SLF001
            [value, value], "artifact-manifest"
        )
