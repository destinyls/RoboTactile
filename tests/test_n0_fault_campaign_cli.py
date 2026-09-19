"""Public N0 fault campaign CLI tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robotactile_benchmark import cli
from robotactile_benchmark.n0_fault_campaign import cli as campaign_cli


def test_generate_cli_builds_full_default_contract(
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
                    campaign_id="pilot",
                    sha256="b" * 64,
                    pair_count=1,
                    cell_count=71,
                    live_request_count=61,
                    unsupported_contract_count=10,
                ),
            ),
        )

    monkeypatch.setattr(campaign_cli, "generate_n0_fault_campaign_bundle", generate)
    args = cli._parser().parse_args(  # noqa: SLF001
        [
            "generate-n0-fault-campaign",
            "--output",
            str(tmp_path / "campaign"),
            "--campaign-id",
            "pilot",
            "--base-clean-request",
            str(tmp_path / "clean.json"),
            "--operator-seed-master",
            "17",
            "--fault-start-index",
            "20",
            "--fault-stop-index",
            "80",
            "--rest-reference",
            f"insert_tube={tmp_path / 'rest'}",
        ]
    )

    payload = campaign_cli.handle_n0_fault_campaign_command(args)

    assert payload is not None
    spec = captured["spec"]
    assert len(spec.operator_ids) == 14
    assert spec.severity_levels == (1, 2, 3, 4, 5)
    assert spec.fault_onset_mode == "fixed_v1"
    assert spec.rest_reference_artifacts["insert_tube"] == (tmp_path / "rest")


def test_generate_cli_accepts_early_seeded_onset_without_fixed_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def generate(output: Path, spec: object) -> tuple[str, object]:
        captured["spec"] = spec
        return (
            "created",
            SimpleNamespace(
                root=output,
                receipt_file_sha256="a" * 64,
                manifest=SimpleNamespace(
                    campaign_id="early-pilot",
                    sha256="b" * 64,
                    pair_count=1,
                    cell_count=2,
                    live_request_count=2,
                    unsupported_contract_count=0,
                ),
            ),
        )

    monkeypatch.setattr(campaign_cli, "generate_n0_fault_campaign_bundle", generate)
    args = cli._parser().parse_args(
        [
            "generate-n0-fault-campaign",
            "--output",
            str(tmp_path / "campaign"),
            "--campaign-id",
            "early-pilot",
            "--base-clean-request",
            str(tmp_path / "clean.json"),
            "--operator",
            "F2_spatial_sensitivity_loss",
            "--severity",
            "5",
            "--operator-seed-master",
            "3",
            "--fault-onset-mode",
            "early_random_onset_v1",
            "--fault-onset-max-index",
            "8",
            "--fault-stop-index",
            "41",
        ]
    )
    campaign_cli.handle_n0_fault_campaign_command(args)
    spec = captured["spec"]
    assert spec.fault_onset_mode == "early_random_onset_v1"
    assert spec.fault_onset_max_index == 8
    assert spec.fault_start_index == 0


def test_run_cli_defaults_to_full_capture_and_training_contract(tmp_path: Path) -> None:
    args = cli._parser().parse_args(  # noqa: SLF001
        [
            "run-n0-fault-campaign",
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--integration-config",
            str(tmp_path / "config.json"),
            "--n0-source-root",
            str(tmp_path / "N0-TWAM"),
        ]
    )

    assert args.capture_profile == "paper_full_v1"
    assert args.action_execution_contract == "robotactile_n0_training_60hz_ee_v1"


def test_tactile_null_cli_selects_only_full_horizon_black_frame_ablation(
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
                    campaign_id="null-pilot",
                    sha256="b" * 64,
                    pair_count=1,
                    cell_count=2,
                    live_request_count=2,
                    unsupported_contract_count=0,
                ),
            ),
        )

    monkeypatch.setattr(campaign_cli, "generate_n0_fault_campaign_bundle", generate)
    args = cli._parser().parse_args(  # noqa: SLF001
        [
            "generate-n0-fault-campaign",
            "--output",
            str(tmp_path / "campaign"),
            "--campaign-id",
            "null-pilot",
            "--base-clean-request",
            str(tmp_path / "clean.json"),
            "--operator-seed-master",
            "17",
            "--fault-start-index",
            "0",
            "--fault-stop-index",
            "600",
            "--diagnostic-tactile-null",
        ]
    )

    payload = campaign_cli.handle_n0_fault_campaign_command(args)

    assert payload is not None
    assert payload["diagnostic_ablation"] == "tactile_null_black_frame_v1"
    assert payload["paper_s1_s5_claim"] is False
    spec = captured["spec"]
    assert spec.operator_ids == ("F1_global_response_drift",)
    assert spec.severity_levels == (5,)
    assert spec.fault_start_index == 0
    assert spec.fault_stop_index == 600
    assert spec.severity_registry == "diagnostic_tactile_null_black_v1"
    assert spec.rest_reference_artifacts == {}


def test_observed_tactile_absence_cli_selects_live_a1(
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
                    campaign_id="absence-pilot",
                    sha256="b" * 64,
                    pair_count=1,
                    cell_count=2,
                    live_request_count=2,
                    unsupported_contract_count=0,
                ),
            ),
        )

    monkeypatch.setattr(campaign_cli, "generate_n0_fault_campaign_bundle", generate)
    args = cli._parser().parse_args(  # noqa: SLF001
        [
            "generate-n0-fault-campaign",
            "--output",
            str(tmp_path / "campaign"),
            "--campaign-id",
            "absence-pilot",
            "--base-clean-request",
            str(tmp_path / "clean.json"),
            "--operator-seed-master",
            "17",
            "--fault-start-index",
            "0",
            "--fault-stop-index",
            "600",
            "--diagnostic-observed-tactile-absence",
        ]
    )

    payload = campaign_cli.handle_n0_fault_campaign_command(args)

    assert payload is not None
    assert payload["diagnostic_ablation"] == "observed_tactile_absent_v1"
    assert payload["paper_s1_s5_claim"] is False
    spec = captured["spec"]
    assert spec.operator_ids == ("A1_stream_absence",)
    assert spec.severity_levels == (5,)
    assert spec.severity_registry == "diagnostic_observed_tactile_absence_v1"


def test_report_cli_exposes_source_bound_output_contract(tmp_path: Path) -> None:
    args = cli._parser().parse_args(  # noqa: SLF001
        [
            "report-n0-fault-campaign",
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--output",
            str(tmp_path / "report"),
            "--bootstrap-seed",
            "2027",
        ]
    )

    assert args.campaign_root == tmp_path / "campaign"
    assert args.output == tmp_path / "report"
    assert args.bootstrap_seed == 2027


def test_duplicate_rest_reference_is_rejected(tmp_path: Path) -> None:
    value = f"insert_tube={tmp_path / 'rest'}"
    with pytest.raises(ValueError, match="duplicate"):
        campaign_cli._rest_reference_mapping([value, value])  # noqa: SLF001
