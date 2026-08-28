"""End-to-end CLI behavior for clean campaign build and partial reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_clean_campaign_support import (
    make_layout,
    write_artifact_and_attempt,
    write_clean_request,
)

from robotactile_benchmark.clean_baseline import (
    CleanCampaignProtocol,
    build_clean_campaign_manifest,
    write_clean_campaign_manifest,
)
from robotactile_benchmark.cli import main
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile


def test_cli_build_is_no_clobber_and_report_requires_explicit_partial(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = make_layout(tmp_path)
    request, _ = write_clean_request(layout)
    request_root = request.parents[3]
    build_args = (
        "clean-campaign-build",
        "--root",
        str(layout.root),
        "--campaign-id",
        "campaign-test",
        "--protocol",
        "diagnostic_v1",
        "--master-seed",
        "20260823",
        "--request-root",
        str(request_root),
        "--bootstrap-resamples",
        "20",
    )

    assert main(build_args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["created"] is True
    assert main(build_args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["created"] is False

    manifest = Path(first["campaign_manifest"])
    output = layout.outputs / "clean-campaigns/campaign-test/summary.json"
    report_args = (
        "clean-campaign-report",
        "--root",
        str(layout.root),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
    )
    assert main(report_args) == 2
    incomplete = json.loads(capsys.readouterr().out)
    assert incomplete["status"] == "incomplete_no_summary_written"
    assert not output.exists()

    assert main((*report_args, "--allow-partial")) == 0
    partial = json.loads(capsys.readouterr().out)
    assert partial["statistically_complete"] is False
    assert partial["paper_claim_eligible"] is False
    assert output.is_file()


def test_cli_manifest_output_refuses_different_bootstrap_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = make_layout(tmp_path)
    request, _ = write_clean_request(layout)
    common = (
        "clean-campaign-build",
        "--root",
        str(layout.root),
        "--campaign-id",
        "campaign-test",
        "--protocol",
        "diagnostic_v1",
        "--request",
        str(request),
    )
    assert main((*common, "--master-seed", "20260823", "--bootstrap-seed", "1")) == 0
    capsys.readouterr()

    with pytest.raises(FileExistsError, match="refusing to replace"):
        main((*common, "--master-seed", "20260823", "--bootstrap-seed", "2"))


def test_cli_report_accepts_compact_capture_only_with_explicit_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = make_layout(tmp_path)
    request_path, request = write_clean_request(layout)
    manifest = build_clean_campaign_manifest(
        deployment_root=layout.root,
        request_paths=(request_path,),
        campaign_id="campaign-test",
        protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
        master_seed=20260823,
        bootstrap_resamples=20,
    )
    manifest_path = (
        layout.requests / "clean-campaigns/campaign-test/campaign_manifest.json"
    )
    write_clean_campaign_manifest(manifest_path, manifest)
    spec = manifest.trials[0]
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        request,
        manifest_ordinal=spec.ordinal,
        request_file_sha256=spec.request_file_sha256,
        capture_profile=LiveCaptureProfile.METRICS_ONLY,
    )
    output = layout.outputs / "clean-campaigns/campaign-test/compact-summary.json"
    common = (
        "clean-campaign-report",
        "--root",
        str(layout.root),
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
    )

    assert main(common) == 2
    rejected = json.loads(capsys.readouterr().out)
    assert rejected["protocol_invalid_count"] == 1
    assert not output.exists()

    assert main((*common, "--allow-compact-capture")) == 0
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["loaded_artifact_count"] == 1
    assert accepted["statistically_complete"] is True
    assert output.is_file()
