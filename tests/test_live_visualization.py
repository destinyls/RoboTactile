"""Real-trace visualization export tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from robotactile_benchmark.closed_loop.contracts import ClosedLoopRunSpec
from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.closed_loop.runner import (
    run_closed_loop_trial_with_evidence,
)
from robotactile_benchmark.execution.capture_profiles import (
    LiveCaptureProfile,
    project_evidence_for_capture,
)
from robotactile_benchmark.execution.live_artifacts_preview import (
    build_live_preview_trace,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.trials import (
    Condition,
    TrialManifest,
    system_manifest_hash,
)
from robotactile_benchmark.visualization.live_artifact import (
    _export_loaded_visualization,
    _uint8_rgb,
)


def _trial(fault: FaultManifest) -> TrialManifest:
    base_system_id = "fake-tactile-policy"
    return TrialManifest(
        task="insert_HDMI",
        initial_seed=10,
        exogenous_seed=20,
        condition=Condition.FAULTED,
        base_system_id=base_system_id,
        executed_system_id=base_system_id,
        dataset_sha256="a" * 64,
        base_system_manifest_sha256=system_manifest_hash(
            base_system_id, "b" * 64, "c" * 64, "qpos8_next_step"
        ),
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        action_spec="qpos8_next_step",
        fault_manifest_sha256=fault.sha256,
        matched_no_touch_system_id=None,
    )


def _artifact(
    profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL,
) -> SimpleNamespace:
    fault = FaultManifest(
        operator_id="A1_stream_absence",
        severity_level=3,
        operator_seed=23,
        start_index=1,
        stop_index=4,
        sensor_slots=("left",),
        observability=Observability.DECLARED,
        parameters={},
    )
    trial = _trial(fault)
    spec = ClosedLoopRunSpec(
        prompt="insert the cable",
        success_predicate_id="fake-success-v1",
        max_control_cycles=4,
        max_observation_steps=5,
        execute_action_steps=1,
        wall_timeout_s=5.0,
    )
    evidence = run_closed_loop_trial_with_evidence(
        trial,
        spec,
        DeterministicFakeBackend(
            make_synthetic_episode(length=10),
            success_predicate_id=spec.success_predicate_id,
        ),
        DeterministicFakePolicy.for_trial(trial),
        fault_manifest=fault,
    )
    return SimpleNamespace(
        capture_profile=profile,
        evidence=project_evidence_for_capture(evidence, profile),
        fault_manifest=fault,
        preview_trace=(
            build_live_preview_trace(evidence.finalization)
            if profile is LiveCaptureProfile.PREVIEW
            else None
        ),
        root_receipt_sha256="f" * 64,
        trial=trial,
    )


def test_visualization_exports_source_bound_preview(tmp_path: Path) -> None:
    output = tmp_path / "visualization"
    result = _export_loaded_visualization(
        _artifact(),
        output,
        fps=20,
        stride=2,
        max_frames=2,
        video=False,
        ffmpeg="ffmpeg",
    )

    receipt = json.loads(result.receipt.read_text(encoding="utf-8"))
    with Image.open(result.preview) as preview:
        assert preview.size == (960, 720)
        assert preview.mode == "RGB"
    assert result.video is None
    assert result.rendered_frame_count == 2
    assert receipt["source_live_artifact_root_sha256"] == "f" * 64
    assert receipt["condition"] == "faulted"
    assert receipt["display_color_profile"]["channel_transform"] == "reverse_rgb"
    assert receipt["preview_step_index"] == 3
    assert receipt["simulator_qualification_claimed"] is False
    assert receipt["video_exported"] is False
    assert set(receipt["members"]) == {"preview.png"}


def test_visualization_uses_n0_twam_univtac_color_conversion() -> None:
    simulator_numeric = np.array([[[11, 22, 33]]], dtype=np.uint8)

    display_rgb = _uint8_rgb(simulator_numeric)

    assert display_rgb.tolist() == [[[33, 22, 11]]]


@pytest.mark.parametrize(
    "identity", [{"n0_action_per_frame": 4}, {"retrained_control_hz": 10}]
)
def test_retrained_visualization_preserves_model_input_rgb(tmp_path, identity):
    from robotactile_benchmark.visualization.live_artifact import _display_profile

    artifact = _artifact()
    artifact.request_identity = identity
    raw = np.array([[[11, 22, 33]]], dtype=np.uint8)
    assert _uint8_rgb(raw, profile=_display_profile(artifact)).tolist() == raw.tolist()
    result = _export_loaded_visualization(
        artifact,
        tmp_path / "retrained",
        fps=10,
        stride=1,
        max_frames=1,
        video=False,
        ffmpeg="ffmpeg",
    )
    receipt = json.loads(result.receipt.read_text())
    assert receipt["display_color_profile"]["channel_transform"] == "identity"


def test_preview_capture_visualizes_and_metrics_only_rejects_frames(
    tmp_path: Path,
) -> None:
    result = _export_loaded_visualization(
        _artifact(LiveCaptureProfile.PREVIEW),
        tmp_path / "preview-capture",
        fps=20,
        stride=1,
        max_frames=None,
        video=False,
        ffmpeg="ffmpeg",
    )
    receipt = json.loads(result.receipt.read_text(encoding="utf-8"))
    assert receipt["capture_profile"] == "preview_v1"
    assert receipt["selected_source_trace_indices"]

    with pytest.raises(ValueError, match="metrics_only_v1"):
        _export_loaded_visualization(
            _artifact(LiveCaptureProfile.METRICS_ONLY),
            tmp_path / "metrics-only",
            fps=20,
            stride=1,
            max_frames=None,
            video=False,
            ffmpeg="ffmpeg",
        )


def test_visualization_is_no_clobber_and_validates_options(tmp_path: Path) -> None:
    output = tmp_path / "visualization"
    output.mkdir()
    with pytest.raises(FileExistsError):
        _export_loaded_visualization(
            _artifact(),
            output,
            fps=20,
            stride=1,
            max_frames=None,
            video=False,
            ffmpeg="ffmpeg",
        )

    with pytest.raises(ValueError, match="fps"):
        _export_loaded_visualization(
            _artifact(),
            tmp_path / "invalid",
            fps=0,
            stride=1,
            max_frames=None,
            video=False,
            ffmpeg="ffmpeg",
        )
