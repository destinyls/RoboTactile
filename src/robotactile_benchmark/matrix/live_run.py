"""Public resumable live-matrix vertical slice."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from robotactile_benchmark.closed_loop.interfaces import ClosedLoopPolicy
from robotactile_benchmark.execution.live_univtac import (
    LiveBackendFactory,
    LivePolicyFactory,
)
from robotactile_benchmark.execution.loading import LoadedLiveUniVTACRun
from robotactile_benchmark.execution.official_act import (
    OfficialACTPolicyLoader,
    build_official_act_live_binding,
    make_official_act_policy_factory,
)
from robotactile_benchmark.matrix.live_executor import LiveMatrixCellExecutor
from robotactile_benchmark.matrix.live_run_config import LiveMatrixRunConfig
from robotactile_benchmark.matrix.manifest import MatrixManifest
from robotactile_benchmark.matrix.results import MatrixCellStatus
from robotactile_benchmark.matrix.runner import run_matrix
from robotactile_benchmark.matrix.summary import MatrixRunResult

LIVE_MATRIX_EVIDENCE_LEVEL = "unqualified_live_univtac_matrix_v1"


def run_live_matrix(
    output: Path,
    manifest: MatrixManifest,
    config: LiveMatrixRunConfig,
    *,
    max_new_cells: Optional[int] = None,
    backend_factory: Optional[LiveBackendFactory] = None,
    policy_factory: Optional[LivePolicyFactory] = None,
    policy_loader: Optional[OfficialACTPolicyLoader] = None,
) -> MatrixRunResult:
    """Run or resume cells while preserving strict content-addressed artifacts."""

    if type(manifest) is not MatrixManifest:
        raise TypeError("manifest must be an exact MatrixManifest")
    if type(config) is not LiveMatrixRunConfig:
        raise TypeError("config must be an exact LiveMatrixRunConfig")
    if config.matrix_manifest_sha256 != manifest.sha256:
        raise ValueError("live matrix config does not bind the matrix manifest")
    selected_policy = policy_factory or _official_policy_factory(config, policy_loader)
    executor = LiveMatrixCellExecutor(
        matrix_output=Path(output),
        template=config.template(),
        resource_resolver=config.resources_for,
        backend_factory=backend_factory,
        policy_factory=selected_policy,
    )
    return run_matrix(Path(output), manifest, executor, max_new_cells=max_new_cells)


def _official_policy_factory(
    config: LiveMatrixRunConfig,
    policy_loader: Optional[OfficialACTPolicyLoader],
) -> LivePolicyFactory:
    def factory(loaded: LoadedLiveUniVTACRun) -> ClosedLoopPolicy:
        binding = build_official_act_live_binding(
            loaded.request,
            artifact_root=config.official_act_artifact_root,
            stats_sha256=config.stats_sha256,
            encoder_sha256=config.encoder_sha256,
        )
        return make_official_act_policy_factory(binding, policy_loader=policy_loader)(
            loaded
        )

    return factory


def live_matrix_run_summary(
    manifest: MatrixManifest, result: MatrixRunResult
) -> dict[str, object]:
    """Return one canonical CLI summary without upgrading evidence claims."""

    counts = {
        status.value: sum(state.status is status for state in result.states)
        for status in MatrixCellStatus
    }
    return {
        "complete": result.summary is not None,
        "evidence_level": LIVE_MATRIX_EVIDENCE_LEVEL,
        "executed_cell_count": result.executed_cell_count,
        "manifest_sha256": manifest.sha256,
        "matrix_id": manifest.matrix_id,
        "pending_cell_count": result.pending_cell_count,
        "reused_cell_count": result.reused_cell_count,
        "simulator_qualification_claimed": False,
        "status_counts": counts,
    }
