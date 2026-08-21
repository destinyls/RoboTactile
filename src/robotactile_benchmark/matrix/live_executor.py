"""Production single-cell live executor with content-addressed artifacts."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from robotactile_benchmark.execution.contracts import (
    ArtifactExportStatus,
    LiveExecutionUnavailableError,
    LiveUniVTACRunRequest,
)
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.execution.live_univtac import (
    LiveBackendFactory,
    LivePolicyFactory,
    LiveUniVTACExecutionResult,
    N0TransportFactory,
    execute_live_univtac_run,
)
from robotactile_benchmark.matrix.contracts import (
    MatrixCellSpec,
    require_sha256,
)
from robotactile_benchmark.matrix.live_executor_contracts import (
    LiveMatrixExecutionTemplate,
    LiveMatrixResourceResolver,
    materialize_live_matrix_request,
)
from robotactile_benchmark.matrix.results import MatrixCellExecution
from robotactile_benchmark.policies.act_loading import ArtifactUnavailableError


def live_matrix_artifact_path(matrix_output: Path, root_sha256: str) -> Path:
    """Resolve the only valid storage location for one verified live root."""

    root = require_sha256(root_sha256, "root_sha256")
    return Path(matrix_output).absolute() / "artifacts" / root


@dataclass(frozen=True)
class LiveMatrixCellExecutor:
    """Execute one cell and retain only independently reloadable evidence."""

    matrix_output: Path
    template: LiveMatrixExecutionTemplate
    resource_resolver: LiveMatrixResourceResolver
    backend_factory: Optional[LiveBackendFactory] = None
    policy_factory: Optional[LivePolicyFactory] = None
    n0_transport_factory: Optional[N0TransportFactory] = None

    def __post_init__(self) -> None:
        output = Path(self.matrix_output).absolute()
        if output.is_symlink() or (output.exists() and not output.is_dir()):
            raise ValueError("matrix output must be a real directory")
        if type(self.template) is not LiveMatrixExecutionTemplate:
            raise TypeError("template must be an exact LiveMatrixExecutionTemplate")
        if not callable(self.resource_resolver):
            raise TypeError("resource_resolver must be callable")
        for name in ("backend_factory", "policy_factory", "n0_transport_factory"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")
        object.__setattr__(self, "matrix_output", output)

    def __call__(self, cell: MatrixCellSpec) -> MatrixCellExecution:
        if type(cell) is not MatrixCellSpec:
            raise TypeError("cell must be an exact MatrixCellSpec")
        artifacts = _prepare_artifacts_directory(self.matrix_output)
        with tempfile.TemporaryDirectory(
            prefix=f".cell-{cell.sha256[:12]}-", dir=artifacts
        ) as temporary:
            staged = Path(temporary) / "artifact"
            try:
                resources = self.resource_resolver(cell)
                request = materialize_live_matrix_request(
                    cell, self.template, resources, output_dir=staged
                )
                execution = self._execute(request)
            except LiveExecutionUnavailableError as error:
                return MatrixCellExecution.unsupported(
                    f"live_execution_unavailable:{error.code}"
                )
            except ArtifactUnavailableError as error:
                return MatrixCellExecution.unsupported(error.code)
            if (
                execution.artifact_export is not ArtifactExportStatus.EXPORTED
                or execution.artifact_receipt is None
            ):
                raise RuntimeError("live matrix cell did not export a typed artifact")
            staged_artifact = load_live_univtac_artifact(staged)
            _validate_execution_links(cell, execution, staged_artifact)
            published = _publish_content_addressed(artifacts, staged, staged_artifact)
            return MatrixCellExecution.from_closed_loop(
                published.evidence.result,
                root_receipt_sha256=published.external_root_sha256,
                evidence_level=published.root_receipt.evidence_level,
            )

    def _execute(self, request: LiveUniVTACRunRequest) -> LiveUniVTACExecutionResult:
        if self.backend_factory is None:
            return execute_live_univtac_run(
                request,
                policy_factory=self.policy_factory,
                n0_transport_factory=self.n0_transport_factory,
                artifact_exporter=write_live_univtac_artifact,
            )
        return execute_live_univtac_run(
            request,
            backend_factory=self.backend_factory,
            policy_factory=self.policy_factory,
            n0_transport_factory=self.n0_transport_factory,
            artifact_exporter=write_live_univtac_artifact,
        )


def _prepare_artifacts_directory(matrix_output: Path) -> Path:
    if matrix_output.is_symlink() or (
        matrix_output.exists() and not matrix_output.is_dir()
    ):
        raise ValueError("matrix output must be a real directory")
    matrix_output.mkdir(parents=True, exist_ok=True)
    if matrix_output.is_symlink() or not matrix_output.is_dir():
        raise ValueError("matrix output must remain a real directory")
    artifacts = matrix_output / "artifacts"
    if artifacts.is_symlink() or (artifacts.exists() and not artifacts.is_dir()):
        raise ValueError("matrix artifact root must be a real directory")
    artifacts.mkdir(exist_ok=True)
    return artifacts


def _validate_execution_links(
    cell: MatrixCellSpec,
    execution: LiveUniVTACExecutionResult,
    artifact: LoadedLiveUniVTACArtifact,
) -> None:
    result = execution.evidence.result
    receipt = execution.artifact_receipt
    if receipt is None:
        raise RuntimeError("live matrix execution lost its artifact receipt")
    checks = (
        artifact.trial == cell.trial,
        artifact.fault_manifest == cell.fault_manifest,
        artifact.evidence.result == result,
        artifact.root_receipt.result_sha256 == result.sha256,
        receipt.result_sha256 == result.sha256,
        receipt.trial_manifest_sha256 == cell.trial.sha256,
        receipt.run_content_sha256 == artifact.run_content_sha256,
        artifact.root_receipt.simulator_qualification_claimed is False,
    )
    if not all(checks):
        raise RuntimeError("live matrix artifact cross-link mismatch")


def _publish_content_addressed(
    artifacts: Path,
    staged: Path,
    staged_artifact: LoadedLiveUniVTACArtifact,
) -> LoadedLiveUniVTACArtifact:
    root_sha256 = staged_artifact.external_root_sha256
    target = live_matrix_artifact_path(artifacts.parent, root_sha256)
    if target.exists() or target.is_symlink():
        existing = load_live_univtac_artifact(target)
        if (
            existing.external_root_sha256 != root_sha256
            or existing.root_receipt != staged_artifact.root_receipt
        ):
            raise FileExistsError("content-addressed live artifact cannot be clobbered")
        return existing
    try:
        os.rename(staged, target)
    except FileExistsError:
        existing = load_live_univtac_artifact(target)
        if existing.root_receipt != staged_artifact.root_receipt:
            raise FileExistsError(
                "content-addressed live artifact race disagrees"
            ) from None
        return existing
    published = load_live_univtac_artifact(target)
    if published.external_root_sha256 != root_sha256:
        raise RuntimeError("published live matrix root hash changed")
    return published
