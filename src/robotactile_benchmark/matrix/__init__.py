"""Four-condition benchmark matrix construction and orchestration."""

from robotactile_benchmark.matrix.builders import (
    build_focused_phase_manifest,
    build_focused_restoration_manifest,
    build_primary_matrix_manifest,
)
from robotactile_benchmark.matrix.contracts import (
    MatrixCellSpec,
    MatrixComparison,
    MatrixGridKind,
    MatrixGridPoint,
)
from robotactile_benchmark.matrix.io import (
    MatrixResumeError,
    load_cell_receipt,
    load_matrix_manifest_file,
    load_matrix_summary,
)
from robotactile_benchmark.matrix.live_executor import (
    LiveMatrixCellExecutor,
    PairedLiveMatrixExecutor,
    live_matrix_artifact_path,
)
from robotactile_benchmark.matrix.live_executor_contracts import (
    LiveMatrixCellResources,
    LiveMatrixExecutionTemplate,
    LiveMatrixResourceResolver,
    materialize_live_matrix_request,
)
from robotactile_benchmark.matrix.live_run import (
    live_matrix_run_summary,
    run_live_matrix,
)
from robotactile_benchmark.matrix.live_run_config import (
    LiveMatrixResourceEntry,
    LiveMatrixRunConfig,
    load_live_matrix_run_config,
)
from robotactile_benchmark.matrix.manifest import MatrixManifest
from robotactile_benchmark.matrix.primary_generation import (
    generate_primary_matrix_bundle,
    load_primary_matrix_generation,
)
from robotactile_benchmark.matrix.primary_generation_contracts import (
    LoadedPrimaryMatrixGeneration,
    PrimaryMatrixGenerationError,
    PrimaryMatrixGenerationReceipt,
    PrimaryMatrixGenerationSpec,
)
from robotactile_benchmark.matrix.results import (
    CellArtifactReference,
    MatrixCellExecution,
    MatrixCellReceipt,
    MatrixCellStatus,
)
from robotactile_benchmark.matrix.runner import (
    MatrixBatchExecutor,
    MatrixCellExecutor,
    run_matrix,
    run_matrix_batch,
)
from robotactile_benchmark.matrix.states import MatrixCellState
from robotactile_benchmark.matrix.summary import MatrixRunResult, MatrixSummary

__all__ = [
    "CellArtifactReference",
    "LiveMatrixCellExecutor",
    "PairedLiveMatrixExecutor",
    "LiveMatrixCellResources",
    "LiveMatrixExecutionTemplate",
    "LiveMatrixResourceResolver",
    "LiveMatrixResourceEntry",
    "LiveMatrixRunConfig",
    "LoadedPrimaryMatrixGeneration",
    "MatrixCellExecution",
    "MatrixBatchExecutor",
    "MatrixCellExecutor",
    "MatrixCellReceipt",
    "MatrixCellSpec",
    "MatrixCellState",
    "MatrixCellStatus",
    "MatrixComparison",
    "MatrixGridKind",
    "MatrixGridPoint",
    "MatrixManifest",
    "MatrixResumeError",
    "MatrixRunResult",
    "MatrixSummary",
    "PrimaryMatrixGenerationError",
    "PrimaryMatrixGenerationReceipt",
    "PrimaryMatrixGenerationSpec",
    "build_focused_phase_manifest",
    "build_focused_restoration_manifest",
    "build_primary_matrix_manifest",
    "generate_primary_matrix_bundle",
    "load_cell_receipt",
    "load_matrix_manifest_file",
    "load_matrix_summary",
    "load_primary_matrix_generation",
    "load_live_matrix_run_config",
    "live_matrix_artifact_path",
    "materialize_live_matrix_request",
    "live_matrix_run_summary",
    "run_live_matrix",
    "run_matrix",
    "run_matrix_batch",
]
