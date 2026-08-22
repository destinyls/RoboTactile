"""Public repo-contained deployment contracts and helpers."""

from robotactile_benchmark.deployment.contracts import (
    DEPLOYMENT_LAYOUT_EVIDENCE_LEVEL,
    DEPLOYMENT_LAYOUT_ID,
    DEPLOYMENT_LAYOUT_SEMANTIC_VERSION,
    DeploymentLayoutError,
    DeploymentLayoutReceipt,
)
from robotactile_benchmark.deployment.doctor import (
    DEPLOYMENT_PROFILES,
    DeploymentDoctorCheck,
    DeploymentDoctorResult,
    diagnose_deployment,
)
from robotactile_benchmark.deployment.layout import (
    DEPLOYMENT_ROOT_ENV,
    LAYOUT_RECEIPT_RELATIVE_PATH,
    DeploymentLayout,
    discover_repository_root,
    initialize_deployment_layout,
    load_deployment_layout_receipt,
    resolve_deployment_root,
)

__all__ = [
    "DEPLOYMENT_LAYOUT_EVIDENCE_LEVEL",
    "DEPLOYMENT_LAYOUT_ID",
    "DEPLOYMENT_LAYOUT_SEMANTIC_VERSION",
    "DEPLOYMENT_PROFILES",
    "DEPLOYMENT_ROOT_ENV",
    "LAYOUT_RECEIPT_RELATIVE_PATH",
    "DeploymentDoctorCheck",
    "DeploymentDoctorResult",
    "DeploymentLayout",
    "DeploymentLayoutError",
    "DeploymentLayoutReceipt",
    "diagnose_deployment",
    "discover_repository_root",
    "initialize_deployment_layout",
    "load_deployment_layout_receipt",
    "resolve_deployment_root",
]
