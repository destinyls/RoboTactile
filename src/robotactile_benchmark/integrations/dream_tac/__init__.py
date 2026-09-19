"""First-class Dream-Tac integration facade."""

from robotactile_benchmark.integrations.dream_tac.adapter import DreamTacPolicyAdapter
from robotactile_benchmark.integrations.dream_tac.artifacts import (
    DreamTacArtifactFile,
    DreamTacArtifactManifest,
    build_dream_tac_artifact_manifest,
    load_dream_tac_artifact_manifest,
    validate_dream_tac_artifact,
)
from robotactile_benchmark.integrations.dream_tac.factory import load_dream_tac_adapter
from robotactile_benchmark.integrations.dream_tac.transport import (
    DreamTacTransportError,
    OfficialDreamTacClient,
)
from robotactile_benchmark.policies.dream_tac import DreamTacGripperMapping

__all__ = [
    "DreamTacArtifactFile",
    "DreamTacArtifactManifest",
    "DreamTacGripperMapping",
    "DreamTacPolicyAdapter",
    "DreamTacTransportError",
    "OfficialDreamTacClient",
    "build_dream_tac_artifact_manifest",
    "load_dream_tac_adapter",
    "load_dream_tac_artifact_manifest",
    "validate_dream_tac_artifact",
]
