"""Hash-bound official ACT artifact contracts."""

from robotactile_benchmark.policies.univtac_official_act_loading import (
    OfficialUniVTACACTArtifactManifest,
    OfficialUniVTACACTLoadRequest,
)

ACTArtifactManifest = OfficialUniVTACACTArtifactManifest
ACTLoadRequest = OfficialUniVTACACTLoadRequest

__all__ = ["ACTArtifactManifest", "ACTLoadRequest"]
