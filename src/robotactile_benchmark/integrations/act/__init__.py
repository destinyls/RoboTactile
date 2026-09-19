"""First-class ACT model integration.

The runtime probe is exposed lazily because its implementation depends on the
shared runtime configuration module.  Eagerly importing it here would make a
low-level ``act.artifacts`` import re-enter ``runtime_config`` while that module
is still being initialized.
"""

from typing import TYPE_CHECKING, Any

from robotactile_benchmark.integrations.act.adapter import ACTPolicyAdapter
from robotactile_benchmark.integrations.act.artifacts import (
    ACTArtifactManifest,
    ACTLoadRequest,
    build_act_artifact_manifest,
    load_act_artifact_manifest,
)
from robotactile_benchmark.integrations.act.factory import load_act_adapter
from robotactile_benchmark.integrations.act.requests import (
    GeneratedACTRequest,
    build_official_act_request,
    write_official_act_request,
)

if TYPE_CHECKING:
    from robotactile_benchmark.integrations.act.runtime_probe import (
        OfficialACTRuntimeProbeResult,
        probe_official_act_runtime,
    )


def __getattr__(name: str) -> Any:
    """Lazily expose runtime-probe symbols without an import cycle."""

    if name in {"OfficialACTRuntimeProbeResult", "probe_official_act_runtime"}:
        from robotactile_benchmark.integrations.act import runtime_probe

        return getattr(runtime_probe, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ACTArtifactManifest",
    "ACTLoadRequest",
    "ACTPolicyAdapter",
    "GeneratedACTRequest",
    "OfficialACTRuntimeProbeResult",
    "build_act_artifact_manifest",
    "build_official_act_request",
    "load_act_adapter",
    "load_act_artifact_manifest",
    "probe_official_act_runtime",
    "write_official_act_request",
]
