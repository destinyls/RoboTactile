"""First-class ACT integration facade tests."""

import subprocess
import sys

from robotactile_benchmark.integrations import PolicyAdapter, get_model_integration
from robotactile_benchmark.integrations.act import (
    ACTArtifactManifest,
    ACTPolicyAdapter,
    OfficialACTRuntimeProbeResult,
    load_act_adapter,
    probe_official_act_runtime,
)
from robotactile_benchmark.integrations.act.runtime_probe import (
    OfficialACTRuntimeProbeResult as RuntimeProbeResultImplementation,
)
from robotactile_benchmark.integrations.act.runtime_probe import (
    probe_official_act_runtime as runtime_probe_implementation,
)
from robotactile_benchmark.policies.univtac_official_act import (
    OfficialUniVTACACTPolicy,
)
from robotactile_benchmark.policies.univtac_official_act_loading import (
    OfficialUniVTACACTArtifactManifest,
    load_official_univtac_act_policy,
)


def test_act_facade_reuses_validated_implementation() -> None:
    assert ACTPolicyAdapter is OfficialUniVTACACTPolicy
    assert ACTArtifactManifest is OfficialUniVTACACTArtifactManifest
    assert load_act_adapter is load_official_univtac_act_policy
    assert OfficialACTRuntimeProbeResult is RuntimeProbeResultImplementation
    assert probe_official_act_runtime is runtime_probe_implementation


def test_act_is_a_first_class_policy_adapter() -> None:
    spec = get_model_integration("act")

    assert spec.factory_path.endswith(":load_act_adapter")
    assert spec.capabilities.matched_no_touch is True
    assert issubclass(ACTPolicyAdapter, object)
    assert PolicyAdapter.__name__ == "ClosedLoopPolicy"


def test_act_facade_is_safe_after_runtime_config_import() -> None:
    """Regression: importing ACT artifacts must not re-enter runtime_config."""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from robotactile_benchmark.integrations.runtime_config "
                "import ACTRuntimeArtifacts; "
                "from robotactile_benchmark.integrations.act "
                "import OfficialACTRuntimeProbeResult, probe_official_act_runtime"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
