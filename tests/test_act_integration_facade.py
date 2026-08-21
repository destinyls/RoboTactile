"""First-class ACT integration facade tests."""

from robotactile_benchmark.integrations import PolicyAdapter, get_model_integration
from robotactile_benchmark.integrations.act import (
    ACTArtifactManifest,
    ACTPolicyAdapter,
    load_act_adapter,
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


def test_act_is_a_first_class_policy_adapter() -> None:
    spec = get_model_integration("act")

    assert spec.factory_path.endswith(":load_act_adapter")
    assert spec.capabilities.matched_no_touch is True
    assert issubclass(ACTPolicyAdapter, object)
    assert PolicyAdapter.__name__ == "ClosedLoopPolicy"
