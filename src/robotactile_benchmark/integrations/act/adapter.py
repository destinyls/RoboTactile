"""ACT implementation exposed through the common policy adapter contract."""

from robotactile_benchmark.policies.univtac_official_act import (
    OfficialUniVTACACTPolicy,
)

ACTPolicyAdapter = OfficialUniVTACACTPolicy

__all__ = ["ACTPolicyAdapter"]
