"""N0-TWAM implementation exposed through the common policy adapter contract."""

from robotactile_benchmark.policies.n0_official import OfficialN0Policy

N0TWAMPolicyAdapter = OfficialN0Policy

__all__ = ["N0TWAMPolicyAdapter"]
