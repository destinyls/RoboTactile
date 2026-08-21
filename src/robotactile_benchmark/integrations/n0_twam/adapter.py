"""N0-TWAM implementation exposed through the common policy adapter contract."""

from robotactile_benchmark.policies.n0 import N0Policy

N0TWAMPolicyAdapter = N0Policy

__all__ = ["N0TWAMPolicyAdapter"]
