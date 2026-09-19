"""N0-VTLA implementation exposed through the common PolicyAdapter contract."""

from robotactile_benchmark.policies.n0_vtla import OfficialN0VTLAPolicy

N0VTLAPolicyAdapter = OfficialN0VTLAPolicy

__all__ = ["N0VTLAPolicyAdapter"]
