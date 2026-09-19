"""Dream-Tac implementation exposed through the common PolicyAdapter contract."""

from robotactile_benchmark.policies.dream_tac import OfficialDreamTacPolicy

DreamTacPolicyAdapter = OfficialDreamTacPolicy

__all__ = ["DreamTacPolicyAdapter"]
