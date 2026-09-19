"""Train-only UniVTAC materialization and source-bound Dream-Tac launchers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .training_constants import DREAM_TAC_TRAINING_PROTOCOL

if TYPE_CHECKING:
    from .kinematics import pack_dream_tac_trajectories


def __getattr__(name: str) -> object:
    """Load array-dependent helpers only when their public symbol is requested."""

    if name == "pack_dream_tac_trajectories":
        from .kinematics import pack_dream_tac_trajectories

        globals()[name] = pack_dream_tac_trajectories
        return pack_dream_tac_trajectories
    raise AttributeError(name)


__all__ = ["DREAM_TAC_TRAINING_PROTOCOL", "pack_dream_tac_trajectories"]
