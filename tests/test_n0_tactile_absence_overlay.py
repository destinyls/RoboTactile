"""Protocol tests for the opt-in pinned N0 tactile-absence overlay."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from robotactile_benchmark.integrations.n0_twam.tactile_absence_overlay import (
    N0_OBSERVED_TACTILE_ABSENCE_OVERLAY_ID,
    install_n0_observed_tactile_absence_overlay,
)


def test_overlay_enforces_immutable_tensor_free_episode_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeServer:
        def infer(self, obs: dict[str, object]) -> dict[str, object]:
            return {"original": True, "obs": obs}

        def _encode_tactile_obs(self, obs: dict[str, object]) -> object:
            return {"encoded": obs}

        def _align_grounded_tactile_frames(
            self, cold_seed: object, continuation: object, target_frames: int
        ) -> object:
            return cold_seed, continuation, target_frames

        def _prepare_latent_input(
            self, *_args: object, **_kwargs: object
        ) -> dict[str, dict[str, object]]:
            return {"action_res_lst": {"noisy_latents": object()}}

    class FakeModel:
        def forward(
            self,
            input_dict: dict[str, object],
            update_cache: int = 0,
            cache_name: str = "pos",
            action_mode: bool = False,
            train_mode: bool = False,
        ) -> tuple[object, ...]:
            return input_dict, update_cache, cache_name, action_mode, train_mode

    server_module = ModuleType("n0_twam.n0_twam_server")
    server_module.TWAM_Server = FakeServer  # type: ignore[attr-defined]
    model_module = ModuleType("models.model")
    model_module.WanTransformer3DModel = FakeModel  # type: ignore[attr-defined]
    torch_module = ModuleType("torch")
    einops_module = ModuleType("einops")
    einops_module.rearrange = lambda value, *_args, **_kwargs: value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "n0_twam.n0_twam_server", server_module)
    monkeypatch.setitem(sys.modules, "models.model", model_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "einops", einops_module)

    overlay_id = install_n0_observed_tactile_absence_overlay()
    assert overlay_id == N0_OBSERVED_TACTILE_ABSENCE_OVERLAY_ID
    assert install_n0_observed_tactile_absence_overlay() == overlay_id

    server = FakeServer()
    server.infer({"reset": True})
    server.infer({"tactile_cond_drop": True, "obs": {}})
    assert server._encode_tactile_obs({"obs": {}}) is None
    prepared = server._prepare_latent_input()
    assert prepared["action_res_lst"]["tactile_cond_drop"] is True
    assert server._align_grounded_tactile_frames(None, None, 4) is None
    with pytest.raises(RuntimeError, match="cannot change"):
        server.infer({"obs": {}, "tactile": {}})

    model = FakeModel()
    assert model.forward({"normal": True})[0] == {"normal": True}
