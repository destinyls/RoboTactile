"""Opt-in inference overlay for training-consistent N0 tactile CFG dropout."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any, Callable, cast

N0_OBSERVED_TACTILE_ABSENCE_OVERLAY_ID = "n0_observed_tactile_absence_overlay_v1"
_OVERLAY_MARKER = "_robotactile_observed_tactile_absence_overlay_v1"
_EPISODE_MODE = "_robotactile_observed_tactile_absent"


def _callable_attr(owner: Any, name: str) -> Callable[..., Any]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise RuntimeError(f"pinned N0 runtime is missing callable {name}")
    return cast(Callable[..., Any], value)


def install_n0_observed_tactile_absence_overlay() -> str:
    """Patch the loaded pinned server only for explicit tactile-drop requests."""

    server_module = importlib.import_module("n0_twam.n0_twam_server")
    model_module = importlib.import_module("models.model")
    torch = importlib.import_module("torch")
    einops = importlib.import_module("einops")
    rearrange = _callable_attr(einops, "rearrange")

    server_type = getattr(server_module, "TWAM_Server", None)
    model_type = getattr(model_module, "WanTransformer3DModel", None)
    if not isinstance(server_type, type) or not isinstance(model_type, type):
        raise RuntimeError("pinned N0 server/model classes are unavailable")
    if getattr(server_type, _OVERLAY_MARKER, False):
        return N0_OBSERVED_TACTILE_ABSENCE_OVERLAY_ID

    original_server_infer = _callable_attr(server_type, "infer")
    original_encode_tactile = _callable_attr(server_type, "_encode_tactile_obs")
    original_align_tactile = _callable_attr(
        server_type, "_align_grounded_tactile_frames"
    )
    original_prepare_input = _callable_attr(server_type, "_prepare_latent_input")
    original_model_forward = _callable_attr(model_type, "forward")

    def server_infer(self: Any, obs: Mapping[str, Any]) -> Any:
        if not isinstance(obs, Mapping):
            raise TypeError("N0 observation must be a mapping")
        if obs.get("reset", False):
            setattr(self, _EPISODE_MODE, None)
            return original_server_infer(self, obs)
        requested = obs.get("tactile_cond_drop", False)
        if type(requested) is not bool:
            raise TypeError("tactile_cond_drop must be boolean")
        active = getattr(self, _EPISODE_MODE, None)
        if active is None:
            setattr(self, _EPISODE_MODE, requested)
        elif active is not requested:
            raise RuntimeError("N0 tactile mode cannot change within an episode")
        if requested and "tactile" in obs:
            raise ValueError("tactile-drop request cannot carry tactile tensors")
        return original_server_infer(self, obs)

    def encode_tactile(self: Any, obs: Mapping[str, Any]) -> Any:
        if getattr(self, _EPISODE_MODE, False):
            if "tactile" in obs:
                raise ValueError("absent tactile mode received tactile tensors")
            return None
        return original_encode_tactile(self, obs)

    def align_tactile(
        self: Any,
        cold_seed: Any,
        continuation: Any,
        target_frames: int,
    ) -> Any:
        if getattr(self, _EPISODE_MODE, False):
            if cold_seed is not None or continuation is not None:
                raise ValueError("absent tactile mode received encoded latents")
            if not isinstance(target_frames, int) or target_frames < 1:
                raise ValueError("grounding target_frames must be positive")
            return None
        return original_align_tactile(self, cold_seed, continuation, target_frames)

    def prepare_input(self: Any, *args: Any, **kwargs: Any) -> Any:
        prepared = original_prepare_input(self, *args, **kwargs)
        if getattr(self, _EPISODE_MODE, False):
            for branch in prepared.values():
                branch["tactile_cond_drop"] = True
        return prepared

    def model_forward(
        self: Any,
        input_dict: Mapping[str, Any],
        update_cache: int = 0,
        cache_name: str = "pos",
        action_mode: bool = False,
        train_mode: bool = False,
    ) -> Any:
        tactile_drop = input_dict.get("tactile_cond_drop", False)
        if type(tactile_drop) is not bool:
            raise TypeError("tactile_cond_drop must be boolean")
        if train_mode or not tactile_drop:
            return original_model_forward(
                self,
                input_dict,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
                train_mode=train_mode,
            )
        forbidden = {
            "tactile_global_latent",
            "tactile_local_latent",
            "tactile_sensor_ids",
            "tactile_noisy_latent",
            "tactile_timesteps",
        }
        overlap = forbidden.intersection(input_dict)
        if overlap:
            raise ValueError(
                f"tactile-drop inference received tactile entries: {sorted(overlap)}"
            )
        if hasattr(self, "blocks"):
            self._set_block_slice_masks(self.blocks, None, None)

        noisy_latents = input_dict["noisy_latents"]
        if action_mode:
            hidden = rearrange(noisy_latents, "b c f h w -> b (f h w) c")
            hidden = self.action_embedder(hidden)
        else:
            hidden = rearrange(
                noisy_latents,
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            hidden = self.patch_embedding_mlp(hidden)
        text_hidden = self._encode_text_condition(input_dict["text_emb"])
        main_token_count = hidden.shape[1]
        rotary_emb = self.rope(input_dict["grid_id"])[:, :, None]
        patch_h, patch_w = (
            (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        )
        latent_time_steps = torch.repeat_interleave(
            input_dict["timesteps"],
            (noisy_latents.shape[-2] // patch_h) * (noisy_latents.shape[-1] // patch_w),
            dim=1,
        )
        condition_embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder
        )
        temb, timestep_proj = condition_embedder(latent_time_steps, dtype=hidden.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        hidden = self._run_main_blocks(
            hidden,
            text_hidden,
            timestep_proj,
            temb,
            rotary_emb,
            update_cache,
            cache_name,
            action_mode,
            main_token_count,
            0,
        )
        scale_shift = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(scale_shift, "b l n c -> b n l c").chunk(2, dim=1)
        hidden = (
            self.norm_out(hidden.float()) * (1.0 + scale.to(hidden.device).squeeze(1))
            + shift.to(hidden.device).squeeze(1)
        ).type_as(hidden)
        if action_mode:
            return self.action_proj_out(hidden)
        hidden = self.proj_out(hidden)
        return rearrange(
            hidden,
            "b l (n c) -> b (l n) c",
            n=math.prod(self.patch_size),
        )

    server_class = cast(Any, server_type)
    model_class = cast(Any, model_type)
    server_class.infer = server_infer
    server_class._encode_tactile_obs = encode_tactile
    server_class._align_grounded_tactile_frames = align_tactile
    server_class._prepare_latent_input = prepare_input
    model_class.forward = model_forward
    setattr(server_type, _OVERLAY_MARKER, True)
    return N0_OBSERVED_TACTILE_ABSENCE_OVERLAY_ID


__all__ = [
    "N0_OBSERVED_TACTILE_ABSENCE_OVERLAY_ID",
    "install_n0_observed_tactile_absence_overlay",
]
