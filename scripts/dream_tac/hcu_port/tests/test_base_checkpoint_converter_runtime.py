"""Focused runtime-adapter tests for the Dream-Tac base converter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.dream_tac.hcu_port import base_checkpoint_converter_runtime as runtime


@dataclass
class _Tokenizer:
    vae_pth: str
    load_mean_std: bool


@dataclass
class _ModelConfig:
    tokenizer: _Tokenizer


@dataclass
class _Model:
    config: _ModelConfig


@dataclass
class _UpstreamConfig:
    model: _Model


def _config(*, load_mean_std: bool) -> _UpstreamConfig:
    return _UpstreamConfig(
        model=_Model(
            config=_ModelConfig(
                tokenizer=_Tokenizer(
                    vae_pth="/path/to/tokenizer.pth",
                    load_mean_std=load_mean_std,
                )
            )
        )
    )


def test_bind_tokenizer_checkpoint_replaces_placeholder(tmp_path: Path) -> None:
    config = _config(load_mean_std=False)
    checkpoint = tmp_path / "tokenizer.pth"

    runtime._bind_tokenizer_checkpoint(config, checkpoint)

    assert config.model.config.tokenizer.vae_pth == str(checkpoint)
    assert config.model.config.tokenizer.load_mean_std is False


def test_bind_tokenizer_checkpoint_rejects_mean_std_loading(tmp_path: Path) -> None:
    config = _config(load_mean_std=True)

    with pytest.raises(ValueError, match="load_mean_std must remain false"):
        runtime._bind_tokenizer_checkpoint(config, tmp_path / "tokenizer.pth")
    assert config.model.config.tokenizer.vae_pth == "/path/to/tokenizer.pth"
