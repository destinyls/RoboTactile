"""Lazy torch/Cosmos runtime adapter for the base-checkpoint converter."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from scripts.dream_tac.training.training_request import (
    DONOR_EXPERIMENT,
    sha256_regular_file,
)

from .base_checkpoint_converter import (
    CONFIG_ROUTER_RELATIVE_PATH,
    EXPERIMENT_CONFIG_RELATIVE_PATH,
    ConversionConfig,
    TensorDescriptor,
    convert_checkpoint,
)


class _TensorValue(Protocol):
    shape: Sequence[int]
    dtype: object

    def numel(self) -> int: ...

    def detach(self) -> _TensorValue: ...

    def cpu(self) -> _TensorValue: ...


class _CheckpointValue(Protocol):
    load_ema_to_reg: bool


class _UpstreamConfigValue(Protocol):
    checkpoint: _CheckpointValue


class _TokenizerConfigValue(Protocol):
    vae_pth: str
    load_mean_std: bool


class _ModelConfigValue(Protocol):
    tokenizer: _TokenizerConfigValue


class _ModelValue(Protocol):
    config: _ModelConfigValue


class _TokenizerUpstreamConfigValue(Protocol):
    model: _ModelValue


_BASE_CHECKPOINT_ENV = "DREAM_TAC_BASE_CHECKPOINT"
_PROBE_ONLY_ENV = "ROBOTACTILE_HCU_CONFIG_PROBE_ONLY"


@contextmanager
def _base_checkpoint_environment(
    environ: MutableMapping[str, str], checkpoint: Path
) -> Iterator[None]:
    previous = {
        name: environ.get(name) for name in (_BASE_CHECKPOINT_ENV, _PROBE_ONLY_ENV)
    }
    environ[_BASE_CHECKPOINT_ENV] = str(checkpoint)
    environ.pop(_PROBE_ONLY_ENV, None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                environ.pop(name, None)
            else:
                environ[name] = value


def _is_within(path: Path, root: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _bind_tokenizer_checkpoint(upstream_config: object, checkpoint: Path) -> None:
    config = cast(_TokenizerUpstreamConfigValue, upstream_config)
    if config.model.config.tokenizer.load_mean_std:
        raise ValueError("upstream tokenizer load_mean_std must remain false")
    config.model.config.tokenizer.vae_pth = str(checkpoint)


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_conversion_inputs(
    config: ConversionConfig,
    *,
    pinned_commit: str,
    critical_source_files: tuple[Path, ...],
) -> dict[str, object]:
    """Verify immutable source/config/checkpoint identity before deserialization."""

    checkout = config.dream_tac_checkout
    if checkout.is_symlink() or not checkout.resolve(strict=True).is_dir():
        raise ValueError("dream_tac_checkout must be a real directory")
    commit = _git(checkout, "rev-parse", "HEAD")
    if commit != pinned_commit:
        raise ValueError("Dream-Tac checkout commit does not match the pin")
    if config.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite output root: {config.output_root}"
        )
    if _is_within(config.output_root, checkout) or _is_within(
        checkout, config.output_root
    ):
        raise ValueError("output_root must be separate from Dream-Tac checkout")
    files = {
        "experiment_config": (
            config.experiment_config,
            config.experiment_config_sha256,
        ),
        "input_checkpoint": (
            config.input_checkpoint,
            config.input_checkpoint_sha256,
        ),
        "tokenizer_checkpoint": (
            config.tokenizer_checkpoint,
            config.tokenizer_checkpoint_sha256,
        ),
    }
    for name, (path, expected) in files.items():
        if path.is_symlink() or not path.resolve(strict=True).is_file():
            raise ValueError(f"{name} must be a real regular file")
        if sha256_regular_file(path) != expected:
            raise ValueError(f"{name} SHA256 mismatch")
    if not _is_within(config.experiment_config, checkout):
        raise ValueError("experiment_config must be inside Dream-Tac checkout")
    expected_config = checkout / EXPERIMENT_CONFIG_RELATIVE_PATH
    if config.experiment_config.resolve() != expected_config.resolve():
        raise ValueError("experiment_config path does not match the pinned contract")
    sources = [
        {
            "relative_path": str(relative),
            "sha256": sha256_regular_file(checkout / relative),
        }
        for relative in critical_source_files
    ]
    return {"commit": commit, "critical_files": sources}


class TorchUpstreamBackend:
    """Production adapter using the pinned upstream loader, wrapper, and DCP."""

    def __init__(self, checkout: Path) -> None:
        self._checkout = checkout.resolve(strict=True)
        checkout_text = str(self._checkout)
        if checkout_text not in sys.path:
            sys.path.insert(0, checkout_text)
        self._torch = importlib.import_module("torch")
        self._dcp = importlib.import_module("torch.distributed.checkpoint")
        self._loader = self._import_checkout_module(
            "cosmos_policy._src.predict2.utils.model_loader"
        )
        self._checkpointer = self._import_checkout_module(
            "cosmos_policy._src.predict2.checkpointer.dcp"
        )

    def _import_checkout_module(self, name: str) -> ModuleType:
        module = importlib.import_module(name)
        source = getattr(module, "__file__", None)
        if not isinstance(source, str) or not _is_within(Path(source), self._checkout):
            raise RuntimeError(f"{name} was imported outside the pinned checkout")
        return module

    def load_flat_checkpoint(self, path: Path) -> object:
        load = cast(object, self._torch.load)
        if not callable(load):
            raise RuntimeError("torch.load is unavailable")
        return load(path, map_location="cpu", weights_only=True)

    def describe_tensor(self, value: object) -> TensorDescriptor | None:
        tensor_type = cast(type[object], self._torch.Tensor)
        if not isinstance(value, tensor_type):
            return None
        tensor = cast(_TensorValue, value)
        shape = tuple(int(item) for item in tensor.shape)
        return TensorDescriptor(shape, str(tensor.dtype), tensor.numel())

    def create_model(self, config: ConversionConfig) -> tuple[object, object]:
        helper = self._import_checkout_module(
            "cosmos_policy._src.imaginaire.utils.config_helper"
        )
        lazy = self._import_checkout_module("cosmos_policy._src.imaginaire.lazy_config")
        with _base_checkpoint_environment(os.environ, config.input_checkpoint):
            previous_cwd = Path.cwd()
            try:
                os.chdir(self._checkout)
                module_name = helper.get_config_module(str(CONFIG_ROUTER_RELATIVE_PATH))
                router = self._import_checkout_module(module_name)
                upstream_config = router.make_config()
                upstream_config = helper.override(
                    upstream_config, ["--", f"experiment={DONOR_EXPERIMENT}"]
                )
                upstream_config.checkpoint.load_path = str(config.input_checkpoint)
                upstream_config.checkpoint.load_training_state = False
                upstream_config.checkpoint.strict_resume = False
                upstream_config.model.config.fsdp_shard_size = 1
                _bind_tokenizer_checkpoint(upstream_config, config.tokenizer_checkpoint)
                upstream_config.validate()
                upstream_config.freeze()
                model = lazy.instantiate(upstream_config.model)
                model.on_train_start()
            finally:
                os.chdir(previous_cwd)
        return model, upstream_config

    def load_model(self, model: object, upstream_config: object, path: Path) -> object:
        return self._loader.load_model_state_dict_from_checkpoint(
            model=model,
            config=upstream_config,
            s3_checkpoint_dir=str(path),
            load_ema_to_reg=self.load_ema_to_reg(upstream_config),
            local_cache_dir=None,
            override_cache=False,
        )

    def model_state_dict(
        self, model: object, *, load_ema_to_reg: bool
    ) -> dict[str, object]:
        wrapper = self._checkpointer.ModelWrapper(
            model, load_ema_to_reg=load_ema_to_reg
        )
        return cast(dict[str, object], wrapper.state_dict())

    def load_ema_to_reg(self, upstream_config: object) -> bool:
        value = cast(_UpstreamConfigValue, upstream_config)
        return bool(value.checkpoint.load_ema_to_reg)

    def save_dcp(self, state: Mapping[str, object], root: Path) -> None:
        self._dcp.save(state_dict=dict(state), checkpoint_id=str(root))

    def empty_like(self, value: object) -> object:
        return self._torch.empty_like(value)

    def load_dcp(self, state: dict[str, object], root: Path) -> None:
        self._dcp.load(state_dict=state, checkpoint_id=str(root))

    def tensor_equal(self, left: object, right: object) -> bool:
        left_cpu = cast(_TensorValue, left).detach().cpu()
        right_cpu = cast(_TensorValue, right).detach().cpu()
        return bool(self._torch.equal(left_cpu, right_cpu))


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert one pinned Dream-Tac flat checkpoint to model-only DCP."
    )
    parser.add_argument("--dream-tac-checkout", required=True, type=Path)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--experiment-config-sha256", required=True)
    parser.add_argument("--input-checkpoint", required=True, type=Path)
    parser.add_argument("--input-checkpoint-sha256", required=True)
    parser.add_argument("--tokenizer-checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer-checkpoint-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--allow-missing-key", action="append", default=[])
    args = parser.parse_args(argv)
    config = ConversionConfig(
        dream_tac_checkout=args.dream_tac_checkout,
        experiment_config=args.experiment_config,
        experiment_config_sha256=args.experiment_config_sha256,
        input_checkpoint=args.input_checkpoint,
        input_checkpoint_sha256=args.input_checkpoint_sha256,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        tokenizer_checkpoint_sha256=args.tokenizer_checkpoint_sha256,
        output_root=args.output_root,
        allowed_missing_keys=tuple(sorted(args.allow_missing_key)),
    )
    path, receipt = convert_checkpoint(config)
    print(
        json.dumps(
            {
                "output": str(path),
                "receipt_sha256": receipt["receipt_sha256"],
                "status": receipt["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = ["TorchUpstreamBackend", "cli_main", "verify_conversion_inputs"]
