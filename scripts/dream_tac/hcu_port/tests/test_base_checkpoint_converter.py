"""Contract tests for the strict flat-PT to model-only DCP converter."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from scripts.dream_tac.hcu_port import base_checkpoint_converter as converter
from scripts.dream_tac.hcu_port import base_checkpoint_converter_runtime as runtime
from scripts.dream_tac.hcu_port.base_checkpoint_converter import (
    CONVERTER_MANIFEST_NAME,
    CONVERTER_PROTOCOL,
    CONVERTER_RECEIPT_NAME,
    DCP_LOAD_RELATIVE_PATH,
    ConversionConfig,
    TensorDescriptor,
    convert_checkpoint,
)
from scripts.dream_tac.hcu_port.receipt import verify_receipt
from scripts.dream_tac.training.identity import receipt_sha256
from scripts.dream_tac.training.training_request import sha256_regular_file


@dataclass(frozen=True)
class _Tensor:
    values: tuple[float, ...]
    shape: tuple[int, ...]
    dtype: str = "torch.bfloat16"


class _FakeBackend:
    def __init__(
        self,
        payload: object,
        template: dict[str, _Tensor],
        *,
        corrupt_roundtrip: bool = False,
    ) -> None:
        self.payload = payload
        self.template = template
        self.corrupt_roundtrip = corrupt_roundtrip
        self.saved: dict[str, _Tensor] = {}
        self.wrapper_modes: list[bool] = []

    def load_flat_checkpoint(self, path: Path) -> object:
        assert path.name == "base.pt"
        return self.payload

    def describe_tensor(self, value: object) -> TensorDescriptor | None:
        if not isinstance(value, _Tensor):
            return None
        return TensorDescriptor(value.shape, value.dtype, len(value.values))

    def create_model(self, config: ConversionConfig) -> tuple[object, object]:
        return dict(self.template), object()

    def load_model(self, model: object, upstream_config: object, path: Path) -> object:
        state = cast(dict[str, _Tensor], model)
        payload = cast(dict[str, object], self.payload)
        if isinstance(payload.get("model"), dict):
            source = cast(dict[str, _Tensor], payload["model"])
        elif isinstance(payload.get("state_dict"), dict):
            source = cast(dict[str, _Tensor], payload["state_dict"])
        else:
            source = cast(dict[str, _Tensor], payload)
        for key, value in source.items():
            if key in state:
                state[key] = value
        return state

    def model_state_dict(
        self, model: object, *, load_ema_to_reg: bool
    ) -> dict[str, object]:
        self.wrapper_modes.append(load_ema_to_reg)
        state = cast(dict[str, _Tensor], model)
        if not load_ema_to_reg:
            return dict(state)
        return {
            key.replace("net.", "net_ema.", 1): value for key, value in state.items()
        }

    def load_ema_to_reg(self, upstream_config: object) -> bool:
        return True

    def save_dcp(self, state: Mapping[str, object], root: Path) -> None:
        root.mkdir(parents=True)
        (root / ".metadata").write_bytes(b"fake-dcp-metadata")
        (root / "__0_0.distcp").write_bytes(b"fake-dcp-shard")
        self.saved = cast(dict[str, _Tensor], dict(state))

    def empty_like(self, value: object) -> object:
        tensor = cast(_Tensor, value)
        return _Tensor(tuple(0.0 for _ in tensor.values), tensor.shape, tensor.dtype)

    def load_dcp(self, state: dict[str, object], root: Path) -> None:
        assert root.name == "model"
        for key, value in self.saved.items():
            state[key] = value
        if self.corrupt_roundtrip:
            key = sorted(state)[0]
            tensor = cast(_Tensor, state[key])
            state[key] = _Tensor((99.0, *tensor.values[1:]), tensor.shape, tensor.dtype)

    def tensor_equal(self, left: object, right: object) -> bool:
        return left == right


@dataclass(frozen=True)
class _Fixture:
    config: ConversionConfig
    template: dict[str, _Tensor]
    source: dict[str, _Tensor]


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Fixture:
    checkout = tmp_path / "source" / "Dream-Tac"
    experiment = checkout / converter.EXPERIMENT_CONFIG_RELATIVE_PATH
    critical = (*converter._CRITICAL_SOURCE_FILES,)
    for relative in (*critical, converter.EXPERIMENT_CONFIG_RELATIVE_PATH):
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{relative}\n", encoding="utf-8")
    _git(checkout, "init", "--quiet")
    _git(checkout, "config", "user.name", "RoboTactile Test")
    _git(checkout, "config", "user.email", "robotactile@example.invalid")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "--quiet", "-m", "fixture")
    commit = _git(checkout, "rev-parse", "HEAD")
    monkeypatch.setattr(converter, "PINNED_COMMIT", commit)
    checkpoint = tmp_path / "artifacts" / "base.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"flat-checkpoint-fixture")
    tokenizer = checkpoint.parent / "tokenizer.pth"
    tokenizer.write_bytes(b"tokenizer-checkpoint-fixture")
    template = {
        "net.base.weight": _Tensor((1.0, 2.0), (2,)),
        "net.tactile_bias.weight": _Tensor((7.0,), (1,)),
    }
    source = {"net.base.weight": _Tensor((3.0, 4.0), (2,))}
    return _Fixture(
        config=ConversionConfig(
            dream_tac_checkout=checkout,
            experiment_config=experiment,
            experiment_config_sha256=sha256_regular_file(experiment),
            input_checkpoint=checkpoint,
            input_checkpoint_sha256=sha256_regular_file(checkpoint),
            tokenizer_checkpoint=tokenizer,
            tokenizer_checkpoint_sha256=sha256_regular_file(tokenizer),
            output_root=tmp_path / "outputs" / "base-dcp",
            allowed_missing_keys=("net.tactile_bias.weight",),
        ),
        template=template,
        source=source,
    )


@pytest.mark.parametrize("top_level", ("plain", "model", "state_dict"))
def test_conversion_is_source_bound_roundtripped_and_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    top_level: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    payload: object
    if top_level == "plain":
        payload = fixture.source
    else:
        payload = cast(
            dict[str, object],
            {top_level: fixture.source, "optimizer": "ignored outer component"},
        )
    backend = _FakeBackend(payload, fixture.template)

    receipt_path, receipt = convert_checkpoint(fixture.config, backend=backend)

    assert receipt_path == fixture.config.output_root / CONVERTER_RECEIPT_NAME
    verify_receipt(receipt)
    assert receipt["status"] == "complete"
    assert receipt["protocol_id"] == CONVERTER_PROTOCOL
    assert receipt["training_launch_performed"] is False
    assert receipt["roundtrip_status"] == "passed"
    assert receipt["source_config_load_ema_to_reg"] is True
    assert receipt["dcp_model_wrapper_load_ema_to_reg"] is False
    assert backend.wrapper_modes == [False, False, False]
    assert set(backend.saved) == {"net.base.weight", "net.tactile_bias.weight"}
    assert receipt["dcp_iteration_root"] == str(
        fixture.config.output_root / "iter_000000000"
    )
    assert receipt["dcp_model_root"] == str(
        fixture.config.output_root / DCP_LOAD_RELATIVE_PATH
    )
    input_identity = cast(dict[str, object], receipt["input_checkpoint"])
    assert input_identity["top_level_format"] == top_level
    assert input_identity["key_count"] == 1
    tokenizer_identity = cast(dict[str, object], receipt["tokenizer_checkpoint"])
    assert tokenizer_identity == {
        "path": str(fixture.config.tokenizer_checkpoint),
        "sha256": fixture.config.tokenizer_checkpoint_sha256,
    }
    assert receipt["tokenizer_load_mean_std"] is False
    dcp_identity = cast(dict[str, object], receipt["dcp_state"])
    assert dcp_identity["key_count"] == 2
    manifest_path = fixture.config.output_root / CONVERTER_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_sha256(manifest, field="converter_manifest_sha256")
    files = cast(list[dict[str, object]], manifest["files"])
    assert {item["relative_path"] for item in files} == {
        "iter_000000000/model/.metadata",
        "iter_000000000/model/__0_0.distcp",
    }
    assert receipt["converter_manifest_sha256"] == manifest["converter_manifest_sha256"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        convert_checkpoint(fixture.config, backend=backend)


def test_missing_keys_require_the_exact_explicit_whitelist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    backend = _FakeBackend(fixture.source, fixture.template)
    config = ConversionConfig(
        **{
            **fixture.config.__dict__,
            "allowed_missing_keys": (),
        }
    )
    with pytest.raises(ValueError, match="unauthorized"):
        convert_checkpoint(config, backend=backend)
    assert not config.output_root.exists()

    unused = ConversionConfig(
        **{
            **fixture.config.__dict__,
            "allowed_missing_keys": (
                "net.tactile_bias.weight",
                "net.not_actually_missing",
            ),
        }
    )
    with pytest.raises(ValueError, match="unused"):
        convert_checkpoint(unused, backend=backend)


@pytest.mark.parametrize(
    ("source", "message"),
    (
        (
            {
                "net.base.weight": _Tensor((3.0, 4.0), (2,)),
                "net.unexpected": _Tensor((1.0,), (1,)),
            },
            "unexpected checkpoint keys",
        ),
        ({"net.base.weight": _Tensor((3.0,), (1,))}, "shape/dtype mismatch"),
        (
            {"net.base.weight": _Tensor((3.0, 4.0), (2,), "torch.float32")},
            "shape/dtype mismatch",
        ),
        ({"net.base.weight": "not-a-tensor"}, "not a tensor"),
    ),
)
def test_invalid_tensor_contracts_fail_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: object,
    message: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    backend = _FakeBackend(source, fixture.template)
    with pytest.raises(ValueError, match=message):
        convert_checkpoint(fixture.config, backend=backend)
    assert not fixture.config.output_root.exists()


def test_ambiguous_top_level_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    backend = _FakeBackend(
        {"model": fixture.source, "state_dict": fixture.source}, fixture.template
    )
    with pytest.raises(ValueError, match="ambiguous"):
        convert_checkpoint(fixture.config, backend=backend)


def test_corrupt_dcp_roundtrip_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    backend = _FakeBackend(fixture.source, fixture.template, corrupt_roundtrip=True)
    with pytest.raises(ValueError, match="DCP roundtrip"):
        convert_checkpoint(fixture.config, backend=backend)
    assert not fixture.config.output_root.exists()
    assert not list(fixture.config.output_root.parent.glob(".base-dcp.*"))


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("input_checkpoint_sha256", "input_checkpoint SHA256 mismatch"),
        ("tokenizer_checkpoint_sha256", "tokenizer_checkpoint SHA256 mismatch"),
    ),
)
def test_identity_mismatch_fails_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    config = ConversionConfig(
        **{
            **fixture.config.__dict__,
            field: "a" * 64,
        }
    )
    with pytest.raises(ValueError, match=message):
        convert_checkpoint(
            config, backend=_FakeBackend(fixture.source, fixture.template)
        )


def test_config_requires_absolute_paths_and_non_placeholder_hashes(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="dream_tac_checkout must be absolute"):
        ConversionConfig(
            dream_tac_checkout=Path("relative"),
            experiment_config=tmp_path / "config.py",
            experiment_config_sha256="1" * 64,
            input_checkpoint=tmp_path / "base.pt",
            input_checkpoint_sha256="2" * 64,
            tokenizer_checkpoint=tmp_path / "tokenizer.pth",
            tokenizer_checkpoint_sha256="3" * 64,
            output_root=tmp_path / "output",
        )

    with pytest.raises(ValueError, match="all-zero"):
        ConversionConfig(
            dream_tac_checkout=tmp_path / "source",
            experiment_config=tmp_path / "config.py",
            experiment_config_sha256="0" * 64,
            input_checkpoint=tmp_path / "base.pt",
            input_checkpoint_sha256="2" * 64,
            tokenizer_checkpoint=tmp_path / "tokenizer.pth",
            tokenizer_checkpoint_sha256="3" * 64,
            output_root=tmp_path / "output",
        )


def test_base_checkpoint_environment_removes_temporary_values(tmp_path: Path) -> None:
    environ: dict[str, str] = {}
    checkpoint = tmp_path / "base.pt"
    with runtime._base_checkpoint_environment(environ, checkpoint):
        assert environ["DREAM_TAC_BASE_CHECKPOINT"] == str(checkpoint)
        assert "ROBOTACTILE_HCU_CONFIG_PROBE_ONLY" not in environ
    assert environ == {}


def test_base_checkpoint_environment_restores_values_after_error(
    tmp_path: Path,
) -> None:
    environ = {
        "DREAM_TAC_BASE_CHECKPOINT": "/previous/base.pt",
        "ROBOTACTILE_HCU_CONFIG_PROBE_ONLY": "1",
        "UNRELATED": "preserved",
    }
    with (
        pytest.raises(RuntimeError, match="fixture failure"),
        runtime._base_checkpoint_environment(environ, tmp_path / "base.pt"),
    ):
        assert environ["DREAM_TAC_BASE_CHECKPOINT"] == str(tmp_path / "base.pt")
        assert "ROBOTACTILE_HCU_CONFIG_PROBE_ONLY" not in environ
        raise RuntimeError("fixture failure")
    assert environ == {
        "DREAM_TAC_BASE_CHECKPOINT": "/previous/base.pt",
        "ROBOTACTILE_HCU_CONFIG_PROBE_ONLY": "1",
        "UNRELATED": "preserved",
    }
