"""Finite-value contract tests for Dream-Tac T5 cache publication."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.dream_tac.training.t5_cache import (
    _expected_inventory,
    _validate_embeddings,
)


@dataclass(frozen=True)
class _Scalar:
    value: bool | int

    def item(self) -> bool | int:
        return self.value


@dataclass(frozen=True)
class _Mask:
    all_value: bool
    count: int

    def all(self) -> _Scalar:
        return _Scalar(self.all_value)

    def sum(self) -> _Scalar:
        return _Scalar(self.count)


@dataclass(frozen=True)
class _Device:
    type: str = "cpu"


@dataclass(frozen=True)
class _Tensor:
    nan_count: int = 0
    inf_count: int = 0
    shape: tuple[int, int, int] = (1, 512, 1024)
    dtype: str = "torch.bfloat16"
    device: _Device = _Device()

    def isfinite(self) -> _Mask:
        return _Mask(self.nan_count == 0 and self.inf_count == 0, 0)

    def isnan(self) -> _Mask:
        return _Mask(self.nan_count == 0, self.nan_count)

    def isinf(self) -> _Mask:
        return _Mask(self.inf_count == 0, self.inf_count)


def test_finite_t5_tensor_inventory_records_explicit_counts() -> None:
    prompt = "one prompt"

    inventory = _validate_embeddings({prompt: _Tensor()}, [prompt])

    assert inventory == _expected_inventory([prompt])


@pytest.mark.parametrize(
    ("tensor", "kind"),
    [
        (_Tensor(nan_count=1024), "NaN"),
        (_Tensor(inf_count=1), "Inf"),
    ],
)
def test_nonfinite_t5_tensor_is_rejected_before_publication(
    tensor: _Tensor,
    kind: str,
) -> None:
    prompt = f"prompt with {kind}"

    with pytest.raises(ValueError, match="T5 tensor is non-finite"):
        _validate_embeddings({prompt: tensor}, [prompt])
