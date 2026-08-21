"""Fail-closed operator Factory/Registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Dict, Tuple, Type

from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.operators.base import FaultOperator


class OperatorRegistry:
    """Register operator classes under unique stable IDs."""

    def __init__(self) -> None:
        self._classes: Dict[str, Type[FaultOperator]] = {}

    def register(self, name: str, operator_class: Type[FaultOperator]) -> None:
        if name in self._classes:
            raise ValueError(f"operator already registered: {name}")
        self._classes[name] = operator_class

    def decorator(
        self, name: str
    ) -> Callable[[Type[FaultOperator]], Type[FaultOperator]]:
        def register_class(operator_class: Type[FaultOperator]) -> Type[FaultOperator]:
            self.register(name, operator_class)
            return operator_class

        return register_class

    def create(self, name: str) -> FaultOperator:
        try:
            operator_class = self._classes[name]
        except KeyError as exc:
            raise KeyError(f"unknown operator: {name}") from exc
        return operator_class()

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._classes))

    def verify_core(self) -> None:
        actual = set(self._classes)
        if actual != set(CORE_OPERATOR_IDS):
            missing = sorted(set(CORE_OPERATOR_IDS) - actual)
            extra = sorted(actual - set(CORE_OPERATOR_IDS))
            raise RuntimeError(
                f"core operator registry mismatch: missing={missing}, extra={extra}"
            )


REGISTRY = OperatorRegistry()
register_operator = REGISTRY.decorator
