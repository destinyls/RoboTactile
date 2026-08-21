"""Canonical 2+7+3+2 operator registry."""

from robotactile_benchmark.constants import CORE_OPERATOR_IDS as EXPECTED_OPERATOR_IDS
from robotactile_benchmark.operators import availability as _availability  # noqa: F401
from robotactile_benchmark.operators import context as _context  # noqa: F401
from robotactile_benchmark.operators import fidelity as _fidelity  # noqa: F401
from robotactile_benchmark.operators import temporal as _temporal  # noqa: F401
from robotactile_benchmark.operators.base import FaultOperator
from robotactile_benchmark.operators.registry import REGISTRY, OperatorRegistry

REGISTRY.verify_core()


def get_operator(operator_id: str) -> FaultOperator:
    """Instantiate one registered operator by stable ID."""

    return REGISTRY.create(operator_id)


def list_operator_ids() -> tuple[str, ...]:
    """List the exact registered core IDs."""

    return REGISTRY.names()


__all__ = [
    "EXPECTED_OPERATOR_IDS",
    "OperatorRegistry",
    "get_operator",
    "list_operator_ids",
]
