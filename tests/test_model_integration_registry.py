"""Public two-model integration registry and config contract."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from robotactile_benchmark.closed_loop.interfaces import ClosedLoopPolicy
from robotactile_benchmark.integrations import (
    PolicyAdapter,
    get_model_integration,
    list_model_integrations,
    load_model_integration_config,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def test_registry_contains_exactly_two_first_class_models() -> None:
    specs = list_model_integrations()

    assert tuple(spec.integration_id for spec in specs) == ("act", "n0_twam")
    assert specs[0].display_name == "ACT"
    assert specs[0].capabilities.matched_no_touch is True
    assert specs[0].capabilities.structural_absence is False
    assert specs[1].display_name == "N0-TWAM"
    assert specs[1].capabilities.matched_no_touch is False
    assert specs[1].capabilities.structural_absence is False
    assert specs[1].capabilities.stateful_commit is True


def test_policy_adapter_is_existing_closed_loop_contract() -> None:
    assert PolicyAdapter is ClosedLoopPolicy


def test_registry_and_nested_capabilities_are_immutable() -> None:
    spec = get_model_integration("act")

    with pytest.raises(FrozenInstanceError):
        spec.display_name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        spec.capabilities.matched_no_touch = False  # type: ignore[misc]


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown model integration"):
        get_model_integration("arbitrary.module:Policy")


@pytest.mark.parametrize("integration_id", ("act", "n0_twam"))
def test_checked_in_config_matches_static_registry(integration_id: str) -> None:
    config = load_model_integration_config(integration_id)

    assert config.integration_id == integration_id
    assert config.spec == get_model_integration(integration_id)
    assert config.artifact_manifest.startswith("examples/")


def test_config_cannot_override_factory_or_registered_identity(tmp_path: Path) -> None:
    source = load_model_integration_config("act").to_dict()
    source["factory"] = "attacker.module:Policy"
    path = tmp_path / "act.json"
    path.write_bytes(_canonical(source))

    with pytest.raises(ValueError, match="fields"):
        load_model_integration_config("act", path)


def test_config_integration_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    source = load_model_integration_config("act").to_dict()
    source["integration_id"] = "n0_twam"
    path = tmp_path / "act.json"
    path.write_bytes(_canonical(source))

    with pytest.raises(ValueError, match="requested integration"):
        load_model_integration_config("act", path)
