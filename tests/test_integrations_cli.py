"""Unified public CLI for ACT and N0-TWAM integrations."""

from __future__ import annotations

import json

import pytest

from robotactile_benchmark.cli import main


def test_integrations_list_is_one_canonical_json_line(capsys: object) -> None:
    assert main(["integrations", "list"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert output.endswith("\n") and output.count("\n") == 1
    payload = json.loads(output)

    assert [item["integration_id"] for item in payload["integrations"]] == [
        "act",
        "n0_twam",
    ]
    assert payload["integrations"][0]["matched_no_touch"] is True
    assert payload["integrations"][1]["matched_no_touch"] is False
    assert payload["integrations"][1]["license_spdx"] == "CC-BY-NC-SA-4.0"


def test_integration_validate_binds_config_registry_and_external_pin(
    capsys: object,
) -> None:
    assert main(["integrations", "validate", "--model", "act"]) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert payload == {
        "config_valid": True,
        "external_commit": "e295378c702b2e87617ebddcad193be3608e00c3",
        "integration_id": "act",
        "license_spdx": "Apache-2.0",
        "release_ready": False,
    }


def test_n0_evaluate_loads_request_before_runtime_effects() -> None:
    with pytest.raises(FileNotFoundError, match="live UniVTAC request"):
        main(
            [
                "evaluate",
                "--model",
                "n0_twam",
                "--request",
                "/does/not/exist.json",
            ]
        )


def test_existing_commands_remain_available() -> None:
    assert main(["validate-registry"]) == 0
