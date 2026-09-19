"""Unified public CLI for first-class model integrations."""

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
        "dream_tac",
        "ftp1_policy",
        "n0_twam",
        "n0_vtla",
    ]
    assert payload["integrations"][0]["matched_no_touch"] is True
    assert payload["integrations"][1]["matched_no_touch"] is False
    assert payload["integrations"][1]["license_spdx"] == "Apache-2.0"
    assert payload["integrations"][1]["external_commit"] == (
        "14bab51d6862fd07124745c55cd395ea5caa9fd3"
    )
    assert payload["integrations"][2]["license_spdx"] == "Apache-2.0"
    assert payload["integrations"][2]["external_commit"] == (
        "89fa681d6c014cce28300946b7526db808e0b1c1"
    )
    assert payload["integrations"][3]["license_spdx"] == "CC-BY-NC-SA-4.0"
    assert payload["integrations"][4]["license_spdx"] == "CC-BY-SA-4.0"
    assert payload["integrations"][4]["external_commit"] == (
        "03a0ce4d7091ca2354864796770715aa212601b7"
    )


def test_integration_validate_binds_config_registry_and_external_pin(
    capsys: object,
) -> None:
    assert main(["integrations", "validate", "--model", "act"]) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert payload == {
        "config_valid": True,
        "external_commit": "05bcd3edb92237107efa40105292a24f1a9fd761",
        "integration_id": "act",
        "license_spdx": "Apache-2.0",
        "release_ready": True,
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
