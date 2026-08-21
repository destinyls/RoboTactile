"""Unified public CLI for ACT and N0-TWAM integrations."""

from __future__ import annotations

import json

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


def test_n0_evaluate_fails_closed_before_loading_request(capsys: object) -> None:
    exit_code = main(
        [
            "evaluate",
            "--model",
            "n0_twam",
            "--request",
            "/does/not/exist.json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert exit_code == 2
    assert payload["integration_id"] == "n0_twam"
    assert payload["status"] == "unsupported_contract"
    assert payload["backend_effects"] == 0
    assert payload["policy_effects"] == 0


def test_existing_commands_remain_available() -> None:
    assert main(["validate-registry"]) == 0
