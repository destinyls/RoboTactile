"""Strict external source-pin tests for first-class integrations."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from robotactile_benchmark.integrations.provenance import (
    IntegrationProvenanceError,
    load_integration_lock,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def test_packaged_lock_has_exact_external_sources_and_license_boundary() -> None:
    lock = load_integration_lock()

    assert lock.schema_version == "robotactile-integrations-lock-v2"
    assert tuple(pin.integration_id for pin in lock.entries) == (
        "act_runtime",
        "curobo",
        "isaaclab",
        "n0_twam",
        "univtac",
    )
    assert lock.by_id("univtac").commit_sha == (
        "05bcd3edb92237107efa40105292a24f1a9fd761"
    )
    assert lock.by_id("univtac").license_spdx == "Apache-2.0"
    assert lock.by_id("n0_twam").commit_sha == (
        "c43a2160dd31c449d92b28eab52c0e2f09e4738a"
    )
    assert lock.by_id("n0_twam").license_spdx == "CC-BY-NC-SA-4.0"
    assert lock.by_id("act_runtime").release_ready is False
    assert lock.by_id("n0_twam").release_ready is True
    assert lock.by_id("isaaclab").license_spdx == "BSD-3-Clause"
    assert lock.by_id("isaaclab").source_directory == "IsaacLab"
    assert lock.by_id("curobo").license_spdx == ("LicenseRef-NVIDIA-NonCommercial")
    assert lock.by_id("curobo").source_directory == "curobo"


def test_lock_and_nested_entries_are_immutable() -> None:
    lock = load_integration_lock()

    with pytest.raises(FrozenInstanceError):
        lock.entries[0].release_ready = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        lock.schema_version = "changed"  # type: ignore[misc]


def test_unknown_pin_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown external integration"):
        load_integration_lock().by_id("unknown")


@pytest.mark.parametrize(
    "mutator",
    (
        lambda document: document.update({"unknown": True}),
        lambda document: document["entries"].append(document["entries"][0]),
        lambda document: document["entries"][0].update({"commit_sha": "main"}),
        lambda document: document["entries"][0].update(
            {"repository_url": "ssh://internal.invalid/model"}
        ),
    ),
)
def test_semantic_lock_tampering_fails_closed(tmp_path: Path, mutator: object) -> None:
    original = load_integration_lock().to_dict()
    assert callable(mutator)
    mutator(original)
    path = tmp_path / "integrations.lock.json"
    path.write_bytes(_canonical(original))

    with pytest.raises(IntegrationProvenanceError):
        load_integration_lock(path)


def test_noncanonical_or_duplicate_json_fails_closed(tmp_path: Path) -> None:
    canonical = load_integration_lock().to_dict()
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"entries":[],"schema_version":"x","schema_version":"x"}\n',
        encoding="utf-8",
    )

    with pytest.raises(IntegrationProvenanceError):
        load_integration_lock(pretty)
    with pytest.raises(IntegrationProvenanceError):
        load_integration_lock(duplicate)
