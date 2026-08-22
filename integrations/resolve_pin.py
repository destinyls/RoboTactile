#!/usr/bin/env python3
"""Resolve one field from the canonical external-integration lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn

_LOCK_PATH = Path(__file__).with_name("integrations.lock.json")
_FIELDS = (
    "artifact_schema",
    "commit_sha",
    "install_script",
    "license_spdx",
    "repository_url",
    "source_directory",
)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _fail(message: str) -> NoReturn:
    raise SystemExit(message)


def resolve_field(integration_id: str, field: str) -> str:
    """Return one allow-listed string without importing RoboTactile."""

    try:
        document = json.loads(
            _LOCK_PATH.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, ValueError, TypeError) as error:
        _fail(f"cannot read canonical integration lock: {error}")
    if not isinstance(document, dict):
        _fail("integration lock must be an object")
    entries = document.get("entries")
    if document.get(
        "schema_version"
    ) != "robotactile-integrations-lock-v2" or not isinstance(entries, list):
        _fail("integration lock version mismatch")
    matches = [
        item
        for item in entries
        if isinstance(item, dict) and item.get("integration_id") == integration_id
    ]
    if len(matches) != 1:
        _fail(f"unknown or duplicated integration id: {integration_id}")
    value = matches[0].get(field)
    if not isinstance(value, str) or not value or value.strip() != value:
        _fail(f"integration field is not a non-empty string: {field}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("integration_id")
    parser.add_argument("field", choices=_FIELDS)
    arguments = parser.parse_args()
    print(resolve_field(arguments.integration_id, arguments.field))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
