"""Read-only legacy migration planning tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.deployment.migration import plan_legacy_migration


def test_migration_plan_hashes_only_recognized_files_without_writes(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy"
    source = legacy / "src/UniVTAC/README.md"
    model = legacy / "artifacts/checkpoints/pull_out_key/policy_last.ckpt"
    unknown = legacy / "notes.txt"
    source.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    model.write_bytes(b"model")
    unknown.write_bytes(b"unknown")
    deployment = tmp_path / "new/deployment"

    plan = plan_legacy_migration(legacy, DeploymentLayout(deployment))

    assert not deployment.exists()
    assert plan.unmapped_file_count == 1
    assert tuple(entry.destination for entry in plan.entries) == (
        deployment / "artifacts/models/act/pull_out_key/policy_last.ckpt",
        deployment / "sources/UniVTAC/README.md",
    )
    by_source = {entry.source: entry for entry in plan.entries}
    assert by_source[model].sha256 == hashlib.sha256(b"model").hexdigest()
    assert by_source[source].size_bytes == 6
    assert plan.to_dict()["writes_performed"] is False
