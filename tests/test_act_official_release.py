"""Tests for the official UniVTAC ACT release installer."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.integrations.act import official_release_io
from robotactile_benchmark.integrations.act.official_release import (
    OFFICIAL_ACT_REVISION,
    OfficialACTInstallPlan,
    OfficialACTReleaseError,
    PlannedOfficialACTFile,
    build_official_act_install_plan,
    builtin_official_act_release_lock,
    install_official_act_release,
    load_official_act_release_lock,
)
from robotactile_benchmark.integrations.act.official_release_io import (
    _install_validated_plan,
)


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class _InterruptedResponse(_Response):
    def __init__(self) -> None:
        super().__init__(b"")
        self._first = True

    def read(self, _size: int = -1) -> bytes:
        if self._first:
            self._first = False
            return b"partial"
        raise OSError("simulated interrupted transfer")


def test_builtin_inventory_has_16_policies_8_stats_and_shared_encoder() -> None:
    lock = builtin_official_act_release_lock()

    assert lock.repository_id == "byml/UniVTAC"
    assert lock.revision == OFFICIAL_ACT_REVISION
    assert len(lock.files) == 25
    assert sum(item.kind == "policy" for item in lock.files) == 16
    assert sum(item.kind == "stats" for item in lock.files) == 8
    assert sum(item.kind == "encoder" for item in lock.files) == 1
    assert tuple(item.remote_path for item in lock.files) == tuple(
        sorted(item.remote_path for item in lock.files)
    )
    assert all(len(item.sha256) == 64 for item in lock.files)


def test_canonical_lock_roundtrip_and_tamper_rejection(tmp_path: Path) -> None:
    value = builtin_official_act_release_lock().to_dict()
    path = tmp_path / "act_artifacts.lock.json"
    path.write_bytes(canonical_json_bytes(value))

    assert load_official_act_release_lock(path).to_dict() == value

    value["files"][0]["sha256"] = "0" * 64  # type: ignore[index]
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(OfficialACTReleaseError, match="inventory mismatch"):
        load_official_act_release_lock(path)


def test_plan_selects_profile_and_shares_stats_download(tmp_path: Path) -> None:
    plan = build_official_act_install_plan(
        tmp_path / "act",
        tasks=("insert_hole",),
        profiles=("univtac", "vision_only"),
        include_reference=True,
    )

    runtime = [
        item for item in plan.files if item.evidence_level.endswith("execution_v1")
    ]
    references = [item for item in plan.files if item.kind.startswith("reference_")]
    assert [item.kind for item in runtime].count("policy") == 2
    assert [item.kind for item in runtime].count("stats") == 1
    stats = next(item for item in runtime if item.kind == "stats")
    assert stats.destinations == (
        "insert_hole/univtac/dataset_stats.pkl",
        "insert_hole/vision_only/dataset_stats.pkl",
    )
    assert len(references) == 4
    assert all(
        path.startswith(f"references/{OFFICIAL_ACT_REVISION}/")
        for item in references
        for path in item.destinations
    )
    assert plan.to_dict()["reference_evidence_level"] == (
        "upstream_reference_only_no_local_execution_v1"
    )


def test_plan_rejects_unpinned_url_and_kind_evidence_mismatch() -> None:
    with pytest.raises(OfficialACTReleaseError, match="not hash-pinned"):
        PlannedOfficialACTFile(
            destinations=("encoder.pth",),
            evidence_level=(
                "artifact_installation_only_no_model_or_simulator_execution_v1"
            ),
            kind="encoder",
            remote_path="checkpoints/encoder.pth",
            sha256="a" * 64,
            size_bytes=1,
            source_url=(
                "https://huggingface.co/datasets/byml/UniVTAC/resolve/"
                "main/checkpoints/encoder.pth?download=true"
            ),
        )
    with pytest.raises(OfficialACTReleaseError, match="kind/evidence"):
        PlannedOfficialACTFile(
            destinations=("reference.json",),
            evidence_level="upstream_reference_only_no_local_execution_v1",
            kind="policy",
            remote_path="checkpoints/encoder.pth",
            sha256="a" * 64,
            size_bytes=1,
            source_url=(
                "https://huggingface.co/datasets/byml/UniVTAC/resolve/"
                f"{OFFICIAL_ACT_REVISION}/checkpoints/encoder.pth?download=true"
            ),
        )


def _tiny_plan(
    root: Path, data: bytes, destinations: tuple[str, ...]
) -> OfficialACTInstallPlan:
    digest = hashlib.sha256(data).hexdigest()
    remote = "checkpoints/encoder.pth"
    item = PlannedOfficialACTFile(
        destinations=destinations,
        evidence_level="artifact_installation_only_no_model_or_simulator_execution_v1",
        kind="encoder",
        remote_path=remote,
        sha256=digest,
        size_bytes=len(data),
        source_url=(
            "https://huggingface.co/datasets/byml/UniVTAC/resolve/"
            f"{OFFICIAL_ACT_REVISION}/{remote}?download=true"
        ),
    )
    return OfficialACTInstallPlan(
        artifact_root=root,
        files=(item,),
        include_reference=False,
        lock_sha256="a" * 64,
        profiles=("univtac",),
        tasks=("insert_hole",),
    )


def test_streamed_install_is_hash_checked_atomic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"frozen official bytes"
    calls = 0

    def urlopen(_request: object, *, timeout: float) -> _Response:
        nonlocal calls
        assert timeout == 5.0
        calls += 1
        return _Response(data)

    monkeypatch.setattr(official_release_io.urllib.request, "urlopen", urlopen)
    root = tmp_path / "act"
    plan = _tiny_plan(root, data, ("encoder.pth", "copy/encoder.pth"))

    first = _install_validated_plan(plan, timeout_s=5.0)
    second = _install_validated_plan(plan, timeout_s=5.0)

    assert calls == 1
    assert first.downloaded_file_count == 1
    assert first.published_destination_count == 2
    assert second.downloaded_file_count == 0
    assert second.reused_destination_count == 2
    assert (root / "encoder.pth").read_bytes() == data
    assert (root / "copy/encoder.pth").read_bytes() == data
    receipt = json.loads(first.receipt_path.read_text(encoding="utf-8"))
    assert receipt["evidence_level"].endswith("no_model_or_simulator_execution_v1")
    assert receipt["install_plan"]["artifact_root"] == str(root)
    assert receipt["install_plan"]["tasks"] == ["insert_hole"]
    assert receipt["install_plan"]["profiles"] == ["univtac"]
    assert receipt["lock_sha256"] == receipt["install_plan"]["lock_sha256"]
    assert (
        receipt["reference_evidence_level"]
        == receipt["install_plan"]["reference_evidence_level"]
    )
    assert receipt["repository_id"] == receipt["install_plan"]["repository_id"]
    assert receipt["repository_type"] == receipt["install_plan"]["repository_type"]
    assert receipt["revision"] == receipt["install_plan"]["revision"]
    assert (
        receipt["plan_sha256"]
        == hashlib.sha256(canonical_json_bytes(receipt["install_plan"])).hexdigest()
    )
    assert first.receipt_path == second.receipt_path


def test_existing_mismatch_and_symlink_escape_fail_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "act"
    root.mkdir()
    (root / "encoder.pth").write_bytes(b"different")
    plan = _tiny_plan(root, b"expected", ("encoder.pth",))
    monkeypatch.setattr(
        official_release_io.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("network must not be used"),
    )

    with pytest.raises(FileExistsError, match="different ACT artifact"):
        _install_validated_plan(plan)

    with pytest.raises(OfficialACTReleaseError, match="safe POSIX path"):
        _tiny_plan(tmp_path / "other", b"x", ("../escape",))


def test_ancestor_symlink_and_receipt_conflict_fail_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        official_release_io.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("network must not be used"),
    )
    root = tmp_path / "act"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "nested").symlink_to(outside, target_is_directory=True)
    plan = _tiny_plan(root, b"expected", ("nested/encoder.pth",))

    with pytest.raises(OfficialACTReleaseError, match="parent is a symlink"):
        _install_validated_plan(plan)
    assert not (outside / "encoder.pth").exists()

    clean_plan = _tiny_plan(root, b"expected", ("encoder.pth",))
    receipt = root / "conflicting-receipt.json"
    receipt.write_bytes(b"different\n")
    with pytest.raises(FileExistsError, match="different ACT install receipt"):
        _install_validated_plan(clean_plan, receipt_path=receipt)
    assert not (root / "encoder.pth").exists()

    (root / "receipt-link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OfficialACTReleaseError, match="parent is a symlink"):
        _install_validated_plan(
            clean_plan,
            receipt_path=root / "receipt-link/install.json",
        )
    assert not (outside / "install.json").exists()

    artifact_link = tmp_path / "artifact-link"
    artifact_link.symlink_to(outside, target_is_directory=True)
    linked_plan = _tiny_plan(
        artifact_link / "child",
        b"expected",
        ("encoder.pth",),
    )
    with pytest.raises(OfficialACTReleaseError, match="root ancestor is a symlink"):
        _install_validated_plan(linked_plan)
    assert not (outside / "child").exists()


@pytest.mark.parametrize("response_kind", ("short", "wrong_hash", "interrupted"))
def test_bad_transfer_leaves_no_artifact_receipt_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_kind: str,
) -> None:
    expected = b"expected"
    if response_kind == "short":
        response: _Response = _Response(b"x")
    elif response_kind == "wrong_hash":
        response = _Response(b"xxxxxxxx")
    else:
        response = _InterruptedResponse()
    monkeypatch.setattr(
        official_release_io.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: response,
    )
    root = tmp_path / "act"
    plan = _tiny_plan(root, expected, ("encoder.pth",))

    with pytest.raises((OfficialACTReleaseError, OSError)):
        _install_validated_plan(plan)

    assert not (root / "encoder.pth").exists()
    assert not tuple((root / "official_release_receipts").glob("*.json"))
    assert not tuple(root.glob(".official-act-release-*"))


def test_public_install_rejects_manual_plan_before_write_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "act"
    plan = _tiny_plan(root, b"manual", ("encoder.pth",))
    monkeypatch.setattr(
        official_release_io.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("network must not be used"),
    )

    with pytest.raises(OfficialACTReleaseError, match="not frozen-lock-derived"):
        install_official_act_release(plan)

    assert not root.exists()


def test_cli_plan_is_network_free_and_canonical(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        (
            sys.executable,
            "scripts/act/install_official_artifacts.py",
            "--artifact-root",
            str(tmp_path / "act"),
            "--task",
            "lift_can",
            "--profile",
            "vision_only",
            "--dry-run",
        ),
        cwd=repository,
        check=True,
        capture_output=True,
        env={"PYTHONPATH": str(repository / "src")},
    )
    value = json.loads(result.stdout)

    assert value["tasks"] == ["lift_can"]
    assert value["profiles"] == ["vision_only"]
    assert not (tmp_path / "act").exists()
    assert result.stdout == canonical_json_bytes(value)
