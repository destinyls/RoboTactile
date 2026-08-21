"""Adversarial loader and output-safety tests for closed-loop artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from robotactile_benchmark.closed_loop.artifacts import load_closed_loop_bundle
from robotactile_benchmark.closed_loop.smoke import write_cpu_smoke_bundle
from robotactile_benchmark.contracts import array_sha256


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


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("fixture JSON must be an object")
    return value


def _root(bundle: Path) -> dict[str, Any]:
    return _json(bundle / "root_receipt.json")


def _update_member(
    root: dict[str, Any],
    old_path: str,
    new_path: str,
    raw: bytes,
) -> None:
    members = root["members"]
    for member in members:
        if member["path"] == old_path:
            member.update(
                path=new_path,
                sha256=_digest(raw),
                size_bytes=len(raw),
            )
            members.sort(key=lambda value: value["path"])
            return
    raise AssertionError(f"missing root member {old_path}")


def _rewrite_json(
    bundle: Path,
    relative: str,
    value: object,
    root: dict[str, Any],
    *,
    raw: bytes | None = None,
) -> None:
    payload = _canonical(value) if raw is None else raw
    (bundle / relative).write_bytes(payload)
    _update_member(root, relative, relative, payload)


def _save_root(bundle: Path, root: dict[str, Any]) -> None:
    (bundle / "root_receipt.json").write_bytes(_canonical(root))


def _npy(
    value: np.ndarray,
    *,
    allow_pickle: bool = False,
    version: tuple[int, int] = (1, 0),
) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(value),
        version=version,
        allow_pickle=allow_pickle,
    )
    return stream.getvalue()


def _replace_all_descriptors(
    value: object, old_path: str, descriptor: dict[str, object]
) -> int:
    changed = 0
    if isinstance(value, dict):
        if value.get("path") == old_path and set(value) == {
            "path",
            "file_sha256",
            "array_sha256",
            "dtype",
            "shape",
        }:
            value.clear()
            value.update(descriptor)
            return 1
        for child in value.values():
            changed += _replace_all_descriptors(child, old_path, descriptor)
    elif isinstance(value, list):
        for child in value:
            changed += _replace_all_descriptors(child, old_path, descriptor)
    return changed


def _replace_array(
    bundle: Path,
    json_name: str,
    descriptor: dict[str, Any],
    value: np.ndarray,
    *,
    allow_pickle: bool = False,
    trailing: bytes = b"",
    version: tuple[int, int] = (1, 0),
) -> None:
    old_path = descriptor["path"]
    raw = _npy(value, allow_pickle=allow_pickle, version=version) + trailing
    file_hash = _digest(raw)
    new_path = f"arrays/{file_hash}.npy"
    content_hash = array_sha256(value)
    if content_hash is None:
        raise AssertionError("test array hash cannot be null")
    replacement: dict[str, object] = {
        "path": new_path,
        "file_sha256": file_hash,
        "array_sha256": content_hash,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }
    document = _json(bundle / json_name)
    if _replace_all_descriptors(document, old_path, replacement) < 1:
        raise AssertionError("descriptor was not found")
    root = _root(bundle)
    old_file = bundle / old_path
    old_file.unlink()
    new_file = bundle / new_path
    new_file.parent.mkdir(parents=True, exist_ok=True)
    new_file.write_bytes(raw)
    _update_member(root, old_path, new_path, raw)
    _rewrite_json(bundle, json_name, document, root)
    _save_root(bundle, root)


class ClosedLoopArtifactTamperTests(unittest.TestCase):
    def _bundle(self, temporary: str, name: str = "bundle") -> Path:
        bundle = Path(temporary) / name
        write_cpu_smoke_bundle(bundle)
        return bundle

    def test_json_with_updated_file_hash_but_stale_cross_link_is_rejected(self) -> None:
        """Trusting only file hashes must break this test."""

        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._bundle(temporary)
            document = _json(bundle / "run_spec.json")
            document["prompt"] = "forged prompt"
            root = _root(bundle)
            _rewrite_json(bundle, "run_spec.json", document, root)
            _save_root(bundle, root)

            with self.assertRaises(ValueError):
                load_closed_loop_bundle(bundle)

    def test_record_array_with_updated_local_hashes_is_rejected(self) -> None:
        """Recomputing array, descriptor, file, and root hashes must not bypass record hashes."""

        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._bundle(temporary)
            document = _json(bundle / "delivery_trace.json")
            descriptor = document["clean_records"][0]["observation"]["tactile"][0][
                "payload"
            ]
            array = np.load(bundle / descriptor["path"], allow_pickle=False).copy()
            array[0, 0, 0] ^= np.uint8(1)
            _replace_array(bundle, "delivery_trace.json", descriptor, array)

            with self.assertRaises(ValueError):
                load_closed_loop_bundle(bundle)

    def test_forged_actions_with_all_local_hashes_are_rejected(self) -> None:
        """Changing executed bytes must fail the Task 3 action/terminal link."""

        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._bundle(temporary)
            document = _json(bundle / "action_trace.json")
            descriptor = document["entries"][0]["executed_actions"]
            array = np.load(bundle / descriptor["path"], allow_pickle=False).copy()
            array[0, 0] += np.float32(0.5)
            _replace_array(bundle, "action_trace.json", descriptor, array)

            with self.assertRaises(ValueError):
                load_closed_loop_bundle(bundle)

    def test_exact_inventory_and_paths_fail_closed(self) -> None:
        """Unknown, missing, symlinked, and escaping members must all be rejected."""

        with tempfile.TemporaryDirectory() as temporary:
            unknown = self._bundle(temporary, "unknown")
            (unknown / "surprise.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_closed_loop_bundle(unknown)

            missing = self._bundle(temporary, "missing")
            (missing / "trial_manifest.json").unlink()
            with self.assertRaises(ValueError):
                load_closed_loop_bundle(missing)

            escaping = self._bundle(temporary, "escaping")
            root = _root(escaping)
            root["members"][0]["path"] = "../escape.npy"
            _save_root(escaping, root)
            with self.assertRaises(ValueError):
                load_closed_loop_bundle(escaping)

            symlinked = self._bundle(temporary, "symlinked")
            member = symlinked / "trial_manifest.json"
            target = Path(temporary) / "outside.json"
            target.write_bytes(member.read_bytes())
            member.unlink()
            member.symlink_to(target)
            with self.assertRaises(ValueError):
                load_closed_loop_bundle(symlinked)

    def test_noncanonical_duplicate_nan_and_oversized_members_are_rejected(
        self,
    ) -> None:
        """Permissive JSON or size handling must break this test."""

        with tempfile.TemporaryDirectory() as temporary:
            noncanonical = self._bundle(temporary, "noncanonical")
            document = _json(noncanonical / "run_spec.json")
            root = _root(noncanonical)
            pretty = (json.dumps(document, indent=2) + "\n").encode("utf-8")
            _rewrite_json(noncanonical, "run_spec.json", document, root, raw=pretty)
            _save_root(noncanonical, root)
            with self.assertRaises(ValueError):
                load_closed_loop_bundle(noncanonical)

            duplicate = self._bundle(temporary, "duplicate")
            raw = (duplicate / "run_spec.json").read_bytes()
            forged = b'{"prompt":"duplicate",' + raw[1:]
            root = _root(duplicate)
            _rewrite_json(duplicate, "run_spec.json", {}, root, raw=forged)
            _save_root(duplicate, root)
            with self.assertRaises(ValueError):
                load_closed_loop_bundle(duplicate)

            nan_json = self._bundle(temporary, "nan-json")
            raw = (nan_json / "run_spec.json").read_bytes().replace(b"5.0", b"NaN")
            root = _root(nan_json)
            _rewrite_json(nan_json, "run_spec.json", {}, root, raw=raw)
            _save_root(nan_json, root)
            with self.assertRaises(ValueError):
                load_closed_loop_bundle(nan_json)

            oversized = self._bundle(temporary, "oversized")
            action = _json(oversized / "action_trace.json")
            descriptor = action["entries"][0]["executed_actions"]
            array = np.load(oversized / descriptor["path"], allow_pickle=False)
            _replace_array(
                oversized,
                "action_trace.json",
                descriptor,
                array,
                trailing=b"x" * (17 * 1024 * 1024),
            )
            with self.assertRaises(ValueError):
                load_closed_loop_bundle(oversized)

    def test_object_nonfinite_and_trailing_npy_payloads_are_rejected(self) -> None:
        """Unsafe dtype, nonfinite values, or trailing bytes must never reach contracts."""

        cases = (
            ("object", np.array(["unsafe"], dtype=object), True, b""),
            ("nonfinite", np.full((1, 8), np.nan, dtype=np.float32), False, b""),
            ("trailing", np.zeros((1, 8), dtype=np.float32), False, b"trailing"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for name, array, allow_pickle, trailing in cases:
                with self.subTest(name=name):
                    bundle = self._bundle(temporary, name)
                    action = _json(bundle / "action_trace.json")
                    descriptor = action["entries"][0]["executed_actions"]
                    _replace_array(
                        bundle,
                        "action_trace.json",
                        descriptor,
                        array,
                        allow_pickle=allow_pickle,
                        trailing=trailing,
                    )
                    with self.assertRaises(ValueError):
                        load_closed_loop_bundle(bundle)

    def test_semantically_equal_npy_v2_payload_is_rejected(self) -> None:
        """Accepting a noncanonical NPY version must break byte reproducibility."""

        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._bundle(temporary)
            action = _json(bundle / "action_trace.json")
            descriptor = action["entries"][0]["executed_actions"]
            array = np.load(bundle / descriptor["path"], allow_pickle=False)
            _replace_array(
                bundle,
                "action_trace.json",
                descriptor,
                array,
                version=(2, 0),
            )

            with self.assertRaises(ValueError):
                load_closed_loop_bundle(bundle)

    def test_unrelated_nonempty_output_is_never_overwritten(self) -> None:
        """Writer must reject rather than replace unrelated user content."""

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "occupied"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_bytes(b"user-owned")

            with self.assertRaises(FileExistsError):
                write_cpu_smoke_bundle(output)

            self.assertEqual(marker.read_bytes(), b"user-owned")
            self.assertEqual({path.name for path in output.iterdir()}, {"keep.txt"})


if __name__ == "__main__":
    unittest.main()
