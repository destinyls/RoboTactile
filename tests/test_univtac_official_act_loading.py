from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np

import robotactile_benchmark.policies.univtac_official_act as official_policy
import robotactile_benchmark.policies.univtac_official_act_loading as loading
from robotactile_benchmark.backends.univtac_contracts import UPSTREAM_COMMIT
from robotactile_benchmark.closed_loop.contracts import ACTION_SPEC, PolicyIdentity
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.policies.univtac_official_act_loading import (
    OfficialUniVTACACTArtifactManifest,
    OfficialUniVTACACTLoadRequest,
    load_official_univtac_act_policy,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeModel:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[Mapping[str, object], bool]] = []

    def load_state_dict(self, state: Mapping[str, object], *, strict: bool) -> object:
        self.calls.append((state, strict))
        if self.fail:
            raise RuntimeError("strict state mismatch")
        return object()


class _FakeRuntime:
    def __init__(self, *, fail: bool = False) -> None:
        self.policy = _FakeModel(fail=fail)
        self.stats: Mapping[str, np.ndarray] | None = None
        self.close_count = 0

    def reset(self) -> None:
        return None

    def get_action(self, observation: Mapping[str, object]) -> np.ndarray:
        return np.zeros((1, 8), dtype=np.float32)

    def close(self) -> None:
        self.close_count += 1


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_count = 0

    def empty_cache(self) -> None:
        self.empty_cache_count += 1


class _FakeTorch:
    def __init__(self) -> None:
        self.loads: list[tuple[Path, object, bool]] = []
        self.cuda = _FakeCuda()

    def load(
        self, path: Path, *, map_location: object, weights_only: bool
    ) -> Mapping[str, object]:
        self.loads.append((path, map_location, weights_only))
        return {"model.weight": object()}

    @staticmethod
    def from_numpy(value: np.ndarray) -> np.ndarray:
        return value


def _write_stats(path: Path, *, dtype: np.dtype[Any] | None = None) -> None:
    normalized = np.dtype("float32") if dtype is None else dtype
    values = {
        "qpos_mean": np.arange(8, dtype=normalized),
        "qpos_std": np.ones(8, dtype=normalized),
        "action_mean": np.arange(8, dtype=normalized),
        "action_std": np.ones(8, dtype=normalized),
        "example_qpos": np.zeros((2, 8), dtype=normalized),
    }
    path.write_bytes(pickle.dumps(values, protocol=4))


class _Fixture:
    def __init__(self, root: Path, profile: OfficialACTProfile) -> None:
        self.root = root
        self.profile = profile
        self.artifacts = root / "checkpoints"
        self.upstream = root / "UniVTAC"
        profile_dir = self.artifacts / "pull_out_key" / profile.value
        profile_dir.mkdir(parents=True)
        self.checkpoint = profile_dir / "policy_last.ckpt"
        self.stats = profile_dir / "dataset_stats.pkl"
        self.encoder = self.artifacts / "encoder.pth"
        self.source = self.upstream / "policy/ACT/act_policy.py"
        config_name = (
            "train_config.yml"
            if profile is OfficialACTProfile.UNIVTAC
            else "train_config_vision.yml"
        )
        self.config = self.upstream / "policy/ACT" / config_name
        self.source.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"checkpoint-v1")
        _write_stats(self.stats)
        self.encoder.write_bytes(b"encoder-v1")
        self.source.write_bytes(b"pinned act source\n")
        self.config.write_bytes(b"pinned train config\n")

    def profile_specs(
        self,
    ) -> Mapping[OfficialACTProfile, official_policy._ProfileSpec]:
        return {
            OfficialACTProfile.UNIVTAC: official_policy._ProfileSpec(
                config_name="train_config.yml",
                config_sha256=_sha(self.config)
                if self.profile is OfficialACTProfile.UNIVTAC
                else "1" * 64,
                camera_names=("cam_high",),
                tactile_names=("tac_left", "tac_right"),
            ),
            OfficialACTProfile.VISION_ONLY: official_policy._ProfileSpec(
                config_name="train_config_vision.yml",
                config_sha256=_sha(self.config)
                if self.profile is OfficialACTProfile.VISION_ONLY
                else "2" * 64,
                camera_names=("cam_high",),
                tactile_names=(),
            ),
        }

    def manifest(self) -> OfficialUniVTACACTArtifactManifest:
        return OfficialUniVTACACTArtifactManifest.for_shared_root(
            task_id="pull_out_key",
            profile=self.profile,
            artifact_root=self.artifacts,
            upstream_root=self.upstream,
            checkpoint_sha256=_sha(self.checkpoint),
            stats_sha256=_sha(self.stats),
            encoder_sha256=_sha(self.encoder),
        )


def _identity(manifest: OfficialUniVTACACTArtifactManifest) -> PolicyIdentity:
    tactile = manifest.profile is OfficialACTProfile.UNIVTAC
    return PolicyIdentity(
        system_id=f"official-{manifest.profile.value}",
        checkpoint_sha256=manifest.checkpoint_sha256,
        config_sha256=manifest.config_sha256,
        action_spec=ACTION_SPEC,
        consumes_tactile=tactile,
        supports_structural_absence=not tactile,
    )


class OfficialManifestTests(unittest.TestCase):
    def test_manifest_pins_shared_layout_source_config_and_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary), OfficialACTProfile.UNIVTAC)
            with (
                patch.object(loading, "_ACT_SOURCE_SHA256", _sha(fixture.source)),
                patch.object(loading, "_PROFILE_SPECS", fixture.profile_specs()),
            ):
                manifest = fixture.manifest()

        self.assertEqual(manifest.upstream_commit, UPSTREAM_COMMIT)
        self.assertEqual(manifest.checkpoint_path.name, "policy_last.ckpt")
        self.assertEqual(manifest.stats_path.name, "dataset_stats.pkl")
        self.assertEqual(manifest.encoder_path, fixture.artifacts / "encoder.pth")
        self.assertEqual(
            manifest.source_path, fixture.upstream / "policy/ACT/act_policy.py"
        )
        self.assertEqual(manifest.config_path.name, "train_config.yml")

    def test_source_config_task_and_profile_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary), OfficialACTProfile.VISION_ONLY)
            specs = fixture.profile_specs()
            with (
                patch.object(loading, "_ACT_SOURCE_SHA256", _sha(fixture.source)),
                patch.object(loading, "_PROFILE_SPECS", specs),
            ):
                manifest = fixture.manifest()
                mutations = (
                    {"source_path": fixture.upstream / "shadow.py"},
                    {"config_path": fixture.upstream / "policy/ACT/train_config.yml"},
                    {"task_id": "unknown-task"},
                    {"profile": OfficialACTProfile.UNIVTAC},
                    {"upstream_commit": "0" * 40},
                )
                for mutation in mutations:
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        replace(manifest, **mutation)


class OfficialLoaderTests(unittest.TestCase):
    def _load(
        self,
        fixture: _Fixture,
        *,
        runtime: _FakeRuntime | None = None,
    ) -> tuple[object, _FakeTorch, _FakeRuntime]:
        fake_torch = _FakeTorch()
        fake_runtime = runtime or _FakeRuntime()
        with (
            patch.object(loading, "_ACT_SOURCE_SHA256", _sha(fixture.source)),
            patch.object(loading, "_PROFILE_SPECS", fixture.profile_specs()),
            patch.object(loading, "_git_head", return_value=UPSTREAM_COMMIT),
            patch.object(loading, "_import_torch", return_value=fake_torch),
            patch.object(
                loading, "_construct_upstream_runtime", return_value=fake_runtime
            ),
        ):
            manifest = fixture.manifest()
            policy = load_official_univtac_act_policy(
                _identity(manifest),
                OfficialUniVTACACTLoadRequest(
                    manifest=manifest,
                    task_id="pull_out_key",
                    profile=fixture.profile,
                    device_name="cuda:2",
                    live=True,
                ),
            )
        return policy, fake_torch, fake_runtime

    def test_live_loader_uses_weights_only_and_strict_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary), OfficialACTProfile.UNIVTAC)
            policy, fake_torch, runtime = self._load(fixture)
            checkpoint_sha256 = _sha(fixture.checkpoint)

        self.assertEqual(
            fake_torch.loads,
            [(fixture.checkpoint, "cpu", True)],
        )
        self.assertEqual(len(runtime.policy.calls), 1)
        self.assertTrue(runtime.policy.calls[0][1])
        self.assertEqual(set(runtime.stats or {}), loading._STAT_KEYS)
        self.assertEqual(policy.identity.checkpoint_sha256, checkpoint_sha256)

    def test_hash_and_identity_mismatches_fail_before_torch_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary), OfficialACTProfile.UNIVTAC)
            with (
                patch.object(loading, "_ACT_SOURCE_SHA256", _sha(fixture.source)),
                patch.object(loading, "_PROFILE_SPECS", fixture.profile_specs()),
                patch.object(loading, "_git_head", return_value=UPSTREAM_COMMIT),
            ):
                manifest = fixture.manifest()
                request = OfficialUniVTACACTLoadRequest(
                    manifest=manifest,
                    task_id="pull_out_key",
                    profile=OfficialACTProfile.UNIVTAC,
                    device_name="cuda:0",
                    live=True,
                )
                for name, path in (
                    ("checkpoint", fixture.checkpoint),
                    ("stats", fixture.stats),
                    ("encoder", fixture.encoder),
                    ("source", fixture.source),
                    ("config", fixture.config),
                ):
                    original = path.read_bytes()
                    path.write_bytes(original + b"mutated")
                    with (
                        self.subTest(name=name),
                        patch.object(loading, "_import_torch") as import_torch,
                        self.assertRaisesRegex(ValueError, f"{name}.*SHA256"),
                    ):
                        load_official_univtac_act_policy(_identity(manifest), request)
                    import_torch.assert_not_called()
                    path.write_bytes(original)

                wrong_identity = replace(_identity(manifest), config_sha256="f" * 64)
                with patch.object(loading, "_import_torch") as import_torch:
                    with self.assertRaisesRegex(ValueError, "config"):
                        load_official_univtac_act_policy(wrong_identity, request)
                    import_torch.assert_not_called()

    def test_task_profile_and_nonlive_requests_fail_before_torch_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary), OfficialACTProfile.VISION_ONLY)
            with (
                patch.object(loading, "_ACT_SOURCE_SHA256", _sha(fixture.source)),
                patch.object(loading, "_PROFILE_SPECS", fixture.profile_specs()),
            ):
                manifest = fixture.manifest()
                base = OfficialUniVTACACTLoadRequest(
                    manifest=manifest,
                    task_id="pull_out_key",
                    profile=OfficialACTProfile.VISION_ONLY,
                    device_name="cuda:0",
                    live=True,
                )
                for request in (
                    replace(base, task_id="insert_HDMI"),
                    replace(base, profile=OfficialACTProfile.UNIVTAC),
                    replace(base, live=False),
                ):
                    with (
                        self.subTest(request=request),
                        patch.object(loading, "_import_torch") as import_torch,
                        self.assertRaises(ValueError),
                    ):
                        load_official_univtac_act_policy(_identity(manifest), request)
                    import_torch.assert_not_called()

    def test_stats_are_restricted_and_validated_after_hash_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary), OfficialACTProfile.UNIVTAC)
            _write_stats(fixture.stats, dtype=np.dtype("float64"))
            with (
                patch.object(loading, "_ACT_SOURCE_SHA256", _sha(fixture.source)),
                patch.object(loading, "_PROFILE_SPECS", fixture.profile_specs()),
                patch.object(loading, "_git_head", return_value=UPSTREAM_COMMIT),
            ):
                manifest = fixture.manifest()
                request = OfficialUniVTACACTLoadRequest(
                    manifest=manifest,
                    task_id="pull_out_key",
                    profile=OfficialACTProfile.UNIVTAC,
                    device_name="cuda:0",
                    live=True,
                )
                with patch.object(loading, "_import_torch") as import_torch:
                    with self.assertRaisesRegex(TypeError, "float32"):
                        load_official_univtac_act_policy(_identity(manifest), request)
                    import_torch.assert_not_called()

    def test_restricted_stats_unpickler_rejects_executable_global(self) -> None:
        class Exploit:
            def __reduce__(self) -> tuple[object, tuple[str]]:
                return (os.system, ("touch should-not-exist",))

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary), OfficialACTProfile.UNIVTAC)
            marker = Path(temporary) / "should-not-exist"
            fixture.stats.write_bytes(pickle.dumps(Exploit(), protocol=4))
            with (
                patch.object(loading, "_ACT_SOURCE_SHA256", _sha(fixture.source)),
                patch.object(loading, "_PROFILE_SPECS", fixture.profile_specs()),
                patch.object(loading, "_git_head", return_value=UPSTREAM_COMMIT),
            ):
                manifest = fixture.manifest()
                request = OfficialUniVTACACTLoadRequest(
                    manifest=manifest,
                    task_id="pull_out_key",
                    profile=OfficialACTProfile.UNIVTAC,
                    device_name="cuda:0",
                    live=True,
                )
                with self.assertRaises(pickle.UnpicklingError):
                    load_official_univtac_act_policy(_identity(manifest), request)
            self.assertFalse(marker.exists())

    def test_strict_load_failure_releases_constructed_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary), OfficialACTProfile.UNIVTAC)
            runtime = _FakeRuntime(fail=True)
            with self.assertRaisesRegex(RuntimeError, "strict state mismatch"):
                self._load(fixture, runtime=runtime)

        self.assertEqual(runtime.close_count, 1)


if __name__ == "__main__":
    unittest.main()
