import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from robotactile_benchmark.cli import main as cli_main
from robotactile_benchmark.fixtures import make_synthetic_rest_references
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.metrics import (
    psnr,
    tactile_gain_retention,
)
from robotactile_benchmark.replay import (
    load_replay_artifacts,
    run_smoke_matrix,
    run_smoke_replay,
)


class MetricTests(unittest.TestCase):
    def test_psnr_analytic_identical_and_maximal_error_pairs(self) -> None:
        import numpy as np

        zeros = np.zeros((4, 4, 3), dtype=np.float32)
        ones = np.ones((4, 4, 3), dtype=np.float32)

        self.assertEqual(psnr(zeros, zeros), float("inf"))
        self.assertAlmostEqual(psnr(zeros, ones), 0.0)

    def test_tgr_follows_paper_contract(self) -> None:
        self.assertAlmostEqual(
            tactile_gain_retention(clean=0.8, no_touch=0.5, fault=0.65), 0.5
        )
        with self.assertRaisesRegex(ValueError, "clean tactile gain"):
            tactile_gain_retention(clean=0.5, no_touch=0.5, fault=0.4)

    def test_metrics_reject_nonfinite_coercive_and_negative_controls(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                tactile_gain_retention(value, 0.2, 0.3)
        with self.assertRaisesRegex(ValueError, "positive"):
            tactile_gain_retention(0.8, 0.5, 0.6, minimum_gain=-1.0)


class OfflineReplayTests(unittest.TestCase):
    def test_cli_builds_a_valid_warmup_for_maximum_t1_delay(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            exit_code = cli_main(
                [
                    "smoke-replay",
                    "--output",
                    output_dir,
                    "--operator",
                    "T1_fixed_source_delay",
                    "--severity",
                    "5",
                ]
            )
            loaded = load_replay_artifacts(Path(output_dir))

        self.assertEqual(exit_code, 0)
        self.assertTrue(loaded["passed"])

    def test_smoke_replay_writes_loadable_deterministic_artifacts(self) -> None:
        manifest = FaultManifest(
            operator_id="F6_history_residual_imprint",
            severity_level=3,
            operator_seed=44,
            start_index=3,
            stop_index=9,
            sensor_slots=("left",),
            observability=Observability.BLIND,
            parameters={
                "rest_reference_sha256": make_synthetic_rest_references().sha256
            },
        )
        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            first = run_smoke_replay(Path(first_dir), manifest)
            second = run_smoke_replay(Path(second_dir), manifest)
            loaded = load_replay_artifacts(Path(first_dir))

        self.assertTrue(first.validation.passed)
        self.assertEqual(first.trace_sha256, second.trace_sha256)
        self.assertEqual(loaded["trace_sha256"], first.trace_sha256)
        self.assertEqual(loaded["manifest_sha256"], manifest.sha256)
        self.assertEqual(len(loaded["episode_artifact_sha256"]), 64)

    def test_replay_loader_rejects_manifest_and_episode_corruption(self) -> None:
        manifest = FaultManifest(
            operator_id="F6_history_residual_imprint",
            severity_level=3,
            operator_seed=44,
            start_index=3,
            stop_index=9,
            sensor_slots=("left",),
            observability=Observability.BLIND,
            parameters={
                "rest_reference_sha256": make_synthetic_rest_references().sha256
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            run_smoke_replay(output_dir, manifest)
            manifest_path = output_dir / "fault_manifest.json"
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stored_manifest["severity_level"] = 4
            manifest_path.write_text(
                json.dumps(stored_manifest, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                load_replay_artifacts(output_dir)

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            run_smoke_replay(output_dir, manifest)
            episode_path = output_dir / "delivered_episode.npz"
            content = bytearray(episode_path.read_bytes())
            content[-1] ^= 1
            episode_path.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "episode artifact hash"):
                load_replay_artifacts(output_dir)

    def test_replay_loader_rejects_unexpected_npz_members_before_loading(self) -> None:
        manifest = FaultManifest(
            operator_id="A1_stream_absence",
            severity_level=1,
            operator_seed=44,
            start_index=3,
            stop_index=9,
            sensor_slots=("left",),
            observability=Observability.BLIND,
            parameters={},
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            run_smoke_replay(output_dir, manifest)
            episode_path = output_dir / "delivered_episode.npz"
            with zipfile.ZipFile(episode_path, mode="a") as archive:
                archive.writestr("unexpected.npy", b"not-an-array")
            report_path = output_dir / "validation_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["episode_artifact_sha256"] = hashlib.sha256(
                episode_path.read_bytes()
            ).hexdigest()
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "NPZ member"):
                load_replay_artifacts(output_dir)

    def test_replay_loader_rejects_coforged_payload_report_and_hash(self) -> None:
        import numpy as np

        manifest = FaultManifest(
            operator_id="A1_stream_absence",
            severity_level=2,
            operator_seed=44,
            start_index=3,
            stop_index=9,
            sensor_slots=("left",),
            observability=Observability.BLIND,
            parameters={},
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            run_smoke_replay(output_dir, manifest)
            episode_path = output_dir / "delivered_episode.npz"
            with np.load(episode_path) as original:
                rgb = original["tactile_rgb"].copy()
                present = original["payload_present"].copy()
                sources = original["source_index"].copy()
            sources[0, 0] = 999999
            np.savez_compressed(
                episode_path,
                tactile_rgb=rgb,
                payload_present=present,
                source_index=sources,
            )
            report_path = output_dir / "validation_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["episode_artifact_sha256"] = hashlib.sha256(
                episode_path.read_bytes()
            ).hexdigest()
            report["trace_sha256"] = "f" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "reconstruction"):
                load_replay_artifacts(output_dir)

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            run_smoke_replay(output_dir, manifest)
            episode_path = output_dir / "delivered_episode.npz"
            original = episode_path.read_bytes()
            with episode_path.open("ab") as stream:
                stream.write(b"untracked-tail")
            self.assertNotEqual(episode_path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "episode artifact hash"):
                load_replay_artifacts(output_dir)

    def test_smoke_matrix_freezes_all_fourteen_by_five_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "matrix"
            summary = run_smoke_matrix(output_dir)
            stored = json.loads(
                (output_dir / "matrix_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["cell_count"], 70)
        self.assertEqual(summary, stored)
        self.assertTrue(all(cell["validation_passed"] for cell in summary["cells"]))
        self.assertEqual(
            len(
                {
                    (cell["operator_id"], cell["severity_level"])
                    for cell in summary["cells"]
                }
            ),
            70,
        )


if __name__ == "__main__":
    unittest.main()
