import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.resources import (
    load_operator_registry,
    load_severity_registry,
    load_source_manifest,
)
from robotactile_benchmark.severity import SEVERITY_PATHS
from robotactile_benchmark.trials import (
    TRIAL_MANIFEST_SEMANTIC_VERSION,
    Condition,
    TerminalStatus,
    TrialManifest,
    build_paired_trial_grid,
    system_manifest_hash,
)


def _trial(**overrides):
    values = {
        "task": "insert_HDMI",
        "initial_seed": 10,
        "exogenous_seed": 20,
        "condition": Condition.CLEAN,
        "base_system_id": "n0-twam-track31",
        "executed_system_id": "n0-twam-track31",
        "dataset_sha256": "a" * 64,
        "base_system_manifest_sha256": system_manifest_hash(
            "n0-twam-track31",
            "b" * 64,
            "c" * 64,
            "qpos8_next_step",
        ),
        "checkpoint_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "action_spec": "qpos8_next_step",
        "fault_manifest_sha256": None,
        "matched_no_touch_system_id": None,
    }
    values.update(overrides)
    return TrialManifest(**values)


def _fault(stop_index: int) -> FaultManifest:
    return FaultManifest(
        operator_id="T1_fixed_source_delay",
        severity_level=1,
        operator_seed=33,
        start_index=2,
        stop_index=stop_index,
        sensor_slots=("left",),
        observability=Observability.BLIND,
        parameters={},
    )


def _expected_release_paths(root: Path) -> set[str]:
    expected = {
        ".gitignore",
        "BENCHMARK_CARD.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "LICENSE",
        "README.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        "uv.lock",
    }
    excluded_directories = {
        ".git",
        "__pycache__",
        "build",
        "dist",
        "output",
        "outputs",
    }
    for directory_name in (
        ".github",
        "configs",
        "docs",
        "examples",
        "integrations",
        "requirements",
        "schemas",
        "scripts",
        "src",
        "tests",
    ):
        for path in (root / directory_name).rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if (
                ".DS_Store" in relative.parts
                or set(relative.parts) & excluded_directories
            ):
                continue
            top_level = relative.parts[0]
            eligible = (
                top_level in {"configs", "requirements", "schemas"}
                or (top_level == "src" and path.suffix == ".py")
                or (
                    top_level == "tests"
                    and path.name.startswith("test_")
                    and path.suffix == ".py"
                )
                or (top_level == "docs" and path.suffix == ".md")
                or top_level == "integrations"
                or (
                    top_level in {".github", "examples"}
                    and path.suffix in {".json", ".md", ".sh", ".yaml", ".yml"}
                )
                or (
                    top_level == "scripts"
                    and (
                        path.suffix in {".py", ".sh"}
                        or path.name in {"README", "README.md"}
                    )
                )
            )
            if eligible:
                expected.add(relative.as_posix())
    return expected


class ResourceContractTests(unittest.TestCase):
    def test_machine_registry_matches_runtime_exactly(self) -> None:
        operator_registry = load_operator_registry()
        severity_registry = load_severity_registry()

        self.assertEqual(
            {entry["operator_id"] for entry in operator_registry["operators"]},
            set(CORE_OPERATOR_IDS),
        )
        self.assertEqual(
            operator_registry["registry_id"], "robotactile_core_2_7_3_2_v2"
        )
        self.assertEqual(operator_registry["semantic_version"], "2.0")
        self.assertEqual(operator_registry["implementation_version"], "0.4.0")
        self.assertEqual(severity_registry["registry_id"], "provisional_engineering_v2")
        self.assertEqual(severity_registry["semantic_version"], "2.0")
        self.assertEqual(set(severity_registry["paths"]), set(CORE_OPERATOR_IDS))
        for operator_id, path in SEVERITY_PATHS.items():
            self.assertEqual(
                tuple(severity_registry["paths"][operator_id]["values"]), path
            )

    def test_resource_payloads_are_json_round_trip_safe(self) -> None:
        for payload in (load_operator_registry(), load_severity_registry()):
            self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_resource_loader_rejects_packaged_digest_drift(self) -> None:
        from robotactile_benchmark import resources as resource_module

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "configs" / "operators"
            path.mkdir(parents=True)
            (path / "core_v2.json").write_text("{}\n", encoding="utf-8")
            with (
                patch.object(resource_module.resources, "files", return_value=root),
                self.assertRaisesRegex(ValueError, "digest drift"),
            ):
                resource_module.load_operator_registry()

    def test_release_manifest_is_packaged_and_matches_the_source_tree(self) -> None:
        root = Path(__file__).resolve().parents[1]
        lines = tuple(line for line in load_source_manifest().splitlines() if line)
        self.assertGreater(len(lines), 40)
        source_tree = root / "src" / "robotactile_benchmark"
        check_local_files = source_tree.exists()
        listed_paths = {line.split("  ", 1)[1] for line in lines}
        if check_local_files:
            self.assertEqual(listed_paths, _expected_release_paths(root))
        for line in lines:
            expected, relative_path = line.split("  ", 1)
            if check_local_files:
                payload = (root / relative_path).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)


class TrialManifestTests(unittest.TestCase):
    def test_three_conditions_share_one_pair_key_and_use_a_real_control(self) -> None:
        grid = build_paired_trial_grid(
            clean=_trial(),
            faulted_manifest=_fault(80),
            no_touch_system_id="n0-twam-track31-no-touch",
            no_touch_checkpoint_sha256="e" * 64,
            no_touch_config_sha256="f" * 64,
        )

        self.assertEqual(tuple(item.condition for item in grid), tuple(Condition))
        self.assertEqual(len({item.pair_key for item in grid}), 1)
        no_touch = next(item for item in grid if item.condition is Condition.NO_TOUCH)
        self.assertEqual(no_touch.executed_system_id, "n0-twam-track31-no-touch")
        self.assertNotEqual(no_touch.checkpoint_sha256, grid[0].checkpoint_sha256)
        self.assertIsNone(no_touch.fault_manifest_sha256)
        for item in grid:
            reloaded = TrialManifest.from_dict(item.to_dict())
            self.assertEqual(item.semantic_version, TRIAL_MANIFEST_SEMANTIC_VERSION)
            self.assertEqual(reloaded, item)
            self.assertEqual(reloaded.sha256, item.sha256)

    def test_v1_restored_era_trial_manifest_fails_closed(self) -> None:
        document = _trial().to_dict()
        document["semantic_version"] = "1.0"

        with self.assertRaisesRegex(ValueError, "semantic version"):
            TrialManifest.from_dict(document)

    def test_invalid_condition_combinations_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "clean"):
            _trial(fault_manifest_sha256="d" * 64)
        with self.assertRaisesRegex(ValueError, "fault manifest"):
            _trial(condition=Condition.FAULTED)
        with self.assertRaisesRegex(ValueError, "matched no-touch"):
            _trial(condition=Condition.NO_TOUCH)
        with self.assertRaisesRegex(ValueError, "executed system"):
            _trial(
                condition=Condition.NO_TOUCH,
                matched_no_touch_system_id="control",
            )
        with self.assertRaisesRegex(ValueError, "base system"):
            _trial(executed_system_id="different-clean-system")
        with self.assertRaises(TypeError):
            _trial(initial_seed=True)

    def test_pair_key_binds_the_base_checkpoint_and_config_identity(self) -> None:
        original = _trial()
        changed = _trial(
            checkpoint_sha256="8" * 64,
            base_system_manifest_sha256=system_manifest_hash(
                "n0-twam-track31",
                "8" * 64,
                "c" * 64,
                "qpos8_next_step",
            ),
        )
        self.assertNotEqual(original.pair_key, changed.pair_key)

    def test_terminal_status_enumerates_failures_instead_of_dropping_trials(
        self,
    ) -> None:
        self.assertEqual(
            {status.value for status in TerminalStatus},
            {
                "success",
                "task_failure",
                "early_stop",
                "timeout",
                "crash",
                "validator_rejected",
                "unsupported_contract",
            },
        )


if __name__ == "__main__":
    unittest.main()
