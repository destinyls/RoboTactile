from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotactile_benchmark.policies.univtac_official_act_loading import (
    _STAT_KEYS,
    _load_stats,
)


def _valid_stats() -> dict[str, np.ndarray]:
    return {
        "qpos_mean": np.arange(8, dtype=np.float32),
        "qpos_std": np.ones(8, dtype=np.float32),
        "action_mean": np.arange(8, dtype=np.float32),
        "action_std": np.ones(8, dtype=np.float32),
        "example_qpos": np.zeros((3, 8), dtype=np.float32),
    }


def _write(path: Path, value: object) -> None:
    path.write_bytes(pickle.dumps(value, protocol=4))


class OfficialACTStatsTests(unittest.TestCase):
    def test_returns_only_four_readonly_exact_float32_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset_stats.pkl"
            _write(path, _valid_stats())
            loaded = _load_stats(path)

        self.assertEqual(set(loaded), _STAT_KEYS)
        for value in loaded.values():
            self.assertEqual(value.dtype, np.float32)
            self.assertEqual(value.shape, (8,))
            self.assertFalse(value.flags.writeable)

    def test_rejects_missing_malformed_nonfinite_and_nonpositive_stats(self) -> None:
        valid = _valid_stats()
        cases = {
            "missing": {
                key: value for key, value in valid.items() if key != "qpos_mean"
            },
            "shape": {**valid, "qpos_mean": np.zeros(9, dtype=np.float32)},
            "dtype": {**valid, "qpos_mean": np.zeros(8, dtype=np.float64)},
            "nonfinite": {
                **valid,
                "action_mean": np.full(8, np.nan, dtype=np.float32),
            },
            "nonpositive": {
                **valid,
                "action_std": np.zeros(8, dtype=np.float32),
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset_stats.pkl"
            for name, value in cases.items():
                _write(path, value)
                with (
                    self.subTest(name=name),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    _load_stats(path)


if __name__ == "__main__":
    unittest.main()
