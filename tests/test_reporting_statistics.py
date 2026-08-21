from __future__ import annotations

import unittest

from robotactile_benchmark.reporting.statistics import (
    bootstrap_kendall_tau_b,
    exact_mcnemar,
    holm_bonferroni,
    kendall_tau_b,
    task_stratified_paired_bootstrap,
)


class ReportingStatisticsTests(unittest.TestCase):
    def test_task_stratified_bootstrap_equal_weights_tasks(self) -> None:
        result = task_stratified_paired_bootstrap(
            {"task_a": (1.0, 1.0), "task_b": (-1.0, -1.0)},
            n_resamples=200,
            confidence_level=0.95,
            seed=17,
        )

        self.assertEqual(result.estimate, 0.0)
        self.assertEqual(result.lower, 0.0)
        self.assertEqual(result.upper, 0.0)
        self.assertEqual(result.task_count, 2)
        self.assertEqual(result.pair_count, 4)

    def test_exact_mcnemar_uses_only_discordant_pairs(self) -> None:
        result = exact_mcnemar(
            reference=(True, True, True, False),
            comparison=(False, False, False, False),
        )

        self.assertEqual(result.reference_only_successes, 3)
        self.assertEqual(result.comparison_only_successes, 0)
        self.assertAlmostEqual(result.p_value, 0.25)
        self.assertAlmostEqual(result.paired_success_delta, -0.75)

    def test_holm_bonferroni_returns_monotone_adjusted_values(self) -> None:
        adjusted = holm_bonferroni({"a": 0.01, "b": 0.03, "c": 0.04})

        self.assertEqual(adjusted, {"a": 0.03, "b": 0.06, "c": 0.06})

    def test_kendall_tau_b_and_bootstrap_handle_rank_reversal(self) -> None:
        result = kendall_tau_b((1.0, 2.0, 3.0, 4.0), (1.0, 3.0, 2.0, 4.0))
        perfect = bootstrap_kendall_tau_b(
            (1.0, 2.0, 3.0, 4.0),
            (10.0, 20.0, 30.0, 40.0),
            n_resamples=200,
            confidence_level=0.95,
            seed=23,
        )

        self.assertAlmostEqual(result.tau, 2.0 / 3.0)
        self.assertEqual((result.concordant, result.discordant), (5, 1))
        self.assertEqual(perfect.estimate, 1.0)
        self.assertEqual(perfect.lower, 1.0)
        self.assertEqual(perfect.upper, 1.0)
        self.assertGreater(perfect.valid_resamples, 0)

    def test_statistics_reject_nonfinite_or_unpaired_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            exact_mcnemar((True,), (True, False))
        with self.assertRaisesRegex(ValueError, "finite"):
            kendall_tau_b((1.0, float("nan")), (1.0, 2.0))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            task_stratified_paired_bootstrap(
                {}, n_resamples=10, confidence_level=0.95, seed=0
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            holm_bonferroni({"bad": 1.1})


if __name__ == "__main__":
    unittest.main()
