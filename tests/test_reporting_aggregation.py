from __future__ import annotations

import hashlib
import unittest

from robotactile_benchmark.reporting import (
    OutcomeRecord,
    ReportingSpec,
    aggregate_benchmark,
)
from robotactile_benchmark.trials import Condition, TerminalStatus

_HEXES = tuple(f"{index:064x}" for index in range(1, 200))


def _outcome(
    *,
    task: str,
    pair: str,
    condition: Condition,
    success: bool | None,
    digest_index: int,
    operator: str | None = None,
    severity: int | None = None,
) -> OutcomeRecord:
    eligible = success is not None
    return OutcomeRecord(
        system_id="touch-policy",
        task=task,
        pair_key=hashlib.sha256(pair.encode("utf-8")).hexdigest(),
        condition=condition,
        terminal_status=(
            TerminalStatus.SUCCESS if success else TerminalStatus.TASK_FAILURE
        )
        if eligible
        else TerminalStatus.UNSUPPORTED_CONTRACT,
        score_eligible=eligible,
        score_success=success,
        source_root_sha256=_HEXES[digest_index],
        operator_id=operator,
        severity_level=severity,
    )


def _fixture() -> tuple[OutcomeRecord, ...]:
    records: list[OutcomeRecord] = []
    pairs = (
        ("task_a", "a1", True, False),
        ("task_a", "a2", True, True),
        ("task_b", "b1", True, False),
        ("task_b", "b2", False, False),
    )
    fault_values = {
        ("A1_stream_absence", 1): (False, True, False, False),
        ("A1_stream_absence", 2): (False, False, False, False),
        ("F1_global_response_drift", 1): (True, True, True, False),
        ("F1_global_response_drift", 2): (True, False, False, False),
    }
    digest_index = 0
    for task, pair, clean, no_touch in pairs:
        records.append(
            _outcome(
                task=task,
                pair=pair,
                condition=Condition.CLEAN,
                success=clean,
                digest_index=digest_index,
            )
        )
        digest_index += 1
        records.append(
            _outcome(
                task=task,
                pair=pair,
                condition=Condition.NO_TOUCH,
                success=no_touch,
                digest_index=digest_index,
            )
        )
        digest_index += 1
    for (operator, severity), values in fault_values.items():
        for (task, pair, _, _), success in zip(pairs, values):
            records.append(
                _outcome(
                    task=task,
                    pair=pair,
                    condition=Condition.FAULTED,
                    success=success,
                    digest_index=digest_index,
                    operator=operator,
                    severity=severity,
                )
            )
            digest_index += 1
    return tuple(records)


class ReportingAggregationTests(unittest.TestCase):
    def test_legacy_v1_reporting_spec_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "semantic version"):
            ReportingSpec(
                system_id="touch-policy",
                matched_control_qualified=True,
                minimum_clean_gain=0.05,
                primary_operator_ids=("A1_stream_absence",),
                bootstrap_resamples=100,
                bootstrap_seed=7,
                semantic_version="1.0",
            )

    def test_aggregation_uses_task_axis_operator_and_severity_equal_weighting(
        self,
    ) -> None:
        summary = aggregate_benchmark(
            _fixture(),
            ReportingSpec(
                system_id="touch-policy",
                matched_control_qualified=True,
                minimum_clean_gain=0.05,
                primary_operator_ids=(
                    "A1_stream_absence",
                    "F1_global_response_drift",
                ),
                primary_severity_levels=(1, 2),
                bootstrap_resamples=200,
                bootstrap_seed=7,
            ),
        )

        by_task = {item.task: item for item in summary.tasks}
        self.assertEqual(by_task["task_a"].clean_sr, 1.0)
        self.assertEqual(by_task["task_a"].no_touch_sr, 0.5)
        self.assertEqual(by_task["task_a"].fault_sr, 0.5)
        self.assertEqual(by_task["task_a"].paired_delta_sr, 0.5)
        self.assertEqual(by_task["task_a"].tgr, 0.0)
        self.assertEqual(by_task["task_b"].fault_sr, 0.125)
        self.assertEqual(by_task["task_b"].tgr, 0.25)
        self.assertAlmostEqual(summary.macro_tgr, 0.125)
        self.assertEqual(summary.macro_paired_delta_sr, 0.4375)
        self.assertEqual(summary.paired_delta_interval.estimate, 0.4375)
        self.assertEqual(summary.worst_cell.operator_id, "A1_stream_absence")
        self.assertEqual(summary.worst_cell.severity_level, 2)
        self.assertEqual(summary.worst_cell.success_rate, 0.0)
        self.assertEqual(summary.worst_cell.paired_delta_sr, 0.75)
        self.assertIsNotNone(summary.worst_cell.paired_delta_lower)
        self.assertIsNotNone(summary.worst_cell.paired_delta_upper)
        assert summary.worst_cell.paired_delta_lower is not None
        assert summary.worst_cell.paired_delta_upper is not None
        self.assertGreaterEqual(summary.worst_cell.paired_delta_lower, 0.0)
        self.assertGreaterEqual(
            summary.worst_cell.paired_delta_upper,
            summary.worst_cell.paired_delta_lower,
        )

    def test_aggregation_reports_native_doses_and_coverage(self) -> None:
        unsupported = _outcome(
            task="task_a",
            pair="a1",
            condition=Condition.FAULTED,
            success=None,
            digest_index=150,
            operator="A1_stream_absence",
            severity=3,
        )
        summary = aggregate_benchmark(
            _fixture() + (unsupported,),
            ReportingSpec(
                system_id="touch-policy",
                matched_control_qualified=True,
                minimum_clean_gain=0.05,
                primary_operator_ids=(
                    "A1_stream_absence",
                    "F1_global_response_drift",
                ),
                primary_severity_levels=(1, 2),
                bootstrap_resamples=100,
                bootstrap_seed=7,
            ),
        )

        curve = next(
            item
            for item in summary.native_dose_curves
            if item.operator_id == "A1_stream_absence" and item.severity_level == 2
        )
        self.assertEqual(curve.native_dose, 0.10)
        self.assertEqual(curve.native_unit, "affected_window_fraction")
        self.assertEqual(curve.success_rate, 0.0)
        self.assertEqual(summary.coverage.requested_count, len(_fixture()) + 1)
        self.assertEqual(summary.coverage.ineligible_count, 1)

    def test_tgr_is_withheld_for_unqualified_or_small_gain_control(self) -> None:
        unqualified = aggregate_benchmark(
            _fixture(),
            ReportingSpec(
                system_id="touch-policy",
                matched_control_qualified=False,
                minimum_clean_gain=0.05,
                primary_operator_ids=(
                    "A1_stream_absence",
                    "F1_global_response_drift",
                ),
                primary_severity_levels=(1, 2),
                bootstrap_resamples=100,
                bootstrap_seed=7,
            ),
        )

        self.assertIsNone(unqualified.macro_tgr)
        self.assertTrue(all(item.tgr is None for item in unqualified.tasks))
        self.assertEqual(
            {item.tgr_ineligibility_reason for item in unqualified.tasks},
            {"matched_control_not_qualified"},
        )

    def test_contract_rejects_duplicate_cells_and_missing_baselines(self) -> None:
        records = _fixture()
        spec = ReportingSpec(
            system_id="touch-policy",
            matched_control_qualified=True,
            minimum_clean_gain=0.05,
            primary_operator_ids=(
                "A1_stream_absence",
                "F1_global_response_drift",
            ),
            primary_severity_levels=(1, 2),
            bootstrap_resamples=100,
            bootstrap_seed=7,
        )
        with self.assertRaisesRegex(ValueError, "duplicate reporting cell"):
            aggregate_benchmark(records + (records[0],), spec)
        with self.assertRaisesRegex(ValueError, "matched clean and no-touch"):
            aggregate_benchmark(records[1:], spec)

    def test_axis_mean_weights_operators_not_their_number_of_severity_points(
        self,
    ) -> None:
        records = (
            _outcome(
                task="task",
                pair="pair",
                condition=Condition.CLEAN,
                success=True,
                digest_index=160,
            ),
            _outcome(
                task="task",
                pair="pair",
                condition=Condition.NO_TOUCH,
                success=False,
                digest_index=161,
            ),
            _outcome(
                task="task",
                pair="pair",
                condition=Condition.FAULTED,
                success=True,
                digest_index=162,
                operator="A1_stream_absence",
                severity=1,
            ),
            _outcome(
                task="task",
                pair="pair",
                condition=Condition.FAULTED,
                success=True,
                digest_index=163,
                operator="A1_stream_absence",
                severity=2,
            ),
            _outcome(
                task="task",
                pair="pair",
                condition=Condition.FAULTED,
                success=False,
                digest_index=164,
                operator="A2_frame_erasure",
                severity=1,
            ),
        )
        summary = aggregate_benchmark(
            records,
            ReportingSpec(
                system_id="touch-policy",
                matched_control_qualified=True,
                minimum_clean_gain=0.05,
                primary_operator_ids=(
                    "A1_stream_absence",
                    "A2_frame_erasure",
                ),
                primary_severity_levels=(1,),
                bootstrap_resamples=100,
                bootstrap_seed=7,
            ),
        )

        self.assertEqual(summary.tasks[0].axis_fault_sr, (("availability", 0.5),))
        self.assertEqual(summary.tasks[0].fault_sr, 0.5)

    def test_primary_coverage_and_tgr_fail_closed_when_a_cell_is_missing(self) -> None:
        records = _fixture()
        incomplete = tuple(
            record
            for record in records
            if not (
                record.condition is Condition.FAULTED
                and record.operator_id == "F1_global_response_drift"
                and record.severity_level == 2
                and record.task == "task_b"
                and record.pair_key == hashlib.sha256(b"b2").hexdigest()
            )
        )
        summary = aggregate_benchmark(
            incomplete,
            ReportingSpec(
                system_id="touch-policy",
                matched_control_qualified=True,
                minimum_clean_gain=0.05,
                primary_operator_ids=(
                    "A1_stream_absence",
                    "F1_global_response_drift",
                ),
                primary_severity_levels=(1, 2),
                bootstrap_resamples=100,
                bootstrap_seed=7,
            ),
        )

        self.assertFalse(summary.coverage.primary_grid_complete)
        self.assertEqual(summary.coverage.expected_primary_cell_count, 16)
        self.assertEqual(summary.coverage.observed_primary_cell_count, 15)
        self.assertIsNone(summary.macro_tgr)
        task_b = next(item for item in summary.tasks if item.task == "task_b")
        self.assertIsNone(task_b.tgr)
        self.assertEqual(
            task_b.tgr_ineligibility_reason,
            "incomplete_or_ineligible_primary_grid",
        )

    def test_unavailable_no_touch_withholds_tgr_but_keeps_clean_fault_delta(
        self,
    ) -> None:
        records = (
            _outcome(
                task="task",
                pair="pair",
                condition=Condition.CLEAN,
                success=True,
                digest_index=170,
            ),
            _outcome(
                task="task",
                pair="pair",
                condition=Condition.NO_TOUCH,
                success=None,
                digest_index=171,
            ),
            _outcome(
                task="task",
                pair="pair",
                condition=Condition.FAULTED,
                success=False,
                digest_index=172,
                operator="A1_stream_absence",
                severity=1,
            ),
        )
        summary = aggregate_benchmark(
            records,
            ReportingSpec(
                system_id="touch-policy",
                matched_control_qualified=True,
                minimum_clean_gain=0.05,
                primary_operator_ids=("A1_stream_absence",),
                primary_severity_levels=(1,),
                bootstrap_resamples=100,
                bootstrap_seed=7,
            ),
        )

        self.assertEqual(summary.macro_clean_sr, 1.0)
        self.assertEqual(summary.macro_fault_sr, 0.0)
        self.assertEqual(summary.macro_paired_delta_sr, 1.0)
        self.assertEqual(summary.paired_delta_interval.estimate, 1.0)
        self.assertIsNone(summary.macro_no_touch_sr)
        self.assertIsNone(summary.macro_tgr)
        self.assertEqual(
            summary.tasks[0].tgr_ineligibility_reason,
            "matched_control_outcomes_ineligible",
        )
        self.assertIsNone(summary.operator_cells[0].no_touch_relative_delta)

    def test_ineligible_clean_pair_prevents_primary_score_completeness(self) -> None:
        records = (
            _outcome(
                task="task",
                pair="pair-a",
                condition=Condition.CLEAN,
                success=True,
                digest_index=173,
            ),
            _outcome(
                task="task",
                pair="pair-a",
                condition=Condition.NO_TOUCH,
                success=False,
                digest_index=174,
            ),
            _outcome(
                task="task",
                pair="pair-a",
                condition=Condition.FAULTED,
                success=False,
                digest_index=175,
                operator="A1_stream_absence",
                severity=1,
            ),
            _outcome(
                task="task",
                pair="pair-b",
                condition=Condition.CLEAN,
                success=None,
                digest_index=176,
            ),
            _outcome(
                task="task",
                pair="pair-b",
                condition=Condition.NO_TOUCH,
                success=False,
                digest_index=177,
            ),
            _outcome(
                task="task",
                pair="pair-b",
                condition=Condition.FAULTED,
                success=False,
                digest_index=178,
                operator="A1_stream_absence",
                severity=1,
            ),
        )
        summary = aggregate_benchmark(
            records,
            ReportingSpec(
                system_id="touch-policy",
                matched_control_qualified=True,
                minimum_clean_gain=0.05,
                primary_operator_ids=("A1_stream_absence",),
                primary_severity_levels=(1,),
                bootstrap_resamples=100,
                bootstrap_seed=7,
            ),
        )

        self.assertTrue(summary.coverage.primary_grid_complete)
        self.assertFalse(summary.coverage.primary_score_complete)
        self.assertEqual(summary.coverage.scored_primary_cell_count, 1)
        self.assertIsNone(summary.tasks[0].tgr)
        self.assertEqual(
            summary.tasks[0].tgr_ineligibility_reason,
            "incomplete_or_ineligible_primary_grid",
        )


if __name__ == "__main__":
    unittest.main()
