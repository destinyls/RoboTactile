from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from robotactile_benchmark.reporting import (
    OutcomeRecord,
    ReportingSpec,
    aggregate_benchmark,
    load_report_bundle,
    write_report_bundle,
)
from robotactile_benchmark.trials import Condition, TerminalStatus


def _record(
    condition: Condition,
    success: bool | None,
    digest: str,
    operator: str | None = None,
    severity: int | None = None,
) -> OutcomeRecord:
    eligible = success is not None
    return OutcomeRecord(
        system_id="system",
        task="task",
        pair_key=hashlib.sha256(b"pair").hexdigest(),
        condition=condition,
        terminal_status=(
            (TerminalStatus.SUCCESS if success else TerminalStatus.TASK_FAILURE)
            if eligible
            else TerminalStatus.UNSUPPORTED_CONTRACT
        ),
        score_eligible=eligible,
        score_success=success,
        source_root_sha256=digest,
        operator_id=operator,
        severity_level=severity,
    )


class ReportingExportTests(unittest.TestCase):
    def test_report_bundle_is_deterministic_loadable_and_source_bound(self) -> None:
        records = (
            _record(Condition.CLEAN, True, "1" * 64),
            _record(Condition.NO_TOUCH, False, "2" * 64),
            _record(
                Condition.FAULTED,
                False,
                "3" * 64,
                "A1_stream_absence",
                1,
            ),
        )
        spec = ReportingSpec(
            system_id="system",
            matched_control_qualified=True,
            minimum_clean_gain=0.05,
            primary_operator_ids=("A1_stream_absence",),
            primary_severity_levels=(1,),
            bootstrap_resamples=100,
            bootstrap_seed=5,
        )
        summary = aggregate_benchmark(records, spec)
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            first_receipt = write_report_bundle(Path(first), summary)
            second_receipt = write_report_bundle(Path(second), summary)
            loaded = load_report_bundle(Path(first))
            first_files = {
                path.relative_to(first).as_posix(): path.read_bytes()
                for path in Path(first).rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second).as_posix(): path.read_bytes()
                for path in Path(second).rglob("*")
                if path.is_file()
            }

        self.assertEqual(first_files, second_files)
        self.assertEqual(first_receipt.sha256, second_receipt.sha256)
        self.assertEqual(loaded.summary.sha256, summary.sha256)
        self.assertEqual(
            loaded.receipt.source_root_sha256,
            ("1" * 64, "2" * 64, "3" * 64),
        )
        self.assertEqual(
            set(first_files),
            {
                "native_dose_curves.csv",
                "native_dose_curves.svg",
                "operator_cells.csv",
                "per_task.csv",
                "report_receipt.json",
                "summary.json",
                "summary_table.tex",
            },
        )
        self.assertEqual(
            hashlib.sha256(first_files["report_receipt.json"]).hexdigest(),
            loaded.receipt_file_sha256,
        )

    def test_writer_never_overwrites_unrelated_nonempty_directory(self) -> None:
        summary = aggregate_benchmark(
            (
                _record(Condition.CLEAN, True, "1" * 64),
                _record(Condition.NO_TOUCH, False, "2" * 64),
                _record(
                    Condition.FAULTED,
                    False,
                    "3" * 64,
                    "A1_stream_absence",
                    1,
                ),
            ),
            ReportingSpec(
                system_id="system",
                matched_control_qualified=True,
                minimum_clean_gain=0.05,
                primary_operator_ids=("A1_stream_absence",),
                primary_severity_levels=(1,),
                bootstrap_resamples=100,
                bootstrap_seed=5,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            output.mkdir()
            sentinel = output / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                write_report_bundle(output, summary)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_report_bundle_marks_unavailable_no_touch_metrics_as_na(self) -> None:
        summary = aggregate_benchmark(
            (
                _record(Condition.CLEAN, True, "1" * 64),
                _record(Condition.NO_TOUCH, None, "2" * 64),
                _record(
                    Condition.FAULTED,
                    False,
                    "3" * 64,
                    "A1_stream_absence",
                    1,
                ),
            ),
            ReportingSpec(
                system_id="system",
                matched_control_qualified=True,
                minimum_clean_gain=0.05,
                primary_operator_ids=("A1_stream_absence",),
                primary_severity_levels=(1,),
                bootstrap_resamples=100,
                bootstrap_seed=5,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            receipt = write_report_bundle(Path(temporary), summary)
            loaded = load_report_bundle(Path(temporary))
            latex = (Path(temporary) / "summary_table.tex").read_text(encoding="utf-8")

        self.assertEqual(receipt.summary_sha256, summary.sha256)
        self.assertIsNone(loaded.summary.macro_no_touch_sr)
        self.assertIn("task & 1.000 & N/A & 0.000", latex)
        self.assertIn("Macro & 1.000 & N/A & 0.000", latex)


if __name__ == "__main__":
    unittest.main()
