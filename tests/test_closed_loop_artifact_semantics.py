"""Self-consistent semantic-forgery tests for closed-loop artifact loading."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from robotactile_benchmark.closed_loop.artifacts import write_closed_loop_bundle
from robotactile_benchmark.closed_loop.capture import (
    ActionTraceEntry,
    ClosedLoopExecutionEvidence,
)
from robotactile_benchmark.closed_loop.contracts import ClosedLoopRunSpec
from robotactile_benchmark.closed_loop.delivery import DeliveryFinalization
from robotactile_benchmark.closed_loop.result_hashes import build_trial_result
from robotactile_benchmark.closed_loop.smoke import write_cpu_smoke_bundle
from robotactile_benchmark.contracts import (
    EvaluationRecord,
    build_evaluation_record,
    canonical_hash,
    delivered_hash,
)
from robotactile_benchmark.runtime import trace_hash
from robotactile_benchmark.validators import validate_delivery


def _clean_trace_hash(records: tuple[EvaluationRecord, ...]) -> str:
    return canonical_hash(
        {
            "namespace": "robotactile_benchmark.closed_loop.clean_trace.v1",
            "records": [
                delivered_hash(record.observation, record.provenance)
                for record in records
            ],
        }
    )


def _rebuild_evidence(
    loaded: object,
    *,
    run_spec: ClosedLoopRunSpec | None = None,
    finalization: DeliveryFinalization | None = None,
    action_entries: tuple[ActionTraceEntry, ...] | None = None,
) -> tuple[ClosedLoopRunSpec, ClosedLoopExecutionEvidence]:
    active_spec = loaded.run_spec if run_spec is None else run_spec
    active_finalization = loaded.finalization if finalization is None else finalization
    active_entries = loaded.action_entries if action_entries is None else action_entries
    original = loaded.result
    result = build_trial_result(
        loaded.trial,
        active_spec,
        initial_state_sha256=original.initial_state_sha256,
        terminal_status=original.terminal_status,
        execution_status=original.execution_status,
        finalization=active_finalization,
        action_entries=tuple(entry.to_hash_entry() for entry in active_entries),
        observation_count=original.observation_count,
        control_cycle_count=original.control_cycle_count,
        failure_stage=original.failure_stage,
        failure_code=original.failure_code,
    )
    return active_spec, ClosedLoopExecutionEvidence(
        result=result,
        finalization=active_finalization,
        action_entries=active_entries,
    )


class ClosedLoopArtifactSemanticForgeryTests(unittest.TestCase):
    def test_outside_window_payload_provenance_mismatch_is_rejected(self) -> None:
        """Rehashing an uncoupled clean/delivered pair must not create valid evidence."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = write_cpu_smoke_bundle(root / "valid")
            clean = list(loaded.finalization.clean_records)
            delivered = list(loaded.finalization.delivered_records)
            provenance = tuple(
                replace(item, payload_sha256="0" * 64)
                if item.slot_id == "left"
                else item
                for item in clean[0].provenance
            )
            forged = build_evaluation_record(clean[0].observation, provenance)
            clean[0] = forged
            delivered[0] = forged
            clean_records = tuple(clean)
            delivered_records = tuple(delivered)
            report = validate_delivery(
                clean_records,
                delivered_records,
                loaded.fault_manifest,
            )
            self.assertTrue(report.passed)
            finalization = DeliveryFinalization(
                clean_records=clean_records,
                delivered_records=delivered_records,
                validation=report,
                clean_trace_sha256=_clean_trace_hash(clean_records),
                delivered_trace_sha256=trace_hash(
                    delivered_records, loaded.fault_manifest
                ),
                manifest_sha256=loaded.fault_manifest.sha256,
            )
            spec, evidence = _rebuild_evidence(
                loaded,
                finalization=finalization,
            )

            with self.assertRaises(ValueError):
                write_closed_loop_bundle(
                    root / "forged",
                    trial=loaded.trial,
                    run_spec=spec,
                    fault_manifest=loaded.fault_manifest,
                    evidence=evidence,
                )

    def test_self_consistent_forged_source_step_is_rejected(self) -> None:
        """Rehashing source_step=999 must not bypass runner chronology."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = write_cpu_smoke_bundle(root / "valid")
            entries = list(loaded.action_entries)
            entries[0] = replace(entries[0], source_step_index=999)
            spec, evidence = _rebuild_evidence(
                loaded,
                action_entries=tuple(entries),
            )

            with self.assertRaises(ValueError):
                write_closed_loop_bundle(
                    root / "forged",
                    trial=loaded.trial,
                    run_spec=spec,
                    fault_manifest=loaded.fault_manifest,
                    evidence=evidence,
                )

    def test_self_consistent_results_cannot_exceed_run_spec_budgets(self) -> None:
        """Rehashing a smaller run budget must not legitimize observed counts."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = write_cpu_smoke_bundle(root / "valid")
            forged_specs = (
                replace(loaded.run_spec, max_control_cycles=4),
                replace(loaded.run_spec, max_observation_steps=5),
            )
            for index, spec in enumerate(forged_specs):
                with self.subTest(spec=spec):
                    _, evidence = _rebuild_evidence(loaded, run_spec=spec)
                    with self.assertRaises(ValueError):
                        write_closed_loop_bundle(
                            root / f"forged-{index}",
                            trial=loaded.trial,
                            run_spec=spec,
                            fault_manifest=loaded.fault_manifest,
                            evidence=evidence,
                        )


if __name__ == "__main__":
    unittest.main()
