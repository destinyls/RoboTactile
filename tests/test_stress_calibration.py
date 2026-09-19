"""Development-only contact template calibration, including task failures."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robotactile_benchmark.contracts import ContactPhase, canonical_hash
from robotactile_benchmark.n0_fault_campaign import stress_calibration as module
from robotactile_benchmark.trials import Condition


def record(phase, index, pixel):
    return SimpleNamespace(
        provenance_for=lambda slot: SimpleNamespace(phase=phase),
        observation=SimpleNamespace(
            step_index=index,
            sensor=lambda slot: SimpleNamespace(
                payload=np.full((12, 12, 3), pixel, dtype=np.uint8)
            ),
        ),
    )


def source(monkeypatch, *, condition=Condition.CLEAN, contact=True):
    terminal = dict(
        score_eligible=True,
        validation_passed=True,
        terminal_status="task_failure",
        score_success=False,
        observation_count=2,
    )
    records = [record(ContactPhase.FREE, 0, 0)]
    if contact:
        records.append(record(ContactPhase.SUSTAINED, 1, 128))
    loaded = SimpleNamespace(
        trial=SimpleNamespace(condition=condition, task="lift_bottle", initial_seed=5),
        evidence=SimpleNamespace(
            result=terminal, finalization=SimpleNamespace(clean_records=records)
        ),
        external_root_sha256="a" * 64,
        capture_profile=SimpleNamespace(value="paper_full_v1"),
    )
    monkeypatch.setattr(module, "load_live_univtac_artifact", lambda path: loaded)
    monkeypatch.setattr(module, "result_to_dict", lambda value: value)


def test_task_failure_is_retained_and_template_binds_source(monkeypatch):
    source(monkeypatch)
    template, receipt = module.calibrate_clean_artifacts(
        (Path("clean"),), allowed_seeds=(5,)
    )
    assert receipt["contact_frame_counts"] == {"left": 1, "right": 1}
    assert template["source_sha256"] == canonical_hash(receipt)


@pytest.mark.parametrize("mode", ["fault", "unknown_seed", "no_contact"])
def test_ineligible_calibration_sources_are_rejected(monkeypatch, mode):
    source(
        monkeypatch,
        condition=Condition.FAULTED if mode == "fault" else Condition.CLEAN,
        contact=mode != "no_contact",
    )
    with pytest.raises(ValueError):
        module.calibrate_clean_artifacts(
            (Path("clean"),), allowed_seeds=(8,) if mode == "unknown_seed" else (5,)
        )
