"""N0 EE8 temporal clocks survive same-snapshot paired resets."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.qualification_fakes import (
    FakeUpstreamScenario,
    make_fake_runtime,
)
from robotactile_benchmark.backends.univtac_contracts import (
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_conversion import UniVTACConversionError
from robotactile_benchmark.backends.univtac_pairing import UniVTACPairedBackendSession
from robotactile_benchmark.closed_loop.contracts import PolicyEpisodeContext
from robotactile_benchmark.contracts import build_evaluation_record
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.streaming import StreamingFaultSession


@pytest.mark.parametrize("actual_ticks", [12, 2])
def test_n0_paired_temporal_clock_and_physical_cadence(actual_ticks: int) -> None:
    config = build_univtac_backend_config(
        "pull_out_key",
        EE8_ACTION_SPEC,
        n0_action_execution_contract=N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    )
    runtime, task = make_fake_runtime(
        config, scenario=FakeUpstreamScenario(native_step_increment=actual_ticks)
    )
    native: Any = task
    native._robotactile_n0_fixed_cadence_enabled = True
    native._robotactile_n0_action_execution_contract = config.action_execution_contract
    session = UniVTACPairedBackendSession(config, runtime)
    action = np.asarray([[0.1, 0, 0, 1, 0, 0, 0, 0.02]], dtype=np.float32)
    operators = (
        None,
        "T1_fixed_source_delay",
        "T2_held_last_freeze",
        "T3_inter_sensor_skew",
    )
    try:
        for index, operator in enumerate(operators):
            backend = session.new_backend()
            backend.reset(
                PolicyEpisodeContext(
                    episode_id=f"paired-{index}",
                    task="pull_out_key",
                    initial_seed=11,
                    exogenous_seed=999,
                    instruction=config.task.prompt,
                    action_spec=EE8_ACTION_SPEC,
                )
            )
            initial = backend.observe()
            assert initial.provenance_for("left").source_time_s == 0
            stream = None
            if operator is not None:
                stream = StreamingFaultSession(
                    FaultManifest(
                        operator_id=operator,
                        severity_level=5,
                        operator_seed=0,
                        start_index=20,
                        stop_index=50,
                        sensor_slots=("left", "right"),
                        observability=Observability.BLIND,
                        parameters={"sample_period_s": 0.1},
                        severity_registry="optical_marker_v1",
                    )
                )
                stream.deliver_one(initial)
            if actual_ticks != 12:
                with pytest.raises(UniVTACConversionError, match="native step"):
                    backend.execute(action)
                backend.close()
                break
            for step in range(1, 24):
                record = backend.execute(action).transitions[0].clean_record
                for slot in ("left", "right"):
                    assert record.provenance_for(slot).source_time_s == pytest.approx(
                        step * 0.1
                    )
                    assert record.observation.sensor(
                        slot
                    ).delivery_time_s == pytest.approx(step * 0.1)
                if stream is not None:
                    stream.deliver_one(record)
            if stream is not None:
                # The clock validator must still reject a genuine cadence jump.
                record = backend.execute(action).transitions[0].clean_record
                malformed = build_evaluation_record(
                    record.observation,
                    tuple(
                        replace(item, source_time_s=2.35) for item in record.provenance
                    ),
                )
                with pytest.raises(ValueError, match="source cadence disagrees"):
                    stream.deliver_one(malformed)
            backend.close()
        if actual_ticks == 12:
            assert task.reset_count == 1
            assert task.restore_count == 3
            assert session.reset_receipt.all_exact
    finally:
        session.close()
