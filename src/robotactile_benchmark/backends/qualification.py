"""Typed CPU qualification for the source-complete UniVTAC backend."""

from __future__ import annotations

import platform

import numpy as np

from robotactile_benchmark.backends.qualification_checks import (
    qualification_action,
    qualification_context,
    run_failure_gate_checks,
    run_terminal_priority_checks,
)
from robotactile_benchmark.backends.qualification_fakes import make_fake_runtime
from robotactile_benchmark.backends.qualification_io import (
    qualification_receipt_bytes,
    write_qualification_receipt,
)
from robotactile_benchmark.backends.qualification_receipt import (
    CPU_FAKE_EVIDENCE_LEVEL,
    QUALIFICATION_CHECKS,
    QUALIFICATION_FAILURE_CODES,
    QUALIFICATION_SCHEMA_VERSION,
    UniVTACQualificationError,
    UniVTACQualificationReceipt,
    action_progress_witness,
    qualification_descriptor,
)
from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend
from robotactile_benchmark.contracts import canonical_hash

__all__ = [
    "CPU_FAKE_EVIDENCE_LEVEL",
    "QUALIFICATION_CHECKS",
    "QUALIFICATION_FAILURE_CODES",
    "QUALIFICATION_SCHEMA_VERSION",
    "UniVTACQualificationError",
    "UniVTACQualificationReceipt",
    "qualification_receipt_bytes",
    "run_cpu_fake_qualification",
    "write_qualification_receipt",
]


def run_cpu_fake_qualification(
    task_id: str = "pull_out_key",
) -> UniVTACQualificationReceipt:
    """Run deterministic upstream-faithful CPU checks without Isaac/CUDA claims."""

    config = build_univtac_backend_config(task_id)
    first_runtime, first_task = make_fake_runtime(config)
    second_runtime, second_task = make_fake_runtime(config)
    first = UniVTACIsaacBackend(config, first_runtime)
    second = UniVTACIsaacBackend(config, second_runtime)
    context = qualification_context(config)
    try:
        first_reset = first.reset(context)
        first_record = first.observe()
        second_reset = second.reset(context)
        second_record = second.observe()
        first_initial_native = first.initial_native_step_id
        second_initial_native = second.initial_native_step_id
        first_joint_witness = first.joint_reorder_witness_sha256
        second_joint_witness = second.joint_reorder_witness_sha256
        initial_joint9 = first.initial_canonical_joint9
        initial_qpos8 = first.initial_model_visible_qpos8
        action = qualification_action()
        transition = first.execute(action).transitions[0]
        action_state = first.latest_state_sha256
        if (
            first_initial_native is None
            or second_initial_native is None
            or first_joint_witness is None
            or second_joint_witness is None
            or initial_joint9 is None
            or initial_qpos8 is None
            or action_state is None
        ):
            raise UniVTACQualificationError("backend omitted required witnesses")
        if first_reset.simulator_state_sha256 != second_reset.simulator_state_sha256:
            raise UniVTACQualificationError("fresh reset state hashes differ")
        if first_record.clean_record_sha256 != second_record.clean_record_sha256:
            raise UniVTACQualificationError("fresh reset record hashes differ")
        if first_record.observation.proprio.shape != (8,):
            raise UniVTACQualificationError("joint9 did not fold to shared qpos8")
        if first_reset.simulator_state_sha256 == first_task.stale_return_sha256:
            raise UniVTACQualificationError("stale reset return was used")
        exact_sequence = ("detach", "cpu", "contiguous", "numpy")
        if not first_task.cuda_conversion_sequences or not all(
            sequence == exact_sequence
            for sequence in first_task.cuda_conversion_sequences
            + second_task.cuda_conversion_sequences
        ):
            raise UniVTACQualificationError("CUDA-like conversion order mismatch")
    finally:
        first.close()
        first.close()
        second.close()
    if first_task.close_count != 1 or second_task.close_count != 1:
        raise UniVTACQualificationError("pair cleanup was not exactly once")
    terminal_results = run_terminal_priority_checks(config)
    run_failure_gate_checks(config)
    action_sha256 = canonical_hash(action)
    action_witness = action_progress_witness(
        first_reset.simulator_state_sha256,
        action_state,
        first_initial_native,
        transition.native_step_id,
        transition.clean_record.observation.step_index,
        action_sha256,
    )
    environment = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "accelerator": "none",
        "isaac_sim_executed": False,
        "runtime_kind": "upstream_faithful_cpu_fake",
    }
    return UniVTACQualificationReceipt(
        task_id=config.task.task_id,
        success_predicate_id=config.task.success_predicate_id,
        upstream_commit=config.upstream_commit,
        task_source_sha256=config.task.task_source_sha256,
        registry_resource_sha256=config.registry_resource_sha256,
        config_sha256=config.sha256,
        config_descriptor=qualification_descriptor(config),
        handshake_sha256=first.handshake.sha256,
        handshake_descriptor=qualification_descriptor(first.handshake),
        first_reset_state_sha256=first_reset.simulator_state_sha256,
        second_reset_state_sha256=second_reset.simulator_state_sha256,
        first_clean_record_sha256=first_record.clean_record_sha256,
        second_clean_record_sha256=second_record.clean_record_sha256,
        first_initial_native_step_id=first_initial_native,
        second_initial_native_step_id=second_initial_native,
        first_joint_reorder_witness_sha256=first_joint_witness,
        second_joint_reorder_witness_sha256=second_joint_witness,
        live_joint_names=first.handshake.live_joint_names,
        canonical_joint_names=config.canonical_joint_names,
        initial_canonical_joint9=tuple(float(value) for value in initial_joint9),
        initial_model_visible_qpos8=tuple(float(value) for value in initial_qpos8),
        action_state_sha256=action_state,
        action_native_step_id=transition.native_step_id,
        action_benchmark_step=transition.clean_record.observation.step_index,
        action_sha256=action_sha256,
        action_progress_witness_sha256=action_witness,
        terminal_priority_results=terminal_results,
        terminal_priority_witness_sha256=canonical_hash(terminal_results),
        predicate_note=config.task.predicate_note,
        environment=environment,
        checks=QUALIFICATION_CHECKS,
        passed=True,
        failure_codes=(),
    )
