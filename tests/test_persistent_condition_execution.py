"""Contracts for task/condition-sharded persistent N0 + Isaac execution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from scripts.retrained_evaluation import persistent_condition as module

SHA_A = "a" * 64
SHA_B = "b" * 64


def _n0_artifact() -> dict[str, object]:
    return {
        "action_hz": 10,
        "action_per_frame": 4,
        "artifact_sha256": SHA_A,
        "checkpoint_sha256": SHA_B,
        "source_tree_sha256": "c" * 64,
        "tasks": {
            "lift_can": {
                "config_sha256": "d" * 64,
                "normalizer_sha256": "e" * 64,
            }
        },
    }


def _work_item(sequence: int, seed: int) -> module.ConditionWorkItem:
    return module.ConditionWorkItem(
        sequence=sequence,
        seed=seed,
        group_relpath=f"seeds/seed-{seed:06d}/groups/n0_twam/lift_can",
        request_relpath="requests/clean.json",
        request_index=0,
        request_file_sha256=SHA_A,
        trial_manifest_sha256=SHA_B,
    )


def _worker_preclose_document(
    manifest: module.ConditionWorkManifest,
) -> dict[str, object]:
    session_id = "persistent-condition-test"
    dispatch: dict[str, object] = {
        "condition_id": manifest.condition_id,
        "policy_reset_completed": True,
        "schema": module.EPISODE_SCHEMA,
        "session_id": session_id,
    }
    dispatch["content_sha256"] = module.canonical_hash(dispatch)
    receipt: dict[str, object] = {
        "app_close_status": "pending",
        "condition_id": manifest.condition_id,
        "dispatches": [dispatch],
        "isaac_application_launch_count": 1,
        "model_identity_sha256": manifest.model_identity_sha256,
        "n0_endpoint_reset_before_observation": True,
        "policy_instance_per_episode": True,
        "receipt_stage": "preclose",
        "runtime_count": len(manifest.items),
        "schema": module.SESSION_SCHEMA,
        "session_id": session_id,
        "task_id": manifest.task_id,
        "task_runtime_per_episode": True,
        "total_duration_s": 10.0,
        "work_manifest_sha256": manifest.content_sha256,
    }
    receipt["content_sha256"] = module.canonical_hash(receipt)
    return receipt


def _aggregate_campaign(
    root: Path,
    condition: str,
    successes: tuple[bool, bool],
    *,
    mismatch_second_seed: bool = False,
) -> Path:
    root.mkdir()
    items = tuple(
        module.ConditionWorkItem(
            sequence=sequence,
            seed=seed,
            group_relpath=f"seeds/seed-{seed:06d}/groups/n0_twam/lift_can",
            request_relpath=(
                "requests/clean.json"
                if condition == "clean"
                else f"requests/optical_marker_extreme_v1/{condition}.json"
            ),
            request_index=0,
            request_file_sha256=SHA_A,
            trial_manifest_sha256=SHA_B,
        )
        for sequence, seed in enumerate((3, 8))
    )
    manifest = module.ConditionWorkManifest.build(
        task_id="lift_can",
        condition_id=condition,
        capture_profile=LiveCaptureProfile.METRICS_ONLY.value,
        evidence_mode="diagnostic",
        binding_relpath="binding.json",
        binding_file_sha256=SHA_A,
        model_identity_sha256=SHA_A,
        reset_equivalence_receipt=None,
        reset_equivalence_receipt_sha256=None,
        items=items,
    )
    module.write_json(root / "work_manifest.json", manifest.to_dict())
    session: dict[str, object] = {
        "condition_id": condition,
        "isaac_application_launch_count": 1,
        "model_identity_sha256": manifest.model_identity_sha256,
        "n0_endpoint_reset_before_observation": True,
        "runtime_count": 2,
        "schema": module.SESSION_SCHEMA,
        "task_id": "lift_can",
        "work_manifest_sha256": manifest.content_sha256,
    }
    session["content_sha256"] = module.canonical_hash(session)
    module.write_json(root / "persistent_condition_session.json", session)
    for item, success in zip(items, successes):
        initial = (
            "c" * 64
            if item.seed == 3
            else (
                "d" * 64 if mismatch_second_seed and condition != "clean" else "e" * 64
            )
        )
        module.write_json(
            root / item.group_relpath / "results" / f"{item.request_index:02d}.json",
            {
                "condition": condition,
                "failure_stage": None,
                "initial_state_sha256": initial,
                "score_eligible": True,
                "score_success": success,
                "seed": item.seed,
                "task": "lift_can",
                "validation_passed": True,
            },
        )
    return root


def test_work_manifest_roundtrip_and_evidence_gates() -> None:
    manifest = module.ConditionWorkManifest.build(
        task_id="lift_can",
        condition_id="clean",
        capture_profile=LiveCaptureProfile.METRICS_ONLY.value,
        evidence_mode="diagnostic",
        binding_relpath="binding.json",
        binding_file_sha256=SHA_A,
        model_identity_sha256=SHA_A,
        reset_equivalence_receipt=None,
        reset_equivalence_receipt_sha256=None,
        items=(_work_item(0, 4), _work_item(1, 9)),
    )
    assert module.ConditionWorkManifest.from_dict(manifest.to_dict()) == manifest
    assert [item.seed for item in manifest.items] == [4, 9]

    with pytest.raises(ValueError, match="requires reset-equivalence"):
        module.ConditionWorkManifest.build(
            task_id="lift_can",
            condition_id="clean",
            capture_profile=LiveCaptureProfile.METRICS_ONLY.value,
            evidence_mode="statistics",
            binding_relpath="binding.json",
            binding_file_sha256=SHA_A,
            model_identity_sha256=SHA_A,
            reset_equivalence_receipt=None,
            reset_equivalence_receipt_sha256=None,
            items=(_work_item(0, 4),),
        )
    with pytest.raises(ValueError, match="paper_full"):
        module.ConditionWorkManifest.build(
            task_id="lift_can",
            condition_id="clean",
            capture_profile=LiveCaptureProfile.METRICS_ONLY.value,
            evidence_mode="claim",
            binding_relpath="binding.json",
            binding_file_sha256=SHA_A,
            model_identity_sha256=SHA_A,
            reset_equivalence_receipt="/proof/reset.json",
            reset_equivalence_receipt_sha256=SHA_B,
            items=(_work_item(0, 4),),
        )


def test_prepare_condition_work_freezes_seed_bound_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _n0_artifact()
    binding_path = tmp_path / "source-binding.json"
    module.write_json(
        binding_path,
        {
            "artifact": str(tmp_path / "artifact.json"),
            "binding_sha256": SHA_A,
            "evaluation": {},
            "model": "n0_twam",
            "tasks": {"lift_can": {"prompt": "lift the can"}},
        },
    )
    proof = tmp_path / "lift_can-reset.json"
    proof_document = {
        "app_close_status": "returned",
        "evidence_level": "live_univtac_same_app_fresh_runtime_probe_v1",
        "runtime_count": 2,
        "same_process_confirmed": True,
        "status": "passed",
        "task_id": "lift_can",
    }
    proof_document["content_sha256"] = module.canonical_hash(proof_document)
    module.write_json(proof, proof_document)
    seen: list[tuple[int, dict[str, Any]]] = []

    def prepare(
        binding: dict[str, Any],
        task: str,
        group: Path,
        seed: int,
        *,
        n0_artifact: dict[str, Any] | None = None,
    ) -> Path:
        assert n0_artifact == artifact
        seen.append((seed, binding["evaluation"]))
        assert task == "lift_can"
        relative = "requests/optical_marker_extreme_v1/T1_fixed_source_delay.json"
        module.write_json(group / "group.json", {"ordered_requests": [relative]})
        module.write_json(group / relative, {"seed": seed})
        return group / "group.json"

    def load_request(path: Path) -> SimpleNamespace:
        seed = int(
            next(part for part in path.parts if part.startswith("seed-")).split("-")[1]
        )
        return SimpleNamespace(seed=seed)

    monkeypatch.setattr(module, "prepare_group", prepare)
    monkeypatch.setattr(module, "load_retrained", lambda _path: artifact)
    monkeypatch.setattr(module, "load_live_univtac_request", load_request)
    monkeypatch.setattr(
        module,
        "load_live_univtac_run",
        lambda request: SimpleNamespace(
            trial=SimpleNamespace(
                initial_seed=request.seed, sha256=SHA_B, task="lift_can"
            )
        ),
    )
    campaign = tmp_path / "campaign"
    path = module.prepare_condition_work(
        binding_path=binding_path,
        task_id="lift_can",
        condition_id="T1_fixed_source_delay",
        campaign=campaign,
        seeds=(3, 11),
        capture_profile=LiveCaptureProfile.METRICS_ONLY,
        evidence_mode="statistics",
        severity_registry="optical_marker_extreme_v1",
        severity_level=5,
        fault_window_mode="early_random_onset_v1",
        fault_onset_max_index=8,
        reset_equivalence_receipt=proof,
        port_override=24567,
    )
    manifest = module.load_work_manifest(path)
    assert [item.seed for item in manifest.items] == [3, 11]
    assert manifest.condition_id == "T1_fixed_source_delay"
    assert manifest.capture_profile == "metrics_only_v1"
    assert manifest.model_identity_sha256 == module._model_identity_sha256(
        module.read_object(campaign / "binding.json"), artifact, "lift_can"
    )
    assert len(seen) == 2
    assert all(
        evaluation["operators"] == ["T1_fixed_source_delay"]
        and evaluation["fault_window_mode"] == "early_random_onset_v1"
        and evaluation["fault_onset_max_index"] == 8
        for _, evaluation in seen
    )
    frozen = module.read_object(campaign / "binding.json")
    assert frozen["port"] == 24567
    assert frozen["server_receipt"].endswith(
        "campaign/server/worker/server_receipt.json"
    )
    assert frozen["binding_sha256"] != SHA_A


def test_statistics_without_reset_proof_fails_before_campaign_creation(
    tmp_path: Path,
) -> None:
    binding_path = tmp_path / "binding.json"
    module.write_json(
        binding_path,
        {"model": "n0_twam", "tasks": {"lift_can": {}}},
    )
    campaign = tmp_path / "campaign"
    with pytest.raises(ValueError, match="requires --reset-equivalence"):
        module.prepare_condition_work(
            binding_path=binding_path,
            task_id="lift_can",
            condition_id="clean",
            campaign=campaign,
            seeds=(0,),
            capture_profile=LiveCaptureProfile.METRICS_ONLY,
            evidence_mode="statistics",
            severity_registry="optical_marker_extreme_v1",
            severity_level=5,
            fault_window_mode="early_random_onset_v1",
            fault_onset_max_index=8,
            reset_equivalence_receipt=None,
        )
    assert not campaign.exists()


def test_aggregate_pairs_conditions_by_seed_and_initial_state(tmp_path: Path) -> None:
    clean = _aggregate_campaign(tmp_path / "clean", "clean", (True, True))
    fault = _aggregate_campaign(
        tmp_path / "fault", "F1_global_response_drift", (False, True)
    )
    result = module.aggregate_condition_shards(
        (clean, fault), tmp_path / "aggregate.json", require_all_conditions=False
    )
    assert result["aggregate_valid"] is True
    assert result["all_episode_initial_states_matched"] is True
    effect = result["paired_effects"][0]  # type: ignore[index]
    assert effect["paired_episode_count"] == 2
    assert effect["fault_induced_failure_count"] == 1
    assert effect["paired_success_rate_drop"] == 0.5


def test_aggregate_rejects_initial_state_mismatch_from_metric_claim(
    tmp_path: Path,
) -> None:
    clean = _aggregate_campaign(tmp_path / "clean", "clean", (True, True))
    fault = _aggregate_campaign(
        tmp_path / "fault",
        "F1_global_response_drift",
        (False, True),
        mismatch_second_seed=True,
    )
    result = module.aggregate_condition_shards(
        (clean, fault), tmp_path / "aggregate.json", require_all_conditions=False
    )
    assert result["aggregate_valid"] is False
    assert result["all_episode_initial_states_matched"] is False
    effect = result["paired_effects"][0]  # type: ignore[index]
    assert effect["initial_state_mismatch_count"] == 1
    assert effect["initial_state_mismatch_seeds"] == [8]


def test_same_task_session_reuses_host_and_rebuilds_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from robotactile_benchmark.execution import same_task_worker_runtime as runtime

    class FakeRequest:
        def __init__(self) -> None:
            self.task_id = "lift_can"
            self.runtime_dir = tmp_path / "runtime"
            self.upstream_root = tmp_path / "UniVTAC"
            self.launcher_args = {"headless": True}
            self.simulator_device = "cuda:0"

    request = FakeRequest()
    loaded = SimpleNamespace(
        request=request,
        backend_config=object(),
        trial=SimpleNamespace(initial_seed=3),
        run_spec=SimpleNamespace(success_predicate_id="lift_can_height_v1"),
    )

    class FakeHost:
        closed = False

        def __init__(self) -> None:
            self.created: list[object] = []
            self.close_count = 0

        @property
        def runtime_count(self) -> int:
            return len(self.created)

        def create_runtime(self, **_kwargs: object) -> object:
            selected = SimpleNamespace(close_runtime=lambda: None)
            self.created.append(selected)
            return selected

        def close(self) -> None:
            self.closed = True
            self.close_count += 1

    host = FakeHost()
    session = runtime.SameTaskIsaacSession.__new__(runtime.SameTaskIsaacSession)
    session._bootstrap = loaded
    session._action_execution_contract = "robotactile_n0_retrained_10hz_ee_v1"
    session._host = host
    monkeypatch.setattr(runtime, "LiveUniVTACRunRequest", FakeRequest)
    monkeypatch.setattr(runtime, "load_live_univtac_run", lambda _request: loaded)
    monkeypatch.setattr(
        runtime,
        "UniVTACIsaacBackend",
        lambda _config, task_runtime, **_kwargs: task_runtime,
    )

    def execute(_request: object, **kwargs: object) -> object:
        backend_factory = kwargs["backend_factory"]
        assert callable(backend_factory)
        backend_factory(loaded)
        return SimpleNamespace()

    monkeypatch.setattr(runtime, "execute_live_univtac_run", execute)
    for _ in range(2):
        session.execute(
            request,  # type: ignore[arg-type]
            policy_factory=lambda _loaded: object(),  # type: ignore[arg-type]
            artifact_exporter=lambda *_args: object(),  # type: ignore[arg-type]
        )
    assert session.runtime_count == 2
    assert host.created[0] is not host.created[1]
    session.close()
    session.close()
    assert host.close_count == 1


def test_execute_condition_work_uses_one_session_for_all_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    artifact = _n0_artifact()
    binding_path = campaign / "binding.json"
    server_receipt = campaign / "server_receipt.json"
    module.write_json(server_receipt, {"ready": True})
    module.write_json(
        binding_path,
        {
            "artifact": str(campaign / "artifact.json"),
            "model": "n0_twam",
            "port": 24567,
            "server_receipt": str(server_receipt),
        },
    )
    binding = module.read_object(binding_path)
    requests: dict[Path, SimpleNamespace] = {}
    items: list[module.ConditionWorkItem] = []
    for sequence, seed in enumerate((2, 7)):
        group = campaign / "seeds" / f"seed-{seed:06d}" / "groups/n0_twam/lift_can"
        request_path = group / "requests/clean.json"
        module.write_json(
            group / "group.json", {"ordered_requests": ["requests/clean.json"]}
        )
        module.write_json(request_path, {"seed": seed})
        request = SimpleNamespace(output_dir=group / "artifacts/clean")
        requests[request_path] = request
        items.append(
            module.ConditionWorkItem(
                sequence=sequence,
                seed=seed,
                group_relpath=group.relative_to(campaign).as_posix(),
                request_relpath="requests/clean.json",
                request_index=0,
                request_file_sha256=module._sha256_file(request_path),
                trial_manifest_sha256=SHA_B,
            )
        )
    manifest = module.ConditionWorkManifest.build(
        task_id="lift_can",
        condition_id="clean",
        capture_profile=LiveCaptureProfile.METRICS_ONLY.value,
        evidence_mode="diagnostic",
        binding_relpath="binding.json",
        binding_file_sha256=module._sha256_file(binding_path),
        model_identity_sha256=module._model_identity_sha256(
            binding, artifact, "lift_can"
        ),
        reset_equivalence_receipt=None,
        reset_equivalence_receipt_sha256=None,
        items=tuple(items),
    )
    work_path = campaign / "work_manifest.json"
    module.write_json(work_path, manifest.to_dict())

    def load_request(path: Path) -> SimpleNamespace:
        return requests[Path(path)]

    def load_run(request: SimpleNamespace) -> SimpleNamespace:
        seed = next(
            item.seed
            for path, selected in requests.items()
            if selected is request
            for item in items
            if f"seed-{item.seed:06d}" in path.parts
        )
        return SimpleNamespace(
            content_sha256=SHA_A,
            request=request,
            trial=SimpleNamespace(initial_seed=seed, sha256=SHA_B, task="lift_can"),
        )

    result = SimpleNamespace(
        initial_state_sha256=SHA_A,
        observation_count=301,
        score_eligible=True,
        score_success=True,
        validation_passed=True,
        failure_stage=None,
    )
    artifacts: dict[Path, SimpleNamespace] = {}
    sessions: list[Any] = []

    class FakeSession:
        def __init__(self, **_kwargs: object) -> None:
            self.count = 0
            self.closed = False
            sessions.append(self)

        @property
        def runtime_count(self) -> int:
            return self.count

        def execute(self, request: SimpleNamespace, **_kwargs: object) -> object:
            self.count += 1
            request.output_dir.mkdir(parents=True)
            artifact = SimpleNamespace(
                capture_profile=LiveCaptureProfile.METRICS_ONLY,
                evidence=SimpleNamespace(
                    initial_diagnostics={
                        "reset_viable": True,
                        "reset_witness": {"reset_viable": True},
                    },
                    result=result,
                ),
                root_receipt_sha256=SHA_A,
                run_content_sha256=SHA_A,
            )
            artifacts[request.output_dir] = artifact
            return SimpleNamespace(
                evidence=SimpleNamespace(result=result),
                loaded=SimpleNamespace(content_sha256=SHA_A),
            )

        def close(self) -> None:
            assert (campaign / "persistent_condition_session.preclose.json").is_file()
            assert not (campaign / "persistent_condition_session.json").exists()
            self.closed = True

    monkeypatch.setattr(module, "load_live_univtac_request", load_request)
    monkeypatch.setattr(module, "load_live_univtac_run", load_run)
    monkeypatch.setattr(module, "load_retrained", lambda _path: artifact)
    monkeypatch.setattr(module, "SameTaskIsaacSession", FakeSession)
    monkeypatch.setattr(
        module, "load_live_univtac_artifact", lambda path: artifacts[Path(path)]
    )
    monkeypatch.setattr(
        module,
        "result_to_dict",
        lambda _result: {
            "failure_stage": None,
            "score_eligible": True,
            "score_success": True,
            "validation_passed": True,
        },
    )

    receipt = module.execute_condition_work(campaign, work_path)
    assert len(sessions) == 1
    assert sessions[0].closed is True
    assert receipt["isaac_application_launch_count"] == 1
    assert receipt["runtime_count"] == 2
    assert receipt["task_runtime_per_episode"] is True
    assert receipt["n0_endpoint_reset_before_observation"] is True
    assert len(receipt["dispatches"]) == 2  # type: ignore[arg-type]
    assert receipt["receipt_stage"] == "final"
    for item in items:
        group = campaign / item.group_relpath
        assert (group / "results/00.json").is_file()
        assert (group / "persistent_condition_episode.json").is_file()
    assert (campaign / "persistent_condition_session.json").is_file()


def test_existing_result_is_rejected_before_session_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    group = campaign / "seeds/seed-000001/groups/n0_twam/lift_can"
    request_path = group / "requests/clean.json"
    binding_path = campaign / "binding.json"
    campaign.mkdir()
    module.write_json(binding_path, {"model": "n0_twam"})
    module.write_json(
        group / "group.json", {"ordered_requests": ["requests/clean.json"]}
    )
    module.write_json(request_path, {})
    module.write_json(group / "results/00.json", {"already": "complete"})
    item = module.ConditionWorkItem(
        sequence=0,
        seed=1,
        group_relpath=group.relative_to(campaign).as_posix(),
        request_relpath="requests/clean.json",
        request_index=0,
        request_file_sha256=module._sha256_file(request_path),
        trial_manifest_sha256=SHA_B,
    )
    manifest = module.ConditionWorkManifest.build(
        task_id="lift_can",
        condition_id="clean",
        capture_profile=LiveCaptureProfile.METRICS_ONLY.value,
        evidence_mode="diagnostic",
        binding_relpath="binding.json",
        binding_file_sha256=module._sha256_file(binding_path),
        model_identity_sha256=SHA_A,
        reset_equivalence_receipt=None,
        reset_equivalence_receipt_sha256=None,
        items=(item,),
    )
    work_path = campaign / "work_manifest.json"
    module.write_json(work_path, manifest.to_dict())
    request = SimpleNamespace(output_dir=group / "artifacts/clean")
    monkeypatch.setattr(module, "load_live_univtac_request", lambda _path: request)
    monkeypatch.setattr(
        module,
        "load_live_univtac_run",
        lambda _request: SimpleNamespace(
            request=request,
            trial=SimpleNamespace(initial_seed=1, sha256=SHA_B, task="lift_can"),
        ),
    )
    launched = False

    class NeverLaunch:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal launched
            launched = True

    monkeypatch.setattr(module, "SameTaskIsaacSession", NeverLaunch)
    with pytest.raises(FileExistsError, match="never repeat"):
        module.execute_condition_work(campaign, work_path)
    assert launched is False


def test_supervisor_leaves_server_output_for_serve_retrained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    code = tmp_path / "code"
    package = tmp_path / "package"
    runtime = tmp_path / "python"
    code.mkdir()
    package.mkdir()
    runtime.touch()
    receipt = campaign / "server/worker/server_receipt.json"
    work_path = campaign / "work_manifest.json"
    manifest = module.ConditionWorkManifest.build(
        task_id="lift_can",
        condition_id="clean",
        capture_profile=LiveCaptureProfile.METRICS_ONLY.value,
        evidence_mode="diagnostic",
        binding_relpath="binding.json",
        binding_file_sha256=SHA_A,
        model_identity_sha256=SHA_A,
        reset_equivalence_receipt=None,
        reset_equivalence_receipt_sha256=None,
        items=(_work_item(0, 3),),
    )
    binding = {
        "artifact": str(tmp_path / "artifact.json"),
        "port": 24567,
        "runtime_python": str(runtime),
        "server_receipt": str(receipt),
    }

    def prepare(**_kwargs: object) -> Path:
        campaign.mkdir()
        module.write_json(work_path, {"prepared": True})
        return work_path

    launches: list[list[str]] = []

    class FakeProcess:
        def __init__(self, pid: int, *, worker: bool) -> None:
            self.pid = pid
            self.worker = worker

        def poll(self) -> None:
            return None

        def wait(self) -> int:
            assert self.worker
            module.write_json(
                campaign / "persistent_condition_session.preclose.json",
                _worker_preclose_document(manifest),
            )
            return 0

    def popen(command: list[str], **_kwargs: object) -> FakeProcess:
        launches.append(command)
        worker = len(launches) == 2
        if not worker:
            assert (campaign / "server").is_dir()
            assert not receipt.parent.exists()
            receipt.parent.mkdir()
        return FakeProcess(1000 + len(launches), worker=worker)

    class FakeSocket:
        def __enter__(self) -> FakeSocket:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def bind(self, address: tuple[str, int]) -> None:
            assert address == ("127.0.0.1", 24567)

    read_object = module.read_object
    monkeypatch.setattr(module, "prepare_condition_work", prepare)
    monkeypatch.setattr(module, "load_work_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        module,
        "read_object",
        lambda path: (
            binding
            if Path(path) == campaign / manifest.binding_relpath
            else read_object(Path(path))
        ),
    )
    monkeypatch.setattr(module, "resolve_isaac_python", lambda _binding: runtime)
    monkeypatch.setattr(
        module,
        "build_process_environments",
        lambda *_args: ({}, {}),
    )
    monkeypatch.setattr(module.socket, "socket", FakeSocket)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module, "wait_ready", lambda *_args: None)
    monkeypatch.setattr(module, "stop_owned", lambda _process: None)
    monkeypatch.setattr(
        module,
        "_summary",
        lambda *_args: {
            "completed_episode_count": 1,
            "planned_episode_count": 1,
        },
    )

    summary = module.supervise(
        binding_path=tmp_path / "binding.json",
        task_id="lift_can",
        condition_id="clean",
        campaign=campaign,
        code=code,
        package=package,
        seeds=(3,),
        capture_profile=LiveCaptureProfile.METRICS_ONLY,
        evidence_mode="diagnostic",
        severity_registry="optical_marker_extreme_v1",
        severity_level=5,
        fault_window_mode="early_random_onset_v1",
        fault_onset_max_index=8,
        reset_equivalence_receipt=None,
    )

    assert len(launches) == 2
    assert launches[0][-4:-2] == ["--output", str(receipt.parent)]
    assert summary["n0_server_launch_count"] == 1
    assert summary["isaac_application_launch_count"] == 1
    final = module.read_object(campaign / "persistent_condition_session.json")
    assert final["app_close_status"] == "system_exit_zero"
    assert final["receipt_stage"] == "final"
    assert final["total_duration_s"] >= 10.0


def test_supervisor_does_not_promote_preclose_after_nonzero_worker_exit(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    manifest = module.ConditionWorkManifest.build(
        task_id="lift_can",
        condition_id="clean",
        capture_profile=LiveCaptureProfile.METRICS_ONLY.value,
        evidence_mode="diagnostic",
        binding_relpath="binding.json",
        binding_file_sha256=SHA_A,
        model_identity_sha256=SHA_A,
        reset_equivalence_receipt=None,
        reset_equivalence_receipt_sha256=None,
        items=(_work_item(0, 3),),
    )
    module.write_json(
        campaign / "persistent_condition_session.preclose.json",
        _worker_preclose_document(manifest),
    )

    with pytest.raises(RuntimeError, match="worker exited 9"):
        module._finalize_worker_session(campaign, manifest, 9)

    assert not (campaign / "persistent_condition_session.json").exists()
