"""CPU-only component contracts for the persistent same-task worker."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import pytest

from robotactile_benchmark.execution.contracts import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
    production_univtac_launcher_args,
)
from robotactile_benchmark.execution.same_task_worker_protocol import (
    SAME_TASK_WORKER_CONTRACT,
    SAME_TASK_WORKER_SEMANTIC_VERSION,
    SameTaskWorkerRequest,
    SameTaskWorkerResponse,
    compute_request_id,
    read_frame,
    write_frame,
)
from robotactile_benchmark.trials import Condition

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


class _NoCloseBytesIO(io.BytesIO):
    def close(self) -> None:
        return


class _Duplex:
    def __init__(self, response: bytes) -> None:
        self.reader = io.BytesIO(response)
        self.writer = io.BytesIO()

    def read(self, size: int = -1) -> bytes:
        return self.reader.read(size)

    def write(self, payload: bytes) -> int:
        return self.writer.write(payload)

    def flush(self) -> None:
        return

    def __enter__(self) -> _Duplex:
        return self

    def __exit__(self, *_args: object) -> None:
        return


class _ClientSocket:
    def __init__(self, stream: _Duplex) -> None:
        self.stream = stream
        self.connected: str | None = None

    def __enter__(self) -> _ClientSocket:
        return self

    def __exit__(self, *_args: object) -> None:
        return

    def connect(self, path: str) -> None:
        self.connected = path

    def makefile(self, _mode: str) -> _Duplex:
        return self.stream


def _request_identity() -> SameTaskWorkerRequest:
    request_id = compute_request_id(
        campaign_manifest_sha256=SHA_A,
        task_id="lift_can",
        ordinal=2,
        attempt_id="attempt-2",
        request_file_sha256=SHA_B,
    )
    return SameTaskWorkerRequest(
        worker_contract=SAME_TASK_WORKER_CONTRACT,
        semantic_version=SAME_TASK_WORKER_SEMANTIC_VERSION,
        request_id=request_id,
        campaign_manifest_sha256=SHA_A,
        task_id="lift_can",
        ordinal=2,
        attempt_id="attempt-2",
        request_file_sha256=SHA_B,
        trial_manifest_sha256=SHA_C,
    )


def _response_wire(response: SameTaskWorkerResponse) -> bytes:
    stream = io.BytesIO()
    write_frame(stream, response.to_document())
    return stream.getvalue()


def _client_args(tmp_path: Path, request_id: str) -> argparse.Namespace:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    socket_path = tmp_path / "worker.sock"
    socket_path.touch()
    return argparse.Namespace(
        socket=socket_path,
        request=request_path,
        lifecycle_journal=tmp_path / "journal",
        request_id=request_id,
        campaign_manifest_sha256=SHA_A,
        task="lift_can",
        ordinal=2,
        attempt_id="attempt-2",
        request_file_sha256=SHA_B,
        trial_manifest_sha256=SHA_C,
    )


def test_client_completed_verifies_artifact_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from robotactile_benchmark.execution import same_task_worker_client as module

    request = _request_identity()
    response = SameTaskWorkerResponse(
        worker_contract=SAME_TASK_WORKER_CONTRACT,
        semantic_version=SAME_TASK_WORKER_SEMANTIC_VERSION,
        request_id=request.request_id,
        task_id=request.task_id,
        ordinal=request.ordinal,
        attempt_id=request.attempt_id,
        status="completed",
        return_code=0,
        artifact_root_sha256=SHA_D,
        exception_code=None,
    )
    duplex = _Duplex(_response_wire(response))
    client_socket = _ClientSocket(duplex)
    args = _client_args(tmp_path, request.request_id)
    monkeypatch.setattr(module, "_validated_journal", lambda *_args: object())
    monkeypatch.setattr(module, "_validated_socket", lambda path: path)
    monkeypatch.setattr(module, "sha256_file", lambda _path: SHA_B)
    monkeypatch.setattr(
        module,
        "load_live_univtac_request",
        lambda _path: SimpleNamespace(task_id="lift_can", output_dir=tmp_path / "out"),
    )
    artifact = SimpleNamespace(external_root_sha256=SHA_D)
    monkeypatch.setattr(module, "load_live_univtac_artifact", lambda _path: artifact)
    monkeypatch.setattr(
        module, "official_act_live_summary", lambda _artifact: {"status": "ok"}
    )
    monkeypatch.setattr(module.socket, "socket", lambda *_args: client_socket)

    assert module.execute_client(args) == {"status": "ok"}
    dispatched = SameTaskWorkerRequest.from_document(
        read_frame(io.BytesIO(duplex.writer.getvalue()))
    )
    assert dispatched == request

    artifact.external_root_sha256 = SHA_C
    duplex.reader = io.BytesIO(_response_wire(response))
    duplex.writer = io.BytesIO()
    with pytest.raises(ValueError, match="artifact SHA256 mismatch"):
        module.execute_client(args)


def test_client_failed_response_preserves_remote_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from robotactile_benchmark.execution import same_task_worker_client as module

    request = _request_identity()
    response = SameTaskWorkerResponse(
        worker_contract=SAME_TASK_WORKER_CONTRACT,
        semantic_version=SAME_TASK_WORKER_SEMANTIC_VERSION,
        request_id=request.request_id,
        task_id=request.task_id,
        ordinal=request.ordinal,
        attempt_id=request.attempt_id,
        status="failed",
        return_code=2,
        artifact_root_sha256=None,
        exception_code="episode_failed",
    )
    duplex = _Duplex(_response_wire(response))
    args = _client_args(tmp_path, request.request_id)
    monkeypatch.setattr(module, "_validated_journal", lambda *_args: object())
    monkeypatch.setattr(module, "_validated_socket", lambda path: path)
    monkeypatch.setattr(module, "sha256_file", lambda _path: SHA_B)
    monkeypatch.setattr(
        module,
        "load_live_univtac_request",
        lambda _path: SimpleNamespace(task_id="lift_can", output_dir=tmp_path / "out"),
    )
    monkeypatch.setattr(module.socket, "socket", lambda *_args: _ClientSocket(duplex))

    with pytest.raises(module.SameTaskWorkerRemoteError) as captured:
        module.execute_client(args)
    assert captured.value.failure_code == "episode_failed"


def _live_request(tmp_path: Path, task_id: str = "lift_can") -> LiveUniVTACRunRequest:
    return LiveUniVTACRunRequest(
        task_id=task_id,
        condition=Condition.CLEAN,
        policy_kind=LivePolicyKind.N0,
        base_system_id="n0-test",
        dataset_sha256=SHA_A,
        checkpoint_sha256=SHA_B,
        config_sha256=SHA_C,
        base_system_manifest_sha256=SHA_D,
        initial_seed=1,
        exogenous_seed=2,
        max_control_cycles=1,
        max_observation_steps=2,
        execute_action_steps=24,
        wall_timeout_s=1.0,
        upstream_root=tmp_path / "upstream",
        runtime_dir=tmp_path / "runtime",
        output_dir=tmp_path / "output",
        fault_manifest_path=None,
        rest_references_path=None,
        restoration_index=None,
        restoration_mode=None,
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name=None,
        simulator_device="cuda:0",
        launcher_args=production_univtac_launcher_args(),
        n0_source_commit="1" * 40,
        n0_normalizer_sha256="2" * 64,
        n0_serve_bundle_sha256="3" * 64,
        n0_prompt_manifest_sha256="4" * 64,
    )


def test_runtime_reuses_app_but_constructs_fresh_episode_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from robotactile_benchmark.execution import same_task_worker_runtime as module

    request = _live_request(tmp_path)
    backend_config = object()
    loaded = SimpleNamespace(
        request=request,
        backend_config=backend_config,
        trial=SimpleNamespace(initial_seed=1),
    )

    class FakeRuntime:
        def close_runtime(self) -> None:
            return

    class FakeHost:
        closed = False

        def __init__(self) -> None:
            self.created: list[FakeRuntime] = []
            self.close_count = 0

        @property
        def runtime_count(self) -> int:
            return len(self.created)

        def create_runtime(self, **_kwargs: object) -> FakeRuntime:
            runtime = FakeRuntime()
            self.created.append(runtime)
            return runtime

        def close(self) -> None:
            self.close_count += 1
            self.closed = True

    host = FakeHost()
    executor = module.SameTaskEpisodeExecutor.__new__(module.SameTaskEpisodeExecutor)
    executor._bootstrap = loaded
    executor._manifest = SimpleNamespace()
    executor._n0_source_root = tmp_path
    executor._n0_host = "127.0.0.1"
    executor._n0_port = 29601
    executor._action_execution_contract = "robotactile_n0_training_60hz_ee_v1"
    executor._host = host
    monkeypatch.setattr(module, "load_live_univtac_run", lambda _request: loaded)
    backends: list[object] = []
    monkeypatch.setattr(
        module,
        "UniVTACIsaacBackend",
        lambda _config, runtime: backends.append(runtime) or object(),
    )

    def fake_execute(_request: object, **kwargs: object) -> object:
        factory = kwargs["backend_factory"]
        assert callable(factory)
        factory(loaded)
        return SimpleNamespace(external_root_sha256=SHA_D)

    monkeypatch.setattr(module, "execute_official_n0_live_run", fake_execute)
    journal = SimpleNamespace(observe=lambda *_args: None)
    attestation = SimpleNamespace()

    executor.execute(
        request, lifecycle_journal=journal, isaac_attestation_request=attestation
    )
    executor.execute(
        request, lifecycle_journal=journal, isaac_attestation_request=attestation
    )

    assert len(host.created) == 2
    assert host.created[0] is not host.created[1]
    assert backends == host.created
    executor.close()
    executor.close()
    assert host.close_count == 1


def test_runtime_rejects_cross_task_episode(tmp_path: Path) -> None:
    from robotactile_benchmark.execution import same_task_worker_runtime as module

    baseline = SimpleNamespace(
        request=_live_request(tmp_path, "lift_can"), backend_config=object()
    )
    changed = SimpleNamespace(
        request=_live_request(tmp_path, "insert_hole"),
        backend_config=baseline.backend_config,
    )
    executor = module.SameTaskEpisodeExecutor.__new__(module.SameTaskEpisodeExecutor)
    executor._bootstrap = baseline
    with pytest.raises(ValueError, match="task-local Isaac app contract"):
        executor._validate_loaded(changed)


def test_server_request_identity_and_reset_proof_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from robotactile_benchmark.clean_baseline.contracts import CleanCampaignProtocol
    from robotactile_benchmark.execution import same_task_worker_server as module

    root = tmp_path / "deployment"
    root.mkdir()
    assert module._validate_reset_proof(
        root, CleanCampaignProtocol.DIAGNOSTIC, None, None
    ) == (None, None)
    for protocol in (CleanCampaignProtocol.PILOT, CleanCampaignProtocol.PAPER):
        with pytest.raises(ValueError, match="requires reset-equivalence"):
            module._validate_reset_proof(root, protocol, None, None)

    proof = root / "artifacts/deployment/reset.json"
    proof.parent.mkdir(parents=True)
    proof.write_text("{}", encoding="utf-8")
    digest = module.sha256_file(proof)
    assert module._validate_reset_proof(
        root, CleanCampaignProtocol.PAPER, proof, digest
    ) == ("artifacts/deployment/reset.json", digest)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module._validate_reset_proof(root, CleanCampaignProtocol.PAPER, proof, SHA_A)

    request = _request_identity()
    entry = SimpleNamespace(
        ordinal=2,
        request_file_sha256=SHA_B,
        trial_manifest_sha256=SHA_C,
        request_relpath="requests/task/request.json",
    )
    request_path = root / entry.request_relpath
    request_path.parent.mkdir(parents=True)
    request_path.write_text("{}", encoding="utf-8")
    server = module.SameTaskWorkerServer.__new__(module.SameTaskWorkerServer)
    server._root = root
    server._manifest = SimpleNamespace(sha256=SHA_A, campaign_id="campaign")
    server._task_id = "lift_can"
    server._entries = {2: entry}
    monkeypatch.setattr(module, "sha256_file", lambda _path: SHA_B)
    journal = SimpleNamespace(
        identity=SimpleNamespace(
            to_dict=lambda: {
                "attempt_id": request.attempt_id,
                "campaign_manifest_sha256": SHA_A,
                "ordinal": 2,
                "request_file_sha256": SHA_B,
                "semantic_version": "1.0",
                "task_id": "lift_can",
                "trial_manifest_sha256": SHA_C,
            }
        )
    )
    monkeypatch.setattr(
        module,
        "campaign_watchdog_paths",
        lambda *_args: (root / "journal", root / "watchdog"),
    )
    monkeypatch.setattr(module.LifecycleStageJournal, "open", lambda _path: journal)

    selected, loaded_path, loaded_journal = server._validate_request(request)
    assert (
        selected is entry and loaded_path == request_path and loaded_journal is journal
    )

    mismatched = SameTaskWorkerRequest(
        **{
            **request.to_document(),
            "task_id": "insert_hole",
            "request_id": compute_request_id(
                campaign_manifest_sha256=SHA_A,
                task_id="insert_hole",
                ordinal=2,
                attempt_id=request.attempt_id,
                request_file_sha256=SHA_B,
            ),
        }  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="campaign/task mismatch"):
        server._validate_request(mismatched)


def test_server_accepts_canonical_manifest_with_distinct_file_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from robotactile_benchmark.clean_baseline.contracts import CleanCampaignProtocol
    from robotactile_benchmark.execution import same_task_worker_server as module

    root = tmp_path / "deployment"
    manifest_path = root / "requests/campaign.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(b"{}\n")
    assert module.sha256_file(manifest_path) != SHA_A

    request_path = root / "requests/task/request.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_bytes(b"{}\n")
    request_sha256 = module.sha256_file(request_path)
    entry = SimpleNamespace(
        ordinal=0,
        task="lift_can",
        request_relpath="requests/task/request.json",
        request_file_sha256=request_sha256,
        trial_manifest_sha256=SHA_B,
    )
    manifest = SimpleNamespace(
        campaign_id="campaign",
        protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
        sha256=SHA_A,
        trials=(entry,),
    )
    source = SimpleNamespace(
        action_execution_contract="robotactile_n0_training_60hz_ee_v1",
        checkpoint_sha256=SHA_A,
        config_sha256=SHA_B,
        n0_source_commit="1" * 40,
        normalizer_sha256=SHA_C,
        prompt_manifest_sha256=SHA_D,
        serve_bundle_sha256="e" * 64,
    )
    context = SimpleNamespace(runtime_source_binding=source)
    integration_config = root / "artifacts/config.json"
    integration_config.parent.mkdir(parents=True)
    integration_config.write_bytes(b"{}\n")
    runtime_manifest = SimpleNamespace(
        task_id="lift_can",
        external_commit=source.n0_source_commit,
        checkpoint_sha256=source.checkpoint_sha256,
        config_sha256=source.config_sha256,
        normalizer_sha256=source.normalizer_sha256,
        prompt_manifest_sha256=source.prompt_manifest_sha256,
        serve_bundle_sha256=source.serve_bundle_sha256,
    )

    monkeypatch.setattr(module, "load_clean_campaign_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        module, "load_source_bound_campaign_context", lambda **_kwargs: context
    )
    monkeypatch.setattr(
        module,
        "resolve_n0_runtime_artifacts",
        lambda _path: SimpleNamespace(manifest=runtime_manifest),
    )
    monkeypatch.setattr(module, "load_live_univtac_request", lambda _path: object())
    monkeypatch.setattr(
        module,
        "load_live_univtac_run",
        lambda _request: SimpleNamespace(trial=SimpleNamespace(sha256=SHA_B)),
    )
    monkeypatch.setattr(module, "SameTaskEpisodeExecutor", lambda **_kwargs: object())

    args = argparse.Namespace(
        root=root,
        campaign_manifest=manifest_path,
        task="lift_can",
        config=integration_config,
        n0_source_root=root / "sources/N0-TWAM",
        n0_host="127.0.0.1",
        n0_port=29603,
        action_execution_contract="robotactile_n0_training_60hz_ee_v1",
        qualification=root / "artifacts/qualification.json",
        n0_server_attestation=root / "outputs/n0.json",
        n0_server_attestation_sha256=SHA_A,
        socket=root / "runtime/worker.sock",
        ready_receipt=root / "outputs/worker.ready.json",
        session_receipt=root / "outputs/worker.session.json",
        reset_equivalence_receipt=None,
        reset_equivalence_receipt_sha256=None,
        restart_generation=0,
    )

    server = module.SameTaskWorkerServer(args)
    assert server._manifest is manifest
    assert server._manifest.sha256 == SHA_A


def test_server_completed_and_failed_responses_are_identity_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robotactile_benchmark.execution import same_task_worker_server as module

    request = _request_identity()
    request_wire = io.BytesIO()
    write_frame(request_wire, request.to_document())
    entry = SimpleNamespace(
        ordinal=2,
        initial_seed=1,
        exogenous_seed=2,
        request_file_sha256=SHA_B,
        trial_manifest_sha256=SHA_C,
    )
    server = module.SameTaskWorkerServer.__new__(module.SameTaskWorkerServer)
    server._task_id = "lift_can"
    server._manifest = SimpleNamespace(campaign_id="campaign")
    server._context = SimpleNamespace(
        qualification_path=Path("qualification.json"),
        n0_server_attestation_path=Path("n0.json"),
        n0_server_attestation_sha256=SHA_A,
    )
    server._dispatches = []
    server._crashed = False
    server._root = Path("/deployment")
    server._executor = SimpleNamespace(
        execute=lambda *_args, **_kwargs: SimpleNamespace(external_root_sha256=SHA_D)
    )
    monkeypatch.setattr(
        server,
        "_validate_request",
        lambda _request: (entry, Path("request.json"), object()),
    )
    monkeypatch.setattr(module, "load_live_univtac_request", lambda _path: object())
    monkeypatch.setattr(
        module,
        "isaac_attestation_output_path",
        lambda *_args, **_kwargs: Path("attestation.json"),
    )
    monkeypatch.setattr(module, "_utc_now", lambda: "2026-08-27T00:00:00+00:00")

    class Connection:
        def __init__(self) -> None:
            self.streams: list[_NoCloseBytesIO] = []

        def makefile(self, _mode: str) -> BinaryIO:
            stream = _NoCloseBytesIO(
                request_wire.getvalue() if not self.streams else b""
            )
            self.streams.append(stream)
            return stream

    connection = Connection()
    server._serve_connection(connection)  # type: ignore[arg-type]
    stream = connection.streams[0]
    stream.seek(len(request_wire.getvalue()))
    response = SameTaskWorkerResponse.from_document(read_frame(stream))
    response.validate_against(request)
    assert response.status == "completed"
    assert response.artifact_root_sha256 == SHA_D
    assert len(server._dispatches) == 1

    server._crashed = False
    server._executor = SimpleNamespace(
        execute=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(module, "emit_crash_marker", lambda *_args: None)
    monkeypatch.setattr(server, "_last_stage", lambda _request: "execute")
    failed_connection = Connection()
    server._serve_connection(failed_connection)  # type: ignore[arg-type]
    failure = SameTaskWorkerResponse.from_document(
        read_frame(io.BytesIO(failed_connection.streams[1].getvalue()))
    )
    failure.validate_against(request)
    assert failure.status == "failed"
    assert failure.exception_code == "same_task_worker_episode_failed"
    assert server._crashed is True
