"""Task-local Isaac app reuse for sequential episode execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, cast

from robotactile_benchmark.backends.univtac_factory import launch_univtac_app_host
from robotactile_benchmark.backends.univtac_host import UniVTACSimulationAppHost
from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend
from robotactile_benchmark.execution.contracts import LiveUniVTACRunRequest
from robotactile_benchmark.execution.isaac_runtime_attestation import (
    IsaacAttestationRequest,
)
from robotactile_benchmark.execution.lifecycle_watchdog import LifecycleStageJournal
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.execution.live_univtac import (
    LiveArtifactExporter,
    LiveBackendFactory,
    LivePolicyFactory,
    LiveUniVTACExecutionResult,
    execute_live_univtac_run,
)
from robotactile_benchmark.execution.loading import (
    LoadedLiveUniVTACRun,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.official_n0 import execute_official_n0_live_run
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    N0TWAMArtifactManifest,
)


class SameTaskEpisodeExecutor:
    """Own one Isaac app and construct a fresh UniVTAC task per episode."""

    def __init__(
        self,
        *,
        bootstrap: LoadedLiveUniVTACRun,
        manifest: N0TWAMArtifactManifest,
        n0_source_root: Path,
        n0_host: str,
        n0_port: int,
        action_execution_contract: str,
    ) -> None:
        if type(bootstrap) is not LoadedLiveUniVTACRun:
            raise TypeError("bootstrap must be an exact LoadedLiveUniVTACRun")
        if type(manifest) is not N0TWAMArtifactManifest:
            raise TypeError("manifest must be an exact N0TWAMArtifactManifest")
        if manifest.task_id != bootstrap.request.task_id:
            raise ValueError("bootstrap task does not match N0 manifest")
        self._bootstrap = bootstrap
        self._manifest = manifest
        self._n0_source_root = Path(n0_source_root).absolute()
        self._n0_host = n0_host
        self._n0_port = n0_port
        self._action_execution_contract = action_execution_contract
        request = bootstrap.request
        self._host: Optional[UniVTACSimulationAppHost] = launch_univtac_app_host(
            bootstrap.backend_config,
            upstream_root=request.upstream_root,
            initial_seed=bootstrap.trial.initial_seed,
            launcher_args=request.launcher_args,
            device=request.simulator_device,
            n0_action_execution_contract=action_execution_contract,
        )

    @property
    def runtime_count(self) -> int:
        host = self._require_host()
        return host.runtime_count

    def execute(
        self,
        request: LiveUniVTACRunRequest,
        *,
        lifecycle_journal: LifecycleStageJournal,
        isaac_attestation_request: IsaacAttestationRequest,
    ) -> LoadedLiveUniVTACArtifact:
        """Execute one request through a newly constructed task runtime."""

        if type(request) is not LiveUniVTACRunRequest:
            raise TypeError("request must be an exact LiveUniVTACRunRequest")
        loaded = load_live_univtac_run(request)
        self._validate_loaded(loaded)
        host = self._require_host()

        def backend_factory(selected: LoadedLiveUniVTACRun) -> UniVTACIsaacBackend:
            self._validate_loaded(selected)
            runtime = host.create_runtime(
                runtime_dir=selected.request.runtime_dir,
                initial_seed=selected.trial.initial_seed,
                stage_observer=lifecycle_journal.observe,
            )
            try:
                return UniVTACIsaacBackend(
                    selected.backend_config,
                    runtime,
                    success_predicate_id=selected.run_spec.success_predicate_id,
                )
            except BaseException:
                runtime.close_runtime()
                raise

        return execute_official_n0_live_run(
            request,
            manifest=self._manifest,
            source_root=self._n0_source_root,
            host=self._n0_host,
            port=self._n0_port,
            api_key=os.environ.get("N0_TWAM_API_KEY"),
            backend_factory=cast(LiveBackendFactory, backend_factory),
            lifecycle_journal=lifecycle_journal,
            isaac_attestation_request=isaac_attestation_request,
            action_execution_contract=self._action_execution_contract,
        )

    def close(self) -> None:
        """Close an active task, if any, and the shared Isaac app exactly once."""

        host = self._host
        if host is None:
            return
        self._host = None
        host.close()

    def _require_host(self) -> UniVTACSimulationAppHost:
        host = self._host
        if host is None or host.closed:
            raise RuntimeError("same-task episode executor is closed")
        return host

    def _validate_loaded(self, loaded: LoadedLiveUniVTACRun) -> None:
        baseline = self._bootstrap
        request = loaded.request
        expected = baseline.request
        if (
            request.task_id != expected.task_id
            or loaded.backend_config != baseline.backend_config
            or request.upstream_root != expected.upstream_root
            or request.launcher_args != expected.launcher_args
            or request.simulator_device != expected.simulator_device
        ):
            raise ValueError("episode changed the task-local Isaac app contract")


class SameTaskIsaacSession:
    """Reuse one Isaac application while rebuilding the task for every episode.

    Unlike :class:`SameTaskEpisodeExecutor`, this class is policy-agnostic.  It
    is used by condition-sharded robustness sweeps where the N0 endpoint is
    already source-bound by the parent supervisor.  Only the application is
    reused: every call leases a newly constructed UniVTAC task runtime and the
    normal closed-loop runner closes that runtime and its policy before return.
    """

    def __init__(
        self,
        *,
        bootstrap: LoadedLiveUniVTACRun,
        action_execution_contract: str,
    ) -> None:
        if type(bootstrap) is not LoadedLiveUniVTACRun:
            raise TypeError("bootstrap must be an exact LoadedLiveUniVTACRun")
        self._bootstrap = bootstrap
        self._action_execution_contract = action_execution_contract
        request = bootstrap.request
        self._host: Optional[UniVTACSimulationAppHost] = launch_univtac_app_host(
            bootstrap.backend_config,
            upstream_root=request.upstream_root,
            initial_seed=bootstrap.trial.initial_seed,
            launcher_args=request.launcher_args,
            device=request.simulator_device,
            n0_action_execution_contract=action_execution_contract,
        )

    @property
    def runtime_count(self) -> int:
        """Return the number of freshly constructed task runtimes."""

        return self._require_host().runtime_count

    def execute(
        self,
        request: LiveUniVTACRunRequest,
        *,
        policy_factory: LivePolicyFactory,
        artifact_exporter: LiveArtifactExporter,
        lifecycle_journal: Optional[LifecycleStageJournal] = None,
    ) -> LiveUniVTACExecutionResult:
        """Run one episode with fresh task, policy, and fault-session state."""

        if type(request) is not LiveUniVTACRunRequest:
            raise TypeError("request must be an exact LiveUniVTACRunRequest")
        if not callable(policy_factory) or not callable(artifact_exporter):
            raise TypeError("policy_factory and artifact_exporter must be callable")
        loaded = load_live_univtac_run(request)
        self._validate_loaded(loaded)
        host = self._require_host()

        def backend_factory(selected: LoadedLiveUniVTACRun) -> UniVTACIsaacBackend:
            self._validate_loaded(selected)
            runtime = host.create_runtime(
                runtime_dir=selected.request.runtime_dir,
                initial_seed=selected.trial.initial_seed,
                stage_observer=(
                    None if lifecycle_journal is None else lifecycle_journal.observe
                ),
            )
            try:
                return UniVTACIsaacBackend(
                    selected.backend_config,
                    runtime,
                    success_predicate_id=selected.run_spec.success_predicate_id,
                )
            except BaseException:
                runtime.close_runtime()
                raise

        return execute_live_univtac_run(
            request,
            backend_factory=cast(LiveBackendFactory, backend_factory),
            policy_factory=policy_factory,
            artifact_exporter=artifact_exporter,
            lifecycle_journal=lifecycle_journal,
            n0_action_execution_contract=self._action_execution_contract,
        )

    def close(self) -> None:
        """Close the last task lease, if any, then the shared application."""

        host = self._host
        if host is None:
            return
        self._host = None
        host.close()

    def _require_host(self) -> UniVTACSimulationAppHost:
        host = self._host
        if host is None or host.closed:
            raise RuntimeError("same-task Isaac session is closed")
        return host

    def _validate_loaded(self, loaded: LoadedLiveUniVTACRun) -> None:
        baseline = self._bootstrap
        request = loaded.request
        expected = baseline.request
        if (
            request.task_id != expected.task_id
            or loaded.backend_config != baseline.backend_config
            or request.upstream_root != expected.upstream_root
            or request.launcher_args != expected.launcher_args
            or request.simulator_device != expected.simulator_device
        ):
            raise ValueError("episode changed the task-local Isaac app contract")


__all__ = ["SameTaskEpisodeExecutor", "SameTaskIsaacSession"]
