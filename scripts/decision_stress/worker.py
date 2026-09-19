"""One simulator and one retained model per task/seed; only missing conditions."""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

from robotactile_benchmark.closed_loop.interfaces import ClosedLoopPolicy
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.live_artifacts import write_live_univtac_artifact
from robotactile_benchmark.execution.loading import (
    LoadedLiveUniVTACRun,
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.official_act import (
    build_official_act_live_binding,
    make_official_act_policy_factory,
)
from robotactile_benchmark.execution.official_n0 import (
    build_official_n0_live_binding,
    make_official_n0_policy_factory,
)
from robotactile_benchmark.execution.paired_live_univtac import (
    _execute_one,
    default_paired_backend_session_factory,
)
from robotactile_benchmark.execution.sequential_policy_pool import SequentialPolicyPool
from robotactile_benchmark.integrations.runtime_config import (
    resolve_act_runtime_artifacts,
    resolve_n0_runtime_artifacts,
)
from scripts.decision_stress.remaining import read, sha, write


def execute(plan_path: Path, n0_source: Path, port: int) -> None:
    plan = read(plan_path)
    root = plan_path.parent
    items = []
    for cell in plan["cells"]:
        path = Path(cell["request"])
        if sha(path) != cell["request_sha256"] or Path(cell["artifact"]).exists():
            raise ValueError(
                "changed request or pre-existing artifact; no automatic rerun"
            )
        item = load_live_univtac_run(load_live_univtac_request(path))
        if item.content_sha256 != cell["run_content_sha256"]:
            raise ValueError("request dependencies changed")
        items.append(item)
    for cell in plan["prior_cells"]:
        if (
            sha(Path(cell["artifact"]) / "terminal_result.json")
            != cell["terminal_file_sha256"]
        ):
            raise ValueError("historical result changed")
    base = items[0]
    if any(
        i.backend_config != base.backend_config
        or i.trial.pair_key != base.trial.pair_key
        for i in items
    ):
        raise ValueError("continuation task/seed/physical contract mismatch")
    with SequentialPolicyPool() as pool:
        contract = None
        if plan["model"] == "n0":
            runtime = resolve_n0_runtime_artifacts(Path(plan["integration_config"]))
            binding = build_official_n0_live_binding(
                base.request,
                manifest=runtime.manifest,
                source_root=n0_source,
                host="127.0.0.1",
                port=port,
            )
            factory = make_official_n0_policy_factory(binding)
            contract = "robotactile_n0_training_60hz_ee_v1"
        else:
            act = resolve_act_runtime_artifacts(Path(plan["integration_config"]))
            act_binding = build_official_act_live_binding(
                base.request,
                artifact_root=act.artifact_root,
                stats_sha256=act.stats_sha256,
                encoder_sha256=act.encoder_sha256,
            )
            raw_factory = make_official_act_policy_factory(act_binding)

            def factory(loaded: LoadedLiveUniVTACRun) -> ClosedLoopPolicy:
                return pool.acquire(
                    (act_binding.load_request, loaded.policy_identity),
                    loaded.policy_identity,
                    lambda: raw_factory(loaded),
                )

        session = default_paired_backend_session_factory(
            base, n0_action_execution_contract=contract
        )
        exporter = partial(
            write_live_univtac_artifact, capture_profile=LiveCaptureProfile.PREVIEW
        )
        try:
            for cell, item in zip(plan["cells"], items):
                started = time.time()
                write(
                    root / "attempts" / f"{cell['condition']}.json",
                    {"started_unix": started},
                )
                execution, witness_index = _execute_one(
                    item,
                    session,
                    policy_factory=factory,
                    n0_transport_factory=None,
                    artifact_exporter=exporter,
                )
                result = execution.evidence.result
                # Historical Clean is cross-process evidence; do not claim same-process replay.
                write(
                    root / "receipts" / f"{cell['condition']}.json",
                    {
                        "started_unix": started,
                        "completed_unix": time.time(),
                        "result_sha256": result.sha256,
                        "witness_index": witness_index,
                        "initial_state_sha256": result.initial_state_sha256,
                        "historical_clean_initial_state_match": result.initial_state_sha256
                        == plan["prior_initial_state_sha256"],
                        "reset_receipt": session.reset_receipt.to_dict(),
                        "simulator_qualification_claimed": False,
                    },
                )
            write(
                root / "execution_complete.json",
                {"completed_unix": time.time(), "episode_count": len(items)},
            )
        finally:
            session.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--n0-source", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    a = p.parse_args()
    execute(a.plan, a.n0_source, a.port)


if __name__ == "__main__":
    main()
