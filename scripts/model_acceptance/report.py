"""Fail-closed five-model evidence matrix; never launch models or episodes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import pprint
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MODELS = ("dream_tac", "act", "ftp1_policy", "n0_twam", "n0_vtla")
ALIASES = {"dream_tac": "dream", "n0_twam": "n0", "n0_vtla": "vtla"}


class GateFailure(ValueError):
    def __init__(self, gate: str, detail: str, path: Path) -> None:
        super().__init__(detail)
        self.gate, self.path = gate, str(path)


def require(ok: Any, gate: str, detail: str, path: Path) -> None:
    if not ok:
        raise GateFailure(gate, detail, path)


def read(path: Path, gate: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        raise GateFailure(gate, f"{type(error).__name__}: {error}", path) from error


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_probe(root: Path) -> dict[str, Any]:
    modules = {}
    for name in (
        "robotactile_benchmark.closed_loop.contracts",
        "robotactile_benchmark.closed_loop.runner",
        "robotactile_benchmark.execution.live_artifacts",
        "scripts.retrained_evaluation.group",
    ):
        try:
            spec = importlib.util.find_spec(name)
            path = Path(spec.origin).resolve() if spec and spec.origin else None
            modules[name] = {
                "path": str(path),
                "sha256": sha(path) if path else None,
                "within_current_code_snapshot": path is not None
                and (root / "code").resolve() in path.parents,
            }
        except (ImportError, OSError, ValueError) as error:
            modules[name] = {"error": str(error), "within_current_code_snapshot": False}
    return {
        "scope": "independent_resolution_at_report_time_not_historical_process_introspection",
        "hostname": socket.gethostname(),
        "unix_time": time.time(),
        "python": sys.executable,
        "current_code_snapshot": str(root / "code"),
        "modules": modules,
    }


def verify_training_config_patch(
    relative: str, patch: dict[str, Any], artifact: dict[str, Any], path: Path
) -> None:
    """Read registered constants and reconstruct the sole template without exec."""
    require(
        relative == "n0_twam/configs/twam_posttrain_cfg.py",
        "source",
        "unregistered upstream source patch",
        path,
    )
    renderer, contract = (
        Path(patch["renderer_path"]),
        Path(patch["policy_contract_path"]),
    )
    require(
        sha(renderer) == patch.get("renderer_sha256")
        and sha(contract) == patch.get("policy_contract_sha256"),
        "source",
        "renderer or policy contract changed",
        path,
    )
    constants = {}
    for node in ast.parse(contract.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "OFFICIAL_COMMIT",
                    "ALLOWED_OFFICIAL_CHANGE",
                }:
                    require(
                        target.id not in constants,
                        "source",
                        "duplicate policy constant",
                        contract,
                    )
                    constants[target.id] = ast.literal_eval(node.value)
    require(
        constants.get("OFFICIAL_COMMIT") == artifact["source"]["source_commit"]
        and constants.get("ALLOWED_OFFICIAL_CHANGE") == relative,
        "source",
        "training policy does not authorize this exact commit/file",
        contract,
    )
    snapshot = Path(artifact["source"]["root"]) / relative
    values = [
        ast.literal_eval(node.value)
        for node in ast.parse(snapshot.read_text()).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_VALUES" for t in node.targets)
    ]
    templates = [
        node.value
        for node in ast.walk(ast.parse(renderer.read_text()))
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.JoinedStr)
        and any(isinstance(t, ast.Name) and t.id == "source" for t in node.targets)
    ]
    require(
        len(values) == 1 and isinstance(values[0], dict) and len(templates) == 1,
        "source",
        "renderer/config AST is not the registered literal-template shape",
        renderer,
    )
    literal = pprint.pformat(values[0], sort_dicts=True, width=88)
    chunks = []
    for node in templates[0].values:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            chunks.append(node.value)
        elif (
            isinstance(node, ast.FormattedValue)
            and isinstance(node.value, ast.Name)
            and node.value.id == "literal"
            and node.conversion == -1
            and node.format_spec is None
        ):
            chunks.append(literal)
        else:
            raise GateFailure(
                "source", "unsupported executable interpolation in renderer", renderer
            )
    rebuilt = "".join(chunks).encode()
    expected = artifact["source"]["python_files"][relative]
    require(
        hashlib.sha256(rebuilt).hexdigest()
        == expected
        == patch.get("snapshot_sha256")
        == patch.get("renderer_reconstructed_sha256")
        and rebuilt == snapshot.read_bytes(),
        "source",
        "registered renderer does not exactly reproduce snapshot config",
        snapshot,
    )


def verify_source_attestation(artifact: dict[str, Any], path: Path) -> dict[str, Any]:
    """Bind a new Git-object proof to the unchanged prepared Python snapshot."""
    proof = read(path, "source")
    source = artifact["source"]
    require(
        proof.get("status") == "passed"
        and proof.get("verification_scope")
        in {
            "independent_git_object_comparison_at_acceptance_time",
            "independent_git_object_comparison_with_registered_training_config",
        }
        and proof.get("normal_python_file_set_equal") is True,
        "source",
        "independent Git-object attestation did not pass",
        path,
    )
    require(
        proof.get("source_commit") == source["source_commit"]
        and Path(str(proof.get("snapshot_root", ""))).resolve()
        == Path(source["root"]).resolve()
        and proof.get("prepared_artifact_sha256") == artifact["artifact_sha256"]
        and proof.get("prepared_source_tree_sha256") == artifact["source_tree_sha256"]
        and artifact["source_tree_sha256"] == source["source_tree_sha256"],
        "source",
        "independent attestation is bound to a different prepared source",
        path,
    )
    ordinary, metadata = (
        proof.get("python_files", {}),
        proof.get("appledouble_files", {}),
    )
    original = source["python_files"]
    require(
        isinstance(ordinary, dict)
        and ordinary
        and isinstance(metadata, dict)
        and not (set(ordinary) & set(metadata))
        and set(ordinary) | set(metadata) == set(original),
        "source",
        "attestation does not partition exactly the prepared Python file set",
        path,
    )
    import re

    patches = proof.get("documented_config_patches", {})
    patched_scope = (
        proof["verification_scope"]
        == "independent_git_object_comparison_with_registered_training_config"
    )
    require(
        isinstance(patches, dict)
        and (
            set(patches) == {"n0_twam/configs/twam_posttrain_cfg.py"}
            if patched_scope
            else not patches
        ),
        "source",
        "source proof does not declare exactly the registered config patch",
        path,
    )
    for relative, item in ordinary.items():
        require(
            not Path(relative).name.startswith("._")
            and isinstance(item, dict)
            and re.fullmatch(
                r"[0-9a-f]{40}|[0-9a-f]{64}", str(item.get("git_blob", ""))
            )
            is not None
            and item.get("actual_sha256") == original[relative],
            "source",
            f"normal source file is not Git-object matched: {relative}",
            path,
        )
        if relative in patches:
            patch = patches[relative]
            require(
                item.get("expected_sha256") == patch.get("upstream_sha256")
                and item.get("expected_sha256") != original[relative],
                "source",
                "registered patch does not bind the upstream SHA",
                path,
            )
            verify_training_config_patch(relative, patch, artifact, path)
        else:
            require(
                item.get("expected_sha256") == original[relative],
                "source",
                f"unregistered source difference: {relative}",
                path,
            )
    require(
        set(patches).issubset(ordinary),
        "source",
        "registered patch missing from full Python file table",
        path,
    )
    for relative, item in metadata.items():
        require(
            Path(relative).name.startswith("._")
            and isinstance(item, dict)
            and item.get("magic_hex") == "00051607"
            and item.get("sha256") == original[relative],
            "source",
            f"unproven AppleDouble exclusion: {relative}",
            path,
        )
        member = Path(source["root"]) / relative
        try:
            with member.open("rb") as stream:
                magic = stream.read(4).hex()
        except OSError as error:
            raise GateFailure("source", str(error), member) from error
        require(
            magic == "00051607",
            "source",
            "actual AppleDouble magic differs from proof",
            member,
        )
    return {
        "path": str(path),
        "sha256": sha(path),
        "verification_scope": proof["verification_scope"],
        "normal_python_files": len(ordinary),
        "appledouble_files": len(metadata),
        "source_commit": proof["source_commit"],
        "prepared_source_tree_sha256": proof["prepared_source_tree_sha256"],
        "normal_python_file_set_equal": True,
        "historical_artifact_modified": False,
        "pure_upstream_python_tree": not patches,
        "documented_config_patches": patches,
        "registered_patch_ast_reconstruction_verified": bool(patches),
    }


def source_gate(
    inventory: dict[str, Any], model: str, server: dict[str, Any], path: Path
) -> dict[str, Any]:
    binding, source = inventory["binding"], inventory["source"]
    expected = binding["source_commit"]
    require(
        source.get("expected_commit") == expected,
        "source",
        "inventory/binding source commit mismatch",
        path,
    )
    head = source.get("git_head")
    verification = (
        "git_head_verified_tree_separately_hashed" if head is not None else None
    )
    attestation = None
    if head is not None:
        require(
            head.get("returncode") == 0 and head.get("stdout", "").strip() == expected,
            "source",
            "recorded Git HEAD differs from bound source commit",
            path,
        )
    else:
        require(
            model == "n0_twam",
            "source",
            "missing Git source identity outside N0 snapshot contract",
            path,
        )
        artifact_path = Path(binding["artifact"])
        artifact = read(artifact_path, "source")
        require(
            artifact["source"].get("source_commit") == expected,
            "source",
            "N0 prepared snapshot commit mismatch",
            artifact_path,
        )
        require(
            server.get("artifact_sha256") == artifact.get("artifact_sha256")
            and server.get("source_verification")
            == artifact["source"].get("verification"),
            "source",
            "N0 server receipt does not bind the prepared source snapshot",
            path,
        )
        verification = artifact["source"].get("verification")
        if verification == "declared_unverified":
            attestation_root = path.parent.parent / "source_attestation"
            selected = attestation_root / "n0_twam_with_training_config.json"
            if not selected.is_file():
                selected = attestation_root / "n0_twam.json"
            attestation = verify_source_attestation(artifact, selected)
        else:
            require(
                verification
                in {
                    "git_head_verified_tree_separately_hashed",
                    "matched_source_snapshot_receipt",
                },
                "source",
                "unsupported prepared source verification status",
                artifact_path,
            )
    files = source.get("source_files_sha256", {})
    require(
        files and source.get("source_tree_digest"),
        "source",
        "missing concrete source member hashes",
        path,
    )
    require(
        hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        == source["source_tree_digest"],
        "source",
        "source member table digest mismatch",
        path,
    )
    source_root = Path(source["source_root"])
    require(
        source_root.resolve() == Path(binding["source_root"]).resolve(),
        "source",
        "source root differs from binding",
        path,
    )
    for relative, expected_sha in files.items():
        member = source_root / relative
        require(
            member.is_file() and sha(member) == expected_sha,
            "source",
            f"source member changed: {relative}",
            member,
        )
    return {
        "commit": expected,
        "git_head": head,
        "snapshot_without_git": head is None,
        "original_verification": verification,
        "independent_attestation": attestation,
        "tree_sha256": source["source_tree_digest"],
        "member_count": len(files),
        "inventory": str(path),
        "current_members_rehashed": True,
    }


def completed(bundle: Any, horizon: int) -> bool:
    actions = sum(
        entry.executed_actions.shape[0] for entry in bundle.evidence.action_entries
    )
    result = bundle.evidence.result
    return bool(
        result.validation_passed is True
        and result.score_eligible
        and (result.score_success is True or actions >= horizon)
    )


def verify_plans(episode: Path, bundle: Any) -> dict[str, Any]:
    from robotactile_benchmark.closed_loop.contracts import ActionPlan

    path = episode / "inference_trace.json"
    if path.is_file():
        traces = json.loads(path.read_text())
    else:
        path = episode / "inference_trace.jsonl"
        traces = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    require(
        isinstance(traces, list) and traces,
        "closed_loop",
        "missing full inference plans",
        path,
    )
    plans = {}
    for item in traces:
        plan = ActionPlan(
            item["action_spec"],
            item["plan_source_step_index"],
            np.asarray(item["planned_action_values"], dtype=np.float32),
        )
        require(
            plan.sha256 == item["action_plan_sha256"]
            and plan.action_spec == bundle.trial.action_spec
            and plan.source_step_index == item["source_step_index"],
            "closed_loop",
            "inference plan hash/spec/source mismatch",
            path,
        )
        require(
            plan.sha256 not in plans, "closed_loop", "duplicate inference plan", path
        )
        plans[plan.sha256] = plan
    for entry in bundle.evidence.action_entries:
        plan = plans.get(entry.action_plan_sha256)
        require(
            plan is not None
            and plan.source_step_index == entry.source_step_index
            and np.array_equal(
                plan.actions[: len(entry.executed_actions)], entry.executed_actions
            ),
            "closed_loop",
            "executed actions differ from inferred prefix",
            path,
        )
    return {
        "path": str(path),
        "sha256": sha(path),
        "plan_count": len(plans),
        "prefixes_verified": True,
        "scope": "independent_report_time_plan_and_executed_prefix_revalidation",
    }


def matrix_row(
    root: Path, model: str, host: str, modules: dict[str, Any]
) -> dict[str, Any]:
    row: dict[str, Any] = {"model": model, "status": "BLOCKED", "gap": None}
    target = root / "execution" / model
    inference_ready = False
    try:
        inventory_path = root / "inventory" / f"{model}.json"
        inventory = read(inventory_path, "weights")
        binding = inventory["binding"]
        weights = inventory.get("weights", [])
        require(
            inventory.get("all_members_verified") is True
            and weights
            and all(
                item.get("matches") is True and item.get("sha256") for item in weights
            ),
            "weights",
            "weight inventory did not verify every member",
            inventory_path,
        )
        require(
            all(item.get("expected_sha256") == item["sha256"] for item in weights)
            or model == "act",
            "weights",
            "weight has no matching expected SHA256",
            inventory_path,
        )
        row["weights"] = {
            "checkpoint_sha256": binding.get("checkpoint_sha256"),
            "checkpoint_root": binding["checkpoint_root"],
            "members": weights,
            "inventory": str(inventory_path),
        }
        if model == "dream_tac":
            from robotactile_benchmark.contracts import canonical_hash

            members = binding["checkpoint_members"]
            require(
                len(members) == 17
                and canonical_hash(members) == binding["checkpoint_sha256"],
                "weights",
                "Dream-Tac complete 17-member DCP table hash mismatch",
                inventory_path,
            )
            row["weights"]["dcp_member_table"] = members
        if model == "act":
            from robotactile_benchmark.integrations.runtime_config import (
                resolve_act_runtime_artifacts,
            )

            resolved = resolve_act_runtime_artifacts(
                Path(binding["integration_config"])
            )
            row["weights"]["checkpoint_sha256"] = resolved.manifest.checkpoint_sha256
            require(
                any(
                    item["sha256"] == resolved.manifest.checkpoint_sha256
                    for item in weights
                ),
                "weights",
                "ACT canonical checkpoint absent from inventory",
                inventory_path,
            )
            for expected in (resolved.stats_sha256, resolved.encoder_sha256):
                require(
                    any(item["sha256"] == expected for item in weights),
                    "weights",
                    "ACT encoder/stats inventory differs from canonical manifest",
                    inventory_path,
                )
        runtime_path = root / "runtime" / f"{model}.json"
        runtime = read(runtime_path, "runtime")
        runtime_exit = read(root / "runtime" / f"{model}.exit.json", "runtime")
        require(
            runtime_exit.get("returncode") == 0
            and runtime.get("cuda_matmul_passed") is True
            and runtime.get("capability") == [8, 0]
            and "A800" in runtime.get("gpu", ""),
            "runtime",
            "actual A800/SM80 CUDA runtime probe failed",
            root / "runtime" / f"{model}.log",
        )
        row["runtime"] = runtime
        execution_path = target / "execution.json"
        execution = read(execution_path, "model_load")
        require(
            execution.get("hostname") == host,
            "model_load",
            "execution host mismatch",
            execution_path,
        )
        require(
            not execution.get("error"),
            "model_load",
            str(execution.get("error")),
            target / "server.log",
        )
        server = {}
        if model != "act":
            receipt_path = Path(
                execution.get(
                    "real_weight_load_receipt", str(target / "server_receipt.json")
                )
            )
            server = read(receipt_path, "model_load")
            require(
                bool(server),
                "model_load",
                "missing real policy load receipt",
                target / "server.log",
            )
            if model in {"dream_tac", "n0_vtla"}:
                require(
                    server.get("status") == "model_loaded"
                    and server.get("binding_sha256") == binding["binding_sha256"],
                    "model_load",
                    "load receipt does not match the frozen binding",
                    target / "server.log",
                )
            if model == "ftp1_policy":
                require(
                    server.get("checkpoint_sha256") == binding["checkpoint_sha256"]
                    and server.get("source_commit") == binding["source_commit"]
                    and server.get("serve_bundle_sha256") == binding["binding_sha256"],
                    "model_load",
                    "FTP-1 load metadata differs from binding",
                    target / "server.log",
                )
        row["real_load"] = {"receipt": server, "execution": str(execution_path)}
        row["source"] = source_gate(inventory, model, server, inventory_path)
        require(
            all(
                item.get("within_current_code_snapshot")
                for item in modules["modules"].values()
            ),
            "module_resolution",
            "report-time modules do not resolve inside current code snapshot",
            root / "code",
        )
        inference_path = target / "inference.json"
        inference = read(inference_path, "fresh_inference")
        fresh = inference.get("inference", {})
        action = fresh.get("action", {})
        require(
            execution.get("fresh_inference_returncode") == 0
            and fresh.get("passed") is True
            and action.get("finite") is True
            and action.get("valid") is True
            and len(action.get("shape", [])) == 2
            and action["shape"][0] > 0
            and action["shape"][1] == 8,
            "fresh_inference",
            "fresh full action-plan inference did not pass",
            target / "inference.log",
        )
        row["fresh_inference"] = {**fresh, "receipt": str(inference_path)}
        if model == "n0_twam":
            from robotactile_benchmark.integrations.n0_twam.retrained import (
                load_retrained,
            )
            from robotactile_benchmark.integrations.n0_twam.retrained_live import (
                validate_server_receipt,
            )

            prepared = load_retrained(Path(binding["artifact"]))
            validate_server_receipt(prepared, server["task"], server, binding["port"])
        inference_ready = True
        native_path = root / "curobo_gpu_probe.json"
        native = read(native_path, "native_runtime")
        require(
            native.get("status") == "passed"
            and native.get("hostname") == host
            and native.get("capability") == [8, 0]
            and native.get("warmup_completed") is True
            and native.get("joint_space_plan_success") is True,
            "native_runtime",
            "same-host SM80 cuRobo GPU plan is unverified",
            native_path,
        )
        from robotactile_benchmark.backends.univtac_contracts import (
            build_univtac_backend_config,
        )
        from robotactile_benchmark.execution.live_artifacts import (
            load_live_univtac_artifact,
        )

        if model in ALIASES:
            audit_path = root / "adopted_audit" / f"{model}.json"
            adopted = read(audit_path, "closed_loop")
            checked = adopted.get("audit", adopted)
            require(
                checked.get("passed") is True
                and inference.get("audit", {}).get("passed") is True,
                "closed_loop",
                "adopted historical audit failed",
                audit_path,
            )
            group = Path(execution["adopted_group"])
            artifact = group / "artifacts/clean"
            launch_path = Path(
                execution.get("historical_launch", str(group / "launch.json"))
            )
            launch = adopted.get("launch") or read(launch_path, "historical_host")
            require(
                launch.get("hostname") == host,
                "historical_host",
                "historical episode hostname mismatch",
                launch_path,
            )
        else:
            require(
                execution.get("full_episode_returncode") in {None, 0},
                "full_episode_interrupted",
                "full episode failed or was interrupted; do not fall back to smoke",
                target / "full_episode.log",
            )
            require(
                execution.get("full_episode_returncode") == 0,
                "closed_loop",
                "full episode absent/failed; 3-cycle smoke cannot qualify",
                target / "full_episode.log",
            )
            episode = target / "completed_episode"
            acceptance_path = episode / "acceptance.json"
            if acceptance_path.is_file():
                acceptance = read(acceptance_path, "closed_loop")
                require(
                    acceptance.get("runtime_acceptance_passed") is True,
                    "closed_loop",
                    "full episode runtime acceptance failed",
                    acceptance_path,
                )
            row["episode_receipt_provenance"] = {
                "original_acceptance_path": str(acceptance_path),
                "original_acceptance_present": acceptance_path.is_file(),
                "validation_scope": "independent_report_time_strict_artifact_and_plan_revalidation",
                "historical_acceptance_recreated": False,
            }
            artifact = episode / "live_artifact"
            launch_path = episode / "launch.json"
            launch = read(launch_path, "closed_loop")
            require(
                launch.get("hostname") == host,
                "closed_loop",
                "full episode host mismatch",
                launch_path,
            )
        bundle = load_live_univtac_artifact(artifact)
        if model not in ALIASES:
            row["inference_trace"] = verify_plans(episode, bundle)
            from robotactile_benchmark.execution.loading import (
                load_live_univtac_request,
                load_live_univtac_run,
            )

            loaded = load_live_univtac_run(
                load_live_univtac_request(episode / "request.json")
            )
            require(
                loaded.trial == bundle.trial and loaded.run_spec == bundle.run_spec,
                "closed_loop",
                "completed episode request does not match artifact",
                episode / "request.json",
            )
        horizon = build_univtac_backend_config(bundle.trial.task).task.action_horizon
        require(
            completed(bundle, horizon),
            "full_episode_incomplete",
            "episode neither succeeded nor executed the complete task horizon",
            artifact,
        )
        expected_checkpoint = row["weights"]["checkpoint_sha256"]
        require(
            bundle.request_identity["checkpoint_sha256"] == expected_checkpoint,
            "closed_loop",
            "closed-loop checkpoint differs from verified weight",
            artifact,
        )
        if model != "act":
            require(
                bundle.request_identity.get("n0_source_commit")
                == binding["source_commit"],
                "source",
                "historical source commit differs from bound source",
                artifact,
            )
            if model != "n0_twam":
                require(
                    bundle.request_identity.get("n0_normalizer_sha256")
                    == binding["normalizer_sha256"],
                    "source",
                    "historical normalizer differs from bound normalizer",
                    artifact,
                )
            else:
                require(
                    bundle.request_identity.get("n0_normalizer_sha256")
                    == prepared["tasks"][bundle.trial.task]["normalizer_sha256"]
                    and bundle.request_identity.get("n0_serve_bundle_sha256")
                    == prepared["artifact_sha256"],
                    "source",
                    "N0 historical task normalizer/prepared artifact mismatch",
                    artifact,
                )
        row["closed_loop"] = {
            "artifact": str(artifact),
            "root_receipt_sha256": bundle.root_receipt_sha256,
            "launch": str(launch_path),
            "hostname": launch["hostname"],
            "task": bundle.trial.task,
            "task_action_horizon": horizon,
            "executed_actions": sum(
                x.executed_actions.shape[0] for x in bundle.evidence.action_entries
            ),
            "termination": bundle.evidence.result.terminal_status.value,
            "success": bundle.evidence.result.score_success,
            "source_commit": bundle.request_identity.get("n0_source_commit"),
            "normalizer_sha256": bundle.request_identity.get("n0_normalizer_sha256"),
            "normalizer_scope": "task-normalizer"
            if model == "n0_twam"
            else "binding-normalizer",
        }
        row["status"] = "READY_CLOSED_LOOP"
    except GateFailure as error:
        row["status"] = (
            "READY_INFERENCE_ONLY"
            if inference_ready
            and error.gate
            in {
                "closed_loop",
                "historical_host",
                "native_runtime",
                "full_episode_interrupted",
                "full_episode_incomplete",
            }
            else "BLOCKED"
        )
        row["gap"] = {"gate": error.gate, "detail": str(error), "path": error.path}
    except Exception as error:
        row["status"] = "READY_INFERENCE_ONLY" if inference_ready else "BLOCKED"
        row["gap"] = {
            "gate": "evidence_validation",
            "detail": f"{type(error).__name__}: {error}",
            "path": str(target),
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    root, output = args.root.absolute(), args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    host = read(root / "host_inventory.json", "host")["hostname"]
    require(
        host == socket.gethostname(),
        "host",
        "report must run on the inventoried host",
        root / "host_inventory.json",
    )
    modules = module_probe(root)
    source_manifest = root / "deployment/execution_source_manifest.json"
    modules["execution_source_manifest"] = {
        "path": str(source_manifest),
        "sha256": sha(source_manifest) if source_manifest.is_file() else None,
        "scope": "separate_snapshot_manifest_not_historical_introspection",
    }
    rows = [matrix_row(root, model, host, modules) for model in MODELS]
    report = {
        "schema": "robotactile-five-model-acceptance-report-v1",
        "hostname": host,
        "root": str(root),
        "module_resolution": modules,
        "models": rows,
        "all_ready_closed_loop": all(
            row["status"] == "READY_CLOSED_LOOP" for row in rows
        ),
        "scope": "runtime_and_closed_loop_acceptance_not_success_rate_or_real_robot_evaluation",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# A800 five-model acceptance",
        "",
        "| Model | Status | Checkpoint SHA256 | Runtime version / Python path | Fresh shape / seconds | Closed-loop artifact | Unique gap |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        fresh, closed, gap = (
            row.get("fresh_inference", {}),
            row.get("closed_loop", {}),
            row["gap"],
        )
        detail = (
            "—" if gap is None else f"{gap['gate']}: {gap['detail']} ({gap['path']})"
        )
        checkpoint = row.get("weights", {}).get("checkpoint_sha256", "—")
        if row["model"] == "dream_tac":
            checkpoint = f"DCP-table: {checkpoint}"
        runtime = row.get("runtime", {})
        python_version = runtime.get("python", "—").splitlines()[0].replace("|", "/")
        runtime_label = (
            f"Python {python_version}; torch {runtime.get('torch', '—')}; "
            f"CUDA {runtime.get('torch_cuda', '—')}<br>{runtime.get('executable', '—')}"
        )
        lines.append(
            f"| {row['model']} | {row['status']} | {checkpoint} | {runtime_label} | "
            f"{fresh.get('action', {}).get('shape', '—')} / {fresh.get('duration_s', '—')} | "
            f"{closed.get('artifact', '—')} | {detail.replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "Full hashes, member tables, runtime versions and module paths: [report.json](report.json).",
            "Module resolution is an independent report-time probe, not historical process introspection.",
            "Task timeout is not task success. Three-cycle smoke does not qualify as a completed episode.",
        ]
    )
    (output / "matrix.md").write_text("\n".join(lines) + "\n")
    sys.exit(0 if report["all_ready_closed_loop"] else 1)


if __name__ == "__main__":
    main()
