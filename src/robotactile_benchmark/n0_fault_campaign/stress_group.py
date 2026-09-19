"""One Clean plus mixed-registry faults in one exact-reset Isaac session."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from robotactile_benchmark.backends.univtac_contracts import (
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
)
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.closed_loop.artifact_values import result_to_dict
from robotactile_benchmark.constants import operator_requires_rest_reference
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.live_artifacts import load_live_univtac_artifact
from robotactile_benchmark.execution.loading import load_live_univtac_request
from robotactile_benchmark.execution.official_n0 import (
    OfficialN0PairedLiveResult,
    execute_official_n0_paired_live_runs,
)
from robotactile_benchmark.execution.paired_receipt_io import (
    write_paired_execution_receipt,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_n0_runtime_artifacts,
)
from robotactile_benchmark.trials import Condition

from .generation import generate_n0_fault_campaign_bundle
from .generation_contracts import N0FaultCampaignGenerationSpec
from .io import file_sha256, load_n0_fault_campaign_bundle
from .stress_metrics import eligible_terminal
from .stress_protocol import StressVariant, validate_protocol
from .stress_provenance import verify_runtime_code


def write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def _verify_group_path(plan: dict[str, Any], seed: int, root: Path) -> None:
    if "group_paths" in plan and str(root.resolve()) != plan["group_paths"][str(seed)]:
        raise ValueError("group root differs from the unique frozen execution path")


def _verify_imported_runtime(plan: dict[str, Any]) -> None:
    code = os.environ.get("ROBOTACTILE_REPOSITORY_ROOT")
    if not code:
        raise ValueError(
            "ROBOTACTILE_REPOSITORY_ROOT is required to verify frozen code"
        )
    package = Path(__file__).resolve().parents[2]
    declared = os.environ.get("ROBOTACTILE_PACKAGE_PATH")
    if declared and Path(declared).resolve() != package:
        raise ValueError("declared package differs from the imported runtime")
    verify_runtime_code(
        Path(code).resolve(strict=True), package, plan["binding"]["code_sha256"]
    )


def prepare_stress_group(
    root: Path,
    *,
    protocol: dict[str, Any],
    clean_request: Path,
    rest_reference: Path | None,
    spatial_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = validate_protocol(protocol)
    base = load_live_univtac_request(clean_request)
    if (
        base.task_id != plan["task"]
        or base.initial_seed not in plan["seeds"]
        or base.exogenous_seed != base.initial_seed
    ):
        raise ValueError("Clean source does not belong to protocol task/seed cohort")
    if (
        base.dataset_sha256 != plan["binding"]["dataset_sha256"]
        or base.checkpoint_sha256 != plan["binding"]["model_sha256"]
    ):
        raise ValueError("Clean source dataset/checkpoint binding mismatch")
    if base.condition is not Condition.CLEAN:
        raise ValueError("base request must be Clean")
    _verify_group_path(plan, base.initial_seed, root)
    if (
        spatial_calibration is not None
        and canonical_hash(spatial_calibration) != plan["spatial_calibration_sha256"]
    ):
        raise ValueError("spatial calibration differs from frozen protocol")
    root.mkdir(parents=True, exist_ok=False)
    write_once(root / "protocol.json", plan)
    selected: list[dict[str, Any]] = []
    variants = [StressVariant(v["family"], v["level"]) for v in plan["variants"]]
    rotation = base.initial_seed % len(variants)
    variants = variants[rotation:] + variants[:rotation]
    for variant in variants:
        if variant.family == "contact_f3" and spatial_calibration is None:
            raise ValueError("frozen contact template is required")
        needs_rest = operator_requires_rest_reference(
            variant.operator, severity_registry=variant.registry
        )
        if needs_rest and rest_reference is None:
            raise ValueError("profile requires a verified rest-reference artifact")
        spec = N0FaultCampaignGenerationSpec(
            campaign_id=f"stress-{variant.label}-{base.initial_seed}",
            base_clean_request_paths=(clean_request,),
            operator_ids=(variant.operator,),
            severity_levels=(variant.level,),
            operator_seed_master=20260919,
            fault_start_index=0 if variant.family == "null" else 5,
            fault_stop_index=base.max_observation_steps,
            rest_reference_artifacts={base.task_id: rest_reference}
            if needs_rest and rest_reference
            else {},
            severity_registry=variant.registry,
            spatial_calibration=spatial_calibration
            if variant.family == "contact_f3" and spatial_calibration
            else {},
        )
        _, bundle = generate_n0_fault_campaign_bundle(
            root / "profiles" / variant.label, spec
        )
        for cell in bundle.manifest.cells:
            is_clean = cell.condition is Condition.CLEAN
            if is_clean and selected:
                continue
            if cell.request_relpath is None:
                raise ValueError("stress group contains a non-executable contract")
            selected.append(
                {
                    "condition": "clean" if is_clean else variant.label,
                    "bundle": bundle.root.relative_to(root).as_posix(),
                    "bundle_manifest_sha256": bundle.manifest.sha256,
                    "generation_receipt_sha256": bundle.receipt_file_sha256,
                    "request": (bundle.root / cell.request_relpath)
                    .relative_to(root)
                    .as_posix(),
                    "request_file_sha256": cell.request_file_sha256,
                }
            )
    manifest = {
        "schema": "n0_noise_stress_group_v2",
        "protocol_sha256": plan["protocol_sha256"],
        "seed": base.initial_seed,
        "selected": selected,
        "unused_profile_clean_requests_executed": False,
    }
    manifest["group_sha256"] = canonical_hash(manifest)
    write_once(root / "group.json", manifest)
    return manifest


def load_stress_group(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    root = root.resolve(strict=True)
    plan = validate_protocol(json.loads((root / "protocol.json").read_text()))
    group = json.loads((root / "group.json").read_text())
    raw = {k: v for k, v in group.items() if k != "group_sha256"}
    if (
        group["group_sha256"] != canonical_hash(raw)
        or group["protocol_sha256"] != plan["protocol_sha256"]
    ):
        raise ValueError("group/protocol hash mismatch")
    if group["seed"] not in plan["seeds"]:
        raise ValueError("group seed not in frozen cohort")
    _verify_group_path(plan, group["seed"], root)
    labels = [s["condition"] for s in group["selected"]]
    expected = {"clean", *(v["label"] for v in plan["variants"])}
    if (
        not labels
        or labels[0] != "clean"
        or set(labels) != expected
        or len(labels) != len(expected)
    ):
        raise ValueError("group conditions differ from frozen protocol")
    requests = []
    for selected in group["selected"]:
        bundle_path = (root / selected["bundle"]).resolve(strict=True)
        path = (root / selected["request"]).resolve(strict=True)
        bundle_path.relative_to(root)
        path.relative_to(bundle_path)
        bundle = load_n0_fault_campaign_bundle(bundle_path)
        if (
            bundle.manifest.sha256 != selected["bundle_manifest_sha256"]
            or bundle.receipt_file_sha256 != selected["generation_receipt_sha256"]
        ):
            raise ValueError("profile bundle changed after group freeze")
        if file_sha256(path) != selected["request_file_sha256"]:
            raise ValueError("selected request hash mismatch")
        request = load_live_univtac_request(path)
        if request.initial_seed != group["seed"] or request.task_id != plan["task"]:
            raise ValueError("selected request seed/task mismatch")
        requests.append(request)
    return plan, group, requests


def run_stress_group(
    root: Path,
    *,
    integration_config: Path,
    n0_source_root: Path,
    host: str,
    port: int,
    capture_profile: LiveCaptureProfile = LiveCaptureProfile.PREVIEW,
) -> dict[str, Any]:
    plan, group, requests = load_stress_group(root)
    _verify_imported_runtime(plan)
    if file_sha256(integration_config) != plan["binding"]["integration_config_sha256"]:
        raise ValueError("integration config differs from frozen protocol")
    if (root / "group_result.json").exists():
        raise FileExistsError(
            "group result already exists; never rerun scored outcomes"
        )
    for request in requests:
        output = request.output_dir
        if (
            output is None
            or output.is_symlink()
            or (output.exists() and any(output.iterdir()))
        ):
            raise FileExistsError("group output already exists or is invalid")
    # A failed attempt cannot be silently replaced by another outcome at this path.
    write_once(
        root / "execution_started.json",
        {
            "protocol_sha256": plan["protocol_sha256"],
            "group_sha256": group["group_sha256"],
            "pid": os.getpid(),
        },
    )
    runtime = resolve_n0_runtime_artifacts(integration_config)
    if runtime.manifest.task_id != plan["task"]:
        raise ValueError("integration task mismatch")
    published: list[dict[str, Any]] = []

    def publish(result: OfficialN0PairedLiveResult) -> None:
        receipt = write_paired_execution_receipt(
            root / "paired_execution_receipt.json", result.paired
        )
        rows = []
        for selected, artifact in zip(group["selected"], result.artifacts):
            rows.append(
                {
                    **result_to_dict(artifact.evidence.result),
                    "seed": group["seed"],
                    "condition": selected["condition"],
                    "protocol_sha256": plan["protocol_sha256"],
                    "attempt_id": group["group_sha256"],
                    "artifact_root_sha256": artifact.external_root_sha256,
                    "exact_snapshot_reset_verified": result.paired.reset_receipt.all_exact,
                }
            )
        accepted = len(rows) == len(group["selected"]) and all(
            eligible_terminal(row) for row in rows
        )
        for row in rows:
            row["group_accepted"] = accepted
        payload = {
            "schema": "n0_noise_stress_group_result_v2",
            "group_sha256": group["group_sha256"],
            "protocol_sha256": plan["protocol_sha256"],
            "group_accepted": accepted,
            "paired_receipt_file_sha256": receipt.file_sha256,
            "simulator_qualification_claimed": False,
            "rows": rows,
        }
        write_once(root / "group_result.json", payload)
        published.append(payload)

    result = execute_official_n0_paired_live_runs(
        requests,
        manifest=runtime.manifest,
        source_root=n0_source_root,
        host=host,
        port=port,
        api_key=os.environ.get("N0_TWAM_API_KEY"),
        action_execution_contract=N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
        capture_profile=capture_profile,
        pre_close_publisher=publish,
    )
    if not published:
        publish(result)
    return published[0]


def _verify_stress_pairing_proof(
    receipt: dict[str, Any],
    requests: list[Any],
    artifacts: list[Any],
    backend_config_sha256: str,
) -> bool:
    """Reconstruct reset/execution links from the persisted pairing proof."""
    from robotactile_benchmark.backends.univtac_pairing import (
        UniVTACPairedResetReceipt,
        UniVTACResetWitness,
    )
    from robotactile_benchmark.execution.paired_live_univtac import (
        PAIRED_EXECUTION_EVIDENCE_LEVEL,
        PAIRED_EXECUTION_SEMANTIC_VERSION,
    )

    if (
        receipt.get("evidence_level") != PAIRED_EXECUTION_EVIDENCE_LEVEL
        or receipt.get("semantic_version") != PAIRED_EXECUTION_SEMANTIC_VERSION
        or receipt.get("simulator_qualification_claimed") is not False
    ):
        raise ValueError("paired execution proof identity mismatch")
    raw_reset = receipt["reset_receipt"]
    reset = UniVTACPairedResetReceipt(
        **{
            **raw_reset,
            "witnesses": tuple(
                UniVTACResetWitness(**w) for w in raw_reset["witnesses"]
            ),
        }
    )
    if reset.sha256 != receipt["reset_receipt_sha256"]:
        raise ValueError("paired reset content hash mismatch")
    if (
        reset.task_id != requests[0].task_id
        or reset.initial_seed != requests[0].initial_seed
        or reset.exogenous_seed != requests[0].exogenous_seed
        or reset.config_sha256 != backend_config_sha256
    ):
        raise ValueError("paired reset does not match selected requests")
    exact = reset.all_exact is True and all(
        witness.exact_match is True
        and witness.equivalence_key == reset.witnesses[0].equivalence_key
        for witness in reset.witnesses
    )
    if not exact:
        raise ValueError("paired reset witnesses are not exactly equivalent")
    executions = receipt["executions"]
    if len(executions) != len(requests) or len(artifacts) != len(requests):
        raise ValueError("paired execution proof coverage mismatch")
    indices = []
    for execution, request, artifact in zip(executions, requests, artifacts):
        result = artifact.evidence.result
        index = execution["witness_index"]
        if (
            type(index) is not int
            or not 0 <= index < len(reset.witnesses)
            or index in indices
        ):
            raise ValueError("invalid or reused reset witness index")
        indices.append(index)
        if (
            execution["condition"] != request.condition.value
            or execution["run_content_sha256"] != artifact.run_content_sha256
            or execution["result_sha256"] != result.sha256
            or execution["terminal_status"] != result.terminal_status.value
            or execution["validation_passed"] is not result.validation_passed
            or result.initial_state_sha256
            != reset.witnesses[index].simulator_state_sha256
        ):
            raise ValueError("paired execution/result/reset witness binding mismatch")
    expected_hash = canonical_hash(
        {
            "namespace": PAIRED_EXECUTION_EVIDENCE_LEVEL,
            "run_content_sha256": tuple(a.run_content_sha256 for a in artifacts),
            "result_sha256": tuple(a.evidence.result.sha256 for a in artifacts),
            "reset_receipt_sha256": reset.sha256,
            "witness_indices": tuple(indices),
        }
    )
    if receipt["group_content_sha256"] != expected_hash:
        raise ValueError("paired execution group content hash mismatch")
    return exact


def read_stress_rows(root: Path) -> list[dict[str, Any]]:
    """Strictly revalidate persisted artifacts before using accepted/provisional rows."""
    plan, group, requests = load_stress_group(root)
    result_path = root / "group_result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    if result is not None:
        if (
            result["group_sha256"] != group["group_sha256"]
            or result["protocol_sha256"] != plan["protocol_sha256"]
        ):
            raise ValueError("persisted result group/protocol mismatch")
        receipt_path = root / "paired_execution_receipt.json"
        if file_sha256(receipt_path) != result["paired_receipt_file_sha256"]:
            raise ValueError("paired receipt changed")
        if len(result["rows"]) != len(requests):
            raise ValueError("persisted result lost group rows")
    rows = []
    artifacts = []
    backend_config_sha256 = None
    for index, (selected, request) in enumerate(zip(group["selected"], requests)):
        assert request.output_dir is not None
        if not (request.output_dir / "root_receipt.json").exists():
            if result is not None:
                raise ValueError("accepted result lost a referenced artifact")
            continue
        artifact = load_live_univtac_artifact(request.output_dir)
        from robotactile_benchmark.contracts import thaw_value
        from robotactile_benchmark.execution.live_artifacts_values import (
            live_request_identity,
        )
        from robotactile_benchmark.execution.loading import load_live_univtac_run

        loaded = load_live_univtac_run(request)
        if (
            thaw_value(artifact.request_identity) != live_request_identity(loaded)
            or artifact.run_content_sha256 != loaded.content_sha256
            or artifact.trial != loaded.trial
        ):
            raise ValueError(
                "live artifact differs from selected request identity/content"
            )
        if backend_config_sha256 is None:
            backend_config_sha256 = loaded.backend_config.sha256
        elif backend_config_sha256 != loaded.backend_config.sha256:
            raise ValueError("paired requests have different backend configurations")
        artifacts.append(artifact)
        terminal = result_to_dict(artifact.evidence.result)
        row = {
            **terminal,
            "seed": group["seed"],
            "condition": selected["condition"],
            "protocol_sha256": plan["protocol_sha256"],
            "attempt_id": group["group_sha256"],
            "artifact_root_sha256": artifact.external_root_sha256,
            "group_accepted": False,
            "exact_snapshot_reset_verified": False,
        }
        if result is not None:
            saved = result["rows"][index]
            if any(
                saved.get(key) != val
                for key, val in row.items()
                if key not in {"group_accepted", "exact_snapshot_reset_verified"}
            ):
                raise ValueError("persisted row differs from revalidated live artifact")
        rows.append(row)
    if result is not None:
        if backend_config_sha256 is None:
            raise ValueError("paired result has no verified backend configuration")
        exact = _verify_stress_pairing_proof(
            json.loads((root / "paired_execution_receipt.json").read_text()),
            requests,
            artifacts,
            backend_config_sha256,
        )
        accepted = (
            len(rows) == len(requests)
            and exact
            and all(eligible_terminal(row) for row in rows)
        )
        if result.get("group_accepted") is not accepted:
            raise ValueError("persisted group acceptance differs from verified proof")
        for row, saved in zip(rows, result["rows"]):
            if (
                saved.get("group_accepted") is not accepted
                or saved.get("exact_snapshot_reset_verified") is not exact
            ):
                raise ValueError(
                    "persisted row flags differ from verified pairing proof"
                )
            row.update(group_accepted=accepted, exact_snapshot_reset_verified=exact)
    for row in rows:
        row["group_root"] = str(root.resolve())
    return rows
