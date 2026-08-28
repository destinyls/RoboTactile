"""Clean campaign manifest, discovery, and no-clobber tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_clean_campaign_support import digest, make_layout, write_clean_request

from robotactile_benchmark.backends.univtac_registry import load_registry
from robotactile_benchmark.clean_baseline import (
    CLEAN_POLICY_SEED_ROLE,
    CLEAN_SEED_DERIVATION,
    CLEAN_SIMULATOR_SEED_ROLE,
    CleanCampaignError,
    CleanCampaignManifest,
    CleanCampaignProtocol,
    CleanCampaignTrialSpec,
    build_clean_campaign_manifest,
    discover_clean_request_paths,
    load_clean_campaign_manifest,
    write_clean_campaign_manifest,
)
from robotactile_benchmark.clean_baseline.contracts import (
    CLEAN_CAMPAIGN_SAMPLING_SEMANTIC_VERSION,
    CLEAN_UNIVTAC_SEED_DERIVATION,
    CleanCampaignSamplingSpec,
)
from robotactile_benchmark.clean_baseline.seeds import (
    derive_clean_campaign_seed,
    derive_univtac_task_seed,
    expected_clean_seed_pairs,
    univtac_task_seed_start,
)
from robotactile_benchmark.execution import LivePolicyKind


def _synthetic_trials(
    per_task: int,
    protocol_id: str = "paper_v1",
    official_eval_seed: int | None = None,
) -> tuple[CleanCampaignTrialSpec, ...]:
    trials = []
    ordinal = 0
    for task in load_registry().tasks:
        for index in range(per_task):
            label = f"{task.task_id}-{index}"
            official_seed = (
                None
                if official_eval_seed is None
                else derive_univtac_task_seed(
                    eval_seed=official_eval_seed,
                    candidate_ordinal=index,
                )
            )
            trials.append(
                CleanCampaignTrialSpec(
                    ordinal=ordinal,
                    task=task.task_id,
                    initial_seed=(
                        official_seed
                        if official_seed is not None
                        else derive_clean_campaign_seed(
                            protocol_id=protocol_id,
                            master_seed=20260823,
                            task_id=task.task_id,
                            task_ordinal=index,
                            role="initial",
                        )
                    ),
                    exogenous_seed=(
                        official_seed
                        if official_seed is not None
                        else derive_clean_campaign_seed(
                            protocol_id=protocol_id,
                            master_seed=20260823,
                            task_id=task.task_id,
                            task_ordinal=index,
                            role="exogenous",
                        )
                    ),
                    base_system_id="paper-policy-v1",
                    dataset_sha256=digest("dataset"),
                    base_system_manifest_sha256=digest("base-system"),
                    checkpoint_sha256=digest("checkpoint"),
                    config_sha256=digest("config"),
                    trial_manifest_sha256=digest(f"trial-{label}"),
                    pair_key=digest(f"pair-{label}"),
                    run_spec_sha256=digest(f"spec-{label}"),
                    run_content_sha256=digest(f"content-{label}"),
                    request_file_sha256=digest(f"request-{label}"),
                    request_relpath=f"requests/paper/{task.task_id}/{index}/request.json",
                    artifact_relpath=f"artifacts/paper/{task.task_id}/{index}",
                    max_control_cycles=task.action_horizon,
                    max_observation_steps=task.action_horizon + 1,
                    execute_action_steps=1,
                )
            )
            ordinal += 1
    return tuple(trials)


def _manifest(
    protocol: CleanCampaignProtocol,
    trials: tuple[CleanCampaignTrialSpec, ...],
    campaign_id: str = "campaign-test",
    sampling: CleanCampaignSamplingSpec | None = None,
) -> CleanCampaignManifest:
    return CleanCampaignManifest(
        campaign_id=campaign_id,
        protocol_id=protocol,
        policy_kind=LivePolicyKind.ACT,
        task_registry_sha256=load_registry().resource_sha256,
        master_seed=(20260823 if sampling is None else sampling.official_eval_seed),
        seed_derivation=(
            CLEAN_SEED_DERIVATION if sampling is None else sampling.seed_protocol
        ),
        simulator_seed_role=CLEAN_SIMULATOR_SEED_ROLE,
        policy_seed_role=CLEAN_POLICY_SEED_ROLE,
        planned_trial_count=len(trials),
        confidence_level=0.95,
        bootstrap_resamples=100,
        bootstrap_seed=29,
        trials=trials,
        semantic_version=(
            "1.0" if sampling is None else CLEAN_CAMPAIGN_SAMPLING_SEMANTIC_VERSION
        ),
        sampling=sampling,
    )


def _official_sampling(
    *, target: int = 100, candidate: int = 102, eval_seed: int = 0
) -> CleanCampaignSamplingSpec:
    return CleanCampaignSamplingSpec(
        seed_protocol=CLEAN_UNIVTAC_SEED_DERIVATION,
        exception_handling="replace_exception_until_target_valid_v1",
        target_valid_trials_per_task=target,
        candidate_trials_per_task=candidate,
        official_eval_seed=eval_seed,
        task_seed_start=univtac_task_seed_start(eval_seed),
        policy_seed_mode="same_as_task_seed_v1",
    )


def test_build_discovers_canonical_requests_and_preserves_seed_roles(
    tmp_path: Path,
) -> None:
    layout = make_layout(tmp_path)
    first, _ = write_clean_request(layout, ordinal=1)
    second, _ = write_clean_request(layout, ordinal=0)
    (first.parent / "trial_set_manifest.json").write_text("ignored", encoding="utf-8")

    discovered = discover_clean_request_paths(first.parents[3])
    manifest = build_clean_campaign_manifest(
        deployment_root=layout.root,
        request_paths=discovered,
        campaign_id="campaign-test",
        protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
        master_seed=20260823,
        bootstrap_resamples=100,
    )

    assert set(discovered) == {first, second}
    assert manifest.master_seed == 20260823
    assert manifest.seed_derivation == CLEAN_SEED_DERIVATION
    assert manifest.simulator_seed_role == "initial_seed"
    assert manifest.policy_seed_role == "exogenous_seed"
    assert {
        (item.initial_seed, item.exogenous_seed) for item in manifest.trials
    } == expected_clean_seed_pairs(
        protocol_id="diagnostic_v1",
        master_seed=20260823,
        task_id="pull_out_key",
        trial_count=2,
    )
    assert manifest.protocol_allows_paper_claim is False
    with pytest.raises(CleanCampaignError, match="master seed derivation"):
        build_clean_campaign_manifest(
            deployment_root=layout.root,
            request_paths=discovered,
            campaign_id="wrong-master",
            protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
            master_seed=1,
        )


def test_build_v2_manifest_from_task_local_candidate_requests(tmp_path: Path) -> None:
    layout = make_layout(tmp_path)
    paths = tuple(
        write_clean_request(
            layout,
            ordinal=ordinal,
            initial_seed=1_000_000 + ordinal,
            exogenous_seed=1_000_000 + ordinal,
        )[0]
        for ordinal in range(2)
    )
    sampling = _official_sampling(target=1, candidate=2)

    manifest = build_clean_campaign_manifest(
        deployment_root=layout.root,
        request_paths=paths,
        campaign_id="official-diagnostic",
        protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
        master_seed=0,
        bootstrap_resamples=100,
        sampling=sampling,
    )

    assert manifest.semantic_version == "2.0"
    assert manifest.sampling == sampling
    assert manifest.seed_derivation == CLEAN_UNIVTAC_SEED_DERIVATION
    assert manifest.planned_trial_count == 2


def test_discovery_and_build_reject_symlink_and_duplicate_input(tmp_path: Path) -> None:
    layout = make_layout(tmp_path)
    request, _ = write_clean_request(layout)
    link = request.parent / "linked.json"
    link.symlink_to(request)
    with pytest.raises(CleanCampaignError, match="symlink"):
        discover_clean_request_paths(request.parents[3])
    link.unlink()

    with pytest.raises(CleanCampaignError, match="duplicate explicit"):
        build_clean_campaign_manifest(
            deployment_root=layout.root,
            request_paths=(request, request),
            campaign_id="campaign-test",
            protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
            master_seed=1,
        )


def test_paper_protocol_requires_all_eight_tasks_and_one_hundred_each() -> None:
    full = _synthetic_trials(100)
    manifest = _manifest(CleanCampaignProtocol.PAPER, full)
    assert manifest.planned_trial_count == 800
    assert manifest.protocol_allows_paper_claim is False

    one_task = tuple(item for item in full if item.task == full[0].task)
    one_task = tuple(
        CleanCampaignTrialSpec.from_dict({**item.to_dict(), "ordinal": index})
        for index, item in enumerate(one_task)
    )
    with pytest.raises(CleanCampaignError, match="eight-task"):
        _manifest(CleanCampaignProtocol.PAPER, one_task)


def test_v1_manifest_bytes_fields_and_hash_remain_unchanged() -> None:
    trials = _synthetic_trials(1, protocol_id="diagnostic_v1")
    manifest = _manifest(CleanCampaignProtocol.DIAGNOSTIC, trials)

    assert manifest.sha256 == (
        "ee1acb8d574a4b9febd640d0a6955998614436294377fbb88a4c30e50fd5d8d1"
    )
    assert "sampling" not in manifest.to_dict()
    assert set(manifest.to_dict()) == {
        "bootstrap_resamples",
        "bootstrap_seed",
        "campaign_id",
        "confidence_level",
        "master_seed",
        "planned_trial_count",
        "policy_kind",
        "policy_seed_role",
        "protocol_id",
        "seed_derivation",
        "semantic_version",
        "simulator_seed_role",
        "task_registry_sha256",
        "trials",
    }
    assert CleanCampaignManifest.from_dict(manifest.to_dict()) == manifest


def test_v2_paper_sampling_freezes_target_candidates_and_task_local_seeds() -> None:
    sampling = _official_sampling()
    trials = _synthetic_trials(102, official_eval_seed=0)
    manifest = _manifest(CleanCampaignProtocol.PAPER, trials, sampling=sampling)

    assert manifest.semantic_version == "2.0"
    assert manifest.protocol_allows_paper_claim is True
    assert manifest.planned_trial_count == 8 * 102
    assert manifest.sampling == sampling
    assert manifest.to_dict()["sampling"] == sampling.to_dict()
    first_task = trials[:102]
    second_task = trials[102:204]
    assert [item.initial_seed for item in first_task] == list(
        range(1_000_000, 1_000_102)
    )
    assert [item.initial_seed for item in second_task] == list(
        range(1_000_000, 1_000_102)
    )
    assert all(item.initial_seed == item.exogenous_seed for item in trials)
    assert CleanCampaignManifest.from_dict(manifest.to_dict()) == manifest


def test_v2_paper_sampling_rejects_seed_or_candidate_drift() -> None:
    trials = _synthetic_trials(102, official_eval_seed=0)
    drifted = list(trials)
    drifted[0] = CleanCampaignTrialSpec.from_dict(
        {**drifted[0].to_dict(), "exogenous_seed": 1_000_777}
    )
    with pytest.raises(CleanCampaignError, match="task-local sampling"):
        _manifest(
            CleanCampaignProtocol.PAPER,
            tuple(drifted),
            sampling=_official_sampling(),
        )
    reordered = list(trials)
    first = reordered[0].to_dict()
    second = reordered[1].to_dict()
    for key in ("initial_seed", "exogenous_seed"):
        first[key], second[key] = second[key], first[key]
    reordered[0] = CleanCampaignTrialSpec.from_dict(first)
    reordered[1] = CleanCampaignTrialSpec.from_dict(second)
    with pytest.raises(CleanCampaignError, match="ordered.*task-local sampling"):
        _manifest(
            CleanCampaignProtocol.PAPER,
            tuple(reordered),
            sampling=_official_sampling(),
        )
    with pytest.raises(CleanCampaignError, match="equal per task"):
        _manifest(
            CleanCampaignProtocol.PAPER,
            trials,
            sampling=_official_sampling(candidate=103),
        )


def test_sampling_spec_rejects_nonofficial_exception_or_seed_mode() -> None:
    values = _official_sampling().to_dict()
    with pytest.raises(CleanCampaignError, match="replacement"):
        CleanCampaignSamplingSpec.from_dict(
            {**values, "exception_handling": "crash_counted_as_failure_v1"}
        )
    with pytest.raises(CleanCampaignError, match="official seed start"):
        CleanCampaignSamplingSpec.from_dict({**values, "task_seed_start": 999_999})


def test_manifest_write_is_idempotent_and_never_clobbers(tmp_path: Path) -> None:
    trials = _synthetic_trials(1, protocol_id="diagnostic_v1")
    manifest = _manifest(CleanCampaignProtocol.DIAGNOSTIC, trials)
    output = tmp_path / "campaign_manifest.json"

    assert write_clean_campaign_manifest(output, manifest) is True
    assert write_clean_campaign_manifest(output, manifest) is False
    assert load_clean_campaign_manifest(output) == manifest
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_clean_campaign_manifest(
            output,
            _manifest(CleanCampaignProtocol.DIAGNOSTIC, trials, "different"),
        )
