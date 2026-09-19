"""Contracts for deterministic Dream-Tac post-normalization statistics."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from scripts.dream_tac.training import materialize, training_preflight
from scripts.dream_tac.training.contracts import build_prompt_manifest
from scripts.dream_tac.training.identity import signed_payload
from scripts.dream_tac.training.materialize import (
    verify_materialization,
    write_statistics_artifacts,
)
from scripts.dream_tac.training.statistics import (
    derive_post_normalization_statistics,
    validate_dataset_statistics,
    validate_post_normalization_statistics,
)
from scripts.dream_tac.training.training_constants import (
    DATASET_DIR_NAME,
    DATASET_POST_NORM_STATS_NAME,
    DATASET_STATS_NAME,
    DREAM_TAC_UPSTREAM_COMMIT,
    GLOBAL_RECEIPT_NAME,
)
from scripts.dream_tac.training.training_preflight import build_preflight_receipt
from scripts.dream_tac.training.training_request import (
    DreamTacTrainingRequest,
    sha256_regular_file,
)


def _source_statistics() -> dict[str, object]:
    return {
        "actions_min": [0.0] * 7,
        "actions_max": [4.0] * 7,
        "actions_mean": [1.0] * 7,
        "actions_std": [2.0] * 7,
        "actions_median": [3.0] * 7,
        "proprio_min": [-2.0] * 6,
        "proprio_max": [2.0] * 6,
        "proprio_mean": [0.0] * 6,
        "proprio_std": [1.0] * 6,
        "proprio_median": [-1.0] * 6,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _install_preflight_artifacts(
    tmp_path: Path,
) -> tuple[DreamTacTrainingRequest, Path]:
    root = tmp_path / "materialized"
    dataset_root = root / DATASET_DIR_NAME
    (dataset_root / "train").mkdir(parents=True)
    stats_path, post_path = write_statistics_artifacts(
        output_root=root, statistics=_source_statistics()
    )
    source = signed_payload(
        {"schema_version": 1, "episodes": []}, field="manifest_sha256"
    )
    _write_json(root / "source_split_manifest.json", source)
    source_sha = cast(str, source["manifest_sha256"])
    cache_keys = [f"prompt-{index}" for index in range(8)]
    prompts = signed_payload(
        {
            "source_manifest_sha256": source_sha,
            "task_count": 8,
            "prompt_count": 8,
            "t5_cache_keys": cache_keys,
        },
        field="prompt_manifest_sha256",
    )
    _write_json(root / "prompt_manifest.json", prompts)
    t5_request = signed_payload(
        {
            "dream_tac_commit": DREAM_TAC_UPSTREAM_COMMIT,
            "prompt_manifest_sha256": prompts["prompt_manifest_sha256"],
            "expected_cache_keys": cache_keys,
        },
        field="t5_request_sha256",
    )
    _write_json(root / "t5_cache_request.json", t5_request)
    t5_path = dataset_root / "t5_embeddings.pkl"
    t5_path.write_bytes(b"deterministic-t5-cache")
    t5_sha = sha256_regular_file(t5_path)
    t5_receipt = signed_payload(
        {
            "status": "complete",
            "dream_tac_commit": DREAM_TAC_UPSTREAM_COMMIT,
            "prompt_manifest_sha256": prompts["prompt_manifest_sha256"],
            "t5_request_sha256": t5_request["t5_request_sha256"],
            "cache_sha256": t5_sha,
            "cache_keys": cache_keys,
        },
        field="t5_cache_receipt_sha256",
    )
    _write_json(root / "t5_cache_receipt.json", t5_receipt)
    receipt = signed_payload(
        {
            "status": "complete",
            "source_manifest_sha256": source_sha,
            "materialized_episode_count": 759,
            "training_ready": False,
            "t5_cache_status": "external_generation_required",
            "dataset_statistics_sha256": sha256_regular_file(stats_path),
        },
        field="receipt_sha256",
    )
    _write_json(root / GLOBAL_RECEIPT_NAME, receipt)
    request = DreamTacTrainingRequest.from_dict(
        {
            "schema_version": "robotactile-dream-tac-training-request-v1",
            "accelerator_contract": "nvidia_cuda_only_v1",
            "dream_tac_commit": DREAM_TAC_UPSTREAM_COMMIT,
            "donor_experiment": ("cosmos_predict2_2b_480p_franka_cut_banana_20260321"),
            "phase": "p2_micro",
            "run_name": "post-norm-test",
            "dream_tac_root": str(tmp_path / "source"),
            "python_executable": str(tmp_path / "runtime" / "python"),
            "materialization_root": str(root),
            "output_root": str(tmp_path / "output"),
            "source_manifest_sha256": source_sha,
            "materialization_receipt_sha256": receipt["receipt_sha256"],
            "t5_cache_sha256": t5_sha,
            "base_checkpoint": {
                "path": str(tmp_path / "base.pt"),
                "sha256": "4" * 64,
            },
            "resume_checkpoint": None,
            "cuda_devices": list(range(8)),
            "nproc_per_node": 8,
            "master_port": 12341,
            "max_iter": 1,
            "save_iter": 1,
            "batch_size": 1,
            "num_workers": 0,
        }
    )
    return request, post_path


def test_derives_exact_minus_one_plus_one_statistics() -> None:
    post = derive_post_normalization_statistics(_source_statistics())

    assert post["actions_min"] == [-1.0] * 7
    assert post["actions_max"] == [1.0] * 7
    assert post["actions_mean"] == [-0.5] * 7
    assert post["actions_std"] == [1.0] * 7
    assert post["actions_median"] == [0.5] * 7
    assert post["proprio_mean"] == [0.0] * 6
    assert post["proprio_std"] == [0.5] * 6
    assert post["proprio_median"] == [-0.5] * 6
    assert validate_post_normalization_statistics(_source_statistics(), post) == post


def test_rejects_invalid_source_statistics() -> None:
    for failure in ("dimension", "finite", "range", "fields"):
        payload = copy.deepcopy(_source_statistics())
        if failure == "dimension":
            cast(list[float], payload["actions_min"]).pop()
        elif failure == "finite":
            cast(list[float], payload["actions_mean"])[0] = float("nan")
        elif failure == "range":
            cast(list[float], payload["actions_max"])[0] = 0.0
        else:
            payload["unexpected"] = [0.0]

        with pytest.raises(ValueError):
            validate_dataset_statistics(payload)


def test_statistics_artifacts_are_deterministic_and_no_clobber(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_paths = write_statistics_artifacts(
        output_root=first, statistics=_source_statistics()
    )
    write_statistics_artifacts(output_root=first, statistics=_source_statistics())
    second_paths = write_statistics_artifacts(
        output_root=second, statistics=_source_statistics()
    )

    assert [path.read_bytes() for path in first_paths] == [
        path.read_bytes() for path in second_paths
    ]
    tampered = derive_post_normalization_statistics(_source_statistics())
    cast(list[float], tampered["actions_mean"])[0] += 0.01
    _write_json(first_paths[1], tampered)
    with pytest.raises(FileExistsError, match="existing artifact differs"):
        write_statistics_artifacts(output_root=first, statistics=_source_statistics())


def test_preflight_requires_valid_post_statistics_and_records_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, post_path = _install_preflight_artifacts(tmp_path)
    monkeypatch.setattr(
        training_preflight, "_source_checkout", lambda _request: {"status": "stub"}
    )
    monkeypatch.setattr(
        training_preflight,
        "verify_bound_file",
        lambda artifact, _name: artifact.path,
    )
    receipt = build_preflight_receipt(
        request, environment_probe=lambda _request: {"status": "stub"}
    )

    data = cast(dict[str, object], receipt["data"])
    assert data["dataset_statistics_post_norm_sha256"] == sha256_regular_file(post_path)
    tampered = derive_post_normalization_statistics(_source_statistics())
    cast(list[float], tampered["proprio_mean"])[0] += 0.01
    _write_json(post_path, tampered)
    with pytest.raises(ValueError, match="post-normalization"):
        training_preflight._data_artifacts(request)

    post_path.unlink()
    with pytest.raises(FileNotFoundError):
        training_preflight._data_artifacts(request)


def test_read_only_verification_requires_post_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    raw = tmp_path / "raw"
    raw.mkdir()
    stats_path, post_path = write_statistics_artifacts(
        output_root=output, statistics=_source_statistics()
    )
    manifest_sha = "a" * 64
    _write_json(output / "source_split_manifest.json", {})
    _write_json(output / "prompt_manifest.json", build_prompt_manifest(manifest_sha))
    _write_json(output / "t5_cache_request.json", {})
    receipt = signed_payload(
        {
            "materialized_episode_count": 759,
            "dataset_statistics_sha256": sha256_regular_file(stats_path),
        },
        field="receipt_sha256",
    )
    _write_json(output / GLOBAL_RECEIPT_NAME, receipt)
    records = tuple(
        SimpleNamespace(task="insert_hole", episode_id=index) for index in range(759)
    )
    monkeypatch.setattr(
        materialize,
        "load_or_create_source_manifest",
        lambda **_kwargs: ({"manifest_sha256": manifest_sha}, records),
    )
    monkeypatch.setattr(materialize, "train759_records", lambda items: items)
    monkeypatch.setattr(
        materialize, "validate_episode_destination", lambda **_kwargs: {}
    )

    verified = verify_materialization(
        raw_root=raw,
        output_root=output,
        artifact_verifier=lambda *_args, **_kwargs: None,
    )
    assert verified["dataset_statistics_post_norm_sha256"] == (
        sha256_regular_file(post_path)
    )
    post_path.unlink()
    with pytest.raises(FileNotFoundError):
        verify_materialization(
            raw_root=raw,
            output_root=output,
            artifact_verifier=lambda *_args, **_kwargs: None,
        )


def test_post_statistics_filename_matches_upstream_contract() -> None:
    assert DATASET_POST_NORM_STATS_NAME == ("dataset_statistics_post_norm_franka.json")
    assert DATASET_STATS_NAME == "dataset_statistics_franka.json"
