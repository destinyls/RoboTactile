"""Frozen seed-3 binding variants preserve each model's transport identity."""

from __future__ import annotations

from robotactile_benchmark.contracts import canonical_hash
from scripts.retrained_evaluation.dream_seed3_completion import (
    extend_dream_binding,
)
from scripts.retrained_evaluation.seed3_three_models import (
    AVAILABILITY_OPERATORS,
    OPERATORS,
    make_binding,
)


def source(model: str) -> dict[str, object]:
    tasks: dict[str, dict[str, object]] = {
        "grasp_classify": {"prompt": "grasp"},
        "lift_can": {"prompt": "lift"},
    }
    if model == "ftp1_policy":
        for task in tasks.values():
            task["transport_metadata"] = {
                "serve_bundle_sha256": "a" * 64,
                "model": "ftp1_policy",
            }
    return {
        "model": model,
        "tasks": tasks,
        "binding_sha256": "a" * 64,
        "evaluation": {"reset_time_limit_s": 600.0},
    }


def test_native_and_zero_fill_have_separate_controls() -> None:
    original = source("dream_tac")
    native = make_binding(original, "native")
    zero = make_binding(original, "zero_fill_v1")
    assert native["evaluation"]["operators"] == list(OPERATORS)
    assert zero["evaluation"]["operators"] == list(AVAILABILITY_OPERATORS)
    assert zero["evaluation"]["a2_end_policy"] == "episode_censored_v1"
    assert zero["evaluation"]["tactile_zero_shape"] == [240, 320, 3]
    assert native["binding_sha256"] != zero["binding_sha256"]
    assert original["binding_sha256"] == "a" * 64


def test_ftp_metadata_uses_new_binding_identity() -> None:
    original = source("ftp1_policy")
    binding = make_binding(original, "native")
    for task in binding["tasks"].values():
        assert (
            task["transport_metadata"]["serve_bundle_sha256"]
            == binding["binding_sha256"]
        )
    before_metadata = dict(binding)
    before_metadata["tasks"] = {
        name: {key: value for key, value in task.items() if key != "transport_metadata"}
        for name, task in binding["tasks"].items()
    }
    before_metadata.pop("binding_sha256")
    assert binding["binding_sha256"] == canonical_hash(before_metadata)
    assert (
        original["tasks"]["grasp_classify"]["transport_metadata"]["serve_bundle_sha256"]
        == "a" * 64
    )


def test_dream_can_extend_only_the_training_prompt_in_t5_cache() -> None:
    dream = source("dream_tac")
    dream["tasks"].pop("lift_can")
    vtla = source("n0_vtla")
    vtla["tasks"]["lift_can"]["prompt"] = "Rotate a lying can so it stands upright"
    enriched = extend_dream_binding(dream, vtla)
    assert enriched["tasks"]["lift_can"]["prompt"] == (
        "Rotate a lying can so it stands upright"
    )
    assert "lift_can" not in dream["tasks"]
