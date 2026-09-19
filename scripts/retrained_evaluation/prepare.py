"""Freeze the four user-supplied train759 checkpoints for a seeded live campaign."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.integrations.n0_twam.retrained import (
    file_sha256,
    load_retrained,
)
from scripts.retrained_evaluation.group import write_json

FOLDERS = {
    "n0_twam": (10000, "retrained-train759-vt-always-on-absee20-step10000"),
    "n0_vtla": (160000, "retrained-mixed8-train759-step160000"),
    "ftp1_policy": (159999, "retrained-joint8-train759-step159999"),
    "dream_tac": (20000, "retrained-train759-step20000"),
}


def file_table(root: Path, files: list[Path]) -> dict[str, Any]:
    if not files:
        raise ValueError(f"no checkpoint members: {root}")
    table = {}
    for path in files:
        before = path.stat()
        digest = file_sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"artifact changed during binding: {path}")
        table[str(path.relative_to(root))] = {
            "sha256": digest,
            "size_bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns,
        }
    return table


def prepare_model(
    model: str, root: Path, output: Path, n0_artifact: Path, dataset: Path
) -> Path:
    target = output / "bindings" / f"{model}.json"
    if target.exists():
        raise FileExistsError(target)
    training_step, folder = FOLDERS[model]
    checkpoint = root / "artifacts/models" / model / folder
    artifact = load_retrained(n0_artifact)
    tasks = {
        name: {"prompt": entry["prompt"]} for name, entry in artifact["tasks"].items()
    }
    binding: dict[str, Any] = {
        "schema": "robotactile-four-retrained-binding-v1",
        "model": model,
        "deployment_root": str(root),
        "training_step": training_step,
        "dataset_manifest": str(dataset),
        "dataset_sha256": file_sha256(dataset),
        "control_hz": 10,
        "tasks": tasks,
        "rest_references": {},
        "tactile_payload": "rgb" if model in ("n0_twam", "n0_vtla") else "rgb_marker",
        "port": {
            "n0_twam": 29621,
            "n0_vtla": 29622,
            "ftp1_policy": 29623,
            "dream_tac": 29624,
        }[model],
    }
    binding["endpoint"] = (
        f"{'http' if model == 'dream_tac' else 'tcp'}://127.0.0.1:{binding['port']}"
    )
    repo = root.parent
    if model == "n0_twam":
        weight = (
            checkpoint / "checkpoint/transformer/diffusion_pytorch_model.safetensors"
        )
        binding["artifact"] = str(n0_artifact)
        binding["source_root"] = artifact["source"]["root"]
        binding["source_commit"] = artifact["source"]["source_commit"]
        binding["normalizer_sha256"] = file_sha256(Path(artifact["normalizer_path"]))
        binding["runtime_python"] = str(root / "runtime/n0-twam/bin/python")
        table = file_table(checkpoint, [weight])
        if (
            table[str(weight.relative_to(checkpoint))]["sha256"]
            != artifact["checkpoint_sha256"]
        ):
            raise ValueError(
                "canonical N0 checkpoint differs from reusable prepared artifact"
            )
        binding["checkpoint_alias_note"] = (
            "prepared serve bundle uses an immutable byte-identical checkpoint copy"
        )
    elif model == "n0_vtla":
        checkpoint = checkpoint / "checkpoint"
        weight = checkpoint / "model.safetensors"
        binding.update(
            source_root=str(repo / "deployment/sources/N0-VTLA"),
            runtime_python=str(repo / "deployment/runtime/n0-vtla/bin/python"),
            asset_id="univtac_mixed8_train759_qpos8",
        )
        norm = checkpoint / "assets" / binding["asset_id"] / "norm_stats.json"
        binding["normalizer_sha256"] = file_sha256(norm)
        table = file_table(checkpoint, [weight, norm])
    elif model == "ftp1_policy":
        weight = checkpoint / "model.safetensors"
        source = root / "sources/ftp1-policy"
        binding.update(
            source_root=str(source),
            runtime_python=str(root / "runtime/ftp1-policy/bin/python"),
        )
        parser = source / "data_processing/parse_data_module/parse_data_univtac.py"
        candidates = [
            node.value
            for node in ast.walk(ast.parse(parser.read_text()))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(key, ast.Name) and key.id == "instruction_map"
                for key in node.targets
            )
        ]
        if len(candidates) != 1:
            raise ValueError("cannot resolve exact FTP training prompt table")
        prompts = ast.literal_eval(candidates[0])
        settings = json.loads(parser.with_name("task_settings.json").read_text())
        for name, entry in tasks.items():
            entry.update(
                prompt=prompts[name],
                use_wrist=settings[name]["camera_type"] == "all",
                domain_name=f"UniVTAC_{name}",
            )
            if not (checkpoint / "normalization" / entry["domain_name"]).is_dir():
                raise FileNotFoundError(
                    f"missing FTP training domain: {entry['domain_name']}"
                )
        norms = file_table(
            checkpoint,
            sorted(
                path
                for path in (checkpoint / "normalization").rglob("*")
                if path.is_file()
            ),
        )
        binding["normalizer_sha256"] = canonical_hash(norms)
        binding["normalizer_members"] = norms
        table = file_table(
            checkpoint,
            [
                weight,
                checkpoint / "model_config.json",
                checkpoint / "train_config.json",
                checkpoint / "tactile_input_config_file.json",
                *sorted((checkpoint / "hpt_tokenizer").glob("*.safetensors")),
            ],
        )
    else:
        checkpoint = checkpoint / "checkpoint"
        extras = checkpoint.parent / "inference-assets"
        binding.update(
            source_root=str(root / "sources/Dream-Tac"),
            runtime_python=str(root / "runtime/dream-tac/bin/python"),
            dataset_stats=str(extras / "dataset_statistics_franka.json"),
            t5_embeddings=str(extras / "t5_embeddings.pkl"),
            tokenizer_checkpoint=str(extras / "tokenizer.pth"),
            experiment_config="cosmos_predict2_2b_480p_franka_hupai_tactile",
        )
        norm = json.loads(Path(binding["dataset_stats"]).read_text())
        # Upstream stores stats by suite. Fail explicitly rather than inventing
        # binary gripper limits if the trained continuous range is unavailable.
        binding["gripper_qpos_min"] = norm["actions_min"][6]
        binding["gripper_qpos_max"] = norm["actions_max"][6]
        binding["normalizer_sha256"] = file_sha256(Path(binding["dataset_stats"]))
        binding["shared_pythonpath"] = (
            str(root / "runtime/dream-tac/lib/python3.11/site-packages")
            + ":"
            + str(root / "runtime/ftp1-policy/lib/python3.11/site-packages")
        )
        layout = extras / "checkpoint-layout"
        layout.mkdir(exist_ok=False)
        (layout / "model").symlink_to(checkpoint, target_is_directory=True)
        binding["serve_checkpoint_root"] = str(layout)
        members = [checkpoint / ".metadata", *sorted(checkpoint.glob("*.distcp"))]
        if len(members) != 17:
            raise ValueError("Dream-Tac requires all 17 DCP members")
        table = file_table(checkpoint, members)
        binding["inference_asset_members"] = file_table(
            extras,
            [
                Path(binding[key])
                for key in ("dataset_stats", "t5_embeddings", "tokenizer_checkpoint")
            ],
        )
        weight = None
    if model != "n0_twam":
        binding["source_commit"] = subprocess.check_output(
            ["git", "-C", binding["source_root"], "rev-parse", "HEAD"], text=True
        ).strip()
    binding["checkpoint_root"] = str(checkpoint)
    binding["checkpoint_members"] = table
    binding["checkpoint_sha256"] = (
        canonical_hash(table)
        if weight is None
        else table[str(weight.relative_to(checkpoint))]["sha256"]
    )
    binding["binding_sha256"] = canonical_hash(binding)
    if model == "ftp1_policy":
        from types import SimpleNamespace

        from scripts.ftp1_policy.serve_official import FTP1PolicyEngine, TaskContract

        shape = SimpleNamespace(
            get_state_dim=lambda: 120,
            get_action_dim=lambda: 120,
            get_action_horizon=lambda: 32,
            model_config=SimpleNamespace(use_tactile_input=True),
        )
        for name, entry in tasks.items():
            entry["transport_metadata"] = FTP1PolicyEngine(
                shape,
                task_id=name,
                source_commit=binding["source_commit"],
                checkpoint_sha256=binding["checkpoint_sha256"],
                serve_bundle_sha256=binding["binding_sha256"],
                task_contract=TaskContract(
                    entry["prompt"],
                    ("top", "wrist") if entry["use_wrist"] else ("top",),
                ),
            ).metadata
    write_json(target, binding)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n0-artifact", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(FOLDERS), default=list(FOLDERS)
    )
    args = parser.parse_args()
    output = args.output.absolute()
    dataset = output / "dataset_identity.json"
    if not dataset.exists():
        write_json(
            dataset,
            {
                "protocol": "official_univtac_seeded_simulator_diagnostic_v1",
                "initial_seed": 0,
                "exogenous_seed": 0,
                "tasks": list(load_retrained(args.n0_artifact)["tasks"]),
                "not_frozen40_replay": True,
                "training_split": "train759_only",
            },
        )
    for model in args.models:
        print(json.dumps({"stage": "binding", "model": model}), flush=True)
        try:
            path = prepare_model(
                model,
                args.deployment_root.absolute(),
                output,
                args.n0_artifact.absolute(),
                dataset,
            )
        except (OSError, ValueError, KeyError) as error:
            write_json(
                output / "preparation_errors" / f"{model}.json",
                {"status": "not_run", "error": str(error), "model": model},
            )
            print(
                json.dumps({"model": model, "status": "not_run", "error": str(error)}),
                flush=True,
            )
        else:
            print(
                json.dumps(
                    {"stage": "binding_ready", "model": model, "path": str(path)}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
