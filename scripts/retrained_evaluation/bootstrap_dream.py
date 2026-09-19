"""Verify asynchronous auxiliary transfer, then enqueue Dream-Tac on the same lock."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from robotactile_benchmark.integrations.n0_twam.retrained import file_sha256
from scripts.retrained_evaluation.group import write_json
from scripts.retrained_evaluation.prepare import prepare_model

EXPECTED = {
    "dataset_statistics_franka.json": (
        1873,
        "19ab1f6a3d3a7f0d849ea00f55eaea9831864ffe0d65edf396fa3c0a844a1b46",
    ),
    "t5_embeddings.pkl": (
        8391599,
        "8da54d62f26e0d6b16f03d32cd2d7db8dd0d0934ce5b2f50608ae38a59ea6db5",
    ),
    "tokenizer.pth": (
        507609880,
        "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("deployment-root", "campaign", "code", "package", "n0-artifact"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root, campaign = args.deployment_root, args.campaign
    assets = (
        root
        / "artifacts/models/dream_tac/retrained-train759-step20000/inference-assets"
    )
    control = campaign / "dream_bootstrap"
    control.mkdir(exist_ok=False)
    started = time.monotonic()
    write_json(
        control / "started.json",
        {"started_unix": time.time(), "expected_assets": EXPECTED},
    )
    while time.monotonic() - started < 43200:
        tokenizer = assets / "tokenizer.pth"
        incoming = assets / "tokenizer-raw-stream.pth"
        if (
            not tokenizer.exists()
            and incoming.is_file()
            and incoming.stat().st_size == EXPECTED["tokenizer.pth"][0]
        ):
            if file_sha256(incoming) != EXPECTED["tokenizer.pth"][1]:
                raise ValueError("completed tokenizer transfer digest mismatch")
            # Claim the verified inode without overwriting any concurrent
            # destination. Earlier incomplete transfers remain untouched.
            os.link(incoming, tokenizer)
        if all(
            (assets / name).is_file() and (assets / name).stat().st_size == size
            for name, (size, _) in EXPECTED.items()
        ):
            break
        time.sleep(30)
    else:
        raise TimeoutError("Dream-Tac auxiliary transfer did not finish within 12h")
    for name, (_, digest) in EXPECTED.items():
        if file_sha256(assets / name) != digest:
            raise ValueError(f"auxiliary transfer digest mismatch: {name}")
    write_json(
        control / "assets_verified.json",
        {"status": "sha256_verified", "members": EXPECTED},
    )
    shared = root / "runtime/ftp1-policy/lib/python3.11/site-packages"
    runtime = root / "runtime/dream-tac/bin/python"
    env = dict(os.environ)
    env["CUDNN_HOME"] = str(shared / "nvidia/cudnn")
    env["CUDA_HOME"] = str(root / "runtime/cuda-toolkit-12.8")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        (str(shared / "nvidia/cudnn/lib"), env.get("LD_LIBRARY_PATH", ""))
    )
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(args.package),
            str(args.code),
            str(root / "runtime/dream-tac/lib/python3.11/site-packages"),
            str(shared),
        )
    )
    # This checks imports, not an extra model episode or a success metric.
    with (control / "runtime_import.log").open("xb") as log:
        checked = subprocess.run(
            [
                str(runtime),
                "-c",
                "import torch, transformer_engine.pytorch, megatron.core, hydra, peft; print('Dream-Tac imports available')",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    if checked.returncode:
        write_json(
            control / "runtime_not_ready.json",
            {
                "status": "not_run",
                "reason": "see runtime_import.log; no Dream-Tac episode launched",
            },
        )
        raise RuntimeError("Dream-Tac runtime imports failed")
    binding = prepare_model(
        "dream_tac",
        root,
        campaign,
        args.n0_artifact,
        campaign / "dataset_identity.json",
    )
    write_json(
        control / "queued.json",
        {"binding": str(binding), "status": "waiting_for_serial_gpu_lock"},
    )
    command = [
        str(runtime),
        "-m",
        "scripts.retrained_evaluation.campaign",
        "--campaign",
        str(campaign),
        "--code",
        str(args.code),
        "--package",
        str(args.package),
        "--models",
        "dream_tac",
        "--wait-for-gpu-lock",
    ]
    os.execve(str(runtime), command, env)


if __name__ == "__main__":
    main()
