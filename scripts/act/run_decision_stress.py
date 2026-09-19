"""Run native ACT paired sessions with the frozen N0 decision-stress noise."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.act.decision_stress import prepare, read, report, write


def run(output: Path, package_root: Path) -> None:
    plan = read(output / "pilot_plan.json")
    if (output / "finished.json").exists():
        return
    if not (package_root / "robotactile_benchmark").is_dir():
        raise FileNotFoundError(package_root)
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join((str(package_root), env.get("PYTHONPATH", ""))),
        CUDA_VISIBLE_DEVICES="0",
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    with (output / ".run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for group in plan["groups"]:
            root = Path(group["path"])
            if (root / "finished.json").exists():
                continue
            if (root / "started.json").exists() or any(
                Path(c["artifact"]).exists() for c in group["cells"]
            ):
                raise RuntimeError(
                    f"partial group exists; preserve artifacts and inspect before resuming: {root}"
                )
            for cell in group["cells"]:
                import hashlib

                if (
                    hashlib.sha256(Path(cell["request"]).read_bytes()).hexdigest()
                    != cell["request_file_sha256"]
                ):
                    raise ValueError("frozen ACT request file changed")
            command = [
                plan["isaac_python"],
                "-m",
                "robotactile_benchmark.cli",
                "live-univtac-paired-run",
                "--config",
                group["integration_config"],
                "--requests",
                *[c["request"] for c in group["cells"]],
                "--receipt",
                str(root / "paired_receipt.json"),
                "--capture-profile",
                plan["capture_profile"],
            ]
            write(
                root / "started.json", {"started_unix": time.time(), "command": command}
            )
            with (root / "execution.log").open("xb") as log:
                child = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    start_new_session=True,
                )
                write(
                    root / "process.json",
                    {"pid": child.pid, "started_unix": time.time(), "command": command},
                )
                returncode = child.wait()
            completed = all(
                (Path(c["artifact"]) / "terminal_result.json").exists()
                for c in group["cells"]
            )
            if not completed or not (root / "paired_receipt.json").exists():
                write(
                    root / "error.json",
                    {
                        "returncode": returncode,
                        "all_terminal": completed,
                        "error": "paired execution did not publish every terminal artifact and receipt",
                    },
                )
                raise RuntimeError(
                    f"ACT group incomplete; inspect {root / 'execution.log'}"
                )
            write(
                root / "finished.json",
                {
                    "completed_unix": time.time(),
                    "returncode": returncode,
                    "teardown_warning": returncode != 0,
                },
            )
        write(output / "pilot_results.json", report(output))
        write(
            output / "finished.json",
            {
                "completed_unix": time.time(),
                "planned_episode_count": plan["planned_episode_count"],
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--rest-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deployment-root", type=Path, required=True)
    p.add_argument("--config-root", type=Path, required=True)
    p.add_argument("--isaac-python", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--package-root", type=Path, required=True)
    q = sub.add_parser("report")
    q.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            reference=args.reference.resolve(strict=True),
            rest_root=args.rest_root.resolve(strict=True),
            output=args.output.absolute(),
            deployment_root=args.deployment_root.resolve(strict=True),
            config_root=args.config_root.resolve(strict=True),
            isaac_python=args.isaac_python.absolute(),
        )
    elif args.command == "run":
        run(args.output.resolve(strict=True), args.package_root.resolve(strict=True))
    else:
        print(
            json.dumps(
                report(args.output.resolve(strict=True)), indent=2, allow_nan=False
            )
        )


if __name__ == "__main__":
    main()
