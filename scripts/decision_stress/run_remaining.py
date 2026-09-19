"""No-clobber sequential supervisor for the remaining ten noise operators."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import time
from pathlib import Path

from scripts.decision_stress.remaining import read, write
from scripts.decision_stress.report import report
from scripts.n0_twam.run_official_early_fault_seed import (
    _check_server_port_available,
    _stop_owned,
    _wait_server,
    official_server_command,
    official_server_environment,
)


def run(output: Path, bundle: Path, repo: Path) -> None:
    plan = read(output / "plan.json")
    if (output / "finished.json").exists():
        return
    dep = repo / "deployment-sm120"
    source = dep / "sources/N0-TWAM"
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join((str(bundle / "package"), str(bundle))),
        CUDA_VISIBLE_DEVICES="0",
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for group in plan["groups"]:
            root = Path(group["path"])
            if (root / "finished.json").exists():
                continue
            if (root / "started.json").exists() or any(
                Path(c["artifact"]).exists() for c in group["cells"]
            ):
                raise RuntimeError(
                    f"partial group preserved; automatic replay prohibited: {root}"
                )
            command = [
                plan["isaac_python"],
                "-m",
                "scripts.decision_stress.worker",
                "--plan",
                str(root / "group_plan.json"),
                "--n0-source",
                str(source),
                "--port",
                str(plan["n0_port"]),
            ]
            write(
                root / "started.json", {"started_unix": time.time(), "command": command}
            )
            server = None
            try:
                if plan["model"] == "n0":
                    port = plan["n0_port"]
                    _check_server_port_available(port)
                    server_env = official_server_environment(
                        root=dep,
                        task=group["task"],
                        save_root=root / "server_dumps",
                        inherited=env,
                    )
                    server_env["PYTHONPATH"] = os.pathsep.join(
                        (str(bundle / "package"), str(source), str(bundle))
                    )
                    with (root / "server.log").open("xb") as log:
                        server = subprocess.Popen(
                            official_server_command(
                                n0_python=dep / "runtime/n0-twam/bin/python",
                                repo=repo,
                                port=port,
                                save_root=root / "server_dumps",
                            ),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,
                            env=server_env,
                            start_new_session=True,
                        )
                    write(
                        root / "server_process.json",
                        {"pid": server.pid, "started_unix": time.time()},
                    )
                    _wait_server(server, port)
                with (root / "execution.log").open("xb") as log:
                    child = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        env=env,
                        cwd=bundle,
                        start_new_session=True,
                    )
                    write(
                        root / "process.json",
                        {
                            "pid": child.pid,
                            "started_unix": time.time(),
                            "command": command,
                        },
                    )
                    returncode = child.wait()
                if not (root / "execution_complete.json").is_file():
                    raise RuntimeError(
                        f"worker incomplete (exit {returncode}); inspect {root / 'execution.log'}"
                    )
                write(
                    root / "finished.json",
                    {
                        "completed_unix": time.time(),
                        "returncode": returncode,
                        "teardown_warning": returncode != 0,
                    },
                )
            except Exception as error:
                write(
                    root / "error.json",
                    {"failed_unix": time.time(), "error": str(error)},
                )
                raise
            finally:
                if server is not None:
                    _stop_owned(server)
        if plan["schema"] == "a2_episode_censored_continuation_v1":
            from scripts.decision_stress.a2_censored import report as a2_report

            result = a2_report(output)
        else:
            result = report(output)
        write(output / "results.json", result)
        write(
            output / "finished.json",
            {
                "completed_unix": time.time(),
                "planned_episode_count": plan["planned_episode_count"],
            },
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    a = p.parse_args()
    run(
        a.output.resolve(strict=True),
        a.bundle.resolve(strict=True),
        a.repo.resolve(strict=True),
    )


if __name__ == "__main__":
    main()
