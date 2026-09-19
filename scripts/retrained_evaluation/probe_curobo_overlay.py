"""Verify private cuRobo binaries and execute GPU planning, not an episode."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import socket
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    overlay = args.overlay.resolve(strict=True)
    receipt = json.loads((overlay / "build_receipt.json").read_text())
    if receipt["returncode"] or not receipt["original_binaries_unchanged"]:
        raise ValueError("overlay did not finish building safely")
    torch = importlib.import_module("torch")
    expected_capability = tuple(map(int, receipt["architecture"].split(".")))
    if torch.cuda.get_device_capability(0) != expected_capability:
        raise ValueError("current GPU does not match the overlay architecture")
    modules = {}
    for name in (
        "geom_cu",
        "kinematics_fused_cu",
        "lbfgs_step_cu",
        "line_search_cu",
        "tensor_step_cu",
    ):
        module = importlib.import_module(f"curobo.curobolib.{name}")
        if module.__file__ is None:
            raise ValueError(f"extension has no binary file: {name}")
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(overlay / "curobo/src"):
            raise ValueError(f"wrong cuRobo extension loaded: {path}")
        architecture = subprocess.check_output(
            [
                str(Path(receipt["cuda_root"]) / "bin/cuobjdump"),
                "--list-elf",
                str(path),
            ],
            text=True,
        )
        if "sm_" + receipt["architecture"].replace(".", "") not in architecture:
            raise ValueError(f"missing GPU architecture: {path}")
        modules[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "architecture": architecture,
        }
    started = time.monotonic()
    motion = importlib.import_module("curobo.wrap.reacher.motion_gen")
    robot = importlib.import_module("curobo.types.robot")
    config = motion.MotionGenConfig.load_from_robot_config(
        "franka.yml", "collision_table.yml", interpolation_dt=1 / 120
    )
    planner = motion.MotionGen(config)
    planner.warmup()
    joints = planner.get_retract_config().view(1, -1)
    target = joints.clone()
    target[:, 0] += 0.02
    result = planner.plan_single_js(
        robot.JointState.from_position(joints),
        robot.JointState.from_position(target),
    )
    torch.cuda.synchronize()
    success = bool(result.success.item())
    evidence = {
        "status": "passed" if success else "planning_failed",
        "evidence_scope": "native_gpu_planner_smoke_not_closed_loop_episode",
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "capability": expected_capability,
        "torch": torch.__version__,
        "modules": modules,
        "warmup_completed": True,
        "joint_space_plan_success": success,
        "elapsed_s": time.monotonic() - started,
    }
    with args.output.open("x") as stream:
        json.dump(evidence, stream, indent=2)
    if not success:
        raise RuntimeError("native GPU planning failed; campaign was not started")


if __name__ == "__main__":
    main()
