# Quickstart

## NVIDIA/Isaac workflow

From the checkout, inspect the two integrations and prepare the canonical ACT
workspace:

```bash
robotactile integrations list
robotactile setup --model act
```

The first `setup` call is allowed to exit with status 2: its JSON output names
the missing source, artifact, transport, or release gates. After installing
sources and placing artifacts, rerun the same command until the doctor passes.
It does not allocate Isaac or run inference. The equivalent lower-level start
is `robotactile deployment init`.

The default
workspace is `RoboTactile/deployment`; see
[deployment_layout.md](deployment_layout.md) and
[external_dependencies.md](external_dependencies.md).

Use [isaac_sim.md](isaac_sim.md) for the canonical Ubuntu 22.04 + Isaac Sim
4.5.0 standalone flow. Do not skip directly to `live-univtac-run`: first
obtain the install receipt, pass the headless GPU smoke, install the pinned
IsaacLab/cuRobo stack, and satisfy the UniVTAC/model artifact gates.

For the complete 14 x 5 benchmark and paper report, continue with
[benchmark_workflow.md](benchmark_workflow.md).
