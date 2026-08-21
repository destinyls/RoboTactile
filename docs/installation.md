# Installation

RoboTactile deliberately separates the dependency-light core from simulator
and model runtimes. Choose one installation level and do not infer a higher
evidence level from a lower one.

| Installation | Provides | Does not provide |
|---|---|---|
| Core wheel | Contracts, operators, artifacts, reporting, live preflight | Torch, Isaac Sim, model weights |
| External checkouts | Frozen ACT, N0-TWAM, and UniVTAC source identities | Dependencies, datasets, checkpoints |
| Isaac deployment | Standalone Isaac, IsaacLab, cuRobo, GPU smoke | UniVTAC task success or model qualification |

## Core development installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements/dev.lock.txt
python -m pip install --no-deps --no-build-isolation .
python -m robotactile_benchmark.cli integrations list
python -m pytest -q
```

On Windows PowerShell, activate the same environment with
`.\.venv\Scripts\Activate.ps1`; subsequent commands still use `python -m ...`.

The equivalent one-command bootstrap is `bash scripts/bootstrap_pip.sh`.
Neither path requires `uv`; `uv.lock` is retained only as an optional
maintainer lock.

Core installation requires neither torch nor Isaac Sim.

Build the release artifacts without accessing the network after dependencies
have been synchronized:

```bash
python -m hatchling build
```

## External repositories

Choose explicit absolute destinations:

```bash
bash integrations/install_univtac.sh /absolute/external/UniVTAC
bash integrations/install_act_runtime.sh /absolute/external/WorldArena
bash integrations/install_n0_twam.sh /absolute/external/N0-TWAM
```

Each script verifies the exact origin and commit and writes a sibling canonical
install receipt. Existing mismatched or dirty checkouts are never overwritten.
The scripts do not install weights, datasets, Isaac Sim, or GPU dependencies.

ACT and N0-TWAM currently remain release-blocked until reviewed local runtime
changes are published in externally accessible commits. Their base commits are
available for development but do not satisfy that final source gate.

## Isaac Sim deployment

For Ubuntu 22.04/NVIDIA installation, use the dedicated
[Isaac Sim 4.5 installation and UniVTAC execution guide](isaac_sim.md). It
covers archive hashing, standalone installation, headless GPU smoke,
IsaacLab/cuRobo, external checkouts, Isaac's bundled Python, request
generation, live execution, and troubleshooting.

Do not use the macOS development environment for live UniVTAC execution. Do
not run the standalone backend from the core development Python and label it as Isaac
evidence.
