# Installation

RoboTactile deliberately separates the dependency-light core from simulator
and model runtimes. Choose one installation level and do not infer a higher
evidence level from a lower one.

| Installation | Provides | Does not provide |
|---|---|---|
| Core wheel | Contracts, operators, artifacts, reporting, live preflight | Torch, Isaac Sim, model weights |
| External checkouts | Frozen UniVTAC (including `policy/ACT`), FTP-1, N0-TWAM, and N0-VTLA source identities | Dependencies, datasets, checkpoints |
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

Initialize the repo-contained workspace, then install to its canonical source
directories:

```bash
robotactile deployment init
bash integrations/install_univtac.sh
python scripts/act/install_official_artifacts.py \
  --artifact-root "$PWD/deployment/artifacts/models/act"
bash integrations/install_n0_vtla.sh
bash scripts/n0_twam/install_official_runtime.sh --root "$PWD/deployment"
bash scripts/ftp1_policy/install_official_runtime.sh --root "$PWD/deployment"
```

For the first-class ACT integration, `install_univtac.sh` is the only model
source checkout step: the adapter is pinned to
`deployment/sources/UniVTAC/policy/ACT/act_policy.py`. The separately retained
`integrations/install_act_runtime.sh` command checks out the legacy
WorldArena/ACTStrict `policy_best.ckpt` path and is not an official ACT install
step.

When the installer must create its deployment-local Python 3.11 fallback with
micromamba, it uses four extraction/bytecode-compilation workers by default.
This avoids spawning one worker per host CPU on shared filesystems. Override
the bounded value only when appropriate with `--mamba-extract-threads N`
(`1 <= N <= 32`).

Each script verifies the exact origin and commit and writes a canonical install
receipt. Existing mismatched or dirty checkouts are never overwritten. The N0
and FTP-1 installers create separate deployment-local Python environments;
neither modifies system Python/CUDA, Isaac, or the other model runtimes, and
neither downloads model weights. On Blackwell, the FTP-1 receipt additionally
requires its isolated PyTorch `2.7.1+cu128` bundle to report CUDA 12.8 and
`sm_120`, then complete a real CUDA tensor kernel. Its PaliGemma tokenizer is
hash-bound under `deployment/artifacts/openpi-data/ftp1-policy`; server launch
must export that path as `OPENPI_DATA_HOME`. See
[Deployment layout](deployment_layout.md) and
[External dependencies](external_dependencies.md) for the complete tree,
immutable commit links, licenses, and HPC override.

The ACT artifact installer pins the official `byml/UniVTAC` dataset at revision
`172331dbbce95bc04c3e59b22f32dc72ba5561ae`. By default it installs the shared
`encoder.pth` and all eight `univtac` task profiles after validating
`integrations/act_artifacts.lock.json`. This is sufficient for Clean/Faulted
evaluation. Install `vision_only` only when running the matched no-touch
control. The files remain outside the wheel and are content-addressed by
`robotactile integrations configure act --task <task> --profile <profile>`, which writes
`artifacts/models/act/configs/<task>/<profile>/integration_config.json`.

Optional upstream logs and metadata are references rather than locally
executed results. The official artifact metadata also does not prove that
frozen40 was excluded from training: label it `ACT-official`; use
`ACT-train759` only for separately retrained, split-manifest-bound weights.

Official ACT inference is in-process inside Isaac Sim's bundled Python. There
is no second ACT venv or model server to install. Do not install the upstream
ACT conda environment over Isaac and do not upgrade or replace Isaac's bundled
Torch/Torchvision; doing so can invalidate IsaacLab, cuRobo, TacEx, and native
extension compatibility.

Official N0 and FTP-1 source/model revisions are pinned, but live readiness
still requires separately downloading and hashing task-specific serving
artifacts. FTP-1 source is Apache-2.0; its checkpoint repository does not
specify a weight license and may also be subject to Gemma terms. Review
[Third-party notices](../THIRD_PARTY_NOTICES.md) before downloading weights.

## Isaac Sim deployment

For Ubuntu 22.04/NVIDIA installation, use the dedicated
[Isaac Sim 4.5 installation and UniVTAC execution guide](isaac_sim.md). It
covers archive hashing, standalone installation, headless GPU smoke,
IsaacLab/cuRobo, external checkouts, Isaac's bundled Python, request
generation, live execution, and troubleshooting.

Do not use the macOS development environment for live UniVTAC execution. Do
not run the standalone backend from the core development Python and label it as Isaac
evidence.
