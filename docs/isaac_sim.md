# Isaac Sim 4.5 Installation and UniVTAC Execution

This guide is the canonical GPU deployment path for RoboTactile. It installs a
downloaded Isaac Sim 4.5.0 package, runs a headless infrastructure smoke,
installs the pinned IsaacLab and cuRobo sources, and shows how to launch one
auditable UniVTAC ACT trial.

The commands below are not executed by installing the RoboTactile wheel. They
must be run explicitly on a compatible NVIDIA Linux host.
Directory ownership is defined in [Deployment layout](deployment_layout.md);
all Git pins and license boundaries are in
[External dependencies](external_dependencies.md).

## Evidence ladder

Keep these outcomes separate:

| Stage | Required evidence | What it proves |
|---|---|---|
| Isaac installation | `isaac_sim_install.json` | The verified archive installed successfully |
| GPU smoke | `isaac_sim_smoke_<run-id>.json` | Isaac's headless example exited zero on one GPU |
| Live preflight | `live_preflight.json` | Exact requests, sources, artifacts, GPU visibility, and Isaac Python are ready without allocation |
| Unqualified live trial | live artifact root receipt | UniVTAC, a policy, and the runner produced a consistent trace |
| Isaac-qualified result | separate qualification receipt | Reserved; RoboTactile does not currently emit this |

The GPU smoke receipt records `evidence_boundary=infrastructure_launch_only`.
Live artifacts record
`evidence_level=unqualified_live_univtac_execution_v1` and
`simulator_qualification_claimed=false`. Neither may be reported as an
Isaac-qualified benchmark result.

## Canonical reference environment

The documented reference target is:

- Ubuntu 22.04, Linux x86-64;
- one NVIDIA GPU visible through `nvidia-smi`;
- RTX 3090 or another supported GPU with at least 16 GB VRAM;
- at least 32 GB system RAM;
- Isaac Sim 4.5.0 standalone archive;
- IsaacLab v2.1.1;
- cuRobo v0.7.7;
- a writable absolute deployment root with sufficient free space.

RTX 3090 is the canonical RoboTactile reference GPU, not a claim that every
driver or workstation configuration is qualified. Check the official Isaac
Sim system and driver requirements before installation.

Live execution is Linux-only. macOS can build the package, run tests, render
figures, and generate requests, but cannot run the Isaac/UniVTAC backend.

## 1. Prepare the source checkout and deployment root

Run from the RoboTactile repository root:

```bash
bash scripts/bootstrap_pip.sh
source .venv/bin/activate
```

Initialize the repository-contained default. For HPC, set
`ROBOTACTILE_DEPLOY_ROOT` to one absolute task-specific directory before this
command. Do not use `/`, `/data`, `/mnt`, or a shared storage root directly.

```bash
export ROBOTACTILE_ROOT="$(pwd -P)"
robotactile deployment init
export DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$ROBOTACTILE_ROOT/deployment}"
robotactile deployment doctor --profile core
```

The deployment scripts redirect their temporary files, caches, Omni user
state, Torch cache, and `HOME` under `$DEPLOY_ROOT/runtime/`. They do not use
the invoking user's home directory and do not call `sudo`.

## 2. Download and independently hash Isaac Sim 4.5.0

Download the official Linux x86-64 standalone archive from NVIDIA. Place it at
a stable path, then record its SHA-256 independently:

```bash
export ISAAC_ARCHIVE="$DEPLOY_ROOT/runtime/isaac-sim-standalone-4.5.0-linux-x86_64.zip"
sha256sum "$ISAAC_ARCHIVE"
export ISAAC_ARCHIVE_SHA256='<independently-recorded-64-hex-sha256>'
```

Do not copy a digest printed after an untrusted transfer into the same command
without comparing it to a separately recorded value. RoboTactile deliberately
requires `--sha256`; the installer will not silently trust the archive it is
about to extract.

Official references:

- [Isaac Sim 4.5 installation index](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/index.html)
- [Isaac Sim 4.5 workstation/archive installation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html)
- [IsaacLab v2.1.1 documentation](https://isaac-sim.github.io/IsaacLab/v2.1.1/)

## 3. Install the standalone archive

```bash
bash scripts/live_univtac/install_isaac_sim_4_5.sh \
  --root "$DEPLOY_ROOT" \
  --archive "$ISAAC_ARCHIVE" \
  --sha256 "$ISAAC_ARCHIVE_SHA256"
```

The installer rejects unsafe zip paths, extracts through a staging directory,
runs `post_install.sh`, and refuses to overwrite a foreign installation. A
matching rerun is an idempotent no-op.

Successful installation creates:

```text
$DEPLOY_ROOT/runtime/isaac-sim-4.5.0/
$DEPLOY_ROOT/artifacts/deployment/isaac_sim_install.json
$DEPLOY_ROOT/logs/isaac-sim-post-install-*.log
```

Inspect the receipt without changing it:

```bash
python3 -m json.tool \
  "$DEPLOY_ROOT/artifacts/deployment/isaac_sim_install.json"
test -x "$DEPLOY_ROOT/runtime/isaac-sim-4.5.0/python.sh"
```

## 4. Run the headless GPU smoke

First verify that the selected physical GPU exists:

```bash
export GPU_INDEX=0
nvidia-smi -i "$GPU_INDEX"
```

Run the frozen infrastructure smoke:

```bash
bash scripts/live_univtac/smoke_isaac_sim_4_5.sh \
  --root "$DEPLOY_ROOT" \
  --gpu "$GPU_INDEX" \
  --run-id rtx3090-headless-v1
```

The command launches Isaac's bundled `hello_world.py --headless` example. A
successful receipt appears at:

```text
$DEPLOY_ROOT/artifacts/deployment/isaac_sim_smoke_rtx3090-headless-v1.json
```

This is an infrastructure launch check only. It does not import UniVTAC, load
a tactile observation, run a policy, or evaluate task success.

## 5. Install pinned IsaacLab and cuRobo

Install against the verified standalone runtime:

```bash
bash scripts/live_univtac/install_isaaclab_v2_1_1.sh \
  --root "$DEPLOY_ROOT"

bash scripts/live_univtac/install_curobo_v0_7_7.sh \
  --root "$DEPLOY_ROOT"
```

The scripts pin:

- IsaacLab v2.1.1 at commit
  `90b79bb2d44feb8d833f260f2bf37da3487180ba`;
- cuRobo v0.7.7 at commit
  `0a50de1ba72db304195d59d9d0b1ed269696047f`.

They use Isaac Sim's bundled Python and create:

```text
$DEPLOY_ROOT/artifacts/deployment/isaaclab_install.json
$DEPLOY_ROOT/artifacts/deployment/curobo_install.json
```

Do not activate a separate conda, venv, or uv environment for the standalone runtime.
The live process must see `isaaclab.app` through Isaac Sim's `python.sh`.

## 6. Materialize the external repositories

RoboTactile does not copy third-party source into its wheel. Install the exact
checkouts into the deployment root:

```bash
bash integrations/install_univtac.sh
bash integrations/install_act_runtime.sh
bash integrations/install_n0_twam.sh
```

Each installer verifies origin, commit, cleanliness, and license identity and
writes a sibling `*.robotactile-install.json` receipt. These commands check out
source only; they do not download datasets or weights.

UniVTAC's modified TacEx/UIPC and task dependencies must also be available to
Isaac Sim's bundled Python according to the pinned upstream checkout. The
current RoboTactile installer does not claim to automate or qualify that
upstream system build. Missing imports fail before a trial is reported.

N0-TWAM is a first-class typed integration, but a production N0 transport and
published serving bundle are not currently registered. The public CLI fails
closed instead of substituting a fake model.

## 7. Install RoboTactile into Isaac's Python

Use the core pip environment to build the wheel, then install that wheel into
the standalone runtime. This keeps the core development environment separate
from Isaac's bundled Python.

```bash
python -m hatchling build -t wheel
export ISAAC_SIM_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
export ROBOTACTILE_WHEEL="$(find "$ROBOTACTILE_ROOT/dist" -maxdepth 1 \
  -name 'robotactile_benchmark-*.whl' -print | sort | tail -n 1)"
test -n "$ROBOTACTILE_WHEEL"

"$ISAAC_SIM_PATH/python.sh" -m pip install --no-deps "$ROBOTACTILE_WHEEL"
"$ISAAC_SIM_PATH/python.sh" -c \
  'import isaaclab.app, numpy, robotactile_benchmark, typing_extensions'
```

If the final import fails, stop. Do not run a trial from the core development Python and
describe it as Isaac execution.

## 8. Generate one four-condition request set

Place the official tactile ACT and matched vision-only artifacts under one
artifact root. Record the checkpoint, statistics, and encoder hashes before
generating requests:

```bash
export UNIVTAC_ROOT="$DEPLOY_ROOT/sources/UniVTAC"
export CHECKPOINT_ROOT="$DEPLOY_ROOT/artifacts/models/act"
export TACTILE_CHECKPOINT_SHA256='<64-hex-tactile-checkpoint-sha256>'
export VISION_CHECKPOINT_SHA256='<64-hex-vision-only-checkpoint-sha256>'
export STATS_SHA256='<64-hex-dataset-stats-sha256>'
export ENCODER_SHA256='<64-hex-encoder-sha256>'

python scripts/live_univtac/generate_pull_out_key_matrix.py \
  --tactile-checkpoint-sha256 "$TACTILE_CHECKPOINT_SHA256" \
  --vision-checkpoint-sha256 "$VISION_CHECKPOINT_SHA256" \
  --stats-sha256 "$STATS_SHA256" \
  --encoder-sha256 "$ENCODER_SHA256" \
  --initial-seed 17 \
  --exogenous-seed 29 \
  --operator T1_fixed_source_delay \
  --severity 3 \
  --fault-start 16 \
  --restoration-index 180
```

The generator creates `clean`, `faulted`, `no_touch`, and `restored` requests
with one matched pair identity. It does not launch Isaac or a policy.

Generate the tactile ACT runtime config from those real files before
preflight. The command computes the hashes; do not edit the all-zero example:

```bash
robotactile integrations configure act --task pull_out_key
robotactile integrations doctor --model act
```

## 9. Run the no-allocation live preflight

Validate the request, both pinned source checkouts, official ACT artifacts,
NVIDIA visibility, and Isaac's bundled Python before allocating the simulator:

```bash
export REQUEST_ROOT='<generated-request-directory>'

robotactile preflight-live \
  --request "$REQUEST_ROOT/requests/clean.json" \
  --config "$DEPLOY_ROOT/artifacts/models/act/integration_config.json"
```

The command fails while either external model pin has `release_ready=false`, a
checkout is dirty, an artifact hash differs, the GPU is not visible, or Isaac
Python lacks required packages. It never imports a learned model, constructs an
AppLauncher, or executes a task. Archive the receipt, but do not report it as
simulator execution or qualification.

## 10. Execute the live ACT requests

Set `REQUEST_ROOT` to the directory printed by the generator. Requests use
logical `cuda:0`; `CUDA_VISIBLE_DEVICES` selects the physical GPU.

The installed console form is `robotactile live-univtac-run`. For the
standalone archive, invoke the same entry point through its bundled Python:

```bash
export REQUEST_ROOT='<generated-request-directory>'

for condition in clean faulted no_touch restored; do
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    "$ISAAC_SIM_PATH/python.sh" -m robotactile_benchmark.cli \
      live-univtac-run \
      --request "$REQUEST_ROOT/requests/$condition.json" \
      --config "$DEPLOY_ROOT/artifacts/models/act/integration_config.json"
done
```

Every successful invocation exports and strictly reloads its live artifact.
The one-line JSON summary includes condition, terminal status, validation,
artifact root hash, evidence level, and the qualification flag.

Do not infer success from process exit alone. Accept a trial only when:

1. the command exits zero;
2. the artifact reload succeeds;
3. `validation_passed` is true;
4. the terminal status is retained exactly as reported;
5. the evidence remains `unqualified_live_univtac_execution_v1`;
6. `simulator_qualification_claimed` remains false.

The complete 14 x 5 matrix and report workflow is documented in
[benchmark_workflow.md](benchmark_workflow.md).

## Troubleshooting

### `nvidia-smi` cannot see the GPU

Stop before installation smoke. Check the NVIDIA driver and container/device
exposure on the host. Do not replace this check with a software test receipt.

### Archive SHA-256 mismatch

Delete or quarantine the transferred archive and obtain it again from the
official source. Never change the expected digest merely to make installation
continue.

### `isaaclab.app` is unavailable

Confirm that the command uses `$ISAAC_SIM_PATH/python.sh`, that the IsaacLab
receipt matches v2.1.1, and that `_isaac_sim` points to the verified standalone
installation.

### First headless launch is slow

Initial shader and extension caches can take several minutes to populate.
Inspect the receipt-linked log under `$DEPLOY_ROOT/logs/`; do not terminate a
healthy launch solely because it is initially quiet.

### UniVTAC task import fails

Verify the external checkout receipt, exact upstream commit, clean worktree,
task source hash, and its TacEx/UIPC dependencies. RoboTactile deliberately
rejects a shadow or modified task module.

### Existing output or receipt is rejected

Installers, request generators, and artifact writers are no-clobber. Reuse an
existing path only when its canonical content is identical; otherwise choose a
new run ID or output directory.

### N0-TWAM live execution is unavailable

This is the expected release boundary until a production typed transport and
published serving artifacts are registered. Adapter contract tests are not a
substitute for live N0 inference.

## What to archive for a paper result

Preserve at minimum:

- the RoboTactile source manifest and wheel SHA-256;
- all external install receipts and exact commits;
- the Isaac installation and GPU smoke receipts;
- request, fault, trial, policy-artifact, and matrix manifests;
- every live artifact root receipt;
- the matrix summary and report receipt;
- logs referenced by receipts;
- GPU, driver, OS, Python, and Isaac versions.

Only a later, separately specified Isaac qualification protocol may promote
these artifacts to an Isaac-qualified result.
