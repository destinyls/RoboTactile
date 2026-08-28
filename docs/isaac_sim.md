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
| CUDA toolkit installation | `cuda_toolkit_install.json` | The verified runfile installed the deployment-local CUDA Toolkit 12.4.1 payload and recorded protected-link restoration |
| GPU smoke | `isaac_sim_smoke_<run-id>.json` | Isaac's headless example exited zero on one GPU |
| Task reset qualification | `univtac_task_reset_<run-id>.json` | One pinned task instantiated, reset, exposed the strict multimodal observation contract, then requested close and exited |
| Live preflight | `live_preflight.json` | Exact requests, sources, artifacts, GPU visibility, and Isaac Python are ready without allocation |
| Unqualified live trial | live artifact root receipt | UniVTAC, a policy, and the runner produced a consistent trace |
| All-task qualification | aggregate receipt plus bound import/reset/pairing receipts | Task/action compatibility only; no policy or success rate |
| Qualified Clean bundle | `clean-campaign-publish` JSON/CSV/LaTeX | Requires a complete `paper_v1` campaign; not real-robot evidence |

The GPU smoke receipt records `evidence_boundary=infrastructure_launch_only`.
Live artifacts record
`evidence_level=unqualified_live_univtac_execution_v1` and
`simulator_qualification_claimed=false`. Neither may be reported as an
Isaac-qualified benchmark result.

## Canonical reference environment

The documented reference target is:

- Ubuntu 22.04, Linux x86-64;
- one NVIDIA GPU visible through `nvidia-smi`;
- RTX 3090, A800, or another supported GPU with at least 16 GB VRAM;
- at least 32 GB system RAM;
- Isaac Sim 4.5.0 standalone archive;
- CUDA Toolkit 12.4.1 runfile (`nvcc` V12.4.131) for native TacEx builds;
- IsaacLab v2.1.1;
- cuRobo v0.7.7;
- a writable absolute deployment root with sufficient free space.

RTX 3090 is the canonical RoboTactile reference GPU, not a claim that every
driver or workstation configuration is qualified. A800 is a supported build
target for the TacEx native components, but remains subject to the same live
qualification gates. Check the official Isaac Sim system and driver
requirements before installation.

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

## 6. Install CUDA Toolkit and materialize external repositories

For A800/RTX 3090, keep the CUDA 12.4.1 path below. For a Blackwell GPU with
compute capability 12.0, use a separate `deployment-sm120` root and the
one-command, N0-only bootstrap:

```bash
bash scripts/live_univtac/bootstrap_blackwell_sm120.sh \
  --root "$PWD/deployment-sm120" \
  --cuda-runfile "$PWD/deployment-sm120/runtime/installers/cuda_12.8.1_570.124.06_linux.run" \
  --cuda-sha256 '<verified-sha256>' \
  --isaac-archive "$PWD/deployment/runtime/isaac-sim-standalone-4.5.0-linux-x86_64.zip" \
  --isaac-sha256 '<verified-sha256>' \
  --wheel "$PWD/dist/robotactile_benchmark-0.4.0-py3-none-any.whl" \
  --gpu 0
```

This path uses CUDA Toolkit 12.8.1 (`nvcc` `V12.8.93`) and compiles all native
extensions for `sm_120`. It does not modify the NVIDIA driver, system Python,
system CUDA, or the legacy runtime. Completion requires both a headless Isaac
smoke and a `cuobjdump` attestation covering the three torch-scatter kernel
extensions, modified UIPC, and all five cuRobo CUDA extensions. The
torch-scatter CUDA-version metadata stub is hashed separately because it has
no device code. That attestation is an infrastructure gate, not closed-loop
success evidence.

RoboTactile does not copy third-party source into its wheel. Install the exact
checkouts into the deployment root:

Download the official Linux x86-64 CUDA Toolkit 12.4.1 runfile from NVIDIA's
[CUDA 12.4.1 archive](https://developer.nvidia.com/cuda-12-4-1-download-archive),
transfer it below the deployment root, and independently record its SHA-256:

```bash
export CUDA_RUNFILE="$DEPLOY_ROOT/runtime/cuda_12.4.1_550.54.15_linux.run"
export CUDA_RUNFILE_SHA256='<independently-recorded-64-hex-sha256>'

bash scripts/live_univtac/install_cuda_toolkit_12_4.sh \
  --root "$DEPLOY_ROOT" \
  --runfile "$CUDA_RUNFILE" \
  --sha256 "$CUDA_RUNFILE_SHA256"
```

The installer invokes only `--silent --toolkit`, the exact deployment-local
`--toolkitpath`, `--no-man-page`, and a deployment-local `--tmpdir`. It never
passes `--driver` and does not alter the existing NVIDIA driver. The NVIDIA
runfile may nevertheless create `/usr/local/cuda` and write its vendor log at
`/var/log/cuda-installer.log` when invoked as root. RoboTactile therefore
refuses to start a new installation if `/usr/local/cuda` already exists. If it
was absent, the script removes it after the runfile only when it is a symlink
whose target is the canonical deployment-local toolkit, allowing the vendor's
single trailing slash; any other
object or target is retained and fails closed for manual inspection. The
vendor log is never deleted or managed. RoboTactile's `HOME`, temporary files,
and own command log remain below `$DEPLOY_ROOT`. A same-hash rerun is a no-op.
A failed owned installation keeps a hash-bound incomplete marker and is safely
retried with the same command; an unowned toolkit destination or different
runfile is rejected without replacement.

Successful installation must produce:

```text
$DEPLOY_ROOT/runtime/cuda-toolkit-12.4/bin/nvcc
$DEPLOY_ROOT/artifacts/deployment/cuda_toolkit_install.json
$DEPLOY_ROOT/logs/cuda-toolkit-12.4.1-runfile-*.log
```

The receipt binds the runfile SHA-256, canonical toolkit path, exact
`12.4|V12.4.131` nvcc identity, `/usr/local/cuda` precondition/restoration
result, and the fact that the vendor log is unmanaged. Select the physical GPU
whose compute capability will be targeted:

```bash
export GPU_INDEX=0
export CUDA_ROOT="$DEPLOY_ROOT/runtime/cuda-toolkit-12.4"
test -x "$CUDA_ROOT/bin/nvcc"
"$CUDA_ROOT/bin/nvcc" --version
```

Both TacEx installers use one strict CUDA contract. With
`--cuda-architecture auto`, Isaac's bundled Torch probes the selected GPU:
A800 resolves to `80` (`sm_80`) and RTX 3090 resolves to `86` (`sm_86`). An
explicit `--cuda-architecture 80` or `86` is accepted only when it matches the
selected GPU; a mismatch fails before compilation. An explicit `--cuda-root`
must resolve below `$DEPLOY_ROOT`. For compatibility, omitting `--cuda-root`
checks `$DEPLOY_ROOT/runtime/cuda-toolkit-12.4` first and then the legacy
`/usr/local/cuda-12.4`; either candidate must still report CUDA 12.4.

```bash
bash integrations/install_univtac.sh
bash integrations/install_act_runtime.sh
bash scripts/n0_twam/install_official_runtime.sh --root "$DEPLOY_ROOT"
bash scripts/live_univtac/install_tacex_univtac.sh \
  --root "$DEPLOY_ROOT" \
  --cuda-root "$CUDA_ROOT" \
  --cuda-architecture auto \
  --gpu "$GPU_INDEX"

bash scripts/live_univtac/install_tacex_uipc_univtac.sh \
  --root "$DEPLOY_ROOT" \
  --cuda-root "$CUDA_ROOT" \
  --cuda-architecture auto \
  --gpu "$GPU_INDEX"

bash scripts/live_univtac/qualify_univtac_task_import.sh \
  --root "$DEPLOY_ROOT"
```

Each installer verifies origin, commit, cleanliness, and license identity and
writes a sibling `*.robotactile-install.json` receipt. These commands check out
source only; they do not download datasets or weights.

The TacEx core installer binds its receipt to the pinned UniVTAC checkout and
keeps Torch 2.7/CUDA 12.8 unchanged. The second installer builds UniVTAC's
embedded modified UIPC for the detected `sm_<compute-capability>` with a
deployment-local CMake 3.26.4/GCC 11.4 toolchain, pinned vcpkg tool commit,
minimal `cpptrace==0.8.3` and `tinygltf==2.9.3` overlays, exact package lock,
caches, and temporary directory. Both receipts bind the canonical CUDA toolkit
path, CUDA 12.4 version, `nvcc` release/build identity, CMake architecture, and decimal
compute capability. UIPC additionally requires those fields to match the TacEx
core receipt. Neither installer activates or modifies the invoking Conda base.
For a host with unreliable GitHub access, the same command accepts
`--tinygltf-archive /absolute/path/to/syoyo-tinygltf-v2.9.3.tar.gz`; the archive
must match the pinned SHA-512 and is copied only into the deployment cache.
The installer records a separate headless Isaac import receipt; missing native
imports fail before a trial is reported.
The qualification command then checks the exact `envs.pull_out_key.TaskCfg`
import after an Isaac headless launch. It deliberately does not instantiate the
environment or execute a simulator step, and its separate receipt preserves
that evidence boundary. Because the pinned UniVTAC task imports `omni.ui`
unconditionally, the qualification explicitly enables the bundled `omni.ui`
extension before importing the task; no desktop display is required.

N0-TWAM is a first-class official websocket integration. Its isolated runtime
installer does not download weights; prepare one task and run the model server
using the commands in [Model integrations](model_integrations.md#n0-twam-end-to-end).
The simulator-side client dependencies are installed only inside standalone
Isaac Python by `scripts/n0_twam/install_isaac_client.sh`.

## 7. Install RoboTactile into Isaac's Python

Use the manifest-bound installer to build the wheel and install it only into
the standalone runtime. This keeps the core development environment separate
from Isaac's bundled Python and records a deployment receipt.

```bash
bash scripts/live_univtac/install_robotactile_isaac.sh \
  --root "$DEPLOY_ROOT"
```

If the target host intentionally has no build tooling, build the pure-Python
wheel on a trusted source host and transfer it under the RoboTactile deployment
root. Then pass `--wheel /absolute/path/to/robotactile_benchmark-0.4.0-py3-none-any.whl`.
The installer rejects a different filename and verifies the packaged source
manifest after installation before writing its receipt.

If the final import fails, stop. Do not run a trial from the core development Python and
describe it as Isaac execution.

## 8. Qualify one task reset and observation

Use a fresh no-clobber run ID to exercise the production backend through task
construction, the upstream reset/pre-move sequence, strict observation
conversion, and clean runtime shutdown:

```bash
bash scripts/live_univtac/qualify_univtac_task_reset.sh \
  --root "$DEPLOY_ROOT" \
  --gpu "$GPU_INDEX" \
  --initial-seed 17 \
  --exogenous-seed 29 \
  --timeout-seconds 900 \
  --run-id pull-out-key-reset-s17-v1
```

The result must contain the frozen `top`/`wrist_l` RGB, left/right tactile RGB,
and qpos8 observation shapes and hashes. The task performs simulator work internally
during reset and pre-move, so the receipt records its post-reset native step
instead of claiming zero simulator steps. The same `initial_seed` is injected
into `TaskCfg.seed` before task construction and into the later reset call. The
factory also seeds Python, NumPy, Torch/CUDA, enables warn-only deterministic
Torch algorithms, disables cuDNN benchmarking, and fixes the cuBLAS workspace
configuration before task import. The receipt records these process controls
alongside `construction_seed`. It also records
`policy_loaded=false`, `action_commands_executed=0`,
`closed_loop_control_cycles=0`, `runtime_close_requested=true`, and
`task_success_evaluated=false`. Isaac may terminate its Python process inside
the close call, so the wrapper records that the process exited after the
fsynced result was published; it does not require post-close Python execution.
This gate is stronger than task import but is still not a policy or benchmark
result.

Then qualify the single-process snapshot/replay gate with a fresh run ID:

```bash
bash scripts/live_univtac/qualify_univtac_task_pairing.sh \
  --root "$DEPLOY_ROOT" \
  --gpu "$GPU_INDEX" \
  --initial-seed 17 \
  --exogenous-seed 29 \
  --timeout-seconds 900 \
  --run-id pull-out-key-pairing-s17-v1
```

This executes exactly one safe qpos action to prove divergence, then restores
the canonical UIPC/PhysX/task/RNG snapshot, verifies its physical-state hash,
and reuses the exact canonical initial sensor frame rather than rerendering it.
This avoids treating nondeterministic RTX refresh noise as an initial-condition
difference while still rejecting any physical snapshot drift. It requires two
exact post-reset witnesses, loads no learned policy, does not evaluate task success, keeps
`simulator_qualification_claimed=false`, and records
`in_process_snapshot_replay_equivalence_v1` only.

## 9. Generate one four-condition request set

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

## 10. Run the no-allocation live preflight

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

## 11. Execute the live ACT requests as one paired session

Set `REQUEST_ROOT` to the directory printed by the generator. Requests use
logical `cuda:0`; `CUDA_VISIBLE_DEVICES` selects the physical GPU.

Use `live-univtac-paired-run`, not four independent simulator processes. For
the standalone archive, invoke the entry point through its bundled Python:

```bash
export REQUEST_ROOT='<generated-request-directory>'

CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  "$ISAAC_SIM_PATH/python.sh" -m robotactile_benchmark.cli \
    live-univtac-paired-run \
    --requests \
      "$REQUEST_ROOT/requests/clean.json" \
      "$REQUEST_ROOT/requests/faulted.json" \
      "$REQUEST_ROOT/requests/no_touch.json" \
      "$REQUEST_ROOT/requests/restored.json" \
    --receipt "$REQUEST_ROOT/paired_execution_receipt.json" \
    --config "$DEPLOY_ROOT/artifacts/models/act/integration_config.json"
```

The group receipt uses
`evidence_level=unqualified_paired_live_univtac_execution_v1`; every exported
condition artifact retains `unqualified_live_univtac_execution_v1`.

The task performs exactly one upstream reset. After the clean post-reset frame
is converted, RoboTactile dumps the current UIPC frame and captures the PhysX
robot/plate state, controller targets, task counters, and NumPy RNG state.
Before every later condition it restores that snapshot in the same process and
compares an exact hash of the UIPC frame, PhysX state, controller targets, task
counters, and RNG state. It then rebinds the cached canonical initial sensor
frame to the condition episode; RTX is not asked to rerender step zero. Native
step, complete canonical top/wrist/tactile/depth/joint state hash, joint reorder
witness, canonical joint9, and model-visible qpos8 therefore remain identical.
A physical or observable mismatch is rejected before an observation reaches
the policy and no paired receipt is published. Every accepted condition still
exports and strictly reloads its own live artifact.

Render a paper-facing 960 x 720 panel and optional MP4 directly from one of
those strictly reloaded artifacts. This does not rerun Isaac Sim and never
upgrades the source evidence level:

```bash
python -m pip install --only-binary=:all: --require-hashes \
  -r requirements/visualization.lock.txt
python -m pip install --no-deps --no-build-isolation .
robotactile visualize-live-artifact \
  --artifact "$LIVE_ARTIFACT_ROOT" \
  --output "$DEPLOY_ROOT/outputs/visualizations/pull-out-key-faulted" \
  --video --fps 20
```

Run these installation commands from the repository root in the core
development environment created in step 1, not through Isaac Sim's
`python.sh`. The visualization lock contains Pillow 11.3.0 and the complete
distribution-hash set copied from `uv.lock`; binary-only mode fails closed on
an unsupported platform instead of compiling an unrecorded source artifact.

Each frame contains the top and wrist RGB views, clean and delivered tactile
payloads for both sensor slots, a labeled absolute-difference diagnostic, and
the exact step/contact/fault metadata. The exporter writes `preview.png`,
optional `preview.mp4`, and a hash-bound `visualization_receipt.json`. Video
export requires `ffmpeg`; the source trace remains the evaluation authority.
The exporter accepts `paper_full_v1` and the bounded keyframes in
`preview_v1`. It rejects `metrics_only_v1` with an explicit error because that
profile intentionally contains no image frames.

Do not infer success from process exit alone. Accept a trial only when:

1. the command exits zero;
2. the artifact reload succeeds;
3. `validation_passed` is true;
4. the terminal status is retained exactly as reported;
5. `reset_receipt.all_exact` is true and every executed condition links to one
   accepted witness;
6. the pairing evidence is
   `in_process_snapshot_replay_equivalence_v1`;
7. each trace remains `unqualified_live_univtac_execution_v1`;
8. `simulator_qualification_claimed` remains false.

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

### N0-TWAM server is unreachable or out of memory

Run `scripts/n0_twam/check_server.py` before launching Isaac. Confirm the exact
task-specific normalizer/config, websocket host/port, and that proxy variables
do not intercept localhost. Official fast serving requires at least 40 GB on
every selected rank because model components are resident before FSDP2
sharding. Multiple RTX 3090 cards do not remove that peak; `--debug-offload`
is a slow functional-smoke path only and must not support latency or throughput
claims. Reserve a separate GPU for Isaac. Adapter contract tests are not a
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
