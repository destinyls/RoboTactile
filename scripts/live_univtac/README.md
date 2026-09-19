# Live UniVTAC runtime deployment

Run every command below from the `RoboTactile/` repository root. For
prerequisites, bundled-Python setup, live execution, acceptance criteria, and
troubleshooting, start with
[`docs/isaac_sim.md`](../../docs/isaac_sim.md). This file is the compact
reference for the scripts in this directory.

These entry points prepare the pinned Linux/NVIDIA runtime used by the live
RoboTactile qualification path. They deliberately do not call UniVTAC's
all-in-one `scripts/install.sh`: no command uses `sudo`, sources a shell rc
file, or writes to the invoking user's `HOME`.

The default root is the checkout's ignored directory:

```text
RoboTactile/deployment
```

Set `ROBOTACTILE_DEPLOY_ROOT` only when an absolute HPC/shared-storage
override is required. See [`docs/deployment_layout.md`](../../docs/deployment_layout.md).

Every upstream process receives deployment-local cache, temporary, Omni,
Python bytecode, CUDA, Torch, pip, and `HOME` paths below `runtime/`.

## Blackwell (`sm_120`) isolated bootstrap

Do not reuse an A800/RTX 3090 native deployment on Blackwell. A copied
torch-scatter, modified UIPC, or cuRobo `.so` can import while still containing
only an `sm_80`/`sm_86` cubin. Create a sibling root ending in
`deployment-sm120` and run the source-bound N0-only bootstrap:

```bash
ROOT="$PWD/deployment-sm120"
CUDA_RUNFILE="$ROOT/runtime/installers/cuda_12.8.1_570.124.06_linux.run"
CUDA_SHA256='<verified-sha256>'
ISAAC_ARCHIVE="$PWD/deployment/runtime/isaac-sim-standalone-4.5.0-linux-x86_64.zip"
ISAAC_SHA256='<verified-sha256>'
WHEEL="$PWD/dist/robotactile_benchmark-0.6.0-py3-none-any.whl"

bash scripts/live_univtac/bootstrap_blackwell_sm120.sh \
  --root "$ROOT" \
  --cuda-runfile "$CUDA_RUNFILE" --cuda-sha256 "$CUDA_SHA256" \
  --isaac-archive "$ISAAC_ARCHIVE" --isaac-sha256 "$ISAAC_SHA256" \
  --wheel "$WHEEL" --gpu 0
```

The command never installs ACT and never writes the legacy `deployment`
runtime. It installs CUDA Toolkit 12.8.1 without a driver, fresh pinned
UniVTAC/IsaacLab/cuRobo sources, TacEx and modified UIPC compiled for `sm_120`,
and isolated Isaac/N0 Python runtimes. It then rejects the deployment unless
`cuobjdump` finds an `sm_120` cubin in the three torch-scatter kernel
extensions, the UIPC CUDA backend, and all five cuRobo extensions. The
torch-scatter `_version_cuda.so` metadata stub is hashed in the attestation but
does not contain or require device code. The decisive receipts are:

```text
deployment-sm120/artifacts/deployment/cuda_native_extensions_sm_120.json
deployment-sm120/artifacts/deployment/blackwell_sm120_bootstrap.json
```

This is infrastructure compatibility evidence. It does not claim a UniVTAC
task reset, policy rollout, or Clean success.

## 1. Install the standalone Isaac Sim archive

Use the official Isaac Sim 4.5.0 Linux standalone archive already downloaded
to the NVIDIA host. Supply the independently recorded 64-character SHA-256;
the installer will not derive trust from the archive it is about to install.

```bash
DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
ARCHIVE="$DEPLOY_ROOT/runtime/isaac-sim-standalone-4.5.0-linux-x86_64.zip"
ARCHIVE_SHA256='<verified-sha256>'

bash scripts/live_univtac/install_isaac_sim_4_5.sh \
  --archive "$ARCHIVE" \
  --sha256 "$ARCHIVE_SHA256"
```

The script validates every zip member against path traversal, extracts into a
deployment-local staging directory, runs `post_install.sh`, and writes:

```text
artifacts/deployment/isaac_sim_install.json
logs/isaac-sim-post-install-*.log
```

A matching rerun is an idempotent no-op. A different receipt or an unowned
destination is rejected rather than overwritten.

## 2. Run the headless infrastructure smoke

```bash
bash scripts/live_univtac/smoke_isaac_sim_4_5.sh \
  --gpu 1 \
  --run-id initial-gpu1
```

This runs the standalone `hello_world.py --headless` example and records the
GPU identity, installation archive hash, command log hash, and explicit
evidence boundary in:

```text
artifacts/deployment/isaac_sim_smoke_initial-gpu1.json
```

## 3. Install the pinned simulator stack and TacEx core

```bash
CUDA_RUNFILE="$DEPLOY_ROOT/runtime/cuda_12.4.1_550.54.15_linux.run"
CUDA_RUNFILE_SHA256='<independently-recorded-64-hex-sha256>'

bash scripts/live_univtac/install_cuda_toolkit_12_4.sh \
  --runfile "$CUDA_RUNFILE" --sha256 "$CUDA_RUNFILE_SHA256"

bash scripts/live_univtac/install_isaaclab_v2_1_1.sh

bash scripts/live_univtac/install_curobo_v0_7_7.sh

bash integrations/install_univtac.sh

CUDA_ROOT="$DEPLOY_ROOT/runtime/cuda-toolkit-12.4"
GPU_INDEX=0

bash scripts/live_univtac/install_tacex_univtac.sh \
  --cuda-root "$CUDA_ROOT" --cuda-architecture auto --gpu "$GPU_INDEX"

bash scripts/live_univtac/install_tacex_uipc_univtac.sh \
  --cuda-root "$CUDA_ROOT" --cuda-architecture auto --gpu "$GPU_INDEX"
bash scripts/live_univtac/install_robotactile_isaac.sh
bash scripts/live_univtac/qualify_univtac_task_import.sh
bash scripts/live_univtac/qualify_univtac_task_reset.sh \
  --gpu 0 --run-id pull-out-key-reset-s17-v1

bash scripts/live_univtac/qualify_univtac_task_pairing.sh \
  --gpu 0 --run-id pull-out-key-pairing-s17-v1
```

The pairing qualifier performs one real simulator action between the canonical
reset and snapshot replay. Replay verifies the exact UIPC/PhysX/task/RNG state
hash and reuses the canonical initial sensor frame instead of introducing RTX
rerender noise. It loads no learned policy and records only
`in_process_snapshot_replay_equivalence_v1`; it is not task-success or Isaac
qualification evidence.

The immutable source pins are:

- IsaacLab `v2.1.1`: `90b79bb2d44feb8d833f260f2bf37da3487180ba`
- cuRobo `v0.7.7`: `0a50de1ba72db304195d59d9d0b1ed269696047f`
- UniVTAC: `05bcd3edb92237107efa40105292a24f1a9fd761`
- vcpkg build tool `2025.04.09`:
  `ce613c41372b23b1f51333815feb3edd87ef8a8b`

The source trees live under `sources/`; all Python installation commands run
through the standalone Isaac Sim `python.sh`. Matching receipts make reruns
no-ops, while a wrong commit or modified tracked worktree fails before an
installer runs.

The TacEx installer uses the embedded source from that exact UniVTAC commit and
builds `torch_scatter==2.1.2` locally with a verified CUDA 12.4/12.8 toolkit. With
`--cuda-architecture auto`, A800 resolves to `sm_80`, RTX 3090 to `sm_86`, and
Blackwell compute capability 12.0 to `sm_120`;
an explicit value must match the selected `--gpu`. The explicit `--cuda-root`
must resolve below the deployment root. If it is omitted, the scripts prefer
`runtime/cuda-toolkit-12.8`, then 12.4 and their legacy `/usr/local` counterparts.
Only CUDA 12.8 is accepted for `sm_100`, `sm_101`, or `sm_120`. Receipts bind toolkit
path/version, `nvcc` release/build identity, and compute architecture, and UIPC requires
an exact match with the TacEx receipt. The installer refuses to publish a
receipt if Torch, NumPy, or the editable imports drift. The modified UIPC gate
uses micromamba only as a deployment-local native-tool bootstrap; its exact
CMake/GCC packages are frozen in `requirements/uipc-toolchain-linux-64.lock.txt`. The vcpkg tool
commit supplies the source of a repository-owned `cpptrace==0.8.3` build recipe;
that version is absent from libuipc's older, source-pinned manifest baseline.
The sibling `tinygltf==2.9.3` recipe keeps the baseline source version while
pinning the regenerated official GitHub tag archive bytes. A single overlay
root avoids platform-specific list separators. Other ports remain resolved
against the embedded baseline. Both overlay identities, the tool commit,
baseline, and exact native-tool lock are recorded in the install receipt.

The CUDA installer accepts only an independently supplied SHA-256 and runs the
official 12.4.1 runfile with toolkit-only, no-man-page, deployment-local
toolkit, and deployment-local temporary-directory flags. It never requests a
driver install or `/usr/local` target. The vendor runfile can still create
`/usr/local/cuda` and write `/var/log/cuda-installer.log` when run as root, so
the wrapper rejects any pre-existing `/usr/local/cuda` and removes a newly
created path only when it targets the deployment-local toolkit, allowing the
vendor's single trailing slash.
It does not remove or manage the vendor log. Its no-clobber receipt records the
runfile digest, toolkit path, `12.4|V12.4.131` nvcc identity, and protected-link
restoration; a matching rerun is a no-op and a verified incomplete attempt can
be retried safely.

On a restricted host, pre-download the exact official tinygltf archive and pass
`--tinygltf-archive PATH`. The installer verifies SHA-512 before publishing it
to the deployment-local vcpkg cache; a different existing cache file fails
closed.

The final task-import qualification launches Isaac headlessly, imports
`envs.pull_out_key`, and constructs only `TaskCfg`. Its receipt explicitly
records zero simulator steps and no task-environment construction; it is a
deployment gate, not closed-loop success evidence.

The subsequent reset qualification uses RoboTactile's production backend to
construct one task, complete the upstream reset/pre-move path, validate one
`top`/`wrist_l`/tactile/qpos8 observation, request runtime close, and observe process
exit. Reset advances the simulator, but no policy or action command is
executed. The no-clobber receipt records the native post-reset step,
observation hashes, physical GPU, exact runtime source manifest, and
qualification script hashes.

For `same_task_worker_v1`, first run
`probe_same_task_app_reuse.py` as documented in
[`docs/benchmark_workflow.md`](../../docs/benchmark_workflow.md). It verifies
two fresh task runtimes inside one Isaac application without loading N0. A
passing probe is deployment evidence for application reuse; it is not a Clean
episode and contributes neither a success nor a failure to benchmark metrics.

The RoboTactile wheel installer verifies the checked-in source manifest, builds
into deployment temporary storage, force-reinstalls only inside standalone
Isaac Python, and binds the wheel/runtime identities to a separate receipt.

## N0 pre-Clean dynamic contract

For the source-bound `lift_bottle` experiment lock and the one-action TAA
simulator probe, use the commands and acceptance boundary in
[`docs/recorded_n0.md`](../../docs/recorded_n0.md#pre-clean-simulator-dynamic-gate).
The probe starts neither the released N0 server nor a Clean episode; it must
pass before either is allowed.

## Source-bound N0 Clean entry point

After generating a frozen Clean campaign and qualification v3, run it through
the N0-only all-task owner:

```bash
export DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
export QUALIFICATION="$DEPLOY_ROOT/artifacts/deployment/<source-bound-qualification-v3>.json"

python scripts/n0_twam/run_clean_campaign_all_tasks.py \
  --root "$DEPLOY_ROOT" \
  --manifest "$DEPLOY_ROOT/requests/clean-campaigns/<campaign>/campaign_manifest.json" \
  --qualification "$QUALIFICATION" \
  --gpus 0 \
  --run-id <new-run-id>
```

Qualification v3 holds eight ordered task-local source/parity bindings;
normalizer and serve-bundle hashes may differ by task. The owner automatically
passes the selected binding into each shard, obtains an N0 rank-0 server
attestation and an Isaac-child attestation, and binds both in attempt v3.
`--n0-server-attestation*` and `--isaac-attestation-output` are internal flags,
not user inputs at this entry point. The lifecycle hard watchdog covers startup
through teardown and artifact export. Only classified `exception_replaced`
attempts are excluded from the model denominator.

Only a complete `paper_v1` publication bundle with exact qualification and
dual-attestation matches can remove `simulator_not_qualified`; legacy v1/v2,
diagnostic, and pilot evidence cannot. The default `official_reproduction`
initial-state policy does not reject reset terminal signals by itself. The
opt-in `replace_initial_terminal_v1` policy is a separate robustness diagnostic
and is not directly comparable with the public 84.5% reference. These commands
describe the implemented gate; they do not claim a newly executed GPU result.

## 4. Freeze the first pull-out-key three-condition matrix

Run the generator from the benchmark repository with the hashes recorded when
the official `policy_last.ckpt`, `dataset_stats.pkl`, and `encoder.pth` files
are transferred to the target host:

```bash
DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
UNIVTAC_ROOT="$DEPLOY_ROOT/sources/UniVTAC"
CHECKPOINT_ROOT="$DEPLOY_ROOT/artifacts/models/act"

source "$ROBOTACTILE_ROOT/.venv/bin/activate"
python scripts/live_univtac/generate_pull_out_key_matrix.py \
  --tactile-checkpoint-sha256 '<univtac-policy-last-sha256>' \
  --vision-checkpoint-sha256 '<vision-only-policy-last-sha256>' \
  --stats-sha256 '<dataset-stats-sha256>' \
  --encoder-sha256 '<encoder-sha256>' \
  --initial-seed 17 \
  --exogenous-seed 29
```

The default operator is `T1_fixed_source_delay` at S3 (four frames), active
from observation 16. The persistent request keeps it active through the
301-observation budget. Override it with `--operator`, `--severity`, and
`--fault-start`. Operators that require a measured
rest-reference bundle fail closed unless `--rest-reference-artifact` points to
the strict three-file artifact created by:

```bash
python -m robotactile_benchmark.cli build-rest-references \
  --source-live-artifact "$DEPLOY_ROOT/artifacts/calibration-clean" \
  --dataset-split calibration \
  --minimum-consecutive-free-records 5 \
  --output "$DEPLOY_ROOT/artifacts/rest-references/pull_out_key-v1"
```

The source must be a separate clean live artifact. The selector requires a
consecutive two-sensor `FREE` interval from the frozen UniVTAC depth-phase
tracker; it never invents, blackens, or borrows a test-episode baseline. Pass
the resulting directory with `--rest-reference-artifact`. Its evidence remains
unqualified until the separate Isaac acceptance protocol is completed.

The atomically published directory contains:

```text
trial_set_manifest.json
fault_manifests/persistent.json
policy_artifacts/{univtac,vision_only}.json
requests/{clean,faulted,no_touch}.json
matrix_receipt.json
```

For a rest-reference operator, the directory also contains the copied and
hash-bound `rest_references/{rest_reference.json,no_contact_validation.json,root_receipt.json}`.

All three requests share the tactile ACT `base_system_manifest_sha256` and one
`pair_key`. Only `no_touch` executes the matched vision-only checkpoint and
config. Each condition has a separate artifact output directory, while
`live-univtac-paired-run` executes all conditions in one shared runtime from a
single canonical snapshot.
The request records `headless=true`, `enable_cameras=true`, and logical
`cuda:0`; select a physical GPU outside the process with, for example,
`CUDA_VISIBLE_DEVICES=1`.

`dataset_sha256` is the exact content hash of the generated frozen
`trial_set_manifest.json`. It identifies the matched task/seeds contract; it
is deliberately not described as a hash of raw episode files. Repeating the
same command is an idempotent no-op. Different content at the same output path
is rejected without replacement.

## Receipt-bound detached jobs

Use the receipt-bound launcher only on hosts that preserve detached process
groups after SSH disconnect. Some managed SSH services later terminate the
entire originating process group; use the Supervisor entry point below on
those hosts. The expected artifact must be a new absolute path below the
deployment root:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/live_univtac/run_detached_job.sh \
  --root "$DEPLOY_ROOT" \
  --job-id n0-pull-out-key-clean-v1 \
  --artifact "$DEPLOY_ROOT/artifacts/live-univtac/n0-twam/pull_out_key/clean" \
  -- \
  "$DEPLOY_ROOT/runtime/isaac-sim-4.5.0/python.sh" \
  -m robotactile_benchmark.cli evaluate \
  --model n0_twam --root "$DEPLOY_ROOT" \
  --request "$DEPLOY_ROOT/requests/n0-twam/pull_out_key/clean.json" \
  --config "$DEPLOY_ROOT/artifacts/models/n0_twam/configs/pull_out_key/integration_config.json"
```

The launcher creates exactly one no-clobber
`outputs/jobs/<job-id>/` directory and prints its paths plus the worker PID and
PGID as JSON. `launch.json` and, after a real child exit, `exit.json` are
atomically published read-only receipts; `job.log` is hash-bound by
`exit.json`. Send `TERM` or `INT` to the recorded worker PID for a forwarded,
recorded shutdown. `SIGKILL`, host loss, or power loss deliberately leaves no
`exit.json`; that state is incomplete evidence, never a successful or failed
trial receipt. Even a zero return code and present artifact still require the
normal strict artifact reload and validation gates.

## Supervisor-managed long-running jobs

Supervisor is an optional operational dependency, not a RoboTactile Python
dependency. Pass both executable paths explicitly; the launcher never assumes
a system or `/opt` installation. The generated config, foreground program,
PID, short Unix socket, logs, and immutable receipts all remain below the
deployment root:

```bash
SUPERVISORD='<absolute-path-to-supervisord>'
SUPERVISORCTL='<absolute-path-to-supervisorctl>'
JOB_ID='n0-pull-out-key-clean-supervised-v1'
ARTIFACT="$DEPLOY_ROOT/artifacts/live-univtac/n0-twam/pull_out_key/clean"

CUDA_VISIBLE_DEVICES=0 bash scripts/live_univtac/run_supervised_job.sh launch \
  --root "$DEPLOY_ROOT" --job-id "$JOB_ID" \
  --supervisord "$SUPERVISORD" --supervisorctl "$SUPERVISORCTL" \
  --artifact "$ARTIFACT" --timeout-seconds 2400 \
  -- \
  "$DEPLOY_ROOT/runtime/isaac-sim-4.5.0/python.sh" \
  -m robotactile_benchmark.cli evaluate \
  --model n0_twam --root "$DEPLOY_ROOT" \
  --request "$DEPLOY_ROOT/requests/n0-twam/pull_out_key/clean.json" \
  --config "$DEPLOY_ROOT/artifacts/models/n0_twam/configs/pull_out_key/integration_config.json"
```

The supervised foreground program is wrapped by GNU `timeout` with `TERM` at
the requested deadline and `KILL` 90 seconds later. The default hard timeout
is 2400 seconds and values below 60 are rejected; it covers simulator startup,
reset, and the complete trial. Manage the same no-clobber job with:

```bash
COMMON=(--root "$DEPLOY_ROOT" --job-id "$JOB_ID" \
  --supervisord "$SUPERVISORD" --supervisorctl "$SUPERVISORCTL")
bash scripts/live_univtac/run_supervised_job.sh status "${COMMON[@]}"
bash scripts/live_univtac/run_supervised_job.sh stop "${COMMON[@]}"
bash scripts/live_univtac/run_supervised_job.sh shutdown "${COMMON[@]}"
```

For source-bound Clean campaigns this Supervisor wrapper is only an outer
operational guard. The owned all-task path additionally publishes its own
identity-bound hard-watchdog evidence across the full child lifecycle.

Each control action atomically publishes a deployment-local receipt. A
Supervisor state of `EXITED`, including reported exit code 0, only describes
process lifecycle. Artifact presence and `root_receipt.json` presence are also
diagnostic fields: neither is an artifact-success claim. Accept the result
only after the benchmark's strict artifact reload and validation gates pass.

## Evidence boundary

An install receipt proves only that the pinned install command exited zero. A
headless smoke receipt proves only that Isaac Sim's infrastructure example
exited zero on the recorded GPU. Neither artifact proves that UniVTAC loaded,
that tactile observations reached a policy, that a paired clean/fault trial
completed, or that a task succeeded. Those claims require the separate live
UniVTAC closed-loop trial and artifact validator.

Likewise, `matrix_receipt.json` proves only deterministic request generation
and identity pairing. It does not prove that a policy or simulator was loaded.
