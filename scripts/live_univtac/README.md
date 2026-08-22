# Live UniVTAC runtime deployment

For prerequisites, bundled-Python setup, live execution, acceptance criteria,
and troubleshooting, start with
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

## 1. Install the standalone Isaac Sim archive

Use the official Isaac Sim 4.5.0 Linux standalone archive already downloaded
to the NVIDIA host. Supply the independently recorded 64-character SHA-256;
the installer will not derive trust from the archive it is about to install.

```bash
DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
ARCHIVE="$DEPLOY_ROOT/runtime/isaac-sim-standalone-4.5.0-linux-x86_64.zip"
ARCHIVE_SHA256='<verified-sha256>'

bash RoboTactile/scripts/live_univtac/install_isaac_sim_4_5.sh \
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
bash RoboTactile/scripts/live_univtac/smoke_isaac_sim_4_5.sh \
  --gpu 1 \
  --run-id initial-gpu1
```

This runs the standalone `hello_world.py --headless` example and records the
GPU identity, installation archive hash, command log hash, and explicit
evidence boundary in:

```text
artifacts/deployment/isaac_sim_smoke_initial-gpu1.json
```

## 3. Install pinned IsaacLab and cuRobo sources

```bash
bash RoboTactile/scripts/live_univtac/install_isaaclab_v2_1_1.sh

bash RoboTactile/scripts/live_univtac/install_curobo_v0_7_7.sh
```

The immutable source pins are:

- IsaacLab `v2.1.1`: `90b79bb2d44feb8d833f260f2bf37da3487180ba`
- cuRobo `v0.7.7`: `0a50de1ba72db304195d59d9d0b1ed269696047f`

The source trees live under `sources/`; all Python installation commands run
through the standalone Isaac Sim `python.sh`. Matching receipts make reruns
no-ops, while a wrong commit or modified tracked worktree fails before an
installer runs.

## 4. Freeze the first pull-out-key four-condition matrix

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
301-observation budget; the restored request resumes valid delivery at
observation 180. Override these with `--operator`, `--severity`,
`--fault-start`, and `--restoration-index`. Operators that require a measured
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
fault_manifests/{persistent,restored}.json
policy_artifacts/{univtac,vision_only}.json
requests/{clean,faulted,no_touch,restored}.json
matrix_receipt.json
```

For a rest-reference operator, the directory also contains the copied and
hash-bound `rest_references/{rest_reference.json,no_contact_validation.json,root_receipt.json}`.

All four requests share the tactile ACT `base_system_manifest_sha256` and one
`pair_key`. Only `no_touch` executes the matched vision-only checkpoint and
config. Each condition has a separate runtime and artifact output directory.
The request records `headless=true`, `enable_cameras=true`, and logical
`cuda:0`; select a physical GPU outside the process with, for example,
`CUDA_VISIBLE_DEVICES=1`.

`dataset_sha256` is the exact content hash of the generated frozen
`trial_set_manifest.json`. It identifies the matched task/seeds contract; it
is deliberately not described as a hash of raw episode files. Repeating the
same command is an idempotent no-op. Different content at the same output path
is rejected without replacement.

## Evidence boundary

An install receipt proves only that the pinned install command exited zero. A
headless smoke receipt proves only that Isaac Sim's infrastructure example
exited zero on the recorded GPU. Neither artifact proves that UniVTAC loaded,
that tactile observations reached a policy, that a paired clean/fault trial
completed, or that a task succeeded. Those claims require the separate live
UniVTAC closed-loop trial and artifact validator.

Likewise, `matrix_receipt.json` proves only deterministic request generation
and identity pairing. It does not prove that a policy or simulator was loaded.
