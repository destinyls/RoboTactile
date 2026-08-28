# Reproducing RoboTactile

This guide is the shortest supported path from a source checkout to a verified
RoboTactile result. It separates package verification, N0-TWAM Clean execution,
ACT robustness matrices, and paper claims so that one stage is never presented
as evidence for another.

All commands start in the repository root. Generated sources, runtimes, model
files, requests, receipts, and results stay below the ignored `deployment/`
tree unless `ROBOTACTILE_DEPLOY_ROOT` selects another task-specific directory.

## Supported reproduction surfaces

| Surface | Public entry point | Current boundary |
|---|---|---|
| 14-operator deterministic replay | `robotactile smoke-replay` | Package/operator evidence only |
| N0-TWAM recorded HDF5 evaluation | `robotactile recorded-n0` | Real recorded observations; no simulator success |
| N0-TWAM UniVTAC Clean | `scripts/n0_twam/run_clean_campaign_all_tasks.py` | One task, all eight tasks, pilot, and paper protocols are implemented |
| ACT UniVTAC robustness matrix | `generate-primary-matrix` + `run-live-matrix` | Current public 14 x 5 live-matrix path |
| N0-TWAM UniVTAC fault campaign | No public campaign driver yet | Low-level N0 live requests and streaming faults exist; do not claim an end-to-end matrix |

The last row is an explicit release limitation. The current primary generator
binds an ACT tactile checkpoint and a separately trained ACT vision-only
checkpoint. Frozen N0-TWAM has no matched vision-only checkpoint, and its A1/A2
structural-absence conditions are `unsupported_contract` because the model
requires both tactile streams.

## Operator map

The machine-readable authority is [`core_v2.json`](../configs/operators/core_v2.json).
The table below is an onboarding summary; severity values remain the native
parameters recorded in each fault manifest rather than interchangeable visual
strength labels.

| ID | Delivered fault | Native unit | Stateful | Extra evidence |
|---|---|---|---|---|
| `A1_stream_absence` | Removes the tactile payload for a declared window | affected window fraction | No | None |
| `A2_frame_erasure` | Removes registered individual frames | erased frame fraction | No | erased offsets |
| `F1_global_response_drift` | Scales the full contact residual about rest | retained global gain | No | rest reference |
| `F2_spatial_sensitivity_loss` | Attenuates the contact residual inside one spatial region without moving it | retained regional gain | No | rest reference |
| `F3_persistent_surface_artifact` | Adds a fixed surface scar across the affected window | scar dose | No | rest reference |
| `F4_local_nonresponsive_patch` | Replaces one local patch with its resting response | failed-area dose | No | rest reference |
| `F5_contact_shape_distortion` | Coherently warps the current contact shape | peak warp displacement in pixels | No | contact phase |
| `F6_history_residual_imprint` | Mixes a prior contact imprint into later release observations | history mix | Yes | rest and release samples |
| `F7_high_load_saturation` | Compresses the current high-response contact field | response-knee dose | No | contact phase |
| `T1_fixed_source_delay` | Delivers a fixed-lag source frame | lag frames | No | source provenance |
| `T2_held_last_freeze` | Repeats the last delivered source frame | held duration in frames | Yes | source provenance |
| `T3_inter_sensor_skew` | Delivers the two tactile streams from different source times | source-index gap | Yes | source provenance |
| `C1_sensor_identity_misrouting` | Routes a tactile stream under the wrong physical sensor identity | misrouted window fraction | No | source provenance |
| `C2_frame_misregistration` | Reprojects a registered tactile frame with a spatial offset | reprojection displacement in pixels | No | rest reference for registered-pixel mode |

## 1. Clone and verify the package

```bash
git clone https://github.com/destinyls/RoboTactile.git
cd RoboTactile
git rev-parse HEAD

bash scripts/bootstrap_pip.sh
source .venv/bin/activate

robotactile validate-registry
robotactile smoke-replay \
  --operator T1_fixed_source_delay \
  --severity 3 \
  --output outputs/reproduction-smoke
make check
python scripts/release/update_source_manifest.py --check
```

Record the Git commit and `release/source_manifest.sha256` with every result.
The bootstrap uses hash-locked pip requirements and does not require `uv`.
Package tests do not start Isaac Sim or establish a learned-policy result.

## 2. Choose and initialize a deployment

```bash
export ROBOTACTILE_DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
export DEPLOY_ROOT="$ROBOTACTILE_DEPLOY_ROOT"
robotactile deployment init --root "$DEPLOY_ROOT"
robotactile deployment show --root "$DEPLOY_ROOT"
```

The complete ownership and directory contract is in
[`deployment_layout.md`](deployment_layout.md). Never use `/`, `/data`, `/mnt`,
or another shared storage root directly as `DEPLOY_ROOT`.

## 3. Install pinned N0-TWAM and UniVTAC sources

```bash
bash integrations/install_univtac.sh
bash scripts/n0_twam/install_official_runtime.sh --root "$DEPLOY_ROOT"
bash scripts/n0_twam/install_robotactile_client.sh --root "$DEPLOY_ROOT"
```

The N0 runtime is isolated at `deployment/runtime/n0-twam`. These commands do
not install model weights into the wheel and do not modify system Python or
system CUDA. On Ubuntu 22.04, provide Python 3.11 explicitly if the
deployment-local micromamba fallback has not been installed yet:

```bash
bash scripts/n0_twam/install_official_runtime.sh \
  --root "$DEPLOY_ROOT" \
  --python /absolute/path/to/python3.11
```

Exact upstream commits, model revisions, and licenses are in
[`external_dependencies.md`](external_dependencies.md).

## 4. Install and qualify the simulator runtime

Follow [`isaac_sim.md`](isaac_sim.md) for the verified Isaac Sim 4.5.0 archive,
headless smoke, deployment-local CUDA toolkit, IsaacLab v2.1.1, cuRobo v0.7.7,
TacEx/UIPC, and the RoboTactile Isaac client.

Minimum distinctions:

- the N0 fast server requires at least 40 GB on every selected GPU;
- co-locating N0 and Isaac on one GPU needs additional measured headroom;
- Blackwell `sm_120` requires a fresh native build and cannot reuse A800/RTX
  3090 native extensions;
- a headless smoke proves infrastructure launch, not task success.

After installation, check the declared deployment without starting an episode:

```bash
robotactile deployment doctor --root "$DEPLOY_ROOT" --profile n0-univtac
```

## 5. Prepare one official N0 task

The following example uses `insert_tube`; replace `TASK` only with a registered
UniVTAC task ID.

```bash
export TASK=insert_tube
export N0_PYTHON="$DEPLOY_ROOT/runtime/n0-twam/bin/python"
test -x "$N0_PYTHON"

"$N0_PYTHON" scripts/n0_twam/prepare_official_artifacts.py \
  --root "$DEPLOY_ROOT/artifacts/models/n0_twam" \
  --task "$TASK"

export N0_CONFIG="$DEPLOY_ROOT/artifacts/models/n0_twam/configs/$TASK/integration_config.json"
test -f "$N0_CONFIG"

robotactile integrations doctor \
  --model n0_twam \
  --root "$DEPLOY_ROOT" \
  --task "$TASK" \
  --config "$N0_CONFIG"
```

Preparation downloads only immutable revisions and generates a task-specific
normalizer, serve bundle, artifact manifest, and integration config. Do not
copy or edit the zero-hash files under `examples/n0_twam/` into a deployment.

## 6. Run one N0 Clean episode

Use a unique campaign and run ID. The example freezes official evaluation seed
2, which derives task seed 3000000 under the released evaluator contract.

```bash
export CAMPAIGN_ID="n0-clean-$TASK-seed2-v1"
export RUN_ID="$CAMPAIGN_ID"
export CAMPAIGN_MANIFEST="$DEPLOY_ROOT/requests/clean-campaigns/$CAMPAIGN_ID/campaign_manifest.json"

"$N0_PYTHON" scripts/n0_twam/generate_clean_campaign_requests.py \
  --root "$DEPLOY_ROOT" \
  --campaign-id "$CAMPAIGN_ID" \
  --protocol diagnostic_v1 \
  --trials-per-task 1 \
  --sampling-contract univtac_official_v1 \
  --official-eval-seed 2 \
  --replacement-reserve-per-task 0 \
  --task "$TASK" \
  --integration-config "$N0_CONFIG" \
  --initial-state-policy official_reproduction \
  --simulator-device cuda:0

"$N0_PYTHON" scripts/n0_twam/run_clean_campaign_all_tasks.py \
  --root "$DEPLOY_ROOT" \
  --manifest "$CAMPAIGN_MANIFEST" \
  --execution-profile quick \
  --capture-profile metrics_only_v1 \
  --gpus 0 \
  --run-id "$RUN_ID" \
  --max-new-trials-per-task 1 \
  --worker-contract fresh_process_v1 \
  --action-execution-contract robotactile_n0_training_60hz_ee_v1
```

The campaign owner starts and stops the task-specific official N0 server,
launches the Isaac child, validates the resulting artifact, and never retries a
normal model failure. A classified infrastructure exception is distinct from a
scoreable terminal failure.

Strictly regenerate the diagnostic summary when needed:

```bash
robotactile clean-campaign-report \
  --root "$DEPLOY_ROOT" \
  --manifest "$CAMPAIGN_MANIFEST" \
  --allow-compact-capture
```

Read these outputs:

```text
deployment/requests/clean-campaigns/<campaign>/campaign_manifest.json
deployment/artifacts/live-univtac/clean-campaigns/<campaign>/<task>/0000/
deployment/outputs/clean-campaigns/<campaign>/attempts/<task>/
deployment/outputs/clean-campaigns/<campaign>/clean_baseline_summary.json
deployment/outputs/clean-campaigns/<campaign>/runs/<run-id>.json
```

The summary fields to report are `loaded_artifact_count`,
`eligible_trial_count`, `success_count`, `eligible_success_rate`,
`macro_task_success_rate`, `terminal_status_counts`,
`protocol_invalid_count`, and `paper_claim_eligible`.

## 7. Select an evidence/capture profile

Execution strictness and stored payload are separate flags:

| Execution | Default capture | Stored evidence | Use |
|---|---|---|---|
| `quick` | `metrics_only_v1` | Metric, actions, termination, diagnostics, hashes | Success-rate development runs |
| `diagnostic` | `preview_v1` | Metrics plus at most 64 deterministic keyframes | Debugging and compact videos |
| `claim` | `paper_full_v1` | Complete RGB, tactile, proprio, actions, diagnostics | Qualified pilot/paper evidence |

To retain a compact video in the one-task command, replace the execution and
capture flags with:

```text
--execution-profile diagnostic --capture-profile preview_v1
```

Then render the verified artifact without rerunning Isaac:

```bash
robotactile visualize-live-artifact \
  --artifact "$DEPLOY_ROOT/artifacts/live-univtac/clean-campaigns/$CAMPAIGN_ID/$TASK/0000" \
  --output "$DEPLOY_ROOT/outputs/visualizations/$CAMPAIGN_ID" \
  --video --fps 20
```

`metrics_only_v1` intentionally contains no video frames.

## 8. Expand Clean evaluation only after the one-task result

Prepare all task-specific artifacts, then run the bounded one-per-task
diagnostic:

```bash
"$N0_PYTHON" scripts/n0_twam/prepare_official_artifacts.py \
  --root "$DEPLOY_ROOT/artifacts/models/n0_twam" \
  --all-tasks --skip-download

bash scripts/n0_twam/run_clean_all_tasks_once.sh \
  --root "$DEPLOY_ROOT" \
  --gpus 0 \
  --campaign-id n0-clean-all8-once-v1 \
  --run-id n0-clean-all8-once-v1
```

This produces eight diagnostic outcomes, not a paper estimate. The formal
`pilot_v1` and `paper_v1` protocols require all eight tasks, respectively 10
and 100 valid outcomes per task, official consecutive sampling, qualification
v3, dual runtime attestations, and full capture. See
[`model_integrations.md`](model_integrations.md#6-run-and-gate-an-official-clean-campaign-v2).

## 9. Robustness evaluation boundary

The deterministic operator layer can be verified immediately:

```bash
robotactile validate-registry
robotactile smoke-matrix --output outputs/operator-smoke-matrix
```

The current complete live primary matrix is ACT-specific:

```bash
robotactile generate-primary-matrix --help
robotactile run-live-matrix --help
```

Do not present those commands as an N0-TWAM fault benchmark. A public N0
fault-campaign generator must additionally bind the N0 task config, preserve
Clean/delivered observation hashes, reset temporal/stateful operators per
episode, and report A1/A2 as unsupported rather than substituting black images.

## 10. Release and result checklist

Before sharing a reproduction:

1. record the Git commit and source-manifest digest;
2. record `integrations/integrations.lock.json` and all install receipts;
3. retain task-specific model artifact/config hashes;
4. retain the campaign or matrix manifest before execution;
5. retain attempt, runtime-attestation, and root receipts;
6. report infrastructure exceptions separately from model outcomes;
7. label diagnostic, pilot, paper, and real-robot evidence separately;
8. never commit model files, datasets, credentials, or generated deployment
   outputs.
