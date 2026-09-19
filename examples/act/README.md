# ACT integration example

ACT is a first-class RoboTactile `PolicyAdapter`. The tactile profile evaluates
`clean` and `faulted`; a separately trained vision-only profile is the matched
`no_touch` control.

The checked-in JSON files are contract examples. All-zero hashes deliberately
fail content validation and must be replaced with hashes from your own artifact
manifest. RoboTactile never downloads weights implicitly.

Validate the static integration and external pin:

```bash
robotactile integrations validate --model act
```

Generate the runnable deployment config from real files instead of editing the
all-zero example:

```bash
robotactile deployment init
robotactile integrations configure act \
  --task pull_out_key --profile univtac
robotactile integrations doctor --model act
```

Generate canonical requests from the validated task/profile configs. Clean is
always generated; configure the vision-only profile and pass
`--include-no-touch` when the matched control is required:

```bash
robotactile integrations configure act \
  --task pull_out_key --profile vision_only

python scripts/act/prepare_requests.py \
  --task pull_out_key \
  --dataset-sha256 "$DATASET_SHA256" \
  --initial-seed 17 \
  --exogenous-seed 29 \
  --include-no-touch
```

Add `--fault-manifest /absolute/fault.json` to include Faulted. Add
`--rest-references /absolute/rest_reference.json` only when that fault operator
requires a certified rest reference. The no-clobber outputs are written to
`deployment/requests/act/<task>/`; `request_generation_summary.json` is the
canonical machine-readable inventory. Generation does not run the model or
simulator and is not closed-loop evidence.

The example request records the public contract shape only. Replace all-zero
hashes, source paths, runtime paths, and output paths with frozen deployment
values before running the live preflight:

```bash
robotactile preflight-live \
  --request examples/act/request.json \
  --config "$PWD/deployment/artifacts/models/act/configs/pull_out_key/univtac/integration_config.json"
```

Preflight validates resources and host readiness without allocating a simulator.
It is not task-success evidence.

Before the first live episode, verify imports and strict checkpoint loading with
the Isaac-local Python. This loads and immediately closes the policy; it does
not start Isaac Sim or claim closed-loop success:

```bash
"$PWD/deployment/runtime/isaac-sim-4.5.0/python.sh" \
  scripts/act/verify_official_runtime.py \
  --integration-config "$PWD/deployment/artifacts/models/act/configs/pull_out_key/univtac/integration_config.json" \
  --load-policy
```

Repeat the probe with the adjacent `vision_only` config before including the
matched No-touch request.

Run one Clean episode inside the prepared Isaac environment:

```bash
"$PWD/deployment/runtime/isaac-sim-4.5.0/python.sh" \
  -m robotactile_benchmark.cli live-univtac-run \
  --root "$PWD/deployment" \
  --request "$PWD/deployment/requests/act/pull_out_key/clean.json" \
  --capture-profile preview_v1
```

For a matched group, pass only the request files that were generated. The
runner restores one canonical simulator snapshot between conditions and writes
the receipt before closing Isaac:

```bash
"$PWD/deployment/runtime/isaac-sim-4.5.0/python.sh" \
  -m robotactile_benchmark.cli live-univtac-paired-run \
  --root "$PWD/deployment" \
  --requests \
    "$PWD/deployment/requests/act/pull_out_key/clean.json" \
    "$PWD/deployment/requests/act/pull_out_key/faulted.json" \
    "$PWD/deployment/requests/act/pull_out_key/no_touch.json" \
  --receipt "$PWD/deployment/outputs/act/pull_out_key/paired_receipt.json" \
  --capture-profile paper_full_v1
```

## Formal fault-campaign reporting

The ad hoc matched group above may include the separately trained
`vision_only` No-touch control. The formal ACT fault campaign is narrower: it
reports only paired Clean and Faulted outcomes from the tactile `univtac`
profile. Each pair has one Clean request, 12 executable payload-preserving
fault requests, and explicit unsupported receipts for A1/A2. A1/A2 never enter
the SR denominator, and the report contains neither No-touch nor Restored
metrics.

First capture one task-bound measured no-contact reference. The calibration
skips task `pre_move`, holds the empty gripper for six rendered control cycles,
and never applies the ACT action returned by the policy:

```bash
export ACT_TASK=grasp_classify
export ACT_BASE_REQUEST="$PWD/deployment/requests/act/$ACT_TASK/clean.json"
export ACT_CONFIG="$PWD/deployment/artifacts/models/act/configs/$ACT_TASK/univtac/integration_config.json"
export ACT_RESET_SOURCE="$PWD/deployment/artifacts/live-univtac/act/$ACT_TASK/successful-clean"
export ACT_RESET_REFERENCE="$PWD/deployment/artifacts/reset-references/act/$ACT_TASK/seed1000000-v1.json"
export ACT_RESET_TRAJECTORY="$PWD/deployment/artifacts/reset-trajectories/act/$ACT_TASK/seed1000000-v1.json"

"$PWD/deployment/runtime/isaac-sim-4.5.0/python.sh" \
  scripts/act/run_rest_calibration.py \
  --root "$PWD/deployment" \
  --base-clean-request "$ACT_BASE_REQUEST" \
  --integration-config "$ACT_CONFIG" \
  --initial-seed 1000000 \
  --exogenous-seed 1000000 \
  --request-output "$PWD/deployment/requests/act-rest/$ACT_TASK/request.json" \
  --live-output "$PWD/deployment/artifacts/live-univtac/act-rest/$ACT_TASK" \
  --rest-output "$PWD/deployment/artifacts/rest-references/act/$ACT_TASK"

robotactile build-act-reset-reference \
  --source-artifact "$ACT_RESET_SOURCE" \
  --output "$ACT_RESET_REFERENCE"

"$PWD/deployment/runtime/isaac-sim-4.5.0/python.sh" \
  -m robotactile_benchmark.cli capture-act-reset-trajectory \
  --base-clean-request "$ACT_BASE_REQUEST" \
  --reset-reference "$ACT_RESET_REFERENCE" \
  --output "$ACT_RESET_TRAJECTORY"
```

`ACT_RESET_SOURCE` must be an independently reloadable, validated successful
Clean artifact. `preview_v1` and `paper_full_v1` retain a legacy frame-0
fallback; `metrics_only_v1` is accepted only when it retains the strict
`initial_diagnostics.task.reset_witness`. A timeout, failed score, validator
rejection, or compact artifact without that witness is rejected.
The resulting canonical JSON binds the task/seed pair, dataset, checkpoint,
config, source run/result, observable simulator-state hash, initial native
step, and model-visible qpos8. It is a
qualification reference, not a synthetic reset and not a replacement for a
successful Clean run.

The trajectory command performs exactly one reset and never constructs or
calls the ACT policy. During this calibration only, the successful reference's
arm solution seeds CuRobo so the four official `pre_move` plans use the same IK
branch. Publication requires the source native step and no more than `2e-3`
model-visible qpos max-absolute drift (radians for arm joints, meters for the
gripper channel). The captured native step and qpos become the physical replay
anchor. The full observation hash remains diagnostic because GPU-rendered
RGB/tactile bytes need not be identical across independent Isaac processes.
The saved dense position/velocity
sequence is replayed through the upstream `move`, `_step`, delay, render,
contact, and tactile physics path; it is not a joint teleport. This separates
successful-source provenance from brittle cross-process pixel equality.
Formal replay still fails closed on any native-step or `1e-5` qpos mismatch.
Clean and all Faulted cells then use one exact in-process canonical snapshot,
and no Faulted cell runs unless the first Clean episode succeeds and validates.

Generate, execute, and report one severity without restarting Isaac between
cells:

```bash
export ACT_CAMPAIGN="$PWD/deployment/requests/act-fault-campaigns/act-$ACT_TASK-s5-v1"
export ACT_MANIFEST="$PWD/deployment/artifacts/models/act/configs/$ACT_TASK/univtac/artifact_manifest.json"

robotactile generate-act-fault-campaign \
  --output "$ACT_CAMPAIGN" \
  --campaign-id "act-$ACT_TASK-s5-v1" \
  --base-clean-request "$ACT_BASE_REQUEST" \
  --artifact-manifest "$ACT_TASK=$ACT_MANIFEST" \
  --deployment-root "$PWD/deployment" \
  --severity 5 \
  --operator-seed-master 20260830 \
  --fault-start-index 0 \
  --fault-stop-index 300 \
  --rest-reference "$ACT_TASK=$PWD/deployment/artifacts/rest-references/act/$ACT_TASK" \
  --reset-reference "$ACT_RESET_REFERENCE" \
  --reset-trajectory "$ACT_RESET_TRAJECTORY"

"$PWD/deployment/runtime/isaac-sim-4.5.0/python.sh" \
  -m robotactile_benchmark.cli run-act-fault-campaign \
  --campaign-root "$ACT_CAMPAIGN" \
  --integration-config "$ACT_CONFIG" \
  --capture-profile paper_full_v1

robotactile report-act-fault-campaign \
  --campaign-root "$ACT_CAMPAIGN" \
  --output "$PWD/deployment/outputs/reports/act-$ACT_TASK-s5-v1"
```

The 13 live requests share the base Clean `runtime_dir`; changing it per fault
cell changes UniVTAC's UIPC workspace and is rejected. The first canonical
reset replays the qualified dense `pre_move` trajectory and then records the
actual reset witness before `policy.reset()` or the first inference. If its
observable simulator state, native step, or qpos8 differs from the trajectory's
captured replay anchor, execution
raises `qualification_reset_not_viable`; no policy action and no Faulted cell
is executed. The witness remains in
`transition_trace.json.initial_diagnostics.task.reset_witness` for every
capture profile. The runner then exports and validates Clean. If Clean is not
successful it raises
`clean_baseline_unqualified` and does not run the 12 Faulted cells. Diagnose or
replace that reset-bound campaign instead of resuming its fault cells.
The tactile ACT checkpoint is loaded once for the pair and its temporal state
is reset for every cell; it is not reconstructed 13 times.
The same pooling rule applies to missing-only resume: one installed ACT
runtime is leased sequentially to every missing cell. The task-run receipt is
published before native Isaac application shutdown, so a clean
`SimulationApp.close()` process exit cannot leave a fully written campaign
without its execution inventory.

F6 has an additional phase prerequisite: the affected Clean window must
contain contact followed by a real `release` sample. A trajectory that remains
in `sustained_contact` until task success is not an F6 failure or success. Its
F6 artifact is validator-ineligible with `NO_RELEASE_SAMPLE`, remains outside
the SR denominator, and keeps a full all-operator report from becoming a
complete claim. Select a prequalified task/episode with a contact-to-release
phase for an all-operator paper claim; never synthesize a release frame or
weaken the validator.

The report exposes only Clean SR, Faulted SR,
`delta SR = Clean SR - Faulted SR`, and
`retention = Faulted SR / Clean SR`. Infrastructure failures, validator
failures, unsupported contracts, and missing artifacts are separate
dispositions rather than model failures. Require `complete=true` before using
the values as a complete campaign claim. The Python API is
`build_act_fault_campaign_report`; see
[Model integrations](../../docs/model_integrations.md#5-report-an-act-fault-campaign)
for the evidence boundary.
