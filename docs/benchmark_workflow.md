# Reproducible benchmark workflow

This document connects N0 Clean request generation, N0 and FTP-1 Clean/Faulted
robustness, ACT robustness-matrix orchestration, and reporting without changing
evidence levels. N0, FTP-1, and ACT use separate public campaign contracts: the
ACT primary matrix must not be used as an N0 or FTP-1 entry point. Commands are
run from the RoboTactile repository root.

Prepare and activate the hash-locked core environment before running the
source-tree commands below:

```bash
bash scripts/bootstrap_pip.sh
source .venv/bin/activate
export ROBOTACTILE_DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
export DEPLOY_ROOT="$ROBOTACTILE_DEPLOY_ROOT"
robotactile deployment init --root "$DEPLOY_ROOT"
export UNIVTAC_ROOT="$DEPLOY_ROOT/sources/UniVTAC"
export CHECKPOINT_ROOT="$DEPLOY_ROOT/artifacts/models/act"
export ACT_CONFIG="$CHECKPOINT_ROOT/integration_config.json"
export ISAAC_SIM_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
```

See [Deployment layout](deployment_layout.md),
[External dependencies](external_dependencies.md), and
[Model integrations](model_integrations.md) for ownership, pins, and config
generation.

## Distinct campaign and matrix contracts

The repository exposes four related but non-interchangeable artifacts:

- an N0 fault-campaign bundle contains one Clean request per frozen pair,
  Clean/Faulted requests for its supported operator cells, and typed receipts
  for contract cells that cannot execute;
- an FTP-1 robustness group contains one Clean request, 12 payload-preserving
  Faulted requests, A1/A2 N/A records, and one ordered same-snapshot plan;
- the `pull_out_key` request set contains three matched requests for one
  operator/severity point;
- `MatrixManifest(kind=primary_14x5_blind)` contains 70 operator-severity
  comparisons and 72 unique executions: one shared clean baseline, one shared
  no-touch baseline, and 70 persistent-fault cells.

`generation_receipt.json` proves only that an N0 request
bundle was materialized. `matrix_receipt.json` likewise proves only that the
ACT requests and identities were generated. A run receipt or report must be
loaded separately before making any execution claim.

## Frozen clean-only campaigns

A clean baseline is frozen independently of the three-condition matrix. One
clean episode may later be paired with all fault conditions sharing its trial
identity; it is not rerun once per operator or severity. Build the authoritative
manifest from already generated canonical requests. For large campaigns,
`--request-root` recursively discovers only files named `request.json` in
sorted order and rejects any symlink or duplicate trial identity:

Generate the complete official N0 request set and its authoritative manifest
directly from the pinned task registry and task-specific integration configs:

```bash
python scripts/n0_twam/generate_clean_campaign_requests.py \
  --root "$DEPLOY_ROOT" \
  --campaign-id n0-official-clean-pilot-v1 \
  --protocol pilot_v1 \
  --trials-per-task 10 \
  --sampling-contract univtac_official_v1 \
  --official-eval-seed 0 \
  --replacement-reserve-per-task 2
```

The public execution entry point is the all-task runner shown below. It owns
each task shard, starts the matching task-specific N0 server, waits for the
official websocket protocol, runs Isaac in a fresh process per seed, and stops
only the process group it created. Direct shard and child-attestation flags are
internal plumbing, not user-facing source-bound workflow arguments.

Fresh Isaac processes remain the default execution contract. A campaign may
explicitly select `--worker-contract same_task_worker_v1` to amortize Isaac
startup across consecutive episodes of one task. This worker is strictly
task-local: task identity, N0 serve pool, normalizer, source binding, and N0
server attestation cannot change during its lifetime. The all-task owner still
crosses a process boundary between tasks; this option is not a cross-task hot
reload mechanism.

Before enabling that contract on a new Isaac/UniVTAC deployment, run the live
same-application reuse probe. It launches one Isaac application, constructs two
fresh runtimes for the same task, and requires both seeded `reset + observe`
sequences to finish after the native renderer/timeline teardown and USD-stage
reconstruction barrier. The probe does not load a policy and is not a Clean
episode or success-rate measurement.

```bash
PROBE_ROOT="$DEPLOY_ROOT/outputs/probes/insert-tube-same-app-v1"
mkdir -p "$PROBE_ROOT/runtimes"

"$DEPLOY_ROOT/runtime/isaac-sim-4.5.0/python.sh" \
  scripts/live_univtac/probe_same_task_app_reuse.py \
  --upstream-root "$DEPLOY_ROOT/sources/UniVTAC" \
  --runtime-root "$PROBE_ROOT/runtimes" \
  --output "$PROBE_ROOT/receipt.json" \
  --task insert_tube \
  --initial-seed 4000000 \
  --initial-seed 4000001
```

Use a new probe directory for every attempt. Acceptance requires
`status=passed`, `runtime_count=2`, `completed_runtime_count=2`, and
`same_process_confirmed=true`; a failed second construction is infrastructure
evidence and must not be counted as a model outcome.

Formal `pilot_v1` and `paper_v1` persistent execution requires
`--reset-equivalence-dir PATH`. Before any shard is launched, the owner requires
a regular, non-symlink `<task>.json` reset-equivalence receipt for every
campaign task. Each proof path and SHA256 is passed to its matching task shard
and recorded in the master run receipt. A `diagnostic_v1` persistent run may
omit this directory, but then it produces no formal persistent-session evidence
and cannot be promoted to a pilot or paper claim. If a diagnostic supplies the
directory, every task proof is bound and verified with the same strict rules.
These proofs establish fresh-process versus same-process equivalence for task
reset, initial observation, policy/KV reset, temporal buffers, pending actions,
and seeded simulator state. They are infrastructure evidence, never additional
episodes or replacements for the frozen Clean denominator. Omitting a proof
from pilot/paper, mixing tasks within one worker, or supplying the proof
directory to the default `fresh_process_v1` contract fails before execution.

```bash
python scripts/n0_twam/run_clean_campaign_all_tasks.py \
  --root "$DEPLOY_ROOT" \
  --manifest "$CAMPAIGN_MANIFEST" \
  --execution-profile claim \
  --qualification "$QUALIFICATION" \
  --gpus 0 \
  --run-id n0-persistent-pilot-v1 \
  --worker-contract same_task_worker_v1 \
  --reset-equivalence-dir \
    "$DEPLOY_ROOT/artifacts/deployment/persistent-reset-equivalence"
```

Before Clean execution, qualify each frozen task and the EE8 action contract
once. This is a compatibility gate, not a statistical experiment:

The production N0 action contract is
`robotactile_n0_training_60hz_ee_v1`: cuRobo supplies a feasible final joint
target, then the simulator advances exactly two 120 Hz ticks and renders one
60 Hz endpoint. `univtac_stock_ee_v1` and the historical
`robotactile_fixed_endpoint_v1` alias are diagnostic/reference contracts and
cannot authorize a `pilot_v1` or `paper_v1` campaign.

```bash
bash scripts/live_univtac/qualify_univtac_all_tasks.sh \
  --root "$DEPLOY_ROOT" \
  --gpu 0 \
  --action-spec ee8_absolute \
  --repetitions 1 \
  --campaign-id n0-ee8-qualification-v1
```

This v1 receipt is only the reset/action-contract base evidence. Build
qualification v3 with `build_source_bound_qualification.py` after collecting
one passing observation-parity artifact for each of the eight tasks and the
four required install receipts. V3 has no single top-level source binding:
every ordered task owns its source and parity binding. Task normalizer and serve
bundle hashes may differ; code, external sources, checkpoint, config, prompt,
action, and input-profile identity must agree. The builder is strict,
no-clobber, and fail-closed. Legacy v1/v2 qualifications remain readable but
cannot authorize paper promotion.

For the bounded eight-task deployment diagnostic, generate exactly one Clean
episode per task and run the task-specific N0 servers sequentially:

```bash
python scripts/n0_twam/generate_clean_campaign_requests.py \
  --root "$DEPLOY_ROOT" \
  --campaign-id n0-clean-all8-once-v1 \
  --protocol diagnostic_v1 \
  --master-seed 20260823 \
  --trials-per-task 1

python scripts/n0_twam/run_clean_campaign_all_tasks.py \
  --root "$DEPLOY_ROOT" \
  --manifest "$DEPLOY_ROOT/requests/clean-campaigns/n0-clean-all8-once-v1/campaign_manifest.json" \
  --execution-profile quick \
  --gpus 0 \
  --run-id all8-once-v1 \
  --max-new-trials-per-task 1
```

The `quick` profile intentionally skips qualification and forces a fresh
process with at most one new trial per task. It writes an unqualified diagnostic
receipt and cannot publish a paper result. Supplying an optional v3
qualification upgrades the attempt binding, but does not make a one-trial
diagnostic statistically sufficient. N0 rank 0 and the Isaac child still write
their ordinary execution evidence.

By default, `quick` uses `metrics_only_v1`: it preserves the terminal metric,
action trace, transition diagnostics, identities, and hashes but omits the
full RGB/tactile/proprio trace. `diagnostic` defaults to `preview_v1`, which
also stores at most 64 deterministic keyframes for visualization. Use
`--capture-profile paper_full_v1` to opt either profile back into full storage.
The all-task runner automatically passes `--allow-compact-capture` only to a
diagnostic report; the default report loader and all paper publication paths
remain full-only.

For unattended deployment, `run_clean_all_tasks_once.sh` runs this quick profile
immediately. `--qualification PATH` remains optional for deployments that want
the stronger source binding; only then does the wrapper wait for that receipt.

Generate a paper candidate inventory with the released evaluator's task-local
consecutive seed sequence and an explicitly preregistered exception reserve:

```bash
python scripts/n0_twam/generate_clean_campaign_requests.py \
  --root "$DEPLOY_ROOT" \
  --campaign-id n0-official-clean-paper-v1 \
  --protocol paper_v1 \
  --trials-per-task 100 \
  --sampling-contract univtac_official_v1 \
  --official-eval-seed 0 \
  --replacement-reserve-per-task 10
```

The reserve size is part of the frozen manifest and must be selected before
execution; it is not changed after observing failures. The command writes,
without clobbering,
`requests/clean-campaigns/<campaign-id>/campaign_manifest.json`. It binds the
task registry, policy kind, official evaluator seed, task-local candidate
order, exception-replacement contract, every realized seed, request-file hash,
trial/pair/run identities, full horizon, and future artifact path.

`robotactile clean-campaign-build` remains available for legacy diagnostic
request trees. A legacy SHA-seeded `paper_v1` manifest is loadable for backward
compatibility but is not paper-claim eligible.

The frozen protocols have intentionally different claim boundaries:

- `diagnostic_v1` accepts a task subset with at least one trial per task;
- `pilot_v1` requires all eight tasks and exactly 10 trials per task;
- `paper_v1` requires the frozen eight-task registry, exactly 100 valid outcomes
  per task, the consecutive UniVTAC seed contract, exception replacement, and
  each task's complete observation horizon. Preregistered reserve candidates
  are outside the denominator unless an earlier candidate raises an exception.

Diagnostic and pilot manifests can never authorize a paper result. A
`paper_v1` manifest only authorizes the sampling protocol; it does not qualify
the simulator or artifacts.

Request generation defaults to `initial_state_policy=official_reproduction`.
Under that contract, reset-time success/early-stop signals are retained as
diagnostics but do not independently reject the episode. Opting into
`replace_initial_terminal_v1` defines a separate robustness diagnostic; its
results must not be compared directly with the public 84.5% value.

After execution, aggregate every planned trial:

```bash
robotactile clean-campaign-report \
  --root "$DEPLOY_ROOT" \
  --manifest "$DEPLOY_ROOT/requests/clean-campaigns/n0-official-clean-paper-v1/campaign_manifest.json"
```

For v2 official campaigns the strict loader scans each task in candidate order.
The hard watchdog covers the full child lifecycle: startup, reset, inference,
action execution, artifact export, and teardown. Official N0 requests carry
`wall_timeout_role=infrastructure_watchdog_v1`, so wall-clock expiry cannot be
serialized as a model outcome. Ordinary success, task failure, early stop, and
timeout at the frozen action/observation horizon are valid outcomes and enter
the denominator. Only a classified execution exception with matching lifecycle
evidence is bound to an `exception_replaced` attempt receipt, excluded from the
denominator, and followed by the next consecutive seed in a fresh N0 server
lifecycle. Once 100 valid outcomes are reached, any attempted later candidate
is an overrun error and all remaining candidates are `unused_reserve`. Invalid
artifacts, malformed or ambiguous receipts, and identity/hash mismatches are
protocol errors, never replacement events. By default an incomplete campaign
exits with status 2 and writes no summary; `--allow-partial` is explicitly
non-claimable.

The summary reports per-task eligible success with Wilson intervals, equal-task
macro success with a task-stratified seeded bootstrap, pooled success, terminal
status counts, missing/ineligible/protocol-invalid inventories, and hashes of
all accepted artifact roots and attempt receipts. Individual live artifacts
remain unqualified. `robotactile clean-campaign-publish` removes
`simulator_not_qualified` only when a complete `paper_v1` summary, qualification
v3, every attempt-v3 receipt, and both runtime attestations are exact matches.
Legacy qualification v1/v2, diagnostic/pilot protocols, partial summaries, and
any mismatch remain fail-closed. This implemented gate is not evidence that a
new GPU campaign has already run.

Requests that omit `wall_timeout_role` retain the historical
`scoring_boundary_v1` behavior solely for artifact compatibility. They are not
valid inputs for a newly generated official N0 Clean campaign.

## N0-TWAM Clean/Faulted robust campaign (P0-P3)

This is the public N0 robustness path. It is deliberately separate from the
ACT primary matrix and accepts only canonical N0 Clean requests as its frozen
pair sources. Generation performs no simulator work. Execution strict-loads
the generated bundle, restores one canonical simulator snapshot for the pair,
runs Clean first, and then runs the executable Faulted cells without changing
the trial identity. Existing artifact or receipt directories are never
overwritten.

### P0: frozen applicability and execution contracts

The N0 campaign keeps all 14 paper operator IDs in its contract inventory, but
only the following 12 have a live N0 delivery contract:

- Fidelity: `F1_global_response_drift`,
  `F2_spatial_sensitivity_loss`, `F3_persistent_surface_artifact`,
  `F4_local_nonresponsive_patch`, `F5_contact_shape_distortion`,
  `F6_history_residual_imprint`, and `F7_high_load_saturation`;
- Temporal: `T1_fixed_source_delay`, `T2_held_last_freeze`, and
  `T3_inter_sensor_skew`;
- Context: `C1_sensor_identity_misrouting` and
  `C2_frame_misregistration`.

`A1_stream_absence` and `A2_frame_erasure` are explicit
`unsupported_contract` cells for the released N0 integration. N0 requires both
tactile streams at its input boundary, so the generator writes a typed
unsupported receipt, generates no live request for that cell, and claims no
simulator execution. It never converts absence into a black, resting, or
duplicated image. These receipts remain visible in completeness accounting but
are excluded from score-eligible Faulted outcomes.

For F1, F2, F3, F4, F6, and registered-pixel C2, every selected task must bind
one separately measured rest-reference artifact. The generator rejects missing,
extra, task-mismatched, modified, or symlinked references. One
`operator_template_seed` is derived per frozen pair/operator and is reused
across severities, so an S1/S5 comparison changes dose rather than spatial or
temporal placement.

Live delivery also depends on task phase, not only model input compatibility:

- F6 requires a delivered contact imprint and a later release sample in the
  affected window;
- F7 requires a delivered high-load/detail sample in the affected window.

Freeze these phase requirements from Clean/contact diagnostics before the
Faulted outcomes are inspected. If a task does not contain a required phase,
repeat `--operator` to generate only the preregistered applicable subset. The
strict validator records a missed phase as `SIGNATURE_NOT_DELIVERED` (and a
more specific failure code); reporting keeps the cell outside the score
denominator. It is neither a model failure nor evidence of robustness. Never
drop an operator merely because its Faulted episode failed the task.

N0 fault execution should explicitly bind:

```text
action_execution_contract = robotactile_n0_training_60hz_ee_v1
```

This is the released training-aligned absolute EE8 contract: cuRobo produces a
feasible final joint target, the simulator advances two 120 Hz physics ticks,
and one 60 Hz observation endpoint is rendered. The other registered EE paths
are diagnostics and must not be mixed into one comparison.

Capture volume is orthogonal to action execution:

| `--capture-profile` | Persisted evidence | Intended use |
|---|---|---|
| `metrics_only_v1` | outcome, actions, terminal state, diagnostics, and hashes | P1 gate and broad P3 diagnostics |
| `preview_v1` | metrics plus at most 64 deterministic keyframes | P2 visual pilot |
| `paper_full_v1` | complete RGB/tactile/proprio/action trace | qualified claim preparation |

The fault-pair CLI defaults to `paper_full_v1`; the commands below select the
smaller diagnostic profiles explicitly. Compact capture does not alter the
score, but `metrics_only_v1` cannot be used to create a video. Every current
pair-run receipt remains
`unqualified_paired_n0_fault_execution_v1` with
`simulator_qualification_claimed=false`; request generation or a local report
does not promote it to paper evidence.

### P1: all-8 Clean gate

First run exactly one diagnostic Clean episode for each frozen UniVTAC task.
This detects task/reset/action/observation misalignment before faults are added:

```bash
python scripts/n0_twam/generate_clean_campaign_requests.py \
  --root "$DEPLOY_ROOT" \
  --campaign-id n0-clean-all8-p1-v1 \
  --protocol diagnostic_v1 \
  --master-seed 20260828 \
  --trials-per-task 1

python scripts/n0_twam/run_clean_campaign_all_tasks.py \
  --root "$DEPLOY_ROOT" \
  --manifest "$DEPLOY_ROOT/requests/clean-campaigns/n0-clean-all8-p1-v1/campaign_manifest.json" \
  --execution-profile quick \
  --gpus 0 \
  --run-id n0-clean-all8-p1-v1 \
  --max-new-trials-per-task 1

robotactile clean-campaign-report \
  --root "$DEPLOY_ROOT" \
  --manifest "$DEPLOY_ROOT/requests/clean-campaigns/n0-clean-all8-p1-v1/campaign_manifest.json"
```

The P1 infrastructure gate requires eight valid task outcomes, no missing
task, and no protocol-invalid or infrastructure-classified episode. The
observed Clean success count and rate are reported as measured; model failures
are not rerun to raise the rate and this one-episode-per-task diagnostic is not
a paper estimate. Choose and record the P2 task/pair only after this gate. The
example below uses `insert_tube`; if a different task is preregistered, replace
all task-bound paths consistently.

### P2: one frozen-pair L3 pilot

Select the P1 Clean request and a calibration-only rest reference. The fault
window must be fixed from the task horizon before observing Faulted outcomes:

```bash
export N0_TASK=insert_tube
export N0_INTEGRATION_CONFIG="$DEPLOY_ROOT/artifacts/models/n0_twam/configs/$N0_TASK/integration_config.json"
export N0_SOURCE_ROOT="$DEPLOY_ROOT/sources/N0-TWAM"
export N0_BASE_CLEAN_REQUEST="$DEPLOY_ROOT/requests/clean-campaigns/n0-clean-all8-p1-v1/trials/$N0_TASK/0000/request.json"
export N0_REST_REFERENCE="$DEPLOY_ROOT/artifacts/rest-references/$N0_TASK-empty-gripper-v1"
export FAULT_START_INDEX='<preregistered-inclusive-control-index>'
export FAULT_STOP_INDEX='<preregistered-exclusive-control-index>'
export P2_ROOT="$DEPLOY_ROOT/requests/fault-campaigns/n0-$N0_TASK-l3-p2-v1"
```

Start the task-specific official N0 server in a separately owned
terminal/process group:

```bash
bash scripts/n0_twam/serve_univtac.sh \
  --root "$DEPLOY_ROOT" \
  --task "$N0_TASK" \
  --gpus 0 \
  --port 29601
```

Then capture a measured empty-gripper reference using seeds reserved for the
calibration split:

```bash
"$ISAAC_SIM_PATH/python.sh" scripts/n0_twam/run_rest_calibration.py \
  --root "$DEPLOY_ROOT" \
  --base-clean-request "$N0_BASE_CLEAN_REQUEST" \
  --integration-config "$N0_INTEGRATION_CONFIG" \
  --n0-source-root "$N0_SOURCE_ROOT" \
  --n0-host 127.0.0.1 \
  --n0-port 29601 \
  --initial-seed 900001 \
  --exogenous-seed 900002 \
  --request-output "$DEPLOY_ROOT/requests/calibration/$N0_TASK-empty-gripper-v1/request.json" \
  --live-output "$DEPLOY_ROOT/artifacts/live-univtac/calibration/$N0_TASK-empty-gripper-v1" \
  --rest-output "$N0_REST_REFERENCE"
```

The command preserves the official constructor/reset until the task `pre_move`
boundary, then holds the undeformed empty gripper while sampling at the native
60 Hz endpoint cadence. Its receipt states `model_actions_applied=false` and
`formal_evaluation_reset_modified=false`. This artifact is calibration
evidence only and has no task success outcome.

Generate the immutable P2 bundle after the rest artifact is finalized:

```bash
robotactile generate-n0-fault-campaign \
  --output "$P2_ROOT" \
  --campaign-id "n0-$N0_TASK-l3-p2-v1" \
  --base-clean-request "$N0_BASE_CLEAN_REQUEST" \
  --severity 3 \
  --operator-seed-master 20260828 \
  --fault-start-index "$FAULT_START_INDEX" \
  --fault-stop-index "$FAULT_STOP_INDEX" \
  --rest-reference "$N0_TASK=$N0_REST_REFERENCE"
```

The default operator selection creates one Clean live request, 12 L3 Faulted
live requests, and two unsupported receipts.

For a new seed-variable early-onset protocol, replace the fixed
`--fault-start-index` argument with `--fault-onset-mode early_random_onset_v1`
and `--fault-onset-max-index 8`, and set `--fault-stop-index` to the base request's
`max_observation_steps`. The generated manifest freezes one nonzero onset per
task/exogenous seed, shared by every operator and severity for that pair; its
index lies in the first third of the planned horizon and no later than frame 8.
T1/T3 causally hold frame 0 until enough history exists, then achieve the
specified lag/skew. T2 freezes from onset through the end of the window.
This is a distinct protocol from fixed/full-episode onset: use a new campaign
ID and report actual active/changed observations and action-query exposure.
An exceptionally short episode may finish before onset; do not relabel or rerun
that outcome as if exposure had occurred. Tactile-null/structural-absence
ablations remain fixed full-horizon protocols and reject randomized onset.

Then run the generated pair once from Isaac Python and report it from the core
environment:

```bash
"$ISAAC_SIM_PATH/python.sh" -m robotactile_benchmark.cli \
  run-n0-fault-campaign \
  --campaign-root "$P2_ROOT" \
  --integration-config "$N0_INTEGRATION_CONFIG" \
  --n0-source-root "$N0_SOURCE_ROOT" \
  --n0-host 127.0.0.1 \
  --n0-port 29601 \
  --capture-profile preview_v1 \
  --action-execution-contract robotactile_n0_training_60hz_ee_v1

robotactile report-n0-fault-campaign \
  --campaign-root "$P2_ROOT" \
  --output "$DEPLOY_ROOT/outputs/reports/n0-$N0_TASK-l3-p2-v1" \
  --bootstrap-seed 20260828
```

The pair runner executes all live cells selected in that immutable bundle; it
does not cherry-pick one operator after seeing an outcome. With one pair, the
report is a diagnostic table. Its deterministic bootstrap and exact paired-test
outputs do not make the sample statistically sufficient for a paper claim;
missing or ineligible pairs remain explicit.

### P3: S1/S5 bracket on the same frozen pair

Use the exact same `N0_BASE_CLEAN_REQUEST`, task, pair identity, fault window,
rest reference, and action contract. Generate a new no-clobber bundle containing
only the two bracket severities. For the example `insert_tube` trace below,
the phase gate excludes F6 and F7 while retaining the ten operators whose
signatures were delivered; other tasks must freeze their own applicable set:

```bash
export P3_ROOT="$DEPLOY_ROOT/requests/fault-campaigns/n0-$N0_TASK-s1-s5-p3-v1"

robotactile generate-n0-fault-campaign \
  --output "$P3_ROOT" \
  --campaign-id "n0-$N0_TASK-s1-s5-p3-v1" \
  --base-clean-request "$N0_BASE_CLEAN_REQUEST" \
  --severity 1 \
  --severity 5 \
  --operator C1_sensor_identity_misrouting \
  --operator C2_frame_misregistration \
  --operator F1_global_response_drift \
  --operator F2_spatial_sensitivity_loss \
  --operator F3_persistent_surface_artifact \
  --operator F4_local_nonresponsive_patch \
  --operator F5_contact_shape_distortion \
  --operator T1_fixed_source_delay \
  --operator T2_held_last_freeze \
  --operator T3_inter_sensor_skew \
  --operator-seed-master 20260828 \
  --fault-start-index "$FAULT_START_INDEX" \
  --fault-stop-index "$FAULT_STOP_INDEX" \
  --rest-reference "$N0_TASK=$N0_REST_REFERENCE"

"$ISAAC_SIM_PATH/python.sh" -m robotactile_benchmark.cli \
  run-n0-fault-campaign \
  --campaign-root "$P3_ROOT" \
  --integration-config "$N0_INTEGRATION_CONFIG" \
  --n0-source-root "$N0_SOURCE_ROOT" \
  --n0-host 127.0.0.1 \
  --n0-port 29601 \
  --capture-profile metrics_only_v1 \
  --action-execution-contract robotactile_n0_training_60hz_ee_v1

robotactile report-n0-fault-campaign \
  --campaign-root "$P3_ROOT" \
  --output "$DEPLOY_ROOT/outputs/reports/n0-$N0_TASK-s1-s5-p3-v1" \
  --bootstrap-seed 20260828
```

This example produces one Clean plus 20 executable Faulted runs. In general,
an explicitly selected set of `N` live operators at S1/S5 produces `1 + 2N`
live cells. Omitting `--operator` instead expands the complete contract
inventory: one Clean, 24 executable Faulted runs, and four A1/A2 unsupported
receipts. Reporting uses `degradation = Clean SR - Faulted SR` and
`retention = Faulted SR / Clean SR`; retention is not clipped and can exceed
one when a Faulted cell happens to improve the binary outcome. The generated
report bundle contains `summary.json`, `per_task.csv`, `operator_cells.csv`,
and `report_receipt.json`, each bound to the strict source artifact hashes.
P0-P3 establish an auditable diagnostic path; they do not state that a GPU run
has already occurred or that the sample size is publication-ready.

### Black-frame tactile-null model-reliance gate

Run this gate before tuning any destructive operator. The
`diagnostic_tactile_null_black_v1` registry is restricted to one full-window,
two-sensor realization and cannot be used as a paper severity registry. Set
`--fault-start-index 0`; set `--fault-stop-index` exactly equal to the base
request's `max_observation_steps`. This diagnostic intentionally does not
consume a rest-reference artifact.

The paired runner executes the ordinary Clean request first and the same
N0-TWAM checkpoint with all-zero black tactile frames second, retaining one
Isaac process and enforcing exact reset equivalence. RGB, proprioception,
instruction, task predicate, and action execution are unchanged. The delivery
validator independently reconstructs zero arrays with the source payload's
shape and dtype and requires `black_frame_max_abs_value == 0`.

Interpret the report's `macro_fault_success_rate` as tactile-null SR only when
`spec.severity_registry == "diagnostic_tactile_null_black_v1"`; the CLI also exposes
the explicit alias `tactile_null_success_rate`. This is a model-reliance
diagnostic, not the matched vision-only policy condition and not a paper
S1--S5 robustness result.

### Tensor-free observed-tactile-absence diagnostic

Black images preserve the tactile tensor branch and may remain a recognizable
input pattern. For the stronger structural test, launch the same pinned N0
server with `--enable-observed-tactile-absence`, then generate a campaign with
`--diagnostic-observed-tactile-absence`. The generator accepts only A1, S5,
both sensor slots, `start_index=0`, and `stop_index=max_observation_steps`.

The benchmark delivery removes both payloads. The policy/client omit the
`tactile` field during infer and KV commit, while the server overlay passes
`tactile_cond_drop=true` into a zero-tactile-token model path matching the
training implementation. The mode is frozen at the first post-reset request;
switching modes inside an episode is rejected. Clean runs on the
overlay-enabled server call the original upstream implementation unchanged.

Report this condition as `observed_tactile_absent_v1`. It is neither the
black-frame diagnostic nor evidence for the published retrained `w/o observed`
ablation. A paired one-seed result is useful to verify model dependence, but is
not a paper-level success-rate estimate.

### Destructive stress-max diagnostic across tasks

The formal `provisional_engineering_v2` S5 values remain immutable. To test
whether an apparently robust policy is simply insensitive to weak injections,
use the separately versioned, non-paper `diagnostic_stress_max_v1` profile:

```bash
robotactile generate-n0-fault-campaign \
  --output "$STRESS_ROOT" \
  --campaign-id "n0-all8-stress-max-v1" \
  --base-clean-request "$TASK_1_REQUEST" \
  --base-clean-request "$TASK_2_REQUEST" \
  --diagnostic-stress-max \
  --severity 5 \
  --operator-seed-master 20260829 \
  --fault-start-index 16 \
  --fault-stop-index 300 \
  --rest-reference "task_1=$TASK_1_REST" \
  --rest-reference "task_2=$TASK_2_REST"
```

Repeat the request and rest-reference bindings for every selected task. The
profile uses implementation-limit doses and preserves the ordinary manifest,
validator, no-clobber, and reporting contracts. Unsupported A1/A2 remain typed
`unsupported_contract`; phase-inapplicable F6/F7 remain validator-ineligible.
Therefore the report must separate requested, executable, eligible, model
failure, infrastructure failure, and validator failure counts. This profile is
useful for pipeline sensitivity diagnosis, not for paper S1--S5 claims.

## FTP-1 Clean/Faulted robustness workflow

FTP-1 uses its own first-class `PolicyAdapter` and official ZMQ worker. It does
not reuse the ACT matched vision-only system or the N0 websocket/KV-cache
contract. The pinned release has six task checkpoints and requires two tactile
tensors. Its formal executable fault set is therefore:

```text
F1 F2 F3 F4 F5 F6 F7 T1 T2 T3 C1 C2
```

A1 structural stream absence and A2 frame erasure are N/A for this fixed
model-input contract. They remain visible in the plan rather than being
converted to black, rest, held-last, or duplicated images. This preserves the
meaning of Availability faults and prevents a payload-preserving perturbation
from being mislabeled as absence.

### 1. Freeze source, task artifacts, and rest evidence

Install and configure exactly one released task before materializing requests:

```bash
export FTP1_TASK=insert_hole
export FTP1_CONFIG="$DEPLOY_ROOT/artifacts/models/ftp1_policy/configs/$FTP1_TASK/integration_config.json"

bash scripts/ftp1_policy/install_official_runtime.sh --root "$DEPLOY_ROOT"
robotactile integrations configure ftp1-policy \
  --root "$DEPLOY_ROOT" --task "$FTP1_TASK"
robotactile integrations doctor \
  --model ftp1_policy --root "$DEPLOY_ROOT" --task "$FTP1_TASK" \
  --config "$FTP1_CONFIG"
```

The checkpoint must already be downloaded from
`MJJJJ1064/ftp1_univtac_finetune` at revision
`620ac69b4fffd2341300cfef1b1d224d56710ed3`. The source must be the pinned
`michaelyuancb/ftp1-policy` commit
`89fa681d6c014cce28300946b7526db808e0b1c1`. See
[Model integrations](model_integrations.md#ftp-1-integration) for task-specific
download paths and the worker command.

F1, F2, F3, F4, F6, and registered-pixel C2 require a task-bound certified rest
reference. Generate it from a separate calibration trace, not the evaluation
Clean episode, and bind the resulting JSON artifact to the group. F5/F7 also
remain subject to contact/high-load applicability validation; F6 additionally
needs a real contact-to-release phase. A validator-ineligible cell is not a
model failure and is not silently counted in SR.

### 2. Materialize one immutable matched group

The following diagnostic profile selects the strongest separately registered
engineering doses. It does not modify or stand in for a paper S1--S5 registry:

```bash
export FTP1_GROUP="$DEPLOY_ROOT/requests/ftp1-policy/$FTP1_TASK/diagnostic-stress-max-v1"
export FTP1_REST='<absolute certified rest-reference artifact root>'
export DATASET_SHA256='<exact 64-hex dataset identity>'

python scripts/ftp1_policy/prepare_robustness_group.py \
  --root "$DEPLOY_ROOT" \
  --config "$FTP1_CONFIG" \
  --task "$FTP1_TASK" \
  --severity-profile diagnostic_stress_max \
  --dataset-sha256 "$DATASET_SHA256" \
  --initial-seed 3000000 \
  --exogenous-seed 3000000 \
  --max-control-cycles 255 \
  --max-observation-steps 256 \
  --wall-timeout-s 1800 \
  --device cuda:0 \
  --rest-reference "$FTP1_REST" \
  --output-root "$FTP1_GROUP"
```

Generation is no-clobber and allocates no simulator. It writes a source-bound
plan receipt, the Clean request, one fault manifest and request for every
executable operator, and explicit A1/A2 N/A entries. The plan owns the exact
ordered request list; do not create an apparently equivalent list by editing
JSON. For a paper protocol, replace the diagnostic profile with the
preregistered severity profile and freeze the full task/seed grid before the
first run.

### 3. Execute Clean and Faulted from one snapshot

Start the task-specific FTP-1 worker first. Read the exact paths from the plan,
then invoke one paired runner from Isaac's Python with Clean first and all 12
generated Faulted request paths in the exact plan order:

```bash
export FTP1_PLAN="$FTP1_GROUP/robustness_plan_receipt.json"
FTP1_REQUESTS=()
while IFS= read -r relative_path; do
  FTP1_REQUESTS+=("$FTP1_GROUP/$relative_path")
done < <(python -c 'import json,sys; [print(x) for x in json.load(open(sys.argv[1]))["paired_run"]["request_paths"]]' "$FTP1_PLAN")
export FTP1_PAIRED_RECEIPT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paired_run"]["receipt_path"])' "$FTP1_PLAN")"

"$ISAAC_SIM_PATH/python.sh" -m robotactile_benchmark.cli \
  live-univtac-paired-run \
  --root "$DEPLOY_ROOT" \
  --config "$FTP1_CONFIG" \
  --requests "${FTP1_REQUESTS[@]}" \
  --receipt "$FTP1_PAIRED_RECEIPT" \
  --capture-profile metrics_only_v1 \
  --ftp1-source-root "$DEPLOY_ROOT/sources/ftp1-policy" \
  --ftp1-endpoint tcp://127.0.0.1:5561
```

The runner performs a single canonical reset, snapshots the full
simulator state, restores that snapshot before every condition, and checks the
pre-delivery multimodal witness. It never reuses the policy's temporal state:
each condition resets FTP-1 while preserving the same simulator initial state.

At every observation FTP-1 reinfers, skips raw chunk index 0, admits the next
20 predictions to its `K=0.01` temporal ensemble, and returns one absolute
qpos8 action. Consequently T1--T3 are injected causally into the actual online
observation stream rather than applied afterward to a saved trajectory.

### 4. Read and report the result boundary

Retain the plan receipt, fault manifests, canonical requests, paired execution
receipt, and every strict live artifact. At minimum report:

- Clean terminal status and `score_success`;
- each Faulted terminal status and `score_success`;
- validator eligibility and delivered-dose evidence per operator;
- Clean-minus-Faulted paired outcome, separately from infrastructure errors;
- task, seed, severity profile, source commit, checkpoint revision, and all
  bound SHA-256 identities.

One matched group is a pipeline diagnostic, not a statistically meaningful
Success Rate. **CODE** means the integration and deterministic contracts pass;
**OFFLINE** means the worker loaded and inferred without Isaac; **CLOSED-LOOP**
means a strict Isaac artifact actually executed and terminated. A paper claim
requires a preregistered multi-seed denominator, qualification, complete cells,
and source-bound aggregation. Until those artifacts exist, the correct result
is “integration implemented; live metric pending,” not 100% or 0% SR.

## 0. Build a rest-reference calibration artifact when required

F1, F2, F3, F4, F6, and registered-pixel C2 require a frozen no-contact RGB
reference. RoboTactile never substitutes a black image, the first test frame,
or an evaluation episode heuristic. First execute a separate `clean` request
from a declared development, validation, or calibration split, then build the
reference from its strict live artifact:

```bash
python -m robotactile_benchmark.cli generate-calibration-request \
  --task pull_out_key \
  --dataset-split calibration \
  --split-manifest-sha256 '<frozen-calibration-split-manifest-sha256>' \
  --base-system-id official-univtac-act.pull_out_key.univtac.policy_last.v1 \
  --checkpoint-sha256 '<univtac-policy-last-sha256>' \
  --config-sha256 acdab30e50fa7280918804c4a196f75a6db6e854a3c7a86a0ddd84791c533397 \
  --initial-seed 17 \
  --exogenous-seed 29

CUDA_VISIBLE_DEVICES=1 "$ISAAC_SIM_PATH/python.sh" \
  -m robotactile_benchmark.cli live-univtac-run \
  --request "$DEPLOY_ROOT/requests/calibration/pull_out_key-calibration-i17-e29/request.json" \
  --config "$ACT_CONFIG"

python -m robotactile_benchmark.cli build-rest-references \
  --source-live-artifact "$DEPLOY_ROOT/artifacts/live-univtac/calibration/pull_out_key-calibration-i17-e29" \
  --dataset-split calibration \
  --minimum-consecutive-free-records 5 \
  --output "$DEPLOY_ROOT/artifacts/rest-references/pull_out_key-v1"
```

The selector uses the evaluator-private UniVTAC phase provenance produced by
the frozen depth-hysteresis tracker. Both sensors must report `FREE` for a
consecutive interval. The longest interval is selected; equal intervals choose
the earliest, and the frozen RGB reference is the interval's center frame.
The three-file artifact binds the source live root, trial/dataset identity,
phase-tracker config, complete qualified record-hash interval, sensor
calibration hashes, and selected payload bytes.

`generate-calibration-request` itself performs no simulator or model work. Its
two-file bundle (`request.json` and `calibration_request_receipt.json`) binds
the declared split-manifest hash, seeds, policy identities, runtime paths, and
resulting trial hash. Re-running it is byte-idempotent; different content at
the same destination is rejected.

This command emits
`evidence_level=unqualified_univtac_no_contact_calibration_v1` and
`simulator_qualification_claimed=false`. It does not turn a software contract
test or an unqualified live trace into physical or Isaac-qualified evidence. The
source root hash must remain externally pinned.

## 1. Generate a three-condition live request set

Record the hashes independently before invoking the generator:

```bash
python scripts/live_univtac/generate_pull_out_key_matrix.py \
  --tactile-checkpoint-sha256 '<univtac-policy-last-sha256>' \
  --vision-checkpoint-sha256 '<vision-only-policy-last-sha256>' \
  --stats-sha256 '<dataset-stats-sha256>' \
  --encoder-sha256 '<encoder-sha256>' \
  --initial-seed 17 \
  --exogenous-seed 29 \
  --operator T1_fixed_source_delay \
  --severity 3 \
  --fault-start 16
```

For a rest-reference operator, pass the strict artifact directory explicitly:

```bash
python scripts/live_univtac/generate_pull_out_key_matrix.py \
  ... \
  --operator F2_spatial_sensitivity_loss \
  --rest-reference-artifact "$DEPLOY_ROOT/artifacts/rest-references/pull_out_key-v1"
```

The command prints the output directory and a receipt-file hash. The directory
contains:

```text
trial_set_manifest.json
fault_manifests/persistent.json
policy_artifacts/univtac.json
policy_artifacts/vision_only.json
requests/clean.json
requests/faulted.json
requests/no_touch.json
matrix_receipt.json
```

When `--rest-reference-artifact` is supplied, the output additionally contains
`rest_references/{rest_reference.json,no_contact_validation.json,root_receipt.json}`.

All conditions share one `pair_key` and tactile base-system manifest. Clean
and faulted select the tactile `univtac` profile. No-touch selects
the independently identified `vision_only` checkpoint/config while preserving
the base-system comparison identity. A required rest-reference operator fails
closed unless `--rest-reference-artifact` is present, strict-loadable, bound to
`pull_out_key`, and its content hash is embedded in the fault manifest.

## 2. Run the no-allocation deployment preflight

Before simulator allocation, validate one generated request together with the
pinned UniVTAC checkout, official ACT artifacts, NVIDIA visibility, and
Isaac's bundled Python:

```bash
REQUEST_ROOT='<generator-output-directory>'

robotactile preflight-live \
  --request "$REQUEST_ROOT/requests/clean.json" \
  --config "$ACT_CONFIG"
```

The command rejects non-canonical requests, unreleased or dirty pins, artifact
drift, an unsupported host, missing NVIDIA visibility, or an incomplete Isaac
Python environment. It writes
`evidence_level=live_preflight_no_simulator_execution_v1` and keeps both
simulator claim flags false. It does not import the learned policy, allocate
Isaac, or establish task success.

## 3. Execute the three official ACT requests from one canonical state

Select the physical GPU outside the request. The request itself uses logical
`cuda:0`.

```bash
REQUEST_ROOT='<generator-output-directory>'

CUDA_VISIBLE_DEVICES=1 "$ISAAC_SIM_PATH/python.sh" \
  -m robotactile_benchmark.cli live-univtac-paired-run \
  --requests \
    "$REQUEST_ROOT/requests/clean.json" \
    "$REQUEST_ROOT/requests/faulted.json" \
    "$REQUEST_ROOT/requests/no_touch.json" \
  --receipt "$REQUEST_ROOT/paired_execution_receipt.json" \
  --root "$DEPLOY_ROOT"
```

The command performs one upstream reset, captures one process-local UIPC/PhysX
snapshot, restores it before each later condition, and rejects any exact
multimodal state mismatch before policy delivery. It prints one canonical JSON
summary and independently reloads every condition artifact. The per-condition
profile is resolved from the canonical `univtac` or `vision_only` integration
config under `artifacts/models/act/configs/<task>/<profile>/`; a single shared
`--config` is deliberately rejected for this mixed-profile group. The
per-condition
evidence remains exactly:

```json
{"evidence_level":"unqualified_live_univtac_execution_v1","simulator_qualification_claimed":false}
```

This tier records that the configured live path ran and the trace is internally
consistent. It does not certify Isaac installation, task semantics, physical
fault realism, or simulator qualification.

## 4. Materialize and run the ACT primary matrix

Generate both canonical inputs directly from the registered operators, the
frozen UniVTAC task horizon, the matched ACT/no-touch identities, and one
strictly loaded calibration artifact. The generator creates 70 comparison
points, 72 unique execution cells (one shared clean, one shared no-touch, and
70 faulted), 70 exact fault manifests, and a self-validating
receipt. It performs no simulator or policy execution.

```bash
python -m robotactile_benchmark.cli generate-primary-matrix \
  --task pull_out_key \
  --dataset-sha256 "$TRIAL_SPLIT_SHA256" \
  --tactile-checkpoint-sha256 "$TACTILE_CHECKPOINT_SHA256" \
  --no-touch-checkpoint-sha256 "$VISION_CHECKPOINT_SHA256" \
  --stats-sha256 "$STATS_SHA256" \
  --encoder-sha256 "$ENCODER_SHA256" \
  --initial-seed 17 \
  --exogenous-seed 29 \
  --fault-start-index 16 \
  --matrix-id pull_out_key-i17-e29 \
  --rest-reference-artifact "$CALIBRATION_ROOT"
```

`--dataset-sha256` is the content hash of the frozen trial/split identity, not
an informal dataset label. The rest-reference directory must pass the strict
calibration loader and match the requested task; the generator never creates a
synthetic substitute. The default fault scope is both tactile sensors. The
default stop and action/observation budgets come from the packaged UniVTAC task
horizon. A T1 warm-up violation, task mismatch, incomplete artifact, unknown
member, or existing non-identical output fails closed.

The resulting request directory contains:

```text
deployment/requests/primary-matrix/pull_out_key-i17-e29/
├── matrix_manifest.json
├── live_matrix_run_config.json
├── primary_matrix_receipt.json
├── fault_manifests/<cell_sha256>.json  # 70 exact manifests
└── rest_references/                    # copied calibrated artifact
```

The public executor consumes the generated `matrix_manifest.json` and the
`live_matrix_run_config.schema.json`-conforming resource document. The latter
binds the official ACT artifact root, statistics/encoder hashes, runtime
devices, and one exact resource entry for every content-addressed cell. First
perform a no-execution materialization check:

```bash
PRIMARY_REQUEST_ROOT="$DEPLOY_ROOT/requests/primary-matrix/pull_out_key-i17-e29"
MATRIX_ID="pull_out_key-i17-e29"
PRIMARY_MATRIX_OUTPUT="$DEPLOY_ROOT/outputs/matrices/$MATRIX_ID"

python -m robotactile_benchmark.cli run-live-matrix \
  --matrix-manifest "$PRIMARY_REQUEST_ROOT/matrix_manifest.json" \
  --run-config "$PRIMARY_REQUEST_ROOT/live_matrix_run_config.json" \
  --max-new-cells 0
```

Then execute the complete matrix as one paired batch from a fresh output:

```bash
CUDA_VISIBLE_DEVICES=1 "$ISAAC_SIM_PATH/python.sh" \
  -m robotactile_benchmark.cli run-live-matrix \
  --matrix-manifest "$PRIMARY_REQUEST_ROOT/matrix_manifest.json" \
  --run-config "$PRIMARY_REQUEST_ROOT/live_matrix_run_config.json"
```

Production execution intentionally does not support a positive
`--max-new-cells`: splitting cells across simulator processes would invalidate
the exact matched-state contract. A completed, strictly verified matrix may be
reused as a whole; a partial independent-reset output must be moved aside and
regenerated under a new matrix lineage. The CLI prints executed, reused, and
pending cell counts plus every terminal status. It never upgrades the
underlying artifacts:
`evidence_level=unqualified_live_univtac_paired_matrix_v1` and
`simulator_qualification_claimed=false` remain explicit.

The same orchestration remains available as a typed Python API. Construct the
primary manifest from one clean `TrialManifest` and exactly 70 blind
`FaultManifest` values:

```python
from pathlib import Path

from robotactile_benchmark.matrix import (
    LiveMatrixCellResources,
    LiveMatrixExecutionTemplate,
    PairedLiveMatrixExecutor,
    build_primary_matrix_manifest,
    run_matrix_batch,
)
from robotactile_benchmark.execution import LivePolicyKind
from robotactile_benchmark.execution.official_act import (
    build_official_act_live_binding,
    make_official_act_policy_factory,
)

manifest = build_primary_matrix_manifest(
    matrix_id="pull-out-key-primary-v1",
    clean=clean_trial,
    fault_manifests=fault_manifests_14_by_5,
    no_touch_system_id="official-univtac-act.pull_out_key.vision_only.policy_last.v1",
    no_touch_checkpoint_sha256=vision_checkpoint_sha256,
    no_touch_config_sha256=vision_config_sha256,
)


def resolve_resources(cell):
    condition = cell.trial.condition.value
    return LiveMatrixCellResources(
        fault_manifest_path=(
            request_root / "fault_manifests" / f"{cell.sha256}.json"
            if cell.fault_manifest is not None
            else None
        ),
        rest_references_path=(
            rest_reference_artifact_root / "rest_reference.json"
            if cell.operator_id in rest_reference_operator_ids
            else None
        ),
        matched_no_touch_artifact_path=(
            vision_checkpoint if condition == "no_touch" else None
        ),
    )


template = LiveMatrixExecutionTemplate(
    policy_kind=LivePolicyKind.ACT,
    max_control_cycles=300,
    max_observation_steps=301,
    execute_action_steps=1,
    wall_timeout_s=1800.0,
    upstream_root=univtac_root,
    runtime_root=runtime_root,
    act_device_name="cuda:0",
    simulator_device="cuda:0",
    launcher_args={"enable_cameras": True, "headless": True},
)


def official_act_policy_factory(loaded):
    binding = build_official_act_live_binding(
        loaded.request,
        artifact_root=official_act_artifact_root,
        stats_sha256=stats_sha256,
        encoder_sha256=encoder_sha256,
    )
    return make_official_act_policy_factory(binding)(loaded)


executor = PairedLiveMatrixExecutor(
    matrix_output=Path("deployment/outputs/matrices/pull-out-key-primary-v1"),
    template=template,
    resource_resolver=resolve_resources,
    policy_factory=official_act_policy_factory,
)

result = run_matrix_batch(
    Path("deployment/outputs/matrices/pull-out-key-primary-v1"),
    manifest,
    executor,
)
```

Here `official_act_policy_factory` is the hash-pinned factory constructed from
`build_official_act_live_binding` and `make_official_act_policy_factory` for the
cell's loaded request. The resource resolver must point to the exact manifest
bytes represented by each cell; `materialize_live_matrix_request` reloads and
cross-checks them before any backend allocation.

The production executor exports each trace, strictly reloads it, publishes it
at `artifacts/<root_receipt_sha256>/`, and writes
`paired_execution_receipt.json`. Exceptions are retained as `crash`;
unsupported contracts and validator rejections remain distinct.
`run_matrix_batch` writes canonical, content-addressed cell receipts only after
the complete paired batch returns. It reuses only a fully completed summary;
partial cross-process resume is deliberately rejected.

For report compatibility, every artifact-bearing cell must be stored at:

```text
<matrix-output>/artifacts/<root_receipt_sha256>/
```

The directory must be the complete software-contract or unqualified-live bundle
whose evidence level appears in the cell receipt. A receipt does not authorize
renaming another bundle or substituting cached score fields.

The final primary layout is:

```text
deployment/outputs/matrices/<matrix-id>/
  matrix_manifest.json
  matrix_summary.json
  paired_execution_receipt.json
  cells/<cell_sha256>.json
  artifacts/<root_receipt_sha256>/...
```

## 5. Freeze reporting choices

Reporting specs must be canonical JSON. The following creates a conservative
spec that withholds TGR because no separate matched-control qualification has
yet been claimed:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.reporting import ReportingSpec

spec = ReportingSpec(
    system_id="official-univtac-act.pull_out_key.univtac.policy_last.v1",
    matched_control_qualified=False,
    minimum_clean_gain=0.05,
    primary_operator_ids=tuple(sorted(CORE_OPERATOR_IDS)),
    primary_severity_levels=(1, 2, 3, 4, 5),
    bootstrap_resamples=10000,
    bootstrap_seed=20260821,
    confidence_level=0.95,
)
path = Path(os.environ["DEPLOY_ROOT"]) / "requests/reporting/act-pull-out-key.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    + "\n",
    encoding="utf-8",
)
PY

export REPORTING_SPEC="$DEPLOY_ROOT/requests/reporting/act-pull-out-key.json"
```

`configs/reporting_spec.example.json` is the tracked canonical example for the
same default ACT system. Keep generated experiment choices in `deployment/`
rather than overwriting tracked source.

Set `matched_control_qualified=true` only when the matched vision-only control
has passed its separately defined qualification protocol. Merely loading or
executing the no-touch checkpoint is not sufficient. If the flag is false, or
if `clean_sr - no_touch_sr < minimum_clean_gain`, TGR is reported as unavailable
with an explicit reason.

## 6. Generate the paper report

```bash
MATRIX_ID="pull_out_key-i17-e29"
PRIMARY_MATRIX_OUTPUT="$DEPLOY_ROOT/outputs/matrices/$MATRIX_ID"

python -m robotactile_benchmark.cli report-matrix \
  --matrix-manifest "$PRIMARY_REQUEST_ROOT/matrix_manifest.json" \
  --matrix-output "$PRIMARY_MATRIX_OUTPUT" \
  --reporting-spec "$REPORTING_SPEC" \
  --output "$DEPLOY_ROOT/outputs/reports/$MATRIX_ID"
```

The command currently accepts only a complete primary matrix. Focused or
incomplete grids fail closed before pooling. The report contains:

```text
summary.json
per_task.csv
operator_cells.csv
native_dose_curves.csv
native_dose_curves.svg
summary_table.tex
report_receipt.json
```

`report_receipt.json` pins the reporting spec, summary, every rendered member,
and all source artifact roots. Its evidence level is
`derived_from_verified_closed_loop_results`; it does not upgrade those source
artifacts to simulator-qualified evidence.

## 7. Metric interpretation

- Success rate (SR) is reported per task and macro-averaged over tasks.
- Paired delta-SR compares faulted outcomes with the matched clean outcomes.
- No-touch-relative delta separates tactile corruption from removal of the
  tactile modality.
- TGR is computed only for a qualified control and a sufficiently large clean
  gain; otherwise it is withheld.
- Native-dose curves stay within one operator. Severity levels are not treated
  as a shared physical scale across operators.
- Task-stratified paired bootstrap resamples pairs within each task and then
  gives tasks equal macro weight.
- Exact two-sided McNemar tests use paired binary outcomes; Holm adjustment is
  applied across the registered operator-severity family.

## 8. Isaac qualification boundary

The task/action receipt alone is not an Isaac-qualified policy result.
Qualification v3 adds eight task-local source/parity bindings, and a Qualified
Clean paper bundle additionally requires a complete frozen `paper_v1` campaign,
strict aggregation, success predicates, attempt v3, and exact matching N0
rank-0 and Isaac-child attestations. Only that dual-attested publication gate
can remove `simulator_not_qualified`; v1/v2 cannot. Fault-matrix and real-robot
claims still require their own acceptance evidence.
