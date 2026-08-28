# Reproducible benchmark workflow

This document connects N0 Clean request generation, ACT robustness-matrix
orchestration, and paper reporting without changing evidence levels. The
current public primary-matrix generator is ACT-only; it must not be presented
as an N0 fault-campaign entry point. Commands are run from the RoboTactile
repository root.

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

## Two different matrices

The repository uses the word *matrix* for two related but non-interchangeable
artifacts:

- the `pull_out_key` request set contains four matched requests for one
  operator/severity/restoration point;
- `MatrixManifest(kind=primary_14x5_blind)` contains 70 operator-severity
  comparisons and 142 unique executions: one shared clean baseline, one shared
  no-touch baseline, 70 persistent-fault cells, and 70 restored cells.

`matrix_receipt.json` from the generator proves only that four requests and
their identities were generated. `matrix_summary.json` from `run_matrix`
proves that every requested primary cell reached an explicit terminal state.

## Frozen clean-only campaigns

A clean baseline is frozen independently of the four-condition matrix. One
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

## 1. Generate a four-condition live request set

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
  --fault-start 16 \
  --restoration-index 180
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
fault_manifests/restored.json
policy_artifacts/univtac.json
policy_artifacts/vision_only.json
requests/clean.json
requests/faulted.json
requests/no_touch.json
requests/restored.json
matrix_receipt.json
```

When `--rest-reference-artifact` is supplied, the output additionally contains
`rest_references/{rest_reference.json,no_contact_validation.json,root_receipt.json}`.

All conditions share one `pair_key` and tactile base-system manifest. Clean,
faulted, and restored select the tactile `univtac` profile. No-touch selects
the independently identified `vision_only` checkpoint/config while preserving
the base-system comparison identity. A required rest-reference operator fails
closed unless `--rest-reference-artifact` is present, strict-loadable, bound to
`pull_out_key`, and its content hash is embedded in both fault manifests.

## 2. Run the no-allocation deployment preflight

Before simulator allocation, validate one generated request together with the
two pinned external checkouts, official ACT artifacts, NVIDIA visibility, and
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

## 3. Execute the four official ACT requests from one canonical state

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
    "$REQUEST_ROOT/requests/restored.json" \
  --receipt "$REQUEST_ROOT/paired_execution_receipt.json" \
  --config "$ACT_CONFIG"
```

The command performs one upstream reset, captures one process-local UIPC/PhysX
snapshot, restores it before each later condition, and rejects any exact
multimodal state mismatch before policy delivery. It prints one canonical JSON
summary and independently reloads every condition artifact. The per-condition
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
points, 142 unique execution cells (one shared clean, one shared no-touch, 70
faulted, and 70 restored), 140 exact fault manifests, and a self-validating
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
  --restoration-index 180 \
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
├── fault_manifests/<cell_sha256>.json  # 140 exact manifests
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
from robotactile_benchmark.trials import RestorationMode

manifest = build_primary_matrix_manifest(
    matrix_id="pull-out-key-primary-v1",
    clean=clean_trial,
    fault_manifests=fault_manifests_14_by_5,
    restoration_index=180,
    restoration_mode=RestorationMode.VALID_STREAM_RESUME,
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

### Optional registered recovery evidence

Recovery lag is not inferred from terminal success. An evaluator that has a
registered, task-and-phase-matched progress signal may publish one sidecar for
each restored artifact:

```python
from robotactile_benchmark.reporting import (
    POLICY_TASK_PROGRESS_SIGNAL_ID,
    RecoveryEvidence,
    RecoveryEvidenceBinding,
    write_recovery_evidence,
)

binding = RecoveryEvidenceBinding(
    clean_live_artifact_root_sha256=clean_root,
    restored_live_artifact_root_sha256=restored_root,
    task=restored_cell.task,
    pair_key=restored_cell.pair_key,
    operator_id=restored_cell.operator_id,
    severity_level=restored_cell.severity_level,
    restoration_index=restored_cell.trial.restoration_index,
)
evidence = RecoveryEvidence.evaluate(
    binding=binding,
    signal_id=POLICY_TASK_PROGRESS_SIGNAL_ID,
    clean_envelope=phase_matched_clean_quality,
    restored_quality=restored_quality,
    tolerance=0.02,
    consecutive_steps=5,
)
write_recovery_evidence(
    Path("deployment/outputs/matrices/<matrix-id>/recovery"), evidence
)
```

`report-matrix` independently reloads and recomputes the lag. If the sidecar is
absent, recovery remains ineligible; if its identities, signal, cached lag, or
canonical bytes disagree, reporting fails closed. This evidence remains
`unqualified_evaluator_recovery_signal_v1` until the evaluator itself passes a
separate simulator qualification.

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

The command currently accepts only a complete primary matrix. Focused phase or
restoration grids fail closed because they require a registered comparison and
recovery identity before pooling. The report contains:

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
- Recovery summaries do not convert an unrecovered trial into zero lag.

## 8. Isaac qualification boundary

The task/action receipt alone is not an Isaac-qualified policy result.
Qualification v3 adds eight task-local source/parity bindings, and a Qualified
Clean paper bundle additionally requires a complete frozen `paper_v1` campaign,
strict aggregation, success predicates, attempt v3, and exact matching N0
rank-0 and Isaac-child attestations. Only that dual-attested publication gate
can remove `simulator_not_qualified`; v1/v2 cannot. Fault-matrix and real-robot
claims still require their own acceptance evidence.
