# Reproducible benchmark workflow

This document connects request generation, one-trial execution, primary matrix
orchestration, and paper reporting without changing evidence levels. Commands
are run from the `robotactile_benchmark` source root.

Prepare and activate the hash-locked core environment before running the
source-tree commands below:

```bash
bash scripts/bootstrap_pip.sh
source .venv/bin/activate
robotactile deployment init
export DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
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

## 3. Execute the four official ACT requests

Select the physical GPU outside the request. The request itself uses logical
`cuda:0`.

```bash
REQUEST_ROOT='<generator-output-directory>'

for condition in clean faulted no_touch restored; do
  CUDA_VISIBLE_DEVICES=1 "$ISAAC_SIM_PATH/python.sh" \
    -m robotactile_benchmark.cli live-univtac-run \
    --request "$REQUEST_ROOT/requests/$condition.json" \
    --config "$ACT_CONFIG"
done
```

Each successful invocation prints one canonical JSON line containing the
external artifact-root hash, condition, terminal status, and validation state.
It also independently reloads the artifact before returning. The evidence is
still exactly:

```json
{"evidence_level":"unqualified_live_univtac_execution_v1","simulator_qualification_claimed":false}
```

This tier records that the configured live path ran and the trace is internally
consistent. It does not certify Isaac installation, task semantics, physical
fault realism, or simulator qualification.

## 4. Materialize and run the primary matrix

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

python -m robotactile_benchmark.cli run-live-matrix \
  --matrix-manifest "$PRIMARY_REQUEST_ROOT/matrix_manifest.json" \
  --run-config "$PRIMARY_REQUEST_ROOT/live_matrix_run_config.json" \
  --max-new-cells 0
```

Then execute a bounded batch and resume with the same command:

```bash
CUDA_VISIBLE_DEVICES=1 "$ISAAC_SIM_PATH/python.sh" \
  -m robotactile_benchmark.cli run-live-matrix \
  --matrix-manifest "$PRIMARY_REQUEST_ROOT/matrix_manifest.json" \
  --run-config "$PRIMARY_REQUEST_ROOT/live_matrix_run_config.json" \
  --max-new-cells 8
```

The CLI prints executed, reused, and pending cell counts plus every terminal
status. It never upgrades the underlying artifacts:
`evidence_level=unqualified_live_univtac_matrix_v1` and
`simulator_qualification_claimed=false` remain explicit.

The same orchestration remains available as a typed Python API. Construct the
primary manifest from one clean `TrialManifest` and exactly 70 blind
`FaultManifest` values:

```python
from pathlib import Path

from robotactile_benchmark.matrix import (
    LiveMatrixCellExecutor,
    LiveMatrixCellResources,
    LiveMatrixExecutionTemplate,
    build_primary_matrix_manifest,
    run_matrix,
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


executor = LiveMatrixCellExecutor(
    matrix_output=Path("deployment/outputs/matrices/pull-out-key-primary-v1"),
    template=template,
    resource_resolver=resolve_resources,
    policy_factory=official_act_policy_factory,
)

result = run_matrix(
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

The production executor exports each trace, strictly reloads it, and publishes
it at `artifacts/<root_receipt_sha256>/`. Exceptions are retained as `crash`;
unsupported contracts and validator rejections remain distinct.
`run_matrix` writes canonical, content-addressed cell receipts and reuses only
receipts that pass complete type/hash/cross-link validation. A partial run can
set `max_new_cells`; subsequent calls resume without re-executing valid cells.

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
path = Path("configs/reporting_spec.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    + "\n",
    encoding="utf-8",
)
PY
```

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
python -m robotactile_benchmark.cli report-matrix \
  --matrix-manifest "$PRIMARY_REQUEST_ROOT/matrix_manifest.json" \
  --reporting-spec configs/reporting_spec.json
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

The repository does not currently emit an Isaac-qualified receipt. Such a tier
requires a separate acceptance protocol binding the simulator build, task
assets, GPU/runtime environment, seeds, full matrix roots, validation results,
and success predicates. Until that protocol exists and passes, keep all live
results and derived reports at their declared unqualified evidence level.
