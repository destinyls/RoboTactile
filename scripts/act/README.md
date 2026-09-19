# ACT operational entry points

Run these commands from the RoboTactile repository root. The first-class
`act` integration uses the pinned UniVTAC `policy/ACT` source and official ACT
artifacts; it does not use the legacy WorldArena/ACTStrict wrapper.

## Matched decision-stress diagnostic

`python -m scripts.act.run_decision_stress` prepares, runs and reports the
two-task/three-seed Clean + F1/F3/T2/T3 pilot. Supply an exported N0
`robotactile_decision_stress_cross_model_reference_v1` JSON to `prepare --reference`;
fault seeds, onset, gain ramp and temporal source maps are preserved exactly,
not resampled using ACT's model identity. The existing all-14 campaign is unchanged.

```bash
python -m scripts.act.run_decision_stress prepare \
  --reference /absolute/path/to/n0_reference.json \
  --rest-root /absolute/path/to/exported-n0-rest-references \
  --output /absolute/path/to/new-act-pilot \
  --deployment-root /absolute/path/to/deployment \
  --config-root /absolute/path/to/deployment/artifacts/models/act/configs \
  --isaac-python /absolute/path/to/isaac-sim-4.5.0/python.sh

python -m scripts.act.run_decision_stress run \
  --output /absolute/path/to/new-act-pilot \
  --package-root /absolute/path/to/isolated-benchmark-package

python -m scripts.act.run_decision_stress report \
  --output /absolute/path/to/new-act-pilot
```

Use the repo-local Python environment for preparation/supervision. Execution
uses the configured Isaac Python and existing `live-univtac-paired-run` entrypoint:
one simulator session and one pooled ACT model per task/seed, five conditions.
ACT retains official chunk50 temporal aggregation and executes one QPOS8 action
per observation. This is a native-policy diagnostic, not a shared-chunk benchmark.
F1/F3 use the same task-specific certified rest artifacts as the N0 pilot;
their content hashes must match the frozen fault manifests. No new rest images
are synthesized or silently substituted for a different calibration.
Up to 64 preview frames are retained. Completed groups are skipped; partially
executed groups require explicit no-clobber continuation, never silent reruns.
`pilot_results.json` separates pending/invalid trials from eligible SR and reports
paired Clean-success-to-fault-failure counts. Three seeds remain exploratory.

## Frozen identities

| Asset | Identity | Local owner |
|---|---|---|
| `univtac/UniVTAC` source | `05bcd3edb92237107efa40105292a24f1a9fd761` | `deployment/sources/UniVTAC` |
| `byml/UniVTAC` ACT artifacts | `172331dbbce95bc04c3e59b22f32dc72ba5561ae` | `deployment/artifacts/models/act` |

The artifact dataset card declares MIT. The artifacts remain external to the
Apache-2.0 RoboTactile wheel; review the exact upstream terms before
redistribution.

## Published checkpoint tree

The frozen Hugging Face revision contains:

```text
checkpoints/
├── encoder.pth
└── <task>/
    ├── univtac/
    │   ├── policy_last.ckpt
    │   ├── dataset_stats.pkl
    │   ├── metadata.json
    │   └── log.log
    └── vision_only/
        ├── policy_last.ckpt
        ├── dataset_stats.pkl
        ├── metadata.json
        └── log.log
```

The eight task IDs are `grasp_classify`, `insert_HDMI`, `insert_hole`,
`insert_tube`, `lift_bottle`, `lift_can`, `pull_out_key`, and
`put_bottle_in_shelf`.

## Install official artifacts

Inspect the deterministic plan without network access or filesystem writes:

```bash
export DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
export ACT_ARTIFACT_ROOT="$DEPLOY_ROOT/artifacts/models/act"

python scripts/act/install_official_artifacts.py \
  --artifact-root "$ACT_ARTIFACT_ROOT" \
  --dry-run
```

The default installation selects all eight tasks and only the `univtac`
profile:

```bash
python scripts/act/install_official_artifacts.py \
  --artifact-root "$ACT_ARTIFACT_ROOT"
```

Select tasks and profiles by repeating their options. This example installs
both profiles for one task and retains upstream reference files:

```bash
python scripts/act/install_official_artifacts.py \
  --artifact-root "$ACT_ARTIFACT_ROOT" \
  --task pull_out_key \
  --profile univtac \
  --profile vision_only \
  --include-reference
```

`--plan` is an alias for `--dry-run`. `--lock-path` may select an explicit
release lock; the repository default is `integrations/act_artifacts.lock.json`.
`--receipt` may select an explicit receipt path inside the artifact root.
Normal installation verifies the frozen lock, never replaces conflicting bytes, and writes a canonical receipt under
`official_release_receipts/install-<plan-sha256-prefix>.json`.

Runtime files are installed as:

```text
deployment/artifacts/models/act/
├── encoder.pth
├── <task>/<profile>/{policy_last.ckpt,dataset_stats.pkl}
├── references/172331dbbce95bc04c3e59b22f32dc72ba5561ae/
│   └── <task>/<profile>/{metadata.json,log.log}
└── official_release_receipts/*.json
```

The `references/` subtree is created only with `--include-reference`. Its
metadata and logs describe upstream runs and are marked
`upstream_reference_only_no_local_execution`; they are not RoboTactile
execution results.

## Choose the profile

- `univtac` consumes both tactile streams. Use it for Clean/Faulted robustness;
  only this profile is required for those runs.
- `vision_only` is a separately trained model used only for the matched
  no-touch control. A black image, rest image, erased frame, or absent tactile
  stream is not equivalent to this control.

## Configure and probe one task

After installing the pinned UniVTAC source and artifacts:

```bash
export ACT_TASK=pull_out_key
export ACT_PROFILE=univtac

bash integrations/install_univtac.sh

robotactile integrations configure act \
  --root "$DEPLOY_ROOT" \
  --task "$ACT_TASK" \
  --profile "$ACT_PROFILE"

robotactile integrations doctor \
  --model act \
  --root "$DEPLOY_ROOT" \
  --task "$ACT_TASK" \
  --profile "$ACT_PROFILE"

"$DEPLOY_ROOT/runtime/isaac-sim-4.5.0/python.sh" \
  scripts/act/verify_official_runtime.py \
  --integration-config \
  "$ACT_ARTIFACT_ROOT/configs/$ACT_TASK/$ACT_PROFILE/integration_config.json" \
  --load-policy
```

The installer and configuration command do not load the model or start Isaac.
The runtime probe loads and immediately closes the model but records
`isaac_sim_started=false` and `closed_loop_execution_claimed=false`. Only a
qualified live run can produce CLOSED-LOOP evidence.

## Split-provenance boundary

The public release metadata does not provide enough evidence to prove that the
frozen40 validation episodes were excluded from training. Label results from
these weights `ACT-official`. Reserve `ACT-train759` for a separately trained
checkpoint whose immutable data manifest proves train759-only training. Do not
pool these result families or treat upstream `metadata.json` as a locally
executed benchmark result.

## Other scripts

| File | Purpose | Evidence produced |
|---|---|---|
| `install_official_artifacts.py` | Install revision-locked official ACT artifacts | Download/install receipt only |
| `verify_official_runtime.py` | Validate imports and optionally strict-load one configured policy | Runtime probe only |
| `prepare_requests.py` | Freeze task-bound Clean/Faulted and optional matched no-touch requests | Request-generation evidence only |
| `run_rest_calibration.py` | Capture measured empty-gripper tactile references for ACT faults | Calibration evidence only; no policy action is applied |

Formal robustness campaigns use the package CLI after rest calibration:

```bash
robotactile build-act-reset-reference --help
robotactile capture-act-reset-trajectory --help
robotactile generate-act-fault-campaign --help
robotactile run-act-fault-campaign --help
robotactile report-act-fault-campaign --help
```

Before generation, derive one reset reference per task/seed pair from a
strictly reloadable successful Clean artifact. `metrics_only_v1` is supported
only when the artifact retains the strict reset witness; preview/full artifacts
also provide a legacy frame-0 fallback:

```bash
robotactile build-act-reset-reference \
  --source-artifact "$SUCCESSFUL_CLEAN_ARTIFACT" \
  --output "$DEPLOY_ROOT/artifacts/reset-references/act/$ACT_TASK/seed-v1.json"

"$DEPLOY_ROOT/runtime/isaac-sim-4.5.0/python.sh" \
  -m robotactile_benchmark.cli capture-act-reset-trajectory \
  --base-clean-request "$BASE_CLEAN_REQUEST" \
  --reset-reference "$DEPLOY_ROOT/artifacts/reset-references/act/$ACT_TASK/seed-v1.json" \
  --output "$DEPLOY_ROOT/artifacts/reset-trajectories/act/$ACT_TASK/seed-v1.json"
```

Pass both files to generation with repeated `--reset-reference PATH` and
`--reset-trajectory PATH` arguments. One task/seed campaign contains 15
contract cells: one Clean, 12 executable
Faulted cells, and typed unsupported receipts for A1/A2. The runner reuses one
canonical Isaac snapshot and one base Clean UIPC workspace within the
invocation. The exact ACT artifact/profile is loaded once and reset between
sequential cells. The successful Clean reference binds provenance and the
source reset neighborhood. Trajectory calibration performs one reset, loads no
policy, requires the same native step and at most `2e-3` qpos max-absolute
drift (radians for arm joints, meters for the gripper channel), and records the
resulting native step and qpos as the physical replay anchor.
Formal reset replays the dense planner output through upstream physics, uses no
joint teleport, and must match that captured native step plus qpos before
policy reset or inference. The cross-process full-observation hash remains a
diagnostic because rendered bytes may vary; same-process Clean/Faulted snapshot
replay remains exact. It never reruns an existing strictly valid artifact, and it stops
before all Faulted cells when either replay qualification or the first Clean
baseline fails.
See [`examples/act/README.md`](../../examples/act/README.md)
for the complete commands.
