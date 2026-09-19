# Model Integrations

RoboTactile registers five first-class model integrations: `act`, `dream_tac`,
`ftp1_policy`, `n0_twam`, and `n0_vtla`. All implement the same
`PolicyAdapter` lifecycle (`reset`, `infer`, `commit`, `abort`, and `close`) and
feed the same fault injector, closed-loop runner, artifact, and reporting
contracts. Model source and weights remain external to the Apache-2.0 wheel.
Their public orchestration surfaces are intentionally model-specific:
N0-TWAM has the official Clean campaign owner, the current
`generate-primary-matrix` / `run-live-matrix` robustness workflow is ACT-only,
N0-VTLA currently releases only `insert_hole`, and FTP-1 uses a dedicated
same-snapshot Clean/Faulted group for its six released tasks. Dream-Tac exposes
its source-bound adapter and official HTTP wire contract, but remains
`release_ready=false` until checkpoint and inference-parity gates are resolved.

Initialize the [deployment layout](deployment_layout.md) and inspect the static
registry first:

```bash
robotactile deployment init
robotactile integrations list
robotactile setup --model act
```

## ACT end-to-end

The complete ACT chain is:

```text
pinned UniVTAC policy/ACT source
  + revision-pinned official <task>/<profile> model artifacts
  -> deployment/artifacts/models/act/configs/<task>/<profile>/artifact_manifest.json
  -> deployment/artifacts/models/act/configs/<task>/<profile>/integration_config.json
  -> request -> integrations doctor -> preflight-live
  -> live-univtac-paired-run/evaluate -> content-addressed live artifacts
```

The first-class `act` ID names only UniVTAC's pinned
`policy/ACT/act_policy.py` implementation and its `policy_last.ckpt` artifact
family. The older WorldArena `deployment/ACTStrict` wrapper uses a different
`policy_best.ckpt` contract. `integrations/install_act_runtime.sh` is retained
only for legacy workspace compatibility; it is not part of this installation
or a prerequisite for official ACT preflight/execution.

### 1. Install sources

```bash
bash integrations/install_univtac.sh
```

The exact commits, licenses, and immutable hyperlinks are listed in
[External dependencies](external_dependencies.md). The UniVTAC repository root
is Apache-2.0, while the nested `policy/ACT` code carries its own MIT notice and
its DETR subtree carries Apache-2.0; see the third-party notices before
redistribution.

### 2. Install the official artifacts

The official ACT artifacts are published in the Hugging Face dataset
[`byml/UniVTAC`](https://huggingface.co/datasets/byml/UniVTAC/tree/172331dbbce95bc04c3e59b22f32dc72ba5561ae/checkpoints),
frozen at revision `172331dbbce95bc04c3e59b22f32dc72ba5561ae`.
The upstream `checkpoints/` tree contains one shared tactile encoder and eight
task directories:

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

`<task>` is one of `grasp_classify`, `insert_HDMI`, `insert_hole`,
`insert_tube`, `lift_bottle`, `lift_can`, `pull_out_key`, or
`put_bottle_in_shelf`.

The repository-owned installer pins that revision, verifies
`integrations/act_artifacts.lock.json`, and refuses to overwrite different
files. Its default selection installs all eight `univtac` profiles. Run
`--help` for task/profile subsets and the no-network/no-write planning mode:

```bash
export DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"

python scripts/act/install_official_artifacts.py \
  --artifact-root "$DEPLOY_ROOT/artifacts/models/act"
```

For task `pull_out_key`, the resulting runtime layout is:

```text
deployment/artifacts/models/act/
├── encoder.pth
├── pull_out_key/
│   ├── univtac/
│   │   ├── policy_last.ckpt
│   │   └── dataset_stats.pkl
│   └── vision_only/
│       ├── policy_last.ckpt
│       └── dataset_stats.pkl
└── configs/pull_out_key/
    ├── univtac/{artifact_manifest.json,integration_config.json}
    └── vision_only/{artifact_manifest.json,integration_config.json}
```

`univtac` consumes the two tactile streams and is the only profile needed for
Clean/Faulted robustness runs. Faults are injected into this same policy's
online observations. `vision_only` is a separately trained matched `no_touch`
system and is downloaded only when that control is requested; a black image,
resting tactile frame, or structurally absent stream is not a substitute.

The installer may retain upstream `metadata.json` and `log.log` under the
revision-scoped `references/` directory when explicitly requested. Those files
are upstream reference material only: they are not RoboTactile execution
artifacts and do not establish local OFFLINE, CLOSED-LOOP, or task-success
evidence.

The release metadata does not provide enough split provenance to prove that
the frozen40 validation episodes were excluded from ACT training. Report these
weights as `ACT-official`. A separately retrained checkpoint whose data
manifest proves train759-only training should be reported as `ACT-train759`;
do not merge the two result families or infer the latter provenance from the
official filename.

### 3. Generate canonical config

Run once per installed profile. The default output is already profile-specific and uses
the canonical `configs/<task>/<profile>/` directory:

```bash
export DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
export ACT_TASK=pull_out_key

robotactile integrations configure act \
  --root "$DEPLOY_ROOT" \
  --task "$ACT_TASK" \
  --profile univtac

robotactile integrations configure act \
  --root "$DEPLOY_ROOT" \
  --task "$ACT_TASK" \
  --profile vision_only
```

The command computes checkpoint, stats, encoder, upstream source, and training
config hashes from the real files. It validates the clean UniVTAC commit and
official layout, emits canonical JSON, and refuses to replace different
existing output. It does not load Torch or run inference.

The resulting tactile config is exactly
`$DEPLOY_ROOT/artifacts/models/act/configs/$ACT_TASK/univtac/integration_config.json`;
the matched no-touch config uses the adjacent `vision_only` directory. Do not
copy either file to the legacy top-level `artifacts/models/act/integration_config.json`.

### 4. Diagnose, preflight, and execute

```bash
export ACT_CONFIG="$DEPLOY_ROOT/artifacts/models/act/configs/$ACT_TASK/univtac/integration_config.json"
export ACT_VISION_CONFIG="$DEPLOY_ROOT/artifacts/models/act/configs/$ACT_TASK/vision_only/integration_config.json"

"$DEPLOY_ROOT/runtime/isaac-sim-4.5.0/python.sh" \
  scripts/act/verify_official_runtime.py \
  --integration-config "$ACT_CONFIG"

"$DEPLOY_ROOT/runtime/isaac-sim-4.5.0/python.sh" \
  scripts/act/verify_official_runtime.py \
  --integration-config "$ACT_VISION_CONFIG"

robotactile integrations doctor \
  --model act \
  --root "$DEPLOY_ROOT" \
  --task "$ACT_TASK" \
  --profile univtac \
  --config "$ACT_CONFIG"

robotactile preflight-live \
  --request "$REQUEST_PATH" \
  --root "$DEPLOY_ROOT" \
  --config "$ACT_CONFIG"

"$DEPLOY_ROOT/runtime/isaac-sim-4.5.0/python.sh" \
  -m robotactile_benchmark.cli live-univtac-run \
  --request "$REQUEST_PATH" \
  --root "$DEPLOY_ROOT" \
  --config "$ACT_CONFIG"
```

The default runtime probes above validate each canonical artifact and import
NumPy, Torch, Torchvision, and IPython from Isaac-local Python. They do not load
the policy or start Isaac. Add `--load-policy` only when a strict
load-and-immediate-close smoke is desired. Even that opt-in mode records
`isaac_sim_started=false` and `closed_loop_execution_claimed=false`; neither
mode is simulator execution or task-success evidence.

ACT is loaded in-process by Isaac Sim's bundled Python; there is no official
second ACT Python runtime or model server in this integration. Preserve the
Torch/Torchvision versions shipped with the qualified Isaac deployment. Do not
install the upstream ACT conda environment over Isaac or use `pip --upgrade`
to replace Isaac's Torch stack. Missing auxiliary imports must be treated as an
Isaac-local compatibility failure and resolved without changing that stack.

Legacy explicit artifact-root and hash arguments remain compatibility inputs,
but the typed config above is the canonical runtime input. `preflight-live`
does not allocate Isaac or the model. A live run remains
`unqualified_live_univtac_execution_v1` until the separate qualification
protocol passes.

The commands above are single-profile operations, so their explicit
`--config` is valid. A mixed Clean/Faulted/No-touch
`live-univtac-paired-run` must instead receive `--root "$DEPLOY_ROOT"` and omit
`--config`: the runner resolves and cross-validates the canonical `univtac` and
`vision_only` configs for the request task. Supplying one profile's config to a
mixed-profile paired session is rejected.

### 5. Report an ACT fault campaign

The formal ACT robustness campaign uses only the tactile `univtac` profile.
For every matched task/seed pair it plans one Clean cell and all 14 Faulted
operators. Twelve payload-preserving operators are executable; A1 stream
absence and A2 frame erasure are explicit `unsupported_contract` cells because
the frozen ACT interface requires both tactile tensors. Those two cells do not
enter a Success Rate denominator. A `vision_only` No-touch run is a separate
matched control and is not part of this report; there is no Restored condition.

The public workflow is exposed as five commands:

```bash
robotactile build-act-reset-reference --help
robotactile capture-act-reset-trajectory --help
robotactile generate-act-fault-campaign --help
robotactile run-act-fault-campaign --help
robotactile report-act-fault-campaign --help
```

The reset-reference command accepts only a strictly reloadable, validated
successful Clean artifact. A new `metrics_only_v1` artifact is sufficient when
it retains the strict reset witness; preview/full artifacts also support the
legacy frame-0 fallback. It records the source-bound task/seed pair,
dataset/checkpoint/config hashes, source run/artifact/result hashes, observable
simulator-state hash, initial native step, and model-visible qpos8.
Then capture one policy-free dense `pre_move` trajectory per pair and generate
the campaign with repeated `--reset-reference PATH` plus
`--reset-trajectory PATH`; both coverage sets must be exact. Calibration uses
the reference qpos only as a CuRobo IK seed, runs one upstream reset, and
publishes only when the successful-source identity and native step match and
model-visible qpos max-absolute drift is at most `2e-3` (radians for arm joints,
meters for the gripper channel). Its captured native step and qpos form the
physical formal replay anchor. The full-observation hash remains diagnostic,
because it includes GPU-rendered RGB/tactile bytes that need not be identical
across independent Isaac processes. Formal reset replays
every dense command through upstream physics; no teleport is used. A compact
artifact without a reset witness, an
unsuccessful or unvalidated artifact, or an identity-inconsistent source fails
closed. The source reference remains provenance evidence; the formal replay
qpos tolerance remains `1e-5` against the captured anchor.

Generation accepts exactly one severity and all 14 contract operators. The
runner executes Clean plus the 12 live Faulted cells from one canonical Isaac
snapshot and supports missing-only resume without overwriting valid artifacts.
All requests in a pair retain the base Clean `runtime_dir`, because UniVTAC
uses that path as the UIPC physics workspace rather than as a logging-only
directory. A mixed-workspace pair is rejected before Isaac starts. After
backend reset, the ACT trajectory profile requires the captured native step and
qpos8 (`1e-5` tolerance) before it calls `policy.reset()` or performs the first
inference. It records the full observable-state hash and whether it matched,
but cross-process render-byte drift alone does not invalidate an otherwise
exact physical endpoint. A native-step or qpos mismatch raises
`qualification_reset_not_viable`, writes explicit reset
diagnostics, and executes no policy action or Faulted cell. The reset witness is
retained under `initial_diagnostics.task.reset_witness` for all capture
profiles. After the first Clean artifact is exported, the runner requires
`terminal_status=success`, `score_success=true`, and
`validation_passed=true`; otherwise it raises
`clean_baseline_unqualified`, closes the session, and does not execute any
Faulted cell. This prevents reset/planner drift from being reported as fault
robustness. The default backend profile, including N0 paths, still requires
simulator-state hash, native step, and qpos simultaneously.
Within that sequential session, one exact ACT artifact/profile is loaded once;
each condition receives a non-owning episode lease and calls the upstream
`reset()` to clear temporal aggregation. Distinct profiles remain distinct
runtimes and every owned runtime is closed when the paired session exits.
Missing-only execution uses the same policy pool and publishes its task-run
receipt before native Isaac shutdown. This avoids repeated checkpoint
construction and preserves the execution inventory on Isaac builds whose
successful application close exits the interpreter.

`F6_history_residual_imprint` is score-eligible only when the affected Clean
window contains a real release sample after contact. A successful grasp trace
that stays in `sustained_contact` is phase-inapplicable: the validator reports
`NO_RELEASE_SAMPLE`, the cell stays outside the SR denominator, and the report
retains a validator-coverage blocker. For a complete all-operator claim, bind a
prequalified task/episode with an observed contact-to-release phase; do not
fabricate release data or reinterpret the ineligible cell as robustness.
The measured-rest calibration and full campaign commands are in
[`examples/act/README.md`](../examples/act/README.md#formal-fault-campaign-reporting).

This gate detects a different physical endpoint after deterministic dense
`pre_move` replay. Within the paired process, snapshot-state and observation
equivalence remain exact for Clean versus every Faulted cell. If the same
trajectory-bound pair still fails the native-step/qpos gate
across fresh processes, fix the simulator replay contract and publish a new
no-clobber campaign instead of relaxing the tolerance or classifying reset
drift as model failure.

Load the strict campaign bundle and completed live artifacts, then build the
content-bound report:

```python
from robotactile_benchmark.act_fault_campaign.io import (
    load_act_fault_campaign_bundle,
)
from robotactile_benchmark.act_fault_campaign.reporting import (
    build_act_fault_campaign_report,
)
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
)

bundle = load_act_fault_campaign_bundle(campaign_root)
artifacts = {
    cell.artifact_relpath: load_live_univtac_artifact(
        bundle.root / cell.artifact_relpath
    )
    for cell in bundle.manifest.cells
    if cell.artifact_relpath is not None
    and (bundle.root / cell.artifact_relpath).is_dir()
}
report = build_act_fault_campaign_report(bundle.manifest, artifacts)
```

The only performance metrics are Clean SR, Faulted SR,
`delta SR = Clean SR - Faulted SR`, and
`retention = Faulted SR / Clean SR` (undefined when Clean SR is zero).
Only model-completed successes and failures enter SR denominators. Unsupported
contracts, infrastructure failures, validator failures, and missing artifacts
remain separate dispositions. Any infrastructure, validator,
missing-artifact, or pairing blocker makes `report.complete` false; such a
report is diagnostic evidence, not a complete robustness claim.

## N0-TWAM end-to-end

RoboTactile uses the released official N0-TWAM websocket server and the
`NeoteAI/n0-twam-univtac-delta` checkpoint. The single-arm adapter converts the
server's first ten `[xyz, rot6d, gripper]` channels to UniVTAC absolute
`[xyz, quat(wxyz), gripper]` actions.

The low-level live request and streaming injector accept N0 observations, but
this release does not expose a public N0 fault-campaign generator. In
particular, the ACT primary generator cannot be reused: it binds two ACT
checkpoints, including a separately trained matched vision-only control that
does not exist for the frozen N0 release. N0 A1/A2 structural absence remains
`unsupported_contract` because the model requires both tactile streams.

```text
official pinned source + base components + UniVTAC delta checkpoint
  -> task-specific per_robot normalizer + verbatim training prompt
  -> official multitask_server websocket
  -> infer -> execute 12/24 slots -> observe 4/8 keyframes -> KV re-ground
  -> the same UniVTAC fault injector, trace, matrix, and report contracts
```

The policy and simulator intentionally keep separate task instructions. For
`pull_out_key`, the N0 policy receives the checkpoint's verbatim training
prompt (`Untwist and extract a key from a lock`), while the UniVTAC reset uses
the frozen simulator-registry prompt (`Grasp the key and pull it completely out
of the lock.`). Replacing either string with the other changes a released
contract; RoboTactile binds each prompt at its own boundary.

The adapter applies no task-specific transformation to released N0 actions.
For every task, including `grasp_classify`, executable EE8 rows are decoded
directly from the server's native action tensor, and the same unmodified tensor
is committed for KV re-grounding with the `identity_v1` transform.
For execution, RoboTactile retains pinned UniVTAC cuRobo planning, applies the
plan's final joint target, advances two 120 Hz physics ticks, and renders once
per endpoint. The resulting
`robotactile_n0_training_60hz_ee_v1` contract reproduces the released HDF5
collection cadence of 60 Hz endpoints and 20 Hz N0 feedback keyframes. It does
not alter the model action or add a task-specific correction. The upstream
variable-waypoint `BaseTask.take_action(..., action_type="ee")` implementation
is retained as the explicit `univtac_stock_ee_v1` diagnostic/reference path;
the historical `robotactile_fixed_endpoint_v1` name is a diagnostic-only alias.

### 1. Install source and isolated runtime

```bash
bash integrations/install_univtac.sh
bash scripts/n0_twam/install_official_runtime.sh --root "$PWD/deployment"
bash scripts/n0_twam/install_robotactile_client.sh --root "$PWD/deployment"
```

The N0 runtime lives under `deployment/runtime/n0-twam`; system Python and
system CUDA are not modified. The installer uses `python -m venv` when
available and otherwise reuses the repo-local micromamba runtime to create a
pinned Python 3.11 environment. The micromamba fallback resolves the pinned
`pip`, `setuptools`, and `wheel` versions in the environment transaction so it
does not self-replace `pip` afterward on shared filesystems.
The second installer binds the RoboTactile client wheel to the current release
source manifest. Re-run it after synchronizing new RoboTactile source; receipts
are content-addressed, so previous source versions remain auditable.

### 2. Download and prepare official tasks

```bash
"$PWD/deployment/runtime/n0-twam/bin/python" \
  scripts/n0_twam/prepare_official_artifacts.py \
  --root "$PWD/deployment/artifacts/models/n0_twam" \
  --task pull_out_key
```

For the frozen eight-task Clean campaign, prepare every task-specific
normalizer and integration config in one idempotent pass:

```bash
"$PWD/deployment/runtime/n0-twam/bin/python" \
  scripts/n0_twam/prepare_official_artifacts.py \
  --root "$PWD/deployment/artifacts/models/n0_twam" \
  --all-tasks --skip-download
```

The downloader uses immutable Hugging Face revisions, omits the unused base
transformer, selects the task's own normalizer, rebuilds minimal serve-pool
metadata, and publishes a content-addressed config:

```text
deployment/artifacts/models/n0_twam/
├── base/{vae,text_encoder,tokenizer,...}/
├── univtac-delta/{transformer,train_meta.json,norm}/
├── serve-bundle/
├── serve-pools/<task>/{train,norm_stat_per_robot.json,serve_bundle_manifest.json}
└── configs/<task>/{artifact_manifest,integration_config}.json
```

Re-running `--all-tasks --skip-download` validates and reuses every complete
task configuration, then prepares only missing tasks. Task routing is explicit:
`grasp_classify` uses its official HDF5 key, while the other seven tasks use
their frozen Rot6D keys; no task silently falls back to `pull_out_key`.

For the pinned cuRobo v0.7.7 runtime, RoboTactile supplies a narrow planning
compatibility hook for `insert_hole` and `insert_tube`: an upstream
zero-distance grasp enables cuRobo's deterministic 5 cm approach cost metric.
The registered contact frame and final grasp target are unchanged; the hook is
source-manifest-bound and does not modify the pinned UniVTAC checkout.
For the final constrained placements of `insert_hole` and `insert_tube`, the
adapter captures the first approach target and its zero-distance endpoint
before execution. The following 2 mm target is constructed as an exact
continuation of that same geometry: 8 mm from a 10 mm anchor for `insert_hole`,
and 48 mm from a 50 mm anchor for `insert_tube`. This avoids recomputing the
grasp transform from a potentially stale UIPC actor pose. The continuation is
accepted only when the captured translation matches the task contract and its
orientation is unchanged; the unstable
partial-pose trajectory metric is omitted. This affects reset/pre-move planning
only; policy actions, episode horizon, target state, and success predicate
remain unchanged.
If cuRobo reports `IK Fail` for that marked 8 mm local placement, the adapter
retries the same start state and goal once with cuRobo's `partial_ik_opt`
fallback. The retry is logged and no other motion query uses this path.
If both global paths still report `IK Fail`, a final bounded DLS fallback is
allowed only when the marked goal is within 2 cm of the current EE pose, has
quaternion alignment at least 0.999, stays within soft joint limits, and needs
at most 0.15 rad on any joint. It emits an interpolated joint path capped at 64
steps and records all gate values in the runtime log.

### 3. Diagnose and start the server

```bash
robotactile integrations doctor \
  --model n0_twam \
  --task pull_out_key \
  --config "$PWD/deployment/artifacts/models/n0_twam/configs/pull_out_key/integration_config.json"

bash scripts/n0_twam/serve_univtac.sh \
  --root "$PWD/deployment" \
  --task pull_out_key \
  --gpus 0
```

The official fast path requires at least 40 GB on **every selected GPU**. The
transformer is FSDP2-sharded, but each rank first loads the full transformer,
VAE and T5; adding more 24 GB RTX 3090 cards does not remove that initialization
peak. A single >=40 GB GPU uses the official `WORLD_SIZE=1` launch path; multiple
selected GPUs use `torchrun`. This only establishes model-server feasibility:
N0-TWAM and Isaac Sim sharing one GPU still requires an empirical peak-memory
and closed-loop stability qualification. For protocol/model functional smoke
only, 3090-class cards may use:

```bash
bash scripts/n0_twam/serve_univtac.sh \
  --root "$PWD/deployment" \
  --task pull_out_key \
  --gpus 4,5,6,7,8 \
  --debug-offload
```

Debug offload keeps VAE/T5 on CPU and can take minutes per chunk. Never use it
for latency, throughput, or real-time benchmark claims. In another shell:

```bash
"$PWD/deployment/runtime/n0-twam/bin/python" \
  scripts/n0_twam/check_server.py \
  --source-root "$PWD/deployment/sources/N0-TWAM"
bash scripts/n0_twam/install_isaac_client.sh --root "$PWD/deployment"
```

### 4. Generate a runnable clean request

Do not copy the zero-hash example. Bind the prepared model manifest and a
frozen task/seeds identity into a canonical request:

```bash
"$PWD/deployment/runtime/n0-twam/bin/python" \
  scripts/n0_twam/generate_clean_request.py \
  --root "$PWD/deployment" \
  --task pull_out_key \
  --initial-seed 17 \
  --exogenous-seed 29 \
  --simulator-device cuda:0

REQUEST_PATH="$PWD/deployment/requests/n0-twam/pull_out_key/clean.json"
```

The generator validates every official artifact hash, atomically writes a
canonical `trial_set_manifest.json`, uses its exact content hash as
`dataset_sha256`, records the EE action specification and 24-slot execution
contract, and refuses to overwrite different existing content. The identity is
the frozen task/seeds/horizon contract, not a raw episode-file hash. An optional
`--dataset-sha256` may assert the expected manifest hash but cannot replace it.

### 5. Execute the closed loop

```bash
CUDA_VISIBLE_DEVICES=0 \
"$PWD/deployment/runtime/isaac-sim-4.5.0/python.sh" \
  -m robotactile_benchmark.cli evaluate \
  --model n0_twam \
  --root "$PWD/deployment" \
  --request "$REQUEST_PATH" \
  --config "$PWD/deployment/artifacts/models/n0_twam/configs/pull_out_key/integration_config.json" \
  --n0-host 127.0.0.1 \
  --n0-port 29601
```

For a remote endpoint, pass host/port at runtime and provide an optional key
only through `N0_TWAM_API_KEY`; none of these are written to requests or model
manifests.

The released checkpoint bundles the exact converter used for training; it
reads the marker-less expert HDF5 tactile `rgb` field rather than
`rgb_marker`. The live N0 backend requests the same UniVTAC `rgb` payload.
The pinned collection path passes simulator RGB arrays directly to OpenCV's
JPEG encoder, whose input contract is BGR; the checkpoint converter then
decodes those bytes with PIL as RGB. The live profile therefore performs
exactly one R/B reversal to reproduce the checkpoint's decoded pixel domain
before websocket transport. N0 requires both tactile streams, so
A1/A2 and matched `no_touch` remain
`unsupported_contract`; black frames are never substituted.

### 6. Run and gate an official Clean campaign v2

The public N0-TWAM page reports an 84.5% average over eight UniVTAC tasks and
100 randomized trials per simulation task. RoboTactile stores that statement
as a reference contract in
`configs/protocols/n0_twam_univtac_paper_v1.json`; the value is contextual
evidence, not a pass threshold.

First create a full-scope 10-valid-outcome-per-task pilot. The reserve is
preregistered candidate inventory for replacing execution exceptions; `2` is a
documented operator choice, not an official constant:

```bash
export DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
export QUALIFICATION="$DEPLOY_ROOT/artifacts/deployment/<source-bound-qualification-v3>.json"

python scripts/n0_twam/generate_clean_campaign_requests.py \
  --root "$DEPLOY_ROOT" \
  --campaign-id n0-official-clean-pilot-v1 \
  --protocol pilot_v1 \
  --trials-per-task 10 \
  --sampling-contract univtac_official_v1 \
  --official-eval-seed 0 \
  --replacement-reserve-per-task 2

python scripts/n0_twam/run_clean_campaign_all_tasks.py \
  --root "$DEPLOY_ROOT" \
  --manifest "$DEPLOY_ROOT/requests/clean-campaigns/n0-official-clean-pilot-v1/campaign_manifest.json" \
  --execution-profile claim \
  --qualification "$QUALIFICATION" \
  --gpus 0 \
  --run-id n0-official-clean-pilot-v1
```

The paper-candidate inventory changes only the frozen campaign identity,
protocol, target, and preregistered reserve. The example reserve of 10 is also
an operator choice fixed before any outcome is observed:

```bash
python scripts/n0_twam/generate_clean_campaign_requests.py \
  --root "$DEPLOY_ROOT" \
  --campaign-id n0-official-clean-paper-v1 \
  --protocol paper_v1 \
  --trials-per-task 100 \
  --sampling-contract univtac_official_v1 \
  --official-eval-seed 0 \
  --replacement-reserve-per-task 10

python scripts/n0_twam/run_clean_campaign_all_tasks.py \
  --root "$DEPLOY_ROOT" \
  --manifest "$DEPLOY_ROOT/requests/clean-campaigns/n0-official-clean-paper-v1/campaign_manifest.json" \
  --execution-profile claim \
  --qualification "$QUALIFICATION" \
  --gpus 0 \
  --run-id n0-official-clean-paper-v1 \
  --publish-paper
```

Both commands are N0-only. The all-task runner verifies qualification v3 and
automatically propagates its task-local binding through the N0 rank-0 server
attestation, Isaac child attestation, and attempt v3. Internal attestation flags
belong to the owned shard/child launchers and must not be passed manually. The
v2 campaign manifest freezes each task's candidate order
as the released evaluator's consecutive sequence, beginning at
`1_000_000 * (1 + official_eval_seed)`; initial and exogenous seeds are equal,
and every task reuses the same task-local sequence. There is no SHA-derived
split between simulator and policy seeds.

The first 10 (pilot) or 100 (paper) valid outcomes per task define the
denominator. The hard watchdog covers startup, reset, inference, action
execution, artifact export, and teardown. Official N0 requests bind it as an
infrastructure-only watchdog; elapsed wall time cannot become a scoreable
failure. Ordinary success, task failure, early stop, and timeout at the frozen
action/observation horizon are valid outcomes and all enter it. Only a
classified execution exception with matching lifecycle evidence is recorded as
`exception_replaced`, excluded from the denominator, and causes the all-task
runner to launch the next consecutive candidate through a fresh N0 server
lifecycle. Once the target is reached, unattempted candidates are
`unused_reserve`; attempting a later candidate is an overrun error. A malformed
artifact, invalid canonical JSON, ambiguous receipt, or identity/hash mismatch
is a protocol error and cannot be relabeled as a replaceable exception.

`pilot_v1` validates the full protocol at lower sample count but is never
paper-claim eligible. `paper_v1` requires all eight frozen tasks and exactly
100 valid outcomes per task; it still needs task/action qualification and every
other comparison gate. The legacy SHA-seeded builder and any
`diagnostic_v1` campaign remain useful only for integration diagnosis and must
not be compared with 84.5%.

Qualification v3 contains eight ordered task-local source and
observation-parity bindings. Normalizer and serve-bundle hashes may differ by
task; code, external source, checkpoint, config, prompt, action, and input
profile must agree. `clean-campaign-publish` removes
`simulator_not_qualified` only for a complete `paper_v1` bundle whose accepted
attempt-v3 receipts exactly match qualification v3 and both runtime
attestations. Legacy v1/v2 qualifications remain readable for compatibility,
but cannot authorize promotion.

The default `initial_state_policy=official_reproduction` records reset-time
terminal signals without rejecting the episode solely for those signals.
`replace_initial_terminal_v1` is an opt-in robustness diagnostic with different
semantics and must not be compared directly with the public 84.5% reference.
The implemented qualification and attestation path does not itself claim that
a new GPU result has been executed.

After execution, strict aggregation and the artifact-only comparison gate are:

```bash
robotactile clean-campaign-report \
  --root "$DEPLOY_ROOT" \
  --manifest "$DEPLOY_ROOT/requests/clean-campaigns/n0-official-clean-paper-v1/campaign_manifest.json"

robotactile n0-protocol-alignment \
  --root "$DEPLOY_ROOT" \
  --manifest "$DEPLOY_ROOT/requests/clean-campaigns/n0-official-clean-paper-v1/campaign_manifest.json"
```

The gate strictly reloads the campaign, every request, live artifact, and
attempt receipt. It reports sampling, seed/reset, terminal,
planning/execution, cadence, predicate, crash-denominator, action-path, model
lineage, observation-parity, and expert-alignment status. Missing public
checkpoint/seed/render information is `UNKNOWN`; a custom endpoint action path,
SHA-derived split simulator/policy seeds, an exception counted as task failure,
or a non-`paper_v1` sample is `FAIL`. No episode is launched and no existing
artifact is modified.

## Dream-Tac integration

Dream-Tac is a separate tactile-policy integration, not an ACT, N0-TWAM,
N0-VTLA, or FTP-1 alias. RoboTactile pins the upstream source at
`14bab51d6862fd07124745c55cd395ea5caa9fd3`. The checkout, runtime, checkpoint,
dataset statistics, and T5 embeddings remain external to the Apache-2.0 wheel.
The root upstream license is Apache-2.0, but selected configuration files
retain conflicting NVIDIA proprietary/confidential header text; review the
exact files and obtain upstream clarification before redistribution.

As of 2026-08-30, the official documented upstream channels did not expose a
downloadable Dream-Tac checkpoint, and no UniVTAC task-aligned checkpoint was
public. RoboTactile therefore accepts a user-supplied serving bundle only. The
manifest labels it `user_supplied_unverified`; hashing it does not turn it into
an official release.

### Official HTTP and action contract

The adapter reproduces the pinned Franka server's `/info` and `/infer` wire
surface:

- `vision['top']` -> `cam_front` RGB;
- `vision['wrist_l']` -> `cam_high` RGB;
- delivered left/right tactile RGB -> `tactile_left` / `tactile_right`;
- EE8 proprio `[xyz, quaternion_wxyz, gripper_qpos]` -> the server's 6D
  `[xyz, roll, pitch, yaw]` state;
- exact task instruction -> `instruction`;
- server output -> an absolute `(20, 7)` `[xyz, roll, pitch, yaw, gripper]`
  chunk;
- every pose is converted to absolute EE8
  `[xyz, quaternion_wxyz, gripper_qpos]` before delivery to the common runner.

Both tactile payloads must be present. Fault injection therefore operates on
the actually delivered tactile tensors; the adapter does not silently restore
clean inputs. The gripper scalar convention is ambiguous across upstream
materials, so RoboTactile never guesses it. The artifact must bind one explicit
mapping—`greater_than_threshold_is_closed_v1` or
`greater_than_threshold_is_open_v1`—plus the calibrated threshold for an
upstream-style bundle. A model retrained with continuous UniVTAC gripper qpos
instead binds `direct_qpos_v1` and both train-derived qpos bounds; this mode
does not threshold or clip the prediction and fails closed outside the bound
range. The compatibility `gripper_threshold` remains recorded but is ignored
by direct-qpos delivery. The artifact also binds `control_hz`; the upstream
preprocessing, model metadata, and real client expose different cadence
values, so a runtime default is not paper evidence.

### Install and content-address one user-supplied bundle

Install only the exact external source checkout:

```bash
robotactile deployment init
bash integrations/install_dream_tac.sh
```

Place user-supplied files under one local artifact root. A minimal shape is:

```text
deployment/artifacts/models/dream_tac/
├── checkpoint/                  # complete model directory
├── dataset_statistics_franka.json
└── t5_embeddings.pkl
```

Then generate a canonical manifest and integration config. Replace every
example value with the exact training/serving contract for the supplied model:

```bash
robotactile integrations configure dream-tac \
  --bundle-root "$PWD/deployment/artifacts/models/dream_tac" \
  --checkpoint-root "$PWD/deployment/artifacts/models/dream_tac/checkpoint" \
  --dataset-stats "$PWD/deployment/artifacts/models/dream_tac/dataset_statistics_franka.json" \
  --t5-embeddings "$PWD/deployment/artifacts/models/dream_tac/t5_embeddings.pkl" \
  --task custom_franka_pick_baguette \
  --instruction 'Pick up the baguette.' \
  --experiment-config cosmos_predict2_2b_480p_franka_pick_and_place_baguette \
  --control-hz 20 \
  --gripper-mapping greater_than_threshold_is_closed_v1 \
  --gripper-threshold 0.5

robotactile integrations doctor \
  --model dream_tac \
  --task custom_franka_pick_baguette
```

`configure` hashes every checkpoint file plus dataset statistics and T5
embeddings, and binds the instruction, experiment config, control cadence,
gripper conversion, source commit, and fixed 20-step action contract. The
checked-in [example](../examples/dream_tac/README.md) is contract-only and uses
placeholder hashes; it is not a model bundle.

### UniVTAC retraining route

Because the documented upstream channels do not supply a task-aligned
checkpoint, RoboTactile also defines a separate `Dream-Tac-UniVTAC` retraining
route. Its CPU materializer is restricted to the frozen `train759` split:
`frozen40` remains held out, and `grasp_classify/90` remains quarantined. It
preserves the source 10 Hz cadence, creates `T - 1` causal samples, binds a
20-step/two-second chunk, requires Vision + both tactile streams for every
sample, and supervises the continuous next-row gripper coordinate.

Models trained by this route must use `direct_qpos_v1`, not thresholded
open/close conversion. The complete P0--P4 commands, artifact boundaries, GPU
requirements, and claim gates are documented in
[Dream-Tac UniVTAC retraining](dream_tac_training.md). P0/P1 data preparation
does not imply that training, offline evaluation, or closed-loop evaluation has
already occurred.

P2/P3 uses the source-bound
`python -m scripts.dream_tac.training.launch_training` entry point and the
strict examples under `configs/experiments/dream_tac/`. It supports one Linux
NVIDIA CUDA node with one to eight local GPUs; it does not use the N0-TWAM HPU
launcher. The separate `scripts/dream_tac/hcu_port/` route currently provides
only source-bound HCU compatibility and CASA correctness probes, not a formal
P2/P3 optimizer-step claim. The Cosmos base checkpoint and locally cached
`google-t5/t5-11b` snapshot remain external prerequisites, and formal training
must expose only train759. Resume is checkpoint-hash-bound and all launcher
plans, logs, and results are no-clobber receipts.

### Current release gate

The pinned HTTP inference path does not apply the paper's CASA contact-aware
inference gate, even though CASA-related logic appears in the training path.
Together with the missing official downloadable checkpoint and missing
UniVTAC task-aligned checkpoint, this prevents a paper-parity or UniVTAC
closed-loop claim. The integration deliberately fails its release doctor at
those gates.

| Evidence level | Current status |
|---|---|
| CODE / contract smoke | Supported: registry, manifest, HTTP validation, observation mapping, and EE8 conversion |
| OFFLINE | Unverified: no official downloadable checkpoint was available through documented channels |
| CLOSED-LOOP | Unverified: no qualified task-aligned UniVTAC model/runtime trace |
| OFFICIAL | Not claimed: current HTTP inference is not CASA-parity and `release_ready=false` |

Do not report a passing source install, schema check, mock-server inference, or
user-supplied checkpoint hash as Dream-Tac Success Rate or paper reproduction.

## FTP-1 integration

FTP-1 is a VTLA/VLA policy integration, not an N0-TWAM alias and not a strict
WAM. RoboTactile freezes the official source at
`89fa681d6c014cce28300946b7526db808e0b1c1` and the public UniVTAC checkpoint
collection at revision
`620ac69b4fffd2341300cfef1b1d224d56710ed3`. The source is Apache-2.0. The
checkpoint repository does not declare a weight license, and Gemma terms may
also apply; the weights must therefore be reviewed and downloaded separately.
Neither source nor weights enter the RoboTactile wheel.

The checkpoint collection exposes exactly these six task routes:

| RoboTactile task | Checkpoint directory | Camera route |
|---|---|---|
| `insert_hole` | `FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1` | head |
| `insert_tube` | `FTP1_UniVTAC_insert_tube_expert_gsmall_ftp1` | head + wrist |
| `lift_bottle` | `FTP1_UniVTAC_lift_bottle_expert_gsmall_ftp1` | head |
| `lift_can` | `FTP1_UniVTAC_lift_can_expert_gsmall_ftp1` | head + wrist |
| `pull_out_key` | `FTP1_UniVTAC_pull_out_key_expert_gsmall_ftp1` | head |
| `put_bottle_in_shelf` | `FTP1_UniVTAC_put_bottle_expert_gsmall_ftp1` | head |

There is no released FTP-1 checkpoint for `grasp_classify` or `insert_HDMI`.
Do not map another task's checkpoint onto either task and report it as an
official FTP-1 result.

### Official observation and action contract

RoboTactile keeps the executable upstream evaluation semantics:

1. Every new closed-loop observation triggers a fresh FTP-1 inference.
2. The raw prediction is a `(32, 120)` chunk. Index 0 is skipped and indices
   `[1:21]` are admitted to the overlapping prediction buffer.
3. Concurrent candidates for the current control step are temporally ensembled
   with `K=0.01`.
4. The checkpoint's `mix` action representation treats the seven arm joints as
   deltas from qpos at that inference and the gripper channel as absolute.
5. The worker returns exactly one absolute qpos8 target for the current
   simulator control step, then reinfers from the next observation.

The head image maps to `camera_ego_rgb_0`; the wrist image is included only for
the two routes shown above. Left and right delivered tactile frames map to the
two areas of `right_tactile_gripper` as `(1, 2, 224, 224, 3)` uint8 data. Arm
state comes from indices 9--15 of the canonical proprio vector and the gripper
from index 44. The pinned executable evaluation code does not perform an
explicit RGB/BGR swap, so the integration freezes `upstream_passthrough_v1`.
That is code-parity evidence, not a claim that the upstream training color
provenance has been independently reconstructed.

### Install one isolated runtime and checkpoint

The installer creates `deployment/sources/ftp1-policy` and
`deployment/runtime/ftp1-policy`. It does not modify system Python, system
CUDA, Isaac, N0-TWAM, or N0-VTLA runtimes.
It uses the HTTPS package registry recorded by the pinned upstream `uv.lock`
instead of inheriting a machine-wide pip mirror. An equivalent HTTPS registry
can be selected per invocation with `ROBOTACTILE_FTP1_PIP_INDEX_URL`.
Before resolving the upstream runtime, it installs the exact official PyTorch
`2.7.1+cu128` 17-package bundle. No runtime receipt is written until CUDA 12.8,
Blackwell `sm_120`, and a real CUDA tensor kernel have all been verified.
The hash-bound PaliGemma tokenizer is provisioned separately under
`deployment/artifacts/openpi-data/ftp1-policy`, never in a user/root cache.

```bash
export DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
export TASK=insert_hole

robotactile deployment init --root "$DEPLOY_ROOT"
bash integrations/install_univtac.sh
bash scripts/ftp1_policy/install_official_runtime.sh --root "$DEPLOY_ROOT"
bash scripts/ftp1_policy/install_isaac_client.sh --root "$DEPLOY_ROOT"

hf download MJJJJ1064/ftp1_univtac_finetune \
  --revision 620ac69b4fffd2341300cfef1b1d224d56710ed3 \
  --include 'FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1/19999/**' \
  --local-dir "$DEPLOY_ROOT/artifacts/models/ftp1_policy"
```

Downloading one task is sufficient for a one-task run. Repeat with the exact
checkpoint directory from the table only when another task is selected.

The FTP-1 model worker and Isaac client remain separate runtimes. The client
installer invokes only
`deployment/runtime/isaac-sim-4.5.0/python.sh -m pip` and pins
`msgpack==1.1.1` plus `pyzmq==27.1.0`. Its receipt is
`deployment/artifacts/deployment/ftp1_policy_isaac_client_install.json`.
N0-TWAM uses the same msgpack version and retains its independent
`n0_twam_isaac_client_install.json` receipt; neither model runtime nor system
Python/CUDA is modified by this client installer.

### Configure, diagnose, and start the worker

```bash
robotactile integrations configure ftp1-policy \
  --root "$DEPLOY_ROOT" \
  --task "$TASK"

export FTP1_CONFIG="$DEPLOY_ROOT/artifacts/models/ftp1_policy/configs/$TASK/integration_config.json"
export FTP1_MANIFEST="$DEPLOY_ROOT/artifacts/models/ftp1_policy/configs/$TASK/artifact_manifest.json"

robotactile integrations doctor \
  --model ftp1_policy \
  --root "$DEPLOY_ROOT" \
  --task "$TASK" \
  --config "$FTP1_CONFIG"
```

`configure` hashes the model, tokenizer, train/model/tactile configs, and full
task normalization inventory. It also binds the source commit, checkpoint
revision, exact prompt, camera route, color contract, and temporal ensemble.
`doctor` is read-only and does not load the model or allocate Isaac.

For `insert_hole`, start the official wrapper worker on one explicit GPU:

```bash
export FTP1_PYTHON="$DEPLOY_ROOT/runtime/ftp1-policy/bin/python"
export FTP1_CHECKPOINT_DIR="$DEPLOY_ROOT/artifacts/models/ftp1_policy/FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1/19999"
export FTP1_CHECKPOINT_SHA256="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_sha256"])' "$FTP1_MANIFEST")"
export FTP1_SERVE_BUNDLE_SHA256="$(python -c 'import json,sys; from robotactile_benchmark.integrations.ftp1_policy.artifacts import FTP1PolicyArtifactManifest; print(FTP1PolicyArtifactManifest.from_dict(json.load(open(sys.argv[1]))).serve_bundle_sha256)' "$FTP1_MANIFEST")"
export OPENPI_DATA_HOME="$DEPLOY_ROOT/artifacts/openpi-data/ftp1-policy"

CUDA_VISIBLE_DEVICES=0 OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
  "$FTP1_PYTHON" scripts/ftp1_policy/serve_official.py \
  --bind 'tcp://*:5561' \
  --source-root "$DEPLOY_ROOT/sources/ftp1-policy" \
  --checkpoint-dir "$FTP1_CHECKPOINT_DIR" \
  --checkpoint-sha256 "$FTP1_CHECKPOINT_SHA256" \
  --serve-bundle-sha256 "$FTP1_SERVE_BUNDLE_SHA256" \
  --domain-name UniVTAC_insert_hole \
  --task insert_hole \
  --device cuda:0
```

The server metadata and the RoboTactile client independently bind these
identities. A mismatched task, prompt, camera route, source commit, checkpoint,
or serving bundle fails before policy action delivery.

### Clean/Faulted robustness contract

FTP-1 requires both tactile tensors. The formal live robustness group therefore
contains Clean plus the 12 payload-preserving operators F1--F7, T1--T3, and
C1--C2. A1 structural stream absence and A2 frame erasure are explicit N/A
cells for this frozen interface; black, rest, or duplicated tensors are not
used to make those cells executable.

The generator writes one source-bound plan receipt, one Clean request, and 12
Faulted request/manifest pairs. The paired runner executes the ordered requests
from one canonical post-reset simulator snapshot, so the exogenous scene and
initial state are matched while the delivered tactile stream differs. See
[the FTP-1 robustness workflow](benchmark_workflow.md#ftp-1-cleanfaulted-robustness-workflow)
for exact commands and rest-reference requirements.

Passing unit tests or `doctor` establishes **CODE** readiness. Hashing an
artifact or running a worker-only inference establishes at most **OFFLINE**
model execution. Only a completed, strict-loadable Isaac trace establishes one
**CLOSED-LOOP** trial; it still does not establish qualification, statistical
sufficiency, or a paper-level Success Rate. No FTP-1 metric is bundled or
claimed merely because this integration exists.

## N0-VTLA integration

N0-VTLA is a separate model family, not an alias for N0-TWAM. RoboTactile pins
the official source at `03a0ce4d7091ca2354864796770715aa212601b7` and the
released UniVTAC checkpoint revision at
`73a514c015c6745a14a3efdca92f25c6cfab5eb5`. The source remains under
CC-BY-SA-4.0 and the weights remain subject to their upstream model terms; none
is copied into the Apache-2.0 wheel.

The adapter reproduces the official `sim_single_arm_tactile` serving contract:

- `vision['top']` -> `observation/image`;
- `vision['wrist_l']` -> `observation/wrist_image`;
- left/right delivered tactile -> the two official tactile wire keys;
- float32 qpos8 -> `state`;
- `reset` clears the server baseline, so the first prediction captures the
  episode tactile reference;
- the model's first seven channels are trained as deltas, but the official
  policy applies `AbsoluteActions` against the supplied state before serving;
  RoboTactile therefore takes the server's leading `(50, 8)` as absolute qpos
  targets without adding state a second time;
- all 50 steps must be executed before reinference.

Install the pinned sources and the lightweight RoboTactile client dependencies:

```bash
robotactile deployment init
bash integrations/install_univtac.sh
bash integrations/install_n0_vtla.sh
python -m pip install '.[n0-vtla]'
```

Install the upstream Python 3.11/CUDA runtime according to its pinned README,
then download the deployable task policy—not the non-deployable base model:

```bash
hf download NeoteAI/n0_VTLA_insert_hole \
  --revision 73a514c015c6745a14a3efdca92f25c6cfab5eb5 \
  --local-dir "$PWD/deployment/artifacts/models/n0_vtla/checkpoint"

robotactile integrations configure n0-vtla --task insert_hole
robotactile integrations doctor --model n0_vtla --task insert_hole
```

Start the official server from its isolated runtime:

```bash
cd "$PWD/deployment/sources/N0-VTLA"
VTLA_ASSET_ID=n0_insert_hole_norm python scripts/serve_zmq.py \
  --config sim_single_arm_tactile \
  --ckpt "$OLDPWD/deployment/artifacts/models/n0_vtla/checkpoint" \
  --addr 'tcp://*:5557' \
  --default-prompt 'insert hole'
```

`integrations configure` and `doctor` establish only source/artifact/transport
readiness. Adapter unit tests establish the RGB/tactile/qpos mapping and full
chunk rule, not live model inference, Isaac execution, or success rate. The
current release does not present the one-task N0-VTLA checkpoint as an all-8
UniVTAC result.

`insert_hole` supports two separately identified evaluation profiles. The
default `official_v1` retains the pinned upstream predicate. The optional
`insert_hole_strict_v1` requires `<5 mm` radial XY error, `>50 mm` insertion,
alignment dot `>0.999`, `<25 mm` in-hand z drift, and 30 consecutive 120 Hz
physics steps. Generate official and strict campaigns separately with
`scripts/n0_vtla/prepare_insert_hole_pair.py --success-profile ...`; see the
[dual success protocol](insert_hole_success_protocols.md). The server,
checkpoint, observation mapping, and 50-step action chunks are unchanged.

## Static examples versus runnable deployment files

`examples/act/`, `examples/dream_tac/`, `examples/ftp1_policy/`,
`examples/n0_twam/`, and `examples/n0_vtla/` document contract shapes only.
Their all-zero hashes are intentionally non-runnable. Do not edit them into
local runtime files; run `integrations configure` so hashes are computed from
real artifacts under `deployment/`.

The checked-in `configs/integrations/*.json` similarly identify the five static
registrations. Generated deployment configs use absolute manifest paths and
are ignored by Git.

## Evidence boundaries

| Operation | Establishes | Does not establish |
|---|---|---|
| `integrations configure` | Real local files were hashed into a canonical config | Model load or inference |
| `integrations doctor` | Declared source/artifact/transport gates passed or failed | Simulator allocation or task success |
| `preflight-live` | Request, source, artifacts, GPU, and Isaac Python readiness | Simulator execution |
| live artifact | One requested runtime produced a strict-loadable trace | Isaac qualification or hardware validity |
| qualification v3 | Eight task-local source/parity identities passed strict verification | A policy outcome or paper sample size |
| dual-attested paper bundle | Complete `paper_v1` attempts exactly match v3, N0 rank-0, and Isaac-child evidence | Real-robot validity or hardware fault calibration |
