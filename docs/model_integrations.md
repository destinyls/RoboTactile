# Model Integrations

RoboTactile registers exactly two first-class model integrations: `act` and
`n0_twam`. Both implement the same `PolicyAdapter` lifecycle (`reset`, `infer`,
`commit`, `abort`, and `close`) and feed the same fault injector, closed-loop
runner, artifact, and reporting contracts. Model source and weights remain
external to the Apache-2.0 wheel. Their public orchestration surfaces are not
yet symmetric: N0-TWAM has the official Clean campaign owner, while the current
`generate-primary-matrix` / `run-live-matrix` robustness workflow is ACT-only.

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
WorldArena + UniVTAC pinned sources
  -> deployment/artifacts/models/act/<task>/<profile>/
  -> artifact_manifest.json
  -> integration_config.json
  -> request -> integrations doctor -> preflight-live
  -> live-univtac-paired-run/evaluate -> content-addressed live artifacts
```

### 1. Install sources

```bash
bash integrations/install_univtac.sh
bash integrations/install_act_runtime.sh
```

The exact commits, licenses, and immutable hyperlinks are listed in
[External dependencies](external_dependencies.md).

### 2. Place real artifacts

For task `pull_out_key`, use the canonical layout for both mandatory profiles:

```text
deployment/artifacts/models/act/
├── encoder.pth
└── pull_out_key/
    ├── univtac/
    │   ├── policy_last.ckpt
    │   └── dataset_stats.pkl
    └── vision_only/
        ├── policy_last.ckpt
        └── dataset_stats.pkl
```

`univtac` consumes the two tactile streams. `vision_only` is a separately
trained matched `no_touch` system; a black image, resting tactile frame, or
structurally absent stream is not a substitute.

### 3. Generate canonical config

Run once per profile. Because the default output names are shared, retain the
two manifests under explicit profile names and choose the relevant config for
each request/matrix stage:

```bash
robotactile integrations configure act \
  --task pull_out_key \
  --profile univtac \
  --manifest-output "$PWD/deployment/artifacts/models/act/univtac_manifest.json" \
  --integration-config-output "$PWD/deployment/artifacts/models/act/univtac_config.json"

robotactile integrations configure act \
  --task pull_out_key \
  --profile vision_only \
  --manifest-output "$PWD/deployment/artifacts/models/act/vision_only_manifest.json" \
  --integration-config-output "$PWD/deployment/artifacts/models/act/vision_only_config.json"
```

The command computes checkpoint, stats, encoder, upstream source, and training
config hashes from the real files. It validates the clean UniVTAC commit and
official layout, emits canonical JSON, and refuses to replace different
existing output. It does not load Torch or run inference.

For a single tactile run, the default names may be used:

```bash
robotactile integrations configure act --task pull_out_key
```

### 4. Diagnose, preflight, and execute

```bash
robotactile integrations doctor \
  --model act \
  --config "$PWD/deployment/artifacts/models/act/univtac_config.json"

robotactile preflight-live \
  --request "$REQUEST_PATH" \
  --config "$PWD/deployment/artifacts/models/act/univtac_config.json"

"$PWD/deployment/runtime/isaac-sim-4.5.0/python.sh" \
  -m robotactile_benchmark.cli live-univtac-run \
  --request "$REQUEST_PATH" \
  --config "$PWD/deployment/artifacts/models/act/univtac_config.json"
```

Legacy explicit artifact-root and hash arguments remain accepted, but the
typed config is the recommended single runtime input. `preflight-live` does
not allocate Isaac or the model. A live run remains
`unqualified_live_univtac_execution_v1` until the separate qualification
protocol passes.

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

## Static examples versus runnable deployment files

`examples/act/` and `examples/n0_twam/` document contract shapes only. Their
all-zero hashes are intentionally non-runnable. Do not edit them into local
runtime files; run `integrations configure` so hashes are computed from real
artifacts under `deployment/`.

The checked-in `configs/integrations/*.json` similarly identify the two static
registrations. Generated deployment configs use absolute manifest paths and
are ignored by Git.

## Evidence boundaries

| Operation | Establishes | Does not establish |
|---|---|---|
| `integrations configure` | Real local files were hashed into a canonical config | Model load or inference |
| `integrations doctor` | Declared source/artifact/transport gates passed or failed | Simulator allocation or task success |
| `preflight-live` | ACT request, source, artifacts, GPU, and Isaac Python readiness | Simulator execution |
| live artifact | One requested runtime produced a strict-loadable trace | Isaac qualification or hardware validity |
| qualification v3 | Eight task-local source/parity identities passed strict verification | A policy outcome or paper sample size |
| dual-attested paper bundle | Complete `paper_v1` attempts exactly match v3, N0 rank-0, and Isaac-child evidence | Real-robot validity or hardware fault calibration |
