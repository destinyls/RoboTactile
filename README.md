# RoboTactile Benchmark

RoboTactile is an auditable robustness benchmark for optical-tactile robot
policies. The package implements the paper's 2 Availability + 7 Fidelity +
3 Temporal + 2 Context operators, model-specific matched-condition protocols,
exact single-process paired execution, and source-bound reports. N0-TWAM uses
a strictly paired Clean/Faulted protocol; ACT additionally supports its
registered matched no-touch control. N0-VTLA is registered as a third external
`PolicyAdapter` with its official ZMQ and 50-step qpos contract. FTP-1 is the
fourth first-class integration: it uses its pinned official wrapper in an
isolated ZMQ runtime and returns one absolute qpos8 command per observation.
Dream-Tac is the fifth first-class integration: RoboTactile binds its external
source, user-supplied serving bundle, official HTTP contract, and explicit
XYZ/RPY/gripper-to-EE8 conversion without copying upstream code or weights
into the wheel.

The public repository and project are named **RoboTactile**. The installable
distribution remains `robotactile-benchmark`, the stable Python import is
`robotactile_benchmark`, and the command-line entry point is `robotactile`.

Start here: [Reproducibility](docs/reproducibility.md) ·
[Installation](docs/installation.md) ·
[Quickstart](docs/quickstart.md) ·
[Isaac Sim installation and execution](docs/isaac_sim.md) ·
[Deployment layout](docs/deployment_layout.md) ·
[External dependencies](docs/external_dependencies.md) ·
[Model integrations](docs/model_integrations.md) ·
[ACT operations](scripts/act/README.md) ·
[Insert-hole success protocols](docs/insert_hole_success_protocols.md) ·
[Recorded N0 evaluation](docs/recorded_n0.md) ·
[N0 decision-time fault pilot](docs/n0_decision_stress.md) ·
[Benchmark workflow](docs/benchmark_workflow.md) ·
[A1/A2 missing-input protocols](docs/tactile_availability.md) ·
[Evidence levels](docs/evidence_levels.md) ·
[Benchmark card](BENCHMARK_CARD.md)

Project information: [Citation](CITATION.cff) · [License](LICENSE) ·
[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) ·
[Third-party notices](THIRD_PARTY_NOTICES.md)

The benchmark treats tactile availability as a typed observation contract.
Structural absence is `payload=None`; a black or resting image is never used
as an Availability fault or as the matched no-touch control.
The opt-in [availability protocols](docs/tactile_availability.md) additionally
provide native N0-VTLA omission and a separately labelled zero-fill compatibility
adapter. Neither changes the default fault definition or historical results.

The first-class ACT integration is the pinned UniVTAC implementation at
`policy/ACT/act_policy.py`; it does not use the legacy WorldArena
`deployment/ACTStrict` wrapper. Its public artifacts come from the pinned
`byml/UniVTAC` dataset revision
`172331dbbce95bc04c3e59b22f32dc72ba5561ae` and are installed with
`scripts/act/install_official_artifacts.py`; they are not bundled in the wheel.
ACT inference runs in-process under Isaac Sim's bundled Python, so do not
replace or independently upgrade Isaac's Torch/Torchvision installation to
satisfy a separate ACT environment.

## Choose a path

| Goal | Start with | Hardware |
|---|---|---|
| Verify the package and 14 operators | [Reproducibility guide](docs/reproducibility.md#1-clone-and-verify-the-package) | Python >= 3.9 |
| Install and configure official UniVTAC ACT artifacts | [ACT integration](docs/model_integrations.md#act-end-to-end) | NVIDIA Linux + Isaac-local Python for live execution |
| Prepare the canonical N0-TWAM workspace | `robotactile setup --model n0_twam` | NVIDIA Linux for live execution |
| Configure N0-VTLA `insert_hole` | [N0-VTLA integration](docs/model_integrations.md#n0-vtla-integration) | Python 3.11 + CUDA upstream runtime |
| Configure one released FTP-1 task | [FTP-1 integration](docs/model_integrations.md#ftp-1-integration) | Python 3.11 + CUDA upstream runtime |
| Inspect or configure a user-supplied Dream-Tac bundle | [Dream-Tac integration](docs/model_integrations.md#dream-tac-integration) | Upstream CUDA runtime; no public aligned checkpoint is bundled |
| Install Isaac and run a headless GPU smoke | [Isaac Sim guide](docs/isaac_sim.md) | NVIDIA Linux |
| Qualify one UniVTAC task reset and observation | `bash scripts/live_univtac/qualify_univtac_task_reset.sh --help` | NVIDIA Linux |
| Run one paired three-condition ACT/UniVTAC set | `robotactile live-univtac-paired-run --help` | NVIDIA Linux + artifacts |
| Run one N0-TWAM Clean episode | [N0 Clean reproduction](docs/reproducibility.md#6-run-one-n0-clean-episode) | Qualified NVIDIA deployment |
| Run an N0-TWAM Clean/Faulted campaign | [N0 robust workflow](docs/benchmark_workflow.md#n0-twam-cleanfaulted-robust-campaign-p0-p3) | Qualified NVIDIA deployment |
| Serve official N0-TWAM UniVTAC weights | [N0-TWAM integration](docs/model_integrations.md#n0-twam-end-to-end) | Each selected GPU >= 40 GB |
| Measure N0 on a real recorded UniVTAC anchor | [Recorded N0 evaluation](docs/recorded_n0.md) | N0 GPU server; no Isaac loop |
| Run the current ACT 14 x 5 live matrix | [Benchmark workflow](docs/benchmark_workflow.md#4-materialize-and-run-the-primary-matrix) | Qualified ACT deployment |
| Run paired FTP-1 Clean/Faulted trials | [FTP-1 robustness workflow](docs/benchmark_workflow.md#ftp-1-cleanfaulted-robustness-workflow) | Qualified UniVTAC + FTP-1 deployment |
| Render a verified live trace as PNG/MP4 | `robotactile visualize-live-artifact --help` | Pillow; ffmpeg for video |
| Inspect the same 14 operators with marker-preserving optical delivery | [Optical 14 and visual validation](docs/optical14.md) | Core Python + Pillow; recorded source bundle |

N0-TWAM shares the same `PolicyAdapter`, observation-delivery, trace, and
reporting contracts. Its official websocket transport, released delta
checkpoint, rot6d-to-EE action bridge, and causal KV-cache re-grounding are
registered; live evidence still requires real weights, a reachable server,
and Isaac Sim. Its public robustness driver generates, runs, and reports
Clean/Faulted-only paired campaigns for 12 live operators by default. The A1/A2
Availability contracts are retained as explicit unsupported receipts: no N0
live request is generated for them and no black/rest image is substituted in
that default workflow. The explicit zero-fill supplement is a separate protocol.
The public ACT primary-matrix generator remains a separate workflow because it
binds tactile and matched vision-only ACT checkpoints.

FTP-1 shares those observation-delivery, online-injection, snapshot-pairing,
artifact, and reporting contracts. The released checkpoints cover six tasks:
`insert_hole`, `insert_tube`, `lift_bottle`, `lift_can`, `pull_out_key`, and
`put_bottle_in_shelf`. For the default fixed tactile-required interface, RoboTactile
executes F1--F7, T1--T3, and C1--C2; A1/A2 are recorded as N/A instead of being
silently replaced with a black, rest, or duplicated tactile tensor. The
integration and orchestration code do not imply that a Clean or Faulted
closed-loop metric has already been measured.

Dream-Tac uses the pinned upstream `/info` and `/infer` HTTP surface. One
request contains two RGB images, two tactile images, a 6D XYZ/RPY state, and
the exact instruction; one response contains a 20 x 7 absolute
XYZ/RPY/gripper chunk. RoboTactile converts each action explicitly to EE8
`[xyz, quaternion_wxyz, gripper_qpos]`. The artifact must bind the control Hz,
gripper threshold, and gripper polarity because the upstream materials do not
define one unambiguous deployment-wide convention. As of 2026-08-30, the
official documented channels did not expose a downloadable checkpoint, the
HTTP inference path omits the paper's CASA inference gate, and no UniVTAC
task-aligned Dream-Tac checkpoint is public. The integration is therefore
`release_ready=false`: CODE/contract smoke is supported, while OFFLINE,
CLOSED-LOOP, and OFFICIAL evidence remain unverified.

## Five-minute package verification

```bash
git clone https://github.com/destinyls/RoboTactile.git
cd RoboTactile
bash scripts/bootstrap_pip.sh
source .venv/bin/activate

robotactile validate-registry
robotactile smoke-replay \
  --operator T1_fixed_source_delay \
  --severity 3 \
  --output outputs/reproduction-smoke
make check
```

This verifies the installable package and deterministic operator contracts. It
does not start Isaac Sim or establish model success. Continue with the
[N0-TWAM reproduction guide](docs/reproducibility.md#5-prepare-one-official-n0-task)
for a real Clean closed loop.

## Reproduction status

| Path | Code | Public orchestration | Claim boundary |
|---|---|---|---|
| Deterministic 14-operator replay | Implemented | `smoke-replay`, `smoke-matrix` | Software contract |
| N0 recorded-HDF5 evaluation | Implemented | `recorded-n0`; `run_frozen40_tactile_causal.py` | Offline action metrics; no simulator success |
| N0 UniVTAC Clean | Implemented | One-task, all-8, pilot, paper campaign owners | Qualification and sample size remain separate gates |
| ACT UniVTAC fault robustness | Implemented | `build-act-reset-reference`, `capture-act-reset-trajectory`, `generate-act-fault-campaign`, `run-act-fault-campaign`, `report-act-fault-campaign` | Official weights are public but not bundled; a source-bound reset witness and dense pre-move replay, successful paired Clean gate, and local qualification remain required |
| N0 UniVTAC fault robustness | Implemented | `generate-n0-fault-campaign`, `run-n0-fault-campaign`, `report-n0-fault-campaign` | Generated requests and diagnostic runs are not paper evidence by themselves |
| FTP-1 UniVTAC Clean/Faulted robustness | Implemented | `prepare_robustness_group.py` + `live-univtac-paired-run` | Requires pinned source/weights and a qualified live deployment; no result is bundled |
| Dream-Tac adapter and HTTP contract | Implemented | Source/artifact/config/adapter smoke only | No public checkpoint or CASA-parity HTTP route; no OFFLINE, CLOSED-LOOP, or OFFICIAL result is claimed |
| UniVTAC `insert_hole` dual success | Implemented | `official_v1` and `insert_hole_strict_v1` | Report as separate SR denominators; strict does not replace official |

## Repository layout

```text
RoboTactile/
├── src/robotactile_benchmark/
│   ├── integrations/{act,dream_tac,ftp1_policy,n0_twam,n0_vtla}/  # first-class integrations
│   ├── backends/                    # UniVTAC facade, lifecycle, and qualification
│   ├── operators/                   # 2 + 7 + 3 + 2 fault operators
│   ├── streaming/                   # causal online fault injection
│   ├── closed_loop/                 # typed runner and trace contracts
│   ├── calibration/                 # measured rest-reference selection/artifacts
│   ├── n0_fault_campaign/           # N0 paired campaign generation/run/report
│   ├── matrix/                      # exact paired matrix + completed-run reuse
│   └── reporting/                   # source-bound metrics and reports
├── configs/                         # frozen runtime registries
├── schemas/                         # public JSON Schemas
├── integrations/                    # external pins and install scripts
├── examples/{act,dream_tac,ftp1_policy,n0_twam,n0_vtla}/  # contract-only examples
├── scripts/{act,ftp1_policy,live_univtac,n0_twam,n0_vtla}/  # runtime entry points
├── docs/                             # installation and evaluation guides
└── deployment/                       # ignored writable workspace
    ├── sources/{UniVTAC,Dream-Tac,N0-TWAM,N0-VTLA,ftp1-policy,IsaacLab,curobo}/
    ├── runtime/{isaac-sim-4.5.0,ftp1-policy,cuda-toolkit-12.4}/
    ├── artifacts/models/{act,dream_tac,ftp1_policy,n0_twam,n0_vtla}/
    ├── requests/{calibration,clean-campaigns,fault-campaigns,primary-matrix}/
    └── outputs/{matrices,reports}/
```

Create and inspect the complete tree with `robotactile deployment init` and
`robotactile deployment show`. The default root is this checkout's
`deployment/`; `ROBOTACTILE_DEPLOY_ROOT` provides an absolute HPC override
without changing the subdirectory contract. See the
[deployment layout guide](docs/deployment_layout.md).

Older workspaces may additionally contain `sources/WorldArena` and the
`integrations/install_act_runtime.sh` compatibility checkout. They belong to
the legacy ACTStrict `policy_best.ckpt` path and are not inputs to the official
UniVTAC ACT adapter or its `policy_last.ckpt` artifact manifest.

The shortest fail-closed preparation path is:

```bash
robotactile integrations list
robotactile setup --model act
```

`setup` initializes the canonical tree, generates model configuration only
when the required files exist, and reports every remaining source, artifact,
transport, or release gate. It never starts Isaac or claims live inference.

## Implemented surface

- immutable observation, fault, trial, execution, matrix, and reporting
  contracts;
- deterministic 14-operator replay with hard delivery validators and an O(T)
  causal streaming engine for closed-loop injection;
- frozen UniVTAC task adapters and dependency-injected contract tests;
- official UniVTAC ACT `policy_last` loading with checkpoint, statistics,
  encoder, config, source, and upstream-commit verification;
- one-command official ACT live execution and independently reloadable live
  artifact bundles;
- official N0-TWAM source/weight pins, per-task normalizer preparation,
  websocket client, absolute EE execution, and real-observation KV grounding;
- official FTP-1 source and checkpoint revision pins, content-addressed task
  artifacts, isolated ZMQ worker, exact temporal ensemble, and qpos8 adapter;
- pinned Dream-Tac source, content-addressed user-supplied serving artifacts,
  official HTTP request validation, and explicit 20-step EE8 action conversion;
- same-snapshot FTP-1 Clean/Faulted execution for the 12 payload-preserving
  Fidelity, Temporal, and Context operators, with A1/A2 emitted as N/A;
- a no-clobber N0 Clean/Faulted campaign bundle, one-snapshot pair runner, and
  strict JSON/CSV report path for 12 executable fault operators, with typed
  unsupported receipts for A1/A2;
- one-process paired ACT execution with a single canonical reset, in-memory
  UIPC/PhysX snapshot replay, successful-Clean provenance, a source-proximate
  dense reset trajectory, capture-derived native-step/qpos hard gates before
  policy inference, diagnostic cross-process render hashes, and a mandatory
  successful Clean gate before Faulted cells;
- deterministic rest-reference calibration from clean UniVTAC traces, with a
  consecutive no-contact predicate, source-root binding, and strict artifacts;
- a clean calibration-request generator that binds split, policy, seed, and
  runtime identities before any simulator allocation;
- a 14 x 5 primary matrix with shared clean/no-touch baselines, one complete
  snapshot-paired GPU batch, explicit crash/unsupported/validator-rejected
  states, and content-addressed receipts;
- a public `generate-primary-matrix` CLI that expands the registered 14 x 5
  grid into 72 exact execution cells and a portable, hash-verified resource
  bundle without allocating the simulator;
- a public `run-live-matrix` CLI whose production path runs all cells in one
  shared runtime and only reuses a fully completed, strictly verified matrix;
- task-stratified reporting with SR, paired delta-SR, gated TGR, native-dose
  curves, bootstrap intervals, exact McNemar tests, and
  Holm correction;
- deterministic JSON, CSV, LaTeX, and SVG report bundles bound to every source
  artifact root.

## Evidence levels

RoboTactile does not promote one evidence tier into another.

| Tier | What it establishes | What it does not establish |
|---|---|---|
| Software contracts | Deterministic delivery, adapter state machines, exact pairing gates, completed-run reuse, and report plumbing | A learned policy, Isaac Sim, task success, or real hardware |
| Live preflight | Frozen request, external checkouts, model artifacts, NVIDIA host, and Isaac Python passed without allocation | Simulator launch, task execution, or task success |
| Unqualified live | The requested official model runtime path executed and produced a hash-verified trace | Isaac qualification, physical calibration, or a publishable simulator result |
| Task/action qualified | All eight frozen tasks passed hash-bound import, reset, and action-contract checks | Policy success or a publishable success rate |
| Legacy v1/v2 qualification | Prior task/action evidence, with at most one global source binding | Paper promotion or exact task-local runtime identity |
| Source-bound qualification v3 | Eight ordered task-local source and observation-parity bindings; each task binds its own normalizer and serve bundle | A policy outcome, statistical sufficiency, real-robot validity, or fault realism |

`live-univtac-run` always writes
`evidence_level=unqualified_live_univtac_execution_v1` and
`simulator_qualification_claimed=false`. Running that command inside Isaac Sim
does not change those fields. The reporting receipt is a deterministic
derivation from verified inputs and also never claims simulator qualification.
`live-univtac-paired-run` additionally emits
`in_process_snapshot_replay_equivalence_v1`; this proves exact equality of the
recorded post-reset multimodal witnesses inside that process, not Isaac
qualification or task success.

The bounded deployment diagnostic executes one Clean episode for each of the
eight tasks through `scripts/n0_twam/run_clean_campaign_all_tasks.py`. This is
an end-to-end functional result, not a statistically sufficient paper estimate.
Use `--execution-profile quick` for one target trial per task or
`--execution-profile diagnostic` for a larger development campaign; neither
requires qualification. Use `--execution-profile claim` for `pilot_v1` and
`paper_v1`; this is the only profile that requires qualification v3 and permits
paper publication. The profile is recorded in the run receipt and does not
change the frozen campaign manifest. Qualification v3 binds eight task-local
source/observation-parity records: the
normalizer and serve bundle may differ by task, while the code, external source,
checkpoint, config, prompt, action, and input-profile identity must agree. The
all-task runner automatically carries that qualification through the N0 rank-0
server attestation, Isaac child attestation, and source-bound attempt v3. Only a
complete `paper_v1` bundle whose two runtime attestations exactly match every
accepted attempt can remove `simulator_not_qualified`. Legacy v1/v2 evidence is
still readable, but cannot be promoted through this gate.

Execution strictness and artifact volume are separate, explicit contracts:

| Execution profile | Default capture | Persisted evidence | Intended use |
|---|---|---|---|
| `quick` | `metrics_only_v1` | terminal result, action hash domain, transition diagnostics | fastest Success Rate smoke |
| `diagnostic` | `preview_v1` | metrics plus at most 64 deterministic clean/delivered keyframes | debugging and compact video |
| `claim` | `paper_full_v1` | complete clean/delivered RGB, tactile, proprio, actions, and diagnostics | pilot/paper evidence |

`--capture-profile` may explicitly upgrade a quick or diagnostic run. A claim
or `--publish-paper` run cannot use compact capture, and compact capture is
currently restricted to the fresh-process worker contract. Compact artifacts
retain the exact terminal result and full-trace hashes, so diagnostic Success
Rate uses the same outcome; omitted frames are declared in
`capture_summary.json` and are never reconstructed or presented as a complete
trace. `metrics_only_v1` cannot generate a video. `preview_v1` can generate a
bounded video from its stored keyframes. Legacy live roots v1.0/v1.1 load as
`paper_full_v1`; compact roots use v1.2 and a distinct non-claim evidence level.

### Official N0-TWAM Clean campaign v2

The smallest full-scope official-sampling run is the 10-valid-outcome-per-task
pilot below. It is N0-only: the runner rejects other policy kinds, forwards raw
N0 EE8 actions without task-specific transforms, uses pinned UniVTAC cuRobo
planning for feasibility, applies the final planned joint target, and advances
exactly two 120 Hz physics ticks per endpoint. This
`robotactile_n0_training_60hz_ee_v1` contract matches the 60 Hz endpoint
cadence of the released training rows. The upstream variable-waypoint EE loop
remains available only as the explicit `univtac_stock_ee_v1` reference path.

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

For a paper candidate, change the protocol to `paper_v1`, the target to 100,
and preregister a fixed reserve (the documented example uses 10 candidates per
task). Each task reuses the official consecutive seed sequence. The owned hard
watchdog covers the full child lifecycle, including startup, reset, execution,
artifact export, and teardown. Generated official N0 requests bind
`wall_timeout_role=infrastructure_watchdog_v1`: elapsed wall time cannot create
a scoreable model result. Ordinary success, failure, early stop, and the
official action-horizon timeout are valid denominator outcomes; only a
classified execution exception
with matching lifecycle evidence becomes `exception_replaced`, is excluded from
the denominator, and advances to the next reserved seed with a fresh N0 server.
Invalid artifacts or identity/hash mismatches are protocol errors, never
replacement events. Internal N0/Isaac attestation flags are passed by the
all-task runner and must not be supplied manually. Legacy SHA-seeded or
`diagnostic_v1` campaigns cannot support comparison with the public paper
number. See [Model integrations](docs/model_integrations.md#6-run-and-gate-an-official-clean-campaign-v2)
for the paper command and full boundary.

### N0-TWAM Clean/Faulted robustness campaign

The N0-specific campaign path is public and independent of the ACT primary
matrix:

```bash
robotactile generate-n0-fault-campaign --help
robotactile run-n0-fault-campaign --help
robotactile report-n0-fault-campaign --help
```

Generation expands one or more canonical N0 Clean requests into immutable
paired bundles. F1-F7, T1-T3, and C1-C2 are live operators. A1 stream absence
and A2 frame erasure remain in the 14-operator inventory as typed unsupported
receipts because the released N0 input contract requires both tactile streams;
they do not allocate Isaac and are not replaced by black/rest frames. Operators
that require a no-contact reference fail closed unless the generator receives
one strict task-bound `--rest-reference TASK=PATH` artifact.

For N0, create that artifact with
`scripts/n0_twam/run_rest_calibration.py`. The calibration-only contract skips
the task `pre_move`, keeps the empty gripper stationary, advances the native
60 Hz observation cadence, and applies no model action to the robot. It writes
the full measured live artifact and strict rest-reference bundle before Isaac
Sim closes; it never reuses a formal Clean/Faulted episode as “rest.”

Operator support and task-phase applicability are distinct. F6 is scoreable
only when the trace contains a prior contact imprint followed by a release
phase; F7 is scoreable only when a high-load response is actually delivered.
A task without those phases must preregister a reduced `--operator` set. A
validator rejection such as `SIGNATURE_NOT_DELIVERED` is reported as an
ineligible cell, not as model failure, model success, or zero SR.

`run-n0-fault-campaign` executes one frozen pair from a single canonical reset,
with Clean first and all executable Faulted cells after it. Bind
`--action-execution-contract robotactile_n0_training_60hz_ee_v1` for the
training-aligned 60 Hz EE8 path. Select `metrics_only_v1`, `preview_v1`, or
`paper_full_v1` independently through `--capture-profile`; the smaller profiles
reduce storage, not the terminal scoring contract. The runner refuses existing
artifacts instead of rerunning or overwriting them.

The strict reporter publishes `summary.json`, `per_task.csv`,
`operator_cells.csv`, and `report_receipt.json`. It defines degradation as
Clean SR minus Faulted SR and retention as Faulted SR divided by Clean SR;
unsupported and missing cells remain explicit. See the complete
[P0-P3 workflow](docs/benchmark_workflow.md#n0-twam-cleanfaulted-robust-campaign-p0-p3)
for the all-8 Clean gate, one-pair L3 pilot, and the same-pair S1/S5 bracket.
These commands implement the protocol but do not claim that any GPU result has
already been collected.

For destructive integration testing only, generation also accepts
`--diagnostic-stress-max --severity 5`. This selects the separately identified
`diagnostic_stress_max_v1` registry (for example C2=30 px, F5=20 px, and
full-window C1/T2). It never changes the paper S1--S5 registry and its
results must be labeled diagnostic rather than reported as S5 benchmark cells.

Before increasing tactile-fault strength, run the stricter model-reliance
ablation `tactile_null_black_frame_v1`. It executes the same N0-TWAM model and
preserves RGB, proprioception, reset, prompt, and action execution, while
replacing both tactile streams with dtype- and shape-preserving all-zero RGB
frames from observation 0 through the complete horizon:

```bash
robotactile generate-n0-fault-campaign \
  --output "$NULL_ROOT" \
  --campaign-id "n0-tactile-null-v1" \
  --base-clean-request "$CLEAN_REQUEST" \
  --diagnostic-tactile-null \
  --severity 5 \
  --operator-seed-master 20260829 \
  --fault-start-index 0 \
  --fault-stop-index "$MAX_OBSERVATION_STEPS"

robotactile run-n0-fault-campaign \
  --campaign-root "$NULL_ROOT" \
  --integration-config "$N0_CONFIG" \
  --n0-source-root "$N0_SOURCE" \
  --capture-profile metrics_only_v1

robotactile report-n0-fault-campaign \
  --campaign-root "$NULL_ROOT" \
  --output "$NULL_REPORT" \
  --bootstrap-seed 20260829
```

The validator requires exact zero-valued pixels on both streams at every
delivered step and exposes `black_frame_max_abs_value`; scoring is invalid
unless that value is zero. The CLI labels the result as a diagnostic ablation
and returns `tactile_null_success_rate` plus Clean-minus-null degradation. If
null SR remains near Clean SR, stronger local tactile-only operators should not
be expected to reduce binary SR on the same task/seed distribution.

Black tactile is still a tensor, not structural absence. To test the released
model without observed tactile tensors, enable the opt-in server overlay and
generate the dedicated A1 diagnostic:

```bash
bash scripts/n0_twam/serve_univtac.sh \
  --root "$DEPLOY_ROOT" \
  --task "$N0_TASK" \
  --gpus 0 \
  --enable-observed-tactile-absence

robotactile generate-n0-fault-campaign \
  --output "$ABSENCE_ROOT" \
  --campaign-id "n0-observed-tactile-absence-v1" \
  --base-clean-request "$CLEAN_REQUEST" \
  --diagnostic-observed-tactile-absence \
  --severity 5 \
  --operator-seed-master 20260829 \
  --fault-start-index 0 \
  --fault-stop-index "$MAX_OBSERVATION_STEPS"
```

The A1 delivery removes both tactile payloads. The policy and websocket client
omit `tactile` entirely and send `tactile_cond_drop=true`; the overlay then
uses the model's training-time zero-tactile-token path for inference and KV
grounding. Clean requests on the same server retain the unmodified upstream
path. This is `observed_tactile_absent_v1`, a non-paper model-reliance
diagnostic—not the published retrained `w/o observed` checkpoint/result.

On a slow first Isaac launch, the upstream 120 s reset watchdog can expire
while compiling/loading the initial physics frame before policy rollout begins.
Set `ROBOTACTILE_UNIVTAC_RESET_TIME_LIMIT_S` to a larger positive value (for
example `600`) for that deployment. This only widens the reset watchdog, cannot
shorten the upstream limit, and does not change seeds, simulation cadence,
observations, actions, or success scoring.

## Isaac Sim and live UniVTAC

The canonical GPU path uses Ubuntu 22.04, an NVIDIA GPU such as RTX 3090,
Isaac Sim 4.5.0 standalone, IsaacLab v2.1.1, and cuRobo v0.7.7. Installation
is explicit: the core wheel never downloads or embeds these dependencies.

Blackwell (`sm_120`) hosts must use a fresh `deployment-sm120` native build;
do not copy A800/3090 `.so` files. The N0-only CUDA 12.8 bootstrap and its
per-binary `cuobjdump` acceptance gate are documented in
[Live UniVTAC runtime deployment](scripts/live_univtac/README.md#blackwell-sm_120-isolated-bootstrap).

```bash
robotactile deployment init
export DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
export ISAAC_ARCHIVE="$DEPLOY_ROOT/runtime/isaac-sim-standalone-4.5.0-linux-x86_64.zip"
export ISAAC_ARCHIVE_SHA256='<independently-recorded-64-hex-sha256>'
export CUDA_RUNFILE="$DEPLOY_ROOT/runtime/cuda_12.4.1_550.54.15_linux.run"
export CUDA_RUNFILE_SHA256='<independently-recorded-64-hex-sha256>'

bash scripts/live_univtac/install_isaac_sim_4_5.sh \
  --archive "$ISAAC_ARCHIVE" \
  --sha256 "$ISAAC_ARCHIVE_SHA256"

bash scripts/live_univtac/smoke_isaac_sim_4_5.sh \
  --gpu 0 --run-id rtx3090-headless-v1

bash scripts/live_univtac/install_cuda_toolkit_12_4.sh \
  --runfile "$CUDA_RUNFILE" \
  --sha256 "$CUDA_RUNFILE_SHA256"

bash scripts/live_univtac/install_isaaclab_v2_1_1.sh
bash scripts/live_univtac/install_curobo_v0_7_7.sh
bash integrations/install_univtac.sh
bash scripts/live_univtac/install_tacex_univtac.sh
bash scripts/live_univtac/install_tacex_uipc_univtac.sh
bash scripts/live_univtac/install_robotactile_isaac.sh
bash scripts/live_univtac/qualify_univtac_task_import.sh
bash scripts/live_univtac/qualify_univtac_task_reset.sh \
  --gpu 0 --run-id pull-out-key-reset-s17-v1
```

The CUDA runfile requests only a deployment-local toolkit, but NVIDIA's vendor
installer can still create `/usr/local/cuda` and write
`/var/log/cuda-installer.log` when invoked as root. RoboTactile refuses to run
if `/usr/local/cuda` already exists; otherwise it removes that path only when
the runfile creates a symlink to the exact deployment-local toolkit (accepting
the vendor's single trailing slash). The
vendor log is neither deleted nor managed by RoboTactile.

The standalone runtime must use its bundled `python.sh`; a core development process
is not silently treated as Isaac execution. Continue with the complete
[Isaac Sim installation and UniVTAC execution guide](docs/isaac_sim.md) for
external checkouts, wheel installation, live preflight, request generation,
live execution, troubleshooting, and artifact acceptance criteria.

Official ACT is imported in-process by that same Isaac-local Python. The ACT
setup therefore installs no second Python runtime and must not overwrite
Isaac's bundled Torch/Torchvision. A source checkout or an ACT-only conda
environment is not a substitute for an Isaac-local import/load smoke.

Passing the headless smoke establishes only
`evidence_boundary=infrastructure_launch_only`. Live traces remain
`unqualified_live_univtac_execution_v1` with
`simulator_qualification_claimed=false` until a separate qualification
protocol is completed.

Before starting UniVTAC, validate the exact request and deployment resources:

```bash
export ACT_TASK=pull_out_key
export ACT_PROFILE=univtac

python scripts/act/install_official_artifacts.py \
  --artifact-root "$DEPLOY_ROOT/artifacts/models/act" \
  --task "$ACT_TASK" \
  --profile "$ACT_PROFILE"

robotactile integrations configure act \
  --root "$DEPLOY_ROOT" \
  --task "$ACT_TASK" \
  --profile "$ACT_PROFILE"

robotactile preflight-live \
  --request "$REQUEST_PATH" \
  --root "$DEPLOY_ROOT" \
  --config "$DEPLOY_ROOT/artifacts/models/act/configs/$ACT_TASK/$ACT_PROFILE/integration_config.json"
```

The receipt always records `simulator_execution_claimed=false` and does not
allocate Isaac, the policy, or a UniVTAC task.

Clean/Faulted runs require only the `univtac` profile. Add the separately
trained `vision_only` artifact only for the matched no-touch control. Upstream
metadata is reference material, not a RoboTactile result; report the released
weights as `ACT-official`, because their public metadata does not prove
frozen40 exclusion. Reserve `ACT-train759` for separately retrained,
split-manifest-bound weights.

## ACT primary robustness workflow

The current public 14 x 5 live-matrix workflow is ACT-specific and has four
explicit stages:

1. generate and execute a separate clean calibration request, then build rest
   references when required;
2. generate the complete hash-bound 14 x 5 primary request bundle;
3. materialize it with `--max-new-cells 0`, then execute one complete paired
   GPU batch from a fresh matrix output;
4. strictly reload the completed matrix and regenerate the source-bound report.

For a mixed Clean/Faulted/No-touch paired session, pass `--root` and omit
`--config`; the runner loads both canonical task configs from
`artifacts/models/act/configs/<task>/{univtac,vision_only}/`. A single-profile
doctor, preflight, or `live-univtac-run` may still receive its explicit
`integration_config.json`.

See [`docs/benchmark_workflow.md`](docs/benchmark_workflow.md) for complete
commands, required directory layouts, matrix executor responsibilities, and
report interpretation. The small request-generation matrix and the full 14 x
5 benchmark matrix are deliberately different artifacts; neither is described
as the other.

The tracked reporting-spec example matches the default `pull_out_key` ACT
system ID. For another task/system, generate the canonical spec as documented
in `docs/benchmark_workflow.md` rather than editing a result after execution.
The reporting command is:

```bash
source .venv/bin/activate
export MATRIX_ID="pull_out_key-i17-e29"
export PRIMARY_REQUEST_ROOT="$DEPLOY_ROOT/requests/primary-matrix/$MATRIX_ID"
export PRIMARY_MATRIX_OUTPUT="$DEPLOY_ROOT/outputs/matrices/$MATRIX_ID"

python -m robotactile_benchmark.cli report-matrix \
  --matrix-manifest "$PRIMARY_REQUEST_ROOT/matrix_manifest.json" \
  --matrix-output "$PRIMARY_MATRIX_OUTPUT" \
  --reporting-spec configs/reporting_spec.example.json \
  --output "$DEPLOY_ROOT/outputs/reports/$MATRIX_ID"
```

It strict-loads `matrix_summary.json`, every cell receipt, and each referenced
closed-loop bundle from
`<matrix-output>/artifacts/<root_receipt_sha256>/`. Unknown evidence types,
missing artifacts, altered hashes, focused grids, and incomplete primary grids
fail closed.

## Artifact and schema contracts

JSON Schemas in [`schemas/`](schemas/) cover rest-reference calibration, the
live request, no-allocation preflight receipt, official ACT artifact manifest,
live root receipt, request-generation receipt, matrix manifest/cell/summary,
reporting inputs, benchmark summary, and report receipt.
Runtime loaders remain the semantic authority: they additionally enforce
canonical JSON, duplicate-key rejection, path safety, content hashes, exact
inventories, and cross-file links.

Every fault manifest materializes its schedule, source map, spatial template,
and transfer parameters. A2 stores its exact `erased_offsets`; it never
resamples gaps during replay. F1--F4, F6, and registered-pixel C2 bind an
immutable measured no-contact `RestReferenceBundle`. F5 and F7 operate on the
current RGB frame and do not composite a second marker lattice.

## Distribution policy

Hash-locked pip environments and source/wheel installation commands are
documented in [`requirements/README.md`](requirements/README.md).

The wheel contains the importable Python package plus runtime configs, JSON
Schemas, license, third-party notices, the external-integration lock, and the
frozen source manifest. The sdist additionally contains `docs/`, `examples/`,
`integrations/` installers, operational `scripts/`, CI metadata, and tests.
Deployment shell/Python scripts are intentionally not copied into the wheel:
they are source-tree operational entry points with sibling-file and
shell-layout dependencies, not importable runtime resources. Use a source
checkout or the sdist for deployment.

After the hash-locked environment has been installed once, both artifacts can
be built without a resolver or network access:

```bash
make build
```

## Upstream and limitations

The official N0-TWAM checkout is frozen at
`c43a2160dd31c449d92b28eab52c0e2f09e4738a`; base and UniVTAC delta weights
are pinned separately to immutable Hugging Face revisions. The package does
not modify N0-TWAM or UniVTAC. FTP-1 source is independently frozen at
`89fa681d6c014cce28300946b7526db808e0b1c1`, and its six-task checkpoint
collection at `620ac69b4fffd2341300cfef1b1d224d56710ed3`. FTP-1 source is
Apache-2.0, but the checkpoint repository does not specify a weight license and
Gemma terms may also apply. The live snapshot adapter uses explicitly audited,
commit-pinned UniVTAC/UIPC/IsaacLab runtime surfaces and fails closed when
those surfaces drift. See
[`docs/upstream_provenance.md`](docs/upstream_provenance.md) and
[`docs/external_dependencies.md`](docs/external_dependencies.md), plus
[`docs/paper_code_traceability.md`](docs/paper_code_traceability.md) for the
frozen sources and evidence limits.

No FTP-1 Success Rate, Isaac-qualified closed-loop result, or real-robot result
is bundled in the current release. A request receipt proves request generation;
an install or headless smoke receipt proves only the operation named by that
receipt; an unqualified live trace proves only its verified execution boundary.

Dream-Tac source is independently frozen at
`14bab51d6862fd07124745c55cd395ea5caa9fd3` and remains external. No Dream-Tac
source, checkpoint, dataset statistics, or T5 embedding artifact is copied into
the wheel. Until an exact task-aligned checkpoint and a CASA-parity inference
route are available and qualified, RoboTactile does not claim Dream-Tac model
execution, UniVTAC closed-loop success, paper-number reproduction, or an
official benchmark result.

Before comparing a Clean result with the public N0-TWAM UniVTAC number, run
the artifact-only protocol gate (it never starts an episode):

```bash
robotactile n0-protocol-alignment \
  --root "$PWD/deployment" \
  --manifest "$PWD/deployment/requests/clean-campaigns/<campaign>/campaign_manifest.json"
```

The command writes canonical JSON and Markdown below
`deployment/outputs/clean-campaigns/<campaign>/parity/`. It separates `FAIL`
(contradictory evidence) from `UNKNOWN` (missing or unpublished evidence), and
will not compute a difference from the public 84.5% reference unless every
paper-comparison gate passes. Request generation defaults to
`initial_state_policy=official_reproduction`, so reset-time terminal signals are
recorded but do not independently reject the episode. The opt-in
`replace_initial_terminal_v1` policy is a separate robustness diagnostic and
must not be compared directly with 84.5%. A `diagnostic_v1` campaign with one
trial per task likewise remains an integration diagnostic, even when all
artifacts are valid. These contracts do not claim that a new GPU campaign has
already been executed.

## Citation, license, and contributions

Use [CITATION.cff](CITATION.cff) when citing RoboTactile. RoboTactile-owned
source is Apache-2.0; external model and simulator repositories retain their
upstream licenses. N0-TWAM source/base assets are distributed separately under
CC-BY-NC-SA-4.0; the released UniVTAC post-trained checkpoint declares
Apache-2.0. Neither is copied into the RoboTactile wheel.

Before opening a contribution, read [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md). Do not submit datasets, checkpoints, credentials,
private hostnames, or generated benchmark outputs.
