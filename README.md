# RoboTactile Benchmark

RoboTactile is an auditable robustness benchmark for optical-tactile robot
policies. The package implements the paper's 2 Availability + 7 Fidelity +
3 Temporal + 2 Context operators, four matched conditions (`clean`,
`faulted`, `no_touch`, and `restored`), exact single-process paired matrix
execution, and source-bound paper reports.

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
[Recorded N0 evaluation](docs/recorded_n0.md) ·
[Benchmark workflow](docs/benchmark_workflow.md) ·
[Evidence levels](docs/evidence_levels.md) ·
[Benchmark card](BENCHMARK_CARD.md)

Project information: [Citation](CITATION.cff) · [License](LICENSE) ·
[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) ·
[Third-party notices](THIRD_PARTY_NOTICES.md)

The benchmark treats tactile availability as a typed observation contract.
Structural absence is `payload=None`; a black or resting image is never used
as an Availability fault or as the matched no-touch control.

## Choose a path

| Goal | Start with | Hardware |
|---|---|---|
| Verify the package and 14 operators | [Reproducibility guide](docs/reproducibility.md#1-clone-and-verify-the-package) | Python >= 3.9 |
| Prepare the canonical N0-TWAM workspace | `robotactile setup --model n0_twam` | NVIDIA Linux for live execution |
| Install Isaac and run a headless GPU smoke | [Isaac Sim guide](docs/isaac_sim.md) | NVIDIA Linux |
| Qualify one UniVTAC task reset and observation | `bash scripts/live_univtac/qualify_univtac_task_reset.sh --help` | NVIDIA Linux |
| Run one paired four-condition ACT/UniVTAC set | `robotactile live-univtac-paired-run --help` | NVIDIA Linux + artifacts |
| Run one N0-TWAM Clean episode | [N0 Clean reproduction](docs/reproducibility.md#6-run-one-n0-clean-episode) | Qualified NVIDIA deployment |
| Serve official N0-TWAM UniVTAC weights | [N0-TWAM integration](docs/model_integrations.md#n0-twam-end-to-end) | Each selected GPU >= 40 GB |
| Measure N0 on a real recorded UniVTAC anchor | [Recorded N0 evaluation](docs/recorded_n0.md) | N0 GPU server; no Isaac loop |
| Run the current ACT 14 x 5 live matrix | [Benchmark workflow](docs/benchmark_workflow.md#4-materialize-and-run-the-primary-matrix) | Qualified ACT deployment |
| Render a verified live trace as PNG/MP4 | `robotactile visualize-live-artifact --help` | Pillow; ffmpeg for video |

N0-TWAM shares the same `PolicyAdapter`, observation-delivery, trace, and
reporting contracts. Its official websocket transport, released delta
checkpoint, rot6d-to-EE action bridge, and causal KV-cache re-grounding are
registered; live evidence still requires real weights, a reachable server,
and Isaac Sim. The public N0 Clean campaign is implemented. The public 14 x 5
primary-matrix generator remains ACT-specific because it binds tactile and
matched vision-only ACT checkpoints; a one-command N0 fault campaign is not
yet part of this release.

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
| N0 recorded-HDF5 evaluation | Implemented | `recorded-n0` | No simulator success |
| N0 UniVTAC Clean | Implemented | One-task, all-8, pilot, paper campaign owners | Qualification and sample size remain separate gates |
| ACT UniVTAC fault robustness | Implemented | 14 x 5 primary matrix | Requires real ACT artifacts and qualification |
| N0 UniVTAC fault robustness | Low-level injector/policy path implemented | **Campaign driver pending** | No end-to-end N0 robustness claim yet |

## Repository layout

```text
RoboTactile/
├── src/robotactile_benchmark/
│   ├── integrations/{act,n0_twam}/  # first-class policy integrations
│   ├── backends/                    # UniVTAC facade, lifecycle, and qualification
│   ├── operators/                   # 2 + 7 + 3 + 2 fault operators
│   ├── streaming/                   # causal online fault injection
│   ├── closed_loop/                 # typed runner and trace contracts
│   ├── calibration/                 # measured rest-reference selection/artifacts
│   ├── matrix/                      # exact paired matrix + completed-run reuse
│   └── reporting/                   # source-bound metrics and reports
├── configs/                         # frozen runtime registries
├── schemas/                         # public JSON Schemas
├── integrations/                    # external pins and install scripts
├── examples/{act,n0_twam}/          # model-specific examples
├── scripts/{live_univtac,n0_twam}/  # Isaac and official N0 deployment entry points
├── docs/                             # installation and evaluation guides
└── deployment/                       # ignored writable workspace
    ├── sources/{UniVTAC,WorldArena,N0-TWAM,IsaacLab,curobo}/
    ├── runtime/{isaac-sim-4.5.0,cuda-toolkit-12.4}/
    ├── artifacts/models/{act,n0_twam}/
    ├── requests/{calibration,four-condition,primary-matrix}/
    └── outputs/{matrices,reports}/
```

Create and inspect the complete tree with `robotactile deployment init` and
`robotactile deployment show`. The default root is this checkout's
`deployment/`; `ROBOTACTILE_DEPLOY_ROOT` provides an absolute HPC override
without changing the subdirectory contract. See the
[deployment layout guide](docs/deployment_layout.md).

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
- one-process paired ACT execution with a single canonical reset, in-memory
  UIPC/PhysX snapshot replay, and an exact pre-delivery equivalence gate;
- deterministic rest-reference calibration from clean UniVTAC traces, with a
  consecutive no-contact predicate, source-root binding, and strict artifacts;
- a clean calibration-request generator that binds split, policy, seed, and
  runtime identities before any simulator allocation;
- a 14 x 5 primary matrix with shared clean/no-touch baselines, one complete
  snapshot-paired GPU batch, explicit crash/unsupported/validator-rejected
  states, and content-addressed receipts;
- a public `generate-primary-matrix` CLI that expands the registered 14 x 5
  grid into 142 exact execution cells and a portable, hash-verified resource
  bundle without allocating the simulator;
- a public `run-live-matrix` CLI whose production path runs all cells in one
  shared runtime and only reuses a fully completed, strictly verified matrix;
- task-stratified reporting with SR, paired delta-SR, gated TGR, native-dose
  curves, recovery summaries, bootstrap intervals, exact McNemar tests, and
  Holm correction;
- deterministic JSON, CSV, LaTeX, and SVG report bundles bound to every source
  artifact root.

## Evidence levels

RoboTactile does not promote one evidence tier into another.

| Tier | What it establishes | What it does not establish |
|---|---|---|
| Software contracts | Deterministic delivery, adapter state machines, exact pairing gates, completed-run reuse, and report plumbing | A learned policy, Isaac Sim, task success, or real hardware |
| Live preflight | Frozen request, external checkouts, model artifacts, NVIDIA host, and Isaac Python passed without allocation | Simulator launch, task execution, or task success |
| Unqualified live | The requested official ACT or N0 runtime path executed and produced a hash-verified trace | Isaac qualification, physical calibration, or a publishable simulator result |
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

Passing the headless smoke establishes only
`evidence_boundary=infrastructure_launch_only`. Live traces remain
`unqualified_live_univtac_execution_v1` with
`simulator_qualification_claimed=false` until a separate qualification
protocol is completed.

Before starting UniVTAC, validate the exact request and deployment resources:

```bash
robotactile preflight-live \
  --request "$REQUEST_PATH" \
  --config "$DEPLOY_ROOT/artifacts/models/act/integration_config.json"
```

The receipt always records `simulator_execution_claimed=false` and does not
allocate Isaac, the policy, or a UniVTAC task.

## ACT primary robustness workflow

The current public 14 x 5 live-matrix workflow is ACT-specific and has four
explicit stages:

1. generate and execute a separate clean calibration request, then build rest
   references when required;
2. generate the complete hash-bound 14 x 5 primary request bundle;
3. materialize it with `--max-new-cells 0`, then execute one complete paired
   GPU batch from a fresh matrix output;
4. strictly reload the completed matrix and regenerate the source-bound report.

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
not modify N0-TWAM or UniVTAC. The live snapshot adapter uses explicitly audited,
commit-pinned UniVTAC/UIPC/IsaacLab runtime surfaces and fails closed when
those surfaces drift. See
[`docs/upstream_provenance.md`](docs/upstream_provenance.md) and
[`docs/external_dependencies.md`](docs/external_dependencies.md), plus
[`docs/paper_code_traceability.md`](docs/paper_code_traceability.md) for the
frozen sources and evidence limits.

No Isaac-qualified closed-loop result or real-robot result is bundled in the
current release. A request receipt proves request generation; an install or
headless smoke receipt proves only the operation named by that receipt; an
unqualified live trace proves only its verified execution boundary.

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
