# RoboTactile Benchmark

RoboTactile is an auditable robustness benchmark for optical-tactile robot
policies. The package implements the paper's 2 Availability + 7 Fidelity +
3 Temporal + 2 Context operators, four matched conditions (`clean`,
`faulted`, `no_touch`, and `restored`), resumable matrix execution, and
source-bound paper reports.

The public repository and project are named **RoboTactile**. The installable
distribution remains `robotactile-benchmark`, the stable Python import is
`robotactile_benchmark`, and the command-line entry point is `robotactile`.

Start here: [Installation](docs/installation.md) ·
[Quickstart](docs/quickstart.md) ·
[Isaac Sim installation and execution](docs/isaac_sim.md) ·
[Deployment layout](docs/deployment_layout.md) ·
[External dependencies](docs/external_dependencies.md) ·
[Model integrations](docs/model_integrations.md) ·
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
| Inspect the benchmark and both model integrations | `robotactile integrations list` | Any supported Python host |
| Prepare the canonical ACT workspace and see missing gates | `robotactile setup --model act` | Any supported Python host |
| Install Isaac and run a headless GPU smoke | [Isaac Sim guide](docs/isaac_sim.md) | NVIDIA Linux |
| Run one four-condition ACT/UniVTAC trial | [Isaac Sim guide](docs/isaac_sim.md#8-generate-one-four-condition-request-set) | NVIDIA Linux + artifacts |
| Run the complete 14 x 5 evaluation and report | [Benchmark workflow](docs/benchmark_workflow.md) | Qualified deployment |

N0-TWAM shares the same `PolicyAdapter`, manifests, runner, and reporting
contracts. Live N0 execution remains fail-closed until a production transport
and published serving artifacts are registered.

## Repository layout

```text
RoboTactile/
├── src/robotactile_benchmark/
│   ├── integrations/{act,n0_twam}/  # first-class policy integrations
│   ├── simulators/univtac/          # simulator facade and qualification
│   ├── operators/                   # 2 + 7 + 3 + 2 fault operators
│   ├── closed_loop/                 # typed runner and trace contracts
│   ├── calibration/                 # measured rest-reference selection/artifacts
│   ├── matrix/                      # resumable evaluation matrix
│   └── reporting/                   # source-bound metrics and reports
├── configs/                         # frozen runtime registries
├── schemas/                         # public JSON Schemas
├── integrations/                    # external pins and install scripts
├── examples/{act,n0_twam}/          # model-specific examples
├── scripts/live_univtac/            # Isaac/UniVTAC deployment entry points
├── docs/                             # installation and evaluation guides
└── deployment/                       # ignored writable workspace
    ├── sources/{UniVTAC,WorldArena,N0-TWAM,IsaacLab,curobo}/
    ├── runtime/isaac-sim-4.5.0/
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
- deterministic rest-reference calibration from clean UniVTAC traces, with a
  consecutive no-contact predicate, source-root binding, and strict artifacts;
- a clean calibration-request generator that binds split, policy, seed, and
  runtime identities before any simulator allocation;
- a 14 x 5 primary matrix with shared clean/no-touch baselines, strict resume,
  explicit crash/unsupported/validator-rejected states, and content-addressed
  receipts;
- a public `generate-primary-matrix` CLI that expands the registered 14 x 5
  grid into 142 exact execution cells and a portable, hash-verified resource
  bundle without allocating the simulator;
- a public `run-live-matrix` CLI with bounded batches and strict resume from an
  exact per-cell resource config;
- task-stratified reporting with SR, paired delta-SR, gated TGR, native-dose
  curves, recovery summaries, bootstrap intervals, exact McNemar tests, and
  Holm correction;
- deterministic JSON, CSV, LaTeX, and SVG report bundles bound to every source
  artifact root.

## Evidence levels

RoboTactile does not promote one evidence tier into another.

| Tier | What it establishes | What it does not establish |
|---|---|---|
| Software contracts | Deterministic delivery, adapter state machines, resume, and report plumbing | A learned policy, Isaac Sim, task success, or real hardware |
| Live preflight | Frozen request, external checkouts, ACT artifacts, NVIDIA host, and Isaac Python passed without allocation | Simulator launch, task execution, or task success |
| Unqualified live | The requested official ACT/runtime path executed and produced a hash-verified trace | Isaac qualification, physical calibration, or a publishable simulator result |
| Isaac-qualified | Reserved for a separate qualification receipt and acceptance protocol | Not currently emitted by this repository |

`live-univtac-run` always writes
`evidence_level=unqualified_live_univtac_execution_v1` and
`simulator_qualification_claimed=false`. Running that command inside Isaac Sim
does not change those fields. The reporting receipt is a deterministic
derivation from verified inputs and also never claims simulator qualification.

## Isaac Sim and live UniVTAC

The canonical GPU path uses Ubuntu 22.04, an NVIDIA GPU such as RTX 3090,
Isaac Sim 4.5.0 standalone, IsaacLab v2.1.1, and cuRobo v0.7.7. Installation
is explicit: the core wheel never downloads or embeds these dependencies.

```bash
robotactile deployment init
export DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
export ISAAC_ARCHIVE="$DEPLOY_ROOT/runtime/isaac-sim-standalone-4.5.0-linux-x86_64.zip"
export ISAAC_ARCHIVE_SHA256='<independently-recorded-64-hex-sha256>'

bash scripts/live_univtac/install_isaac_sim_4_5.sh \
  --archive "$ISAAC_ARCHIVE" \
  --sha256 "$ISAAC_ARCHIVE_SHA256"

bash scripts/live_univtac/smoke_isaac_sim_4_5.sh \
  --gpu 0 --run-id rtx3090-headless-v1

bash scripts/live_univtac/install_isaaclab_v2_1_1.sh
bash scripts/live_univtac/install_curobo_v0_7_7.sh
```

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

## End-to-end workflow

The reproducible workflow has four explicit stages:

1. generate and execute a separate clean calibration request, then build rest
   references when required;
2. generate the complete hash-bound 14 x 5 primary request bundle;
3. materialize it with `--max-new-cells 0`, then execute bounded GPU batches;
4. resume the matrix and regenerate the source-bound report.

See [`docs/benchmark_workflow.md`](docs/benchmark_workflow.md) for complete
commands, required directory layouts, matrix executor responsibilities, and
report interpretation. The small request-generation matrix and the full 14 x
5 benchmark matrix are deliberately different artifacts; neither is described
as the other.

The reporting command is:

```bash
source .venv/bin/activate
python -m robotactile_benchmark.cli report-matrix \
  --matrix-manifest "$DEPLOY_ROOT/requests/primary-matrix/run/matrix_manifest.json" \
  --reporting-spec configs/reporting_spec.json
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

The N0-TWAM checkout is frozen at branch `UniVTAC-PostTraining`, commit
`9036c130409f8cf5494b12489fea339f7213b9d6`. The package does not modify or
import private paths from N0-TWAM or UniVTAC. See
[`docs/upstream_provenance.md`](docs/upstream_provenance.md) and
[`docs/external_dependencies.md`](docs/external_dependencies.md), plus
[`docs/paper_code_traceability.md`](docs/paper_code_traceability.md) for the
frozen sources and evidence limits.

No Isaac-qualified closed-loop result or real-robot result is bundled in the
current release. A request receipt proves request generation; an install or
headless smoke receipt proves only the operation named by that receipt; an
unqualified live trace proves only its verified execution boundary.

## Citation, license, and contributions

Use [CITATION.cff](CITATION.cff) when citing RoboTactile. RoboTactile-owned
source is Apache-2.0; external model and simulator repositories retain their
upstream licenses. In particular, N0-TWAM is distributed separately under
CC-BY-NC-SA-4.0 and is not copied into the RoboTactile wheel.

Before opening a contribution, read [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md). Do not submit datasets, checkpoints, credentials,
private hostnames, or generated benchmark outputs.
