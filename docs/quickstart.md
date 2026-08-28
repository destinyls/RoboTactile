# Quickstart

Run commands from the RoboTactile repository root. The complete, evidence-
bounded walkthrough is in [`reproducibility.md`](reproducibility.md).

## Verify the package

```bash
bash scripts/bootstrap_pip.sh
source .venv/bin/activate

robotactile validate-registry
robotactile smoke-replay \
  --operator T1_fixed_source_delay \
  --severity 3 \
  --output outputs/reproduction-smoke
make check
```

This verifies the package, the frozen 14-operator registry, and deterministic
replay. It does not allocate Isaac Sim or run a learned policy.

## Prepare the N0-TWAM deployment

```bash
export ROBOTACTILE_DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
export DEPLOY_ROOT="$ROBOTACTILE_DEPLOY_ROOT"

robotactile deployment init --root "$DEPLOY_ROOT"
robotactile integrations list
robotactile setup --model n0_twam --root "$DEPLOY_ROOT"
```

The first `setup` call may exit with status 2. Its canonical JSON output lists
the missing source, artifact, transport, or simulator gates; it does not start
Isaac or inference. Install the exact dependencies using
[`installation.md`](installation.md),
[`external_dependencies.md`](external_dependencies.md), and
[`isaac_sim.md`](isaac_sim.md), then rerun the same doctor.

For a copy-paste one-task N0 Clean campaign, continue at
[`reproducibility.md#6-run-one-n0-clean-episode`](reproducibility.md#6-run-one-n0-clean-episode).
For the bounded all-eight diagnostic, use
`scripts/n0_twam/run_clean_all_tasks_once.sh` only after every task-specific
artifact has been prepared.

## Select execution and capture explicitly

| Execution | Capture | Intended result |
|---|---|---|
| `quick` | `metrics_only_v1` | Fast diagnostic Success Rate |
| `diagnostic` | `preview_v1` | Metrics plus at most 64 keyframes/video |
| `claim` | `paper_full_v1` | Qualified pilot or paper evidence |

`quick` and `diagnostic` cannot publish a paper result. `claim` requires the
frozen pilot/paper protocol and source-bound qualification v3.

## Robustness workflow status

- ACT: the current public `generate-primary-matrix` and `run-live-matrix`
  commands implement the complete 14 x 5 live robustness matrix.
- N0-TWAM: Clean campaigns and low-level online fault delivery are implemented,
  but this release does not yet expose a public N0 fault-campaign generator.
- Recorded N0: [`recorded_n0.md`](recorded_n0.md) evaluates all operators on
  real HDF5 observations without claiming simulator task success.

Do not reuse the ACT primary-matrix command as an N0 result by editing generated
JSON. The formal ACT matrix and paper report are documented in
[`benchmark_workflow.md`](benchmark_workflow.md).
