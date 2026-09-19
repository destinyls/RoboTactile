# Operational scripts

Run every command in this directory from the RoboTactile repository root.
Scripts resolve sibling files from that root and write only below the selected
deployment directory. They are included in the source distribution, but not
in the importable wheel.

## Entry-point map

| Directory or script | Purpose | Normal user entry point |
|---|---|---|
| `bootstrap_pip.sh` | Create the hash-locked development environment | `bash scripts/bootstrap_pip.sh` |
| `act/` | Generate official ACT requests and verify an Isaac-local ACT runtime | [`examples/act/README.md`](../examples/act/README.md) |
| `dream_tac/training/` | Materialize and verify train-only UniVTAC inputs for Dream-Tac retraining | [`docs/dream_tac_training.md`](../docs/dream_tac_training.md) |
| `ftp1_policy/` | Install and serve the pinned FTP-1 runtime; prepare matched robustness groups | [`ftp1_policy/README.md`](ftp1_policy/README.md) |
| `n0_twam/` | Install, prepare, serve, and evaluate official N0-TWAM | `run_clean_all_tasks_once.sh` or the campaign owner |
| `retrained_evaluation/` | Bind retrained models to serial groups; run N0-TWAM task/condition shards with persistent N0 + Isaac | [`retrained_evaluation/README.md`](retrained_evaluation/README.md) |
| `live_univtac/` | Install and qualify Isaac Sim, UniVTAC, TacEx, IsaacLab, and cuRobo | [`live_univtac/README.md`](live_univtac/README.md) |
| `release/` | Build or verify the content-addressed source inventory | `update_source_manifest.py --check` |

## Public versus internal entry points

Use these public orchestration layers:

- `scripts/act/prepare_requests.py` for canonical ACT Clean/Faulted/matched
  No-touch requests, and `scripts/act/verify_official_runtime.py` for the
  read-only artifact/import/optional strict-load probe;
- `python -m scripts.dream_tac.training.materialize_train759` from the pinned
  external Dream-Tac runtime to plan, materialize, or verify its train759-only
  UniVTAC training dataset;
- `scripts/n0_twam/run_clean_all_tasks_once.sh` for the bounded eight-task N0
  Clean diagnostic;
- `scripts/n0_twam/generate_clean_campaign_requests.py` followed by
  `scripts/n0_twam/run_clean_campaign_all_tasks.py` for an explicitly frozen
  N0 Clean campaign;
- `scripts/n0_twam/run_frozen40_tactile_causal.py` for the complete 40-episode
  recorded Clean/structural-tactile-absence action diagnostic;
- `python -m scripts.retrained_evaluation.persistent_condition run` for one
  N0-TWAM task/condition across a seed range with one persistent N0 server and
  one persistent Isaac application;
- `scripts/ftp1_policy/prepare_robustness_group.py` followed by
  `robotactile live-univtac-paired-run` for one FTP-1 same-snapshot
  Clean/Faulted group;
- `robotactile generate-n0-fault-campaign` for canonical N0 Clean/Faulted
  requests, and `scripts/n0_twam/run_contact_ablation_pilot.py` only for the
  non-paper Clean/structural-absence/Clean-action-replay diagnostic;
- `robotactile generate-primary-matrix` and `robotactile run-live-matrix` for
  the current ACT primary robustness matrix;
- `robotactile visualize-live-artifact` to render an existing verified trace.

Files named `run_clean_task_shard.py`, lifecycle probes, attestation builders,
and task qualification helpers are owned plumbing. They are intentionally
available for audit and targeted diagnosis, but external users should not call
them as substitutes for the public campaign owner.

## Current N0 robustness boundary

The streaming injector and live N0 policy path accept typed fault manifests,
and the public N0 campaign generator records unsupported cells explicitly.
The released primary-matrix generator remains ACT-specific because it binds
tactile and matched vision-only ACT checkpoints. Do not translate an ACT
matrix command into an N0 result by editing JSON manually.

The exact supported reproduction surfaces and commands are maintained in
[`docs/reproducibility.md`](../docs/reproducibility.md).

## FTP-1 boundary

FTP-1 has a repo-local Python 3.11 runtime and six released task checkpoints.
The installer never writes to system Python/CUDA, Isaac, N0-TWAM, or N0-VTLA.
On Blackwell it installs and verifies the official cu128 bundle with a real
`sm_120` CUDA tensor probe. The PaliGemma tokenizer is hash-bound in the
deployment-local `OPENPI_DATA_HOME`, not `~/.cache/openpi`.
The robustness group executes F1--F7, T1--T3, and C1--C2; A1/A2 remain explicit
N/A for the fixed tactile-required input contract. Generating requests proves
only CODE readiness. A worker health/inference check is OFFLINE evidence. Only
a strict live artifact produced by the paired Isaac runner is CLOSED-LOOP
evidence, and no bundled script output should be described as a paper metric
until the declared live denominator is complete.

## Safety and generated state

- Do not place credentials, datasets, checkpoints, simulator archives, or
  generated results in `scripts/`.
- Use `ROBOTACTILE_DEPLOY_ROOT` or `--root` for a task-specific deployment.
- Installers reject broad roots, dirty pinned checkouts, mismatched hashes, and
  existing foreign destinations.
- `temp/`, `deployment/`, `outputs/`, and model files are ignored and are not
  release source.
