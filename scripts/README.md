# Operational scripts

Run every command in this directory from the RoboTactile repository root.
Scripts resolve sibling files from that root and write only below the selected
deployment directory. They are included in the source distribution, but not
in the importable wheel.

## Entry-point map

| Directory or script | Purpose | Normal user entry point |
|---|---|---|
| `bootstrap_pip.sh` | Create the hash-locked development environment | `bash scripts/bootstrap_pip.sh` |
| `n0_twam/` | Install, prepare, serve, and evaluate official N0-TWAM | `run_clean_all_tasks_once.sh` or the campaign owner |
| `live_univtac/` | Install and qualify Isaac Sim, UniVTAC, TacEx, IsaacLab, and cuRobo | [`live_univtac/README.md`](live_univtac/README.md) |
| `release/` | Build or verify the content-addressed source inventory | `update_source_manifest.py --check` |

## Public versus internal entry points

Use these public orchestration layers:

- `scripts/n0_twam/run_clean_all_tasks_once.sh` for the bounded eight-task N0
  Clean diagnostic;
- `scripts/n0_twam/generate_clean_campaign_requests.py` followed by
  `scripts/n0_twam/run_clean_campaign_all_tasks.py` for an explicitly frozen
  N0 Clean campaign;
- `robotactile generate-primary-matrix` and `robotactile run-live-matrix` for
  the current ACT primary robustness matrix;
- `robotactile visualize-live-artifact` to render an existing verified trace.

Files named `run_clean_task_shard.py`, lifecycle probes, attestation builders,
and task qualification helpers are owned plumbing. They are intentionally
available for audit and targeted diagnosis, but external users should not call
them as substitutes for the public campaign owner.

## Current N0 robustness boundary

The streaming injector and live N0 policy path accept typed fault manifests,
but the released primary-matrix generator is currently ACT-specific because
it binds tactile and matched vision-only ACT checkpoints. There is not yet a
public N0 14-operator campaign generator. Do not translate an ACT matrix
command into an N0 result by editing JSON manually.

The exact supported reproduction surfaces and commands are maintained in
[`docs/reproducibility.md`](../docs/reproducibility.md).

## Safety and generated state

- Do not place credentials, datasets, checkpoints, simulator archives, or
  generated results in `scripts/`.
- Use `ROBOTACTILE_DEPLOY_ROOT` or `--root` for a task-specific deployment.
- Installers reject broad roots, dirty pinned checkouts, mismatched hashes, and
  existing foreign destinations.
- `temp/`, `deployment/`, `outputs/`, and model files are ignored and are not
  release source.
