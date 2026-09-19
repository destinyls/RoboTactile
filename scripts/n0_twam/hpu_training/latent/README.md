# Official N0-TWAM train759 latent preprocessing

This directory orchestrates the two encoder CLIs already present in the pinned
official N0-TWAM checkout. It does not copy or patch either encoder.

Pinned encoder identities at official commit
`cdd87b6a141667123ad2c25f452478afdb71e287`:

- `script/encode_lerobot_n0_latents.py`:
  `c731103d95059b5ad769a0f28d2b68ba664c02f3b95c4557dbc632d27a6d14ff`
- `script/encode_tactile_latent.py`:
  `bd0975bf9330f60f6cd16adb9fbd28b6f96b577d09cbf06f0654ed3348eb7b79`

The orchestration fails if either digest, the official Git commit, or the
official worktree mutation boundary differs.

## Certified input boundary

The physical input must be `train759/` containing exactly eight task-level
LeRobot v2.1 repositories. Every dense LeRobot episode ID is bound through
`_robotactile_task_receipt.json` to a source HDF5 identity whose split is
`train` in `source_split_manifest.json`. Any frozen40 or quarantine identity,
missing mapping, duplicate source, or non-10-Hz repository is rejected before
an encoder starts.

The complete signed `train759_receipt.json` and all eight signed per-task
receipts are mandatory. The loader verifies the exact dense episode maps,
source hashes/ranges, converted frame counts, task repo/norm-key identity, and
the three normalization artifact hashes before scheduling work.

No files below frozen40 or quarantine directories are traversed.

## Official encoder invocations

The plan contains 16 explicit work units: one Vision and one tactile unit for
each task. Work is balanced so each of the two explicitly configured nodes runs
four Vision and four tactile workers, one per HCU. No internal node address is
stored in this repository.

Workers run in the same image-ID/plugin-SHA-pinned HCU Docker recipe used by
formal training; the host virtualenv is not used for PyTorch. Each transient
container is no-clobber and cleanup is limited to its exact created ID.

Each worker passes an explicit `--episodes` list to the unchanged official CLI:

- Vision: `--target-fps 10`, 256×256,
  `observation.images.top` and `observation.images.wrist_l`.
- tactile: `--target-fps 10`, 128×128, both tactile keys,
  `--mode both --local-mode current`.

Thus every episode must end with two Vision files, two GlobalTactile files and
two LocalTactile files. Source and target cadence remain the true 10 Hz; no
frame duplication or 30-Hz conversion is performed.

Existing valid outputs are loaded and verified, then reused. Only episode IDs
with one or more missing outputs are sent to the official encoder. Existing but
invalid/corrupt outputs cause a hard failure and are never overwritten. The
official `--overwrite` flag is never passed.

## Execute

Copy and edit the example outside the source tree, then run from a cluster node
with key-only access to both formal nodes and shared `/mnt/data/task` storage:

```bash
ROOT=/mnt/data/task/n0_twam_track31_retrain_official_20260829
ROBOTACTILE=/mnt/data/task/RoboTactile

cp "$ROBOTACTILE/scripts/n0_twam/hpu_training/latent/latent.example.json" \
  "$ROOT/requests/latent.json"

python "$ROBOTACTILE/scripts/n0_twam/hpu_training/latent/run_preprocessing.py" \
  --latent-spec "$ROOT/requests/latent.json" \
  --cluster-spec "$ROOT/requests/cluster.json" \
  --run-root "$ROOT/runs/latent" \
  --run-id train759-official-latents-2n16-v1
```

One run directory is no-clobber and contains:

- `plan.json` and 16 explicit episode shard files;
- two node preflight logs and 16 worker logs;
- 16 worker receipts, including reused/encoded episode IDs and payload metadata;
- `final_inventory.json` covering all 759 episodes and all 4,554 required files.

An interrupted run leaves already completed official outputs intact. Start a
new run ID; the successor validates them and submits only still-missing episode
IDs. `--inventory-only` verifies an already complete dataset without invoking
either encoder.
