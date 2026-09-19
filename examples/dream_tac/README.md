# Dream-Tac integration example

Dream-Tac is a first-class RoboTactile `PolicyAdapter` backed by the pinned
upstream Franka HTTP service. Its source is frozen at
`14bab51d6862fd07124745c55cd395ea5caa9fd3` and remains outside the
RoboTactile wheel. Checkpoints, dataset statistics, and T5 embeddings also
remain external.

The adapter sends two RGB images, two delivered tactile RGB images, the exact
instruction, and a 6D XYZ/RPY state to `/infer`. It requires an absolute
`(20, 7)` XYZ/RPY/gripper response and converts every row to EE8
`[xyz, quaternion_wxyz, gripper_qpos]`. The artifact binds both `control_hz`
and an explicit gripper contract. This upstream-style example uses a calibrated
polarity/threshold; a UniVTAC-retrained model instead uses continuous
`direct_qpos_v1` plus its train-derived bounds. Do not infer either contract
from the example.

Install the source checkout and configure real user-supplied files:

```bash
robotactile deployment init
bash integrations/install_dream_tac.sh

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
```

`artifact_manifest.example.json` documents the schema shape only. Its paths
do not exist and its all-zero file hashes deliberately prevent runtime use.
Generate a canonical manifest from real files instead of editing the example.

## Train a UniVTAC-aligned model instead

When no suitable checkpoint is available, use the separate train-only
materialization route. It freezes the complete source grid as
`train759/frozen40/quarantine1`, writes only `train759`, preserves 10 Hz, and
creates `T - 1` next-row action targets with a 20-step/two-second horizon:

```bash
python -m scripts.dream_tac.training.materialize_train759 \
  --raw-root /absolute/path/to/UniVTAC-HDF5 \
  --output-root /absolute/path/to/dream-tac-univtac \
  --dry-run

python -m scripts.dream_tac.training.materialize_train759 \
  --raw-root /absolute/path/to/UniVTAC-HDF5 \
  --output-root /absolute/path/to/dream-tac-univtac

python -m scripts.dream_tac.training.materialize_train759 \
  --raw-root /absolute/path/to/UniVTAC-HDF5 \
  --output-root /absolute/path/to/dream-tac-univtac \
  --verify
```

This CPU stage produces source, dataset, prompt, statistics, and receipt
artifacts. It does not download a base model, synthesize T5 embeddings, train a
model, or launch Isaac Sim. Run it from the pinned external Dream-Tac runtime
whose Franka dependency group supplies HDF5/OpenCV support; RoboTactile's core
wheel intentionally remains NumPy-only. See
[Dream-Tac UniVTAC retraining](../../docs/dream_tac_training.md) for the P0--P4
workflow and evidence gates. A trained UniVTAC model must use continuous
`direct_qpos_v1` gripper delivery with train-derived bounds.

As of 2026-08-30, the official documented channels did not expose a
downloadable Dream-Tac checkpoint, no UniVTAC task-aligned checkpoint was
public, and the upstream HTTP inference path omitted the paper's CASA inference
gate. This example therefore establishes no OFFLINE, CLOSED-LOOP, OFFICIAL, or
Success Rate result; the integration remains `release_ready=false`.
