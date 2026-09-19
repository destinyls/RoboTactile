# UniVTAC train759 materialization

This external data layer writes the LeRobot v2.1 repositories consumed by the
pinned official N0-TWAM checkout. It never edits upstream model code and never
materializes `frozen40` or the quarantined episode into the training tree.

Freeze the source manifest once before starting parallel task jobs:

```bash
python scripts/n0_twam/hpu_training/data/materialize_train759.py \
  --raw-root /mnt/data/task/UniVTAC \
  --output-root /mnt/data/task/n0_twam_train759_absee20 \
  --manifest-only
```

Then run one or more task transactions. Completed task repositories are
validated and skipped on retry; existing unreceipted destinations are rejected.

```bash
python scripts/n0_twam/hpu_training/data/materialize_train759.py \
  --raw-root /mnt/data/task/UniVTAC \
  --output-root /mnt/data/task/n0_twam_train759_absee20 \
  --tasks grasp_classify insert_HDMI
```

The final physical dataset is `train759/<task>/`, with eight task-local repos.
Each repo contains complete `meta/`, `data/`, and `videos/` plus
`_robotactile_task_receipt.json`. Once all eight repos exist, the last task job
also writes:

- `norm_stat_absee.json`
- `norm_stat_absee_per_repo.json`
- `norm_stat_absee_raw_report.json`
- `train759_receipt.json`

The source cadence is 10 FPS. N0's temporal compression therefore requires
`action_per_frame=4`; `12` would be the 30 FPS recipe and is not valid for these
physical source rows. State and action use the official 20D single-arm absEE
schema, with active channels `0..9`, padding channels `10..19`, and strict
`action[t] = state[t+1]` targets.
