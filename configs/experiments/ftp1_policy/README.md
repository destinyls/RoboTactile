# FTP-1 UniVTAC train759

FTP-1's released UniVTAC artifacts are task-specific. This RoboTactile recipe
is a new joint fine-tuning route: all eight train domains feed one
`MultiZarrDataset`, one normalization run, one optimizer, and one checkpoint.
It initializes directly from the official FTP-1 base model, never from a
task-specific fine-tune.

The formal request enforces:

- the exact `train759` split; `frozen40` and the quarantined episode are absent;
- pinned FTP-1 source, pretrained checkpoint, and UniVTAC dataset revisions;
- RGB plus both tactile pads on every training sample, with tactile dropout 0;
- eight instruction-conditioned domains and domain-homogeneous batches, so
  head-only and head+wrist tasks remain schema-compatible;
- 160,000 joint optimizer steps, preserving the former aggregate budget of
  20,000 updates per task, with deterministic seed 42 on one NVIDIA CUDA GPU;
- an exact RGB contract check against source HDF5 for external cameras and
  both tactile pads before normalization starts;
- no W&B requirement and no modification of system Python/CUDA.

Copy `train759_all8.example.json` into the deployment request tree, replace the
absolute placeholder paths, then run from the repository root:

```bash
python -m scripts.ftp1_policy.training.orchestrate \
  --request /absolute/request.json \
  --phase preflight

python -m scripts.ftp1_policy.training.orchestrate \
  --request /absolute/request.json \
  --phase all \
  --verify-source-hashes
```

`--phase parse --task TASK` is the only task-selective mode and only converts
one task. `--phase prepare` converts all eight tasks and performs one joint
normalization. `--phase train` launches one joint training run after prepare.
`--phase all` performs the complete sequence. A task selector is rejected for
joint normalization and training.

The pinned upstream parser currently reverses RGB channels twice. RoboTactile
keeps that checkout clean and applies a source-owned correction at the parser
output boundary, then verifies first/middle/last episodes and frames with
zero pixel difference. Zarr data without this RGB receipt cannot enter joint
normalization or training.

The public FTP-1 release still contains six task-specific checkpoints. The
single all-eight checkpoint produced here is a RoboTactile training artifact,
not an upstream released checkpoint. Its eight-task capability must be
reported only after per-task evaluation.
