# Dream-Tac UniVTAC training entry points

This materializer reuses the frozen RoboTactile UniVTAC split and writes only
`train759`. It never creates a `val/` tree or materializes `frozen40` or the
quarantined episode.

Run these modules from the pinned external Dream-Tac Python 3.10/CUDA runtime,
with the RoboTactile checkout on `PYTHONPATH`. Do not install that stack into
system Python, Isaac Sim, or N0-TWAM. Dream-Tac P2/P3 supports one NVIDIA CUDA
node with one to eight local GPUs. Do not use the N0-TWAM HPU launcher. The
separate `scripts/dream_tac/hcu_port/` tooling is a bounded experimental HCU
compatibility route and does not yet provide a formal P2/P3 training launcher.

## P0/P1: train759 and T5 inputs

```bash
python -m scripts.dream_tac.training.materialize_train759 \
  --raw-root /absolute/UniVTAC-HDF5 \
  --output-root /absolute/dream-tac-univtac --dry-run

python -m scripts.dream_tac.training.materialize_train759 \
  --raw-root /absolute/UniVTAC-HDF5 \
  --output-root /absolute/dream-tac-univtac
```

The dataset is written below `dataset/train/<task>/episode_<id>/`. Every HDF5
contains qpos6 at source row `t`, absolute action7 from row `t+1`, and paths to
all four 10 Hz RGB/tactile MP4 streams. Quaternion WXYZ signs are made
continuous before XYZ Euler conversion and episode-level unwrap. The HDF5
`task_name` is the exact natural-language T5 cache key.

Generate real embeddings with the pinned upstream checkout; the materializer
only writes an identity-bound request and never fabricates tensors:

```bash
python -m scripts.dream_tac.training.t5_cache \
  --dream-tac-root /absolute/Dream-Tac \
  --output-root /absolute/dream-tac-univtac

python -m scripts.dream_tac.training.t5_cache \
  --output-root /absolute/dream-tac-univtac \
  --verify
```

The pinned T5 path uses `google-t5/t5-11b`, `local_files_only=True`, and the
pinned Cosmos encoder on CPU FP32. All eight prompts are batched only to their
maximum active token length, then each finite result is converted to BF16 and
zero-padded to `[1,512,1024]`. The signed receipt records the runtime contract
and explicit NaN/Inf counts; a non-finite tensor is rejected before cache
publication. Pre-populate the exact licensed model snapshot in the runtime
cache; this command does not authorize a substitute encoder or an unrecorded
download. Use `--verify` for read-only receipt/hash verification.

## P2/P3: one-node NVIDIA launch

Copy one checked-in example to a machine-local request and replace every
absolute path and all-zero digest:

- `configs/experiments/dream_tac/p2_micro.example.json`;
- `configs/experiments/dream_tac/p3_full.example.json`;
- schema: `configs/experiments/dream_tac/training_request.schema.json`.

Then progress through the gates explicitly:

```bash
python -m scripts.dream_tac.training.launch_training \
  --request /absolute/requests/dream-tac-p2.json \
  --print-command

python -m scripts.dream_tac.training.launch_training \
  --request /absolute/requests/dream-tac-p2.json \
  --preflight-only

python -m scripts.dream_tac.training.launch_training \
  --request /absolute/requests/dream-tac-p2.json \
  --dry-run

python -m scripts.dream_tac.training.launch_training \
  --request /absolute/requests/dream-tac-p2.json
```

Only the final command starts `torchrun`. `--print-command` is ungated and
write-free; `--preflight-only` writes a content-addressed receipt but does not
start the trainer; `--dry-run` executes upstream config validation without
optimizer steps. The launcher requires a clean pinned Dream-Tac checkout, the
real Cosmos-Predict2 2B base file and SHA256, a complete train759 receipt, a
verified T5 cache, Linux/Python 3.10/CUDA, and an exact GPU list.

Receipts live under
`<output_root>/_robotactile_receipts/<run_name>/<request_sha256>/`. A result
receipt proves the recorded subprocess exit and log identity, not checkpoint
quality or task success. Resume must bind the exact checkpoint file and
SHA256; if a same-job `latest_checkpoint.txt` exists, it must name that exact
file. Never remove receipts or overwrite logs to relaunch an ambiguous run.

See [the full P0--P4 contract](../../../docs/dream_tac_training.md) for the
split, temporal/action, base/T5, resume, hardware, promotion, and evidence
boundaries.
