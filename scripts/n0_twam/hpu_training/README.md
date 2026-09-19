# Official N0-TWAM training on the Lingchu cluster

This directory is an external RoboTactile orchestration layer. It deliberately
does not patch N0-TWAM training, model, dataset, or launcher code.

## Source and mutation boundary

- Official repository: `https://github.com/neoteai/N0-TWAM.git`
- Pinned commit: `cdd87b6a141667123ad2c25f452478afdb71e287`
- The only permitted upstream worktree change is
  `n0_twam/configs/twam_posttrain_cfg.py`.
- The launcher fails before training if `HEAD`, `origin`, the generated config
  digest, or the upstream worktree boundary differs.
- Passwords are never accepted. Node SSH must already work with a key and
  `BatchMode=yes`.

The cluster topology is supplied explicitly in `cluster.json`: exactly two
nodes with eight accelerator processes per node. The committed example uses
RFC 5737 documentation addresses; replace them only in the uncommitted run
request. The same request supports the three bounded launch modes below.

## Training contract

The generated config fixes the official validated UniVTAC recipe:

- physical training input is exactly a directory named `train759`;
- `val_dataset_path = None`, so the official trainer never creates a validation
  loader;
- both RGB keys and both tactile keys are present;
- `tactile_optional=False`, `synthetic_tactile_data=False`;
- `use_local_tactile=True`, `tactile_global_zero=False`;
- `tactile_cfg_prob=cfg_prob=noisy_cond_prob_tactile=0.0`;
- `tactile_diffusion_loss_weight=1.0`;
- `max_latent_frames=5`: the official dataset loader randomly samples a
  contiguous five-latent window from long episodes, retaining all train759
  episodes while bounding activation memory to the proven 2-node recipe;
- official 20D absolute-EE action schema, while preserving UniVTAC's real
  10 Hz source/LeRobot cadence: adjacent latent anchors have `frame_stride=1`,
  so each latent frame corresponds to exactly four actions
  (`action_per_frame=4`, metadata horizon 4). Frames are never duplicated to
  imitate the official documentation's separate 30 Hz/horizon-12 example.
- both model fields must point to the same official `NeoteAI/n0-twam-base`
  bundle; a UniVTAC post-trained checkpoint is rejected to avoid split leakage.

Consequently, an older qpos8-only conversion is not accepted as input to this
official action schema. Prepare an official-schema 20D absolute-EE, true-10-Hz
`train759` and a 20D q01/q99 normalizer in RoboTactile before launching.
Frozen40 may be retained for
later evaluation, but must not appear below `dataset_path`.

Every launch mode requires the complete signed conversion receipt and the
`final_inventory.json` produced by `latent/run_preprocessing.py`. Before
creating the official config, the launcher binds all 759 dense LeRobot IDs to
train-only source hashes, verifies the certified global/per-task normalizer
paths, and stats all 4,554 inventoried Vision/GlobalTactile/LocalTactile files.
Set `latent_inventory_path` in the copied training request to that completed
inventory; an incomplete preprocessing run cannot start training.

The runtime preflight also opens one source-bound Vision/tactile latent pair per
task and requires actual `frame_ids` stride 1 with `fps=ori_fps=10`. This is a
bounded witness over the complete inventory already validated and statted by
the controller.

## Run modes

Run these commands from a cluster node that can reach both nodes using
SSH keys and sees the shared `/mnt/data/task` filesystem.

```bash
ROOT=/mnt/data/task/n0_twam_track31_retrain_official_20260829
ROBOTACTILE=/mnt/data/task/RoboTactile

bash "$ROBOTACTILE/scripts/n0_twam/hpu_training/prepare_official_checkout.sh" \
  "$ROOT/source/N0-TWAM"

cp "$ROBOTACTILE/scripts/n0_twam/hpu_training/cluster.example.json" \
  "$ROOT/requests/cluster.json"
cp "$ROBOTACTILE/scripts/n0_twam/hpu_training/training.example.json" \
  "$ROOT/requests/training.json"
```

Edit the copied JSON paths and replace both documentation-only node addresses.
Do not put passwords or tokens in either file. Use a fresh `run-id` for every
command; the run directory and transient container names are no-clobber.

### 1. Single-card, one-step smoke

This selects the configured `master_addr`, exposes one HCU, starts one official
`torchrun` rank, and generates a config with exactly one optimizer step:

```bash
python "$ROBOTACTILE/scripts/n0_twam/hpu_training/launch_cluster.py" \
  --repo "$ROOT/source/N0-TWAM" \
  --training-spec "$ROOT/requests/training.json" \
  --cluster-spec "$ROOT/requests/cluster.json" \
  --run-root "$ROOT/runs" \
  --run-id n0-twam-single-card-smoke-v1 \
  --launch-mode single-card-smoke
```

### 2. Two-node distributed smoke

This starts 2 nodes x 8 HCU ranks with the official world-sharded FSDP2/RCCL
environment. The
default is two optimizer steps; `--smoke-steps` is fail-closed to 1 through 20:

```bash
python "$ROBOTACTILE/scripts/n0_twam/hpu_training/launch_cluster.py" \
  --repo "$ROOT/source/N0-TWAM" \
  --training-spec "$ROOT/requests/training.json" \
  --cluster-spec "$ROOT/requests/cluster.json" \
  --run-root "$ROOT/runs" \
  --run-id n0-twam-distributed-smoke-v1 \
  --launch-mode distributed-smoke \
  --smoke-steps 2
```

### 3. Formal two-node training

Formal mode uses the `num_steps` from `training.json` and starts one official
`torchrun` per node, with all 16 ranks joining the same static rendezvous:

```bash
python "$ROBOTACTILE/scripts/n0_twam/hpu_training/launch_cluster.py" \
  --repo "$ROOT/source/N0-TWAM" \
  --training-spec "$ROOT/requests/training.json" \
  --cluster-spec "$ROOT/requests/cluster.json" \
  --run-root "$ROOT/runs" \
  --run-id n0-twam-train759-vt-always-on-2n16-v1 \
  --launch-mode formal
```

The launcher first performs one lightweight contract/runtime preflight on every
selected node and then immediately starts training. It does not restart failed
runs. Each run directory is no-clobber and contains `status.json`, per-node
preflight logs, per-node torchrun logs, and the model output directory.
An exclusive lock under the checkout's `.git/` directory prevents two runs from
rewriting the single permitted config concurrently.

The runtime is pinned to image
`docker-registry-sh.tencentcloudcr.com/hg/hg_wm:v3.5` by its full image ID and
to the validated `librccl-net-shca.so.0.0.0` by SHA256. It mounts the proven
HCU/RDMA devices, `/opt/hyhal`, `/etc/hfm`, RCCL plugin, and shared `/mnt/data`,
then applies the proven 2-node RCCL transport environment. The pinned official
trainer uses FSDP2's default global mesh across all 16 ranks; RoboTactile records
that effective world-shard topology rather than labeling it HSDP. Every preflight
or training rank uses a unique transient container. An existing same-name
container causes a failure and is never removed; cleanup only stops/removes the
exact container ID created by this invocation.

After the official trainer returns successfully, the external runtime wrapper
performs one final all-rank barrier, flushes a per-rank success witness, and
uses a vendor-scoped fast exit. This avoids a known Torch 2.5 HIP/RCCL
communicator-destructor hang; exception paths never use the success exit.

Use `--preflight-only` only when validating a newly installed runtime; it is not
a required step for every experiment because the normal launch already includes
the same fail-closed preflight.
