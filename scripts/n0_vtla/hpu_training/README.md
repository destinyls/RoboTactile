# N0-VTLA UniVTAC training on Lingchu HCU

This directory implements both the official task-specific N0-VTLA UniVTAC
adaptation and a RoboTactile mixed-task extension. Neither path uses the frozen
validation episodes.

## Fixed experiment contract

- upstream source: `neoteai/N0-VTLA@03a0ce4d7091ca2354864796770715aa212601b7`;
- initialization: `NeoteAI/n0-vtla-base@ec12548b1bde68e377b02d6b659eb258aabd3306`;
- config: `sim_single_arm_tactile`;
- official reproduction: one policy per UniVTAC task;
- RoboTactile mixed8: one language-conditioned checkpoint for all eight tasks;
- state/action: 8D joint space, `state[t]`, `action[t] = state[t+1]`;
- model transform: seven joint deltas plus absolute gripper;
- 10 Hz, horizon 50, global batch 64, bf16;
- Vision and both tactile streams always present; no tactile/VL dropout;
- normalization statistics come from the exact training view only;
- `frozen40` is excluded from data views, normalization, checkpoint selection,
  and training.

The public repository describes one policy per task. The mixed8 path is an
explicit RoboTactile extension and must not be reported as an exact reproduction
of that task-specific recipe.

## Data view

The certified source split has 759 train episodes, 40 frozen evaluation
episodes, and one quarantined corrupt episode. Build task-local qpos8 views by
hardlinking the already validated videos and rewriting only Parquet state/action:

```bash
python -m scripts.n0_vtla.hpu_training.materialize_qpos8 \
  --raw-root /mnt/data/task/UniVTAC \
  --source-manifest /path/to/source_split_manifest.json \
  --source-train-root /path/to/validated/train759 \
  --output-root /path/to/univtac_qpos8_train759
```

Every published task root contains
`_robotactile_n0_vtla_qpos8_receipt.json` with `validation_data_used=false`.
Existing task roots are verified and reused; they are never overwritten.

Build one single-root mixed8 view from those eight certified task views:

```bash
python -m scripts.n0_vtla.hpu_training.materialize_mixed8 \
  --source-root PROJECT_ROOT/data/univtac_qpos8_train759 \
  --output-root PROJECT_ROOT/data/univtac_qpos8_train759
```

This rewrites only global `episode_index`, `index`, and `task_index` fields in
759 small Parquet files. The 3,036 RGB/tactile videos remain zero-copy
hardlinks. `meta/tasks.jsonl` contains eight exact prompts, so the official
`prompt_from_task=True` pipeline conditions one checkpoint on the current task.

## HCU runtime

The vendor Docker image owns PyTorch/HIP/RCCL. Install only CPU-side Python
dependencies into an isolated overlay:

```bash
bash scripts/n0_vtla/hpu_training/install_hcu_overlay.sh \
  /mnt/data/task/PROJECT/runtime/n0_vtla_py310_overlay
```

The requirements intentionally exclude Torch, CUDA, Triton, NVIDIA packages,
and JAX CUDA plugins. `runtime_shims/sitecustomize.py` is opt-in and only adapts
the upstream `DDP(init_sync=...)` call for vendor PyTorch 2.5.1; it does not
replace the accelerator runtime.

Before formal training, run one single-HCU optimizer step and one 8-rank
single-node smoke. A running process or finite first loss is start evidence,
not a completed checkpoint.

Launch each 8-HCU, 20,000-step task-specific formal run with
`launch_formal_task.sh TASK PROJECT_ROOT RUN_ID`. The launcher is no-clobber,
survives SSH disconnects, and records its PID, state, exit code, and log below
`PROJECT_ROOT/logs/supervisor/RUN_ID`. Formal runs use the official eight data
workers per rank; smoke runs retain two workers.

For one eight-task model, compute the unified train759 normalizer through
`docker_runtime.sh norm mixed8 PROJECT_ROOT 1 1 RUN_ID`, then run an eight-rank
one-step smoke. Start the exposure-matched formal run with:

```bash
bash PROJECT_ROOT/tooling/hpu_training/launch_formal_mixed8.sh \
  PROJECT_ROOT RUN_ID
```

The mixed8 claim profile trains for 160,000 optimizer steps from the same base
checkpoint. At 64 samples per step, its expected per-task exposure matches eight
20,000-step task-specific runs. A 20,000-step mixed pilot has only one eighth of
that per-task exposure and is not claim-equivalent.

### Four-node mixed8 formal training

The accelerated mixed8 launcher starts exactly four unique nodes with eight
HCUs per node. This 4x8 HCU topology uses DDP world size 32, microbatch one per
rank, two gradient-accumulation microsteps, effective global batch 64, and the
same 160,000-step claim profile. It always initializes from the pinned N0-VTLA
base checkpoint; resuming from the stopped single-node optimizer state is
forbidden.

Run the launcher from a login node that can reach all four workers by SSH. The
addresses and ports below are documentation-only placeholders and must be
replaced with the assigned cluster values. `--master-addr` must exactly equal
the first value passed to `--nodes`.

```bash
NODE_0=192.0.2.10
NODE_1=192.0.2.11
NODE_2=192.0.2.12
NODE_3=192.0.2.13
SSH_PORT=2222
MASTER_PORT=29640

python PROJECT_ROOT/tooling/hpu_training/launch_formal_mixed8_4node.py \
  --project-root PROJECT_ROOT \
  --run-id RUN_ID \
  --nodes "$NODE_0" "$NODE_1" "$NODE_2" "$NODE_3" \
  --ssh-user root \
  --ssh-port "$SSH_PORT" \
  --master-addr "$NODE_0" \
  --master-port "$MASTER_PORT"
```

Before training, the persistent supervisor concurrently checks the pinned
container, HCU devices, train-only data, fresh base, and source SHA contracts on
all nodes. It then runs one real 4-by-8 `torchrun` collective preflight: all 32
ranks must publish the expected node/local-rank map, produce a finite
`all_reduce` sum, and pass a final barrier. The formal train phase is not started
if any local or collective preflight fails.

Each `RUN_ID` owns one no-clobber cluster directory:

```text
PROJECT_ROOT/logs/supervisor/RUN_ID/
├── request.json
├── launch.json
├── state.json
├── launcher.log
├── launcher.pid
├── worker.pid
├── preflight-node-00-ADDRESS.log ... preflight-node-03-ADDRESS.log
└── train-node-00-ADDRESS.log ... train-node-03-ADDRESS.log
```

`request.json` binds the exact topology, base checkpoint, and training-semantic
source SHA256 values. `launch.json` records the immutable launch and supervisor
identity. `state.json` is the current cluster-level phase, final exit code, and
per-node return-code authority; a running state is not a completed checkpoint.
Reusing a `RUN_ID`, log directory, checkpoint directory, per-node log, or
container name is rejected instead of overwritten. The supervisor survives the
initiating SSH disconnect, while a login-node reboot still requires an explicit
recovery decision.

The HCU formal path preserves effective global batch 64 in both topologies. A
single node uses eight ranks times microbatch one times eight accumulation
microsteps; four nodes use 32 ranks times microbatch one times two accumulation
microsteps. All non-boundary microsteps use DDP `no_sync`; clipping, AdamW
update, LR advancement, and checkpointing occur once per effective batch. A
source-SHA-bound runtime patch also releases an already-detached prefix graph
that the pinned upstream policy otherwise retains until the end of forward.
Neither patch changes the official checkout or enables validation-data access.

Mixed8 additionally installs a source-bound sampler. With eight ranks, each
physical global microbatch contains one sample from each task; with 32 ranks, it
contains four samples from each task. Eight single-node microsteps or two
four-node microsteps therefore both yield eight samples per task in every
optimizer update. The unified quantile normalizer consumes all 144,484 train
frames using a divisor batch size; it is never averaged from task-local
statistics.
