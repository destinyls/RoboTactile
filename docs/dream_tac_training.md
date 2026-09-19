# Dream-Tac UniVTAC retraining

This document defines the RoboTactile route for training a new
`Dream-Tac-UniVTAC` model when no official, task-aligned Dream-Tac checkpoint
is available. It does not describe a reproduction of the paper's released
Franka weights: the public Dream-Tac repository supplies code, but its
documented channels did not expose a downloadable checkpoint as of
2026-08-30.

The route is split into five evidence-gated phases. P0 and P1 are CPU data
preparation and contract validation. P2 and P3 require the external pinned
Dream-Tac/Cosmos training runtime and accelerator resources. P4 additionally requires
the qualified UniVTAC/Isaac Sim runtime. Finishing an earlier phase never
establishes a later evidence level.

## NVIDIA deployment prerequisite

The formal P2/P3 launcher supports one Linux NVIDIA CUDA node with one to eight
local GPUs. It must not be routed through the separately pinned N0-TWAM HPU
launcher. An experimental, source-bound HCU route is documented in
[`scripts/dream_tac/hcu_port/README.md`](../scripts/dream_tac/hcu_port/README.md).
It now includes a strict flat-PT-to-DCP converter and a launcher for exactly one
single-HCU optimizer step. The implementation does not establish that the step
has been executed, and one completed step would still not establish HCU P2/P3
training, throughput, convergence, or model quality. The exact formal NVIDIA
runtime, Cosmos base, T5 cache, request schema, preflight, no-clobber receipt,
and resume instructions are in
[Dream-Tac NVIDIA training deployment](models/dream_tac/nvidia_training.md).

## Frozen dataset contract

The source is an exact `8 x 100` UniVTAC HDF5 grid. RoboTactile freezes its
membership before creating any Dream-Tac files:

| Split | Membership | Count | Permitted use |
|---|---|---:|---|
| `train759` | Every episode except the rows below | 759 | Training, train-only statistics, and training-time development |
| `frozen40` | Episode IDs `0, 1, 2, 3, 5` in every task | 40 | One held-out evaluation after the training/checkpoint rule is frozen |
| `quarantine1` | `grasp_classify/clean/90.hdf5` | 1 | No training, tuning, statistics, or evaluation |

The eight tasks are `grasp_classify`, `insert_HDMI`, `insert_hole`,
`insert_tube`, `lift_bottle`, `lift_can`, `pull_out_key`, and
`put_bottle_in_shelf`. Every source file is content-addressed in
`source_split_manifest.json`. The materializer emits training files only for
the 759 training episodes; `frozen40` and `quarantine1` remain identities in
the source manifest and are never copied into the training tree.

Do not mount `frozen40` inside the formal training container. In particular,
do not use it for early stopping, prompt selection, normalization, checkpoint
selection, or debugging examples.

## Temporal and modality contract

The conversion preserves the source cadence exactly:

- source and target cadence: `10 Hz`, with no frame duplication or resampling;
- observation row: source row `t`;
- supervision row: source end-effector pose and continuous
  `embodiment/joint[t + 1, 7]` gripper state;
- converted episode length: `T - 1` causal samples;
- action dimension: seven absolute values `[xyz, rpy_xyz, gripper_qpos]`;
- source WXYZ quaternion signs are made episode-continuous before conversion,
  then each XYZ Euler axis is unwrapped across the episode;
- action horizon: 20 steps, hence exactly two seconds at 10 Hz;
- all training samples include front RGB, wrist RGB, left tactile RGB, right
  tactile RGB, and proprioception;
- missing or malformed tactile is an error. It is never replaced by zeros and
  there is no tactile-dropout schedule.

The continuous gripper scalar is not a binary open/close label. A model trained
with this contract must be configured with `direct_qpos_v1`; the adapter then
copies the finite predicted scalar to EE8 without thresholding or clipping.
The manifest can additionally bind the train-derived minimum and maximum, and
inference fails closed when a prediction leaves that range. The legacy
threshold mappings remain available only for independently calibrated upstream
Franka bundles.

## P0: materialize `train759` on CPU

Run P0/P1 from the isolated runtime or container created for the pinned
external Dream-Tac source. Its Franka dependency group supplies `h5py` and
OpenCV for real HDF5/MP4 conversion. Mount this RoboTactile checkout read-only,
mount the raw dataset read-only, and give the process a dedicated output root.
Do not install those dependencies into RoboTactile's dependency-light core
environment, Isaac Sim, or system Python. The main wheel intentionally remains
NumPy-only and does not promise real HDF5/MP4 conversion by itself. In the
commands below, `python` means the activated external Dream-Tac runtime. The
input root must contain the complete eight-task source grid described above.

First perform a read-only plan:

```bash
python -m scripts.dream_tac.training.materialize_train759 \
  --raw-root /absolute/path/to/UniVTAC-HDF5 \
  --output-root /absolute/path/to/dream-tac-univtac \
  --dry-run
```

Then materialize into a dedicated, non-clobber output root:

```bash
python -m scripts.dream_tac.training.materialize_train759 \
  --raw-root /absolute/path/to/UniVTAC-HDF5 \
  --output-root /absolute/path/to/dream-tac-univtac
```

The public entry point emits:

```text
dream-tac-univtac/
├── source_split_manifest.json
├── dataset/
│   ├── train/<task>/episode_<id>/
│   │   ├── episode_<id>.hdf5
│   │   ├── episode_<id>_cam_front.mp4
│   │   ├── episode_<id>_cam_high.mp4
│   │   ├── episode_<id>_tactile_rectify_left.mp4
│   │   ├── episode_<id>_tactile_rectify_right.mp4
│   │   └── _robotactile_dream_tac_episode.json
│   └── dataset_statistics_franka.json
├── prompt_manifest.json
├── t5_cache_request.json
├── dream_tac_train759_receipt.json
└── t5_cache_receipt.json             # generated only by the later T5 step
```

Each materialized episode contains the Dream-Tac HDF5 control/state streams
and four matching MP4 streams. Episode receipts and the dataset receipt bind
the source file, causal row count, 10 Hz cadence, 20-step horizon, four required
modalities, and generated-file hashes. Existing foreign output is rejected;
rerunning must not silently overwrite a prior dataset.

Only the upstream-discoverable `dataset/train/` tree is created; there is no
`dataset/val/` materialization that could expose `frozen40` to training code.
For bounded/resumable conversion, add `--tasks <task> [<task> ...]`; a subset
returns partial status, and the global complete receipt is published only
after all 759 training episodes validate.

`t5_cache_request.json` is a content-addressed request for the eight canonical
task prompts. P0 deliberately does not fabricate `t5_embeddings.pkl`; generate
that cache with the pinned upstream text encoder before training and retain its
identity alongside the checkpoint. Consequently, even a complete P0 receipt
records `training_ready=false` and `t5_cache_status=external_generation_required`.

## P1: verify and inspect before GPU use

Verify the materialized tree without changing it:

```bash
python -m scripts.dream_tac.training.materialize_train759 \
  --raw-root /absolute/path/to/UniVTAC-HDF5 \
  --output-root /absolute/path/to/dream-tac-univtac \
  --verify
```

The verification must pass before a GPU job is submitted. It checks exact
source split identity, generated hashes, train-only membership, causal lengths,
10 Hz timing, four-stream presence, and statistics provenance. In addition,
perform these visual/semantic checks on representative episodes:

1. Compare the source HDF5 front, wrist, left tactile, and right tactile pixels
   with the decoded materialized streams.
2. Confirm that sample `t` uses observation/state at `t` and action target at
   `t + 1`, including the continuous gripper coordinate.
3. Inspect quaternion-to-RPY conversion around wrap boundaries rather than
   treating a numerically discontinuous angle as physical motion.
4. Load one batch through the pinned Dream-Tac dataset class and confirm two
   visual streams, two non-empty tactile streams, proprioception, and a
   `(20, 7)` target chunk.
5. Replay a source expert action sequence in the qualified UniVTAC scene and
   verify physical object motion before diagnosing any learned policy.

P0/P1 completion is `CODE` evidence only. It is not model inference, offline
accuracy, simulator success, or paper reproduction.

### Generate the real T5 cache before P2

After P0/P1 passes, run the pinned upstream text encoder in the external
Dream-Tac runtime. The command consumes the exact prompt and cache request
identities; it does not invent substitute embeddings:

The pinned encoder calls `google-t5/t5-11b` with `local_files_only=True` and
places the model on CUDA. Therefore this step is not part of the CPU-only P0
materialization: the exact T5 tokenizer/model snapshot must already exist in
the runtime's Hugging Face cache, and the selected NVIDIA GPU must have enough
memory to load it. An air-gapped host without that cache must fail before P2;
do not replace the encoder, enable an implicit network download, or use a
different T5 size while retaining the same experiment label.

```bash
python -m scripts.dream_tac.training.t5_cache \
  --dream-tac-root /absolute/path/to/pinned/Dream-Tac \
  --output-root /absolute/path/to/dream-tac-univtac

python -m scripts.dream_tac.training.t5_cache \
  --output-root /absolute/path/to/dream-tac-univtac \
  --verify
```

Keep `dataset/t5_embeddings.pkl` and its verification receipt bound to the
prompt manifest and source commit. A generated cache is still a preprocessing
artifact, not an `OFFLINE` model result. The current receipt binds the produced
cache bytes and prompt/upstream identities; separately retain the exact
`google-t5/t5-11b` snapshot/license identity because a content hash alone does
not establish where those weights came from.

## Experimental HCU single-step gate

The HCU route is deliberately narrower than P2. It exists to answer one
question before any HCU-scale training is planned: can the exact Dream-Tac
model, train759 batch, optimizer, scheduler, and checkpoint stack complete one
source-bound optimizer step on one HCU?

Before that attempt, place the legally obtained Cosmos base and matching
`tokenizer/tokenizer.pth` on the shared filesystem, and generate the public
T5-11B cache on a machine that already has the exact local snapshot. The HCU
runtime is offline and must never attempt an implicit download. Convert the
flat Cosmos base with
`scripts.dream_tac.hcu_port.base_checkpoint_converter`; its generated
`.../iter_000000000` **iteration root**, not the original `.pt` and not the
nested `model/` directory, is the HCU `checkpoint.load_path`.

Then use the strict request and gate sequence:

```bash
cp configs/experiments/dream_tac/hcu_optimizer_step.example.json \
  /absolute/requests/dream-tac-hcu-one-step.json

python -m scripts.dream_tac.hcu_port.launch_optimizer_step \
  --request /absolute/requests/dream-tac-hcu-one-step.json \
  --print-command
python -m scripts.dream_tac.hcu_port.launch_optimizer_step \
  --request /absolute/requests/dream-tac-hcu-one-step.json \
  --preflight-only
python -m scripts.dream_tac.hcu_port.launch_optimizer_step \
  --request /absolute/requests/dream-tac-hcu-one-step.json
```

The first command is ungated and write-free; the second checks artifacts and
the exact one-HCU HIP runtime without launching; only the third may execute the
single step. The request fixes `max_iter=1`, `save_iter=1`, `batch_size=1`,
`nproc_per_node=1`, and no resume. Consult the HCU README for the exact
conversion command, artifact gates, receipts, and success predicate. Do not
promote an overlay, config dry-run, CASA random-tensor backward, converter
receipt, preflight, or launch plan into optimizer-step evidence.

## P2/P3 launcher

The formal NVIDIA launcher remains unchanged. Use
`python -m scripts.dream_tac.training.launch_training` with a request
created from the strict P2/P3 examples. Progress through `--print-command`,
`--preflight-only`, and `--dry-run` before omitting the mode flag to start one
single-node `torchrun`. Exact request fields, mode boundaries, receipt paths,
resume rules, and the remaining paper-grade provenance limits are documented
in [Dream-Tac NVIDIA training deployment](models/dream_tac/nvidia_training.md).

## P2: bounded micro-train

Initialize from the compatible Cosmos-Predict2 2B base checkpoint, not random
weights and not an unverified file presented as an official Dream-Tac release.
The current v1 launcher intentionally has no task-filter field: P2 is a bounded
run over the same eight-task train759 mixture as P3, not a single-task result.
Use it to run, in order:

1. configuration-only load;
2. one-batch forward/backward;
3. a 100-step throughput and memory probe;
4. a 500--1,000-step overfit/micro-train;
5. checkpoint save, reload, and resume;
6. a separate normal-versus-tactile-null prediction-drift diagnostic on frozen
   training examples.

Do not start P3 unless the trainable tactile/CASA path receives gradients,
losses remain finite, action loss decreases, tactile removal changes the
prediction, and the resulting server emits finite `(20, 7)` chunks. A failure
here is first a data/action/timing-contract problem, not justification for a
longer run.

The upstream 2B training stack is an external GPU workload. Its published
example uses eight local processes, but that is not a guarantee that any named
GPU/topology will fit this UniVTAC configuration. Size the P3 topology from the
measured P2 peak allocated/reserved memory and throughput. A 24 GB
A5000-class card remains suitable for P0/P1 and selected probes, not a presumed
fit for the unmodified full 2B fine-tune. LoRA or a frozen-backbone route is a
separate resource-constrained ablation and must be labelled as such.

## P3: eight-task training

After P2 passes, train one language-conditioned eight-task model:

- sample the 94 `grasp_classify` and 95 episodes from each remaining task from
  `train759` only;
- keep Vision + both tactile streams + proprioception active for every sample;
- use statistics and T5 prompt caches derived exclusively from `train759`;
- bind source hashes, base checkpoint, upstream commits, complete config,
  random seeds, environment, and every output checkpoint;
- select the training rule using a train-only development partition;
- freeze the rule before the single `frozen40` evaluation.

Resume must bind the exact checkpoint file and SHA256, and a same-job latest
marker must name that same file. Never delete a plan/result receipt or overwrite
a log to force a retry; use the content-addressed resume procedure in the
[NVIDIA deployment guide](models/dream_tac/nvidia_training.md#receipts-and-resume).

The resulting model is `Dream-Tac-UniVTAC`, not an official Dream-Tac
checkpoint. Offline inference on `frozen40` is `OFFLINE` evidence only and does
not establish simulator task success.

After training, create one source-bound task artifact using the exact prompt
and `actions_min[6]` / `actions_max[6]` from the train-only statistics
artifact:

```bash
robotactile integrations configure dream-tac \
  --bundle-root "$DREAM_TAC_BUNDLE" \
  --checkpoint-root "$DREAM_TAC_BUNDLE/checkpoint" \
  --dataset-stats "$DREAM_TAC_BUNDLE/dataset/dataset_statistics_franka.json" \
  --t5-embeddings "$DREAM_TAC_BUNDLE/dataset/t5_embeddings.pkl" \
  --task "$TASK_ID" \
  --instruction "$EXACT_TRAINING_PROMPT" \
  --experiment-config "$EXACT_EXPERIMENT_CONFIG" \
  --control-hz 10 \
  --gripper-mapping direct_qpos_v1 \
  --gripper-qpos-min "$TRAIN_GRIPPER_QPOS_MIN" \
  --gripper-qpos-max "$TRAIN_GRIPPER_QPOS_MAX"
```

`gripper_threshold` remains a backwards-compatible manifest field but is not
applied by `direct_qpos_v1`.

## P4: history-aware serving and closed loop

The pinned upstream HTTP endpoint does not establish paper CASA-parity. Before
closed-loop promotion, add a source-bound session protocol that carries
`session_id`, monotonic `cycle_index`, and delivered tactile history. Reset
must clear history, and Faulted evaluation must not read a hidden clean tactile
stream. Bind the server to:

- the exact `10 Hz` control cadence;
- a full 20-step chunk followed by re-inference;
- `direct_qpos_v1` and train-derived gripper bounds;
- the exact camera, color, tactile, proprioception, and task-prompt contracts;
- the existing RoboTactile Clean/Faulted injector and task success predicate.

Promotion order is one Clean smoke episode, eight tasks with one diagnostic
episode each, the declared multi-seed Clean protocol, and finally the matched
Clean/Faulted robustness matrix. Only valid Isaac traces with completed task
predicates are `CLOSED-LOOP` evidence. They remain results for the retrained
`Dream-Tac-UniVTAC` artifact, not `OFFICIAL` reproduction of Dream-Tac's real
Franka experiments.

## Evidence boundary

| Claim | Minimum required evidence |
|---|---|
| `CODE` | P0/P1 conversion and contract tests pass |
| `OFFLINE` | A content-addressed trained model executes on the frozen evaluation inputs |
| `CLOSED-LOOP` | Qualified UniVTAC/Isaac episodes produce valid terminal artifacts and success predicates |
| `OFFICIAL` | Upstream-published task-aligned weights and protocol parity; currently unavailable |

Never turn a source checkout, successful conversion, decreasing training loss,
mock HTTP response, or expert replay into a Success Rate claim.
