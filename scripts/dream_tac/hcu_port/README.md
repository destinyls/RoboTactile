# Dream-Tac HCU compatibility and single-step route

This directory contains a bounded source overlay, compatibility probes, a
flat-PT-to-DCP converter, and a fail-closed launcher for exactly one optimizer
step on one Hygon HCU. The formal NVIDIA P2/P3 launcher remains unchanged at
`scripts/dream_tac/training/launch_training.py`; this HCU route must not be
used as an implicit replacement for a full training run.

## Apply the isolated source overlay

The overlay is bound to upstream commit
`14bab51d6862fd07124745c55cd395ea5caa9fd3`. Keep one clean pinned checkout and
create a separate target checkout without hardlinks. The tool refuses the
pinned source itself, a dirty checkout, a wrong commit, an unexpected source
digest, a shared source/target inode, or an existing receipt.

```bash
git clone https://github.com/LYFCLOUDFAN/Dream-Tac.git /absolute/source/Dream-Tac-pinned
git -C /absolute/source/Dream-Tac-pinned checkout \
  14bab51d6862fd07124745c55cd395ea5caa9fd3
git clone --no-hardlinks /absolute/source/Dream-Tac-pinned \
  /absolute/source/Dream-Tac-HCU-port-v5

python -m scripts.dream_tac.hcu_port.overlay \
  --pinned-source-checkout /absolute/source/Dream-Tac-pinned \
  --target-checkout /absolute/source/Dream-Tac-HCU-port-v5 \
  --receipt /absolute/receipts/dream-tac-hcu-overlay-v5.json
```

The v5 overlay currently applies seven digest-bound compatibility changes:

1. `minimal_v4_dit.py` tries the Transformer Engine legacy
   `attention.apply_rotary_pos_emb` API first and falls back to
   `attention.rope.apply_rotary_pos_emb` on `ImportError`.
2. `cosmos_policy_experiment_configs.py` returns `None` only when
   `ROBOTACTILE_HCU_CONFIG_PROBE_ONLY=1`. A real launch must provide
   `DREAM_TAC_BASE_CHECKPOINT`, which is passed through upstream
   `get_checkpoint_path` validation.
3. The upstream `CheckpointConfig.load_path` annotation accepts `str | None`,
   allowing that explicit probe-only `None` without weakening
   the real-training environment contract.
4. When and only when `ROBOTACTILE_HCU_TRAINING=1`, upstream distributed setup
   skips NVIDIA NVML affinity and `libcudart` L2 tuning while preserving the
   normal `torch.cuda` compatibility API and NCCL/RCCL initialization.
5. On HIP with the legacy PyTorch SDPA API, generic attention selects the
   probe-validated Flash fallback instead of unavailable efficient attention.
   NVIDIA SM routing and the configured model attention backend are unchanged.
6. On HIP only, the four model RMSNorm sites use Dream-Tac's source-native,
   FP32-accumulation implementation because the packaged Transformer Engine
   RMSNorm returns incorrect forward values and fails backward. NVIDIA keeps
   the original Transformer Engine RMSNorm path.
7. The iteration logger omits intentional non-finite sentinel metrics for
   sample subsets absent from a batch. This changes console/W&B reporting only:
   model outputs, aggregate loss, backward, and optimizer behavior are unchanged.

The source manifest, preimage/output SHA256 values, pure-Python wheel pins, and
their hashes are embedded in the signed, no-clobber receipt. The source remains
an external pinned integration; the overlay does not copy Dream-Tac into the
RoboTactile wheel.

## Install the isolated pure-Python additions

The HCU image supplies PyTorch/DTK and compiled backends. Install only the four
missing pure-Python wheels into an isolated target directory. No dependency is
vendored here.

```bash
python -m pip install \
  --target /absolute/runtime/hcu-port/site \
  --no-deps \
  --require-hashes \
  -r scripts/dream_tac/hcu_port/pure_python_requirements.txt
```

The locked wheels are `ftfy==6.3.1`, `wcwidth==0.2.13`,
`webdataset==0.2.111`, and `braceexpand==0.1.7`. Add the isolated directory to
`PYTHONPATH`; do not install these packages into system Python.

## Run the compatibility probe

Run it inside the candidate DTK/HIP runtime:

```bash
python -m scripts.dream_tac.hcu_port.probe \
  --output /absolute/receipts/dream-tac-hcu-probe.json
```

The default required imports are `cosmos_policy`, `flash_attn`,
`transformer_engine`, `natten`, `xformers`, `h5py`, `cv2`, and `transformers`.
Use repeated `--required-module` options to replace that set. If an exact CASA
backend module is known, test it explicitly:

```bash
python -m scripts.dream_tac.hcu_port.probe \
  --output /absolute/receipts/dream-tac-hcu-casa-probe.json \
  --casa-module package.path.to.casa_backend
```

The receipt records separate runtime, imports, BF16 forward/backward, and
optional CASA phases. An identical existing receipt is verified; a different
receipt is never overwritten. Even a passing receipt is labelled
`compatibility_probe_only_not_training_success`.

## Run the CASA BF16 correctness micro

After the source overlay and import probe pass, run one small, deterministic
CASA forward/backward on a single HCU:

```bash
PYTHONPATH=/absolute/runtime/hcu-port/site:/absolute/source/Dream-Tac-HCU-port-v2 \
python -m scripts.dream_tac.hcu_port.casa_micro \
  --dream-tac-checkout /absolute/source/Dream-Tac-HCU-port-v2 \
  --device-index 0 \
  --output /absolute/receipts/dream-tac-hcu-casa-bf16-micro-v1.json
```

The micro calls the pinned
`self_attention_with_tactile_outer_bias_chunked` implementation with seed 0,
BF16 `q/k/v` tensors shaped `[2, 64, 4, 16]`, and requested backend
`flashbias_sdpa`. It requires a finite, nonzero output and loss, then checks
present, finite, nonzero gradients for `q`, `k`, `v`, `a`, `b`, `gamma`, and
the projection weight and bias. Runtime warnings—including an unavailable
memory-efficient backend—are recorded but do not fail a correctness pass when
all numerical checks pass.

The function source file and upstream commit are SHA256-bound. A checkout may
contain the six expected HCU overlay edits elsewhere; any dirty entries are
recorded. The receipt boundary is
`casa_random_tensor_forward_backward_only_not_training`: it does not prove a
training optimizer step, fused-kernel selection, performance parity, model
quality, or closed-loop behavior.

## Required offline artifacts

Treat the HCU host as air-gapped. Prepare these artifacts on an authorized
networked machine, record their SHA256 values, and transfer them to immutable
paths before running preflight:

- the authorized Cosmos-Predict2 2B flat base
  `model-480p-16fps.pt`;
- the matching authorized tokenizer checkpoint
  `tokenizer/tokenizer.pth`;
- the public `google-t5/t5-11b` snapshot in the encoding runtime cache and the
  generated train759-bound `dataset/t5_embeddings.pkl` plus
  `t5_cache_receipt.json`;
- the complete train759 materialization and source manifests;
- passing v5 overlay and CASA micro receipts from the exact HCU checkout and
  selected device.

These tools never download weights, accept licenses, or synthesize a missing
T5 cache. Do not place access tokens in requests, commands, receipts, or Git.

## Convert the flat Cosmos base to DCP

The HCU training checkpointer expects a DCP iteration tree. Retain the flat
`model-480p-16fps.pt` as provenance, but **do not** pass it directly to HCU
`checkpoint.load_path`. Convert it once in the HCU runtime:

```bash
export DREAM_TAC_HCU_ROOT=/absolute/source/Dream-Tac-HCU-port-v2
export COSMOS_ARTIFACT_ROOT=/absolute/artifacts/Cosmos-Predict2-2B-Video2World
export COSMOS_FLAT_PT="$COSMOS_ARTIFACT_ROOT/model-480p-16fps.pt"
export COSMOS_TOKENIZER_PT="$COSMOS_ARTIFACT_ROOT/tokenizer/tokenizer.pth"
export COSMOS_DCP_OUTPUT="${COSMOS_ARTIFACT_ROOT}-dcp"

python -m scripts.dream_tac.hcu_port.base_checkpoint_converter \
  --dream-tac-checkout "$DREAM_TAC_HCU_ROOT" \
  --experiment-config \
    "$DREAM_TAC_HCU_ROOT/cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py" \
  --experiment-config-sha256 "$(sha256sum "$DREAM_TAC_HCU_ROOT/cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py" | awk '{print $1}')" \
  --input-checkpoint "$COSMOS_FLAT_PT" \
  --input-checkpoint-sha256 "$(sha256sum "$COSMOS_FLAT_PT" | awk '{print $1}')" \
  --tokenizer-checkpoint "$COSMOS_TOKENIZER_PT" \
  --tokenizer-checkpoint-sha256 "$(sha256sum "$COSMOS_TOKENIZER_PT" | awk '{print $1}')" \
  --output-root "$COSMOS_DCP_OUTPUT"
```

The converter rejects unknown or implicitly missing tensors, loads through
the pinned upstream model loader, performs a DCP round trip, and publishes
without overwriting. The tokenizer checkpoint is content-bound and injected
into the upstream config before model instantiation; `load_mean_std` must stay
disabled. The only source-only exception is the exact 28-key TransformerEngine
FP8 cross-attention `_extra_state` set that upstream itself skips in non-strict
cross-backend loading; the converter records that set in its signed receipt and
still rejects partial or unrelated unexpected keys. The converter always serializes DCP with
`ModelWrapper(model, load_ema_to_reg=False)`, matching the training
checkpointer's `net.*` key contract; the source config's
`load_ema_to_reg` value is recorded separately and cannot rename DCP keys.
The output root also contains `converter_manifest.json`, with size and SHA256
for every file under `iter_000000000/model`, and the signed
`converter_receipt.json`. Bind the generated artifacts as follows:

```text
base_checkpoint.path     = .../model-480p-16fps.pt
base_dcp_root            = .../Cosmos-Predict2-2B-Video2World-dcp/iter_000000000
base_dcp_receipt.path    = .../Cosmos-Predict2-2B-Video2World-dcp/converter_receipt.json
checkpoint.load_path     = base_dcp_root
```

Conversion proves checkpoint-format compatibility only. It performs no
optimizer step and establishes no model-quality result. Do not use
`--allow-missing-key` unless an exact, reviewed upstream key exception has
been documented; the default is strict equality.

## Run exactly one HCU optimizer step

Copy `configs/experiments/dream_tac/hcu_optimizer_step.example.json` into a
writable request directory, replace every placeholder path and SHA256, and
keep the fixed values `nproc_per_node=1`, `max_iter=1`, `save_iter=1`,
`batch_size=1`, and `resume_checkpoint=null`.

The HCU vendor runtime is activated by the content-bound
`hcu_environment_script`. The example uses `/opt/hyhal/env.sh`; record its real
SHA256 from the target runtime rather than copying the placeholder digest. Both
the environment probe and the optimizer process use the same noninteractive
`bash --noprofile --norc` wrapper: it sources that path from a positional
argument and then `exec`s the payload. Request paths and commands are never
interpolated into shell source text.

For this dense Dream-Tac experiment, the HCU launcher treats NVIDIA/CUDA
`natten` and `xformers` wheels as optional and records their availability in
the environment receipt. They are not used by the selected
`flashbias_sdpa` tactile-attention path. The CUDA compatibility probe above
keeps its broader default import inventory for NVIDIA installations.

```bash
python -m scripts.dream_tac.hcu_port.launch_optimizer_step \
  --request /absolute/requests/dream-tac-hcu-one-step.json \
  --print-command

python -m scripts.dream_tac.hcu_port.launch_optimizer_step \
  --request /absolute/requests/dream-tac-hcu-one-step.json \
  --preflight-only

python -m scripts.dream_tac.hcu_port.launch_optimizer_step \
  --request /absolute/requests/dream-tac-hcu-one-step.json
```

`--print-command` is write-free and explicitly ungated. `--preflight-only`
checks the authorized flat base and tokenizer hashes, converted DCP tree and
round-trip receipt, train759/T5 identities, exact six-file overlay, CASA
gradient receipt, the real non-symlink/nonempty HCU environment script and its
SHA256, Python 3.10 HIP runtime, one visible HCU, and RCCL through the PyTorch
NCCL API. It does not launch training. There is intentionally no HCU
`--dry-run` mode.

The final command launches once and refuses ambiguous replay. A successful
result requires process return code zero, all required upstream load/train
completion log markers, no `NaN`/`Inf`, no `Training from scratch.`, a
`latest_checkpoint.txt` naming `iter_000000001`, and nonempty DCP `model`,
`optim`, `scheduler`, and `trainer` components. Receipts are written under:

```text
<output_root>/_robotactile_receipts/<run_name>/<request_sha256>/
├── optimizer_step_preflight_receipt.json
├── optimizer_step_environment_receipt.json
├── optimizer_step_resolved_experiment.json
├── optimizer_step_launch_plan.json
├── optimizer_step.log
└── optimizer_step_launch_result.json
```

Only a result with `optimizer_step_completed=true` proves one real optimizer
step. It still does not prove throughput, multi-HCU support, convergence,
checkpoint quality, offline accuracy, or closed-loop success. Intentional NaN
sentinels for inapplicable sample subsets are filtered by the overlay before
logging; an actual non-finite aggregate or applicable component remains a
hard failure. The result validator uses identifier-aware token boundaries, so
runtime names such as `TORCH_NCCL_NAN_CHECK` are not confused with numeric
`NaN`, while standalone `nan`, `inf`, and `infinity` remain hard failures.

## Two-node train759 training

After the source-bound one-step v11 result passes, create a request from
`configs/experiments/dream_tac/hcu_distributed_2node.example.json`. This route
keeps the one-step API unchanged and fixes the distributed topology to two
nodes, eight HCU processes per node, and an HSDP mesh with shard size 8
(node-local shard, cross-node replicate). Both ranks use the complete eight-task
`train759` mixture with vision, two tactile streams, and proprio always enabled.

`distributed-smoke` accepts 1--20 iterations. `formal` requires more than 1000
iterations; the example uses 20,000. Batch size is per rank, and receipts record
`effective_global_batch = 16 * batch_size * grad_accum_iter`.

```bash
python -m scripts.dream_tac.hcu_port.launch_distributed \
  --request /absolute/requests/dream-tac-hcu-2node.json \
  --print-plan

python -m scripts.dream_tac.hcu_port.launch_distributed \
  --request /absolute/requests/dream-tac-hcu-2node.json
```

The launcher verifies the exact completed optimizer-step request/result pair,
then starts the two SSH node ranks concurrently. It intentionally loads the
converted base DCP with a fresh optimizer; the single-rank v11 optimizer state
is evidence, not a 16-rank resume checkpoint. The container image ID, RCCL
plugin file/hash, source files, request, plan, status snapshots, per-node logs,
and no-clobber container names are recorded. The runtime removes only the
transient container it created after exit and never touches a pre-existing
container. A run name/request hash may execute only once; use a new request for
an explicit successor.

The v2 request distinguishes container paths from their shared-host mount
sources. `dream_tac_host_root` is mounted read-only at the source request's
`dream_tac_root`, while `runtime_host_root` is mounted read-only at `/probe`.
Both host paths must be below `/mnt/data` and are available identically on the
two selected nodes.
