# Dream-Tac NVIDIA training deployment

This document is the machine-facing deployment contract for RoboTactile's
Dream-Tac UniVTAC P2/P3 launcher. The dataset, temporal/action, training-phase,
evaluation, and claim contracts remain in
[Dream-Tac UniVTAC retraining](../../dream_tac_training.md).

## Runtime and accelerator boundary

Dream-Tac training is an NVIDIA CUDA workload. The pinned upstream environment
declares Linux, Python 3.10, CUDA-enabled PyTorch, and a `cu128` Franka group;
its Dockerfile is based on CUDA 12.8.1 with cuDNN. A compatible host must expose
its NVIDIA GPU through `nvidia-smi` and, when using the recommended container
route, through NVIDIA Container Toolkit. Host compatibility must be proved by
loading CUDA PyTorch and allocating one tensor before a training process is
submitted; the presence of a GPU name in an inventory file is not sufficient.

Hygon HCU/HPU is **not** a supported backend of this formal NVIDIA launcher. In
particular, do not launch Dream-Tac through `scripts/n0_twam/hpu_training/`:
that route is a separately pinned N0-TWAM two-node contract and is neither a
CUDA nor a Cosmos-Predict2 runtime. The bounded experimental HCU route lives in
[`scripts/dream_tac/hcu_port/README.md`](../../../scripts/dream_tac/hcu_port/README.md).
It includes an isolated four-file overlay, compatibility/CASA probes, a strict
flat-PT-to-DCP converter, and a launcher for exactly one single-HCU optimizer
step. None of those additions changes this NVIDIA launcher or its request
schema. Source/probe/preflight evidence is not an optimizer-step result, and a
single completed step would not establish throughput, convergence, full HCU
training, checkpoint quality, or closed-loop success.

Keep all Dream-Tac dependencies inside the pinned external checkout's local
environment or container. Do not install or upgrade them in system Python,
Isaac Sim, or either N0-TWAM runtime. A normal deployment keeps these identities
separate:

```text
deployment/sources/Dream-Tac/              # pinned source, commit 14bab51...
deployment/runtime/dream-tac/              # external CUDA environment/cache
<shared-data>/dream-tac-univtac/           # P0 dataset and T5 cache
<shared-runs>/dream-tac/<run-id>/           # one no-clobber training run
deployment/artifacts/models/dream_tac/     # promoted checkpoint only
```

The upstream Docker image is an environment base, not a checkpoint-bearing
training appliance. Building it does not download the Cosmos base checkpoint,
the T5 weights, the UniVTAC data, or a Dream-Tac task checkpoint. Consult the
[pinned upstream setup](https://github.com/LYFCLOUDFAN/Dream-Tac/tree/14bab51d6862fd07124745c55cd395ea5caa9fd3#1-environment-setup)
and retain the exact image identity used by the run.

## Install the isolated source and runtime

Initialize the repository-contained deployment and install only the pinned
Dream-Tac checkout:

```bash
cd /absolute/RoboTactile
robotactile deployment init
bash integrations/install_dream_tac.sh

export ROBOTACTILE_ROOT="$PWD"
export DREAM_TAC_ROOT="$ROBOTACTILE_ROOT/deployment/sources/Dream-Tac"
export DREAM_TAC_PYTHON="$ROBOTACTILE_ROOT/deployment/runtime/dream-tac/bin/python"
test "$(git -C "$DREAM_TAC_ROOT" rev-parse HEAD)" = \
  14bab51d6862fd07124745c55cd395ea5caa9fd3
```

The repository-contained route uses a Python 3.10 virtual environment outside
the source checkout and a non-editable PEP 517 build/install. It does not alter
system Python, Isaac Sim, N0-TWAM, or the pinned Dream-Tac tree:

```bash
python3.10 -m venv "$ROBOTACTILE_ROOT/deployment/runtime/dream-tac"
"$DREAM_TAC_PYTHON" -m pip install --upgrade pip
"$DREAM_TAC_PYTHON" -m pip install \
  "$DREAM_TAC_ROOT/cosmos_policy[cu128]"
"$DREAM_TAC_PYTHON" -m pip freeze --all > \
  "$ROBOTACTILE_ROOT/deployment/runtime/dream-tac/pip-freeze.txt"
test -z "$(git -C "$DREAM_TAC_ROOT" status --porcelain=v1 --untracked-files=all)"
```

Run this inside the recommended CUDA container or another compatible NVIDIA
host runtime. `python_executable` in every training request must be the
resulting absolute `$DREAM_TAC_PYTHON`. Archive `pip-freeze.txt` with the run
environment. This pip-based compatibility path is documented but has not yet
been validated on a physical NVIDIA host; only a passing source/artifact/CUDA
preflight establishes local launch readiness.

The pinned upstream README instead recommends
`uv sync --extra cu128 --group franka`, but the pinned commit contains no
`uv.lock`. Running that command in the formal source checkout can create an
untracked lock and fail the clean-tree gate. It is an upstream reference, not
the RoboTactile source-bound main path; if investigated, use a separate build
checkout and record the generated environment identity.

Before materialization or T5 generation, mount the RoboTactile checkout into
that environment so `scripts.dream_tac.training` is importable. Before formal
training, `dream_tac_root` must point to the untouched pinned source checkout.
Preflight rejects tracked or untracked changes.

## Required external assets

Do not allocate a formal GPU run until every row below has a local immutable
identity. None of these large assets is included in the RoboTactile wheel.

| Asset | Required identity and boundary |
|---|---|
| Dream-Tac source | Exact commit `14bab51d6862fd07124745c55cd395ea5caa9fd3`; install it under `deployment/sources/Dream-Tac` and do not edit its shared config in place |
| CUDA runtime | The pinned upstream Docker/runtime plus the resolved Python, PyTorch, CUDA, GPU, and container-image identity |
| Cosmos base | A legitimately obtained `Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt`; bind its absolute path, byte size, and SHA256 instead of leaving the upstream `/path/to/...` placeholder |
| UniVTAC training data | Complete `dream_tac_train759_receipt.json`; training may see only `dataset/train/`, never `frozen40` or `quarantine1` |
| Text conditioning | Real `dataset/t5_embeddings.pkl` plus `t5_cache_receipt.json`, generated for exactly the eight canonical prompts |
| Training request | A run-specific experiment/config identity, process topology, output root, base/data/T5 hashes, and optional checkpoint for resume |

The Cosmos base and `google-t5/t5-11b` assets have their own access and license
terms. RoboTactile neither downloads them nor grants redistribution rights.
Record the exact upstream revisions used in addition to the content hashes.
The pinned T5 implementation uses `local_files_only=True` and CUDA, so the
exact tokenizer/model snapshot must already be present in the runtime cache.

The flat Cosmos `.pt` remains the correct base input for this formal NVIDIA
route. The experimental HCU checkpointer has a different loading contract: it
must first convert that exact bound flat file into a model-only DCP tree and
pass the generated `iter_000000000` **iteration root** to
`checkpoint.load_path`. Do not point the HCU request at the flat `.pt` or at
the nested `iter_000000000/model` directory. The converter receipt proves only
format conversion and round-trip identity; it is not training evidence.

## NVIDIA host preflight

Use explicit absolute paths in the run request. The following checks are
read-only and must be run from the isolated Dream-Tac runtime; substitute the
actual shared-storage roots before use:

```bash
export ROBOTACTILE_ROOT=/absolute/RoboTactile
export DREAM_TAC_ROOT="$ROBOTACTILE_ROOT/deployment/sources/Dream-Tac"
export DREAM_TAC_PYTHON="$ROBOTACTILE_ROOT/deployment/runtime/dream-tac/bin/python"
export DREAM_TAC_DATA_ROOT=/absolute/shared-data/dream-tac-univtac
export COSMOS_BASE_CHECKPOINT=/absolute/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt

test "$(git -C "$DREAM_TAC_ROOT" rev-parse HEAD)" = \
  14bab51d6862fd07124745c55cd395ea5caa9fd3
test -f "$COSMOS_BASE_CHECKPOINT"
nvidia-smi
"$DREAM_TAC_PYTHON" -c '
import torch
assert torch.cuda.is_available()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print(torch.ones(1, device="cuda"))
'

cd "$ROBOTACTILE_ROOT"
"$DREAM_TAC_PYTHON" -m scripts.dream_tac.training.materialize_train759 \
  --raw-root /absolute/UniVTAC-HDF5 \
  --output-root "$DREAM_TAC_DATA_ROOT" \
  --verify
"$DREAM_TAC_PYTHON" -m scripts.dream_tac.training.t5_cache \
  --output-root "$DREAM_TAC_DATA_ROOT" \
  --verify
sha256sum "$COSMOS_BASE_CHECKPOINT"
```

When using Docker, prove that the same checks succeed inside the exact image
that will run training. The upstream launch requires `--gpus all` and either
`--ipc=host` or an explicitly sized shared-memory segment. A host-side CUDA
probe is not evidence for an untested container image.

## Create and gate a P2/P3 request

Start from
`configs/experiments/dream_tac/p2_micro.example.json` or
`configs/experiments/dream_tac/p3_full.example.json`, copy it to a writable
request directory, and replace every absolute path and all-zero SHA256. The
strict schema is
`configs/experiments/dream_tac/training_request.schema.json`. Do not edit the
checked-in examples into machine-local state.

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

The request binds the clean Dream-Tac checkout, its Python executable, P0
materialization root, output root, source/global/T5 identities, Cosmos base,
GPU list/topology, training length, save cadence, per-rank batch size, and data
workers. `cuda_devices` must contain exactly `nproc_per_node` unique devices;
the current launcher accepts one to eight GPUs on one node only. `p2_micro`
permits at most 1,000 iterations, while `p3_full` requires more than 1,000.
The output root must be disjoint from both source and data roots.

The fixed donor is
`cosmos_predict2_2b_480p_franka_cut_banana_20260321`, but the resolved launch
overrides its example data and job fields. It forces the train759 dataset,
both tactile streams, no tactile dropout, CASA, a 20-step chunk, no validation,
disabled W&B, the content-bound checkpoint, and
`IMAGINAIRE_OUTPUT_ROOT=<output_root>`. Do not run the donor config directly:
its upstream dataset and base-checkpoint paths are placeholders and it does
not by itself encode the RoboTactile split contract.

| Mode | What it proves |
|---|---|
| `--print-command` | Parses the request and prints the command/environment; it performs no artifact or CUDA gates and writes nothing |
| `--preflight-only` | Verifies clean pinned source, Python 3.10/CUDA dependencies, exact visible GPUs, train759/no-`val` data, T5/base hashes, and resume identity; it writes a preflight receipt but starts no trainer |
| `--dry-run` | Runs the same gates, then upstream configuration dry-run; it performs no optimizer steps |
| no mode flag | Runs the same gates, then one single-node `torchrun` exactly once for that content-addressed request |

## Receipts and resume

Receipts are written under:

```text
<output_root>/_robotactile_receipts/<run_name>/<request_sha256>/
├── preflight_receipt.json
├── resolved_experiment.json
├── <mode>_launch_plan.json
├── <mode>.log
└── <mode>_launch_result.json
```

A launch plan without a result is treated as ambiguous and is never replayed
automatically. Inspect the process and log, then create an explicit
content-addressed resume request. A launch result records the subprocess return
code and log hash; it does **not** by itself certify that a usable checkpoint
exists, that loss improved, or that evaluation succeeded.

`resume_checkpoint` is either `null` or an exact regular-file path plus
SHA256. When the same job already has
`checkpoints/latest_checkpoint.txt`, preflight requires the bound resume file
to be exactly the file named by that marker; this prevents upstream's implicit
latest-checkpoint priority from silently overriding the request. When no
same-job marker exists, a bound checkpoint is an explicit resume source.
Resume enables both `load_training_state` and `strict_resume`.

Never delete a plan/result receipt or overwrite a log to force a retry. Create
a new content-addressed request, bind the inspected checkpoint, and normally
use a successor `run_name`. A warm start from the Cosmos base keeps
`resume_checkpoint=null`; it is not a resume because optimizer/scheduler state
is intentionally absent.

## Known v1 provenance limits

The v1 launcher is sufficient for a source-bound P2/P3 process launch, but it
is not yet a complete paper-grade training record. The request fixes
`PYTHONHASHSEED=0`, while an explicit random/NumPy/Torch/dataloader seed is not
a request field. The preflight records the resolved Python/PyTorch/CUDA/GPU
environment but does not bind a container digest or full dependency lock. The
T5 receipt binds generated bytes, not the underlying `google-t5/t5-11b`
snapshot. Finally, `launch_result.json` is a process receipt rather than an
immutable inventory of output checkpoints. Retain these identities externally
for diagnostics, and close these schema/receipt gaps before using P3 for a
paper-level reproducibility claim.
