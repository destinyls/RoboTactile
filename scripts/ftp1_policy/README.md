# FTP-1 operational entry points

Run these commands from the RoboTactile repository root. FTP-1 is a first-class
`PolicyAdapter` integration with an isolated worker; it does not run through
ACT, N0-TWAM, or N0-VTLA.

## Frozen identities

| Asset | Identity | License boundary |
|---|---|---|
| `michaelyuancb/ftp1-policy` | `89fa681d6c014cce28300946b7526db808e0b1c1` | Apache-2.0 source |
| `MJJJJ1064/ftp1_univtac_finetune` | `620ac69b4fffd2341300cfef1b1d224d56710ed3` | Weight license not specified upstream; review Gemma terms |

The six released tasks are `insert_hole`, `insert_tube`, `lift_bottle`,
`lift_can`, `pull_out_key`, and `put_bottle_in_shelf`. Source and checkpoints
remain outside the RoboTactile wheel.

## Scripts

| File | Purpose | Evidence produced |
|---|---|---|
| `install_official_runtime.sh` | Clone the exact source, install the Blackwell cu128 bundle, and create `deployment/runtime/ftp1-policy` | Install receipt only after a real CUDA kernel probe |
| `provision_openpi_tokenizer.py` | Download and hash-check PaliGemma tokenizer into a deployment-local `OPENPI_DATA_HOME` | No-clobber tokenizer receipt |
| `install_isaac_client.sh` | Install the pinned binary RPC client only into `deployment/runtime/isaac-sim-4.5.0` | Isaac-client install receipt only |
| `serve_official.py` | Load one task checkpoint and serve the binary ZMQ policy contract | Runtime logs/metadata; no simulator result |
| `prepare_robustness_group.py` | Freeze one Clean plus 12 Faulted requests and A1/A2 N/A records | Plan/request evidence; no execution |
| `training/orchestrate.py` | Prepare RGB-correct train759 and train one joint all-eight FTP-1 model | Source, RGB, conversion, joint normalization, and single-checkpoint receipts |

`requirements-runtime.txt` is installer input, not a user entry point.

## Train one model on all eight UniVTAC tasks

The upstream release uses one fine-tuned checkpoint per task and publishes six
of the eight UniVTAC tasks. RoboTactile additionally provides a joint route for
one model covering all eight tasks, including `grasp_classify` and
`insert_HDMI`. One eight-entry `MultiZarrDataset` is normalized once and sent
through one optimizer/checkpoint tree. The task is conditioned by each Zarr
episode's language instruction; domain-homogeneous batches keep different
camera schemas compatible.

The request contract pins the upstream source, FTP-1 pretrained checkpoint,
official UniVTAC snapshot, and frozen `train759` manifest. RGB and both tactile
pads are always active with `non_tactile_dropout_ratio=0.0`. A RoboTactile
wrapper corrects the pinned parser's duplicate channel reversal without
modifying the external checkout, and exact HDF5-to-Zarr RGB checks gate every
task before joint normalization.

See [`configs/experiments/ftp1_policy/README.md`](../../configs/experiments/ftp1_policy/README.md)
for the no-`uv`, single-GPU joint workflow. Training outputs and source
data remain under the selected deployment root and do not modify the N0-TWAM,
N0-VTLA, ACT, Isaac, or system Python environments.

## Install and configure `insert_hole`

```bash
export DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$PWD/deployment}"
export TASK=insert_hole

bash scripts/ftp1_policy/install_official_runtime.sh --root "$DEPLOY_ROOT"
bash scripts/ftp1_policy/install_isaac_client.sh --root "$DEPLOY_ROOT"

hf download MJJJJ1064/ftp1_univtac_finetune \
  --revision 620ac69b4fffd2341300cfef1b1d224d56710ed3 \
  --include 'FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1/19999/**' \
  --local-dir "$DEPLOY_ROOT/artifacts/models/ftp1_policy"

robotactile integrations configure ftp1-policy \
  --root "$DEPLOY_ROOT" --task "$TASK"
robotactile integrations doctor \
  --model ftp1_policy --root "$DEPLOY_ROOT" --task "$TASK"
```

When `DEPLOY_ROOT` is on slow shared storage, set
`ROBOTACTILE_LOCAL_SCRATCH=/tmp` for this installer. Only disposable pip
downloads/builds use node-local scratch; the source, runtime, receipt, model,
and evaluation artifacts remain under `DEPLOY_ROOT`, and scratch is removed on
exit.

The installer defaults to the HTTPS Tsinghua PyPI registry frozen by the
reviewed upstream `uv.lock`, so a machine-wide pip mirror cannot silently
change dependency resolution. Set `ROBOTACTILE_FTP1_PIP_INDEX_URL` only when
an equivalent HTTPS mirror is required; the constraints file still fixes the
reviewed direct versions.

The server runtime installs PyTorch `2.7.1+cu128`, torchvision
`0.22.1+cu128`, Triton `3.3.1`, and the 14 exact CUDA wheels declared by the
official PyTorch cu128 wheel before resolving FTP-1. This 17-package bundle is
read back from package metadata after installation. The runtime receipt is
written only when `torch.version.cuda == "12.8"`, the compiled architecture
list contains `sm_120` (or `compute_120`), and a real `torch.ones(...,
device="cuda")` kernel completes. These packages remain confined to
`deployment/runtime/ftp1-policy`; system CUDA and the N0/ACT/Isaac runtimes are
not modified.

OpenPI's PaliGemma tokenizer is provisioned independently at
`deployment/artifacts/openpi-data/ftp1-policy/big_vision/paligemma_tokenizer.model`.
The provisioner binds the 4,264,023-byte object to SHA-256
`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`,
publishes it without overwriting an existing file, and writes a separate
receipt. It never uses `~/.cache/openpi` or a root cache.

The configured artifact manifest hashes the model, HPT tactile tokenizers,
normalization inventory, and all serving configs. It also binds the task
prompt, camera route, checkpoint revision, source commit, and temporal/action
contract.

`install_isaac_client.sh` installs exactly `msgpack==1.1.1` and
`pyzmq==27.1.0` through Isaac Sim's own `python.sh`. The msgpack version is the
same version used by the N0-TWAM Isaac client, so installing both clients does
not invalidate the N0-TWAM dependency contract or its separate receipt. The
script does not modify system Python/CUDA, `runtime/ftp1-policy`, or
`runtime/n0-twam`.

## Serve

Read the manifest hashes and launch the task worker:

```bash
export FTP1_MANIFEST="$DEPLOY_ROOT/artifacts/models/ftp1_policy/configs/$TASK/artifact_manifest.json"
export FTP1_CHECKPOINT_DIR="$DEPLOY_ROOT/artifacts/models/ftp1_policy/FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1/19999"
export FTP1_CHECKPOINT_SHA256="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_sha256"])' "$FTP1_MANIFEST")"
export FTP1_SERVE_BUNDLE_SHA256="$(python -c 'import json,sys; from robotactile_benchmark.integrations.ftp1_policy.artifacts import FTP1PolicyArtifactManifest; print(FTP1PolicyArtifactManifest.from_dict(json.load(open(sys.argv[1]))).serve_bundle_sha256)' "$FTP1_MANIFEST")"
export OPENPI_DATA_HOME="$DEPLOY_ROOT/artifacts/openpi-data/ftp1-policy"

CUDA_VISIBLE_DEVICES=0 OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
  "$DEPLOY_ROOT/runtime/ftp1-policy/bin/python" \
  scripts/ftp1_policy/serve_official.py \
  --bind 'tcp://*:5561' \
  --source-root "$DEPLOY_ROOT/sources/ftp1-policy" \
  --checkpoint-dir "$FTP1_CHECKPOINT_DIR" \
  --checkpoint-sha256 "$FTP1_CHECKPOINT_SHA256" \
  --serve-bundle-sha256 "$FTP1_SERVE_BUNDLE_SHA256" \
  --domain-name UniVTAC_insert_hole \
  --task insert_hole \
  --device cuda:0
```

The worker reproduces the pinned online policy: one inference per fresh
observation, raw `(32, 120)` prediction, skip index 0, admit indices `[1:21]`,
ensemble overlapping candidates with `K=0.01`, convert the `mix` representation
against inference-time qpos, and return one absolute qpos8 action.

## Prepare and run robustness

```bash
python scripts/ftp1_policy/prepare_robustness_group.py --help
robotactile live-univtac-paired-run --help
```

The generated group contains Clean and F1--F7/T1--T3/C1--C2 Faulted requests.
A1/A2 are explicit N/A because this checkpoint contract requires both tactile
tensors. Operators needing a rest image require `--rest-reference` pointing to
a certified task-bound artifact. The paired runner must receive Clean first and
the 12 generated request paths in plan order; it restores one canonical
post-reset simulator snapshot before every condition.

Use `metrics_only_v1` for a bounded first run. It preserves metrics, actions,
termination, diagnostics, and hashes without RGB/tactile video payloads. This
changes storage volume, not inference, fault injection, or success scoring.

## Evidence boundary

- Passing tests/configuration is **CODE**, not a model result.
- Loading the worker or returning a contract-valid action is **OFFLINE**, not
  simulator success.
- A strict paired Isaac artifact is **CLOSED-LOOP** for that task/seed only.

Do not report an FTP-1 Success Rate until the preregistered task/seed grid has
completed with qualified, source-bound artifacts and an explicit denominator.
The integration itself contains no fabricated or generated metric.
