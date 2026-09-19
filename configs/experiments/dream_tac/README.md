# Dream-Tac UniVTAC training requests

## Formal NVIDIA P2/P3 requests

`p2_micro.example.json` and `p3_full.example.json` are strict, source-bound
requests. Replace every absolute path and all-zero SHA256 placeholder with the
identities produced by P0/P1, the real T5 cache, and the compatible
Cosmos-Predict2 2B base file.

The launcher is NVIDIA CUDA only and supports one node with one to eight local
GPUs. It does not claim HCU/HPU support or multi-node support.

```bash
python -m scripts.dream_tac.training.launch_training \
  --request configs/experiments/dream_tac/p2_micro.example.json \
  --print-command

python -m scripts.dream_tac.training.launch_training \
  --request configs/experiments/dream_tac/p2_micro.example.json \
  --preflight-only

python -m scripts.dream_tac.training.launch_training \
  --request configs/experiments/dream_tac/p2_micro.example.json \
  --dry-run

python -m scripts.dream_tac.training.launch_training \
  --request configs/experiments/dream_tac/p2_micro.example.json
```

`--print-command` is explicitly ungated and write-free. Every other mode runs
the artifact and NVIDIA environment gates first. A new run refuses an existing
same-job `latest_checkpoint.txt`. To resume, bind the exact checkpoint file and
SHA256 in `resume_checkpoint`; if a same-job latest marker exists, the bound
file must be exactly the file named by that marker because upstream gives it
priority over `checkpoint.load_path`.

## Experimental single-HCU optimizer-step request

`hcu_optimizer_step.example.json` is a separate request governed by
`hcu_optimizer_step_request.schema.json`. It does not modify, wrap, or replace
the NVIDIA launcher above. Its scope is fixed to one HCU, one process, one
batch, one optimizer step, one checkpoint save, and no resume.

Before creating the request, obtain the authorized Cosmos flat base and
matching `tokenizer/tokenizer.pth`, prepare the public T5-11B snapshot/cache on
a networked machine, and transfer all artifacts to the offline HCU host. The
HCU request must bind:

- the original flat base and SHA256 as `base_checkpoint` provenance;
- the converter-produced `.../iter_000000000` directory as `base_dcp_root`;
- the generated `converter_receipt.json` and its file SHA256;
- the tokenizer checkpoint, train759 source/materialization/T5 identities,
  v2 four-file overlay receipt, and passing CASA BF16 receipt;
- the exact Python 3.10 HIP runtime roots, one device, output root, and port.

The flat `.pt` cannot be used directly as HCU `checkpoint.load_path`.
Likewise, `base_dcp_root` must be the iteration directory containing `model/`,
not the nested `model/` directory itself. See
[`scripts/dream_tac/hcu_port/README.md`](../../../scripts/dream_tac/hcu_port/README.md)
for the exact conversion command.

Copy the example to a writable request directory and replace every placeholder
path and digest. Then run:

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

`--print-command` checks no gates and writes nothing. `--preflight-only`
verifies bound artifacts and the exact one-HCU HIP environment but executes no
optimizer step. There is no HCU `--dry-run` mode. Only the final command can
launch, and only a no-clobber `optimizer_step_launch_result.json` with
`optimizer_step_completed=true` is evidence for one optimizer step. It is not
evidence for full training, convergence, model quality, or task success. No
such completed HCU result is claimed by these checked-in examples.
