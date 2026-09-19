# Five-model acceptance

This tool reads existing live artifacts. It never starts a simulator, campaign,
or model service, and never executes inferred actions. Run `infer` only against
already-running, exclusively reserved services: policy reset changes their state.
ACT runs in the selected Python process and requires its official integration
config rather than a four-model retrained binding.

From the RoboTactile source root:

```bash
python -m scripts.model_acceptance.run --manifest /absolute/acceptance.json \
  --output /absolute/new-audit-output --mode audit
python -m scripts.model_acceptance.run --manifest /absolute/acceptance.json \
  --output /absolute/new-inference-output --mode infer --timeout 900
```

The manifest is one JSON object containing `models`, a list with exactly one
row for each ID: `act`, `dream_tac`, `ftp1_policy`, `n0_twam`, `n0_vtla`.
All paths are absolute; `package` is the directory containing the installed
`robotactile_benchmark` package. Each row follows this schema:

```json
{
  "model": "n0_twam",
  "binding_path": "/deployment/campaign/bindings/n0_twam.json",
  "code": "/source/RoboTactile",
  "package": "/deployment/isolated-wheel-install",
  "live_artifact": "/deployment/campaign/groups/n0_twam/TASK/artifacts/clean",
  "source_root": "/deployment/sources/N0-TWAM",
  "runtime_python": "/deployment/runtime/n0-twam/bin/python",
  "request_path": "/deployment/campaign/groups/n0_twam/TASK/clean.request.json",
  "server_receipt": "/deployment/campaign/servers/n0_twam/TASK/worker/server_receipt.json"
}
```

`request_path` is mandatory only for `infer`; it must reference the exact
captured trial and run specification. Use the actual generated request path,
not the illustrative name above. `server_receipt` is optional and overrides
the binding's receipt path for N0-TWAM. The other services use their bound
endpoint and adapter metadata checks. For ACT, `binding_path` must point to
`artifacts/models/act/configs/TASK/PROFILE/integration_config.json` and
`runtime_python` should be its Isaac Python launcher. No service is started.

The parent creates independent model JSON receipts and logs, plus summary.json.
Each subprocess uses its selected runtime with `PYTHONPATH=package:code` plus
the retrained binding's `shared_pythonpath`. CUDA/CUDNN and OPENPI paths follow
the existing campaign policy environment, while caller GPU selection is kept.
ACT config is not interpreted as a retrained binding.
Model failures/timeouts do not skip later models.
Output directories must be new. A nonzero exit means at least one check failed.

Audit strictly reloads the entire bundle, verifies its model/checkpoint binding,
and inspects retained observation arrays, action arrays, step order and tactile
delivery times. A passing audit requires nonempty finite Nx8 actions, retained
observations, consistent record links, and a score-eligible validated episode.
Historical N0 `policy_kind=n0` maps only to `model=n0_twam`. Checkpoint and
source checks still apply. Saved action traces contain executed prefixes and
plan hashes, not full planned arrays; audit explicitly reports the planned
arrays as unavailable. Only a new inference call measures the full plan shape.
Task success is reported separately: a valid failed task can pass integrity
acceptance. Preview captures report retained versus total observation counts
and cannot prove that all observations were preserved. The loader's simulator
qualification flag is preserved; a root receipt alone does not establish an
official simulator or real-robot result.

Inference strictly loads the original request, uses the exact captured step-0
observation and real policy identity, resets the policy, then calls infer once.
It uses `group.policy_factory(binding, loaded)` for four retrained models and
the official ACT loader for ACT. Input array hashes and returned ActionPlan
hash/shape/finite status are saved. Inference pass is independent of historical
episode score eligibility; the audit result remains visible separately.
This is OFFLINE inference, not a new CLOSED-LOOP result. Native imports, CUDA
architecture compatibility, weights and service availability are verified only
when the actual respective model load/inference reaches those boundaries.

## New bounded ACT/FTP-1 episode

This explicit entrypoint starts one real simulator episode; it never starts a
model service. Run it in the compatible Isaac Python environment. FTP-1 requires
an already running exclusive service, while ACT loads in-process:

```bash
/absolute/isaac-python.sh -m scripts.model_acceptance.episode \
  --model act --binding /absolute/act/integration_config.json \
  --output /absolute/new-act-episode --task grasp_classify --seed 110911 --max-cycles 3
/absolute/isaac-python.sh -m scripts.model_acceptance.episode \
  --model ftp1_policy --binding /absolute/ftp1-binding.json \
  --output /absolute/new-ftp1-episode --task grasp_classify --seed 110911 --max-cycles 3
```

The request uses 3 control cycles, 4 observations and 1800 seconds wall timeout
by default. It preserves weights, action contract and official success predicate.
ACT generates a hashed seeded diagnostic dataset identity; FTP-1 preserves its
binding dataset identity. Both use a private runtime directory inside the new
output. Existing output directories are rejected.

Outputs include launch.json (hostname/GPU inventory), canonical request.json,
full live_artifact/, inference_trace.jsonl (flushed after every inference), final
inference_trace.json and acceptance.json. Each inference retains the entire
planned action array, shape, finite check, plan hash, input hashes and duration;
the acceptance receipt checks that executed prefixes match those plans. This is
bounded runtime acceptance, not an SR experiment. Budget exhaustion/timeout is
never counted as task success. A valid bounded execution can pass runtime checks
while reporting task failure. Startup/export errors preserve the launch/error
receipt and any already written traces.
Critical inference_trace.json and acceptance.json are published by the artifact
exporter before simulator teardown, which may exit the Isaac interpreter.

## Final fail-closed matrix

Run on the inventoried host with the current execution snapshot on PYTHONPATH:

```bash
PYTHONPATH=/absolute/output-root/code/src:/absolute/output-root/code \
  /absolute/isaac-python.sh -m scripts.model_acceptance.report \
  --root /absolute/output-root --output /absolute/new-report
```

Inputs are `host_inventory.json`, `inventory/MODEL.json`, `runtime/MODEL.json`
and `.exit.json`, `curobo_gpu_probe.json`, `execution/MODEL/` receipts, and
`adopted_audit/{dream_tac,n0_twam,n0_vtla}.json`. Adopted audits can contain
their hostname-bearing launch object. ACT/FTP-1 require completed_episode/
with the canonical request, full live artifact, acceptance and inference trace.
For older runs where Isaac teardown prevented post-close JSON publication,
acceptance.json may be absent: the report independently reloads the strict live
artifact and inference_trace.jsonl, verifies plan hashes/source/spec/executed
prefixes and completion, and labels this report-time revalidation. It never
creates or claims an original historical acceptance receipt.

`READY_CLOSED_LOOP` requires independent weights/source/runtime/load/inference
gates plus same-host valid completed execution. A completed execution either
succeeds early or actually executes the task's full action horizon. A valid
3-cycle smoke alone can yield `READY_INFERENCE_ONLY`. Failed/interrupted or
prematurely timed-out full runs retain `READY_INFERENCE_ONLY` when inference
has independently passed. Simulator/cuRobo failures also cannot erase verified
model inference. `BLOCKED` means a weight/source/model-runtime/load/inference
gate failed. Each model reports only its first
failed necessary gate and exact evidence/log path.

Source members are rehashed against the inventory. N0's non-Git snapshot is
validated using the prepared artifact and source-bound server receipt; its
historical task normalizer is compared with the prepared task normalizer, never
with the binding's aggregate hash. ACT's unpinned inventory values are checked
against its canonical integration manifest. Dream's complete 17-member DCP
table and aggregate hash are retained. Full new plans are reconstructed and
their hashes/executed prefixes revalidated.

Outputs are report.json and matrix.md. Module resolution is a separate probe at
report generation time and explicitly does not claim introspection of a past
process. The execution source manifest, when present at
deployment/execution_source_manifest.json, is referenced separately.
