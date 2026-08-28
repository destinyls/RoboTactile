# N0-TWAM integration example

N0-TWAM is a first-class RoboTactile `PolicyAdapter` using the official
websocket server, 20-D rot6d chunks, UniVTAC absolute EE execution, and causal
KV-cache re-grounding. The external source/base model remain CC-BY-NC-SA-4.0;
the released UniVTAC delta checkpoint declares Apache-2.0. None is included in
the RoboTactile wheel.

The frozen model requires two tactile streams. A1/A2 and matched `no_touch`
therefore return `unsupported_contract`; black images are never substituted.
All-zero hashes in the example files are intentionally non-runnable and must be
replaced by a content-addressed model-bundle manifest.

```bash
robotactile integrations validate --model n0_twam
```

Prepare the official source and one task without hand-editing hashes:

```bash
robotactile deployment init
bash scripts/n0_twam/install_official_runtime.sh --root "$PWD/deployment"
bash scripts/n0_twam/install_robotactile_client.sh --root "$PWD/deployment"
"$PWD/deployment/runtime/n0-twam/bin/python" \
  scripts/n0_twam/prepare_official_artifacts.py \
  --root "$PWD/deployment/artifacts/models/n0_twam" \
  --task pull_out_key
robotactile integrations doctor --model n0_twam --task pull_out_key
"$PWD/deployment/runtime/n0-twam/bin/python" \
  scripts/n0_twam/generate_clean_request.py \
  --root "$PWD/deployment" \
  --task pull_out_key \
  --initial-seed 17 \
  --exogenous-seed 29 \
  --simulator-device cuda:0
```

The generator writes a canonical `trial_set_manifest.json` and uses its exact
content hash as `dataset_sha256`; that field is a frozen task/seeds identity,
not a claim about a raw episode file. Select the physical Isaac GPU with
`CUDA_VISIBLE_DEVICES` when the request is executed.

See `docs/model_integrations.md` for server launch and `evaluate` commands.
Contract tests do not establish live TWAM inference or simulator success.

For the real HDF5 open-loop robustness path, including the pinned
`lift_bottle/clean/90.hdf5` example, 12-step target semantics, all-operator
command, and artifact boundary, see `docs/recorded_n0.md`. This path runs the
official N0 model but intentionally does not claim Isaac task success.
