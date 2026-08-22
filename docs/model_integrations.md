# Model Integrations

RoboTactile registers exactly two first-class model integrations: `act` and
`n0_twam`. Both implement the same `PolicyAdapter` lifecycle (`reset`, `infer`,
`commit`, `abort`, and `close`) and feed the same fault injector, closed-loop
runner, artifact, matrix, and reporting protocols. Model source and weights
remain external to the Apache-2.0 wheel.

Initialize the [deployment layout](deployment_layout.md) and inspect the static
registry first:

```bash
robotactile deployment init
robotactile integrations list
robotactile setup --model act
```

## ACT end-to-end

The complete ACT chain is:

```text
WorldArena + UniVTAC pinned sources
  -> deployment/artifacts/models/act/<task>/<profile>/
  -> artifact_manifest.json
  -> integration_config.json
  -> request -> integrations doctor -> preflight-live
  -> live-univtac-run/evaluate -> content-addressed live artifact
```

### 1. Install sources

```bash
bash integrations/install_univtac.sh
bash integrations/install_act_runtime.sh
```

The exact commits, licenses, and immutable hyperlinks are listed in
[External dependencies](external_dependencies.md).

### 2. Place real artifacts

For task `pull_out_key`, use the canonical layout for both mandatory profiles:

```text
deployment/artifacts/models/act/
├── encoder.pth
└── pull_out_key/
    ├── univtac/
    │   ├── policy_last.ckpt
    │   └── dataset_stats.pkl
    └── vision_only/
        ├── policy_last.ckpt
        └── dataset_stats.pkl
```

`univtac` consumes the two tactile streams. `vision_only` is a separately
trained matched `no_touch` system; a black image, resting tactile frame, or
structurally absent stream is not a substitute.

### 3. Generate canonical config

Run once per profile. Because the default output names are shared, retain the
two manifests under explicit profile names and choose the relevant config for
each request/matrix stage:

```bash
robotactile integrations configure act \
  --task pull_out_key \
  --profile univtac \
  --manifest-output "$PWD/deployment/artifacts/models/act/univtac_manifest.json" \
  --integration-config-output "$PWD/deployment/artifacts/models/act/univtac_config.json"

robotactile integrations configure act \
  --task pull_out_key \
  --profile vision_only \
  --manifest-output "$PWD/deployment/artifacts/models/act/vision_only_manifest.json" \
  --integration-config-output "$PWD/deployment/artifacts/models/act/vision_only_config.json"
```

The command computes checkpoint, stats, encoder, upstream source, and training
config hashes from the real files. It validates the clean UniVTAC commit and
official layout, emits canonical JSON, and refuses to replace different
existing output. It does not load Torch or run inference.

For a single tactile run, the default names may be used:

```bash
robotactile integrations configure act --task pull_out_key
```

### 4. Diagnose, preflight, and execute

```bash
robotactile integrations doctor \
  --model act \
  --config "$PWD/deployment/artifacts/models/act/univtac_config.json"

robotactile preflight-live \
  --request "$REQUEST_PATH" \
  --config "$PWD/deployment/artifacts/models/act/univtac_config.json"

"$PWD/deployment/runtime/isaac-sim-4.5.0/python.sh" \
  -m robotactile_benchmark.cli live-univtac-run \
  --request "$REQUEST_PATH" \
  --config "$PWD/deployment/artifacts/models/act/univtac_config.json"
```

Legacy explicit artifact-root and hash arguments remain accepted, but the
typed config is the recommended single runtime input. `preflight-live` does
not allocate Isaac or the model. A live run remains
`unqualified_live_univtac_execution_v1` until the separate qualification
protocol passes.

## N0-TWAM end-to-end

The N0 chain is deliberately fail-closed:

```text
N0-TWAM + UniVTAC pinned sources
  -> explicit checkpoint/config/normalizer files
  -> artifact_manifest.json
  -> integration_config.json
  -> integrations doctor
  -> production typed gateway (not registered in this release)
```

### 1. Install sources and place artifacts

```bash
bash integrations/install_univtac.sh
bash integrations/install_n0_twam.sh
```

Default model paths are:

```text
deployment/artifacts/models/n0_twam/
├── checkpoint.pt
├── config.json
└── normalizer.json
```

Alternative filenames are supported through `--checkpoint`,
`--model-config`, `--normalizer`, and `--bundle-root`. All paths are recorded
absolutely in the generated manifest and must stay below its bundle root.

### 2. Generate and diagnose config

```bash
robotactile integrations configure n0-twam
robotactile integrations doctor --model n0_twam
```

Configuration hashes the three real resources and binds the fixed external
commit. The doctor validates the N0-TWAM and UniVTAC checkouts, manifest, and
files, then reports the missing production typed gateway as a failed gate.
It never starts a fake backend to manufacture a pass.

Endpoint addresses and credentials are runtime-only inputs for a future
gateway registration. They must come from command arguments or environment
variables and must never be stored in a tracked config, request, README, or
install receipt.

The frozen N0 contract requires both tactile streams. A1, A2, and matched
`no_touch` therefore remain `unsupported_contract`; black frames are never
substituted. `robotactile evaluate --model n0_twam ...` exits fail-closed
before reading a request until the production transport and serving bundle are
registered.

## Static examples versus runnable deployment files

`examples/act/` and `examples/n0_twam/` document contract shapes only. Their
all-zero hashes are intentionally non-runnable. Do not edit them into local
runtime files; run `integrations configure` so hashes are computed from real
artifacts under `deployment/`.

The checked-in `configs/integrations/*.json` similarly identify the two static
registrations. Generated deployment configs use absolute manifest paths and
are ignored by Git.

## Evidence boundaries

| Operation | Establishes | Does not establish |
|---|---|---|
| `integrations configure` | Real local files were hashed into a canonical config | Model load or inference |
| `integrations doctor` | Declared source/artifact/transport gates passed or failed | Simulator allocation or task success |
| `preflight-live` | ACT request, source, artifacts, GPU, and Isaac Python readiness | Simulator execution |
| live artifact | One requested runtime produced a strict-loadable trace | Isaac qualification or hardware validity |
