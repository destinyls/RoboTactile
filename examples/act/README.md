# ACT integration example

ACT is a first-class RoboTactile `PolicyAdapter`. The tactile profile evaluates
`clean`, `faulted`, and `restored`; a separately trained vision-only profile is
the matched `no_touch` control.

The checked-in JSON files are contract examples. All-zero hashes deliberately
fail content validation and must be replaced with hashes from your own artifact
manifest. RoboTactile never downloads weights implicitly.

Validate the static integration and external pin:

```bash
robotactile integrations validate --model act
```

Generate the runnable deployment config from real files instead of editing the
all-zero example:

```bash
robotactile deployment init
robotactile integrations configure act \
  --task pull_out_key --profile univtac
robotactile integrations doctor --model act
```

The example request records the public contract shape only. Replace all-zero
hashes, source paths, runtime paths, and output paths with frozen deployment
values before running the live preflight:

```bash
robotactile preflight-live \
  --request examples/act/request.json \
  --config "$PWD/deployment/artifacts/models/act/integration_config.json"
```

Preflight validates resources and host readiness without allocating a simulator.
It is not task-success evidence.
