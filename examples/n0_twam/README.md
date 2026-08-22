# N0-TWAM integration example

N0-TWAM is a first-class RoboTactile `PolicyAdapter` using the typed qpos8,
two-phase grounding protocol. The external runtime remains licensed under
CC-BY-NC-SA-4.0 and is not included in the Apache-2.0 wheel.

The frozen model requires two tactile streams. A1/A2 and matched `no_touch`
therefore return `unsupported_contract`; black images are never substituted.
All-zero hashes in the example files are intentionally non-runnable and must be
replaced by a content-addressed model-bundle manifest.

```bash
robotactile integrations validate --model n0_twam
```

After placing real files under
`deployment/artifacts/models/n0_twam/`, generate and diagnose the runnable
deployment config without hand-editing hashes:

```bash
robotactile deployment init
robotactile integrations configure n0-twam
robotactile integrations doctor --model n0_twam
```

Live N0 execution remains fail-closed until a production transport and published
serving bundle are registered. Contract tests do not establish live TWAM
inference or simulator success.
