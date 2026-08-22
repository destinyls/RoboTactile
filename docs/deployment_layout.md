# Deployment Layout

RoboTactile uses the repository checkout as the discoverable project root and
one ignored `deployment/` tree as the writable runtime workspace. This keeps
source, third-party repositories, model files, requests, outputs, and evidence
receipts visibly separate without requiring a second manually assembled root.

## Root resolution

Every deployment-aware command uses the same precedence:

1. explicit `--root PATH`;
2. `ROBOTACTILE_DEPLOY_ROOT`;
3. `<RoboTactile checkout>/deployment`.

The default is suitable for a local paper reproduction:

```bash
robotactile deployment show
robotactile deployment init
robotactile deployment doctor --profile core
```

For an HPC shared filesystem, set one absolute task-specific override. The
directory contract below does not change:

```bash
export ROBOTACTILE_DEPLOY_ROOT=/absolute/shared/project/RoboTactile-deployment
robotactile deployment init
```

Filesystem roots and broad storage roots such as `/`, `/data`, and `/mnt` are
rejected. The tracked checkout and deployment root have different ownership:
`ROBOTACTILE_ROOT` may denote the checkout, but it never denotes writable
deployment state.

## Complete tree

`robotactile deployment init` creates this idempotent layout and a canonical
`artifacts/deployment/layout_receipt.json`:

```text
RoboTactile/
├── src/ configs/ integrations/ docs/ tests/  # tracked release source
└── deployment/                               # ignored writable state
    ├── sources/
    │   ├── UniVTAC/
    │   ├── WorldArena/
    │   ├── N0-TWAM/
    │   ├── IsaacLab/
    │   └── curobo/
    ├── runtime/
    │   ├── isaac-sim-4.5.0/
    │   ├── cache/ home/ locks/ tmp/
    │   └── omni-cache/ pip-cache/
    ├── artifacts/
    │   ├── models/{act,n0_twam}/
    │   ├── deployment/
    │   ├── preflight/
    │   ├── rest-references/
    │   └── live-univtac/
    ├── requests/{calibration,four-condition,primary-matrix}/
    ├── outputs/{matrices,reports}/
    └── logs/
```

| Directory | Owner | Contents |
|---|---|---|
| `sources/` | external installers | Exact detached Git checkouts; never imported into the main wheel |
| `runtime/` | deployment scripts | Isaac, caches, isolated `HOME`, locks, and temporary staging |
| `artifacts/models/` | user transfer/configure step | ACT and N0-TWAM files plus generated manifests/configs |
| `artifacts/deployment/` | installers | Install, layout, and infrastructure-smoke receipts |
| `artifacts/preflight/` | preflight CLI | No-allocation readiness receipts |
| `artifacts/live-univtac/` | live runner | Content-addressed closed-loop artifacts |
| `requests/` | generators | Immutable calibration, four-condition, and primary-matrix bundles |
| `outputs/` | runner/reporter | Resumable matrix state and source-bound reports |
| `logs/` | deployment scripts | Hash-bound command logs |

Nothing under `deployment/` is tracked or included in the wheel/sdist.
Checkpoints, datasets, simulator archives, credentials, outputs, and nested
repositories must remain there or in another ignored absolute deployment root.

## Commands and defaults

```bash
# Inspect only; performs no writes.
robotactile deployment show

# Create the known directories and no-clobber receipt.
robotactile deployment init

# Check expected files/directories for one runtime profile.
robotactile deployment doctor --profile act-univtac
robotactile deployment doctor --profile n0-univtac
```

The calibration, four-condition, primary-matrix, preflight, live-matrix, and
reporting CLIs use this layout when their path arguments are omitted. Existing
explicit absolute paths remain supported and are never silently moved.

## Read-only legacy migration plan

The planner inventories recognized legacy files, computes each SHA-256, and
prints its proposed destination. It performs no copy, replacement, or delete:

```bash
robotactile deployment migrate-plan --legacy-root /absolute/legacy/deployment
```

The JSON result records `writes_performed=false` and `source_deleted=false`.
Review the plan before copying data with a separate, user-controlled tool.

## Cleanup and evidence boundary

The safe cleanup boundary is the exact resolved deployment root printed by
`robotactile deployment show`; never target its parent or a shared storage
root. Deleting it removes generated and external state, not tracked source, but
may destroy expensive simulator installs and model artifacts.

A layout receipt proves directory initialization only. An install receipt,
preflight receipt, headless smoke receipt, live artifact, and qualification
receipt each prove different operations. Creating this tree does not establish
model inference, simulator launch, task success, or Isaac qualification.
