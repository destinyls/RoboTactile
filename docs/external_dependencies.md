# External Dependencies

RoboTactile's Apache-2.0 wheel does not vendor simulator, model, or upstream
task repositories. `integrations/integrations.lock.json` is the only authority
for repository URL, exact commit, license boundary, install entry point, source
directory, and release readiness.

## Frozen inventory

| Role | Immutable source | License at the pinned commit | Default directory | Installer |
|---|---|---|---|---|
| UniVTAC task/simulator | [UniVTAC `05bcd3e`](https://github.com/univtac/UniVTAC/tree/05bcd3edb92237107efa40105292a24f1a9fd761) | Apache-2.0 | `deployment/sources/UniVTAC` | `integrations/install_univtac.sh` |
| ACT runtime source | [WorldArena `e295378`](https://github.com/WorldArena2/WorldArena-2.0/tree/e295378c702b2e87617ebddcad193be3608e00c3) | Apache-2.0 | `deployment/sources/WorldArena` | `integrations/install_act_runtime.sh` |
| N0-TWAM source | [N0-TWAM `c43a216`](https://github.com/neoteai/N0-TWAM/tree/c43a2160dd31c449d92b28eab52c0e2f09e4738a) | CC-BY-NC-SA-4.0 | `deployment/sources/N0-TWAM` | `scripts/n0_twam/install_official_runtime.sh` |
| N0-TWAM base weights | [n0-twam-base `dafcb05`](https://huggingface.co/NeoteAI/n0-twam-base/tree/dafcb053902cc7a51310780fdfdb71ff8e79c0be) | CC-BY-NC-SA-4.0 | `deployment/artifacts/models/n0_twam/base` | `scripts/n0_twam/prepare_official_artifacts.py` |
| N0-TWAM UniVTAC delta weights | [n0-twam-univtac-delta `7694e63`](https://huggingface.co/NeoteAI/n0-twam-univtac-delta/tree/7694e63707a8c9e69e1a1242c4ed74ee39b7bb51) | Apache-2.0 | `deployment/artifacts/models/n0_twam/univtac-delta` | `scripts/n0_twam/prepare_official_artifacts.py` |
| IsaacLab v2.1.1 | [IsaacLab `90b79bb`](https://github.com/isaac-sim/IsaacLab/tree/90b79bb2d44feb8d833f260f2bf37da3487180ba) | BSD-3-Clause | `deployment/sources/IsaacLab` | `scripts/live_univtac/install_isaaclab_v2_1_1.sh` |
| cuRobo v0.7.7 | [cuRobo `0a50de1`](https://github.com/NVlabs/curobo/tree/0a50de1ba72db304195d59d9d0b1ed269696047f) | NVIDIA non-commercial (`LicenseRef-NVIDIA-NonCommercial`) | `deployment/sources/curobo` | `scripts/live_univtac/install_curobo_v0_7_7.sh` |
| Modified UIPC build tool | [vcpkg `ce613c4`](https://github.com/microsoft/vcpkg/tree/ce613c41372b23b1f51333815feb3edd87ef8a8b) | MIT | `deployment/sources/vcpkg-2025.04.09` | `scripts/live_univtac/install_tacex_uipc_univtac.sh` |

The cuRobo license above is the license in the pinned v0.7.7 commit; it must
not be replaced with the license of a newer branch. N0-TWAM and cuRobo impose
non-commercial restrictions. Read their upstream license files before use or
redistribution.

TacEx and modified UIPC source are embedded in, and therefore source-bound to,
the pinned UniVTAC checkout. They are installed through two direct scripts and
separate receipts. The native UIPC installer additionally freezes the vcpkg
tool commit, the older libuipc ports baseline, micromamba archive digest, and
every Conda toolchain package digest. The repository-owned
`cpptrace==0.8.3` recipe is mirrored from the pinned tool checkout. The sibling
`tinygltf==2.9.3` recipe keeps the baseline source version but re-pins the
regenerated official GitHub archive bytes. Every other port stays on libuipc's
embedded baseline. Missing or changed digests are deployment readiness
failures, not permission to fetch a mutable branch.

## Install exact source checkouts

From the repository root, the generic integrations require no destination
argument:

```bash
robotactile deployment init
bash integrations/install_univtac.sh
bash integrations/install_act_runtime.sh
bash scripts/n0_twam/install_official_runtime.sh --root "$PWD/deployment"
```

An explicit absolute destination remains supported:

```bash
bash integrations/install_univtac.sh /absolute/external/UniVTAC
```

Each installer verifies the exact origin, detached commit, and complete clean
inventory. It refuses dirty or mismatched existing checkouts and writes a
sibling `*.robotactile-install.json` receipt. The IsaacLab and cuRobo scripts
additionally require the verified Isaac Sim runtime and use its `python.sh`.

## Readiness and distribution boundary

```bash
robotactile integrations list
robotactile deployment doctor --profile act-univtac
robotactile integrations doctor --model act
robotactile integrations doctor --model n0_twam
```

The official N0 source, websocket protocol, and post-trained checkpoint are
registered. Readiness still requires a task-specific normalizer/config,
reachable live server, and a hash-matched request. Source or weight
availability alone does not establish inference, closed-loop execution, or
task success.

No external checkout, model weight, dataset, Isaac archive, secret, or
generated receipt is copied into the RoboTactile wheel. See
[Third-party notices](../THIRD_PARTY_NOTICES.md) for the distribution summary
and [Deployment layout](deployment_layout.md) for storage ownership.
