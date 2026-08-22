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
| N0-TWAM | [N0-TWAM `9036c13`](https://github.com/destinyls/N0-TWAM/tree/9036c130409f8cf5494b12489fea339f7213b9d6) | CC-BY-NC-SA-4.0 | `deployment/sources/N0-TWAM` | `integrations/install_n0_twam.sh` |
| IsaacLab v2.1.1 | [IsaacLab `90b79bb`](https://github.com/isaac-sim/IsaacLab/tree/90b79bb2d44feb8d833f260f2bf37da3487180ba) | BSD-3-Clause | `deployment/sources/IsaacLab` | `scripts/live_univtac/install_isaaclab_v2_1_1.sh` |
| cuRobo v0.7.7 | [cuRobo `0a50de1`](https://github.com/NVlabs/curobo/tree/0a50de1ba72db304195d59d9d0b1ed269696047f) | NVIDIA non-commercial (`LicenseRef-NVIDIA-NonCommercial`) | `deployment/sources/curobo` | `scripts/live_univtac/install_curobo_v0_7_7.sh` |

The cuRobo license above is the license in the pinned v0.7.7 commit; it must
not be replaced with the license of a newer branch. N0-TWAM and cuRobo impose
non-commercial restrictions. Read their upstream license files before use or
redistribution.

TacEx/UIPC are not independently listed in the lock because this release has
not frozen a separately verified URL, commit, license, and direct installer for
them. If the pinned UniVTAC runtime requires them transitively, their absence is
a deployment readiness failure, not permission to fetch a mutable branch.

## Install exact source checkouts

From the repository root, the generic integrations require no destination
argument:

```bash
robotactile deployment init
bash integrations/install_univtac.sh
bash integrations/install_act_runtime.sh
bash integrations/install_n0_twam.sh
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

The ACT runtime and N0-TWAM lock entries remain `release_ready=false` until
reviewed integration-side changes and real serving artifacts are published at
immutable public commits. N0-TWAM also remains fail-closed until a production
typed gateway handshake is registered. Source availability alone does not
establish live inference.

No external checkout, model weight, dataset, Isaac archive, secret, or
generated receipt is copied into the RoboTactile wheel. See
[Third-party notices](../THIRD_PARTY_NOTICES.md) for the distribution summary
and [Deployment layout](deployment_layout.md) for storage ownership.
