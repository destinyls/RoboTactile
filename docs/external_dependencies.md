# External Dependencies

RoboTactile's Apache-2.0 wheel does not vendor simulator, model, or upstream
task repositories. `integrations/integrations.lock.json` is the authority for
repository URL, exact commit, install entry point, source directory, and release
readiness. `integrations/act_artifacts.lock.json` separately freezes the ACT
Hugging Face revision, file inventory, sizes, and SHA256 values. Repository-root
and nested file-license scopes are expanded in this document and
`THIRD_PARTY_NOTICES.md`; a root SPDX identifier does not erase a more specific
nested notice.

## Frozen inventory

| Role | Immutable source | License at the pinned commit | Default directory | Installer |
|---|---|---|---|---|
| UniVTAC task/simulator and official ACT source | [UniVTAC `05bcd3e`](https://github.com/univtac/UniVTAC/tree/05bcd3edb92237107efa40105292a24f1a9fd761) | Apache-2.0 repository root; `policy/ACT` MIT; `policy/ACT/detr` Apache-2.0 | `deployment/sources/UniVTAC` | `integrations/install_univtac.sh` |
| Official UniVTAC ACT artifacts | [`byml/UniVTAC` `172331d`](https://huggingface.co/datasets/byml/UniVTAC/tree/172331dbbce95bc04c3e59b22f32dc72ba5561ae/checkpoints) | Upstream dataset card declares MIT; review exact artifact terms before redistribution | `deployment/artifacts/models/act` | `scripts/act/install_official_artifacts.py` |
| Legacy ACTStrict compatibility source (not the first-class `act` integration) | [WorldArena `e295378`](https://github.com/WorldArena2/WorldArena-2.0/tree/e295378c702b2e87617ebddcad193be3608e00c3) | MIT repository root; `visual-tactile_world_model_pipeline` has a separate Apache-2.0 notice | `deployment/sources/WorldArena` | legacy-only `integrations/install_act_runtime.sh` |
| Dream-Tac source | [Dream-Tac `14bab51`](https://github.com/LYFCLOUDFAN/Dream-Tac/tree/14bab51d6862fd07124745c55cd395ea5caa9fd3) | Apache-2.0 at repository root; review conflicting file-level notices | `deployment/sources/Dream-Tac` | `integrations/install_dream_tac.sh` |
| FTP-1 policy source | [ftp1-policy `89fa681`](https://github.com/michaelyuancb/ftp1-policy/tree/89fa681d6c014cce28300946b7526db808e0b1c1) | Apache-2.0 source license | `deployment/sources/ftp1-policy` | `scripts/ftp1_policy/install_official_runtime.sh` |
| FTP-1 UniVTAC checkpoints | [ftp1_univtac_finetune `620ac69`](https://huggingface.co/MJJJJ1064/ftp1_univtac_finetune/tree/620ac69b4fffd2341300cfef1b1d224d56710ed3) | Upstream model terms; no Apache-2.0 weight claim | `deployment/artifacts/models/ftp1_policy` | pinned `hf download` command |
| N0-TWAM source | [N0-TWAM `c43a216`](https://github.com/neoteai/N0-TWAM/tree/c43a2160dd31c449d92b28eab52c0e2f09e4738a) | CC-BY-NC-SA-4.0 | `deployment/sources/N0-TWAM` | `scripts/n0_twam/install_official_runtime.sh` |
| N0-TWAM base weights | [n0-twam-base `dafcb05`](https://huggingface.co/NeoteAI/n0-twam-base/tree/dafcb053902cc7a51310780fdfdb71ff8e79c0be) | CC-BY-NC-SA-4.0 | `deployment/artifacts/models/n0_twam/base` | `scripts/n0_twam/prepare_official_artifacts.py` |
| N0-TWAM UniVTAC delta weights | [n0-twam-univtac-delta `7694e63`](https://huggingface.co/NeoteAI/n0-twam-univtac-delta/tree/7694e63707a8c9e69e1a1242c4ed74ee39b7bb51) | Apache-2.0 | `deployment/artifacts/models/n0_twam/univtac-delta` | `scripts/n0_twam/prepare_official_artifacts.py` |
| N0-VTLA source | [N0-VTLA `03a0ce4`](https://github.com/neoteai/N0-VTLA/tree/03a0ce4d7091ca2354864796770715aa212601b7) | CC-BY-SA-4.0 | `deployment/sources/N0-VTLA` | `integrations/install_n0_vtla.sh` |
| N0-VTLA UniVTAC checkpoint | [n0_VTLA_insert_hole `73a514c`](https://huggingface.co/NeoteAI/n0_VTLA_insert_hole/tree/73a514c015c6745a14a3efdca92f25c6cfab5eb5) | Upstream model terms, including Gemma terms | `deployment/artifacts/models/n0_vtla/checkpoint` | pinned `hf download` command |
| IsaacLab v2.1.1 | [IsaacLab `90b79bb`](https://github.com/isaac-sim/IsaacLab/tree/90b79bb2d44feb8d833f260f2bf37da3487180ba) | BSD-3-Clause | `deployment/sources/IsaacLab` | `scripts/live_univtac/install_isaaclab_v2_1_1.sh` |
| cuRobo v0.7.7 | [cuRobo `0a50de1`](https://github.com/NVlabs/curobo/tree/0a50de1ba72db304195d59d9d0b1ed269696047f) | NVIDIA non-commercial (`LicenseRef-NVIDIA-NonCommercial`) | `deployment/sources/curobo` | `scripts/live_univtac/install_curobo_v0_7_7.sh` |
| Modified UIPC build tool | [vcpkg `ce613c4`](https://github.com/microsoft/vcpkg/tree/ce613c41372b23b1f51333815feb3edd87ef8a8b) | MIT | `deployment/sources/vcpkg-2025.04.09` | `scripts/live_univtac/install_tacex_uipc_univtac.sh` |

The cuRobo license above is the license in the pinned v0.7.7 commit; it must
not be replaced with the license of a newer branch. N0-TWAM and cuRobo impose
non-commercial restrictions. Read their upstream license files before use or
redistribution.

The first-class ACT adapter is source-bound only to the UniVTAC row above and
loads `policy/ACT/act_policy.py`. The WorldArena row documents an older
ACTStrict `policy_best.ckpt` compatibility path; it is not an additional source
requirement for official UniVTAC ACT. The repository-level and nested license
notices are listed separately because neither scope should be inferred from the
other.

The official ACT release contains the shared `encoder.pth` and, for each of
eight tasks, `policy_last.ckpt` plus `dataset_stats.pkl` under both `univtac`
and `vision_only`. RoboTactile does not vendor these files in its wheel. The
installer fetches only revision
`172331dbbce95bc04c3e59b22f32dc72ba5561ae`, verifies it against the tracked
release lock, and writes a no-clobber installation receipt. Upstream
`metadata.json` and `log.log` are optional reference material, not local
RoboTactile results.

Dream-Tac's pinned repository has an Apache-2.0 root license, while selected
configuration files retain NVIDIA proprietary/confidential header text.
RoboTactile records the root license in the lock but does not override those
file-level notices. Review the exact upstream files before redistribution and
seek upstream clarification where the notices conflict. The integration keeps
the checkout external, imports no upstream source into the wheel, and does not
redistribute a checkpoint.

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
python scripts/act/install_official_artifacts.py \
  --artifact-root "$PWD/deployment/artifacts/models/act"
bash integrations/install_dream_tac.sh
bash integrations/install_n0_vtla.sh
bash scripts/ftp1_policy/install_official_runtime.sh --root "$PWD/deployment"
bash scripts/n0_twam/install_official_runtime.sh --root "$PWD/deployment"
```

Do not run `integrations/install_act_runtime.sh` for the first-class ACT path.
That script only preserves the legacy WorldArena/ACTStrict checkout contract
for old workspaces and does not install the official ACT model or its Python
dependencies.

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
robotactile integrations doctor --model dream_tac
robotactile integrations doctor --model ftp1_policy --task insert_hole
robotactile integrations doctor --model n0_twam
robotactile integrations doctor --model n0_vtla --task insert_hole
```

Official ACT readiness requires the pinned UniVTAC checkout, hash-matched
official or separately trained model artifacts, and
`artifacts/models/act/configs/<task>/<profile>/integration_config.json`. ACT is
loaded in-process by Isaac Sim's bundled Python; it has no separate official
model-server runtime. Preserve Isaac's Torch/Torchvision stack rather than
installing the upstream ACT conda environment over it.

The upstream release does not expose sufficient split provenance to prove that
frozen40 was excluded from training. Use `ACT-official` for results produced
with the pinned release and reserve `ACT-train759` for a separately retrained,
data-manifest-bound checkpoint. Artifact availability alone does not establish
runtime loading, closed-loop execution, or Success Rate.

The official N0-TWAM websocket, N0-VTLA ZMQ, and FTP-1 ZMQ protocols plus their
released checkpoints are registered. Readiness still requires a task-specific
normalizer/config, reachable live server, and a hash-matched request. Source or
weight availability alone does not establish inference, closed-loop execution,
or task success.

Dream-Tac is intentionally `release_ready=false`. As of 2026-08-30, its
official documented channels did not expose a downloadable checkpoint, and no
UniVTAC task-aligned checkpoint was public. A user-supplied checkpoint,
dataset statistics, and T5 embeddings can be content-addressed locally, but
the current upstream HTTP `/infer` path also omits the paper's CASA inference
gate. Consequently source installation, configuration, and adapter smoke may
establish CODE/contract readiness only; they do not establish OFFLINE model
parity, CLOSED-LOOP evaluation, or an OFFICIAL Dream-Tac result.

No external checkout, model weight, dataset, Isaac archive, secret, or
generated receipt is copied into the RoboTactile wheel. See
[Third-party notices](../THIRD_PARTY_NOTICES.md) for the distribution summary
and [Deployment layout](deployment_layout.md) for storage ownership.
