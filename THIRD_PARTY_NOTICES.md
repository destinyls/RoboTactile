# Third-Party Notices

RoboTactile-owned source is licensed under Apache-2.0. External integrations
are installed separately and remain governed by their upstream licenses.

| Integration | Repository | Frozen commit | License |
|---|---|---|---|
| UniVTAC simulator and official ACT source | [source](https://github.com/univtac/UniVTAC/tree/05bcd3edb92237107efa40105292a24f1a9fd761) | `05bcd3edb92237107efa40105292a24f1a9fd761` | Apache-2.0 repository root; `policy/ACT` MIT; `policy/ACT/detr` Apache-2.0 |
| Official UniVTAC ACT artifacts | [dataset](https://huggingface.co/datasets/byml/UniVTAC/tree/172331dbbce95bc04c3e59b22f32dc72ba5561ae/checkpoints) | `172331dbbce95bc04c3e59b22f32dc72ba5561ae` | Upstream dataset card declares MIT; review the exact artifact terms before use or redistribution |
| Legacy WorldArena/ACTStrict compatibility source | [source](https://github.com/WorldArena2/WorldArena-2.0/tree/e295378c702b2e87617ebddcad193be3608e00c3) | `e295378c702b2e87617ebddcad193be3608e00c3` | MIT repository root; `visual-tactile_world_model_pipeline` includes a separate Apache-2.0 notice |
| Dream-Tac source | [source](https://github.com/LYFCLOUDFAN/Dream-Tac/tree/14bab51d6862fd07124745c55cd395ea5caa9fd3) | `14bab51d6862fd07124745c55cd395ea5caa9fd3` | Apache-2.0 root license; conflicting file-level notices require review |
| N0-TWAM source and base model | [source](https://github.com/neoteai/N0-TWAM/tree/c43a2160dd31c449d92b28eab52c0e2f09e4738a) | `c43a2160dd31c449d92b28eab52c0e2f09e4738a` | CC-BY-NC-SA-4.0 |
| N0-TWAM UniVTAC delta checkpoint | [model](https://huggingface.co/NeoteAI/n0-twam-univtac-delta) | `7694e63707a8c9e69e1a1242c4ed74ee39b7bb51` | Apache-2.0 |
| N0-VTLA source | [source](https://github.com/neoteai/N0-VTLA/tree/03a0ce4d7091ca2354864796770715aa212601b7) | `03a0ce4d7091ca2354864796770715aa212601b7` | CC-BY-SA-4.0 |
| N0-VTLA UniVTAC checkpoint | [model](https://huggingface.co/NeoteAI/n0_VTLA_insert_hole/tree/73a514c015c6745a14a3efdca92f25c6cfab5eb5) | `73a514c015c6745a14a3efdca92f25c6cfab5eb5` | Upstream model terms, including Gemma terms |
| FTP-1 policy source | [source](https://github.com/michaelyuancb/ftp1-policy/tree/89fa681d6c014cce28300946b7526db808e0b1c1) | `89fa681d6c014cce28300946b7526db808e0b1c1` | Apache-2.0 |
| FTP-1 UniVTAC checkpoints | [model](https://huggingface.co/MJJJJ1064/ftp1_univtac_finetune/tree/620ac69b4fffd2341300cfef1b1d224d56710ed3) | `620ac69b4fffd2341300cfef1b1d224d56710ed3` | Upstream does not specify a weight license; Gemma terms may also apply |
| IsaacLab v2.1.1 | [source](https://github.com/isaac-sim/IsaacLab/tree/90b79bb2d44feb8d833f260f2bf37da3487180ba) | `90b79bb2d44feb8d833f260f2bf37da3487180ba` | BSD-3-Clause |
| cuRobo v0.7.7 | [source](https://github.com/NVlabs/curobo/tree/0a50de1ba72db304195d59d9d0b1ed269696047f) | `0a50de1ba72db304195d59d9d0b1ed269696047f` | NVIDIA non-commercial license |
| vcpkg build recipes | [source](https://github.com/microsoft/vcpkg/tree/ce613c41372b23b1f51333815feb3edd87ef8a8b) | `ce613c41372b23b1f51333815feb3edd87ef8a8b` | MIT |

N0-TWAM's Creative Commons license includes non-commercial and share-alike
conditions. The pinned cuRobo v0.7.7 license also limits use to research or
evaluation purposes. Neither is represented as an OSI-approved open-source
license. No third-party runtime source or model artifact is copied into the
RoboTactile wheel. The source distribution contains two minimal MIT-licensed
vcpkg overlay recipes for pinned `cpptrace` and `tinygltf` sources.

RoboTactile's first-class `act` integration imports only the pinned UniVTAC
`policy/ACT/act_policy.py` implementation. Its nested MIT notice and the DETR
Apache-2.0 notice remain applicable within their respective upstream scopes.
The WorldArena entry records the older ACTStrict `policy_best.ckpt`
compatibility path; it is not an additional source for official UniVTAC ACT.

The official `byml/UniVTAC` artifact release provides `policy_last.ckpt`,
`dataset_stats.pkl`, and the shared tactile `encoder.pth` outside the
RoboTactile wheel. RoboTactile pins the exact revision above and verifies its
file inventory through `integrations/act_artifacts.lock.json`; it does not
vendor or relicense those files. The upstream dataset card declares MIT, but
users remain responsible for reviewing the exact artifact terms before use or
redistribution. Upstream logs and metadata are reference material and are not
represented as locally executed RoboTactile results.

N0-VTLA source is CC-BY-SA-4.0. Its released weights are distributed
separately and may incorporate Gemma-derived components governed by additional
model terms. Users must review those terms before downloading or redistributing
the checkpoint.

FTP-1 source at the frozen commit is Apache-2.0. The separate checkpoint
repository publishes six task-specific UniVTAC checkpoints but does not declare
a checkpoint-weight license in its repository metadata. The serving artifacts
also reference Gemma-derived components. Users must review the upstream files
and [Gemma Terms of Use](https://ai.google.dev/gemma/terms) before downloading,
using, or redistributing those weights. RoboTactile does not interpret the
source license as a license grant for the checkpoint files.

Dream-Tac's repository root declares Apache-2.0, but selected configuration
files at the pinned commit retain NVIDIA proprietary/confidential header text.
RoboTactile does not interpret the root license as cancelling those notices.
The exact upstream checkout must be reviewed before redistribution and any
conflict should be clarified with the upstream maintainers. Dream-Tac source,
runtime dependencies, checkpoints, dataset statistics, and T5 embeddings all
remain external to the RoboTactile wheel.

As of 2026-08-30, the official documented Dream-Tac channels did not expose a
downloadable checkpoint. RoboTactile therefore accepts only an explicitly
user-supplied, hash-bound serving bundle and does not claim that the bundle is
an official or UniVTAC-aligned release.

The official N0 source and released checkpoint are consumed without vendoring
or patching. FTP-1 source, runtime dependencies, and weights are likewise kept
outside the RoboTactile wheel. Dream-Tac and legacy ACTStrict readiness remain
tracked separately in the external integration lock; the first-class ACT
source identity is the pinned UniVTAC checkout above.
