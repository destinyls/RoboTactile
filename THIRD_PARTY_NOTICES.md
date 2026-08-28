# Third-Party Notices

RoboTactile-owned source is licensed under Apache-2.0. External integrations
are installed separately and remain governed by their upstream licenses.

| Integration | Repository | Frozen commit | License |
|---|---|---|---|
| UniVTAC | [source](https://github.com/univtac/UniVTAC/tree/05bcd3edb92237107efa40105292a24f1a9fd761) | `05bcd3edb92237107efa40105292a24f1a9fd761` | Apache-2.0 |
| ACT runtime | [source](https://github.com/WorldArena2/WorldArena-2.0/tree/e295378c702b2e87617ebddcad193be3608e00c3) | `e295378c702b2e87617ebddcad193be3608e00c3` | Apache-2.0 |
| N0-TWAM source and base model | [source](https://github.com/neoteai/N0-TWAM/tree/c43a2160dd31c449d92b28eab52c0e2f09e4738a) | `c43a2160dd31c449d92b28eab52c0e2f09e4738a` | CC-BY-NC-SA-4.0 |
| N0-TWAM UniVTAC delta checkpoint | [model](https://huggingface.co/NeoteAI/n0-twam-univtac-delta) | `7694e63707a8c9e69e1a1242c4ed74ee39b7bb51` | Apache-2.0 |
| IsaacLab v2.1.1 | [source](https://github.com/isaac-sim/IsaacLab/tree/90b79bb2d44feb8d833f260f2bf37da3487180ba) | `90b79bb2d44feb8d833f260f2bf37da3487180ba` | BSD-3-Clause |
| cuRobo v0.7.7 | [source](https://github.com/NVlabs/curobo/tree/0a50de1ba72db304195d59d9d0b1ed269696047f) | `0a50de1ba72db304195d59d9d0b1ed269696047f` | NVIDIA non-commercial license |
| vcpkg build recipes | [source](https://github.com/microsoft/vcpkg/tree/ce613c41372b23b1f51333815feb3edd87ef8a8b) | `ce613c41372b23b1f51333815feb3edd87ef8a8b` | MIT |

N0-TWAM's Creative Commons license includes non-commercial and share-alike
conditions. The pinned cuRobo v0.7.7 license also limits use to research or
evaluation purposes. Neither is represented as an OSI-approved open-source
license. No third-party runtime source or model artifact is copied into the
RoboTactile wheel. The source distribution contains two minimal MIT-licensed
vcpkg overlay recipes for pinned `cpptrace` and `tinygltf` sources.

The official N0 source and released checkpoint are consumed without vendoring
or patching. ACT release readiness remains tracked separately in the external
integration lock.
