# Upstream Provenance and Runtime Boundary

## N0-TWAM

- source URL: `https://github.com/destinyls/N0-TWAM.git`
- requested branch: `UniVTAC-PostTraining`
- frozen commit: `9036c130409f8cf5494b12489fea339f7213b9d6`
- checkout: `paper/N0-TWAM`

The first network clone was interrupted after repository initialization and had
no HEAD. The final checkout was reconstructed from a previously verified clone
of the same GitHub repository, mapped to the exact remote-tracking ref, reset to
the GitHub URL, and checked with `git fsck --full --no-dangling`. Local HEAD and
the frozen `origin/UniVTAC-PostTraining` ref match.

The benchmark consumes `qpos8_next_step`, not the legacy 20-D delta-EEF client.
N0 inference is stateful: reset, infer, execute, observe, and commit are a
serialized transaction. Faults must affect both infer observations and committed
keyframes. The adapter uses the upstream Track 3.1 serve keys
`observation.images.top`, `observation.images.wrist_l`,
`observation.images.tactile_a`, and `observation.images.tactile_b`. A commit is
not considered complete until a matching typed ACK; reset/infer/commit use
episode-bound transaction IDs and exact request digests. Each record must also
match the reset episode, task, and seed. Commit carries the exact action and
qpos8 anchor from the preceding acknowledged infer, and the next infer starts
from the last acknowledged committed keyframe. Failure puts the adapter in an
aborted state requiring hard reset. Duplicate commit is forbidden. Live
handshake construction also checks checkpoint/config digests against the trial
manifest and carries frozen 8-D joint/gripper bounds. The audited upstream
server does not yet implement this typed envelope, so this remains the required
server-patch contract rather than a live transport result.

Frozen N0-TWAM requires two tactile payloads. A1/A2 therefore return
`unsupported_contract` for this system; they are not converted to black images.

## UniVTAC

- audited sibling checkout: `../UniVTAC`
- branch: `main`
- frozen commit: `05bcd3edb92237107efa40105292a24f1a9fd761`

The adapter consumes the raw UniVTAC `step`, two `rgb_marker` images, vision,
and joint9. It rejects caller/raw-step disagreement, binds the two physical
sensor identities, far-plane value, calibration ID, and calibration-config
hash in an explicit alias manifest, derives logical source/delivery time from
the 120-Hz simulation step, and maps actions to
absolute qpos8 only at the model boundary. The reset return is not trusted; an
online backend must explicitly observe after reset.

The current macOS workspace does not provide Linux/NVIDIA Isaac Sim 4.5,
IsaacLab 2.1.1, cuRobo, modified TacEx/UIPC, model checkpoints, or task assets.
Consequently local results are offline contract/replay evidence only.

The current adapter is a typed client-side contract. The audited N0-TWAM branch
does not yet emit the required `robotactile-n0-v1` handshake metadata and has no
registered qpos8 serve configuration, so local unit tests must not be described
as a live N0 server integration.
