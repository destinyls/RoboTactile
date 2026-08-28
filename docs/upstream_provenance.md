# Upstream Provenance and Runtime Boundary

## N0-TWAM

- source URL: `https://github.com/neoteai/N0-TWAM.git`
- frozen commit: `c43a2160dd31c449d92b28eab52c0e2f09e4738a`
- default checkout: `deployment/sources/N0-TWAM`
- base weights: `NeoteAI/n0-twam-base@dafcb053902cc7a51310780fdfdb71ff8e79c0be`
- UniVTAC weights: `NeoteAI/n0-twam-univtac-delta@7694e63707a8c9e69e1a1242c4ed74ee39b7bb51`

The benchmark consumes the released server's `(20,2,12)` rot6d chunk. The
single-arm adapter converts channels 0--9 to
`ee8_absolute = [xyz, quat(wxyz), gripper]`; the unused second-arm channels are
not executed. The first cold chunk skips frame zero and executes 12 slots;
subsequent chunks execute 24. N0 inference is stateful: reset, infer, execute,
observe, and KV-cache re-ground are serialized without automatic retries.
Faults affect both infer observations and the 4/8 observed grounding keyframes.
The adapter uses the official serve keys
`observation.images.top`, `observation.images.wrist_l`,
`observation.images.tactile_a`, and `observation.images.tactile_b`. A commit is
not complete until the official server returns. It carries the raw native
action and exact chunk-start EE anchor; replaying an ambiguous commit is
forbidden because it could advance the model time axis twice. Local source,
checkpoint, config, task normalizer, prompt, and assembled-bundle hashes must
match before simulator allocation.

Frozen N0-TWAM requires two tactile payloads. A1/A2 therefore return
`unsupported_contract` for this system; they are not converted to black images.

## UniVTAC

- default checkout: `deployment/sources/UniVTAC`
- branch: `main`
- frozen commit: `05bcd3edb92237107efa40105292a24f1a9fd761`

The adapter consumes raw UniVTAC `step`, vision, tactile RGB, EE pose and
joint9. The released N0 checkpoint's bundled exact converter reads the
marker-less HDF5 `rgb` field, so the N0 EE adapter uses UniVTAC `rgb`; the
legacy qpos path retains the registry's `rgb_marker` payload. The pinned
collector writes simulator RGB arrays through OpenCV's BGR JPEG interface and
the checkpoint converter decodes them with PIL; the live N0 profile applies
one R/B reversal to match that decoded training domain. The adapter rejects
caller/raw-step disagreement, binds
the two physical
sensor identities, far-plane value, calibration ID, and calibration-config
hash in an explicit alias manifest, derives logical source/delivery time from
the 120-Hz simulation step, and maps actions to absolute qpos8 (ACT) or
absolute EE8 (N0) only at the model boundary. The reset return is not trusted; an
online backend must explicitly observe after reset.

N0 EE8 endpoints use the source-bound
`robotactile_n0_training_60hz_ee_v1` execution contract. UniVTAC's cuRobo
planner still supplies collision/IK feasibility, but only its final joint
target is applied before exactly two 120 Hz physics ticks and one endpoint
render. This matches the released HDF5 `save_frequency=2` collection cadence.
The pinned upstream variable-waypoint EE loop is preserved as
`univtac_stock_ee_v1` for explicit reference diagnostics, not as the default
N0 Clean executor.

The complete measured `joint9` retains both Franka finger joints. UniVTAC sends
one shared scalar gripper command, but its pinned observation implementation
(`get_gripper_qpos`) reads `panda_finger_joint1`; the model-visible qpos8 follows
that rule. The two measured finger positions are not forced equal because
contact physics may make them differ even when their command target is shared.

The current macOS workspace does not provide Linux/NVIDIA Isaac Sim 4.5,
IsaacLab 2.1.1, cuRobo, modified TacEx/UIPC, model checkpoints, or task assets.
Consequently local results are offline contract/replay evidence only.

The official websocket metadata is intentionally minimal. RoboTactile binds
source and artifacts locally and treats endpoint reachability as a separate
runtime check. Unit tests exercise conversions and state machines only; they
are not live N0 inference evidence.

The canonical five-source inventory, pinned commit hyperlinks, licenses, and
install commands are maintained in
[External dependencies](external_dependencies.md). The writable directory
contract is maintained in [Deployment layout](deployment_layout.md).
