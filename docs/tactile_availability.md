# A1/A2: structural absence and explicit input adaptation

A1 removes continuous tactile payloads; A2 removes frames on a seeded schedule.
Both remain `payload=None` in the fault delivery trace, not black RGB or T2 freeze.

| Mode | Model input | Availability |
|---|---|---|
| `required` (unchanged default) | Both actual tactile images required | A1/A2 unsupported |
| `native_missing_v1` | Omit only unavailable named slots | N0-VTLA integration |
| `zero_fill_v1` | Fill missing slots with raw uint8 RGB zeros before preprocessing | Shared live compatibility path for N0-TWAM, N0-VTLA, FTP-1 and Dream-Tac |

Shared code support does not establish closed-loop validation for all four models.
Native omission is not silently emulated for other model families. Keep native and
zero results separate: the latter evaluates a model plus a fixed input strategy.
Mode/shape are bound into request and artifact content hashes. Legacy default
serialization is unchanged; explicit retrained groups get distinct system/config IDs.

## N0-VTLA baseline

The opt-in server overlay leaves the external pinned checkout unchanged. Fixed
wire slot mapping prevents a missing left image from shifting right into left.
Client/server protocol mismatch is an error.

- Native: each slot's first actually available frame establishes its baseline,
  only from that query onward (`first_available_per_slot_v1`). Before then omit
  that slot. Returning frames reuse an established baseline.
- Zero-fill: zeros are actual model inputs. An initially missing slot establishes
  a black baseline which stays black after recovery
  (`first_delivered_including_fill_v1`). This is part of the control's semantics.
- Never initialize using a hidden Clean, rest-reference or future frame. Reset
  clears baselines and retains the existing RNG reset.
- `<server-receipt-stem>.availability.jsonl` records query/slot availability,
  baseline establishment and input/baseline hashes, without raw images. A
  collection witness is not proof of completed inference: also check action traces.

Complete inputs preserve the old image/baseline pairing. That does not assert
bitwise-identical stochastic GPU trajectories.

## Prepare and run only A1/A2

Use an existing retrained checkpoint binding. Run from RoboTactile with its Python
runtime. The code/package supplied below must contain this implementation in both
the Isaac and model-server process environments; no system Python/CUDA change is
needed. Paths are explicit placeholders, not new dependencies to install.

```bash
python -m scripts.retrained_evaluation.availability_campaign prepare \
  --binding /absolute/path/to/existing/binding.json \
  --task put_bottle_in_shelf --seeds 0 1 2 \
  --mode native_missing_v1 --output /absolute/path/to/new-native-supplement

python -m scripts.retrained_evaluation.availability_campaign run \
  --output /absolute/path/to/new-native-supplement \
  --code /absolute/path/to/RoboTactile \
  --package /absolute/path/to/RoboTactile/src
```

For the zero control, prepare **another output directory** with
`--mode zero_fill_v1 --zero-shape H W 3`. Replace H/W with native tactile dimensions
from the deployment contract; do not guess or inspect hidden Clean pixels after
failure. Present images must match the configured shape, without resizing.
Run protocols serially, never simultaneously on the same GPU/endpoint.

Each supplement runs **Clean + A1 + A2 per seed** from the existing paired-snapshot
runner, retaining the official success predicate, action chunk and full-episode
S5 schedule. The new Clean is a control for the changed serving protocol; it
does not overwrite/relabel historical Clean. The previous 12 fault conditions
are not repeated. Prepare is non-executing. `started.json` prevents duplicate
launches after interruption; infrastructure failure requires explicit handling.

For custom scope, the existing group entrypoint accepts `binding.evaluation`:

```json
{
  "tactile_availability_mode": "native_missing_v1",
  "operators": ["A1_stream_absence", "A2_frame_erasure"],
  "sensor_slots": ["left", "right"],
  "fault_window_mode": "full_episode_v1",
  "severity_registries": ["optical_marker_extreme_v1"],
  "severity_level": 5,
  "capture_profile": "paper_full_v1"
}
```

Zero mode additionally requires `tactile_zero_shape: [H, W, 3]`. Standard
`optical_marker_v1` supports S1-S5; extreme is S5-only. Reuse exactly the same
seed/slot/schedule across models and modes. At extreme A2 dose, every inference
query can fall on an erased frame even though some sensor frames survived.
Report actual query-time exposure; never alter the schedule after seeing results.

## A2 endpoint and interrupted-run recovery

Legacy A2 requires an observed clean frame after its last planned erasure. That
strict default is unchanged. For full-episode online experiments, explicitly use
`--a2-end-policy episode_censored_v1` when preparing the supplement (or set
`binding.evaluation.a2_end_policy`). The field is bound into the fault manifest.
Every observed payload and unmasked frame is still validated. When the episode
ends during erasure, the report records `a2_resume_status=right_censored`: recovery
after termination is unobservable, not a demonstrated sensor recovery or failure.
The official task-success predicate is unchanged. This protocol neither adds a
clean final frame nor changes the erasure dose, schedule, or episode horizon.

Finalization, artifact export and independent artifact reload share the same
validator. Failures are recorded in `execution_failure.json` before Isaac closes;
process exit code 0 alone cannot mark a group complete.

If Clean and A1 have valid published artifacts but A2 has none, a no-clobber
successor may reference the old results and execute only A2. It verifies the
original request/artifact hashes and requires an exact historical Clean reset
witness before inference. The resulting `recovery_receipt.json` declares
**cross-process reference recovery**, not same-process paired replay. Adopted
results retain their original source files and code provenance. Never overwrite
them, rerun a scored failure, or merge native and zero protocols. An unexported
attempt remains listed separately with unknown outcome.

When that historical reference cannot be reproduced, do not relax the digest,
native-step or joint checks just to admit the episode. Start a separate matched
Clean/A2 pair in one Isaac process, with its own canonical snapshot:

```bash
python -m scripts.retrained_evaluation.a2_same_process prepare \
  --binding /absolute/path/to/n0-vtla/binding.json \
  --output /absolute/path/to/new-clean-a2-pair --seed 0
python -m scripts.retrained_evaluation.a2_same_process launch \
  --output /absolute/path/to/new-clean-a2-pair \
  --code /absolute/path/to/frozen/code --package /absolute/path/to/private/package
```

This targeted entrypoint runs N0-VTLA `put_bottle_in_shelf`, native omission,
full-episode extreme A2, and a **new** matched Clean control. It does not rerun A1,
reuse the failed cross-process reset, replace historical scores, or claim a
three-seed aggregate. `summary.json` is published before RGB/tactile video export.
Task timeout is a valid negative result; reset/injection crashes are not model SR.

## Shelf-only 50-to-8 replanning ablation

The default N0-VTLA protocol still executes all 50 predicted actions. To opt into
`receding_horizon_50x8_v1`, prepare a **new** same-process Clean/A2 pair:

```bash
python -m scripts.retrained_evaluation.a2_same_process prepare \
  --binding /absolute/path/to/n0-vtla/binding.json \
  --output /absolute/path/to/new-clean-a2-50x8-pair --seed 0 \
  --execution-profile receding_horizon_50x8_v1
python -m scripts.retrained_evaluation.a2_same_process launch \
  --output /absolute/path/to/new-clean-a2-50x8-pair \
  --code /absolute/path/to/frozen/code --package /absolute/path/to/private/package
```

Both frozen code and private package must include this profile before launch.
Preparation performs no GPU execution. Profile selection is fixed at preparation;
launch cannot mutate an existing plan. General group bindings can equivalently set
`evaluation.n0_vtla_execution_profile` to `receding_horizon_50x8_v1`.

- Only retrained N0-VTLA `put_bottle_in_shelf` at 10 Hz accepts this profile.
  Other models/tasks reject it rather than silently changing their execution.
- Prediction stays `[50, 32]` on the wire and `[50, 8]` at the adapter. Execute
  the first eight rows, discard the remaining 42, then query using the newest
  delivered RGB/tactile/proprio. Do not reset baseline/RNG between chunks.
- The task budget remains 300 actions, not six queries: up to 38 predictions,
  37 eight-action prefixes and one four-action terminal remainder. Replanning
  interval is 0.8 simulation seconds; inference wall time is additional cost.
- The checkpoint, official success predicate, physics cadence, reset pairing and
  full-episode A2 erasure schedule are unchanged. Profile is bound into config,
  request and result identity; default serialization omits the new optional field.
- Upstream recommends full-chunk execution because tail gripper actions can be
  lost under repeated replanning. Treat this as a deployment-protocol ablation,
  inspect gripper execution and placement errors, and do not pool SR with 50-to-50.
- Extreme A2's surviving offsets (25, 75, ..., 275) also miss every 8-step query.
  Report this as complete query-time absence, not demonstrated intermittent
  recovery. Never reschedule retained frames after seeing success outcomes.

## Reporting

Per-seed reports and real RGB/tactile videos retain model failure as well as
success. Report success/eligible, invalid, unsupported, missing and coverage.
Unsupported/invalid is not a fabricated failed episode. Do not compare averages
over different silently filtered condition sets; show the common supported set
and coverage. Native/zero tables never pool. Three seeds are diagnostic, not
paper-level statistical replication. Old artifacts remain unchanged.
