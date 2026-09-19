# Recorded N0-TWAM robustness evaluation

This path measures how the released N0-TWAM UniVTAC policy's 12-step EE8
prediction changes when one real recorded tactile observation is perturbed. It
is the fastest real-model experiment in RoboTactile and does not require Isaac
Sim. It is deliberately classified as `recorded_model_only_n0_v1`: predicted
actions are not executed, so this path never reports task success or a
closed-loop success rate.

## What is reused

The loader does not implement a second observation adapter. Every HDF5 frame is
decoded and passed through the production `convert_raw_observation` contract,
including canonical joint ordering, depth-based contact phase tracking, and the
marker-less GelSight `rgb` alias used by the released N0 checkpoint. Faults are
applied by the same 14-operator registry and must pass the normal delivery
validator before model inference.

The official N0 native contract requires both tactile streams. Therefore A1
and A2 are retained in the result matrix as `unsupported_contract`; the runner
does not replace missing input with black pixels. The other 12 operators are
executed normally.

## N0 chunk interpretation

The official network returns `[20, 2, 12]`. During a cold prediction, frame 0
is the conditioning frame and is not emitted. Frame 1 supplies 12 executable
EE8 rows. For the recorded target:

- H0 is aligned with the selected anchor frame;
- H1-H11 are future targets;
- there is no six-history/six-future split.

Artifacts report H0, H1-H11, and H0-H11 expert error separately, plus each
fault's H0-H11 drift relative to the clean prediction.

## Reference episode used by the paper example

The initial real episode is:

| Field | Value |
|---|---|
| Task | `lift_bottle` |
| Source | `deployment/artifacts/datasets/univtac_raw/lift_bottle/clean/90.hdf5` |
| Source SHA-256 | `223dbfe7f4781c95fe01e01fd9743652d1233e78488c4ec1dbf0ed927afe4f15` |
| Rest index | `0` |
| Registered fault start | `17` |
| Prediction anchor | `186` |
| Bilateral canonical release anchor for F6 | `280` |

The runner re-hashes the source and derives contact phases from frame 0 through
the selected anchor. The table is documentation, not a hash bypass.

## Training-versus-live observation parity

After a diagnostic or Clean live artifact exists, compare its actual stored
model-boundary arrays with the checkpoint's raw HDF5 domain:

```bash
python scripts/n0_twam/audit_observation_parity.py \
  --hdf5 deployment/artifacts/datasets/univtac_raw/lift_bottle/clean/90.hdf5 \
  --training-seed 90 \
  --live-artifact deployment/artifacts/live-univtac/<campaign>/lift_bottle/0000 \
  --checkpoint-converter deployment/artifacts/models/n0_twam/univtac-delta/norm/convert_univtac_single_rot6d.py \
  --teacher-forced-report deployment/outputs/diagnostics/n0-teacher-forced-history/lift_bottle-90-area-fix.json \
  --applied-channel-transform reverse_rgb \
  --output deployment/outputs/diagnostics/n0-observation-parity/<run>.json \
  --panel-output deployment/outputs/diagnostics/n0-observation-parity/<run>.png
```

The command reports nominal checkpoint FPS, physical HDF5/native-step cadence,
live native-step cadence when a v1.1 transition trace is present, RGB-order
witnesses, spatial high-frequency ratios, duplicate tactile frames, and
model-target endpoint tracking. A report is diagnostic evidence, not a success
receipt. Direct trajectory comparison is marked comparable only when the
training and live seeds and selected starts are identical.

## Pre-Clean simulator dynamic gate

Before starting the N0 server, freeze the exact `lift_bottle` experiment and
run one simulator-only action. This gate reuses the production HDF5 loader,
EE8 backend, released live RGB transform, and fixed-cadence cuRobo path; it
does not run an N0 prediction and cannot report task success.

After activating the repository-local RoboTactile environment:

```bash
python scripts/n0_twam/prepare_official_artifacts.py \
  --root deployment/artifacts/models/n0_twam \
  --task lift_bottle --device cuda --skip-download \
  --config-label source-c43a216

python scripts/n0_twam/freeze_experiment_lock.py \
  --integration-config deployment/artifacts/models/n0_twam/configs/lift_bottle-source-c43a216/integration_config.json \
  --univtac-root deployment/sources/UniVTAC \
  --n0-root deployment/sources/N0-TWAM \
  --wheel deployment/artifacts/wheels/<robotactile-wheel>.whl \
  --hdf5 deployment/artifacts/datasets/univtac_raw/lift_bottle/clean/90.hdf5 \
  --converter deployment/artifacts/models/n0_twam/univtac-delta/norm/convert_univtac_single_rot6d.py \
  --planned-live-artifact deployment/artifacts/live-univtac/n0-twam/lift_bottle/seed100 \
  --output deployment/requests/n0-twam/lift_bottle/experiment_lock.json
```

The command validates the pinned N0/UniVTAC Git checkouts and the existing
official artifact manifest, then binds the source manifest, wheel, checkpoint,
serve bundle, per-task normalizer, converter, HDF5, task registry, camera
configuration, and wrist-camera USD files. It also records seed 100 while
requiring the planned live artifact to be absent.

Run the probe once with the standalone Isaac Python:

```bash
deployment/runtime/isaac-sim-4.5.0/python.sh \
  scripts/live_univtac/probe_n0_dynamic_contract_isaac.py \
  --upstream-root deployment/sources/UniVTAC \
  --runtime-dir deployment/runtime/live-univtac \
  --hdf5 deployment/artifacts/datasets/univtac_raw/lift_bottle/clean/90.hdf5 \
  --output-dir deployment/outputs/n0-diagnostics/lift-bottle-dynamic-seed90 \
  --initial-seed 90 --exogenous-seed 20260825 \
  --antialiasing-mode TAA
```

The write-once bundle contains `probe.json`, exact `.npy` inputs, and
`comparison_panel.png`. It selects the nearest HDF5 EE8 row with a reserved
successor, executes that successor only when the 2 mm / 1 degree / 0.2 mm
state gate passes (otherwise it safely holds the live pose), and records:

- configured/effective USD camera facts plus frame and timestamp deltas;
- the empirically HDF5-matched TAA settings and static temporal variation;
- HDF5/live-pre/live-post Top, Wrist, tactile RGB and tactile depth;
- joint/EE/gripper state errors and task-object predicate witnesses;
- cuRobo status, proof that the stock variable-length `move()` loop was not
  used, physics-step delta 2, exactly one TAA render after both 120 Hz physics
  ticks, 60 Hz endpoints, and the derived 20 Hz keyframe cadence.

The endpoint-only render is part of the production N0 contract, not a display
optimization. Official UniVTAC collection uses `save_frequency=2`: it advances
two 120 Hz physics ticks, renders once, and stores one 60 Hz HDF5 row. Rendering
after the intermediate tick changes RTX temporal accumulation and is rejected
by the cadence gate.

The benchmark explicitly selects TAA for every live UniVTAC policy instead of
inheriting an IsaacLab default. This renderer contract is model-independent, so
EE8 policies such as N0-TWAM and QPOS8 policies such as FTP-1 receive the same
camera rendering. On the pinned headless stack, a controlled DLSS row-174 replay kept
robot, object, and tactile state aligned but produced strong speckle artifacts
and reduced Top/Wrist server-pixel PSNR to 23.33/15.98 dB. The matched TAA
replay reached 36.85/36.07 dB. Renderer selection is therefore bound to measured
training-pixel parity rather than inferred from a framework default.

Because UniVTAC HDF5 does not store the complete live task-object state, the
pixel metrics are explicitly labeled
`robot_state_matched_not_object_registered`; they are diagnostic and never
promoted to pixel-registered ground truth.

## Run

First start the pinned official server in one terminal. The fast path requires
at least 40 GB on every selected GPU; choose the GPU list for the local host.

```bash
bash scripts/n0_twam/serve_univtac.sh \
  --root "$PWD/deployment" \
  --task lift_bottle \
  --gpus 0 \
  --port 29601
```

Run the client in the repo-local runtime that contains `h5py`, Pillow,
`msgpack`, and `websockets`. After activating that runtime, the command uses
plain `python`:

```bash
python -m robotactile_benchmark.cli recorded-n0 \
  --hdf5 deployment/artifacts/datasets/univtac_raw/lift_bottle/clean/90.hdf5 \
  --integration-config deployment/artifacts/models/n0_twam/configs/lift_bottle/integration_config.json \
  --source-root deployment/sources/N0-TWAM \
  --task lift_bottle \
  --episode-id univtac-lift_bottle-clean-90 \
  --rest-index 0 \
  --fault-start-index 17 \
  --anchor-index 186 \
  --release-index 280 \
  --severity 3 \
  --output deployment/outputs/recorded-n0/lift_bottle-90-s3
```

With no `--operator`, all 14 registered operators are represented. Repeat
`--operator` to select a subset. Repeat `--severity` to request several levels;
the default is level 3. A full five-level run uses five explicit flags.

C1 swaps the same two streams at a single anchor, so its five exposure
fractions are not identifiable in a one-shot protocol. The artifact marks this
fact with `severity_identifiable_at_anchor=false`; exposure-dependent C1
severity belongs in the later streaming closed-loop experiment.

F6 is evaluated at release index 280, not at the contact anchor. The runner
loads the same source once through index 291, obtains a separate clean-release
prediction, and measures F6 against the complete release H0-H11 target. All
other executable operators use contact anchor 186.

## Artifact contract

The output is a no-clobber, content-addressed directory containing:

- `experiment.json` with source, policy, chunk, manifest, validation, and metric
  contracts;
- canonical `.npy` members for expert, clean, and fault predictions;
- `root_receipt.json` binding the exact inventory and hashes.

The evidence level remains recorded model inference even when all delivery
validators pass. Isaac closed-loop execution is a separate subsequent stage.

For unattended execution, `scripts/n0_twam/run_recorded_experiment.py` owns
exactly one official server process, waits for protocol metadata, runs the same
CLI once, terminates only the server it started, and writes a separate lifecycle
receipt under `deployment/outputs/recorded-n0-runs/`.

## Complete frozen40 tactile-causal calculation

The cohort runner evaluates the 40 declared raw HDF5 episodes directly. It
starts one task-specific N0 server at a time, reuses it for the five episodes of
that task, and compares paired Clean and structural observed-tactile-absence
predictions at each episode's deterministic strongest-contact anchor:

```bash
python scripts/n0_twam/run_frozen40_tactile_causal.py \
  --root "$PWD/deployment" \
  --data-root "$PWD/deployment/artifacts/datasets/univtac_frozen40" \
  --split-manifest "$PWD/configs/protocols/univtac_frozen40_hdf5_v1.json" \
  --gpus 0 \
  --output "$PWD/deployment/outputs/recorded-n0/frozen40-tactile-causal-v1.json"
```

The loader consumes the released tactile field
`tactile/{left,right}_gsmini/rgb_marker` and requires a complete 12-step EE8
expert horizon. The output retains expert, Clean, and absence action chunks,
source hashes, selected contact strengths, per-episode metrics, per-task means,
the all-40 micro mean, and the ten-episode `grasp_classify`/`lift_can` primary
subgroup. `success_rate_claimed=false` is mandatory because neither predicted
chunk is executed in Isaac Sim.

## Align a live failure to expert demonstrations

`expert-alignment` is a diagnostic command for a completed, verified live
artifact. It never starts Isaac Sim or the N0 server. The command strictly
loads the content-addressed live bundle and one or more complete expert HDF5
episodes, ranks the initial states by mean normalized Top/Wrist RGB MAE (with
EE translation as a tie breaker), then compares:

- initial EE translation, quaternion, gripper, camera, and tactile differences;
- left, right, and bilateral contact landmarks derived with the benchmark's
  frozen depth hysteresis;
- grasp close/minimum/re-open, 10 mm lift, and rotation landmarks;
- absolute EE8 and relative-motion errors over normalized trajectory progress.

```bash
python -m robotactile_benchmark.cli expert-alignment \
  --artifact deployment/artifacts/live-univtac/clean-campaigns/n0-clean-lift-bottle-once-v1/lift_bottle/0000 \
  --expert-hdf5 deployment/artifacts/datasets/univtac_raw/lift_bottle/clean/6.hdf5 \
  --expert-hdf5 deployment/artifacts/datasets/univtac_raw/lift_bottle/clean/90.hdf5 \
  --task lift_bottle \
  --output deployment/outputs/diagnostics/n0-clean-lift-bottle-once-v1/expert_alignment_v1.json
```

The result is `recorded_expert_vs_live_closed_loop_diagnostic_v1`, not a task
metric. Expert HDF5 files do not carry a live success receipt, normalized
progress is not temporal correspondence, and v1.0 live artifacts do not expose
object pose or contact actor identity.

New live artifacts use root receipt v1.1 and add `transition_trace.json`. The
sidecar pins initial task diagnostics, every backend transition, success
predicate subconditions for `lift_bottle`, and contact phases. The strict loader remains compatible
with existing root receipt v1.0 artifacts, where these fields are explicitly
absent rather than reconstructed.

Compact diagnostic artifacts use root receipt v1.2. `metrics_only_v1` retains
the typed terminal result, action trace, transition diagnostics, and complete
trace hashes without observation arrays. `preview_v1` additionally stores a
separate, bounded keyframe subset; it is not a `DeliveryFinalization` and never
claims to contain the full trace. Legacy v1.0/v1.1 roots are interpreted as
`paper_full_v1`. Only `paper_full_v1` is accepted by claim and paper gates.
