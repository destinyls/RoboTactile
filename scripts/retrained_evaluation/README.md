# Retrained four-model Isaac evaluation

For A1/A2 structural absence, use the opt-in
[native-missing and zero-fill protocols](../../docs/tactile_availability.md).
The supplementary runner creates separate Clean+A1+A2 groups and reports;
it never silently fills missing inputs or changes historical runs.

This opt-in runner uses the explicit train759 checkpoints, training prompts,
normalizers, action representations and tactile routes. It never substitutes a
released checkpoint. Existing N0-TWAM and other released-model entrypoints are
unchanged. Run from the source repository with the matching wheel on PYTHONPATH.

`prepare.py` writes content-addressed per-model bindings into a new campaign;
its `--help` documents deployment and N0 prepared-artifact arguments. Preparation
hashes weights and binds the simulator diagnostic dataset identity. It is not
a closed-loop result. Dream-Tac additionally needs its training dataset statistics,
T5 embeddings, VAE tokenizer and a separate compatible runtime; DCP weights alone
are insufficient.

After bindings exist, the execution entrypoint is:

```bash
python -m scripts.retrained_evaluation.campaign \
  --campaign /absolute/deployment/outputs/CAMPAIGN \
  --code /absolute/source-snapshot \
  --package /absolute/isolated-wheel-install \
  --models n0_twam n0_vtla ftp1_policy dream_tac
```

The campaign serializes GPU ownership. Every task/model reuses one policy server.
Within one seed, Clean and fault conditions share one Isaac session and matched
task snapshot; the default multi-seed runner starts a new Isaac process for the
next seed. It preserves all existing group directories instead of rerunning
them. A pre-episode startup failure needs an explicit no-clobber recovery; it is
not a model failure.

## Persistent 100-rollout condition shards (N0-TWAM)

For large N0-TWAM statistics, use `persistent_condition` instead of the default
paired-group runner. One invocation is deliberately fixed to one `(task,
condition, GPU)` and starts N0 and Isaac exactly once. It then executes the seed
range serially. Between seeds it closes the old task runtime, releases renderer
and timeline state, reconstructs a distinct USD stage, builds a fresh task and
policy object, calls the N0 reset RPC, and creates a new fault-delivery session.
It does **not** hot-reload tasks or reuse an episode state.

This task reconstruction is intentional. UniVTAC's task configuration and its
deterministic seed hook are construction-bound; calling `task.reset()` with a
different seed on the same task object would violate the current seed contract.
The optimized boundary is therefore “persistent N0 process + persistent Isaac
application”, not “persistent seed-bound task object”.

First produce the task-local same-application reset proof documented in
[`docs/benchmark_workflow.md`](../../docs/benchmark_workflow.md). A statistical
or paper run requires that passed receipt; a diagnostic can omit it. Then launch
one condition shard, for example:

```bash
python -m scripts.retrained_evaluation.persistent_condition run \
  --binding /absolute/bindings/n0_twam.json \
  --task lift_can \
  --condition F1_global_response_drift \
  --campaign /absolute/outputs/lift-can/F1 \
  --code /absolute/RoboTactile \
  --package /absolute/RoboTactile/src \
  --seed-start 0 \
  --seed-count 100 \
  --capture-profile metrics_only_v1 \
  --evidence-mode statistics \
  --severity-registry optical_marker_extreme_v1 \
  --severity-level 5 \
  --fault-window early_random_onset_v1 \
  --fault-onset-max-index 8 \
  --gpu-id 0 \
  --port 29501 \
  --reset-equivalence-receipt /absolute/proofs/lift_can.json
```

Use `--condition clean` for the matched Clean shard. F1--F7, T1--T3, and
C1--C2 accept their registered operator IDs. A1/A2 additionally require an
explicit availability adapter, for example `--tactile-availability-mode
zero_fill_v1 --tactile-zero-shape H W C`; this never turns black images into a
claim of native structural-absence support.

Map the 15 conditions (Clean + 14 failures) onto 15 GPUs and keep the sixteenth
GPU as a replacement/visualization worker. On each host, assign a distinct
`--gpu-id` in `0..7` and a distinct `--port` to every concurrently running
condition. The selected physical GPU is exposed as logical `cuda:0` to both N0
and Isaac. Every condition writes an immutable
`work_manifest.json`, per-seed result and episode receipt, one
`persistent_condition_session.json`, and a final `summary.json`. The session
receipt must report `isaac_application_launch_count=1`,
`runtime_count=seed_count`, and one distinct runtime ordinal per seed. Existing
result or artifact paths fail before Isaac starts, so a completed rollout is
never silently repeated.

Evidence modes are intentionally distinct:

- `diagnostic`: reset proof optional; suitable for engineering checks only;
- `statistics`: reset proof required; `metrics_only_v1` is allowed for scalable
  SR aggregation, but full RGB/tactile videos are unavailable;
- `claim`: reset proof and `paper_full_v1` are both required.

The worker exits on an infrastructure exception instead of silently replacing a
missing episode. A no-clobber successor campaign may then schedule only the
missing seeds. A model timeout or unsuccessful terminal predicate is a valid
model outcome and does not restart N0 or Isaac.

After all shards finish, aggregate them with one `--campaign` argument per
condition:

```bash
python -m scripts.retrained_evaluation.persistent_condition aggregate \
  --campaign /absolute/outputs/lift-can/clean \
  --campaign /absolute/outputs/lift-can/F1 \
  --campaign /absolute/outputs/lift-can/F2 \
  --campaign /absolute/outputs/lift-can/OTHER_CONDITIONS \
  --output /absolute/outputs/lift-can/aggregate.json
```

The default aggregate requires exactly Clean + 14 failure shards with the same
ordered seeds and the same model identity (checkpoint, source tree, dataset,
task config, and normalizer). For every fault and seed it compares
`initial_state_sha256` against Clean before computing paired SR drop and
fault-induced failures. Any missing/invalid episode, incomplete N0 reset, or
initial-state mismatch sets `aggregate_valid=false`. `--allow-subset` exists
only for incremental diagnostic analysis.

## Shared storage with different GPU architectures

An SM120-only cuRobo binary cannot execute on an SM80 A800. Do not rebuild an
editable cuRobo installation in place while another host uses the shared tree.
Build a private source/binary overlay using the existing Isaac Python and CUDA
toolkit, without pip installation or dependency changes:

```bash
python scripts/retrained_evaluation/build_curobo_overlay.py \
  --source /absolute/deployment/sources/curobo \
  --output /absolute/RoboTactile/deployment/runtime/a800-curobo-v1 \
  --isaac-python /absolute/deployment/runtime/isaac-sim-4.5.0/python.sh \
  --cuda-root /absolute/deployment/runtime/cuda-toolkit-12.8 \
  --architecture 8.0 --jobs 2

/absolute/RoboTactile/deployment/runtime/a800-curobo-v1/isaac-python.sh \
  scripts/retrained_evaluation/probe_curobo_overlay.py \
  --overlay /absolute/RoboTactile/deployment/runtime/a800-curobo-v1 \
  --output /absolute/RoboTactile/deployment/runtime/a800-curobo-v1/gpu_probe.json
```

The output must not exist. The build preserves original binaries and records
their before/after hashes. The probe checks all five loaded extension paths and
GPU architecture, then runs planner warmup and a real CUDA joint-space plan;
this is a native GPU smoke test, not an evaluation episode. Only after it passes,
set `isaac_python` in a **new A800-only binding** to the generated launcher and
recompute its binding hash. Preserve the existing host's binding and runtime.
Use a new campaign for a pre-episode failure; never rerun a valid episode.

## First diagnostic matrix

- Eight tasks, seed 0, one Clean per task/model.
- Standard `optical_marker_v1` S5 and separately labelled
  `optical_marker_extreme_v1` S5. These are not one severity curve.
- Without route-matched certified-rest, only F5/T1/T2/T3/C1 are scheduled.
  The other seven operators are `not_run_missing_calibration`.
- A1/A2 remain `unsupported_contract` for required-tactile native interfaces;
  black images are not structural absence.
- Retrained 10 Hz actions hold 12 physical steps at 120 Hz. Temporal manifests
  and delivered/source timestamps use the actual observation cadence.
- Official UniVTAC success is unchanged. There is no Restored condition.
- `preview_v1` saves at most 64 frames; metrics/actions/termination are retained.

Per-condition JSON is published before simulator teardown in
`groups/MODEL/TASK/results/`. `paired_receipt.json` witnesses completed groups.
`reports/` contains immutable snapshots; delivery/reset/export infrastructure
errors are excluded from model SR and retained explicitly in raw results.
Never interpret missing results or `model_loaded` as success/failure. One seed
per task is a diagnostic, not a paper-level success-rate estimate.

The current deployment and execution history is documented in
[`plan/retrained_four_model_clean_robustness.md`](../../plan/retrained_four_model_clean_robustness.md).
