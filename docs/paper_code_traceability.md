# Paper-to-Code Traceability

This page maps paper contracts to the implementation without promoting a
software check into simulator or hardware evidence.
Commands abbreviated as `pytest` or `robotactile` in the table are run through
the activated, hash-locked `.venv` environment from the package root.

| Paper contract | Implementation | Executable verification | Evidence status |
|---|---|---|---|
| model-visible payload vs evaluator provenance | `contracts.py` | `pytest -q tests/test_contracts.py` | contract test |
| non-test no-contact reference bundle | `rest_references.py`, `operators/base.py` | `pytest -q tests/test_trials_resources.py tests/test_validators.py` | self-attested reference contract; no external hardware calibration |
| frozen operator instance and exact 2+7+3+2 registry | `operator_parameters.py`, `manifests.py`, `operators/registry.py` | `robotactile validate-registry` | deterministic code contract |
| A1/A2 structural absence | `operators/availability.py` | `pytest -q tests/test_operators.py tests/test_validators.py` | synthetic intervention |
| F1--F7 fidelity signatures | `operators/fidelity.py`, `operators/fidelity_transforms.py` | `pytest -q tests/test_operators.py tests/test_fidelity_single_lattice.py tests/test_validators.py` | mechanism proxy; not trace-calibrated |
| T1--T3 source mappings | `operators/temporal.py`, `streaming/temporal.py` | `pytest -q tests/test_operators.py tests/test_validators.py tests/test_streaming_faults.py` | synthetic intervention and batch/streaming hash parity |
| C1/C2 binding failures | `operators/context.py` | `pytest -q tests/test_operators.py tests/test_validators.py` | synthetic intervention |
| delivery hard gate | `validators.py`, `operator_validators.py`, `signature_validators.py`, `reference_signatures.py` | `pytest -q tests/test_validators.py` | independent software validation of the registered intervention |
| closed-loop runner and content-addressed artifacts | `closed_loop/runner.py`, `closed_loop/artifacts.py`, `closed_loop/artifact_validation.py` | `pytest -q tests/test_closed_loop_runner.py tests/test_closed_loop_artifacts.py` | deterministic software contracts only |
| UniVTAC observation/backend boundary | `adapters/univtac.py`, `backends/univtac_factory.py`, `backends/univtac_isaac.py` | `pytest -q tests/test_univtac_backend.py tests/test_univtac_factory.py` | dependency-injected contract tests; not an Isaac run |
| official UniVTAC ACT policy loading and live CLI | `policies/univtac_official_act.py`, `execution/official_act.py`, `execution/live_univtac.py`, `cli.py` | `pytest -q tests/test_univtac_official_act_policy.py tests/test_official_act_live_cli.py` | code and injected live-path tests; emitted live artifacts remain unqualified |
| typed N0 gateway and qpos8/cache transaction | `transport/n0_client.py`, `transport/n0_codec.py`, `transport/n0_commit.py` | `pytest -q tests/test_n0_codec.py tests/test_n0_transport_contracts.py tests/test_n0_transport_lifecycle.py tests/test_n0_transport_integrity.py tests/test_n0_transport_authority.py tests/test_n0_policy.py` | typed software contract tests; no deployed N0 service result |
| clean/faulted/no-touch/restored pairing | `trials.py`, `schemas/trial_manifest.schema.json`, `scripts/live_univtac/generate_pull_out_key_matrix.py` | `pytest -q tests/test_trials_resources.py tests/test_pull_out_key_matrix_generator.py` | hash-bound request generation; not task execution |
| primary 14 x 5 matrix | `matrix/primary_generation.py`, `matrix/builders.py`, `matrix/runner.py`, `matrix/io.py` | `pytest -q tests/test_primary_matrix_generation.py tests/test_matrix_contracts.py tests/test_matrix_runner.py` | 70 comparisons, 142 unique executions with shared clean/no-touch controls, exact resources, and strict resume; request generation is not execution |
| no-allocation live deployment gate | `execution/preflight.py`, `execution/preflight_contracts.py` | `robotactile preflight-live --help` and `pytest -q tests/test_live_preflight.py` | source/artifact/host readiness only; no model or simulator allocation |
| matrix-to-report derivation | `reporting/matrix_adapter.py`, `reporting/aggregation.py`, `reporting/bundle.py` | `pytest -q tests/test_reporting_matrix_adapter.py tests/test_reporting_aggregation.py tests/test_reporting_export.py` | strict-loads source artifacts and writes deterministic JSON/CSV/LaTeX/SVG; does not add an evidence tier |
| restored-condition recovery evidence | `reporting/recovery_evidence.py`, `reporting/recovery_evidence_matrix.py` | `pytest -q tests/test_reporting_recovery_evidence.py` | registered behavioral sidecar API is implemented; lag stays ineligible when aligned clean/restored evidence is absent |
| deterministic frozen replay | `replay.py` | `robotactile smoke-replay --operator F6_history_residual_imprint --severity 3 --output outputs/smoke` | bounded synthetic replay |
| recorded-clean UniVTAC replay and V3 diagnostic atlas | `data/real_univtac/train759/lift_bottle/episode_000380`, `../robotactile_iclr2027/tools/real_univtac_*v3*.py` | source hashes, per-array contracts, delivery validators, semantic witnesses, and artifact parity recorded in the V3 receipts | recorded simulator clean input with synthetic fault injection; qualitative only |
| local deployment helpers | `scripts/live_univtac/install_isaac_sim_4_5.sh`, `install_isaaclab_v2_1_1.sh`, `install_curobo_v0_7_7.sh`, `smoke_isaac_sim_4_5.sh` | `pytest -q tests/test_live_univtac_deploy_scripts.py` | source-tree code and static contract tests only; installation or headless execution is not claimed |

The severity registry is deliberately named `provisional_engineering_v2`.
Levels are monotone only within an operator; they are neither physical
calibration nor comparable dose across operators. The independent pixel
matcher checks registered synthetic signatures without calling the injector
under test. It does not establish correspondence to damaged hardware.

## Evidence tiers

| Tier | Established by this repository | Explicit limit |
|---|---|---|
| software contracts | deterministic delivery, backend/policy protocols, resume, artifact integrity, and report plumbing | no learned-policy task result, Isaac Sim, or hardware claim |
| live preflight | source, artifacts, GPU visibility, and Isaac Python are ready | no simulator allocation or task result |
| unqualified live | a requested official runtime path executed and produced a strict-loadable trace | not Isaac qualification and not a publishable task-result matrix |
| Isaac-qualified | reserved for a separate qualification receipt and acceptance protocol | no such result is currently bundled |
| real robot | reserved for hardware-calibrated execution evidence | no such result is currently bundled |

Reporting preserves the input evidence boundary; deterministic aggregation
does not upgrade software-contract or unqualified-live evidence.

## Reproducible verification

Run the current checks from the package root rather than relying on embedded
test counts or build hashes, which become stale whenever the source changes:

```bash
bash scripts/bootstrap_pip.sh
source .venv/bin/activate
python -m pytest -q
python -m ruff check .
python -m mypy src
python -m robotactile_benchmark.cli validate-registry

python -m robotactile_benchmark.cli smoke-matrix \
  --output outputs/smoke-matrix
python -m robotactile_benchmark.cli closed-loop-smoke \
  --output outputs/closed-loop-smoke
python -m robotactile_benchmark.cli preflight-live --help
```

For a materialized primary matrix, regenerate the paper report with:

```bash
python -m robotactile_benchmark.cli report-matrix \
  --matrix-manifest outputs/primary/matrix_manifest.json \
  --matrix-output outputs/primary \
  --reporting-spec configs/reporting_spec.json \
  --output outputs/report
```

The report bridge strict-loads every matrix receipt and referenced artifact.
Focused or incomplete primary grids, unknown evidence types, altered hashes,
missing artifacts, and unregistered recovery signals fail closed.

## Stable recorded-clean V3 artifacts

These hashes identify the frozen qualitative V3 inputs and figures; they are
retained because the recorded-clean artifacts themselves are stable:

- matrix / diagnostic payload / diagnostic sidecar SHA256:
  `dbc033ffa62631c05017dc8f9c676da5770d2e04f959e3a6b2a116be526cf19e`,
  `6ed93eafe4bddbb8dd006f19c92d60fd3beef4a7e9b01de3b98dc2e0182b0b0e`,
  `31ce2441a0302a282c6bf5c4e49eb597b4e10cba31d106ad50b6f1cc6b2c775a`;
- diagnostic-atlas PNG / PDF / receipt SHA256:
  `b5be8f45da27e6792c9ed4c6b387a6008e5e7e7983d226708a3e868cc036cd3a`,
  `b30d00aa0f91b3f06cd3389f013abefeb7e70910585f1c2616d903e6b77d2751`,
  `c623747c0f8371e4b6683c6c26091b6833e816c32205825b5ad14c2200d31dfc`;
- raw-gallery PNG / PDF / receipt SHA256:
  `4ec4dd57b73ee51b30de7654a5b75b2855be36be49b172fe06f0d98c6de8acba`,
  `4ca4f6ae6b754fb62ce88730282dcd490d8429bd28f9000bd22b163bd7bbd755`,
  `00d0e9ec566d7f3743fdea0ede9aba066744260f2ee4a20b849ee95c12edeba7`.

## Evidence boundary

The repository currently contains code contracts, no-allocation live preflight,
recorded-clean qualitative replay, and paths for unqualified live execution.
It contains **no Isaac-qualified result, no verified simulator task-result
matrix, and no real-robot result**. A generated request, installation helper,
headless smoke, unqualified live trace, or deterministic report proves only
the operation named by its own receipt.
