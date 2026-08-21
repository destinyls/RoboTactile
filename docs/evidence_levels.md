# Evidence Levels

| Level | Establishes | Excludes |
|---|---|---|
| Software contract tests | contracts, runner, artifacts, matrix plumbing | learned model and simulator |
| `live_preflight_no_simulator_execution_v1` | request, source, artifact, GPU, and Isaac-Python readiness | simulator allocation and task execution |
| `unqualified_live_univtac_execution_v1` | verified runtime trace | Isaac acceptance and publishable task result |
| Isaac-qualified receipt | frozen environment acceptance | real-robot behavior |
| source-bound task results | qualified trials and statistics | hardware fault calibration |

Evidence levels are monotonic only through explicit new receipts. Running an
unqualified command on a GPU does not change its evidence level.
