# RoboTactile Benchmark Card

## Purpose

RoboTactile evaluates optical-tactile robot policies under 14 registered
Availability, Fidelity, Temporal, and Context operators. Each model integration
uses a preregistered applicability contract and matched trial identity.
N0-TWAM, N0-VTLA, FTP-1, and Dream-Tac use Clean/Faulted pairs; ACT can
additionally use its registered matched no-touch checkpoint.

## Model integrations

ACT, Dream-Tac, FTP-1, N0-TWAM, and N0-VTLA are first-class integrations behind
one `PolicyAdapter`. ACT has a registered matched vision-only profile. Frozen
FTP-1 and N0-TWAM cannot represent structural tactile absence through their
official executable transports. F1-F7, T1-T3, and C1-C2 are executable fault
operators for both integrations. A1 stream absence and A2 frame erasure produce
typed `unsupported_contract` receipts, no live request, and no simulator claim;
a black/rest image is never substituted. Dream-Tac remains CODE-ready only
because no task-aligned public checkpoint is registered.

## Evidence levels

- Software contracts: deterministic contracts and protocol plumbing.
- Live preflight: frozen deployment resources and host readiness, without allocation.
- Unqualified live: a hash-verified runtime trace without simulator acceptance.
- Qualification v3: eight ordered task-local source/observation-parity bindings,
  including a task-specific normalizer and serve bundle.
- Qualified task result: requires source-bound attempt v3 evidence from both the
  N0 rank-0 server and the Isaac child.
- Real robot: outside the current UniVTAC release.

No software test, synthetic operator image, or installation receipt is treated
as hardware fault calibration or task-performance evidence.

## Reproducibility

Operators, severity paths, trial manifests, source maps, traces, outcomes, and
reports are content addressed. External repositories and model artifacts must
match immutable commits and SHA-256 manifests.

Clean baselines use an independent preregistered campaign manifest. It records
the master seed and deterministic seed roles, canonical request hashes, exact
trial/run identities, and all planned artifact paths. `paper_v1` requires all
eight frozen UniVTAC tasks with at least 100 complete-horizon trials per task;
`diagnostic_v1` and `pilot_v1` cannot support paper claims.

An N0 robustness bundle expands canonical Clean requests into one Clean cell
per frozen pair, executable Faulted cells, and explicit unsupported receipts.
Its operator-template seed is invariant across severities for the same
pair/operator. Operators F1, F2, F3, F4, F6, and registered-pixel C2 require a
strict task-bound rest-reference artifact. Generation, execution, and reporting
are no-clobber and source-hash bound; missing live artifacts remain explicit
report blockers rather than implicit failures.

N0 Clean execution is source-bound to
`robotactile_n0_training_60hz_ee_v1`: every absolute EE8 endpoint advances two
120 Hz physics ticks and one render, matching the 60 Hz training-row cadence.
The upstream variable-waypoint EE executor is retained only as an explicit
reference diagnostic.

UniVTAC `insert_hole` retains the pinned upstream success predicate as
`official_v1`. RoboTactile additionally registers `insert_hole_strict_v1` with
5 mm radial XY error, 50 mm insertion, 0.999 alignment, 25 mm in-hand z drift,
and a 30-step hold at 120 Hz. Official SR and strict SR are separate metrics
with separate run-spec identities and must not be pooled in one denominator.

Fault-pair capture is separately declared as `metrics_only_v1`, `preview_v1`,
or `paper_full_v1`. Compact capture preserves outcome/action/termination hashes
but cannot be promoted into missing full frames. N0 reports use
`degradation = Clean SR - Faulted SR` and
`retention = Faulted SR / Clean SR`; the latter is not clipped.

Clean aggregation is intention-to-treat at the planned-inventory boundary. A
live artifact is accepted only with one matching successful command-attempt
receipt and log hash. Missing, invalid, ineligible, and protocol-invalid trials
remain explicit. Per-task success uses Wilson intervals; the equal-task macro
uses a seeded task-stratified bootstrap. Partial summaries are available only
through an explicit diagnostic flag and remain non-claimable.

The hard watchdog covers startup, reset, policy execution, artifact export, and
teardown. Official N0 Clean requests treat it as infrastructure-only; elapsed
wall time never becomes a scoreable failure. Ordinary model outcomes, including
an official action-horizon timeout, remain in the denominator. Only a classified
execution exception with matching lifecycle evidence is `exception_replaced`
and excluded before a preregistered reserve candidate is used.

## Qualification and publication gate

`qualify_univtac_all_tasks.sh` checks import, reset, and the declared action
contract once for every frozen task. Qualification v3 augments those receipts
with eight task-local observation-parity/source bindings; normalizer and serve
bundle identities may be task-specific. The N0-only all-task runner then creates
and propagates the N0 rank-0 attestation, Isaac child attestation, and attempt v3
without user-supplied child flags. `clean-campaign-publish` removes
`simulator_not_qualified` only for a complete `paper_v1` bundle with exact
qualification/attestation matches. Legacy v1/v2 qualifications remain loadable
but cannot authorize promotion. Diagnostic one-episode-per-task campaigns are
non-claimable by construction.

The public P0-P3 diagnostic sequence is an all-8 one-Clean-per-task
infrastructure gate, one frozen-pair L3 pilot, and an S1/S5 bracket using the
same source pair and operator-template placement. A P1 model failure is retained
as measured, not rerun. P2/P3 reports with one pair remain diagnostics and do
not establish a statistically sufficient paper claim.

## Limitations

Synthetic delivery faults approximate model-visible consequences rather than
complete sensor mechanics. The public N0 generate/run/report workflow is
implemented, but the repository does not bundle an Isaac-qualified full N0
result campaign or real-robot evaluation.

Individual Clean artifacts remain unqualified evidence. Paper promotion is a
separate dual-attested bundle gate and does not establish real-robot validity.
The official-comparison default is `official_reproduction`, which records but
does not reject reset-time terminal signals. The opt-in
`replace_initial_terminal_v1` behavior is a separate robustness diagnostic and
is not directly comparable with the public 84.5% reference. This repository
does not claim a newly executed GPU result merely because these gates exist.
