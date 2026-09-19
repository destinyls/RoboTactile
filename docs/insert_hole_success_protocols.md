# UniVTAC `insert_hole` dual success protocols

RoboTactile preserves the pinned UniVTAC predicate and adds one independently
identified strict predicate. The frozen machine-readable contract is
[`configs/protocols/univtac_insert_hole_dual_success_v1.json`](../configs/protocols/univtac_insert_hole_dual_success_v1.json).

| Metric | XY | Insertion | Alignment | In-hand drift | Stable hold |
|---|---:|---:|---:|---:|---:|
| `official_v1` | each axis `< 10 mm` | `> 40 mm` | dot `> 0.99` | absolute in-hand z drift `< 40 mm` | first passing step |
| `insert_hole_strict_v1` | radial error `< 5 mm` | `> 50 mm` | dot `> 0.999` | absolute in-hand z drift `< 25 mm` | at least 30 consecutive 120 Hz physics steps (`0.25 s`) |

The inequalities are strict. A value exactly on a threshold does not pass.
`XY error` in the strict profile is the Euclidean norm of target-relative x/y;
the official predicate remains the upstream axis-wise test. `In-hand drift`
retains the upstream definition—the absolute z displacement of the object in
the gripper frame—so the strict result changes the threshold rather than the
physical quantity.

Every `insert_hole` transition stores target-relative xyz, radial XY error,
insertion depth, alignment dot, in-hand z drift, all subconditions, and both
predicate states. A strict run also stores its consecutive physics-step count.
When upstream official success fires before the strict hold completes, the
backend records that official success, releases only the upstream success
latch, and continues the same closed loop. It does not reset the task or model.
The counter resets to zero on any strict-geometry failure.

## Scoring and reporting

The pinned UniVTAC predicate is normative. Only `official_v1` determines
benchmark task success and contributes to the headline SR, model ranking, and
paper main-table result. `insert_hole_strict_v1` is diagnostic-only: it cannot
change an official success into a failure, change an official failure into a
success, or enter the official denominator.

The selected profile is part of the request and `ClosedLoopRunSpec` content
identity. Therefore official and strict artifacts cannot silently share a run
hash. Report them as two metrics:

- `insert_hole_official_sr`: the normative upstream-comparable benchmark result;
- `insert_hole_strict_v1_sr`: a separately labeled RoboTactile diagnostic.

Use the same trial inventory, seeds, model, condition, and runtime contract for
paired comparison, but never combine the two profiles in one denominator. A
strict artifact retains `official_ever_success`, so the official outcome is
auditable even when the strict run continues beyond upstream termination.

## N0-VTLA request generation

The default and benchmark-scoring path remains official:

```bash
python scripts/n0_vtla/prepare_insert_hole_pair.py \
  --campaign-id n0-vtla-insert-hole-official-v1 \
  --success-profile official_v1
```

Generate a separate no-clobber strict campaign with the same seeds:

```bash
python scripts/n0_vtla/prepare_insert_hole_pair.py \
  --campaign-id n0-vtla-insert-hole-strict-v1 \
  --success-profile insert_hole_strict_v1
```

The option changes only success evaluation and terminal timing. It does not
change observations, policy inputs, actions, reset geometry, fault injection,
or the official N0-VTLA server.
