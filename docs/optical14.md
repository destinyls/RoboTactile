# Optical 14: production injection and visual validation

`optical_marker_v1` keeps the existing fourteen operator IDs and adds an opt-in
engineering realization for GelSight RGB with dark markers. Legacy
`provisional_engineering_v2` manifests retain their old behavior. The new profile
is an observation-level G1 proxy, not calibrated elastomer damage or a new
closed-loop success-rate result. Never mix the two registries in one severity curve.

## One production path

Both `runtime.apply_fault` and `StreamingFaultSession` execute the same optical
delivery implementation. The gallery consumes those outputs, including actual
missing payloads and routed source timestamps. It has no display-only replacement
for F2/F4/F6/C2. External camera RGB and proprioception remain byte-exact.

```python
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.streaming.session import StreamingFaultSession

manifest = FaultManifest(
    operator_id="F2_spatial_sensitivity_loss",
    severity_level=3,
    operator_seed=23,
    start_index=100,
    stop_index=150,
    sensor_slots=("right",),
    observability=Observability.BLIND,
    severity_registry="optical_marker_v1",
    parameters={
        "sample_period_s": 1 / 60,  # Actual observations, not the physics tick.
        "rest_reference_sha256": rest_references.sha256,
    },
)
session = StreamingFaultSession(manifest, rest_references)
for clean_record in observation_stream:
    delivered_record = session.deliver_one(clean_record)
```

The caller supplies existing `EvaluationRecord` objects and a bound
`RestReferenceBundle`. C2 additionally requires `realization="registered_pixels"`.
F7 now requires a rest reference in this registry. F5 records displacement as a
fraction of the smaller native image dimension, not an ambiguous pixel dose.
T1 needs enough pre-window
history. T1/T2/T3 validate the declared sample period against genuine source
timestamps. A mismatch is an error, not a silent change of delay duration.
Transport adapters must still declare which operator contracts they can consume;
an RGB-only model does not automatically support structural A1/A2 absence.

## Observable meanings

| ID | New realization | What is preserved / limited |
|---|---|---|
| A1 | Continuous absent payloads | Structural missingness, not black RGB |
| A2 | Intermittent absent records with later resumption | Actual source gaps |
| F1 | Per-channel gain and offset ramp | Geometry unchanged; photometry intentionally changes |
| F2 | Compact central optical-response attenuation | Current markers; no warp; exact identity outside support |
| F3 | Persistent soft coating-appearance scar | Live contact and current markers; not a crack mechanics model |
| F4 | Peripheral stuck-at-rest optical field with a narrow transition | Current markers; no imported rest marker lattice |
| F5 | Fold-free current-frame warp | One coherent warped lattice; contact gated |
| F6 | Fast loading and slow recovery of marker-free optical response | Same-episode causal history; current marker pixels |
| F7 | Soft-knee compression of optical response relative to rest | Current markers; no pressure calibration claim |
| T1 | Common source delay | Left/right synchronization retained |
| T2 | One held source then resumption | Payload hash/source time frozen while delivery advances |
| T3 | One-sided source delay | Explicit inter-sensor skew; this profile does not claim jitter |
| C1 | Complete left/right source swap | Actual physical-source provenance |
| C2 | Registered integer image translation | An image-registration fault, not physical remounting |

Dark marker cores are detected at max RGB <= 65 and protected with a two-pixel
guard. Local non-marker averaging estimates the optical field under the markers.
This is not recovered metric deformation. The current marker pixels are copied
back exactly for F2/F3/F4/F6/F7. A large dark region without usable surrounding
shading is rejected as inapplicable. The profile is not universal for translucent,
bright-marker, or marker-free tactile sensors.

F6 uses `buildup_tau_s=0.03` and severity-dependent recovery time constants
`[0.15, 0.30, 0.60, 1.20, 2.40]` seconds, with mixture gains
`[0.12, 0.22, 0.35, 0.50, 0.68]`. Each pixel uses the loading time constant when
the current response magnitude exceeds its history, otherwise the recovery time
constant. The update coefficient is exp(-actual elapsed source time / tau).
Evidence is extra optical response at locally unloading pixels, not an increase
in whole-image brightness. No historical marker images are accumulated.

F2 keeps its center at `(0.5, 0.5)`. The flat attenuation core and outer support
radii are respectively `0.16` and `0.28` times the smaller image dimension.
Retained response gains are `[0.90, 0.75, 0.55, 0.30, 0.10]` for S1-S5.
These are explicit G1 engineering settings, selected to cover central contact
edges without any coordinate shift, not parameters fitted to real sensor damage.

## Reference correction discovered by visual iteration

The legacy `pull_out_key/55` and `insert_HDMI/90` image-based low-change frames
are not true no-contact frames. UniVTAC records after `pre_move` has grasped the
object. Their raw depth remains inside the gel's contact range. Subtracting these
loaded frames imports object contours into F2/F4/F7.

The corrected source preparation therefore binds a **calibration-derived
zero-indentation optical reference** from the official Taxim renderer. This is
a counterfactual sensor reference, not an observed no-contact episode frame.
Recorded depth is first rerendered and compared with recorded marker-free RGB;
source/calibration hashes and JPEG-level errors are saved. No Isaac or model
inference is needed for this small optical calculation. No third-party calibration
asset is copied into the RoboTactile wheel.

F6's recorded demonstration uses **local unloading while the object remains
grasped**, supported by depth changes. A fully detached release must not be
claimed from this example. Reference qualification for a formal live campaign
remains a separate sensor/version-specific requirement.

## Reproduce and inspect

Activate the project environment. The gallery itself needs only core dependencies
and Pillow; source preparation additionally needs h5py, OpenCV, torch and
torchvision, plus an existing pinned UniVTAC checkout and its calibration assets.
The source preparation runs on the CPU in an isolated project environment; it
does not install or initialize Isaac and must not change a running model runtime.
Tested preparation dependencies: Python 3.12, numpy 2.2.6, h5py 3.14.0,
opencv-python-headless 4.11.0.86, torch 2.8.0, torchvision 0.23.0 and Pillow 11.3.
The tested UniVTAC checkout was `05bcd3edb92237107efa40105292a24f1a9fd761`;
the preparation receipt additionally hashes the exact renderer and calibration
assets. Data and calibration assets remain external to the wheel.

Place the two authorized original HDF5 files in
`deployment/artifacts/optical14-sources/raw/{pull_out_key/clean/55.hdf5,insert_HDMI/clean/90.hdf5}`.
With the external UniVTAC checkout at `deployment/src/UniVTAC`, run from the
RoboTactile repository root (change only that path if using another checkout):

```bash
PYTHONPATH=src python scripts/extract_optical14_raw.py \
  --raw-root deployment/artifacts/optical14-sources/raw \
  --output-root deployment/artifacts/optical14-sources/raw_standalone_v1

PYTHONPATH=src python scripts/prepare_optical14_sources.py \
  --source-root deployment/artifacts/optical14-sources/raw_standalone_v1 \
  --raw-root deployment/artifacts/optical14-sources/raw \
  --univtac-root deployment/src/UniVTAC \
  --output-root deployment/artifacts/optical14-sources/calibrated_standalone_v1

PYTHONPATH=src python scripts/visualize_optical14.py \
  --source-root deployment/artifacts/optical14-sources/calibrated_standalone_v1 \
  --output-root deployment/artifacts/optical14-gallery-v7science

PYTHONPATH=src python scripts/assess_optical14.py \
  --gallery-root deployment/artifacts/optical14-gallery-v7science \
  --source-root deployment/artifacts/optical14-sources/calibrated_standalone_v1 \
  --output-root deployment/artifacts/optical14-scientific-v2detail
```

Output directories must be new. Keep failed iterations separately: a directory
without a completed gallery receipt is not a complete result. Inspect native
clean/delivered images, the same fixed-scale absolute-difference image, severity
strips, source-time strips, and F6 depth/optical unloading together. Do not tune
severity against policy failure rates or silently replace a failed operator's
visualization with another processing algorithm.

F2 uses the fixed clean HDMI witness, with its ROI still centered. F4 uses the
fixed clean key witness with a peripheral ROI. S1-S5 reuse the exact same clean
frame; frames are never selected by maximum fault contrast. The standalone
extractor has no dependency on the manuscript directory. It does not relabel a
low-change image as release. The gallery records the source-code hashes before
and after production delivery and refuses a completed receipt if they change.

The gallery is explicitly a local development application of the production
API. This work does not automatically qualify or deploy a new live campaign,
modify existing model configurations, or replace legacy published results.

For the Chinese design decisions and limitations see
[the visual-validation record](optical14_validation_zh.md) and
[the literature-grounded scientific assessment](optical14_scientific_review_zh.md).

The assessment measures eligible non-marker core response rather than treating
whole-image MAE as sensitivity. F6 recovery and F7 transfer probes are explicitly
analytical stimuli, not recorded hardware experiments. Its optional `--source-root`
adds fixed-ROI native pixel zooms; their replayed production trace hashes must
exactly match the gallery. No contrast normalization or fault-strength tuning is
performed by the zoom panel. `observation_level_acceptance_passed` is not physical
calibration or paper-result certification. Inapplicable no-contact/sub-knee/no-history
cases have specific failure reasons, not fabricated successful delivery evidence.

The profile's functional tests cover the fourteen IDs, legacy preservation,
marker continuity under displacement, locality, severity, causal history,
cadence mismatch, round trips and altered-output rejection. These checks support
software correctness, not physical calibration or closed-loop robustness claims.

## Chinese mechanism panels

For a readable main-figure selection (F4/C2, T2 and T1/T3), use the same prepared
source and gallery. Pass a local CJK font path; the font is not bundled in the
wheel. For example, on macOS:

```bash
PYTHONPATH=src python scripts/render_optical14_paper_panels.py \
  --source-root deployment/artifacts/optical14-sources/calibrated_standalone_v1 \
  --gallery-root deployment/artifacts/optical14-gallery-v7science \
  --output-root deployment/artifacts/optical14-paper-panels-v2 \
  --font '/System/Library/Fonts/STHeiti Light.ttc'
```

On Linux, supply an installed Noto Sans CJK font instead. Outputs must be new.
The source-specific layout intentionally binds `pull_out_key/raw55`, observation
121 and its actual clock. It generates three PNGs and `panel_receipt.json`,
including exact production traces, freeze/resumption checks and source-time maps.
No model runs are needed. F2 and F6 are deliberately not presented as visually
conclusive standalone main figures; see the Chinese scientific review for why.

## Stronger, explicitly separate stress profile

`optical_marker_stress_v1` increases all fourteen operator doses relative to
standard optical S5, while retaining the same production delivery path. It accepts
level 5 only; it is not a replacement for standard S1-S5. See the
[Chinese strength table, reproduction command and visual limitations](optical14_stress_zh.md).
The stronger F2 example makes central contact-edge suppression visible, but the
stronger F6 example still does not establish physically calibrated recovery.
