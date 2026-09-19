"""Pillow gallery of production deliveries, with native pixels and provenance.

Inputs are the raw_fault_visualization_sources v1 extractor's JSON/NPZ bundle.
The renderer does not implement faults, recolor payloads, or generate images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast, overload

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    SENSOR_SLOTS,
    operator_requires_rest_reference,
)
from robotactile_benchmark.contracts import (
    Array,
    ContactPhase,
    EvaluationRecord,
    ObservationRecord,
    SensorObservation,
    SensorProvenance,
    array_sha256,
    build_evaluation_record,
    canonical_hash,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.rest_references import ReferenceSplit, RestReferenceBundle
from robotactile_benchmark.runtime import apply_fault

LOGGER = logging.getLogger(__name__)
REGISTRY = "optical_marker_v1"
BOUNDARY = (
    "Recorded UniVTAC simulator RGB; production synthetic fault delivery. "
    "Development visualization only; RGB phase proxy is not force/contact ground truth. "
    "No calibration certification, policy evaluation, closed-loop or real-robot claim."
)
ORDER = sorted(CORE_OPERATOR_IDS, key=lambda value: ("AFTC".index(value[0]), value))
FOCUS = {"F2", "F4", "F6", "F7"}
PHASES = (
    ContactPhase.FREE,
    ContactPhase.ONSET,
    ContactPhase.SUSTAINED,
    ContactPhase.RELEASE,
)
WITNESS_RULES = {
    "A1": "first registered affected offset",
    "A2": "first registered erased offset",
    "C1": "first registered affected offset",
    "F1": "last observation inside fault window",
    "F6": "last frame of qualified sustained clean-depth local-unloading segment",
    "T2": "last registered held-frame observation",
    "default": "fixed source contact_peak selected before corruption",
}


def production_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = [
        root / name
        for name in (
            "runtime.py",
            "operator_parameters.py",
            "constants.py",
            "severity.py",
            "signature_validators.py",
            "validators.py",
            "operator_validators.py",
            "contracts.py",
            "manifests.py",
            "rest_references.py",
        )
    ]
    for directory in ("optical", "operators", "streaming"):
        files.extend((root / directory).glob("*.py"))
    hashes = {str(path.relative_to(root)): file_hash(path) for path in sorted(files)}
    for path in sorted((root.parents[1] / "schemas").glob("*.json")):
        hashes[f"schemas/{path.name}"] = file_hash(path)
    return hashes


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_array_hash(value: Array) -> str:
    """The extractor's array hash differs from the runtime's canonical hash."""
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_source(root: Path, spec: dict[str, Any]) -> dict[str, Array]:
    artifact = (root / spec["artifact"]).resolve(strict=True)
    if not artifact.is_relative_to(root.resolve()):
        raise ValueError("source artifact escapes source root")
    if file_hash(artifact) != spec["artifact_sha256"]:
        raise ValueError("source artifact hash mismatch")
    with np.load(artifact, allow_pickle=False) as archive:
        if set(archive.files) != set(spec["arrays"]):
            raise ValueError("source array set mismatch")
        arrays = {key: archive[key] for key in archive.files}
    for key, value in arrays.items():
        descriptor = spec["arrays"][key]
        if (str(value.dtype), list(value.shape), source_array_hash(value)) != (
            descriptor["dtype"],
            descriptor["shape"],
            descriptor["sha256"],
        ):
            raise ValueError(f"source array hash/shape/dtype mismatch: {key}")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"nonfinite source array: {key}")
    length = len(arrays["source_time_s"])
    if arrays["tactile_rgb"].shape != (length, 2, 240, 320, 3):
        raise ValueError("expected native 240x320 two-slot tactile frames")
    if arrays["tactile_rgb"].dtype != np.uint8:
        raise ValueError("expected uint8 native source")
    if not np.array_equal(arrays["reference_indices"], spec["reference_indices"]):
        raise ValueError("rest-reference index mismatch")
    times = arrays["source_time_s"]
    differences = np.diff(times)
    if not len(differences) or not np.all(differences > 0):
        raise ValueError("source timestamps must increase")
    if not np.allclose(differences, differences[0], atol=1e-9, rtol=0):
        raise ValueError("optical gallery requires uniform recorded observation clock")
    if not np.allclose(times, (arrays["source_step"] - arrays["source_step"][0]) / 120):
        raise ValueError(
            "source steps disagree with extractor's 120 Hz simulator clock"
        )
    return arrays


def build_records(
    arrays: dict[str, Array], spec: dict[str, Any], receipt_hash: str
) -> tuple[tuple[EvaluationRecord, ...], RestReferenceBundle]:
    episode = f"{spec['task']}/raw{spec['raw_episode_id']}"
    calibration = {
        slot: canonical_hash(
            {
                "source": spec["artifact_sha256"],
                "slot": slot,
                "binding": "development_rgb_identity_only",
            }
        )
        for slot in SENSOR_SLOTS
    }
    records = []
    for index, time_s in enumerate(arrays["source_time_s"]):
        sensors, provenance = [], []
        for column, slot in enumerate(SENSOR_SLOTS):
            payload = arrays["tactile_rgb"][index, column]
            sensors.append(
                SensorObservation(
                    slot_id=slot,
                    payload=payload,
                    payload_present=True,
                    declared_validity=True,
                    delivery_index=index,
                    delivery_time_s=float(time_s),
                    visible_source_time_s=float(time_s),
                    frame_id=f"{episode}/{slot}/{index}",
                    calibration_id=f"{episode}/{slot}/development-identity",
                )
            )
            provenance.append(
                SensorProvenance(
                    slot_id=slot,
                    physical_source_id=slot,
                    source_index=index,
                    source_time_s=float(time_s),
                    payload_sha256=array_sha256(payload),
                    calibration_sha256=calibration[slot],
                    phase=PHASES[int(arrays["depth_phase_code"][index, column])],
                )
            )
        observation = ObservationRecord(
            episode_id=episode,
            task=spec["task"],
            seed=int(spec["raw_episode_id"]),
            step_index=index,
            tactile=tuple(sensors),
            vision={"wrist": arrays["wrist_rgb"][index]},
            proprio=arrays["proprio"][index],
        )
        records.append(build_evaluation_record(observation, tuple(provenance)))
    if (
        spec.get("rest_reference_kind")
        != "official_calibration_zero_indentation_optical_field"
        or "optical_reference_rgb" not in arrays
    ):
        raise ValueError(
            "recorded RGB change proxy cannot certify a no-contact rest reference"
        )
    reference = arrays["optical_reference_rgb"]
    if reference.shape != (2, 240, 320, 3) or reference.dtype != np.uint8:
        raise ValueError("invalid calibrated optical reference shape/dtype")
    predicate = {
        "proxy_id": "official_taxim_zero_indentation_plane_invariance_v1",
        "reference_sha256": source_array_hash(reference),
        "scope": "calibration-derived optical field; not recorded free sensor",
    }
    rest = RestReferenceBundle(
        reference_id=f"{episode}/calibrated-optical-rest",
        dataset_split=ReferenceSplit.CALIBRATION,
        split_manifest_sha256=receipt_hash,
        source_artifact_sha256=spec["artifact_sha256"],
        no_contact_predicate_id=predicate["proxy_id"],
        no_contact_validation_sha256=canonical_hash(predicate),
        no_contact_verified=True,
        qualified_record_ids={
            slot: f"{episode}/calibration-zero-indentation/{slot}"
            for slot in SENSOR_SLOTS
        },
        calibration_sha256=calibration,
        payloads={slot: reference[column] for column, slot in enumerate(SENSOR_SLOTS)},
    )
    return tuple(records), rest


def make_manifest(
    operator: str,
    severity: int,
    arrays: dict[str, Array],
    spec: dict[str, Any],
    rest: RestReferenceBundle,
) -> FaultManifest:
    peak, length = int(spec["contact_peak"]), len(arrays["source_time_s"])
    start, stop = max(1, peak - 16), min(length - 1, peak + 10)
    if operator.startswith("F6_"):
        witness = spec["f6_local_unloading_witness"]
        start, stop = (
            int(witness["start_index"]),
            min(length, int(witness["stop_index"])),
        )
    parameters: dict[str, Any] = {
        "sample_period_s": float(np.diff(arrays["source_time_s"])[0])
    }
    if operator_requires_rest_reference(operator, severity_registry=REGISTRY):
        parameters["rest_reference_sha256"] = rest.sha256
    if operator.startswith("C2_"):
        parameters["realization"] = "registered_pixels"
    slots = SENSOR_SLOTS if operator.startswith(("C1_", "T1_", "T3_")) else ("right",)
    if operator.startswith("F6_"):
        slots = (spec["f6_local_unloading_witness"]["slot"],)
    return FaultManifest(
        operator_id=operator,
        severity_level=severity,
        operator_seed=20260906,
        start_index=start,
        stop_index=stop,
        sensor_slots=slots,
        observability=Observability.BLIND,
        parameters=parameters,
        severity_registry=REGISTRY,
    )


def font(size: int = 18) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def tile(payload: Array | None, label: str, scale: int = 1) -> Image.Image:
    result = Image.new("RGB", (320 * scale, 240 * scale + 54), "#f7f8fa")
    drawing = ImageDraw.Draw(result)
    drawing.text((8, 5), label, fill="#202634", font=font(16))
    if payload is None:
        drawing.rectangle((0, 54, result.width - 1, result.height - 1), fill="#e6e8ed")
        drawing.text(
            (20, 130),
            "NO PAYLOAD\nnot a black tactile frame",
            fill="#a02020",
            font=font(17),
        )
    else:
        image = Image.fromarray(payload)
        if scale != 1:
            image = image.resize((320 * scale, 240 * scale), Image.Resampling.NEAREST)
        result.paste(image, (0, 54))
    return result


@overload
def delta(clean: Array, delivered: None) -> None: ...


@overload
def delta(clean: Array, delivered: Array) -> Array: ...


def delta(clean: Array, delivered: Array | None) -> Array | None:
    if delivered is None:
        return None
    return cast(
        Array,
        np.abs(delivered.astype(np.int16) - clean.astype(np.int16)).astype(np.uint8),
    )


def roi_diagnostic(clean: Array, manifest: FaultManifest) -> Image.Image:
    """Annotate only a separate diagnostic copy, never delivered image payloads."""
    from robotactile_benchmark.optical.fields import spatial_weight

    dead = manifest.operator_id.startswith("F4_")
    weight = spatial_weight(clean.shape, manifest.parameters, dead_patch=dead)
    overlay = Image.fromarray(clean.copy())
    cx, cy = (float(value) for value in manifest.parameters["center_xy"])
    center_x, center_y = cx * (clean.shape[1] - 1), cy * (clean.shape[0] - 1)
    core = float(
        manifest.parameters["radius_fraction"]
        if dead
        else manifest.parameters["core_radius_fraction"]
    )
    support = (
        core + float(manifest.parameters["edge_feather_fraction"])
        if dead
        else float(manifest.parameters["support_radius_fraction"])
    )
    drawing = ImageDraw.Draw(overlay)
    for fraction, color in ((support, "#ffee00"), (core, "#ff9900")):
        radius = fraction * min(clean.shape[:2])
        drawing.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            outline=color,
            width=1,
        )
    mask = np.repeat(np.rint(weight[..., None] * 255).astype(np.uint8), 3, axis=-1)
    return grid(
        [
            tile(
                np.asarray(overlay),
                "DIAGNOSTIC ONLY: support/core\nYellow=outer, orange=inner",
            ),
            tile(mask, "Production spatial_weight / 0-1\nWhite=1; black=0"),
        ],
        2,
        f"{manifest.operator_id}: fixed ROI diagnostic\nNative clean and delivered cards remain unannotated",
    )


def grid(tiles: list[Image.Image], columns: int, title: str) -> Image.Image:
    width, height = tiles[0].size
    total_width = columns * (width + 12) + 12
    heading_font = font(20)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = []
    for paragraph in title.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if (
                line
                and measure.textlength(candidate, font=heading_font) > total_width - 24
            ):
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    heading_height = 30 + len(lines) * 24
    result = Image.new(
        "RGB",
        (
            total_width,
            ((len(tiles) + columns - 1) // columns) * (height + 12)
            + heading_height
            + 12,
        ),
        "white",
    )
    ImageDraw.Draw(result).text(
        (12, 12), "\n".join(lines), fill="#172338", font=heading_font, spacing=4
    )
    for index, image in enumerate(tiles):
        result.paste(
            image,
            (
                12 + (index % columns) * (width + 12),
                heading_height + (index // columns) * (height + 12),
            ),
        )
    return result


def display_index(manifest: FaultManifest, spec: dict[str, Any]) -> int:
    """Choose only from frozen clean source anchors and registered schedules."""
    prefix = manifest.operator_id.split("_")[0]
    if prefix in {"A1", "C1"}:
        index = manifest.start_index + int(manifest.parameters["affected_offsets"][0])
    elif prefix == "A2":
        index = manifest.start_index + int(manifest.parameters["erased_offsets"][0])
    elif prefix == "F1":
        index = manifest.stop_index - 1
    elif prefix == "F6":
        index = int(spec["f6_local_unloading_witness"]["indices"][-1])
    elif prefix == "T2":
        index = (
            manifest.start_index + int(manifest.parameters["hold_duration_frames"]) - 1
        )
    else:
        index = int(spec["contact_peak"])
    if not manifest.start_index <= index < manifest.stop_index:
        raise ValueError("frozen clean witness lies outside registered fault window")
    return index


def sequence_caption(prefix: str, task: str, episode: int) -> str:
    title = f"{prefix} sequence | one continuous {task} raw{episode} episode\n"
    title += "Times are recorded observation times"
    if prefix == "F6":
        title += "; F6 is LOCAL unloading, not global release"
    return title


def source_map(
    records: tuple[EvaluationRecord, ...], arrays: dict[str, Array]
) -> list[dict[str, Any]]:
    return [
        {
            "delivery_index": index,
            "delivery_time_s": float(arrays["source_time_s"][index]),
            "slots": {
                slot: {
                    "source_index": record.provenance_for(slot).source_index,
                    "source_simulator_step": (
                        None
                        if record.provenance_for(slot).source_index is None
                        else int(
                            arrays["source_step"][
                                record.provenance_for(slot).source_index
                            ]
                        )
                    ),
                    "source_time_s": record.provenance_for(slot).source_time_s,
                    "physical_source_id": record.provenance_for(
                        slot
                    ).physical_source_id,
                    "payload_present": record.observation.sensor(slot).payload_present,
                    "payload_sha256": record.provenance_for(slot).payload_sha256,
                }
                for slot in SENSOR_SLOTS
            },
        }
        for index, record in enumerate(records)
    ]


def render_gallery(
    source_root: Path, output_root: Path, severity: int = 3
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"no-clobber output already exists: {output_root}")
    initial_production_hashes = production_hashes()
    receipt_path = source_root / "source_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema_id") != "robotactile.raw_fault_visualization_sources"
        or receipt.get("schema_version") != "1.0"
        or receipt.get("pixel_contract") != "opencv_decode_only_uint8_rgb"
        or receipt.get("source_encoding_contract") != "opencv_imencode_rgb_input_v1"
        or any(
            receipt.get(key) is not False
            for key in (
                "fault_operator_applied",
                "normalization_applied",
                "enhancement_applied",
                "generated_content",
                "contact_labels_are_force_ground_truth",
            )
        )
    ):
        raise ValueError("source receipt does not attest untouched recorded RGB")
    sources = {spec["task"]: spec for spec in receipt["sources"]}
    if set(sources) != {"pull_out_key", "insert_HDMI"}:
        raise ValueError(
            "gallery requires pull_out_key and insert_HDMI source episodes"
        )
    calibration = receipt.get("calibration_reference", {})
    if calibration.get("kind") != "official_calibration_zero_indentation_optical_field":
        raise ValueError(
            "gallery requires verified official calibration reference preparation"
        )
    preparation_path = (source_root / calibration["artifact"]).resolve(strict=True)
    if (
        not preparation_path.is_relative_to(source_root.resolve())
        or file_hash(preparation_path) != calibration["artifact_sha256"]
    ):
        raise ValueError("calibration reference preparation hash mismatch")
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root / "source_receipt.json", receipt)
    write_json(
        output_root / "reference_preparation.json",
        json.loads(preparation_path.read_text()),
    )
    for path in source_root.glob("*_local_unloading_curve.png"):
        (output_root / path.name).write_bytes(path.read_bytes())
    entries, atlas_cards, clocks = [], {}, {}
    availability = {}
    for task in ("pull_out_key", "insert_HDMI"):
        spec = dict(sources[task])
        spec["f6_local_unloading_witness"] = receipt["f6_local_unloading_witness"]
        arrays = load_source(source_root, spec)
        reference_tiles = []
        for column, slot in enumerate(SENSOR_SLOTS):
            old_index = int(arrays["reference_indices"][column])
            reference_tiles.extend(
                [
                    tile(
                        arrays["tactile_rgb"][old_index, column],
                        f"Old RGB proxy {slot}[{old_index}]\nObject held; not no-contact",
                    ),
                    tile(
                        arrays["optical_reference_rgb"][column],
                        "Official zero-indentation optical field\nCalibration-derived, marker-free",
                    ),
                ]
            )
        grid(
            reference_tiles,
            2,
            f"{task}: reference correction\n"
            "Recorded alleged rest vs pinned official zero-indentation renderer",
        ).save(output_root / f"{task}_reference_comparison.png")
        clean, rest = build_records(arrays, spec, file_hash(receipt_path))
        period = float(np.diff(arrays["source_time_s"])[0])
        clocks[task] = {
            "observation_hz": 1 / period,
            "sample_period_s": period,
            "simulator_hz": 120,
            "simulator_steps_per_observation": int(np.diff(arrays["source_step"])[0]),
            "seed_semantics": "raw episode replay identifier; simulator seed unknown",
        }
        for operator in ORDER:
            selected_task = (
                receipt["f6_local_unloading_witness"]["task"]
                if operator.startswith("F6_")
                else "insert_HDMI"
                if operator.startswith("F2_")
                else "pull_out_key"
            )
            if task != selected_task:
                continue
            prefix = operator.split("_")[0]
            manifest = make_manifest(operator, severity, arrays, spec, rest)
            result = apply_fault(clean, manifest, rest_references=rest)
            if not result.validation.passed:
                raise RuntimeError(
                    f"{operator} delivery validation failed: {result.validation.failures}"
                )
            index = display_index(manifest, spec)
            display_slot = manifest.sensor_slots[-1]
            before = clean[index].observation.sensor(display_slot).payload
            if before is None:
                raise ValueError("clean witness must have a recorded payload")
            after = result.records[index].observation.sensor(display_slot).payload
            time_s = float(arrays["source_time_s"][index])
            header = f"{operator} | S{severity} | {task} raw{spec['raw_episode_id']} | obs {index}, {time_s:.3f}s"
            header += f"\n{1 / period:g} Hz observations / 120 Hz simulation; native RGB, no enhancement"
            if prefix == "F6":
                header += "; LOCAL unloading, not global release"
            panels = [
                tile(before, f"Recorded clean / {display_slot}"),
                tile(after, f"Production delivery / {display_slot}"),
                tile(delta(before, after), "Absolute delta / raw 0-255"),
            ]
            if prefix in {"C1", "T1", "T3"}:
                left_before = clean[index].observation.sensor("left").payload
                if left_before is None:
                    raise ValueError("clean left witness must have a recorded payload")
                left_after = result.records[index].observation.sensor("left").payload
                panels.extend(
                    [
                        tile(left_before, "Recorded clean / left"),
                        tile(left_after, "Production delivery / left"),
                        tile(delta(left_before, left_after), "Absolute delta / left"),
                    ]
                )
            atlas_cards[operator] = grid(panels, 3, header)
            large_panels = [
                tile(before, f"Recorded clean / {display_slot}", 2),
                tile(after, f"Production delivery / {display_slot}", 2),
                tile(delta(before, after), "Absolute delta / raw 0-255", 2),
            ]
            if prefix in {"C1", "T1", "T3"}:
                assert left_before is not None
                large_panels.extend(
                    [
                        tile(left_before, "Recorded clean / left", 2),
                        tile(left_after, "Production delivery / left", 2),
                        tile(
                            delta(left_before, left_after), "Absolute delta / left", 2
                        ),
                    ]
                )
            large = grid(large_panels, 3, header)
            large.save(output_root / f"{prefix}_card.png")
            if prefix in {"F2", "F4"}:
                roi_diagnostic(before, manifest).save(
                    output_root / f"{prefix}_roi_diagnostic.png"
                )
            entry = {
                "operator_id": operator,
                "source_task": task,
                "source_episode": spec["raw_episode_id"],
                "display_index": index,
                "display_time_s": time_s,
                "display_slot": display_slot,
                "display_selection": WITNESS_RULES.get(
                    prefix, WITNESS_RULES["default"]
                ),
                "manifest": manifest.to_dict(),
                "manifest_sha256": manifest.sha256,
                "rest_reference_sha256": rest.sha256,
                "rest_reference_scope": "official calibration-derived zero-indentation optical field; not recorded no-contact frame",
                "validation": asdict(result.validation),
                "trace_sha256": result.trace_sha256,
                "delivery_source_map": source_map(result.records, arrays),
                "display_mean_absolute_delta": None
                if after is None
                else float(np.mean(delta(before, after))),
            }
            if prefix in {"A1", "A2"}:
                availability[prefix] = {
                    "presence": [
                        int(record.observation.sensor(display_slot).payload_present)
                        for record in result.records
                    ],
                    "times_s": arrays["source_time_s"].tolist(),
                    "start_index": manifest.start_index,
                    "stop_index": manifest.stop_index,
                    "source_task": task,
                    "slot": display_slot,
                }
            if prefix in {"T1", "T2", "T3", "F6"}:
                if prefix == "F6":
                    segment = spec["f6_local_unloading_witness"]["indices"]
                    indices = sorted(
                        set(
                            [
                                manifest.start_index,
                                segment[0] - 1,
                                *segment,
                                min(len(clean) - 1, segment[-1] + 5),
                            ]
                        )
                    )
                    entry["local_unloading_witness"] = spec[
                        "f6_local_unloading_witness"
                    ]
                elif prefix == "T2":
                    duration = int(manifest.parameters["hold_duration_frames"])
                    indices = sorted(
                        set(
                            [
                                manifest.start_index - 1,
                                manifest.start_index,
                                manifest.start_index + duration - 1,
                                min(len(clean) - 1, manifest.start_index + duration),
                                manifest.stop_index - 1,
                                manifest.stop_index,
                            ]
                        )
                    )
                else:
                    indices = np.linspace(
                        manifest.start_index - 1, manifest.stop_index, 6, dtype=int
                    ).tolist()
                sequence = []
                for frame in indices:
                    for slot in (
                        SENSOR_SLOTS if prefix in {"T1", "T3"} else (display_slot,)
                    ):
                        provenance = result.records[frame].provenance_for(slot)
                        sequence.extend(
                            [
                                tile(
                                    clean[frame].observation.sensor(slot).payload,
                                    f"Clean {slot} {frame} / {arrays['source_time_s'][frame]:.3f}s",
                                ),
                                tile(
                                    result.records[frame]
                                    .observation.sensor(slot)
                                    .payload,
                                    f"Delivery {slot} {frame}\nsrc {provenance.source_index} / {provenance.source_time_s:.3f}s",
                                ),
                            ]
                        )
                grid(
                    sequence,
                    4 if prefix in {"T1", "T3"} else 2,
                    sequence_caption(prefix, task, int(spec["raw_episode_id"])),
                ).save(output_root / f"{prefix}_sequence.png")
                if prefix == "F6":
                    column = SENSOR_SLOTS.index(display_slot)
                    mask_tiles = [
                        tile(
                            np.repeat(
                                arrays["depth_local_unloading_mask"][
                                    frame, column, ..., None
                                ],
                                3,
                                axis=-1,
                            ).astype(np.uint8)
                            * 255,
                            f"Clean-depth local-unload mask {frame}",
                        )
                        for frame in indices
                    ]
                    grid(
                        mask_tiles,
                        3,
                        "F6 local unloading witness masks from recorded depth\n"
                        "White: previously depth<28.5mm, now>=28.5mm; no global release claim",
                    ).save(output_root / "F6_depth_masks.png")
                entry["sequence_indices"] = indices
            del result
            if prefix in FOCUS:
                strips, severities = (
                    [tile(before, f"Recorded clean / {display_slot}")],
                    [],
                )
                for level in range(1, 6):
                    level_manifest = make_manifest(operator, level, arrays, spec, rest)
                    level_result = apply_fault(
                        clean, level_manifest, rest_references=rest
                    )
                    if not level_result.validation.passed:
                        raise RuntimeError(
                            f"{operator} S{level}: {level_result.validation.failures}"
                        )
                    payload = (
                        level_result.records[index]
                        .observation.sensor(display_slot)
                        .payload
                    )
                    strips.append(tile(payload, f"S{level} / same observation {index}"))
                    if payload is None:
                        raise ValueError(
                            "fidelity severity witness must have a delivered payload"
                        )
                    severities.append(
                        {
                            "severity_level": level,
                            "manifest": level_manifest.to_dict(),
                            "validation": asdict(level_result.validation),
                            "trace_sha256": level_result.trace_sha256,
                            "display_mean_absolute_delta": float(
                                np.mean(delta(before, payload))
                            ),
                        }
                    )
                    del level_result
                grid(
                    strips,
                    6,
                    f"{prefix} S1-S5 | same native source {task} raw{spec['raw_episode_id']}, obs {index}\n"
                    "Production deliveries; shared raw RGB scale; no independent contrast normalization",
                ).save(output_root / f"{prefix}_S1-S5.png")
                entry["severity_sweep"] = severities
            write_json(output_root / f"{prefix}_provenance.json", entry)
            entries.append(entry)
            LOGGER.info("validated and rendered %s", operator)
        del clean, rest, arrays
    timeline = Image.new("RGB", (1100, 330), "white")
    drawing = ImageDraw.Draw(timeline)
    drawing.text(
        (16, 12),
        "Availability timeline: actual delivered payload presence\nGreen=present (1); red=absent (0); no fabricated tactile RGB",
        fill="#172338",
        font=font(20),
    )
    for row, prefix in enumerate(("A1", "A2")):
        values = availability[prefix]
        left = max(0, values["start_index"] - 5)
        right = min(len(values["presence"]), values["stop_index"] + 5)
        y = 112 + row * 103
        drawing.text(
            (16, y), f"{prefix}\n{values['slot']}", fill="#172338", font=font(18)
        )
        for index in range(left, right):
            x0 = 95 + (index - left) / (right - left) * 985
            x1 = 95 + (index + 1 - left) / (right - left) * 985
            present = values["presence"][index]
            drawing.rectangle(
                (x0, y, x1 - 2, y + 35), fill="#369361" if present else "#cb4444"
            )
        for index in (left, values["start_index"], values["stop_index"], right - 1):
            x = 95 + (index - left) / (right - left) * 985
            drawing.line((x, y - 5, x, y + 38), fill="black", width=1)
            drawing.text(
                (min(x, 999), y + 42),
                f"{index}\n{values['times_s'][index]:.3f}s",
                fill="black",
                font=font(15),
            )
    timeline.save(output_root / "availability_timeline.png")
    width = max(card.width for card in atlas_cards.values())
    height = sum(atlas_cards[operator].height for operator in ORDER) + 105
    atlas = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(atlas).text(
        (12, 12),
        "RoboTactile optical_marker_v1 / 14 production operators\n"
        "Recorded simulator inputs; development visualization, not benchmark certification\n"
        "Each row: native clean / production fault / raw absolute delta (no scaling)",
        font=font(18),
        fill="#172338",
    )
    y = 105
    for operator in ORDER:
        atlas.paste(atlas_cards[operator], (0, y))
        y += atlas_cards[operator].height
    atlas.save(output_root / "all14_atlas.png")
    final_production_hashes = production_hashes()
    if initial_production_hashes != final_production_hashes:
        raise RuntimeError(
            "production source changed during generation; gallery is incomplete and must be regenerated"
        )
    summary = {
        "schema_id": "robotactile.optical14_gallery",
        "schema_version": "1.0",
        "severity_registry": REGISTRY,
        "severity_level": severity,
        "evidence_boundary": BOUNDARY,
        "source_receipt_sha256": file_hash(receipt_path),
        "clocks": clocks,
        "operator_count": len(entries),
        "all_delivery_validations_passed": True,
        "generated_image_content": False,
        "production_source_sha256": initial_production_hashes,
        "production_source_unchanged_during_generation": True,
        "display_witness_rules": WITNESS_RULES,
        "display_selection_uses_fault_outputs": False,
        "rendering": {
            "native_pixel_size": [320, 240],
            "large_card_zoom": "2x nearest",
            "delta": "absolute uint8 RGB difference, unscaled",
            "missing_payload": "labeled neutral card; no replacement tactile frame",
        },
        "output_sha256": {
            path.name: file_hash(path)
            for path in sorted(output_root.iterdir())
            if path.is_file()
        },
    }
    write_json(output_root / "gallery_receipt.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--severity", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    render_gallery(
        args.source_root.resolve(), args.output_root.resolve(), args.severity
    )
