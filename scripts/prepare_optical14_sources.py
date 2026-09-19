"""Verify official optical rerender parity and bind a calibration-only rest field.

No recorded payload is replaced. Original RGB phase codes are retained as
unqualified change proxies; depth metrics do not automatically relabel phases.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any, cast

import cv2
import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

from robotactile_benchmark.contracts import Array
from robotactile_benchmark.visualization.optical14 import (
    file_hash,
    font,
    grid,
    load_source,
    source_array_hash,
    tile,
    write_json,
)

LOGGER = logging.getLogger(__name__)
LOCAL_UNLOADING_FRACTION = 0.02
MIN_UNLOADING_OBSERVATIONS = 3


def unloading_segments(values: Array) -> list[list[int]]:
    indices = np.flatnonzero(values >= LOCAL_UNLOADING_FRACTION * 240 * 320)
    if not len(indices):
        return []
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return [run.tolist() for run in runs if len(run) >= MIN_UNLOADING_OBSERVATIONS]


def official_renderer(univtac_root: Path) -> tuple[Any, dict[str, str]]:
    """Load the standalone official CPU module; never initialize Isaac."""
    source = univtac_root / "third_party/TacEx/source"
    package = source / "tacex/tacex/simulation_approaches/gpu_taxim"
    calibration = (
        source / "tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/calibs/640x480"
    )
    if importlib.util.find_spec("torch_scatter") is None:
        optional = types.ModuleType("torch_scatter")

        def unavailable_shadow(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "torch_scatter shadow path is unavailable; only with_shadow=False is authorized"
            )

        optional.__dict__["scatter_min"] = unavailable_shadow
        optional.__spec__ = importlib.util.spec_from_loader(
            "torch_scatter", loader=None
        )
        sys.modules["torch_scatter"] = optional
    sys.path.insert(0, str(package.resolve()))
    from sim.taxim_torch import TaximTorch

    hashes = {
        str(path.relative_to(univtac_root)): file_hash(path)
        for path in [
            package / "sim/taxim_torch.py",
            package / "sim/taxim_impl.py",
            *[
                calibration / name
                for name in (
                    "params.json",
                    "dataPack.npz",
                    "gelmap.npy",
                    "polycalib.npz",
                    "shadowTable.npz",
                )
            ],
            source / "tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py",
        ]
    }
    return TaximTorch(calib_folder=calibration, device="cpu"), hashes


def render(renderer: Any, depth: Array, pressure: Array) -> Array:
    with torch.no_grad():
        output = renderer.render_direct(
            torch.from_numpy(depth.copy()),
            with_shadow=False,
            press_depth=torch.from_numpy(pressure.copy()),
            orig_hm_fmt=False,
        )
        return cast(Array, (output.movedim(1, 3) * 255).to(torch.uint8).cpu().numpy())


def prepare(
    source_root: Path,
    raw_root: Path,
    univtac_root: Path,
    output_root: Path,
    max_mae: float = 5.0,
) -> None:
    if output_root.exists():
        raise FileExistsError(output_root)
    if not np.isfinite(max_mae) or not 0 < max_mae <= 5.0:
        raise ValueError("rerender JPEG-tolerance MAE must be finite and in (0, 5]")
    torch.set_num_threads(2)
    receipt = json.loads((source_root / "source_receipt.json").read_text())
    renderer, hashes = official_renderer(univtac_root)
    reference = render(
        renderer, np.full((1, 240, 320), 34, np.float32), np.zeros(1, np.float32)
    )[0]
    # Validate the physical invariance: any entirely distant plane gives zero indentation.
    alternate = render(
        renderer, np.full((1, 240, 320), 40, np.float32), np.zeros(1, np.float32)
    )[0]
    if not np.array_equal(reference, alternate):
        raise RuntimeError(
            "zero-indentation reference changes with distant plane distance"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    provenance: dict[str, Any] = {
        "reference_kind": "official_calibration_zero_indentation_optical_field",
        "not_a_recorded_no_contact_frame": True,
        "not_force_or_contact_certification": True,
        "shadow_dependency": "fail-on-call optional import stub; with_shadow=False only",
        "renderer_and_asset_sha256": hashes,
        "renderer_device": "cpu",
        "torch_version": torch.__version__,
        "zero_indentation_input": {
            "shape": [1, 240, 320],
            "depth_mm": 34.0,
            "press_depth_mm": 0.0,
            "orig_hm_fmt": False,
        },
        "zero_indentation_plane_invariance_passed": True,
        "reference_array_sha256": source_array_hash(reference),
        "max_allowed_recorded_rgb_mae": max_mae,
        "phase_labels": "original RGB change proxies preserved; no global no-contact/release claim",
        "local_unloading_rule": {
            "threshold_fraction_of_full_image": LOCAL_UNLOADING_FRACTION,
            "minimum_consecutive_observations": MIN_UNLOADING_OBSERVATIONS,
            "pixel_test": "depth[t-1] <28.5mm AND depth[t] >=28.5mm",
            "phase_release_means": "local surface unloading only; gripper remains in contact",
        },
        "episodes": [],
    }
    for spec in receipt["sources"]:
        if Path(spec["artifact"]).name != spec["artifact"]:
            raise ValueError("source artifact must use an unambiguous plain filename")
        raw_path = (raw_root / spec["raw_relative_path"]).resolve(strict=True)
        if not raw_path.is_relative_to(raw_root.resolve()):
            raise ValueError("raw path escapes root")
        if file_hash(raw_path) != spec["raw_hdf5_sha256"]:
            raise ValueError("raw HDF5 hash mismatch")
        arrays = load_source(source_root, spec)
        selected = sorted(
            {
                0,
                *spec["reference_indices"],
                spec["contact_peak"],
                *(
                    value
                    for value in [spec["release_start"], spec["release_confirmation"]]
                    if value is not None
                ),
            }
        )
        checks, metrics = [], {}
        unloading_masks = []
        per_slot_phases = []
        with h5py.File(raw_path, "r") as handle:
            for slot in ("left", "right"):
                depth = handle[f"tactile/{slot}_gsmini/depth"][:]
                minima = depth.min((1, 2))
                pressure = np.maximum(28.5 - minima, 0).astype(np.float32)
                contact = depth < 28.5
                unloaded = np.zeros(len(depth), dtype=np.int64)
                newly_loaded = np.zeros(len(depth), dtype=np.int64)
                unloaded[1:] = (contact[:-1] & ~contact[1:]).sum((1, 2))
                newly_loaded[1:] = (~contact[:-1] & contact[1:]).sum((1, 2))
                mask = np.zeros_like(contact)
                mask[1:] = contact[:-1] & ~contact[1:]
                unloading_masks.append(mask)
                segments = unloading_segments(unloaded)
                phases = np.where(pressure > 0, 2, 0).astype(np.uint8)
                for segment in segments:
                    phases[segment] = 3
                per_slot_phases.append(phases)
                metrics[slot] = {
                    "depth_min_mm": minima.tolist(),
                    "depth_max_mm": depth.max((1, 2)).tolist(),
                    "renderer_indentation_mm": pressure.tolist(),
                    "pixels_below_nominal_gel_surface": contact.sum((1, 2)).tolist(),
                    "newly_unloaded_pixels": unloaded.tolist(),
                    "newly_loaded_pixels": newly_loaded.tolist(),
                    "largest_local_unloading_indices": np.argsort(-unloaded)[
                        :10
                    ].tolist(),
                    "zero_indentation_indices": np.flatnonzero(pressure == 0).tolist(),
                    "qualified_local_unloading_segments": segments,
                }
                predictions = render(renderer, depth[selected], pressure[selected])
                for position, index in enumerate(selected):
                    encoded = np.frombuffer(
                        handle[f"tactile/{slot}_gsmini/rgb"][index], dtype=np.uint8
                    )
                    recorded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    difference = np.abs(
                        predictions[position].astype(np.int16)
                        - recorded.astype(np.int16)
                    )
                    check = {
                        "slot": slot,
                        "index": index,
                        "mae_uint8": float(difference.mean()),
                        "p99_absolute_error": float(np.percentile(difference, 99)),
                        "max_absolute_error": int(difference.max()),
                        "recorded_rgb_array_sha256": source_array_hash(recorded),
                        "rerender_array_sha256": source_array_hash(
                            predictions[position]
                        ),
                    }
                    checks.append(check)
                    LOGGER.info(
                        "%s %s[%s] rerender RGB MAE %.4f",
                        spec["task"],
                        slot,
                        index,
                        check["mae_uint8"],
                    )
        episode = {
            "task": spec["task"],
            "raw_episode_id": spec["raw_episode_id"],
            "rerender_checks": checks,
            "depth_metrics": metrics,
        }
        provenance["episodes"].append(episode)
        write_json(output_root / "reference_preparation.json", provenance)
        if any(check["mae_uint8"] > max_mae for check in checks):
            raise RuntimeError(
                "official rerender does not match recorded RGB within declared JPEG tolerance; see reference_preparation.json"
            )
        arrays["optical_reference_rgb"] = np.stack([reference, reference])
        arrays["depth_local_unloading_mask"] = np.stack(unloading_masks, axis=1)
        arrays["depth_phase_code"] = np.stack(per_slot_phases, axis=1)
        reference_panels = []
        for column, slot in enumerate(("left", "right")):
            index = int(arrays["reference_indices"][column])
            reference_panels.extend(
                [
                    tile(
                        arrays["tactile_rgb"][index, column],
                        f"Old RGB proxy reference: {slot}[{index}]\nObject remains grasped; not no-contact",
                    ),
                    tile(
                        reference,
                        "Official zero-indentation optical field\nCalibration-derived; marker-free",
                    ),
                ]
            )
        grid(
            reference_panels,
            2,
            f"{spec['task']} reference correction\n"
            "Left: untouched recorded image. Right: same official renderer + pinned calibration.",
        ).save(output_root / f"{spec['task']}_reference_comparison.png")
        curve = Image.new("RGB", (1000, 360), "white")
        drawing = ImageDraw.Draw(curve)
        drawing.text(
            (14, 8),
            f"{spec['task']} clean depth: newly locally unloaded pixels / full image\n"
            "Criterion >=2% for >=3 observations; this is NOT global contact release",
            font=font(18),
            fill="black",
        )
        drawing.line((60, 90, 60, 310, 980, 310), fill="black", width=2)
        maximum = max(
            0.04, max(max(m["newly_unloaded_pixels"]) / 76800 for m in metrics.values())
        )
        for slot, color in (("left", "#2059af"), ("right", "#c15b24")):
            values = np.asarray(metrics[slot]["newly_unloaded_pixels"]) / 76800
            points = [
                (60 + i / (len(values) - 1) * 920, 310 - value / maximum * 200)
                for i, value in enumerate(values)
            ]
            drawing.line(points, fill=color, width=2)
            drawing.text(
                (780, 72 if slot == "left" else 92), slot, font=font(17), fill=color
            )
        threshold_y = 310 - LOCAL_UNLOADING_FRACTION / maximum * 200
        drawing.line((60, threshold_y, 980, threshold_y), fill="#999999", width=1)
        drawing.text(
            (65, threshold_y - 20), "2% threshold", font=font(15), fill="#555555"
        )
        drawing.text(
            (60, 320),
            f"0 s                                      recorded observation time (60 Hz)                              {(len(values) - 1) / 60:.3f} s",
            font=font(16),
            fill="black",
        )
        curve.save(output_root / f"{spec['task']}_local_unloading_curve.png")
        artifact = output_root / spec["artifact"]
        np.savez_compressed(artifact, **cast(dict[str, Any], arrays))
        spec["original_artifact_sha256"] = spec["artifact_sha256"]
        spec["artifact_sha256"] = file_hash(artifact)
        spec["arrays"] = {
            key: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": source_array_hash(value),
            }
            for key, value in arrays.items()
        }
        spec["rest_reference_kind"] = provenance["reference_kind"]
        spec["phase_semantics"] = (
            "depth_phase_code per slot: 0 zero-indentation; 2 contact; 3 local unloading, not global release"
        )
    candidates = [
        (
            sum(
                episode["depth_metrics"][slot]["newly_unloaded_pixels"][index]
                for index in segment
            ),
            episode["task"],
            slot,
            segment,
        )
        for episode in provenance["episodes"]
        for slot in ("left", "right")
        for segment in episode["depth_metrics"][slot][
            "qualified_local_unloading_segments"
        ]
    ]
    if not candidates:
        raise RuntimeError(
            "no sustained local unloading segment meets declared clean-depth rule"
        )
    _, task, slot, segment = max(candidates)
    receipt["f6_local_unloading_witness"] = {
        "task": task,
        "slot": slot,
        "indices": segment,
        "start_index": max(0, segment[0] - 10),
        "stop_index": segment[-1] + 15,
        "selection": "largest summed newly-unloaded area among qualifying sustained depth segments",
        "not_global_release": True,
    }
    receipt["calibration_reference"] = {
        "artifact": "reference_preparation.json",
        "artifact_sha256": file_hash(output_root / "reference_preparation.json"),
        "kind": provenance["reference_kind"],
        "original_recorded_rgb_unchanged": True,
        "phase_codes_are_unqualified_rgb_change_proxies": True,
    }
    receipt["parent_source_receipt_sha256"] = file_hash(
        source_root / "source_receipt.json"
    )
    write_json(output_root / "source_receipt.json", receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source-root", "raw-root", "univtac-root", "output-root"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--max-mae", type=float, default=5.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare(
        args.source_root.resolve(),
        args.raw_root.resolve(),
        args.univtac_root.resolve(),
        args.output_root.resolve(),
        args.max_mae,
    )


if __name__ == "__main__":
    main()
