"""Frozen calibration templates; no online outcome or future-frame adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robotactile_benchmark.contracts import thaw_value


def validate_spatial_calibration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a JSON template with both sensor slots and three nested doses."""
    data = thaw_value(value)
    if not isinstance(data, dict) or set(data) != {"version", "source_sha256", "slots"}:
        raise ValueError(
            "spatial_calibration fields must be version/source_sha256/slots"
        )
    if data["version"] != "frozen_contact_scar_v1":
        raise ValueError("unsupported spatial calibration version")
    sha = data["source_sha256"]
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(c not in "0123456789abcdef" for c in sha)
    ):
        raise ValueError("spatial calibration requires source SHA256")
    if not isinstance(data["slots"], dict) or set(data["slots"]) != {"left", "right"}:
        raise ValueError("spatial calibration requires both slots")
    for template in data["slots"].values():
        if not isinstance(template, dict) or set(template) != {
            "center_xy",
            "slope",
            "area_fractions",
        }:
            raise ValueError("invalid scar template fields")
        center, areas = template["center_xy"], template["area_fractions"]
        if (
            not isinstance(center, list)
            or len(center) != 2
            or not isinstance(areas, list)
            or len(areas) != 3
        ):
            raise ValueError("scar template needs a center and three area fractions")
        numbers = [*center, template["slope"], *areas]
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x)
            for x in numbers
        ):
            raise ValueError("scar template parameters must be finite numbers")
        if any(not 0 <= x <= 1 for x in center) or not -2 <= template["slope"] <= 2:
            raise ValueError("scar template center or slope out of bounds")
        if any(not 0 < x <= 1 for x in areas) or areas != sorted(areas):
            raise ValueError("scar area fractions must be positive, bounded, nested")
    return data


def ranked_spatial_mask(
    shape: tuple[int, ...],
    center: tuple[float, float],
    fraction: float,
    *,
    slope: float | None = None,
) -> NDArray[np.bool_]:
    """Exact rounded pixel area with stable row-major ties and nested support."""
    height, width = shape[:2]
    yy, xx = np.mgrid[:height, :width]
    x = (xx + 0.5) / width - center[0]
    y = (yy + 0.5) / height - center[1]
    distance = (
        np.maximum(np.abs(x), np.abs(y)) if slope is None else np.abs(y - slope * x)
    )
    order = np.argsort(distance.ravel(), kind="stable")
    mask = np.zeros(height * width, dtype=bool)
    mask[order[: int(round(fraction * height * width))]] = True
    return np.asarray(mask.reshape(height, width), dtype=np.bool_)


def calibrate_contact_scars(
    response_maps: Mapping[str, NDArray[Any]], source_sha256: str
) -> dict[str, Any]:
    """Fit horizontal scars to aggregate nonnegative Clean contact-response maps.

    Caller freezes independent calibration data and its digest; test labels and
    online frames are never inputs. Fractions target 30/60/85 percent weighted
    response on these maps, not guaranteed coverage on subsequent test frames.
    """
    slots = {}
    if set(response_maps) != {"left", "right"}:
        raise ValueError("calibration needs left and right response maps")
    for slot, raw in response_maps.items():
        weights = np.asarray(raw, dtype=np.float64)
        if (
            weights.ndim != 2
            or not weights.size
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError(
                "calibration maps must be finite nonnegative 2D with signal"
            )
        height, width = weights.shape
        yy, xx = np.mgrid[:height, :width]
        center = [
            float(np.sum(weights * (xx + 0.5) / width) / weights.sum()),
            float(np.sum(weights * (yy + 0.5) / height) / weights.sum()),
        ]
        order = np.argsort(
            np.abs((yy + 0.5) / height - center[1]).ravel(), kind="stable"
        )
        cumulative = np.cumsum(weights.ravel()[order]) / weights.sum()
        areas = [
            float((np.searchsorted(cumulative, dose) + 1) / weights.size)
            for dose in (0.30, 0.60, 0.85)
        ]
        slots[slot] = {"center_xy": center, "slope": 0.0, "area_fractions": areas}
    return validate_spatial_calibration(
        {
            "version": "frozen_contact_scar_v1",
            "source_sha256": source_sha256,
            "slots": slots,
        }
    )
