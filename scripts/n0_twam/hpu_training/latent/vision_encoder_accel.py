#!/usr/bin/env python3
"""Run the pinned official Vision encoder with UMT5 on its visible HCU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

OFFICIAL_VISION_ENCODER_SHA256 = (
    "c731103d95059b5ad769a0f28d2b68ba664c02f3b95c4557dbc632d27a6d14ff"
)
OFFICIAL_MODULE_NAME = "_robotactile_pinned_official_vision_encoder"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _option_value(arguments: Sequence[str], option: str) -> str:
    indices = [index for index, value in enumerate(arguments) if value == option]
    if len(indices) != 1 or indices[0] + 1 >= len(arguments):
        raise ValueError(f"official arguments must contain exactly one {option}")
    return arguments[indices[0] + 1]


def _load_official_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(OFFICIAL_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned official Vision encoder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[OFFICIAL_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def run_official_encoder(
    *,
    official_script: Path,
    official_arguments: Sequence[str],
    expected_sha256: str = OFFICIAL_VISION_ENCODER_SHA256,
) -> None:
    """Invoke official main while overriding only UMT5's placement argument."""
    resolved_script = official_script.resolve(strict=True)
    actual_sha256 = _sha256_file(resolved_script)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "official Vision encoder SHA256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    forwarded = list(official_arguments)
    encoder_device = _option_value(forwarded, "--device")
    if encoder_device != "cuda:0":
        raise ValueError("accelerated Vision encoding requires --device cuda:0")

    module = _load_official_module(resolved_script)
    original_loader = getattr(module, "load_text_encoder", None)
    official_main = getattr(module, "main", None)
    if not callable(original_loader) or not callable(official_main):
        raise TypeError("pinned official Vision encoder API is incomplete")

    def load_text_encoder_on_visible_hcu(
        text_encoder_path: object,
        torch_dtype: object,
        torch_device: object,
    ) -> object:
        if str(torch_device) != "cpu":
            raise RuntimeError(
                "pinned official main no longer requests CPU UMT5 placement"
            )
        return original_loader(
            text_encoder_path,
            torch_dtype=torch_dtype,
            torch_device=encoder_device,
        )

    module.__dict__["load_text_encoder"] = load_text_encoder_on_visible_hcu
    previous_argv = sys.argv
    sys.argv = [str(resolved_script), *forwarded]
    try:
        official_main()
    finally:
        sys.argv = previous_argv


def _parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-script", type=Path, required=True)
    parsed, official_arguments = parser.parse_known_args(argv)
    if not official_arguments:
        parser.error("official encoder arguments are required")
    return parsed, official_arguments


def main(argv: Sequence[str] | None = None) -> int:
    args, official_arguments = _parse_args(argv)
    run_official_encoder(
        official_script=args.official_script,
        official_arguments=official_arguments,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
