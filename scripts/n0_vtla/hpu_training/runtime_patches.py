"""Source-bound, semantics-preserving HCU runtime patches for N0-VTLA."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

POLICY_MODULE = "n0vtla.models_pytorch.n0vtla_policy"
POLICY_SHA256 = "3d533441976572132eaccfb5e7d420913d9342c3df9b0a61a3f4fd8224c25ca0"
_DETACH_NEEDLE = (
    "        z, g, _has_tac = self._compute_z(vl_ctx.detach(), prefix_pad_masks)\n"
)
_DETACH_REPLACEMENT = (
    "        vl_ctx = vl_ctx.detach()\n"
    "        z, g, _has_tac = self._compute_z(vl_ctx, prefix_pad_masks)\n"
)


def patch_policy_source(source_path: Path) -> str:
    """Release an already-detached, otherwise-unused prefix autograd graph."""
    source_bytes = source_path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != POLICY_SHA256:
        raise RuntimeError(f"N0-VTLA policy source SHA256 mismatch: {digest}")
    source = source_bytes.decode()
    if source.count(_DETACH_NEEDLE) != 1:
        raise RuntimeError("N0-VTLA detach patch anchor is not unique")
    return source.replace(_DETACH_NEEDLE, _DETACH_REPLACEMENT)


def install_detached_vl_ctx_policy() -> Path:
    """Install the exact pinned policy module with the HCU lifetime patch in memory."""
    spec = importlib.util.find_spec(POLICY_MODULE)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate {POLICY_MODULE}")
    source_path = Path(spec.origin).resolve()
    patched_source = patch_policy_source(source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[POLICY_MODULE] = module
    try:
        exec(compile(patched_source, str(source_path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(POLICY_MODULE, None)
        raise
    return source_path
