"""Source-bound gradient-accumulation fallback for pinned N0-VTLA.

This module intentionally patches only the exact ``scripts/train_pytorch.py``
revision certified below. Each rank sees one physical sample at a time while
gradient accumulation is derived from the supported world size, preserving the
configured global batch of 64 for both 8-rank and 32-rank HCU runs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import textwrap
from pathlib import Path

TRAIN_MODULE = "scripts.train_pytorch"
TRAIN_PYTORCH_SHA256 = (
    "468209e74cc12cb183f96d00ba42861dc04fce6712b6ae366432c1436cd7b36b"
)

_IMPORT_ANCHOR = "import dataclasses\n"
_BATCH_SETUP_ANCHOR = """\
    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    append_trace_file("before_batch_size_log")
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )
    append_trace_file("after_batch_size_log")

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    trace_debug("before_build_datasets")
    loader, data_config = build_datasets(config)
    trace_debug("loader_ready")
"""
_BATCH_SETUP_REPLACEMENT = """\
    # Source-bound HCU fallback: one physical sample per rank.  Derive gradient
    # accumulation so every supported topology keeps the global batch at 64.
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    micro_batch_size = 1
    if config.batch_size != 64:
        raise ValueError(
            "gradient-accumulation fallback requires global batch size 64, "
            f"got {config.batch_size}"
        )
    if world_size not in (8, 32):
        raise ValueError(
            "gradient-accumulation fallback requires world_size 8 or 32, "
            f"got {world_size}"
        )
    physical_global_batch_size = world_size * micro_batch_size
    if config.batch_size % physical_global_batch_size != 0:
        raise ValueError(
            "global batch size must be divisible by the physical global batch, "
            f"got {config.batch_size} and {physical_global_batch_size}"
        )
    gradient_accumulation_steps = (
        config.batch_size // physical_global_batch_size
    )
    effective_batch_size = world_size * micro_batch_size * gradient_accumulation_steps
    if effective_batch_size != config.batch_size:
        raise ValueError(
            "gradient-accumulation fallback requires "
            "world_size * micro_batch_size * gradient_accumulation_steps "
            f"== config.batch_size, got {effective_batch_size} != {config.batch_size}"
        )
    append_trace_file("before_batch_size_log")
    logging.info(
        "Using micro batch size per rank: %d, gradient accumulation: %d "
        "(effective global batch: %d)",
        micro_batch_size,
        gradient_accumulation_steps,
        effective_batch_size,
    )
    append_trace_file("after_batch_size_log")

    # The upstream loader interprets batch_size as the global physical batch
    # and divides it by world_size.  Preserve the immutable official config,
    # but give the loader a derived global physical batch of one per rank.
    loader_config = dataclasses.replace(
        config, batch_size=world_size * micro_batch_size
    )
    trace_debug("before_build_datasets")
    loader, data_config = build_datasets(loader_config)
    trace_debug("loader_ready")
"""
_SAMPLE_LOADER_ANCHOR = '        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)\n'
_SAMPLE_LOADER_REPLACEMENT = (
    "        sample_data_loader = _data.create_data_loader(\n"
    '            loader_config, framework="pytorch", shuffle=False\n'
    "        )\n"
)
_STATIC_GRAPH_ANCHOR = (
    "            static_graph=world_size >= 8,  # Enable for 8+ GPUs\n"
)
_STATIC_GRAPH_REPLACEMENT = (
    "            # Vendor PyTorch 2.5.1 asserts when DDP no_sync is combined with static_graph.\n"
    "            static_graph=False,\n"
)
_ACCUMULATOR_ANCHOR = """\
    sample_loss_clip = float(os.environ.get("VTLA_SAMPLE_LOSS_CLIP", "0"))
    if is_main and sample_loss_clip > 0:
        logging.info(f"sample-loss-clip ENABLED (threshold={sample_loss_clip})")
"""
_ACCUMULATOR_REPLACEMENT = (
    _ACCUMULATOR_ANCHOR
    + """\
    micro_step = 0
    accumulated_loss = torch.zeros((), device=device, dtype=torch.float32)
    optim.zero_grad(set_to_none=True)
"""
)
_LR_PHASE_START = """\
            # Update LR (per-group scale for the predictor group, 1.0 otherwise). Phase-A curriculum:
"""
_LR_PHASE_END = """\
                    logging.info(
                        f"phase-A curriculum: boundary at step={global_step} "
                        f"(feat unfrozen, vl mask {'ramping' if global_step == phase_a_steps else 'at config value'})"
                    )
"""
_FORWARD_START = "            # Forward pass\n"
_FORWARD_END = '            trace_step0("step0 backward_done")\n'
_BACKWARD_ANCHOR = "            loss.backward()\n"
_POST_BACKWARD_START = """\

            # Log memory usage after backward pass
"""
_POST_BACKWARD_END = """\
                trace_step0("step0 after_pbar")
"""
_POST_BACKWARD_REPLACEMENT = """\

            accumulated_loss.add_(loss.detach().to(dtype=torch.float32))
            micro_step += 1
            if not is_accumulation_boundary:
                continue

            # Every rank contributes its complete local accumulation window so
            # the reported loss is the global 64-sample mean, not rank 0's
            # two- or eight-sample local mean.
            global_loss_sum = accumulated_loss.clone()
            if use_ddp:
                torch.distributed.all_reduce(
                    global_loss_sum, op=torch.distributed.ReduceOp.SUM
                )
            window_loss = float(global_loss_sum.item()) / (
                world_size * gradient_accumulation_steps
            )
            accumulated_loss.zero_()

            if global_step < 5 and is_main and torch.cuda.is_available() and not use_ddp:
                log_memory_usage(device, global_step, "after_backward")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.optimizer.clip_gradient_norm
            )
            trace_step0("step0 clip_done")

            # Preserve the upstream non-finite guard at optimizer-step scope.
            if not torch.isfinite(grad_norm):
                nonfinite_grad_skips += 1
                if is_main:
                    logging.warning(
                        f"non-finite grad_norm={grad_norm} at step {global_step}; skipping optimizer step "
                        f"(total skips={nonfinite_grad_skips})"
                    )
                optim.zero_grad(set_to_none=True)
                trace_step0("step0 optimizer_skipped_nonfinite")
            else:
                optim.step()
                optim.zero_grad(set_to_none=True)
            trace_step0("step0 optimizer_done")

            if is_main:
                trace_step0("step0 before_collect_stats")
                infos.append(
                    {
                        "loss": window_loss,
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )
                trace_step0("step0 after_collect_stats")
                print(
                    f"STEP {global_step} loss={window_loss} lr={optim.param_groups[0]['lr']}",
                    flush=True,
                )

            if is_main and (global_step % config.log_interval == 0):
                trace_step0("step0 before_log_interval_block")
                elapsed = time.time() - start_time

                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []
                trace_step0("step0 after_log_interval_block")

            global_step += 1
            trace_step0("step0 before_save_checkpoint")
            save_checkpoint(model, optim, global_step, config, is_main, data_config)
            trace_step0("step0 after_save_checkpoint")

            if pbar is not None:
                trace_step0("step0 before_pbar")
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{window_loss:.4f}",
                        "lr": f"{optim.param_groups[0]['lr']:.2e}",
                        "step": global_step,
                    }
                )
                trace_step0("step0 after_pbar")
"""


def _replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"N0-VTLA gradient-accumulation patch anchor {label!r} "
            f"must occur once, got {count}"
        )
    return source.replace(anchor, replacement)


def _replace_bounded_block(
    source: str,
    *,
    start: str,
    end: str,
    replacement: str,
    label: str,
) -> str:
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError(
            f"N0-VTLA gradient-accumulation block {label!r} boundaries are not unique"
        )
    start_index = source.index(start)
    end_index = source.index(end, start_index) + len(end)
    return source[:start_index] + replacement + source[end_index:]


def patch_train_source(source_path: Path) -> str:
    """Return the compiled-source fallback after fail-closed provenance checks."""

    source_bytes = source_path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != TRAIN_PYTORCH_SHA256:
        raise RuntimeError(f"N0-VTLA train source SHA256 mismatch: {digest}")
    source = source_bytes.decode("utf-8")

    source = _replace_once(
        source, _IMPORT_ANCHOR, "import contextlib\nimport dataclasses\n", "import"
    )
    source = _replace_once(
        source,
        _BATCH_SETUP_ANCHOR,
        _BATCH_SETUP_REPLACEMENT,
        "batch-setup",
    )
    source = _replace_once(
        source,
        _SAMPLE_LOADER_ANCHOR,
        _SAMPLE_LOADER_REPLACEMENT,
        "sample-loader",
    )
    source = _replace_once(
        source,
        _STATIC_GRAPH_ANCHOR,
        _STATIC_GRAPH_REPLACEMENT,
        "ddp-static-graph",
    )
    source = _replace_once(
        source,
        _ACCUMULATOR_ANCHOR,
        _ACCUMULATOR_REPLACEMENT,
        "accumulator-state",
    )

    if source.count(_LR_PHASE_START) != 1 or source.count(_LR_PHASE_END) != 1:
        raise RuntimeError(
            "N0-VTLA gradient-accumulation block 'lr-phase' boundaries are not unique"
        )
    lr_start = source.index(_LR_PHASE_START)
    lr_end = source.index(_LR_PHASE_END, lr_start) + len(_LR_PHASE_END)
    lr_phase_block = source[lr_start:lr_end]
    lr_phase_replacement = (
        "            accumulation_index = micro_step % gradient_accumulation_steps\n"
        "            is_accumulation_boundary = (\n"
        "                accumulation_index == gradient_accumulation_steps - 1\n"
        "            )\n"
        "            if accumulation_index == 0:\n"
        + textwrap.indent(lr_phase_block, "    ")
    )
    source = source[:lr_start] + lr_phase_replacement + source[lr_end:]

    if source.count(_FORWARD_START) != 1 or source.count(_FORWARD_END) != 1:
        raise RuntimeError(
            "N0-VTLA gradient-accumulation block 'forward-backward' boundaries are not unique"
        )
    forward_start = source.index(_FORWARD_START)
    forward_end = source.index(_FORWARD_END, forward_start) + len(_FORWARD_END)
    forward_block = source[forward_start:forward_end]
    forward_block = _replace_once(
        forward_block,
        _BACKWARD_ANCHOR,
        "            (loss / gradient_accumulation_steps).backward()\n",
        "backward",
    )
    forward_replacement = (
        "            sync_context = (\n"
        "                model.no_sync()\n"
        "                if use_ddp and not is_accumulation_boundary\n"
        "                else contextlib.nullcontext()\n"
        "            )\n"
        "            with sync_context:\n" + textwrap.indent(forward_block, "    ")
    )
    source = source[:forward_start] + forward_replacement + source[forward_end:]

    source = _replace_bounded_block(
        source,
        start=_POST_BACKWARD_START,
        end=_POST_BACKWARD_END,
        replacement=_POST_BACKWARD_REPLACEMENT,
        label="optimizer-step",
    )
    compile(source, str(source_path), "exec")
    return source


def install_gradient_accumulation_train() -> Path:
    """Install the exact pinned training module with the fallback in memory."""

    spec = importlib.util.find_spec(TRAIN_MODULE)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate {TRAIN_MODULE}")
    source_path = Path(spec.origin).resolve()
    patched_source = patch_train_source(source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[TRAIN_MODULE] = module
    try:
        exec(compile(patched_source, str(source_path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(TRAIN_MODULE, None)
        raise
    return source_path
