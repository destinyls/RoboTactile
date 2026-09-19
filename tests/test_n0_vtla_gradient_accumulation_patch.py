import hashlib
from pathlib import Path

import pytest

from scripts.n0_vtla.hpu_training import gradient_accumulation_patch as patch_module
from scripts.n0_vtla.hpu_training.gradient_accumulation_patch import (
    TRAIN_PYTORCH_SHA256,
    patch_train_source,
)


def _pinned_source() -> Path:
    return Path("../N0-VTLA/scripts/train_pytorch.py")


def test_pinned_train_source_sha_and_patched_source_compile() -> None:
    source_path = _pinned_source()

    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == TRAIN_PYTORCH_SHA256
    patched = patch_train_source(source_path)

    assert "micro_batch_size = 1" in patched
    assert "if world_size not in (8, 32)" in patched
    assert "config.batch_size != 64" in patched
    assert (
        "gradient_accumulation_steps = (\n        config.batch_size // physical_global_batch_size\n    )"
        in patched
    )
    assert "gradient_accumulation_steps = 8" not in patched
    assert (
        "effective_batch_size = world_size * micro_batch_size * gradient_accumulation_steps"
        in patched
    )
    assert "model.no_sync()" in patched
    assert "if use_ddp and not is_accumulation_boundary" in patched
    assert "static_graph=False" in patched
    assert "static_graph=world_size >= 8" not in patched
    assert "(loss / gradient_accumulation_steps).backward()" in patched
    assert "global_loss_sum = accumulated_loss.clone()" in patched
    assert "torch.distributed.all_reduce(" in patched
    assert "global_loss_sum, op=torch.distributed.ReduceOp.SUM" in patched
    assert "world_size * gradient_accumulation_steps" in patched
    assert patched.count("optim.step()") == 1
    assert patched.count("global_step += 1") == 1
    assert (
        patched.count(
            "save_checkpoint(model, optim, global_step, config, is_main, data_config)"
        )
        == 1
    )
    compile(patched, str(source_path), "exec")


def test_patch_rejects_non_pinned_source(tmp_path: Path) -> None:
    source_path = tmp_path / "train_pytorch.py"
    source_path.write_text("pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        patch_train_source(source_path)


def test_patch_rejects_non_unique_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _pinned_source().read_text(encoding="utf-8")
    source = source.replace(
        patch_module._IMPORT_ANCHOR,
        patch_module._IMPORT_ANCHOR * 2,
        1,
    )
    source_path = tmp_path / "train_pytorch.py"
    source_path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(
        patch_module,
        "TRAIN_PYTORCH_SHA256",
        hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(RuntimeError, match="anchor 'import'.*got 2"):
        patch_train_source(source_path)
