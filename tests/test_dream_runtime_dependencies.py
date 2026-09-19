"""Dream-Tac's HTTP server dependencies must be installed in its own overlay."""

from pathlib import Path


def test_dream_runtime_includes_cors_without_mutating_shared_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    installer = root / "scripts/retrained_evaluation/install_dream_runtime.sh"
    text = installer.read_text()
    assert "flask-cors==6.0.1" in text
    assert "pytz==2025.2" in text
    assert "peft==0.17.1" in text
    assert "peft==0.15.2" not in text
    assert "flash-attn==2.7.3" in text
    assert 'TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"' in text
    assert 'FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80;90;100;120}"' in text
    assert 'MAX_JOBS="${MAX_JOBS:-2}"' in text
    assert 'export PATH="$dream_site/bin:$PATH"' in text
    assert '--target "$dream_site"' in text
    assert '--target "$shared_site"' not in text
    assert "--no-deps" in text
    assert "Refusing to overwrite Dream-Tac runtime" in text
