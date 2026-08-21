"""Wheel/sdist policy keeps third-party runtimes outside the core package."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hatch_inventory_declares_release_metadata_and_external_exclusions() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert (
        '"integrations/integrations.lock.json" = "robotactile_benchmark/integrations/integrations.lock.json"'
        in project
    )
    assert (
        '"THIRD_PARTY_NOTICES.md" = "robotactile_benchmark/THIRD_PARTY_NOTICES.md"'
        in project
    )
    assert '"/.gitignore"' in project
    assert '"/examples"' in project
    assert '"/integrations"' in project
    assert '"/requirements"' in project
    assert '"/.github"' in project
    for forbidden in ("N0-TWAM", "UniVTAC", "*.ckpt", "*.pth", '"/outputs"'):
        assert forbidden not in project


def test_external_lock_and_installers_are_separate_artifacts() -> None:
    assert (ROOT / "integrations/integrations.lock.json").is_file()
    assert (ROOT / "integrations/install_act_runtime.sh").is_file()
    assert (ROOT / "integrations/install_n0_twam.sh").is_file()
    assert (ROOT / "integrations/install_univtac.sh").is_file()
    assert not (ROOT / "src/robotactile_benchmark/N0-TWAM").exists()
    assert not (ROOT / "src/robotactile_benchmark/UniVTAC").exists()


def test_gitignore_excludes_data_outputs_weights_and_secrets() -> None:
    ignored = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())

    assert {"data/", "outputs/", "checkpoints/"} <= ignored
    assert {"*.ckpt", "*.pth", "*.pt"} <= ignored
    assert {".env", "*.pem", "*.key", "credentials.json", "settings.json"} <= ignored
