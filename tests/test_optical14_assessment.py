"""Scientific summaries must not reuse outputs or accept incomplete evidence."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def assessment_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "assess_optical14.py"
    spec = importlib.util.spec_from_file_location("optical14_assessment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assessment_no_clobber_before_reading_sources(tmp_path, assessment_module):
    with pytest.raises(FileExistsError):
        assessment_module.assess(tmp_path / "missing", tmp_path)


@pytest.mark.parametrize("failure", ["incomplete", "hash_mismatch", "escaped_path"])
def test_assessment_rejects_bad_gallery_before_writing(
    tmp_path, assessment_module, monkeypatch, failure
):
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    payload = gallery / "sample.png"
    payload.write_bytes(b"unchanged-file-not-an-image")
    receipt = {
        "operator_count": 13 if failure == "incomplete" else 14,
        "all_delivery_validations_passed": True,
        "production_source_sha256": {},
        "output_sha256": {"sample.png": "0" * 64},
    }
    if failure == "escaped_path":
        outside = tmp_path / "outside.png"
        outside.write_bytes(b"outside")
        receipt["output_sha256"] = {
            "../outside.png": assessment_module.file_hash(outside)
        }
    (gallery / "gallery_receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(assessment_module, "production_hashes", lambda: {})
    output = tmp_path / "new-assessment"
    with pytest.raises(ValueError):
        assessment_module.assess(gallery, output)
    assert not output.exists()
