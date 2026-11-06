"""Regression guards for the scoped offline performance measurement harness."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents.__getitem__(1) / "tools/measure_ocr_optimizations.py"
SPEC = importlib.util.spec_from_file_location("measure_ocr_optimizations_tested", SCRIPT)
measure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure)


def test_existing_output_is_never_removed(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("user data")
    with pytest.raises(FileExistsError):
        measure.create_output(target)
    assert marker.read_text() == "user data"


def test_symlink_output_is_rejected(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        measure.create_output(link)
    assert link.is_symlink()
    assert target.is_dir()


def test_new_output_is_created(tmp_path):
    target = tmp_path / "parent" / "new"
    assert measure.create_output(target) == target
    assert target.is_dir()


def test_distribution_retains_outlier():
    result = measure.distribution([1.0, 2.0, 3.0, 4.0, 100.0])
    assert result == {"n": 5, "median_s": 3.0, "min_s": 1.0, "max_s": 100.0, "mad_s": 1.0}


def test_empty_distribution_rejected():
    with pytest.raises(ValueError):
        measure.distribution([])


def samples(current_hash="same"):
    return [
        {
            "scenario": "components", "variant": variant, "repeat": repeat,
            "result": {
                "verification_sha256": signature,
                "cold": {"total_s": elapsed},
                "warm": {"total_s": elapsed / 2},
            },
        }
        for repeat in range(5)
        for variant, signature, elapsed in (
            ("baseline", "same", 2.0), ("current", current_hash, 1.0),
        )
    ]


def test_summary_ratio_and_repeat_counts():
    rows = measure.summarize(samples())
    assert len(rows) == 2
    for row in rows:
        assert row["baseline"]["n"] == row["current"]["n"] == 5
        assert row["equivalent"] is True
        assert row["baseline_over_current"] == 2.0
        assert row["ranges_overlap"] is False


def test_no_speedup_for_inequivalent_outputs():
    for row in measure.summarize(samples("different")):
        assert row["equivalent"] is False
        assert row["baseline_over_current"] is None


@pytest.mark.parametrize("rows", [
    {}, {"page.png": (None, "failure")}, {"page.png": (None, None)},
    {"page.png": ([], None)},
])
def test_failed_or_empty_real_ocr_is_not_a_measurement(rows):
    with pytest.raises(ValueError):
        measure.real_signature(rows)


def test_real_signature_preserves_text_confidence_geometry():
    blocks = [{"text": "文字", "confidence": 0.95, "box": list((list((2, 3)), list((4, 5))))}]
    got = measure.real_signature({"/private/example/page.png": (blocks, None)})
    assert got == [{"file": "page.png", "blocks": blocks}]


def test_redaction_prefers_longest_path():
    got = measure.redact(
        "/private/home/repo/input.png /private/home/other",
        [("/private/home", "<home>"), ("/private/home/repo", "<repo>")],
    )
    assert got == "<repo>/input.png <home>/other"


def test_digest_is_mapping_order_independent():
    assert measure.digest({"a": 1, "b": 2}) == measure.digest({"b": 2, "a": 1})


def test_offline_controls_are_explicit():
    assert measure.OFFLINE["NOVEL_FORMATTER_MODEL_AUTO_DOWNLOAD"] == "0"
    for key in ("NOVEL_FORMATTER_OCR_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        assert measure.OFFLINE[key] == "1"
    assert measure.OFFLINE["NOVEL_FORMATTER_NDLOCR_PROVIDER"] == "cpu"
