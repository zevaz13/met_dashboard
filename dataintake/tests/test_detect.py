"""Regression tests for content-based file-type detection.

Run: uv run pytest dataintake/tests -q
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

sys.modules.pop("detect", None)

import detect  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_detects_ssvep_from_mat_contents_not_filename():
    domain, sub_id, meta = detect.detect_file(FIXTURES / "MET999.mat")
    assert domain == "ssvep"
    assert sub_id == "MET000"
    assert meta["filename_mismatch"] == "MET999"
    assert meta["group"] == "CTR"
    assert meta["subgroup"] == "NA"
    assert meta["session"] == 2


def test_detects_beh_from_header_and_reads_group_from_part_type():
    domain, sub_id, meta = detect.detect_file(FIXTURES / "MET999_beh.csv")
    assert domain == "beh"
    assert sub_id == "MET999"
    assert "filename_mismatch" not in meta
    assert meta["group"] == "CTR"


def test_detects_fm100_from_field_count_and_strips_session_suffix():
    domain, sub_id, meta = detect.detect_file(FIXTURES / "MET999_FM100.txt")
    assert domain == "fm100"
    assert sub_id == "MET999"
    assert "filename_mismatch" not in meta


def test_unrecognized_file_is_flagged_not_guessed_at(tmp_path):
    garbage = tmp_path / "notes.txt"
    garbage.write_text("just some notes, not a data file")
    domain, sub_id, meta = detect.detect_file(garbage)
    assert domain is None
    assert sub_id is None
    assert "reason" in meta
