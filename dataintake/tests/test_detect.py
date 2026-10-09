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


def test_detects_beh_from_json_signature_with_unknown_group_and_default_session():
    domain, sub_id, meta = detect.detect_file(FIXTURES / "MET999_beh.json")
    assert domain == "beh"
    assert sub_id == "MET999"
    assert meta["group"] == "UNKNOWN"
    assert meta["sessions"] == [1]


def test_json_without_behavioral_signature_is_unrecognized(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"metadata": {"mode": "ssvep"}, "columns": [], "data": []}')
    domain, sub_id, meta = detect.detect_file(path)
    assert domain is None
    assert "reason" in meta


def test_malformed_json_is_unrecognized(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    domain, _, meta = detect.detect_file(path)
    assert domain is None
    assert "reason" in meta


def test_detects_minimal_beh_csv_with_only_red_and_green(tmp_path):
    path = tmp_path / "METXXX.csv"
    path.write_text("Red,Green\n1000,500\n1100,520\n")
    domain, sub_id, meta = detect.detect_file(path)
    assert domain == "beh"
    assert sub_id == "METXXX"
    assert meta["group"] == "UNKNOWN"
    assert meta["sessions"] == [1]


def test_beh_csv_uses_subid_column_when_present_and_matches_columns_case_insensitively(tmp_path):
    path = tmp_path / "METXXX.csv"
    path.write_text("subid,RED,green\nMET048,1017,1511\nMET048,1604,1128\n")
    domain, sub_id, meta = detect.detect_file(path)
    assert domain == "beh"
    assert sub_id == "MET048"
    assert meta["filename_mismatch"] == "METXXX"


def test_beh_csv_with_multiple_sub_ids_is_unrecognized(tmp_path):
    path = tmp_path / "many.csv"
    path.write_text("SubID,Red,Green\nMET001,1,2\nMET002,3,4\n")
    domain, _, meta = detect.detect_file(path)
    assert domain is None
    assert "SubID" in meta["reason"]


def test_csv_with_red_but_no_green_is_unrecognized_and_says_why(tmp_path):
    path = tmp_path / "half.csv"
    path.write_text("SubID,Red\nMET001,1\n")
    domain, _, meta = detect.detect_file(path)
    assert domain is None
    assert "Green" in meta["reason"]
