"""Regression tests for enumerating what data exists for a subject.

Run: uv run pytest dataremoval/tests -q
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

sys.modules.pop("inventory", None)

import inventory  # noqa: E402

METADATA_COLUMNS = ["filename", "sub_id", "session", "group", "subgroup"]
BEH_COLUMNS = ["SubID", "Red", "Green", "RunNumber", "session", "PartType", "Date", "FolderOrg"]


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "ssveps" / "files").mkdir(parents=True)
    (tmp_path / "manualTest").mkdir()
    (tmp_path / "standardizedScores").mkdir()
    return tmp_path


def _write_ssvep(data_dir, rows):
    pd.DataFrame(rows, columns=METADATA_COLUMNS).to_csv(data_dir / "ssveps" / "files" / "metadata.csv", index=False)


def _write_beh(data_dir, rows):
    pd.DataFrame(rows, columns=BEH_COLUMNS).to_csv(data_dir / "manualTest" / "behavioral_table.csv", index=False)


def _fm100_line(ref: str) -> str:
    fields = ["", "", "", ref] + ["0"] * 10  # REFERENCE_COL == 3; exact field count doesn't matter here
    return ",".join(fields)


def _write_fm100(data_dir, refs):
    lines = [_fm100_line(ref) for ref in refs]
    glitch = lines[0] if lines else _fm100_line("MET000")
    (data_dir / "standardizedScores" / "repeatedSessionsPY.txt").write_text("\n".join([glitch] + lines) + "\n")


def test_rows_for_finds_ssvep_sessions(data_dir):
    _write_ssvep(data_dir, [{"filename": "x.mat", "sub_id": "MET999", "session": 1, "group": "CTR", "subgroup": "NA"}])
    _write_beh(data_dir, [])
    _write_fm100(data_dir, [])

    rows = inventory.rows_for(data_dir, "MET999")

    assert {"domain": "ssvep", "session": 1} in rows


def test_rows_for_finds_beh_sessions(data_dir):
    _write_ssvep(data_dir, [])
    _write_beh(
        data_dir,
        [
            {"SubID": "MET999", "Red": 1, "Green": 2, "RunNumber": 1, "session": 1, "PartType": 1, "Date": "x", "FolderOrg": "x"},
            {"SubID": "MET999", "Red": 3, "Green": 4, "RunNumber": 2, "session": 1, "PartType": 1, "Date": "x", "FolderOrg": "x"},
        ],
    )
    _write_fm100(data_dir, [])

    rows = inventory.rows_for(data_dir, "MET999")

    assert rows == [{"domain": "beh", "session": 1}]  # one row per distinct session, not per click


def test_rows_for_finds_fm100_sessions_including_suffixed(data_dir):
    _write_ssvep(data_dir, [])
    _write_beh(data_dir, [])
    _write_fm100(data_dir, ["MET999", "MET999b"])

    rows = inventory.rows_for(data_dir, "MET999")

    assert {"domain": "fm100", "session": 1} in rows
    assert {"domain": "fm100", "session": 2} in rows


def test_rows_for_ignores_other_subjects(data_dir):
    _write_ssvep(data_dir, [{"filename": "x.mat", "sub_id": "MET001", "session": 1, "group": "CTR", "subgroup": "NA"}])
    _write_beh(data_dir, [])
    _write_fm100(data_dir, [])

    assert inventory.rows_for(data_dir, "MET999") == []


def test_rows_for_handles_missing_store_files(data_dir):
    # no _write_* calls at all -- none of the three files exist yet
    assert inventory.rows_for(data_dir, "MET999") == []


def test_list_subjects_is_the_union_across_domains(data_dir):
    _write_ssvep(data_dir, [{"filename": "x.mat", "sub_id": "MET001", "session": 1, "group": "CTR", "subgroup": "NA"}])
    _write_beh(data_dir, [{"SubID": "MET002", "Red": 1, "Green": 2, "RunNumber": 1, "session": 1, "PartType": 1, "Date": "x", "FolderOrg": "x"}])
    _write_fm100(data_dir, ["MET003"])

    assert inventory.list_subjects(data_dir) == ["MET001", "MET002", "MET003"]
