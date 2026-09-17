"""Regression tests for committing new behavioral data into the master CSV.

Run: uv run pytest dataintake/tests -q
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

sys.modules.pop("commit", None)
sys.modules.pop("errors", None)

import commit  # noqa: E402
from errors import DuplicateKeyError  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BEH_FIXTURE = FIXTURES / "MET999_beh.csv"
BEH_COLUMNS = ["SubID", "Red", "Green", "RunNumber", "session", "PartType", "Date", "FolderOrg"]


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "manualTest").mkdir()
    return tmp_path


def _master_path(data_dir):
    return data_dir / "manualTest" / "behavioral_table.csv"


def test_appends_new_subject_rows_to_empty_master(data_dir):
    pd.DataFrame(columns=BEH_COLUMNS).to_csv(_master_path(data_dir), index=False)

    commit.commit_beh(BEH_FIXTURE, data_dir)

    master = pd.read_csv(_master_path(data_dir))
    assert len(master) == 25
    assert set(master["SubID"]) == {"MET999"}
    assert set(master["PartType"]) == {1}


def test_appends_alongside_existing_other_subject_rows(data_dir):
    existing = pd.DataFrame([{"SubID": "MET001", "Red": 100, "Green": 200, "RunNumber": 1, "session": 1, "PartType": 2, "Date": "01_JAN", "FolderOrg": "x"}])
    existing.to_csv(_master_path(data_dir), index=False)

    commit.commit_beh(BEH_FIXTURE, data_dir)

    master = pd.read_csv(_master_path(data_dir))
    assert len(master) == 26
    assert "MET001" in set(master["SubID"])
    assert "MET999" in set(master["SubID"])


def test_raises_on_duplicate_session_without_overwrite(data_dir):
    existing = pd.DataFrame([{"SubID": "MET999", "Red": 1, "Green": 2, "RunNumber": 1, "session": 1, "PartType": 1, "Date": "01_JAN", "FolderOrg": "x"}])
    existing.to_csv(_master_path(data_dir), index=False)

    with pytest.raises(DuplicateKeyError):
        commit.commit_beh(BEH_FIXTURE, data_dir)

    master = pd.read_csv(_master_path(data_dir))
    assert len(master) == 1  # untouched


def test_overwrite_replaces_existing_subject_session_rows(data_dir):
    existing = pd.DataFrame([{"SubID": "MET999", "Red": 1, "Green": 2, "RunNumber": 1, "session": 1, "PartType": 1, "Date": "01_JAN", "FolderOrg": "x"}])
    existing.to_csv(_master_path(data_dir), index=False)

    commit.commit_beh(BEH_FIXTURE, data_dir, overwrite=True)

    master = pd.read_csv(_master_path(data_dir))
    assert len(master) == 25
    assert 2034 in set(master["Red"])  # fixture's second row, not the stale placeholder


def test_group_override_rewrites_part_type(data_dir):
    pd.DataFrame(columns=BEH_COLUMNS).to_csv(_master_path(data_dir), index=False)

    commit.commit_beh(BEH_FIXTURE, data_dir, group="CVD")

    master = pd.read_csv(_master_path(data_dir))
    assert set(master["PartType"]) == {2}


def test_sub_id_override_renames_subject(data_dir):
    pd.DataFrame(columns=BEH_COLUMNS).to_csv(_master_path(data_dir), index=False)

    commit.commit_beh(BEH_FIXTURE, data_dir, sub_id="MET123")

    master = pd.read_csv(_master_path(data_dir))
    assert set(master["SubID"]) == {"MET123"}
