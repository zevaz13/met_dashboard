"""Regression tests for removing behavioral data, with backup + restore.

Run: uv run pytest dataremoval/tests -q
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

sys.modules.pop("delete", None)
sys.modules.pop("errors", None)

import delete  # noqa: E402
from errors import RowNotFoundError  # noqa: E402

BEH_COLUMNS = ["SubID", "Red", "Green", "RunNumber", "session", "PartType", "Date", "FolderOrg"]


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "manualTest").mkdir()
    return tmp_path


@pytest.fixture
def backup_dir(tmp_path):
    d = tmp_path / "deletedData"
    d.mkdir()
    return d


def _master_path(data_dir):
    return data_dir / "manualTest" / "behavioral_table.csv"


def _seed(data_dir):
    rows = [
        {"SubID": "MET999", "Red": 1, "Green": 2, "RunNumber": 1, "session": 1, "PartType": 1, "Date": "01_JAN", "FolderOrg": "x"},
        {"SubID": "MET999", "Red": 3, "Green": 4, "RunNumber": 2, "session": 1, "PartType": 1, "Date": "01_JAN", "FolderOrg": "x"},
        {"SubID": "MET999", "Red": 5, "Green": 6, "RunNumber": 1, "session": 2, "PartType": 1, "Date": "01_FEB", "FolderOrg": "x"},
        {"SubID": "MET001", "Red": 9, "Green": 9, "RunNumber": 1, "session": 1, "PartType": 2, "Date": "01_JAN", "FolderOrg": "x"},
    ]
    pd.DataFrame(rows).to_csv(_master_path(data_dir), index=False)


def test_delete_removes_only_the_matching_subject_session(data_dir, backup_dir):
    _seed(data_dir)

    delete.delete_beh(data_dir, backup_dir, "MET999", 1)

    master = pd.read_csv(_master_path(data_dir))
    assert len(master) == 2
    assert not ((master["SubID"] == "MET999") & (master["session"] == 1)).any()
    assert ((master["SubID"] == "MET999") & (master["session"] == 2)).any()
    assert (master["SubID"] == "MET001").any()


def test_delete_writes_a_backup_with_the_removed_rows(data_dir, backup_dir):
    _seed(data_dir)

    backup_path = delete.delete_beh(data_dir, backup_dir, "MET999", 1)

    assert backup_path.exists()
    backup = pd.read_csv(backup_path)
    assert len(backup) == 2
    assert set(backup["SubID"]) == {"MET999"}
    assert set(backup["session"]) == {1}


def test_delete_raises_when_row_does_not_exist(data_dir, backup_dir):
    _seed(data_dir)

    with pytest.raises(RowNotFoundError):
        delete.delete_beh(data_dir, backup_dir, "MET999", 99)

    master = pd.read_csv(_master_path(data_dir))
    assert len(master) == 4  # untouched


def test_dry_run_writes_no_backup_and_removes_nothing(data_dir, backup_dir):
    _seed(data_dir)

    result = delete.delete_beh(data_dir, backup_dir, "MET999", 1, dry_run=True)

    assert result is None
    assert list(backup_dir.iterdir()) == []
    master = pd.read_csv(_master_path(data_dir))
    assert len(master) == 4  # untouched


def test_dry_run_still_raises_when_row_does_not_exist(data_dir, backup_dir):
    _seed(data_dir)

    with pytest.raises(RowNotFoundError):
        delete.delete_beh(data_dir, backup_dir, "MET999", 99, dry_run=True)


def test_restore_puts_the_exact_rows_back(data_dir, backup_dir):
    _seed(data_dir)
    backup_path = delete.delete_beh(data_dir, backup_dir, "MET999", 1)

    delete.restore_beh(data_dir, backup_path)

    master = pd.read_csv(_master_path(data_dir))
    assert len(master) == 4
    assert ((master["SubID"] == "MET999") & (master["session"] == 1)).sum() == 2
