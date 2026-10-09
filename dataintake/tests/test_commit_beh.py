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


def test_session_override_relabels_every_row(data_dir):
    pd.DataFrame(columns=BEH_COLUMNS).to_csv(_master_path(data_dir), index=False)

    commit.commit_beh(BEH_FIXTURE, data_dir, session=2)

    master = pd.read_csv(_master_path(data_dir))
    assert set(master["session"]) == {2}


def test_session_override_duplicate_check_uses_overridden_session(data_dir):
    existing = pd.DataFrame([{"SubID": "MET999", "Red": 1, "Green": 2, "RunNumber": 1, "session": 2, "PartType": 1, "Date": "01_JAN", "FolderOrg": "x"}])
    existing.to_csv(_master_path(data_dir), index=False)

    with pytest.raises(DuplicateKeyError):
        commit.commit_beh(BEH_FIXTURE, data_dir, session=2)

    master = pd.read_csv(_master_path(data_dir))
    assert len(master) == 1  # untouched, session-1 write never attempted


def test_dry_run_writes_nothing_on_a_clear_key(data_dir):
    pd.DataFrame(columns=BEH_COLUMNS).to_csv(_master_path(data_dir), index=False)

    commit.commit_beh(BEH_FIXTURE, data_dir, dry_run=True)

    master = pd.read_csv(_master_path(data_dir))
    assert len(master) == 0


def test_dry_run_still_raises_on_duplicate(data_dir):
    existing = pd.DataFrame([{"SubID": "MET999", "Red": 1, "Green": 2, "RunNumber": 1, "session": 1, "PartType": 1, "Date": "01_JAN", "FolderOrg": "x"}])
    existing.to_csv(_master_path(data_dir), index=False)

    with pytest.raises(DuplicateKeyError):
        commit.commit_beh(BEH_FIXTURE, data_dir, dry_run=True)

    master = pd.read_csv(_master_path(data_dir))
    assert len(master) == 1  # untouched


JSON_FIXTURE = FIXTURES / "MET999_beh.json"


def test_json_commit_converts_to_master_rows(data_dir):
    pd.DataFrame(columns=BEH_COLUMNS).to_csv(_master_path(data_dir), index=False)

    commit.commit_beh(JSON_FIXTURE, data_dir, group="UNKNOWN", session=1)

    master = pd.read_csv(_master_path(data_dir))
    assert list(master.columns) == BEH_COLUMNS
    assert len(master) == 3
    assert set(master["SubID"]) == {"MET999"}
    assert list(master["RunNumber"]) == [1, 2, 3]
    assert list(master["Red"]) == [2033, 1624, 1637]
    assert list(master["Green"]) == [614, 691, 789]
    assert set(master["session"]) == {1}
    assert set(master["PartType"]) == {0}
    assert set(master["Date"]) == {"08_OCT_26"}
    assert set(master["FolderOrg"]) == {"TestUPstairs"}


def test_json_commit_honors_sub_id_group_and_session_overrides(data_dir):
    pd.DataFrame(columns=BEH_COLUMNS).to_csv(_master_path(data_dir), index=False)

    commit.commit_beh(JSON_FIXTURE, data_dir, sub_id="MET050", group="CVD", session=2)

    master = pd.read_csv(_master_path(data_dir))
    assert set(master["SubID"]) == {"MET050"}
    assert set(master["PartType"]) == {2}
    assert set(master["session"]) == {2}


def test_json_commit_raises_on_duplicate_and_overwrite_replaces(data_dir):
    pd.DataFrame(columns=BEH_COLUMNS).to_csv(_master_path(data_dir), index=False)
    commit.commit_beh(JSON_FIXTURE, data_dir, group="UNKNOWN", session=1)

    with pytest.raises(DuplicateKeyError):
        commit.commit_beh(JSON_FIXTURE, data_dir, group="UNKNOWN", session=1)

    commit.commit_beh(JSON_FIXTURE, data_dir, group="CTR", session=1, overwrite=True)
    master = pd.read_csv(_master_path(data_dir))
    assert len(master) == 3
    assert set(master["PartType"]) == {1}


def test_csv_commit_accepts_unknown_group(data_dir):
    pd.DataFrame(columns=BEH_COLUMNS).to_csv(_master_path(data_dir), index=False)

    commit.commit_beh(BEH_FIXTURE, data_dir, group="UNKNOWN")

    assert set(pd.read_csv(_master_path(data_dir))["PartType"]) == {0}


def test_minimal_csv_commit_fills_defaults(data_dir, tmp_path):
    pd.DataFrame(columns=BEH_COLUMNS).to_csv(_master_path(data_dir), index=False)
    path = tmp_path / "METXXX.csv"
    path.write_text("Red,Green\n1000,500\n1100,520\n1200,540\n")

    commit.commit_beh(path, data_dir)

    master = pd.read_csv(_master_path(data_dir))
    assert list(master.columns) == BEH_COLUMNS
    assert set(master["SubID"]) == {"METXXX"}
    assert list(master["Red"]) == [1000, 1100, 1200]
    assert list(master["RunNumber"]) == [1, 2, 3]
    assert set(master["session"]) == {1}
    assert set(master["PartType"]) == {0}
    assert set(master["FolderOrg"]) == {"METXXX.csv"}
    assert master["Date"].str.fullmatch(r"\d{2}_[A-Z]{3}_\d{2}").all()


def test_minimal_csv_commit_honors_overrides(data_dir, tmp_path):
    pd.DataFrame(columns=BEH_COLUMNS).to_csv(_master_path(data_dir), index=False)
    path = tmp_path / "METXXX.csv"
    path.write_text("SubID,Red,Green\nMET048,1017,1511\nMET048,1604,1128\n")

    commit.commit_beh(path, data_dir, sub_id="MET060", group="PD", session=2)

    master = pd.read_csv(_master_path(data_dir))
    assert set(master["SubID"]) == {"MET060"}
    assert set(master["PartType"]) == {3}
    assert set(master["session"]) == {2}


def test_minimal_csv_keeps_run_number_and_session_when_present(data_dir, tmp_path):
    pd.DataFrame(columns=BEH_COLUMNS).to_csv(_master_path(data_dir), index=False)
    path = tmp_path / "x.csv"
    path.write_text("SubID,Red,Green,RunNumber,session\nMET070,1,2,5,3\nMET070,3,4,6,3\n")

    commit.commit_beh(path, data_dir)

    master = pd.read_csv(_master_path(data_dir))
    assert list(master["RunNumber"]) == [5, 6]
    assert set(master["session"]) == {3}
