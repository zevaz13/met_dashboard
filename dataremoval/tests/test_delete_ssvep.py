"""Regression tests for removing SSVEP data, with backup + restore.

Run: uv run pytest dataremoval/tests -q
"""

import json
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

METADATA_COLUMNS = ["filename", "sub_id", "session", "group", "subgroup"]
RUNMAP_COLUMNS = ["sub_id", "session", "run", "red_idx", "green_idx", "value"]
BASELINES_COLUMNS = ["sub_id", "session", "run", "trial", "value"]


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "ssveps" / "files").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def backup_dir(tmp_path):
    d = tmp_path / "deletedData"
    d.mkdir()
    return d


def _files_dir(data_dir):
    return data_dir / "ssveps" / "files"


def _seed(data_dir):
    metadata = [
        {"filename": "a.mat", "sub_id": "MET999", "session": 1, "group": "CTR", "subgroup": "NA"},
        {"filename": "b.mat", "sub_id": "MET001", "session": 1, "group": "PD", "subgroup": "NA"},
    ]
    runmap = [
        {"sub_id": "MET999", "session": 1, "run": 1, "red_idx": 0, "green_idx": 0, "value": 1.5},
        {"sub_id": "MET001", "session": 1, "run": 1, "red_idx": 0, "green_idx": 0, "value": 2.5},
    ]
    baselines = [
        {"sub_id": "MET999", "session": 1, "run": 1, "trial": 1, "value": 0.5},
        {"sub_id": "MET001", "session": 1, "run": 1, "trial": 1, "value": 0.6},
    ]
    pd.DataFrame(metadata, columns=METADATA_COLUMNS).to_csv(_files_dir(data_dir) / "metadata.csv", index=False)
    pd.DataFrame(runmap, columns=RUNMAP_COLUMNS).to_csv(_files_dir(data_dir) / "runmap.csv", index=False)
    pd.DataFrame(baselines, columns=BASELINES_COLUMNS).to_csv(_files_dir(data_dir) / "baselines.csv", index=False)


def test_delete_removes_only_the_matching_subject_session(data_dir, backup_dir):
    _seed(data_dir)

    delete.delete_ssvep(data_dir, backup_dir, "MET999", 1)

    metadata = pd.read_csv(_files_dir(data_dir) / "metadata.csv")
    runmap = pd.read_csv(_files_dir(data_dir) / "runmap.csv")
    baselines = pd.read_csv(_files_dir(data_dir) / "baselines.csv")
    assert set(metadata["sub_id"]) == {"MET001"}
    assert set(runmap["sub_id"]) == {"MET001"}
    assert set(baselines["sub_id"]) == {"MET001"}


def test_delete_writes_a_backup_with_the_removed_rows(data_dir, backup_dir):
    _seed(data_dir)

    backup_path = delete.delete_ssvep(data_dir, backup_dir, "MET999", 1)

    assert backup_path.exists()
    backup = json.loads(backup_path.read_text())
    assert backup["metadata_row"]["sub_id"] == "MET999"
    assert len(backup["runmap_rows"]) == 1
    assert len(backup["baseline_rows"]) == 1


def test_delete_raises_when_row_does_not_exist(data_dir, backup_dir):
    _seed(data_dir)

    with pytest.raises(RowNotFoundError):
        delete.delete_ssvep(data_dir, backup_dir, "MET999", 99)

    metadata = pd.read_csv(_files_dir(data_dir) / "metadata.csv")
    assert len(metadata) == 2  # untouched


def test_dry_run_writes_no_backup_and_removes_nothing(data_dir, backup_dir):
    _seed(data_dir)

    result = delete.delete_ssvep(data_dir, backup_dir, "MET999", 1, dry_run=True)

    assert result is None
    assert list(backup_dir.iterdir()) == []
    metadata = pd.read_csv(_files_dir(data_dir) / "metadata.csv")
    assert len(metadata) == 2  # untouched


def test_restore_puts_the_exact_rows_back(data_dir, backup_dir):
    _seed(data_dir)
    backup_path = delete.delete_ssvep(data_dir, backup_dir, "MET999", 1)

    delete.restore_ssvep(data_dir, backup_path)

    metadata = pd.read_csv(_files_dir(data_dir) / "metadata.csv")
    runmap = pd.read_csv(_files_dir(data_dir) / "runmap.csv")
    baselines = pd.read_csv(_files_dir(data_dir) / "baselines.csv")
    assert set(metadata["sub_id"]) == {"MET999", "MET001"}
    assert set(runmap["sub_id"]) == {"MET999", "MET001"}
    assert set(baselines["sub_id"]) == {"MET999", "MET001"}
