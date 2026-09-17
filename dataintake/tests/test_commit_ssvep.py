"""Regression tests for committing new SSVEP .mat data into the derived CSVs.

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
SSVEP_FIXTURE = FIXTURES / "MET999.mat"  # content SubID is actually MET000, session 2

METADATA_COLUMNS = ["filename", "sub_id", "session", "group", "subgroup"]
RUNMAP_COLUMNS = ["sub_id", "session", "run", "red_idx", "green_idx", "value"]
BASELINES_COLUMNS = ["sub_id", "session", "run", "trial", "value"]


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "ssveps" / "files").mkdir(parents=True)
    return tmp_path


def _files_dir(data_dir):
    return data_dir / "ssveps" / "files"


def _write_stores(data_dir, metadata_rows=(), runmap_rows=(), baselines_rows=()):
    pd.DataFrame(list(metadata_rows), columns=METADATA_COLUMNS).to_csv(_files_dir(data_dir) / "metadata.csv", index=False)
    pd.DataFrame(list(runmap_rows), columns=RUNMAP_COLUMNS).to_csv(_files_dir(data_dir) / "runmap.csv", index=False)
    pd.DataFrame(list(baselines_rows), columns=BASELINES_COLUMNS).to_csv(_files_dir(data_dir) / "baselines.csv", index=False)


def test_appends_new_subject_to_empty_stores(data_dir):
    _write_stores(data_dir)

    commit.commit_ssvep(SSVEP_FIXTURE, data_dir)

    metadata = pd.read_csv(_files_dir(data_dir) / "metadata.csv", keep_default_na=False)
    assert len(metadata) == 1
    row = metadata.iloc[0]
    assert (row["sub_id"], row["session"], row["group"], row["subgroup"]) == ("MET000", 2, "CTR", "NA")

    runmap = pd.read_csv(_files_dir(data_dir) / "runmap.csv")
    baselines = pd.read_csv(_files_dir(data_dir) / "baselines.csv")
    assert len(runmap) == 10 * 10 * 4
    assert len(baselines) == 4 * 4
    assert (_files_dir(data_dir) / "grid.json").exists()


def test_raises_on_duplicate_key_without_overwrite(data_dir):
    _write_stores(data_dir, metadata_rows=[{"filename": "x.mat", "sub_id": "MET000", "session": 2, "group": "CTR", "subgroup": "NA"}])

    with pytest.raises(DuplicateKeyError):
        commit.commit_ssvep(SSVEP_FIXTURE, data_dir)

    metadata = pd.read_csv(_files_dir(data_dir) / "metadata.csv")
    assert len(metadata) == 1  # untouched


def test_overwrite_replaces_runmap_and_baselines_but_leaves_metadata_group_untouched(data_dir):
    _write_stores(
        data_dir,
        metadata_rows=[{"filename": "x.mat", "sub_id": "MET000", "session": 2, "group": "HAND_CORRECTED", "subgroup": "NA"}],
        runmap_rows=[{"sub_id": "MET000", "session": 2, "run": 1, "red_idx": 0, "green_idx": 0, "value": -999}],
        baselines_rows=[{"sub_id": "MET000", "session": 2, "run": 1, "trial": 1, "value": -999}],
    )

    commit.commit_ssvep(SSVEP_FIXTURE, data_dir, overwrite=True, group="SHOULD_NOT_APPLY")

    metadata = pd.read_csv(_files_dir(data_dir) / "metadata.csv")
    assert len(metadata) == 1
    assert metadata.iloc[0]["group"] == "HAND_CORRECTED"

    runmap = pd.read_csv(_files_dir(data_dir) / "runmap.csv")
    assert len(runmap) == 10 * 10 * 4
    assert -999 not in set(runmap["value"])


def test_sub_id_and_group_override_apply_on_a_new_key(data_dir):
    _write_stores(data_dir)

    commit.commit_ssvep(SSVEP_FIXTURE, data_dir, sub_id="MET777", group="PD", subgroup="protan")

    metadata = pd.read_csv(_files_dir(data_dir) / "metadata.csv", keep_default_na=False)
    row = metadata.iloc[0]
    assert (row["sub_id"], row["group"], row["subgroup"]) == ("MET777", "PD", "protan")


def test_session_override_targets_a_different_session(data_dir):
    _write_stores(data_dir)

    commit.commit_ssvep(SSVEP_FIXTURE, data_dir, session=5)

    metadata = pd.read_csv(_files_dir(data_dir) / "metadata.csv")
    assert metadata.iloc[0]["session"] == 5
    runmap = pd.read_csv(_files_dir(data_dir) / "runmap.csv")
    assert set(runmap["session"]) == {5}
    baselines = pd.read_csv(_files_dir(data_dir) / "baselines.csv")
    assert set(baselines["session"]) == {5}


def test_session_override_duplicate_check_uses_overridden_session(data_dir):
    _write_stores(data_dir, metadata_rows=[{"filename": "x.mat", "sub_id": "MET000", "session": 5, "group": "CTR", "subgroup": "NA"}])

    with pytest.raises(DuplicateKeyError):
        commit.commit_ssvep(SSVEP_FIXTURE, data_dir, session=5)

    # the fixture's own embedded session (2) is untouched and still free
    commit.commit_ssvep(SSVEP_FIXTURE, data_dir, session=2)
    metadata = pd.read_csv(_files_dir(data_dir) / "metadata.csv")
    assert set(metadata["session"]) == {5, 2}


def test_dry_run_writes_nothing_on_a_clear_key(data_dir):
    _write_stores(data_dir)

    commit.commit_ssvep(SSVEP_FIXTURE, data_dir, dry_run=True)

    metadata = pd.read_csv(_files_dir(data_dir) / "metadata.csv")
    assert len(metadata) == 0


def test_dry_run_still_raises_on_duplicate(data_dir):
    _write_stores(data_dir, metadata_rows=[{"filename": "x.mat", "sub_id": "MET000", "session": 2, "group": "CTR", "subgroup": "NA"}])

    with pytest.raises(DuplicateKeyError):
        commit.commit_ssvep(SSVEP_FIXTURE, data_dir, dry_run=True)

    metadata = pd.read_csv(_files_dir(data_dir) / "metadata.csv")
    assert len(metadata) == 1  # untouched
