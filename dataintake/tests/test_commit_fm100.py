"""Regression tests for committing new FM100 data into the master raw file.

repeatedSessionsPY.txt has no real header row -- its first line is a
byte-identical duplicate of the second (a data-export glitch the loader
always skips via skiprows=1). Fixtures here replicate that shape so tests
exercise the same file structure the loader actually reads.

Run: uv run pytest dataintake/tests -q
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

sys.modules.pop("commit", None)
sys.modules.pop("errors", None)

import commit  # noqa: E402
from errors import DuplicateKeyError  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FM100_FIXTURE = FIXTURES / "MET999_FM100.txt"
FM100_ROW = FM100_FIXTURE.read_text().splitlines()[0]


def _row_for(sub_id: str) -> str:
    fields = FM100_ROW.split(",")
    fields[3] = sub_id  # REFERENCE_COL
    return ",".join(fields)


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "standardizedScores").mkdir()
    return tmp_path


def _master_path(data_dir):
    return data_dir / "standardizedScores" / "repeatedSessionsPY.txt"


def _write_master(data_dir, data_rows):
    # line 0 is the always-skipped glitch duplicate; reuse the first data row.
    glitch = data_rows[0] if data_rows else _row_for("MET000")
    _master_path(data_dir).write_text("\n".join([glitch] + data_rows) + "\n")


def _sessions_for(data_dir, sub_id, loader_ref_col=3):
    lines = _master_path(data_dir).read_text().splitlines()[1:]
    out = []
    for line in lines:
        fields = line.split(",")
        ref = fields[loader_ref_col].strip()
        if ref.rstrip("bc") == sub_id:
            out.append(ref)
    return out


def test_appends_new_subject_as_session_one_no_suffix(data_dir):
    _write_master(data_dir, [])

    written = commit.commit_fm100(FM100_FIXTURE, data_dir)

    assert written == 1
    assert "MET999" in _sessions_for(data_dir, "MET999")


def test_second_submission_for_same_subject_gets_b_suffix(data_dir):
    _write_master(data_dir, [_row_for("MET999")])

    commit.commit_fm100(FM100_FIXTURE, data_dir)

    assert set(_sessions_for(data_dir, "MET999")) == {"MET999", "MET999b"}


def test_raises_on_duplicate_explicit_session_without_overwrite(data_dir):
    _write_master(data_dir, [_row_for("MET999")])

    with pytest.raises(DuplicateKeyError):
        commit.commit_fm100(FM100_FIXTURE, data_dir, session=1)

    assert _sessions_for(data_dir, "MET999") == ["MET999"]  # untouched


def test_overwrite_replaces_existing_session(data_dir):
    _write_master(data_dir, [_row_for("MET999")])

    commit.commit_fm100(FM100_FIXTURE, data_dir, session=1, overwrite=True)

    assert _sessions_for(data_dir, "MET999") == ["MET999"]


def test_sub_id_override_targets_a_different_subject(data_dir):
    _write_master(data_dir, [])

    commit.commit_fm100(FM100_FIXTURE, data_dir, sub_id="MET500")

    assert _sessions_for(data_dir, "MET500") == ["MET500"]
    assert _sessions_for(data_dir, "MET999") == []


def test_missing_master_file_raises_clear_error(data_dir):
    with pytest.raises(FileNotFoundError):
        commit.commit_fm100(FM100_FIXTURE, data_dir)
