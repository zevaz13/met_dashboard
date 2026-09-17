"""Regression tests for removing FM100 data, with backup + restore.

Run: uv run pytest dataremoval/tests -q
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

sys.modules.pop("delete", None)
sys.modules.pop("errors", None)

import delete  # noqa: E402
from errors import RowNotFoundError  # noqa: E402


def _fm100_line(ref: str) -> str:
    fields = ["", "", "", ref] + ["0"] * 10  # REFERENCE_COL == 3
    return ",".join(fields)


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "standardizedScores").mkdir()
    return tmp_path


@pytest.fixture
def backup_dir(tmp_path):
    d = tmp_path / "deletedData"
    d.mkdir()
    return d


def _master_path(data_dir):
    return data_dir / "standardizedScores" / "repeatedSessionsPY.txt"


def _seed(data_dir):
    lines = [_fm100_line("MET999"), _fm100_line("MET999b"), _fm100_line("MET001")]
    glitch = lines[0]
    _master_path(data_dir).write_text("\n".join([glitch] + lines) + "\n")


def _refs(data_dir):
    lines = _master_path(data_dir).read_text().splitlines()[1:]
    return [line.split(",")[3] for line in lines if line.strip()]


def test_delete_removes_only_the_matching_session(data_dir, backup_dir):
    _seed(data_dir)

    delete.delete_fm100(data_dir, backup_dir, "MET999", 1)

    refs = _refs(data_dir)
    assert "MET999" not in refs
    assert "MET999b" in refs
    assert "MET001" in refs


def test_delete_writes_a_backup_with_the_removed_line(data_dir, backup_dir):
    _seed(data_dir)

    backup_path = delete.delete_fm100(data_dir, backup_dir, "MET999", 2)

    assert backup_path.exists()
    content = backup_path.read_text().strip()
    assert content.split(",")[3] == "MET999b"


def test_delete_raises_when_session_does_not_exist(data_dir, backup_dir):
    _seed(data_dir)

    with pytest.raises(RowNotFoundError):
        delete.delete_fm100(data_dir, backup_dir, "MET999", 3)

    assert len(_refs(data_dir)) == 3  # untouched


def test_dry_run_writes_no_backup_and_removes_nothing(data_dir, backup_dir):
    _seed(data_dir)

    result = delete.delete_fm100(data_dir, backup_dir, "MET999", 1, dry_run=True)

    assert result is None
    assert list(backup_dir.iterdir()) == []
    assert "MET999" in _refs(data_dir)


def test_restore_puts_the_exact_line_back(data_dir, backup_dir):
    _seed(data_dir)
    backup_path = delete.delete_fm100(data_dir, backup_dir, "MET999", 1)

    delete.restore_fm100(data_dir, backup_path)

    assert "MET999" in _refs(data_dir)
