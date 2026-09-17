"""Enumerate what data already exists for a subject, across all three domain
stores -- the reverse of dataintake's detect.py, which enumerates incoming
files instead of existing ones.
"""

from pathlib import Path

import pandas as pd

from _domain_loaders import load_domain_loader


def _ssvep_metadata_path(data_dir: Path) -> Path:
    return Path(data_dir) / "ssveps" / "files" / "metadata.csv"


def _beh_master_path(data_dir: Path) -> Path:
    return Path(data_dir) / "manualTest" / "behavioral_table.csv"


def _fm100_master_path(data_dir: Path) -> Path:
    return Path(data_dir) / "standardizedScores" / "repeatedSessionsPY.txt"


def _ssvep_sub_ids_and_sessions(data_dir: Path) -> list[tuple[str, int]]:
    path = _ssvep_metadata_path(data_dir)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return list(zip(df["sub_id"], df["session"]))


def _beh_sub_ids_and_sessions(data_dir: Path) -> list[tuple[str, int]]:
    path = _beh_master_path(data_dir)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if len(df) == 0:
        return []
    return sorted(set(zip(df["SubID"], df["session"])))


def _fm100_sub_ids_and_sessions(data_dir: Path) -> list[tuple[str, int]]:
    path = _fm100_master_path(data_dir)
    if not path.exists():
        return []
    loader = load_domain_loader("fm100")
    lines = path.read_text().splitlines()[1:]  # line 0 is the always-skipped glitch duplicate
    out = []
    for line in lines:
        if not line.strip():
            continue
        fields = line.split(",")
        sub_id, session = loader._parse_session_and_id(fields[loader.REFERENCE_COL].strip())
        out.append((sub_id, session))
    return out


def rows_for(data_dir: Path, sub_id: str) -> list[dict]:
    """Every existing (domain, session) row for sub_id, across all three stores."""
    rows = []
    for domain, pairs in (
        ("ssvep", _ssvep_sub_ids_and_sessions(data_dir)),
        ("beh", _beh_sub_ids_and_sessions(data_dir)),
        ("fm100", _fm100_sub_ids_and_sessions(data_dir)),
    ):
        for pair_sub_id, session in pairs:
            if pair_sub_id == sub_id:
                rows.append({"domain": domain, "session": int(session)})
    return rows


def list_subjects(data_dir: Path) -> list[str]:
    """Every sub_id appearing in any of the three stores, sorted."""
    sub_ids = set()
    for pairs in (
        _ssvep_sub_ids_and_sessions(data_dir),
        _beh_sub_ids_and_sessions(data_dir),
        _fm100_sub_ids_and_sessions(data_dir),
    ):
        sub_ids.update(sub_id for sub_id, _ in pairs)
    return sorted(sub_ids)
