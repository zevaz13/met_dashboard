"""Remove existing data from a domain's real store, with a backup written
first -- and the exact inverse, restoring a backup back into that store.

Each backup is written in a shape that stands alone (fm100/beh backups even
happen to be re-intakable via dataintake's own detection), but restore_*
re-inserts a backup's contents directly rather than routing through
dataintake's commit_* functions -- restoring must reproduce the exact
original data, not re-run override/dedup logic that has nothing to do with
restoring.
"""

import json
import time
from pathlib import Path

import pandas as pd

from _domain_loaders import load_domain_loader
from errors import RowNotFoundError

BEH_COLUMNS = ["SubID", "Red", "Green", "RunNumber", "session", "PartType", "Date", "FolderOrg"]


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S") + f"{time.time_ns() % 1_000_000:06d}"


def _beh_master_path(data_dir: Path) -> Path:
    return Path(data_dir) / "manualTest" / "behavioral_table.csv"


def delete_beh(data_dir: Path, backup_dir: Path, sub_id: str, session: int, *, dry_run: bool = False) -> Path | None:
    """Remove sub_id's session rows from behavioral_table.csv, after backing
    them up to backup_dir. Raises RowNotFoundError if no such rows exist.
    dry_run performs the same check and raises the same way, but writes
    nothing and returns None."""
    master_path = _beh_master_path(data_dir)
    master = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame(columns=BEH_COLUMNS)

    mask = (master["SubID"] == sub_id) & (master["session"] == session)
    if not mask.any():
        raise RowNotFoundError(f"no behavioral data for {sub_id} session {session}")

    if dry_run:
        return None

    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{_timestamp()}_{sub_id}_beh_session{session}.csv"
    master[mask].to_csv(backup_path, index=False)

    master[~mask].to_csv(master_path, index=False)
    return backup_path


def restore_beh(data_dir: Path, backup_path: Path) -> None:
    """Re-insert a delete_beh backup's exact rows back into behavioral_table.csv."""
    master_path = _beh_master_path(data_dir)
    master = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame(columns=BEH_COLUMNS)
    backup = pd.read_csv(backup_path)
    pd.concat([master, backup], ignore_index=True).to_csv(master_path, index=False)


def _fm100_master_path(data_dir: Path) -> Path:
    return Path(data_dir) / "standardizedScores" / "repeatedSessionsPY.txt"


def delete_fm100(data_dir: Path, backup_dir: Path, sub_id: str, session: int, *, dry_run: bool = False) -> Path | None:
    """Remove sub_id's session line from repeatedSessionsPY.txt, after backing
    it up to backup_dir. Raises RowNotFoundError if no such line exists.
    dry_run performs the same check and raises the same way, but writes
    nothing and returns None."""
    loader = load_domain_loader("fm100")
    master_path = _fm100_master_path(data_dir)
    master_lines = master_path.read_text().splitlines() if master_path.exists() else []
    glitch_line, data_lines = (master_lines[0], master_lines[1:]) if master_lines else (None, [])

    target_line = None
    remaining_lines = []
    for line in data_lines:
        if not line.strip():
            continue
        fields = line.split(",")
        line_sub_id, line_session = loader._parse_session_and_id(fields[loader.REFERENCE_COL].strip())
        if target_line is None and line_sub_id == sub_id and line_session == session:
            target_line = line
        else:
            remaining_lines.append(line)

    if target_line is None:
        raise RowNotFoundError(f"no fm100 data for {sub_id} session {session}")

    if dry_run:
        return None

    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{_timestamp()}_{sub_id}_fm100_session{session}.txt"
    backup_path.write_text(target_line + "\n")

    master_path.write_text("\n".join([glitch_line] + remaining_lines) + "\n")
    return backup_path


def restore_fm100(data_dir: Path, backup_path: Path) -> None:
    """Re-insert a delete_fm100 backup's exact line back into repeatedSessionsPY.txt."""
    line = Path(backup_path).read_text().strip()
    master_path = _fm100_master_path(data_dir)
    master_lines = master_path.read_text().splitlines()
    master_path.write_text("\n".join(master_lines + [line]) + "\n")


def _ssvep_files_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "ssveps" / "files"


def _read_or_empty_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(columns=columns)


def delete_ssvep(data_dir: Path, backup_dir: Path, sub_id: str, session: int, *, dry_run: bool = False) -> Path | None:
    """Remove sub_id's session rows from metadata.csv/runmap.csv/baselines.csv,
    after backing them up to backup_dir. Raises RowNotFoundError if no such
    (sub_id, session) exists. dry_run performs the same check and raises the
    same way, but writes nothing and returns None."""
    loader = load_domain_loader("ssvep")
    files_dir = _ssvep_files_dir(data_dir)
    metadata_df = _read_or_empty_csv(files_dir / "metadata.csv", loader.CSV_COLUMNS["metadata"])
    runmap_df = _read_or_empty_csv(files_dir / "runmap.csv", loader.CSV_COLUMNS["runmap"])
    baselines_df = _read_or_empty_csv(files_dir / "baselines.csv", loader.CSV_COLUMNS["baselines"])

    metadata_mask = (metadata_df["sub_id"] == sub_id) & (metadata_df["session"] == session)
    if not metadata_mask.any():
        raise RowNotFoundError(f"no ssvep data for {sub_id} session {session}")

    if dry_run:
        return None

    runmap_mask = (runmap_df["sub_id"] == sub_id) & (runmap_df["session"] == session)
    baselines_mask = (baselines_df["sub_id"] == sub_id) & (baselines_df["session"] == session)

    backup = {
        "metadata_row": metadata_df[metadata_mask].iloc[0].to_dict(),
        "runmap_rows": runmap_df[runmap_mask].to_dict("records"),
        "baseline_rows": baselines_df[baselines_mask].to_dict("records"),
    }
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{_timestamp()}_{sub_id}_ssvep_session{session}.json"
    backup_path.write_text(json.dumps(backup, indent=2))

    loader.write_derived_csv(files_dir / "metadata.csv", metadata_df[~metadata_mask], "metadata")
    loader.write_derived_csv(files_dir / "runmap.csv", runmap_df[~runmap_mask], "runmap")
    loader.write_derived_csv(files_dir / "baselines.csv", baselines_df[~baselines_mask], "baselines")
    return backup_path


def restore_ssvep(data_dir: Path, backup_path: Path) -> None:
    """Re-insert a delete_ssvep backup's exact rows back into
    metadata.csv/runmap.csv/baselines.csv."""
    loader = load_domain_loader("ssvep")
    backup = json.loads(Path(backup_path).read_text())
    files_dir = _ssvep_files_dir(data_dir)

    metadata_df = _read_or_empty_csv(files_dir / "metadata.csv", loader.CSV_COLUMNS["metadata"])
    runmap_df = _read_or_empty_csv(files_dir / "runmap.csv", loader.CSV_COLUMNS["runmap"])
    baselines_df = _read_or_empty_csv(files_dir / "baselines.csv", loader.CSV_COLUMNS["baselines"])

    metadata_df = pd.concat([metadata_df, pd.DataFrame([backup["metadata_row"]])], ignore_index=True)
    runmap_df = pd.concat([runmap_df, pd.DataFrame(backup["runmap_rows"])], ignore_index=True)
    baselines_df = pd.concat([baselines_df, pd.DataFrame(backup["baseline_rows"])], ignore_index=True)

    loader.write_derived_csv(files_dir / "metadata.csv", metadata_df, "metadata")
    loader.write_derived_csv(files_dir / "runmap.csv", runmap_df, "runmap")
    loader.write_derived_csv(files_dir / "baselines.csv", baselines_df, "baselines")
