"""Write detected intake files into their domain's real data store.

Each commit_* function reuses its domain's own already-vendored format
knowledge (column names, group mappings, raw-column positions) rather than
redefining it, and never touches more than one domain's loader.py per call.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from _domain_loaders import load_domain_loader
from detect import BEH_HEADER, beh_csv_frame
from errors import DuplicateKeyError


def commit_beh(
    path: Path,
    data_dir: Path,
    *,
    sub_id: str | None = None,
    group: str | None = None,
    session: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> int:
    """Append path's rows to <data_dir>/manualTest/behavioral_table.csv.
    Returns the number of rows written. Raises DuplicateKeyError if any
    (sub_id, session) in the incoming file already exists and overwrite is
    False -- nothing is written in that case.

    `session`, if given, relabels every row in the file to that session --
    a blind, file-wide override, same as sub_id/group. The caller (the
    intake page) is responsible for only offering this when a file actually
    has a single session to begin with.

    `dry_run` runs every check (including the DuplicateKeyError check) but
    writes nothing -- used to validate every file in a multi-file batch
    before committing any of them, so one file's conflict can't leave an
    earlier file in the same batch already written with no way back."""
    loader = load_domain_loader("beh")
    df = _beh_json_frame(path) if path.suffix.lower() == ".json" else beh_csv_frame(path)

    if sub_id is not None:
        df["SubID"] = sub_id
    if session is not None:
        df["session"] = session
    if group is not None:
        part_type_by_group = {g: pt for pt, g in loader.PART_TYPE_GROUP.items()}
        if group not in part_type_by_group:
            raise ValueError(f"unknown group {group!r}, expected one of {sorted(part_type_by_group)}")
        df["PartType"] = part_type_by_group[group]

    master_path = Path(data_dir) / "manualTest" / "behavioral_table.csv"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame(columns=BEH_HEADER)

    new_keys = set(zip(df["SubID"], df["session"]))
    existing_keys = set(zip(master["SubID"], master["session"])) if len(master) else set()
    duplicates = new_keys & existing_keys
    if duplicates and not overwrite:
        raise DuplicateKeyError(f"behavioral data already exists for {sorted(duplicates)}")

    if dry_run:
        return len(df)

    if duplicates:
        master = master[~master.set_index(["SubID", "session"]).index.isin(pd.MultiIndex.from_tuples(duplicates))]

    pd.concat([master, df[BEH_HEADER]], ignore_index=True).to_csv(master_path, index=False)
    return len(df)


def _beh_json_frame(path: Path) -> pd.DataFrame:
    """Convert a behavioral JSON export to the master table's rows. SubID is
    the filename stem; session 1 and PartType 0 (unknown group) are
    placeholders for the caller's overrides to replace."""
    d = json.loads(path.read_text())
    raw = pd.DataFrame(d["data"], columns=d["columns"])
    meta = d["metadata"]
    return pd.DataFrame(
        {
            "SubID": path.stem.split("_")[0],
            "Red": raw["Red"],
            "Green": raw["Green"],
            "RunNumber": raw["TrialNumber"],
            "session": 1,
            "PartType": 0,
            "Date": datetime.fromisoformat(meta["saved_at"]).strftime("%d_%b_%y").upper(),
            "FolderOrg": meta["experiment_name"],
        }
    )


def commit_fm100(
    path: Path, data_dir: Path, *, sub_id: str | None = None, session: int | None = None, overwrite: bool = False, dry_run: bool = False
) -> int:
    """Append path's raw line(s) to <data_dir>/standardizedScores/repeatedSessionsPY.txt.

    Each line's session is auto-assigned to the next free slot (1/2/3, encoded
    as the reference-field suffix none/b/c, per FM100's own convention) unless
    `session` is given explicitly. Raises DuplicateKeyError if the target
    session already exists and overwrite is False. Returns rows written.

    `dry_run` runs the same validation without writing -- see commit_beh's
    docstring for why (multi-file batch validation)."""
    loader = load_domain_loader("fm100")
    master_path = Path(data_dir) / "standardizedScores" / "repeatedSessionsPY.txt"
    if not master_path.exists():
        raise FileNotFoundError(f"{master_path} does not exist -- the master FM100 export must already exist before intake can append to it")

    master_lines = master_path.read_text().splitlines()
    glitch_line, data_lines = master_lines[0], master_lines[1:]

    incoming_lines = [line for line in path.read_text().splitlines() if line.strip()]
    new_lines = []
    for raw_line in incoming_lines:
        fields = raw_line.split(",")
        raw_sub_id, _ = loader._parse_session_and_id(fields[loader.REFERENCE_COL].strip())
        target_sub_id = sub_id if sub_id is not None else raw_sub_id

        existing_sessions = _fm100_sessions_for(data_lines, loader, target_sub_id)
        target_session = session if session is not None else _fm100_next_session(existing_sessions)

        if target_session in existing_sessions:
            if not overwrite:
                raise DuplicateKeyError(f"fm100 data already exists for {target_sub_id} session {target_session}")
            data_lines = _fm100_drop_session(data_lines, loader, target_sub_id, target_session)

        fields[loader.REFERENCE_COL] = _fm100_reference(target_sub_id, target_session)
        new_lines.append(",".join(fields))

    if dry_run:
        return len(new_lines)

    master_path.write_text("\n".join([glitch_line] + data_lines + new_lines) + "\n")
    return len(new_lines)


def _fm100_sessions_for(data_lines: list[str], loader, sub_id: str) -> set[int]:
    sessions = set()
    for line in data_lines:
        if not line.strip():
            continue
        fields = line.split(",")
        line_sub_id, line_session = loader._parse_session_and_id(fields[loader.REFERENCE_COL].strip())
        if line_sub_id == sub_id:
            sessions.add(line_session)
    return sessions


def _fm100_next_session(existing_sessions: set[int]) -> int:
    for candidate in (1, 2, 3):
        if candidate not in existing_sessions:
            return candidate
    raise ValueError("no session slots left -- fm100's suffix convention only supports sessions 1-3")


def _fm100_drop_session(data_lines: list[str], loader, sub_id: str, session: int) -> list[str]:
    kept = []
    for line in data_lines:
        if not line.strip():
            kept.append(line)
            continue
        fields = line.split(",")
        line_sub_id, line_session = loader._parse_session_and_id(fields[loader.REFERENCE_COL].strip())
        if line_sub_id == sub_id and line_session == session:
            continue
        kept.append(line)
    return kept


def _fm100_reference(sub_id: str, session: int) -> str:
    return {1: sub_id, 2: sub_id + "b", 3: sub_id + "c"}[session]


def suggest_fm100_session(data_dir: Path, sub_id: str) -> int:
    """Next free fm100 session slot (1/2/3) for sub_id, given data_dir's
    current repeatedSessionsPY.txt -- for the intake UI to show a live,
    accurate default next to its editable session field. commit_fm100
    recomputes this itself at commit time regardless, so the two can never
    drift apart. A missing master file (nothing committed yet) suggests 1."""
    loader = load_domain_loader("fm100")
    master_path = Path(data_dir) / "standardizedScores" / "repeatedSessionsPY.txt"
    if not master_path.exists():
        return 1
    data_lines = master_path.read_text().splitlines()[1:]
    return _fm100_next_session(_fm100_sessions_for(data_lines, loader, sub_id))


def _read_or_empty_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(columns=columns)


def commit_ssvep(
    path: Path,
    data_dir: Path,
    *,
    sub_id: str | None = None,
    group: str | None = None,
    subgroup: str | None = None,
    session: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> int:
    """Write path's .mat data into <data_dir>/ssveps/files/{metadata,runmap,baselines}.csv.

    Raises DuplicateKeyError if (sub_id, session) already exists and overwrite
    is False. On an overwrite, only runmap/baselines rows are replaced --
    metadata.csv's group/subgroup for an existing key is never touched here,
    matching update_derived.py's own policy (hand-correct metadata.csv
    directly to change it). Returns 1 (one file committed).

    `dry_run` runs the same validation without writing -- see commit_beh's
    docstring for why (multi-file batch validation)."""
    import json

    loader = load_domain_loader("ssvep")
    d = loader.load_ssvep(str(path))
    if sub_id is not None:
        d["SubID"] = sub_id
    if group is not None:
        d["group"] = group
    if subgroup is not None:
        d["subgroup"] = subgroup
    if session is not None:
        d["session"] = session

    metadata_row, runmap_rows, baseline_rows = loader.to_rows(d, path.name)
    key = (metadata_row["sub_id"], metadata_row["session"])

    files_dir = Path(data_dir) / "ssveps" / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = files_dir / "metadata.csv"
    runmap_path = files_dir / "runmap.csv"
    baselines_path = files_dir / "baselines.csv"
    grid_path = files_dir / "grid.json"

    metadata_df = _read_or_empty_csv(metadata_path, loader.CSV_COLUMNS["metadata"])
    runmap_df = _read_or_empty_csv(runmap_path, loader.CSV_COLUMNS["runmap"])
    baselines_df = _read_or_empty_csv(baselines_path, loader.CSV_COLUMNS["baselines"])

    existing_keys = set(zip(metadata_df["sub_id"], metadata_df["session"])) if len(metadata_df) else set()
    if key in existing_keys and not overwrite:
        raise DuplicateKeyError(f"ssvep data already exists for {key[0]} session {key[1]}")

    if dry_run:
        return 1

    if key in existing_keys:
        key_index = pd.MultiIndex.from_tuples([key])
        runmap_df = runmap_df[~runmap_df.set_index(["sub_id", "session"]).index.isin(key_index)]
        baselines_df = baselines_df[~baselines_df.set_index(["sub_id", "session"]).index.isin(key_index)]
    else:
        metadata_df = pd.concat([metadata_df, pd.DataFrame([metadata_row])], ignore_index=True)

    runmap_df = pd.concat([runmap_df, pd.DataFrame(runmap_rows)], ignore_index=True)
    baselines_df = pd.concat([baselines_df, pd.DataFrame(baseline_rows)], ignore_index=True)

    loader.write_derived_csv(metadata_path, metadata_df, "metadata")
    loader.write_derived_csv(runmap_path, runmap_df, "runmap")
    loader.write_derived_csv(baselines_path, baselines_df, "baselines")

    if not grid_path.exists():
        grid = {
            "baseDIM": d["baseDIM"],
            "mapDIM": d["mapDIM"],
            "redArray": d["redArray"].tolist(),
            "greenArray": d["greenArray"].tolist(),
        }
        grid_path.write_text(json.dumps(grid, indent=2))

    return 1
