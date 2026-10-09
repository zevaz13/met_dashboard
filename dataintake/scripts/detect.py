"""Detect and classify files dropped into the data-intake directory.

Detection is content-first: a file's structure decides its domain, never its
name alone -- a file can be misnamed (see MET999.mat, whose embedded SubID is
actually MET000; the fixture of the same name in dataintake/tests/fixtures/
keeps that mismatch on purpose). Extension only narrows which check runs;
the actual signature is checked before a domain is assigned.
"""

import json
from datetime import datetime
from pathlib import Path

from _domain_loaders import load_domain_loader

BEH_HEADER = ["SubID", "Red", "Green", "RunNumber", "session", "PartType", "Date", "FolderOrg"]
BEH_JSON_REQUIRED_COLUMNS = {"TrialNumber", "Red", "Green"}
FM100_FIELD_COUNT = 102
SSVEP_REQUIRED_KEYS = {"SubID", "session", "group", "subgroup", "runMap", "baselines", "redArray", "greenArray"}


def detect_file(path: Path) -> tuple[str | None, str | None, dict]:
    """Return (domain, sub_id, meta). domain/sub_id are None, and meta holds
    a "reason", when the file doesn't match any known domain's signature."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".mat":
        return _detect_ssvep(path)

    if suffix == ".json":
        return _detect_beh_json(path)

    if suffix in (".csv", ".txt"):
        first_line = _first_line(path)
        if first_line is not None:
            header = {c.strip().lower() for c in first_line.split(",")}
            if {"red", "green"} <= header:
                return _detect_beh(path)
            if header & {"red", "green"}:
                missing = "Green" if "red" in header else "Red"
                return None, None, {"reason": f"behavioral CSV needs both Red and Green columns, missing {missing}: {path.name}"}
            if len(first_line.split(",")) == FM100_FIELD_COUNT:
                return _detect_fm100(path)

    return None, None, {"reason": f"unrecognized file: {path.name}"}


def _first_line(path: Path) -> str | None:
    if path.stat().st_size == 0:
        return None
    with path.open() as f:
        return f.readline().rstrip("\n")


def _filename_sub_id(path: Path) -> str:
    return path.stem.split("_")[0]


def _detect_ssvep(path: Path):
    loader = load_domain_loader("ssvep")
    try:
        d = loader.load_ssvep(str(path))
    except Exception as exc:
        return None, None, {"reason": f"could not read .mat file: {exc}"}

    missing = SSVEP_REQUIRED_KEYS - d.keys()
    if missing:
        return None, None, {"reason": f"missing expected fields: {sorted(missing)}"}

    sub_id = str(d["SubID"])
    meta = {"session": int(d["session"]), "group": str(d["group"]), "subgroup": str(d["subgroup"])}
    _flag_filename_mismatch(path, sub_id, meta)
    return "ssvep", sub_id, meta


def beh_csv_frame(path: Path):
    """A behavioral CSV in the master table's schema. Only Red and Green
    (any case) are required; every other BEH_HEADER column that's missing
    gets a default the intake page lets the user override."""
    import pandas as pd

    df = pd.read_csv(path)
    canonical = {c.lower(): c for c in BEH_HEADER}
    df = df.rename(columns={c: canonical[c.strip().lower()] for c in df.columns if c.strip().lower() in canonical})
    defaults = {
        "SubID": _filename_sub_id(path),
        "RunNumber": range(1, len(df) + 1),
        "session": 1,
        "PartType": 0,
        "Date": datetime.fromtimestamp(path.stat().st_mtime).strftime("%d_%b_%y").upper(),
        "FolderOrg": path.name,
    }
    for column, value in defaults.items():
        if column not in df.columns:
            df[column] = value
    return df[BEH_HEADER]


def _detect_beh(path: Path):
    df = beh_csv_frame(path)
    sub_ids = df["SubID"].unique()
    if len(sub_ids) != 1:
        return None, None, {"reason": f"expected exactly one SubID per file, found {list(sub_ids)}"}

    loader = load_domain_loader("beh")
    sub_id = str(sub_ids[0])
    part_type = int(df["PartType"].iloc[0])
    meta = {
        "group": loader.PART_TYPE_GROUP.get(part_type, "UNKNOWN"),
        "sessions": sorted(int(s) for s in df["session"].unique()),
    }
    _flag_filename_mismatch(path, sub_id, meta)
    return "beh", sub_id, meta


def _detect_beh_json(path: Path):
    """Behavioral task export: no subject ID, group, or session inside the
    file, so sub_id comes from the filename and group/session are defaults
    the intake page lets the user override."""
    try:
        d = json.loads(path.read_text())
        is_beh = d["metadata"]["mode"] == "behavioral" and BEH_JSON_REQUIRED_COLUMNS <= set(d["columns"])
    except (ValueError, KeyError, TypeError):
        return None, None, {"reason": f"not a behavioral JSON export: {path.name}"}
    if not is_beh:
        return None, None, {"reason": f"not a behavioral JSON export: {path.name}"}
    return "beh", _filename_sub_id(path), {"group": "UNKNOWN", "sessions": [1], "format": "json"}


def _detect_fm100(path: Path):
    # No "session" in meta here on purpose: the raw line carries no reliable
    # session marker of its own (a fresh export is always unsuffixed), so the
    # only trustworthy answer is commit.suggest_fm100_session(data_dir, sub_id),
    # which needs to know the *target* data store, not just this file.
    loader = load_domain_loader("fm100")
    fields = _first_line(path).split(",")
    raw_id = fields[loader.REFERENCE_COL].strip()
    sub_id, _ = loader._parse_session_and_id(raw_id)
    meta = {}
    _flag_filename_mismatch(path, sub_id, meta)
    return "fm100", sub_id, meta


def _flag_filename_mismatch(path: Path, sub_id: str, meta: dict) -> None:
    filename_sub_id = _filename_sub_id(path)
    if filename_sub_id != sub_id:
        meta["filename_mismatch"] = filename_sub_id
