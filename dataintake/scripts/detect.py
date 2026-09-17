"""Detect and classify files dropped into the data-intake directory.

Detection is content-first: a file's structure decides its domain, never its
name alone -- a file can be misnamed (see MET999.mat, whose embedded SubID is
actually MET000; the fixture of the same name in dataintake/tests/fixtures/
keeps that mismatch on purpose). Extension only narrows which check runs;
the actual signature is checked before a domain is assigned.
"""

from pathlib import Path

from _domain_loaders import load_domain_loader

BEH_HEADER = ["SubID", "Red", "Green", "RunNumber", "session", "PartType", "Date", "FolderOrg"]
FM100_FIELD_COUNT = 102
SSVEP_REQUIRED_KEYS = {"SubID", "session", "group", "subgroup", "runMap", "baselines", "redArray", "greenArray"}


def detect_file(path: Path) -> tuple[str | None, str | None, dict]:
    """Return (domain, sub_id, meta). domain/sub_id are None, and meta holds
    a "reason", when the file doesn't match any known domain's signature."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".mat":
        return _detect_ssvep(path)

    if suffix in (".csv", ".txt"):
        first_line = _first_line(path)
        if first_line is not None:
            if [c.strip() for c in first_line.split(",")] == BEH_HEADER:
                return _detect_beh(path)
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


def _detect_beh(path: Path):
    import pandas as pd

    df = pd.read_csv(path)
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


def _detect_fm100(path: Path):
    loader = load_domain_loader("fm100")
    fields = _first_line(path).split(",")
    raw_id = fields[loader.REFERENCE_COL].strip()
    sub_id, session = loader._parse_session_and_id(raw_id)
    meta = {"session": session}
    _flag_filename_mismatch(path, sub_id, meta)
    return "fm100", sub_id, meta


def _flag_filename_mismatch(path: Path, sub_id: str, meta: dict) -> None:
    filename_sub_id = _filename_sub_id(path)
    if filename_sub_id != sub_id:
        meta["filename_mismatch"] = filename_sub_id
