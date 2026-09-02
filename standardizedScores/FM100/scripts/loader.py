"""Load the Farnsworth-Munsell 100 Hue (FM100) test data.

Raw data: /home/sebas/data/standardizedScores/repeatedSessionsPY.txt (read in
place, not copied into the repo) -- one row per (subject, session).
"""

import os

import numpy as np
import pandas as pd

# Data lives outside this repo's git history (per-subject data is never
# committed) -- MET_DASHBOARD_DATA_DIR overrides where it's found, defaulting
# to <repo_root>/data (standardizedScores/FM100/scripts/ -> repo root is
# three levels up).
DATA_DIR = os.environ.get(
    "MET_DASHBOARD_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"),
)

RAW_PATH = os.path.join(DATA_DIR, "standardizedScores", "repeatedSessionsPY.txt")
SSVEP_METADATA_PATH = os.path.join(DATA_DIR, "ssveps", "files", "metadata.csv")

N_CAPS = 85
# 0-indexed raw-file columns. The raw export has no real header row (its
# first line is a byte-identical duplicate of the second, MET000, row -- a
# data export glitch, not a header; skiprows=1 in load_fm100_raw drops it).
# Columns beyond these (an apparent duration field and an unlabeled numeric
# field) have inconsistent formats across rows ("8", "6 min", "715") and
# aren't used by any analysis here, so they're dropped rather than guessed at.
REFERENCE_COL = 3
DATE_COL = 5
SEX_COL = 8
CAPS_START_COL = 16  # 85 consecutive columns from here: the arranged cap IDs


def _parse_session_and_id(raw_id: str) -> tuple[str, int]:
    """Extract sub_id and session number from the raw Reference field.
    MET000 -> (MET000, 1); MET000b -> (MET000, 2); MET000c -> (MET000, 3)."""
    if raw_id.endswith("b"):
        return raw_id[:-1], 2
    if raw_id.endswith("c"):
        return raw_id[:-1], 3
    return raw_id, 1


def load_fm100_raw(path: str = RAW_PATH, *, ssvep_metadata_path: str = SSVEP_METADATA_PATH) -> pd.DataFrame:
    """Tidy FM100 table: sub_id, session, group, subgroup, sex, date, caps --
    one row per (subject, session). caps is a length-85 int array, the cap
    IDs in the order the participant placed them (position i -> cap ID).

    group/subgroup are looked up live from ssvep_metadata_path by sub_id,
    same pattern as beh/scripts/loader.py -- these are the same
    participants, and subgroup is genuinely shared data. A subject absent
    from that file (e.g. MET047, who has no SSVEP or behavioral data) gets
    group='UNKNOWN', subgroup='NA'."""
    raw = pd.read_csv(path, header=None, skiprows=1, skip_blank_lines=True, encoding="utf-8-sig")

    sub_ids, sessions = [], []
    for ref in raw.iloc[:, REFERENCE_COL]:
        sub_id, session = _parse_session_and_id(str(ref).strip())
        sub_ids.append(sub_id)
        sessions.append(session)

    caps = raw.iloc[:, CAPS_START_COL : CAPS_START_COL + N_CAPS].to_numpy(dtype=int)

    df = pd.DataFrame(
        {
            "sub_id": sub_ids,
            "session": sessions,
            "sex": raw.iloc[:, SEX_COL].to_numpy(),
            "date": raw.iloc[:, DATE_COL].to_numpy(),
            "caps": list(caps),
        }
    )

    ssvep_meta = pd.read_csv(ssvep_metadata_path, keep_default_na=False)
    lookup = ssvep_meta.drop_duplicates("sub_id").set_index("sub_id")
    df["group"] = df["sub_id"].map(lookup["group"]).fillna("UNKNOWN")
    df["subgroup"] = df["sub_id"].map(lookup["subgroup"]).fillna("NA")

    return df[["sub_id", "session", "group", "subgroup", "sex", "date", "caps"]]


def subjects_in_group(df: pd.DataFrame, *, group: str | None = None, subgroup: str | None = None) -> list[str]:
    """Subject IDs matching group/subgroup (mirrors beh's/ssveps' function
    of the same name -- independently implemented per project, by
    convention, not by design; see beh/README.md's Tests section)."""
    sub = df
    if group is not None:
        sub = sub[sub["group"] == group]
    if subgroup is not None:
        sub = sub[sub["subgroup"] == subgroup]
    return sorted(sub["sub_id"].unique())
