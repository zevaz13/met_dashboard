"""Load the manual/behavioral point-matching task data.

Raw data: /home/sebas/data/manualTest/behavioral_table.csv (read in place,
not copied into the repo) -- one row per (subject, session, click), already
tidy, no build step needed.

CVD subgroup (protan/deutan) isn't recorded in this file -- it's looked up
live from ssveps/files/metadata.csv at load time, since these are the same
participants who completed the SSVEP experiment and subgroup is genuinely
shared data, not duplicated entry (decided over a persisted copy: no build
step to remember to rerun, always consistent with ssveps' own hand-corrected
values).
"""

import os

import pandas as pd

# Data lives outside this repo's git history (per-subject data is never
# committed) -- MET_DASHBOARD_DATA_DIR overrides where it's found, defaulting
# to <repo_root>/data (beh/scripts/ -> repo root is two levels up).
DATA_DIR = os.environ.get(
    "MET_DASHBOARD_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data"),
)

RAW_PATH = os.path.join(DATA_DIR, "manualTest", "behavioral_table.csv")
SSVEP_METADATA_PATH = os.path.join(DATA_DIR, "ssveps", "files", "metadata.csv")

# PartType -> group (0 = group not yet assigned, e.g. JSON intake), confirmed against ssveps/files/metadata.csv's own group
# column for every one of the 43 subjects present in both datasets.
PART_TYPE_GROUP = {0: "UNKNOWN", 1: "CTR", 2: "CVD", 3: "PD", 4: "HD"}


def load_behavioral(path: str = RAW_PATH, *, ssvep_metadata_path: str = SSVEP_METADATA_PATH) -> pd.DataFrame:
    """Tidy behavioral table: sub_id, session, click, red, green, group,
    subgroup, date, folder -- one row per click (a single red/green match
    attempt). `click` is the raw file's `RunNumber`, renamed because it
    numbers individual clicks within a session, not a full grid repeat like
    ssveps' "run" -- there is no grid here, just a sequence of point matches.

    `group` is derived from `PartType`. `subgroup` (protan/deutan/NA) is
    looked up from `ssvep_metadata_path` by sub_id; a subject absent from
    that file (tested behaviorally but not, or not yet, on SSVEP) gets
    subgroup 'NA', matching ssveps' own convention for non-CVD subjects."""
    raw = pd.read_csv(path)
    df = raw.rename(
        columns={
            "SubID": "sub_id",
            "Red": "red",
            "Green": "green",
            "RunNumber": "click",
            "PartType": "part_type",
            "Date": "date",
            "FolderOrg": "folder",
        }
    )
    df["group"] = df["part_type"].map(PART_TYPE_GROUP)

    ssvep_meta = pd.read_csv(ssvep_metadata_path, keep_default_na=False)
    subgroup_by_sub = ssvep_meta.drop_duplicates("sub_id").set_index("sub_id")["subgroup"]
    df["subgroup"] = df["sub_id"].map(subgroup_by_sub).fillna("NA")

    return df[["sub_id", "session", "click", "red", "green", "group", "subgroup", "date", "folder"]]


def subjects_in_group(df: pd.DataFrame, *, group: str | None = None, subgroup: str | None = None) -> list[str]:
    """Subject IDs matching group/subgroup (mirrors ssveps'
    analysis.subjects_in_group). No session argument -- unlike ssveps' grids,
    this data has no per-session structure worth double-counting against;
    a subject's rows across every session they have are just more of their
    own clicks."""
    sub = df
    if group is not None:
        sub = sub[sub["group"] == group]
    if subgroup is not None:
        sub = sub[sub["subgroup"] == subgroup]
    return sorted(sub["sub_id"].unique())
