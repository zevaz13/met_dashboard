"""Merge FM100/behavioral/SSVEP loader output into one cross-modality
per-participant roster, and summarize it by group.

Takes already-loaded tidy DataFrames (from FM100's, beh's, and ssveps'
own loaders) rather than reading files itself -- the roster is a
dashboard-layer aggregation over data each project already validates,
not a fourth independent data source.
"""

import pandas as pd


def build_roster(fm100_df: pd.DataFrame, beh_df: pd.DataFrame, ssvep_meta_df: pd.DataFrame) -> pd.DataFrame:
    """One row per sub_id, the union of every subject appearing in any of
    the three sources. group/subgroup are resolved from ssvep_meta_df
    first (the authoritative source beh's and FM100's own loaders both
    already defer to), falling back to beh_df's group (already derived
    from PartType there) for a subject with no SSVEP record, else
    'UNKNOWN'/'NA'. *_sessions counts come from each source independently
    (nunique per subject), zero for a subject missing from that source."""
    sub_ids = sorted(set(fm100_df["sub_id"]) | set(beh_df["sub_id"]) | set(ssvep_meta_df["sub_id"]))
    roster = pd.DataFrame({"sub_id": sub_ids})

    ssvep_lookup = ssvep_meta_df.drop_duplicates("sub_id").set_index("sub_id")
    beh_lookup = beh_df.drop_duplicates("sub_id").set_index("sub_id")

    roster["group"] = roster["sub_id"].map(ssvep_lookup["group"])
    roster["group"] = roster["group"].fillna(roster["sub_id"].map(beh_lookup["group"])).fillna("UNKNOWN")

    roster["subgroup"] = roster["sub_id"].map(ssvep_lookup["subgroup"])
    roster["subgroup"] = roster["subgroup"].fillna(roster["sub_id"].map(beh_lookup["subgroup"])).fillna("NA")

    for col, source in [("fm100_sessions", fm100_df), ("beh_sessions", beh_df), ("ssvep_sessions", ssvep_meta_df)]:
        counts = source.groupby("sub_id")["session"].nunique()
        roster[col] = roster["sub_id"].map(counts).fillna(0).astype(int)

    return roster[["sub_id", "group", "subgroup", "fm100_sessions", "beh_sessions", "ssvep_sessions"]]


def build_group_summary(roster_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (group, subgroup) pair -- CVD splits into
    protan/deutan/NA rather than one lumped row, matching how the rest of
    the dashboard already treats subgroup as the more informative split.
    Per modality: total_sessions (data volume, repeat visits included) and
    n_with_data (coverage -- how many of the group have any data there)."""
    rows = []
    for (group, subgroup), sub in roster_df.groupby(["group", "subgroup"]):
        row = {"group": group, "subgroup": subgroup, "n_participants": sub["sub_id"].nunique()}
        for prefix, col in [("fm100", "fm100_sessions"), ("beh", "beh_sessions"), ("ssvep", "ssvep_sessions")]:
            row[f"{prefix}_total_sessions"] = int(sub[col].sum())
            row[f"{prefix}_n_with_data"] = int((sub[col] > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)
