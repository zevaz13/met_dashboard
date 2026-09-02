"""Cross-session test-retest reliability of the behavioral centroid/shape
features (M4, PLANbeh.md).

CVD is excluded at the call site (not enforced here) -- most CVD subjects
have only 1 session (6/7 deutan, 6/8 protan), too few for a per-subgroup
reliability check; HC (21/23 with 2+ sessions) and PD (6/8) are usable.

orientation_deg is periodic (folds to [0, 180), same as
ssvep_beh_fm100/scripts/fm100_features.py's VKS_Angle) -- a naive linear ICC
would be wrong for it near the wrap point, so it's checked with
pingouin.circ_corrcc (after circ_axial folding) instead, same split
fm100_features.reliability_table already makes between its magnitude and
angle features.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg

from features import subject_session_features
from loader import subjects_in_group

_SSVEPS_SCRIPTS = str(Path(__file__).resolve().parents[2] / "ssveps" / "scripts")
if _SSVEPS_SCRIPTS in sys.path:
    sys.path.remove(_SSVEPS_SCRIPTS)
sys.path.insert(0, _SSVEPS_SCRIPTS)
import reliability as ssveps_reliability  # noqa: E402 -- imports from "analysis" internally (unique name), no further defense needed

MAGNITUDE_FEATURES = ["centroid_red", "centroid_green", "along_var", "perp_var"]
ANGLE_FEATURE = "orientation_deg"


def paired_subjects(df: pd.DataFrame, *, group: str | None = None, subgroup: str | None = None, sessions: tuple[int, int] = (1, 2)) -> list[str]:
    """Subject IDs present at both of sessions (default 1 and 2), optionally
    filtered by group/subgroup -- the same "paired at two specific sessions"
    convention ssveps/scripts/reliability.py's own paired_subjects uses,
    rather than "any 2 sessions", so results stay comparable across
    subjects with a differing number of total sessions."""
    matching = set(subjects_in_group(df, group=group, subgroup=subgroup))
    per_session = subject_session_features(df)
    per_session = per_session[per_session["sub_id"].isin(matching)]
    s1 = set(per_session.loc[per_session["session"] == sessions[0], "sub_id"])
    s2 = set(per_session.loc[per_session["session"] == sessions[1], "sub_id"])
    return sorted(s1 & s2)


def paired_sessions(df: pd.DataFrame, *, group: str | None = None, subgroup: str | None = None, sessions: tuple[int, int] = (1, 2)) -> pd.DataFrame:
    """Centroid/shape features for subjects present at both of sessions, one
    row per subject, each feature suffixed _session{n} -- the shape
    reliability_table needs, same convention as
    ssvep_beh_fm100/scripts/fm100_features.py's paired_sessions. Raises
    ValueError below 3 qualifying subjects (feature_icc's own minimum)."""
    sub_ids = paired_subjects(df, group=group, subgroup=subgroup, sessions=sessions)
    if len(sub_ids) < 3:
        raise ValueError(f"only {len(sub_ids)} subjects have data at both session {sessions[0]} and {sessions[1]} (need >= 3)")

    per_session = subject_session_features(df)
    s1 = per_session[per_session["session"] == sessions[0]].set_index("sub_id")
    s2 = per_session[per_session["session"] == sessions[1]].set_index("sub_id")
    features = [ANGLE_FEATURE] + MAGNITUDE_FEATURES
    table = pd.DataFrame({"sub_id": sub_ids})
    for feat in features:
        table[f"{feat}_session{sessions[0]}"] = s1.loc[sub_ids, feat].to_numpy()
        table[f"{feat}_session{sessions[1]}"] = s2.loc[sub_ids, feat].to_numpy()
    return table


def reliability_table(df: pd.DataFrame, *, group: str | None = None, subgroup: str | None = None, sessions: tuple[int, int] = (1, 2)) -> pd.DataFrame:
    """Cross-session reliability for centroid_red/centroid_green/along_var/
    perp_var (ICC(A,1) via ssveps.reliability.feature_icc) and
    orientation_deg (circ_corrcc, periodic). Returns one row per feature:
    feature, n, statistic ('icc' or 'circ_r'), value, p_value -- same shape
    as ssvep_beh_fm100/scripts/fm100_features.reliability_table, so
    ssvep_beh_fm100/scripts/plotting.py's plot_reliability_table is directly
    reusable on the result."""
    paired = paired_sessions(df, group=group, subgroup=subgroup, sessions=sessions)
    n = len(paired)

    rows = []
    for feat in MAGNITUDE_FEATURES:
        result = ssveps_reliability.feature_icc(paired[f"{feat}_session{sessions[0]}"].to_numpy(), paired[f"{feat}_session{sessions[1]}"].to_numpy())
        rows.append({"feature": feat, "n": n, "statistic": "icc", "value": result["icc"], "p_value": result["pval"]})

    angle1 = pg.circ_axial(np.deg2rad(paired[f"{ANGLE_FEATURE}_session{sessions[0]}"].to_numpy()), 2)
    angle2 = pg.circ_axial(np.deg2rad(paired[f"{ANGLE_FEATURE}_session{sessions[1]}"].to_numpy()), 2)
    r, p_value = pg.circ_corrcc(angle1, angle2)
    rows.append({"feature": ANGLE_FEATURE, "n": n, "statistic": "circ_r", "value": r, "p_value": p_value})

    return pd.DataFrame(rows)
