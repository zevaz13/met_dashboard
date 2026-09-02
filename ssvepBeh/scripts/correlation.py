"""Correlation between per-subject behavioral and EEG features -- the
convergent-validity check for whether the EEG-based test measures the same
individual differences the (more directly interpretable) behavioral test
does, across the whole clinical spectrum and within each group. This is the
"justifying our EEG tests" analysis PLANssvepvsBeh.md M1 asks for --
overlap.py's spatial tests ask whether the two agree on *where* the metamer
is per subject; this module asks whether they agree on *how severe* a
subject's deficiency is, relative to other subjects.

EEG features come from ssveps/files/subject_troughs.csv (the project's own
persisted, canonical ramp_gaussian trough table) -- read directly rather
than recomputed, the same reuse-not-rebuild convention beh/ and
standardizedScores/FM100/ already use for ssveps/files/metadata.csv.
ramp_slope_red/green and ramp_intercept are always defined (ssveps' M9:
its most reliable EEG outcomes, ICC 0.85 for ramp_slope_red; ramp_intercept
correlates r=-0.93 with M8's gain, so it stands in for gain without needing
that pipeline's CTR-template dependency) -- unlike fitted_red/green/depth,
which depend on fitted_valid and are NaN for roughly half of CVD subjects.

Behavioral features come from beh/scripts/comparisons.py-equivalent
centroid math (recomputed inline, one line) and features.py (M2's PCA
shape features: orientation_deg, along_var, perp_var).
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pingouin as pg
from statsmodels.stats.multitest import multipletests

_BEH_SCRIPTS = str(Path(__file__).resolve().parents[2] / "beh" / "scripts")
# Always move to sys.path[0], not just insert-if-absent: another module may
# have already put this same path further back in sys.path (e.g. beh's own
# test file, earlier in a combined pytest run), where a later insert (like
# overlap.py's for ssveps/scripts, below) would still search-order ahead of
# it -- "already present" is not "resolved first".
if _BEH_SCRIPTS in sys.path:
    sys.path.remove(_BEH_SCRIPTS)
sys.path.insert(0, _BEH_SCRIPTS)
# beh/scripts/features.py itself does `from loader import subjects_in_group`
# internally -- if some earlier import in this process already cached a
# *different* loader.py (ssveps/ and standardizedScores/FM100/ each have
# their own) under the bare name "loader", that internal import would
# silently resolve to the wrong file instead of erroring, unless dropped
# first. "plotting" isn't needed by features.py but is dropped too, for the
# same reason overlap.py's own imports need it dropped.
for _name in ("loader", "plotting", "features"):
    sys.modules.pop(_name, None)
import features as beh_features  # noqa: E402 -- beh/scripts/features.py

# Data lives outside this repo's git history (per-subject data is never
# committed) -- MET_DASHBOARD_DATA_DIR overrides where it's found, defaulting
# to <repo_root>/data (ssvepBeh/scripts/ -> repo root is two levels up).
DATA_DIR = os.environ.get(
    "MET_DASHBOARD_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data"),
)
SUBJECT_TROUGHS_PATH = os.path.join(DATA_DIR, "ssveps", "files", "subject_troughs.csv")

DEFAULT_BEH_FEATURES = ["beh_red", "beh_green", "orientation_deg", "along_var", "perp_var"]
DEFAULT_EEG_FEATURES = ["eeg_red", "eeg_green", "ramp_slope_red", "ramp_slope_green", "ramp_intercept"]


def subject_features_table(beh_df: pd.DataFrame, session: int, *, subject_troughs_path: str = SUBJECT_TROUGHS_PATH) -> pd.DataFrame:
    """One row per subject present in both beh_df and the EEG trough table
    at `session`: sub_id, group, subgroup, beh_red, beh_green,
    orientation_deg, along_var, perp_var (behavioral -- from that subject's
    pooled clicks across every session, since beh has no session-level
    structure that corresponds to the EEG's) and eeg_red, eeg_green,
    ramp_slope_red, ramp_slope_green, ramp_intercept (EEG, at `session`)."""
    troughs = pd.read_csv(subject_troughs_path)
    troughs = troughs[troughs["session"] == session]

    sub_ids = sorted(set(beh_df["sub_id"]) & set(troughs["sub_id"]))

    rows = []
    for sub_id in sub_ids:
        clicks = beh_df.loc[beh_df["sub_id"] == sub_id, ["red", "green"]]
        shape = beh_features.subject_shape_features(beh_df, sub_id)
        trough_row = troughs.loc[troughs["sub_id"] == sub_id].iloc[0]
        rows.append(
            {
                "sub_id": sub_id,
                "group": trough_row["group"],
                "subgroup": trough_row["subgroup"],
                "beh_red": clicks["red"].mean(),
                "beh_green": clicks["green"].mean(),
                "orientation_deg": shape["orientation_deg"],
                "along_var": shape["along_var"],
                "perp_var": shape["perp_var"],
                "eeg_red": trough_row["red"],
                "eeg_green": trough_row["green"],
                "ramp_slope_red": trough_row["ramp_slope_red"],
                "ramp_slope_green": trough_row["ramp_slope_green"],
                "ramp_intercept": trough_row["ramp_intercept"],
            }
        )
    return pd.DataFrame(rows)


def feature_correlations(
    table: pd.DataFrame,
    *,
    beh_features: list[str] = DEFAULT_BEH_FEATURES,
    eeg_features: list[str] = DEFAULT_EEG_FEATURES,
    group: str | None = None,
    subgroup: str | None = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """Correlation (pingouin.corr) between every beh_features x eeg_features
    pair in table, optionally filtered to one group/subgroup first.
    method='spearman' (default) -- robust to the small, possibly-nonlinear
    per-group samples here, same reasoning as M2's Mann-Whitney choice over
    a parametric test; pass method='pearson' for a linear-only check.

    Returns a tidy long table: beh_feature, eeg_feature, r, p_value, n,
    group, subgroup ('all' where unfiltered)."""
    sub = table
    if group is not None:
        sub = sub[sub["group"] == group]
    if subgroup is not None:
        sub = sub[sub["subgroup"] == subgroup]

    rows = []
    for beh_feat in beh_features:
        for eeg_feat in eeg_features:
            result = pg.corr(sub[beh_feat], sub[eeg_feat], method=method)
            rows.append(
                {
                    "beh_feature": beh_feat,
                    "eeg_feature": eeg_feat,
                    "r": result["r"].iloc[0],
                    "p_value": result["p_val"].iloc[0],
                    "n": int(result["n"].iloc[0]),
                    "group": group or "all",
                    "subgroup": subgroup or "all",
                }
            )
    return pd.DataFrame(rows)


def correct_multiple_comparisons(result: pd.DataFrame, *, method: str = "holm", alpha: float = 0.05) -> pd.DataFrame:
    """Adds p_corrected and significant columns to a feature_correlations
    result (or any DataFrame with a p_value column), via
    statsmodels.stats.multitest.multipletests.

    method='holm' (default) controls the family-wise error rate --
    conservative, the appropriate first-pass correction before treating any
    individual pooled/per-group correlation as confirmed rather than
    exploratory (feature_correlations' default feature sets give 25 tests
    per group/pooled call). Pass method='fdr_bh' (Benjamini-Hochberg) for
    more power at the cost of a weaker guarantee (expected false-discovery
    rate, not family-wise error).

    Correction is scoped to whatever rows are passed in -- call once per
    group/pooled result (not on a concatenation of several) if separate
    per-block corrections are wanted, which is how this project's own
    results are organized and reported (see 01_explore.ipynb)."""
    result = result.copy()
    reject, p_corrected, _, _ = multipletests(result["p_value"], alpha=alpha, method=method)
    result["p_corrected"] = p_corrected
    result["significant"] = reject
    return result
