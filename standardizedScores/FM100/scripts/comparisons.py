"""Group comparisons and HC-vs-PD offset quantification for FM100 scores
(M2, PLANScores.md).

Mirrors beh/scripts/features.py's group-comparison convention (Mann-Whitney
U + effect size via pingouin, one test per feature rather than a single
omnibus statistic) rather than reusing it directly -- this module stays
self-contained (no import from ssvep_beh_fm100/ or any other combination
module), matching this project's layering convention: base modalities
(beh/, ssveps/, standardizedScores/FM100/) don't depend on the modules that
combine them.

FEATURES is the same headline set ssvep_beh_fm100/scripts/fm100_features.py
already treats as FM100's severity/type battery: TES (severity), PES_RG/
PES_BY (the red-green/blue-yellow axis split), VKS_MajRad/VKS_MinRad
(confusion-ellipse size), VKS_Angle (confusion-ellipse direction).
VKS_Angle is periodic (folds to [0, 180), an ellipse axis has no intrinsic
direction) -- pooled circularly here (mirroring
ssvep_beh_fm100/scripts/fm100_features.py's own _circ_mean_deg_axial), but
compared with plain Mann-Whitney like every other feature, the same
approximation beh/scripts/features.py's compare_shape_feature already makes
for orientation_deg (valid as long as a group's angles don't straddle the
0/180 wrap point; not a fully circular-safe test in general).
"""

import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.stats.multitest import multipletests

from loader import subjects_in_group
from scores import build_scores

FEATURES = ["TES", "PES_RG", "PES_BY", "VKS_MajRad", "VKS_MinRad", "VKS_Angle"]
ANGLE_FEATURE = "VKS_Angle"
MAJORITY_FEATURES = 4  # > half of len(FEATURES) (6) -- M3's "outlier on most/all features" exclusion rule


def _circ_mean_deg_axial(angles_deg: np.ndarray) -> float:
    """Circular mean of 180deg-periodic ('axial') angles in degrees, folded
    back into [0, 180) -- same formula as
    ssvep_beh_fm100/scripts/fm100_features.py's helper of the same purpose,
    not imported from there (see module docstring)."""
    doubled = pg.circ_axial(np.deg2rad(angles_deg), 2)
    return float(np.degrees(pg.circ_mean(doubled)) / 2 % 180)


def subject_pooled_scores(df: pd.DataFrame) -> pd.DataFrame:
    """One row per subject: each of FEATURES averaged (linear mean) across
    that subject's available sessions, except VKS_Angle (circular mean).
    Adds n_sessions. group/subgroup taken from the first row (constant per
    subject)."""
    scores = build_scores(df)
    rows = []
    for sub_id, sub in scores.groupby("sub_id"):
        row = {"sub_id": sub_id, "group": sub["group"].iloc[0], "subgroup": sub["subgroup"].iloc[0], "n_sessions": len(sub)}
        for feat in FEATURES:
            row[feat] = _circ_mean_deg_axial(sub[feat].to_numpy()) if feat == ANGLE_FEATURE else float(sub[feat].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def group_pooled_scores(df: pd.DataFrame, *, group: str | None = None, subgroup: str | None = None) -> pd.DataFrame:
    """subject_pooled_scores for every subject in a group/subgroup, one row
    per subject."""
    sub_ids = subjects_in_group(df, group=group, subgroup=subgroup)
    pooled = subject_pooled_scores(df)
    return pooled[pooled["sub_id"].isin(sub_ids)].reset_index(drop=True)


def compare_fm100_feature(
    df: pd.DataFrame,
    feature: str,
    *,
    group1: str | None = None,
    subgroup1: str | None = None,
    group2: str | None = None,
    subgroup2: str | None = None,
) -> dict:
    """Mann-Whitney U test + effect size (pingouin.mwu) on one FM100 feature
    between two groups/subgroups, on subject_pooled_scores (one independent
    observation per subject). feature is one of FEATURES. Returns {feature,
    u_val, p_value, rbc, cles, n1, n2} -- same shape as
    beh/scripts/features.py's compare_shape_feature, run once per feature
    rather than a single multivariate test for the same reason: it shows
    which feature (if any) drives a given group difference, and doesn't
    assume the smaller protan/deutan samples are Gaussian."""
    x = group_pooled_scores(df, group=group1, subgroup=subgroup1)[feature]
    y = group_pooled_scores(df, group=group2, subgroup=subgroup2)[feature]
    result = pg.mwu(x, y)
    return {
        "feature": feature,
        "u_val": float(result["U_val"].iloc[0]),
        "p_value": float(result["p_val"].iloc[0]),
        "rbc": float(result["RBC"].iloc[0]),
        "cles": float(result["CLES"].iloc[0]),
        "n1": len(x),
        "n2": len(y),
    }


def estimate_offset(profiles1: np.ndarray, profiles2: np.ndarray, *, n_boot: int = 2000, seed: int | None = 0) -> dict:
    """Quantifies "group 2's error profile looks like group 1's + a
    constant" (M2). profiles1/profiles2 are (n_subjects, N_CAPS) arrays --
    one smoothed per-subject profile each (e.g.
    plotting._group_profiles' output), not the two group means directly.

    Point estimate: c = mean(mean(profiles2) - mean(profiles1)) across all
    N_CAPS cap positions.

    Significance/CI **resamples subjects, not cap positions**: the 85 cap
    positions are correlated points along one smoothed curve per subject,
    not independent observations, so a per-position t-test would
    pseudoreplicate the same way beh/scripts/comparisons.py's
    unit='point' is explicitly documented to -- this bootstraps subjects
    (with replacement, independently within each group, n_boot times),
    recomputing c on each resample, same "resample the actual unit of
    replication" approach as ssveps/scripts/analysis.py's bootstrap_ci.
    p_value is two-sided, from the fraction of bootstrap replicates that
    cross 0, with the (1 + count) / (1 + n_boot) correction used by every
    permutation/bootstrap test elsewhere in this project (a resampled
    p-value can never legitimately be exactly 0).

    Returns {offset, ci_lower, ci_upper, p_value, r_squared}: r_squared is
    the standard regression R² for the fixed-gain-1 model "mean2 ~= mean1 +
    c" -- 1 - SS_residual/SS_total, where SS_total is mean2's *own*
    across-position variance (how much PD's profile naturally varies
    cap-to-cap) and SS_residual is what's left after predicting each
    position as mean1's value there plus the constant c. High R² means
    "knowing HC's shape and adding one number predicts PD's shape well";
    it is *not* 1 - var(diff - mean(diff)) / var(diff), which is
    mathematically always 0 (a vector's variance around its own mean is
    definitionally unchanged by subtracting that mean) and would silently
    say nothing. Same comparison ssveps/scripts/analysis.py's
    fit_gain_shape (M8) makes for a multiplicative gain instead of an
    additive offset -- this is that same R² with the gain term fixed at 1."""
    profiles1, profiles2 = np.asarray(profiles1), np.asarray(profiles2)
    mean1, mean2 = profiles1.mean(axis=0), profiles2.mean(axis=0)
    diff = mean2 - mean1
    offset = float(diff.mean())
    fitted = mean1 + offset
    ss_residual = np.sum((mean2 - fitted) ** 2)
    ss_total = np.sum((mean2 - mean2.mean()) ** 2)
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan")

    rng = np.random.default_rng(seed)
    n1, n2 = len(profiles1), len(profiles2)
    boot_offsets = np.empty(n_boot)
    for i in range(n_boot):
        boot1 = profiles1[rng.integers(0, n1, n1)].mean(axis=0)
        boot2 = profiles2[rng.integers(0, n2, n2)].mean(axis=0)
        boot_offsets[i] = (boot2 - boot1).mean()
    ci_lower, ci_upper = np.percentile(boot_offsets, [2.5, 97.5])
    p_value = min(1.0, 2 * min((1 + (boot_offsets <= 0).sum(), 1 + (boot_offsets >= 0).sum())) / (1 + n_boot))

    return {"offset": offset, "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "p_value": float(p_value), "r_squared": float(r_squared)}


def correct_multiple_comparisons(result: pd.DataFrame, *, method: str = "holm", alpha: float = 0.05) -> pd.DataFrame:
    """Adds p_corrected and significant columns to a compare_fm100_feature
    battery (or any DataFrame with a p_value column), via
    statsmodels.stats.multitest.multipletests. A self-contained copy of
    ssvepBeh/scripts/correlation.py's function of the same name and same
    behavior -- not imported from there, matching this module's own
    no-cross-import convention (see module docstring: base modalities don't
    depend on the modules that combine them).

    method='holm' (default) controls the family-wise error rate --
    conservative, the appropriate first pass before treating any individual
    feature comparison as confirmed rather than exploratory. Pass
    method='fdr_bh' (Benjamini-Hochberg) for more power at the cost of a
    weaker guarantee.

    Correction is scoped to whatever rows are passed in -- call once per
    comparison pair (FEATURES gives 6 tests per pair), not on a
    concatenation of several pairs, matching how 02_group_comparisons.ipynb
    and 04_hc_vs_pd.ipynb report results (one family per pair)."""
    result = result.copy()
    reject, p_corrected, _, _ = multipletests(result["p_value"], alpha=alpha, method=method)
    result["p_corrected"] = p_corrected
    result["significant"] = reject
    return result


def tukey_outlier_mask(values: np.ndarray) -> np.ndarray:
    """Classic Tukey boxplot rule: True where a value falls more than
    1.5*IQR beyond its own [Q1, Q3] box -- the same rule matplotlib's own
    boxplot(showfliers=True) uses to draw fliers. Exposed here so
    plotting.plot_feature_boxplot and subject_feature_outliers below share
    one definition of "outlier" (M3)."""
    values = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (values < lower) | (values > upper)


def subject_feature_outliers(df: pd.DataFrame, *, group: str | None = None, subgroup: str | None = None, features: list[str] = FEATURES) -> pd.DataFrame:
    """One row per subject in the group/subgroup, one bool column per
    feature in `features` (tukey_outlier_mask against that group's own
    distribution on subject_pooled_scores) plus n_flagged (how many
    features flag that subject) -- the input to M3's CTR-exclusion
    criterion for the offset re-run: outlier on most/all of FEATURES
    (n_flagged >= MAJORITY_FEATURES)."""
    pooled = group_pooled_scores(df, group=group, subgroup=subgroup)
    out = pooled[["sub_id"]].copy()
    for feat in features:
        out[feat] = tukey_outlier_mask(pooled[feat].to_numpy())
    out["n_flagged"] = out[features].sum(axis=1)
    return out.reset_index(drop=True)
