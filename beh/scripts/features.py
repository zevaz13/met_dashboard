"""Per-subject shape features from PCA on each subject's pooled (red, green)
clicks -- orientation, along-line spread, and perpendicular tightness, as a
complement to comparisons.py's group-mean Hotelling T^2 (M2, PLANbeh.md).

M1's subject-mean Hotelling T^2 already separated every group pair on
centroid location alone, including protan vs deutan. These features instead
describe each subject's line *shape*: where it points (orientation), how far
clicks spread along it (along_var), and how tightly they cluster off of it
(perp_var) -- protan and deutan clouds were noted as differing in shape as
much as position.

orientation_deg is folded into [0, 180) -- a fitted line has no direction, so
0deg and 180deg are the same orientation. This means group comparisons on
orientation_deg (compare_shape_feature) assume a group's angles don't
straddle the 0/180 wrap point; true for this dataset's isoluminance lines
(consistently positive slope, well clear of the wrap), but not a fully
circular-safe statistic in general.
"""

import numpy as np
import pandas as pd
import pingouin as pg

from loader import subjects_in_group


def _points_pca(points: np.ndarray) -> dict:
    """PCA on a raw (N, 2) red/green points array -- the core eigendecomposition
    shared by _subject_pca (one subject's own points) and group_outliers (a
    whole group/subgroup's pooled points, M4)."""
    if len(points) < 2:
        raise ValueError(f"fewer than 2 points ({len(points)}), can't fit a PCA line")
    mean = points.mean(axis=0)
    centered = points - mean
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered, rowvar=False))  # ascending
    return {"mean": mean, "centered": centered, "pc1": eigvecs[:, 1], "along_var": eigvals[1], "perp_var": eigvals[0], "n": len(points)}


def _subject_pca(df: pd.DataFrame, sub_id: str) -> dict:
    points = df.loc[df["sub_id"] == sub_id, ["red", "green"]].to_numpy()
    try:
        return _points_pca(points)
    except ValueError as e:
        raise ValueError(f"{sub_id} has {e}") from None


def _orientation_deg(fit: dict) -> float:
    """Angle of a _points_pca/_subject_pca fit's first principal component,
    folded into [0, 180) -- a fitted line has no direction."""
    return float(np.degrees(np.arctan2(fit["pc1"][1], fit["pc1"][0])) % 180)


def subject_shape_features(df: pd.DataFrame, sub_id: str) -> dict:
    """PCA on one subject's pooled (red, green) clicks (every session).

    Returns {orientation_deg, along_var, perp_var, n}:
    - orientation_deg: angle of the first principal component, folded into
      [0, 180).
    - along_var: variance along that first component (spread along the
      fitted line).
    - perp_var: variance along the second component (scatter off the line --
      how tight/consistent the match is).

    Needs at least 2 points; raises ValueError otherwise."""
    fit = _subject_pca(df, sub_id)
    return {"orientation_deg": _orientation_deg(fit), "along_var": float(fit["along_var"]), "perp_var": float(fit["perp_var"]), "n": fit["n"]}


def subject_pca_line(df: pd.DataFrame, sub_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Endpoints of the fitted PCA line, spanning the subject's actual data
    extent along the first principal component -- for overlaying on a cloud
    plot (plotting.plot_subject_cloud's show_fit)."""
    fit = _subject_pca(df, sub_id)
    proj = fit["centered"] @ fit["pc1"]
    return fit["mean"] + proj.min() * fit["pc1"], fit["mean"] + proj.max() * fit["pc1"]


def subject_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """subject_shape_features, plus centroid, computed separately per
    (sub_id, session) rather than pooled across a subject's sessions --
    what a test-retest reliability check needs (M4), unlike
    subject_shape_features/group_features which deliberately pool every
    session together for M2's shape-comparison use.

    One row per (sub_id, session) with >= 2 clicks: sub_id, session,
    centroid_red, centroid_green, orientation_deg, along_var, perp_var, n."""
    rows = []
    for (sub_id, session), sub in df.groupby(["sub_id", "session"]):
        points = sub[["red", "green"]].to_numpy()
        if len(points) < 2:
            continue
        fit = _points_pca(points)
        rows.append(
            {
                "sub_id": sub_id,
                "session": session,
                "centroid_red": float(fit["mean"][0]),
                "centroid_green": float(fit["mean"][1]),
                "orientation_deg": _orientation_deg(fit),
                "along_var": float(fit["along_var"]),
                "perp_var": float(fit["perp_var"]),
                "n": fit["n"],
            }
        )
    return pd.DataFrame(rows)


def within_session_scatter(df: pd.DataFrame, sub_id: str) -> float:
    """One subject's average within-session click consistency (M4): for each
    of their sessions, the RMS Euclidean distance of that session's clicks to
    that session's own centroid; averaged across sessions (each session
    weighted equally, same "average per subject/session first" convention
    plot_group_fm100/plot_troughs_boxplot use elsewhere in this project) so a
    subject with more sessions doesn't dominate a later group comparison.

    Deliberately per-session, not pooled across sessions like
    subject_shape_features -- the question this answers is "how much does
    this subject's hand wander around wherever they're aiming *within one
    sitting*", not cross-session drift in where they aim, which is a
    different thing (and the reason PD's motor symptoms might plausibly
    inflate the former without moving the latter)."""
    sub = df[df["sub_id"] == sub_id]
    session_rms = []
    for _, session_points in sub.groupby("session")[["red", "green"]]:
        points = session_points.to_numpy()
        centroid = points.mean(axis=0)
        session_rms.append(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))
    return float(np.mean(session_rms))


def outlier_mask(pca: dict, points: np.ndarray, *, n_std: float = 2.0) -> np.ndarray:
    """Which rows of points fall outside the ellipse at n_std standard
    deviations along each of pca's principal axes (a _points_pca/_subject_pca
    fit) -- True = outlier. pca and points need not come from the same
    subject/group (M4's group-level check applies one shared group ellipse to
    each individual subject's own points)."""
    centered = points - pca["mean"]
    along = (centered @ pca["pc1"]) / (n_std * np.sqrt(pca["along_var"]))
    pc2 = np.array([-pca["pc1"][1], pca["pc1"][0]])  # perpendicular to pc1, in the (ascending eigh) plane
    perp = (centered @ pc2) / (n_std * np.sqrt(pca["perp_var"]))
    return (along**2 + perp**2) > 1.0


def subject_outliers(df: pd.DataFrame, sub_id: str, *, n_std: float = 2.0) -> dict:
    """One subject's own points classified against their own fitted ellipse
    (M4, per-participant outlier check). Returns {pca, points, outlier_mask}."""
    points = df.loc[df["sub_id"] == sub_id, ["red", "green"]].to_numpy()
    pca = _points_pca(points)
    return {"pca": pca, "points": points, "outlier_mask": outlier_mask(pca, points, n_std=n_std)}


def group_outliers(df: pd.DataFrame, *, group: str | None = None, subgroup: str | None = None, n_std: float = 2.0) -> dict:
    """One ellipse fit to a whole group/subgroup's pooled clicks, then
    applied back to each individual subject's own points (M4's group-level
    outlier check -- "does this subject's data look like an outlier relative
    to the group as a whole", not relative to their own cloud).

    Returns {pca, table}: pca is the group-level fit (for drawing the shared
    ellipse); table has one row per click (sub_id, red, green, is_outlier)
    across every subject in the group, so results can be grouped by sub_id
    to see how many of a given subject's points were flagged."""
    sub_ids = subjects_in_group(df, group=group, subgroup=subgroup)
    sub = df[df["sub_id"].isin(sub_ids)]
    points = sub[["red", "green"]].to_numpy()
    pca = _points_pca(points)
    table = sub[["sub_id", "red", "green"]].copy()
    table["is_outlier"] = outlier_mask(pca, points, n_std=n_std)
    return {"pca": pca, "table": table.reset_index(drop=True)}


def group_features(df: pd.DataFrame, *, group: str | None = None, subgroup: str | None = None) -> pd.DataFrame:
    """subject_shape_features for every subject in a group/subgroup, one row
    per subject, indexed by sub_id."""
    sub_ids = subjects_in_group(df, group=group, subgroup=subgroup)
    rows = {sub_id: subject_shape_features(df, sub_id) for sub_id in sub_ids}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("sub_id")


def compare_shape_feature(
    df: pd.DataFrame,
    feature: str,
    *,
    group1: str | None = None,
    subgroup1: str | None = None,
    group2: str | None = None,
    subgroup2: str | None = None,
) -> dict:
    """Mann-Whitney U test + effect size (pingouin.mwu) on one shape feature
    between two groups/subgroups. feature is one of subject_shape_features's
    keys ('orientation_deg', 'along_var', 'perp_var').

    Returns {feature, u_val, p_value, rbc, cles, n1, n2}. rbc (rank-biserial
    correlation) and cles (common language effect size) come straight from
    pingouin -- unlike Hotelling T^2's single omnibus stat, this is
    per-feature, so run it once per feature to see which shape property (if
    any) drives a given group difference."""
    x = group_features(df, group=group1, subgroup=subgroup1)[feature]
    y = group_features(df, group=group2, subgroup=subgroup2)[feature]
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
