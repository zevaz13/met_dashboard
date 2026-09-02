"""Hotelling's T-squared comparisons between groups' (red, green) point
distributions.

Uses the `hotelling` PyPI package (Hotelling 1931; pooled-covariance
two-sample test, unequal n supported) rather than a hand-rolled
implementation, per project decision. This is *not* a port of
`beh/templateCode/Hot_Tsqd_2samplesPaired.m` -- that MATLAB function is the
paired one-sample-on-differences test (requires equal-length, row-matched x
and y, e.g. the same subjects' session 1 vs session 2), which is a different
question from "do these two different, unequal-sized groups of different
subjects differ" that every comparison in PLANbeh.md M1 actually asks.
`hotelling.stats.hotelling_t2(x)` (single argument) does implement that same
paired/one-sample case if a within-subject comparison is wanted later --
just pass it `x = session1_points - session2_points`.
"""

import numpy as np
import pandas as pd
from hotelling.stats import hotelling_t2

from loader import subjects_in_group


def group_points(df: pd.DataFrame, *, group: str | None = None, subgroup: str | None = None, unit: str = "subject") -> np.ndarray:
    """(red, green) observations for one group/subgroup, shaped for
    hotelling_t2.

    unit='subject' (default): one row per subject -- that subject's own mean
    (red, green) across every click/session they have. This is the
    statistically independent unit (clicks from the same subject are not
    independent of each other) and what compare_groups uses by default.

    unit='point': every individual click from every subject/session in the
    group, pooled. Pseudoreplicated -- do not read the resulting p-value as
    if every point were an independent draw -- but it's the only way to see
    a group's actual point-cloud *shape* (spread, orientation) rather than
    just its central tendency, which matters specifically for protan/deutan:
    at n=7-8 subjects, a subject-mean point cloud is too sparse to show
    shape at all, while each of those subjects contributes 20+ clicks."""
    sub_ids = subjects_in_group(df, group=group, subgroup=subgroup)
    sub = df[df["sub_id"].isin(sub_ids)]
    if unit == "subject":
        return sub.groupby("sub_id")[["red", "green"]].mean().to_numpy()
    if unit == "point":
        return sub[["red", "green"]].to_numpy()
    raise ValueError(f"unknown unit: {unit!r} (expected 'subject' or 'point')")


def compare_groups(
    df: pd.DataFrame,
    *,
    group1: str | None = None,
    subgroup1: str | None = None,
    group2: str | None = None,
    subgroup2: str | None = None,
    unit: str = "subject",
) -> dict:
    """Two-sample Hotelling T^2 between two groups'/subgroups' (red, green)
    distributions (group_points, then hotelling.stats.hotelling_t2).
    unit='subject' (independent, default) or 'point' (pseudoreplicated, for
    seeing point-cloud shape -- see group_points).

    Returns {t2_stat, f_stat, p_value, pooled_cov, n1, n2, unit}."""
    x = group_points(df, group=group1, subgroup=subgroup1, unit=unit)
    y = group_points(df, group=group2, subgroup=subgroup2, unit=unit)
    t2_stat, f_stat, p_value, pooled_cov = hotelling_t2(x, y)
    return {
        "t2_stat": t2_stat,
        "f_stat": f_stat,
        "p_value": p_value,
        "pooled_cov": pooled_cov,
        "n1": len(x),
        "n2": len(y),
        "unit": unit,
    }
