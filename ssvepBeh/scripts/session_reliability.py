"""Cross-session reliability of the behavioral-EEG relationship (M1,
PLANssvepvsBeh.md) -- is a spatial-overlap or individual-differences
correlation finding stable when the EEG side is measured at a different
session, or does it evaporate? A relationship that looks significant at one
session but isn't recoverable at the other isn't a usable clinical signal,
whatever its p-value looked like in isolation.

Behavioral data has no session correspondence to ssveps' own EEG sessions
(see correlation.py's subject_features_table) -- so this holds the
behavioral side fixed (each subject's pooled clicks, as everywhere else in
this project) and asks whether the EEG side's relationship to it replicates
across ssveps' session 1 and session 2.

MIN_PAIRED_SUBJECTS = 3, matching ssveps/scripts/reliability.py's own
minimum for a correlation to be even nominally interpretable (a 2-point
Spearman correlation is always +-1 or undefined -- not a number worth
reporting). This project's own real paired-subject counts, computed
directly from ssveps/files/subject_troughs.csv: pooled n=19, CTR n=13,
PD n=4, protan n=2, CVD (combined) n=2, deutan n=0. Per-subtype reliability
is not assessable with today's data -- see
docs/ssvepbeh_reliability_gaps.md for what that means and what would fix it.
"""

import pandas as pd

import correlation
import overlap

analysis = overlap.analysis

MIN_PAIRED_SUBJECTS = 3


def paired_subjects(*, group: str | None = None, subgroup: str | None = None, subject_troughs_path: str = correlation.SUBJECT_TROUGHS_PATH) -> list[str]:
    """Subject IDs with EEG trough data at both session 1 and session 2,
    optionally filtered by group/subgroup -- computed directly from
    ssveps/files/subject_troughs.csv (which already carries group/subgroup
    columns), rather than importing ssveps/scripts/reliability.py's own
    paired_subjects (avoids yet another cross-project loader.py/plotting.py
    -style name collision for what's a one-line computation here)."""
    troughs = pd.read_csv(subject_troughs_path)
    if group is not None:
        troughs = troughs[troughs["group"] == group]
    if subgroup is not None:
        troughs = troughs[troughs["subgroup"] == subgroup]
    s1 = set(troughs.loc[troughs["session"] == 1, "sub_id"])
    s2 = set(troughs.loc[troughs["session"] == 2, "sub_id"])
    return sorted(s1 & s2)


def _require_min_subjects(sub_ids: list[str], *, group: str | None, subgroup: str | None) -> None:
    if len(sub_ids) < MIN_PAIRED_SUBJECTS:
        raise ValueError(
            f"only {len(sub_ids)} subjects have EEG data at both sessions for group={group!r} subgroup={subgroup!r} "
            f"(need >= {MIN_PAIRED_SUBJECTS}) -- not enough to assess reliability, see docs/ssvepbeh_reliability_gaps.md"
        )


def session_overlap_comparison(
    beh_df: pd.DataFrame,
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    group: str | None = None,
    subgroup: str | None = None,
    sub_ids: list[str] | None = None,
    normalize: dict | None = analysis.DEFAULT_NORMALIZE,
    n_perm: int = 5000,
    seed: int | None = None,
) -> pd.DataFrame:
    """weighted_overlap_test and click_value_test run at both EEG session 1
    and session 2, on the same paired subjects and the same (session-
    pooled) behavioral data -- one row per session, so results can be
    compared directly. Raises ValueError if fewer than
    MIN_PAIRED_SUBJECTS subjects have EEG data at both sessions."""
    if sub_ids is None:
        sub_ids = paired_subjects(group=group, subgroup=subgroup)
    _require_min_subjects(sub_ids, group=group, subgroup=subgroup)

    rows = []
    for session in (1, 2):
        r_overlap = overlap.group_overlap(beh_df, runmap_df, baselines_df, metadata_df, session, sub_ids=sub_ids, normalize=normalize, n_perm=n_perm, seed=seed)
        r_click = overlap.group_click_value_test(beh_df, runmap_df, baselines_df, metadata_df, session, sub_ids=sub_ids, normalize=normalize, n_perm=n_perm, seed=seed)
        rows.append(
            {
                "session": session,
                "n": len(sub_ids),
                "weighted_overlap_obs": r_overlap["obs_stat"],
                "weighted_overlap_p": r_overlap["p_value"],
                "click_value_obs": r_click["obs_mean"],
                "click_value_p": r_click["p_value"],
            }
        )
    return pd.DataFrame(rows)


def session_correlation_comparison(
    beh_df: pd.DataFrame,
    *,
    group: str | None = None,
    subgroup: str | None = None,
    sub_ids: list[str] | None = None,
    beh_features: list[str] = correlation.DEFAULT_BEH_FEATURES,
    eeg_features: list[str] = correlation.DEFAULT_EEG_FEATURES,
    method: str = "spearman",
) -> pd.DataFrame:
    """feature_correlations run using EEG session 1 vs. session 2 trough
    data (same behavioral features both times, pooled across every beh
    session regardless) for the same paired subjects -- one row per
    (beh_feature, eeg_feature, session), for comparing r/p across sessions
    directly. Raises ValueError under the same minimum as
    session_overlap_comparison."""
    if sub_ids is None:
        sub_ids = paired_subjects(group=group, subgroup=subgroup)
    _require_min_subjects(sub_ids, group=group, subgroup=subgroup)

    rows = []
    for session in (1, 2):
        table = correlation.subject_features_table(beh_df, session=session)
        table = table[table["sub_id"].isin(sub_ids)]
        result = correlation.feature_correlations(table, beh_features=beh_features, eeg_features=eeg_features, method=method)
        result["session"] = session
        rows.append(result)
    return pd.concat(rows, ignore_index=True)
