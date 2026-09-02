"""Within-subject vs. between-subject variance decomposition via MixedLM (M7).

Replaces the point-estimate within/between SD split in
`docs/ssvep_analyses.md` proposal 3 with a proper random-intercept mixed
model per group (subject as the random effect, run as the residual), plus a
subject-level bootstrap CI on each variance component -- see
`docs/methods.md` for why a separate model per group was chosen over one
pooled model with group-specific variance components.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis import DEFAULT_NORMALIZE, run_mean_values, subjects_in_group


def group_run_values(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    *,
    group: str | None = None,
    subgroup: str | None = None,
    normalize: dict | None = DEFAULT_NORMALIZE,
) -> dict[str, np.ndarray]:
    """sub_id -> run_mean_values for every subject matching group/subgroup at
    this session -- the input variance_components/within_subject_cv expect."""
    sub_ids = subjects_in_group(metadata_df, session, group=group, subgroup=subgroup)
    return {sub_id: run_mean_values(runmap_df, baselines_df, sub_id, session, normalize=normalize) for sub_id in sub_ids}


def _long_frame(values_by_subject: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({"subject": sub, "value": v} for sub, vals in values_by_subject.items() for v in vals)


def _fit_components(values_by_subject: dict[str, np.ndarray]) -> tuple[float, float]:
    """(within_sd, between_sd) from a random-intercept-only MixedLM
    (`value ~ 1`, grouped by subject) fit to values_by_subject: the residual
    SD is within-subject (run-to-run) variability, the random-intercept SD is
    between-subject variability."""
    df = _long_frame(values_by_subject)
    result = smf.mixedlm("value ~ 1", df, groups=df["subject"]).fit(reml=True)
    return float(np.sqrt(result.scale)), float(np.sqrt(result.cov_re.iloc[0, 0]))


def variance_components(values_by_subject: dict[str, np.ndarray], *, n_boot: int = 2000, seed: int | None = 0) -> dict:
    """Within-subject and between-subject SD for one group, each with a
    subject-level bootstrap 95% CI (resample subjects with replacement,
    refit, repeat -- same percentile-bootstrap approach as
    analysis.bootstrap_ci, but returning two correlated statistics from one
    shared set of resamples rather than one, so it isn't built on top of that
    generic helper).

    Each bootstrap resample is relabeled with fresh synthetic subject ids
    (0..n-1) rather than reusing the original sub_id -- so drawing the same
    real subject twice in one resample is correctly treated as two
    independent subjects for that replicate, not as one subject with double
    the runs (which would silently inflate its weight and distort the
    decomposition).

    A resample whose MixedLM fails to converge is dropped rather than
    counted; check n_boot_used against n_boot if the CI looks unexpectedly
    wide for a small group."""
    within_sd, between_sd = _fit_components(values_by_subject)

    sub_ids = list(values_by_subject)
    rng = np.random.default_rng(seed)
    within_reps, between_reps = [], []
    for _ in range(n_boot):
        drawn = rng.choice(sub_ids, size=len(sub_ids), replace=True)
        resampled = {i: values_by_subject[sub_id] for i, sub_id in enumerate(drawn)}
        try:
            w, b = _fit_components(resampled)
        except Exception:
            continue
        within_reps.append(w)
        between_reps.append(b)

    def ci(reps: list[float]) -> tuple[float, float]:
        return float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))

    return {
        "within_sd": within_sd,
        "between_sd": between_sd,
        "within_ci": ci(within_reps),
        "between_ci": ci(between_reps),
        "n_subjects": len(sub_ids),
        "n_boot_used": len(within_reps),
    }


def within_subject_cv(values_by_subject: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-subject within-subject coefficient of variation (SD/|mean| of that
    subject's own run values, ddof=1) -- response magnitude scales with the
    SD of the noise around it, so a raw within-subject SD comparison across
    groups conflates response size with actual consistency; CV corrects for
    that. Used to check whether within-subject noise is genuinely flat across
    groups once this scaling is accounted for (proposal 3's finding)."""
    return {
        sub_id: float(np.std(vals, ddof=1) / abs(np.mean(vals)))
        for sub_id, vals in values_by_subject.items()
        if len(vals) > 1
    }
