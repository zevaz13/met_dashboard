"""Cluster-based permutation testing between two groups' mean grids.

Replicates ssveps/templateCode/permTestingcomparisons/*.m (cluster-based
permutation testing in the style of Maris & Oostenveld, adapted from EEG
time-frequency maps to this project's 10x10 red/green grid) -- generalized to
any two groups via analysis.subjects_in_group/mean_grid, rather than the
template's copy-pasted scripts with hardcoded per-comparison subject lists.
Three functions replicate the template's three levels of sophistication (kept
separate, matching the template's own progression, rather than collapsed into
one function):

  - permutation_test_size        -- group_permTesting_01JULY25.m
  - permutation_test_weighted    -- groupPermTesting_clustSize_custWeight.m
  - permutation_test_directional -- group_permTesting_positive_negative_clusters.m

n1/n2 control how many subjects each permutation draws from each group before
shuffling labels. **They default to the full group sizes, so no subject is
discarded** and each permutation is a plain relabelling of everyone -- the
standard permutation test, and the same set of subjects the observed
difference map is computed from.

The MATLAB templates hardcoded a subsample per comparison (e.g. 30 of 33 HC
vs 5 of 7 CVD in group_permTesting_01JULY25.m). n1/n2 are kept as parameters
so that behaviour can be reproduced, but subsampling is not the default:
discarding subjects widens the null distribution and shrinks every z-score,
making the test more conservative for no gain. Measured on PD vs CTR at
session 1 (6 vs 21 subjects), drawing a balanced 6 vs 6 dropped max |z| from
1.81 to 1.43. Note also that any n1/n2 below the full group size makes the
permuted statistic sample-size-mismatched against the observed one.

One correctness fix vs. the template (confirmed): permutation_test_directional's
negative-cluster null uses each permutation's most extreme (min, i.e. most
negative) cluster weight, not max -- the template's max() on negative sums
picked the *weakest* negative cluster each time, biasing the negative null
toward small magnitudes and making the negative-cluster threshold too lenient.
"""

import numpy as np
from scipy.ndimage import label
from scipy.stats import norm

from analysis import DEFAULT_NORMALIZE, mean_grid, subjects_in_group

CONNECTIVITY_8 = np.ones((3, 3))


def _group_grid_stack(
    runmap_df, baselines_df, metadata_df, session, *, group: str | None, subgroup: str | None, normalize: dict | None
) -> np.ndarray:
    """Stack of shape (n_subjects, n_red, n_green): each subject's mean_grid."""
    sub_ids = subjects_in_group(metadata_df, session, group=group, subgroup=subgroup)
    return np.stack([mean_grid(runmap_df, baselines_df, sub_id, session, normalize=normalize) for sub_id in sub_ids])


def _null_diff_maps(grids1: np.ndarray, grids2: np.ndarray, n1: int, n2: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """n_perm permuted (fake-group-A mean - fake-group-B mean) maps: each
    permutation independently draws n1/n2 subjects (without replacement) from
    the FULL pool of grids1/grids2, pools them, then randomly reassigns n1/n2
    fake group labels across the pooled subsample.

    With the default n1/n2 (the full group sizes) the draw keeps everyone and
    the step reduces to a plain shuffle of the real group labels."""
    diffs = np.empty((n_perm, *grids1.shape[1:]))
    condlabels = np.arange(n1 + n2) >= n1  # False = fake group A (n1), True = fake group B (n2)
    for i in range(n_perm):
        sample1 = grids1[rng.choice(len(grids1), n1, replace=False)]
        sample2 = grids2[rng.choice(len(grids2), n2, replace=False)]
        pooled = np.concatenate([sample1, sample2])
        fake = rng.permutation(condlabels)
        diffs[i] = pooled[~fake].mean(axis=0) - pooled[fake].mean(axis=0)
    return diffs


def _clusters(zmap: np.ndarray, sig_thresh: float) -> list[tuple[np.ndarray, ...]]:
    """Suprathreshold (|z| >= sig_thresh) connected clusters (8-connectivity,
    matching MATLAB bwconncomp's 2D default), as a list of index tuples."""
    mask = np.abs(zmap) >= sig_thresh
    labeled, n = label(mask, structure=CONNECTIVITY_8)
    return [np.nonzero(labeled == i) for i in range(1, n + 1)]


def _max_size_and_weight(zmap: np.ndarray, sig_thresh: float) -> tuple[int, float]:
    """Max cluster size and max cluster weight (sum of |z|) in zmap, pooling
    both signs; (0, 0.0) if no suprathreshold clusters."""
    clusters = _clusters(zmap, sig_thresh)
    if not clusters:
        return 0, 0.0
    sizes = [idx[0].size for idx in clusters]
    weights = [np.abs(zmap[idx]).sum() for idx in clusters]
    return max(sizes), max(weights)


def _max_directional_stats(zmap: np.ndarray, sig_thresh: float) -> dict:
    """Max cluster size/weight per sign: positive clusters use their largest
    (most positive) values; negative clusters use their most extreme (min,
    most negative) values -- symmetric tails of one max-statistic null."""
    pos_sizes, pos_weights, neg_sizes, neg_weights = [], [], [], []
    for idx in _clusters(zmap, sig_thresh):
        w = zmap[idx].sum()
        if w > 0:
            pos_sizes.append(idx[0].size)
            pos_weights.append(w)
        elif w < 0:
            neg_sizes.append(idx[0].size)
            neg_weights.append(w)
    return {
        "pos_size": max(pos_sizes, default=0),
        "pos_weight": max(pos_weights, default=0.0),
        "neg_size": max(neg_sizes, default=0),
        "neg_weight": min(neg_weights, default=0.0),
    }


def _setup(
    runmap_df, baselines_df, metadata_df, session, *,
    group1, subgroup1, group2, subgroup2, normalize, n1, n2, n_perm, pval, seed,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """Shared setup for every permutation_test_* function: the observed
    z-scored difference map, the null distribution's own z-scored permuted
    maps (for building max-cluster-statistic null distributions), the
    significance threshold, and the resolved n1/n2."""
    grids1 = _group_grid_stack(runmap_df, baselines_df, metadata_df, session, group=group1, subgroup=subgroup1, normalize=normalize)
    grids2 = _group_grid_stack(runmap_df, baselines_df, metadata_df, session, group=group2, subgroup=subgroup2, normalize=normalize)

    n1 = len(grids1) if n1 is None else n1
    n2 = len(grids2) if n2 is None else n2

    rng = np.random.default_rng(seed)
    obsdiff = grids1.mean(axis=0) - grids2.mean(axis=0)
    null_diffs = _null_diff_maps(grids1, grids2, n1, n2, n_perm, rng)
    null_mean, null_std = null_diffs.mean(axis=0), null_diffs.std(axis=0)

    zdiff = (obsdiff - null_mean) / null_std
    null_zmaps = (null_diffs - null_mean) / null_std
    sig_thresh = norm.ppf(1 - pval / 2)
    return zdiff, null_zmaps, sig_thresh, n1, n2


def permutation_test_size(
    runmap_df, baselines_df, metadata_df, session, *,
    group1: str | None = None, subgroup1: str | None = None,
    group2: str | None = None, subgroup2: str | None = None,
    normalize: dict | None = DEFAULT_NORMALIZE,
    n1: int | None = None, n2: int | None = None,  # subjects drawn per permutation; default = full group sizes
    n_perm: int = 1000, pval: float = 0.05, seed: int | None = None,
) -> dict:
    """Cluster-size-corrected two-tailed permutation test between two groups'
    mean grids (replicates group_permTesting_01JULY25.m). Suprathreshold
    clusters (both signs pooled) in the observed map are kept only if their
    size exceeds the 100*(1-pval) percentile of every permutation's own max
    cluster size."""
    zdiff, null_zmaps, sig_thresh, n1, n2 = _setup(
        runmap_df, baselines_df, metadata_df, session,
        group1=group1, subgroup1=subgroup1, group2=group2, subgroup2=subgroup2,
        normalize=normalize, n1=n1, n2=n2, n_perm=n_perm, pval=pval, seed=seed,
    )

    null_sizes = np.array([_max_size_and_weight(null_zmaps[i], sig_thresh)[0] for i in range(n_perm)])
    size_thresh = np.percentile(null_sizes, 100 * (1 - pval))

    zthresh_uncorrected = np.where(np.abs(zdiff) >= sig_thresh, zdiff, 0.0)
    zthresh_corrected = zthresh_uncorrected.copy()
    for idx in _clusters(zdiff, sig_thresh):
        if idx[0].size <= size_thresh:
            zthresh_corrected[idx] = 0.0

    return {
        "zdiff": zdiff,
        "zthresh_uncorrected": zthresh_uncorrected,
        "zthresh_corrected": zthresh_corrected,
        "size_thresh": size_thresh,
        "null_sizes": null_sizes,
        "sig_thresh": sig_thresh,
        "n1": n1,
        "n2": n2,
    }


def permutation_test_weighted(
    runmap_df, baselines_df, metadata_df, session, *,
    group1: str | None = None, subgroup1: str | None = None,
    group2: str | None = None, subgroup2: str | None = None,
    normalize: dict | None = DEFAULT_NORMALIZE,
    n1: int | None = None, n2: int | None = None,  # subjects drawn per permutation; default = full group sizes
    n_perm: int = 1000, pval: float = 0.05, seed: int | None = None,
) -> dict:
    """Adds cluster-weight (sum of |z| in the cluster) correction alongside
    cluster size (replicates groupPermTesting_clustSize_custWeight.m), plus a
    per-cluster p-value: the fraction of permutations whose own max cluster
    weight exceeds that cluster's weight."""
    zdiff, null_zmaps, sig_thresh, n1, n2 = _setup(
        runmap_df, baselines_df, metadata_df, session,
        group1=group1, subgroup1=subgroup1, group2=group2, subgroup2=subgroup2,
        normalize=normalize, n1=n1, n2=n2, n_perm=n_perm, pval=pval, seed=seed,
    )

    null_sizes = np.empty(n_perm)
    null_weights = np.empty(n_perm)
    for i in range(n_perm):
        null_sizes[i], null_weights[i] = _max_size_and_weight(null_zmaps[i], sig_thresh)
    size_thresh = np.percentile(null_sizes, 100 * (1 - pval))
    weight_thresh = np.percentile(null_weights, 100 * (1 - pval))

    zthresh_uncorrected = np.where(np.abs(zdiff) >= sig_thresh, zdiff, 0.0)
    zthresh_size_corrected = zthresh_uncorrected.copy()
    zthresh_weight_corrected = zthresh_uncorrected.copy()
    cluster_results = []
    for idx in _clusters(zdiff, sig_thresh):
        size = idx[0].size
        weight = np.abs(zdiff[idx]).sum()
        if size <= size_thresh:
            zthresh_size_corrected[idx] = 0.0
        if weight <= weight_thresh:
            zthresh_weight_corrected[idx] = 0.0
        cluster_results.append({"size": size, "weight": weight, "pvalue": (null_weights > weight).mean()})

    return {
        "zdiff": zdiff,
        "zthresh_uncorrected": zthresh_uncorrected,
        "zthresh_size_corrected": zthresh_size_corrected,
        "zthresh_weight_corrected": zthresh_weight_corrected,
        "size_thresh": size_thresh,
        "weight_thresh": weight_thresh,
        "null_sizes": null_sizes,
        "null_weights": null_weights,
        "cluster_results": cluster_results,
        "sig_thresh": sig_thresh,
        "n1": n1,
        "n2": n2,
    }


def permutation_test_directional(
    runmap_df, baselines_df, metadata_df, session, *,
    group1: str | None = None, subgroup1: str | None = None,
    group2: str | None = None, subgroup2: str | None = None,
    normalize: dict | None = DEFAULT_NORMALIZE,
    n1: int | None = None, n2: int | None = None,  # subjects drawn per permutation; default = full group sizes
    n_perm: int = 1000, pval: float = 0.05, seed: int | None = None,
) -> dict:
    """Separates positive- and negative-going clusters, each corrected by both
    size and weight against its own one-tailed null distribution (replicates
    group_permTesting_positive_negative_clusters.m, with the negative-cluster
    min/max fix described in this module's docstring), plus a per-cluster
    p-value extending the same logic group_permTesting_positive_negative_clusters.m
    didn't itself compute for the directional case (group_permTesting_
    clustSize_custWeight.m/protansVsDeutans_28JUL25.m do this for the pooled,
    non-directional case)."""
    zdiff, null_zmaps, sig_thresh, n1, n2 = _setup(
        runmap_df, baselines_df, metadata_df, session,
        group1=group1, subgroup1=subgroup1, group2=group2, subgroup2=subgroup2,
        normalize=normalize, n1=n1, n2=n2, n_perm=n_perm, pval=pval, seed=seed,
    )

    null_stats = [_max_directional_stats(null_zmaps[i], sig_thresh) for i in range(n_perm)]
    null_pos_sizes = np.array([s["pos_size"] for s in null_stats])
    null_pos_weights = np.array([s["pos_weight"] for s in null_stats])
    null_neg_sizes = np.array([s["neg_size"] for s in null_stats])
    null_neg_weights = np.array([s["neg_weight"] for s in null_stats])

    pos_size_thresh = np.percentile(null_pos_sizes, 100 * (1 - pval / 2))
    pos_weight_thresh = np.percentile(null_pos_weights, 100 * (1 - pval / 2))
    neg_size_thresh = np.percentile(null_neg_sizes, 100 * (1 - pval / 2))
    neg_weight_thresh = np.percentile(null_neg_weights, pval / 2 * 100)  # low percentile: most extreme negative

    zthresh_uncorrected = np.where(np.abs(zdiff) >= sig_thresh, zdiff, 0.0)
    zthresh_size_pos = zthresh_uncorrected.copy()
    zthresh_size_neg = zthresh_uncorrected.copy()
    zthresh_weight_pos = zthresh_uncorrected.copy()
    zthresh_weight_neg = zthresh_uncorrected.copy()
    cluster_results = []

    for idx in _clusters(zdiff, sig_thresh):
        size = idx[0].size
        weight = zdiff[idx].sum()
        if weight > 0:
            zthresh_size_neg[idx] = 0.0
            zthresh_weight_neg[idx] = 0.0
            if size <= pos_size_thresh:
                zthresh_size_pos[idx] = 0.0
            if weight <= pos_weight_thresh:
                zthresh_weight_pos[idx] = 0.0
            pvalue = (null_pos_weights > weight).mean()
            sign = "pos"
        else:
            zthresh_size_pos[idx] = 0.0
            zthresh_weight_pos[idx] = 0.0
            if size <= neg_size_thresh:
                zthresh_size_neg[idx] = 0.0
            if weight >= neg_weight_thresh:
                zthresh_weight_neg[idx] = 0.0
            pvalue = (null_neg_weights < weight).mean()
            sign = "neg"
        cluster_results.append({"sign": sign, "size": size, "weight": weight, "pvalue": pvalue})

    return {
        "zdiff": zdiff,
        "zthresh_uncorrected": zthresh_uncorrected,
        "zthresh_size_pos": zthresh_size_pos,
        "zthresh_size_neg": zthresh_size_neg,
        "zthresh_weight_pos": zthresh_weight_pos,
        "zthresh_weight_neg": zthresh_weight_neg,
        "pos_size_thresh": pos_size_thresh,
        "neg_size_thresh": neg_size_thresh,
        "pos_weight_thresh": pos_weight_thresh,
        "neg_weight_thresh": neg_weight_thresh,
        "null_pos_sizes": null_pos_sizes,
        "null_neg_sizes": null_neg_sizes,
        "null_pos_weights": null_pos_weights,
        "null_neg_weights": null_neg_weights,
        "cluster_results": cluster_results,
        "sig_thresh": sig_thresh,
        "n1": n1,
        "n2": n2,
    }
