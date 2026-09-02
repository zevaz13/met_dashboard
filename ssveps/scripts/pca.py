"""Principal component analysis of the 100-cell response grid (M10).

Treats each subject's mean grid as one 100-dimensional observation instead of
either collapsing it to a single summary number (loses information) or
running 100 independent cell-wise tests (pays a heavy multiple-comparisons
price on cells that are highly correlated with their neighbours anyway --
see docs/ssvep_analyses.md proposal 7). PCA finds the handful of axes that
actually vary across subjects; docs/methods.md explains why component count
is decided by permutation rather than a fixed cutoff or covariance shrinkage.
"""

import numpy as np
import pandas as pd

from analysis import DEFAULT_NORMALIZE, mean_grid, subjects_in_group


def pixel_matrix(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    *,
    normalize: dict | None = DEFAULT_NORMALIZE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Every subject's mean grid at this session, flattened and stacked into
    an (n_subjects, 100) matrix -- the input fit_pca/permutation_component_
    count operate on. Returns (metadata rows in matrix row order, matrix);
    row i of the metadata frame describes row i of the matrix."""
    sub_ids = subjects_in_group(metadata_df, session)
    meta = metadata_df[metadata_df["session"] == session].set_index("sub_id").loc[sub_ids].reset_index()
    X = np.stack([mean_grid(runmap_df, baselines_df, sub_id, session, normalize=normalize).ravel() for sub_id in sub_ids])
    return meta, X


def fit_pca(X: np.ndarray) -> dict:
    """PCA via SVD on mean-centered X (n_subjects, n_features). No covariance
    shrinkage/regularization -- see permutation_component_count for how this
    project decides how many of the resulting components are worth trusting
    instead.

    Returns {mean, components, scores, explained_variance,
    explained_variance_ratio}. components[k] is the k-th principal axis (a
    unit vector of length n_features, e.g. reshape to (10, 10) to view as a
    grid); scores[:, k] is every subject's projection onto it. Component sign
    is arbitrary (an SVD convention, not a data property) -- don't read
    meaning into whether a component is "positive" or "negative" without
    checking which way its loading grid points."""
    mean = X.mean(axis=0)
    centered = X - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n = X.shape[0]
    explained_variance = S**2 / (n - 1)
    return {
        "mean": mean,
        "components": Vt,
        "scores": centered @ Vt.T,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance / explained_variance.sum(),
    }


def permutation_component_count(X: np.ndarray, *, n_perm: int = 2000, alpha: float = 0.05, seed: int | None = 0) -> dict:
    """How many principal components carry more structure than chance (a
    numpy-native version of Horn's 1965 parallel analysis -- the standard way
    to decide component count without an arbitrary fixed cutoff or a
    covariance-shrinkage method this project doesn't otherwise use).

    Permute each column (grid cell) independently across subjects -- this
    destroys the cross-cell correlation PCA looks for while preserving each
    cell's own marginal distribution -- and redo PCA on the permuted data.
    Repeated n_perm times, this builds a null distribution of the
    explained-variance-ratio spectrum you'd see from 100 cells with no real
    shared structure at all. A real component's explained-variance ratio
    should beat that null; once one doesn't, every later one is judged
    against a null it's already indistinguishable from, so n_components_real
    is the length of the leading run that clears its own rank's (1-alpha)
    null percentile, not a simple per-rank count (which could overstate the
    result if a later rank clears by chance after an earlier one failed).

    Returns {observed_ratio, null_ratio_threshold, n_components_real} -- the
    first two are same-length arrays (one entry per component) suitable for
    plotting directly against each other as a scree plot with its noise
    floor."""
    n_features = X.shape[1]
    observed = fit_pca(X)["explained_variance_ratio"]

    rng = np.random.default_rng(seed)
    null_ratios = np.empty((n_perm, len(observed)))
    for i in range(n_perm):
        permuted = np.column_stack([rng.permutation(X[:, j]) for j in range(n_features)])
        null_ratios[i] = fit_pca(permuted)["explained_variance_ratio"]
    threshold = np.quantile(null_ratios, 1 - alpha, axis=0)

    n_real = 0
    for obs, thr in zip(observed, threshold):
        if obs <= thr:
            break
        n_real += 1

    return {"observed_ratio": observed, "null_ratio_threshold": threshold, "n_components_real": n_real}
