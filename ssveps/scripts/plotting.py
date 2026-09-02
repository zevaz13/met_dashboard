"""Heatmap plots for SSVEP grid data, raw or baseline-normalized.

Axes: red is always x, green is always y. Color follows the dataviz skill: raw
values are a magnitude job (sequential, one hue) while normalized values are
signed/polarity (diverging, centered on zero) -- both ramps are the skill's
validated default palette. Every function accepts optional clim=(vmin, vmax)
and cmap overrides; multi-panel functions use one shared clim/cmap across all
their panels by default.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, LinearSegmentedColormap

from analysis import (
    DEFAULT_NORMALIZE,
    flatten_runs,
    interpolate_grid,
    load_grid_axes,
    mean_grid,
    mean_grid_across_subjects,
    normalized_grid,
    pooled_baseline_values,
    pooled_pixels,
    raw_grid,
    subjects_in_group,
)
from reliability import session_pair_values

# Every multi-panel figure in this module wraps to at most this many panels per
# row, via _multi_panel_figure. Change it here and every such figure follows;
# individual calls can override it with _multi_panel_figure(..., max_cols=).
MAX_PANEL_COLS = 5

SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("sequential_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIVERGING_BLUE_RED = LinearSegmentedColormap.from_list("diverging_blue_red", ["#2a78d6", "#f0efec", "#e34948"])
DISTRIBUTION_COLOR = "#256abf"

METHOD_LABELS = {"percent": "% change from baseline", "db": "dB change from baseline", "zscore": "baseline z-score"}


def _default_cmap(diverging: bool) -> Colormap:
    return DIVERGING_BLUE_RED if diverging else SEQUENTIAL_BLUE


def _auto_clim(grids: list[np.ndarray], *, diverging: bool) -> tuple[float, float]:
    values = np.concatenate([g.ravel() for g in grids])
    if diverging:
        vmax = float(np.abs(values).max())
        return -vmax, vmax
    return float(values.min()), float(values.max())


def _label_for(normalize: dict | None) -> str:
    return "raw value" if normalize is None else METHOD_LABELS[normalize["method"]]


def _multi_panel_figure(n: int, *, max_cols: int = MAX_PANEL_COLS, panel_size: float = 4.5) -> tuple[plt.Figure, list[plt.Axes]]:
    """A grid of n panels, wrapped to at most max_cols per row (so it stays
    readable on screen), with any unused trailing panels hidden."""
    n_cols = min(n, max_cols)
    n_rows = -(-n // max_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_size * n_cols, panel_size * n_rows), squeeze=False)
    flat_axes = axes.ravel()
    for ax in flat_axes[n:]:
        ax.axis("off")
    return fig, list(flat_axes[:n])


def _nice_ticks(n: int, vmin: float, vmax: float, *, max_ticks: int = 10) -> tuple[list[int], list[str]]:
    """Up to max_ticks evenly spaced pixel positions (0..n-1) and their
    interpolated physical-value labels, spanning [vmin, vmax]."""
    positions = np.linspace(0, n - 1, min(n, max_ticks)).round().astype(int)
    values = vmin + positions / (n - 1) * (vmax - vmin)
    return positions.tolist(), [f"{v:.0f}" for v in values]


def _plot_heatmap(ax: plt.Axes, grid: np.ndarray, *, cmap: Colormap, vmin: float, vmax: float, label: str) -> None:
    """grid is indexed [red_idx, green_idx]; it is transposed for display so
    that red lands on x and green on y (imshow puts axis 0 on the y axis)."""
    im = ax.imshow(grid.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    red_vals, green_vals = load_grid_axes()
    ax.set_xticks(range(len(red_vals)))
    ax.set_xticklabels([f"{v:.0f}" for v in red_vals], rotation=45, ha="right")
    ax.set_xlabel("red")
    ax.set_yticks(range(len(green_vals)))
    ax.set_yticklabels([f"{v:.0f}" for v in green_vals])
    ax.set_ylabel("green")
    plt.colorbar(im, ax=ax, label=label)


def plot_run(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    sub_id: str,
    session: int,
    run: int,
    *,
    normalize: dict | None = None,
    clim: tuple[float, float] | None = None,
    cmap: Colormap | str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap for one participant/session/run. normalize, if given, is a dict
    of scope/trials/method kwargs for analysis.normalized_grid."""
    grid = raw_grid(runmap_df, sub_id, session, run) if normalize is None else normalized_grid(
        runmap_df, baselines_df, sub_id, session, run, **normalize
    )
    vmin, vmax = clim if clim is not None else _auto_clim([grid], diverging=normalize is not None)
    if ax is None:
        _, ax = plt.subplots()
    _plot_heatmap(ax, grid, cmap=cmap or _default_cmap(normalize is not None), vmin=vmin, vmax=vmax, label=_label_for(normalize))
    ax.set_title(f"{sub_id} session {session} run {run}")
    return ax


def plot_all_runs(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    sub_id: str,
    session: int,
    *,
    normalize: dict | None = None,
    clim: tuple[float, float] | None = None,
    cmap: Colormap | str | None = None,
) -> plt.Figure:
    """One heatmap per run of a participant/session (handles 3- or 4-run
    sessions), sharing one color scale across all panels by default."""
    runs = sorted(runmap_df.query("sub_id == @sub_id and session == @session")["run"].unique())
    grids = [
        raw_grid(runmap_df, sub_id, session, run)
        if normalize is None
        else normalized_grid(runmap_df, baselines_df, sub_id, session, run, **normalize)
        for run in runs
    ]
    vmin, vmax = clim if clim is not None else _auto_clim(grids, diverging=normalize is not None)
    resolved_cmap = cmap or _default_cmap(normalize is not None)
    label = _label_for(normalize)

    fig, axes = _multi_panel_figure(len(runs))
    for ax, run, grid in zip(axes, runs, grids):
        _plot_heatmap(ax, grid, cmap=resolved_cmap, vmin=vmin, vmax=vmax, label=label)
        ax.set_title(f"{sub_id} session {session} run {run}")
    fig.suptitle(f"{sub_id} session {session} -- all runs")
    fig.tight_layout()
    return fig


def plot_mean_run(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    sub_id: str,
    session: int,
    *,
    normalize: dict | None = None,
    clim: tuple[float, float] | None = None,
    cmap: Colormap | str | None = None,
) -> plt.Axes:
    """Heatmap of the mean grid across all runs of a participant/session."""
    grid = mean_grid(runmap_df, baselines_df, sub_id, session, normalize=normalize)
    vmin, vmax = clim if clim is not None else _auto_clim([grid], diverging=normalize is not None)
    _, ax = plt.subplots()
    _plot_heatmap(ax, grid, cmap=cmap or _default_cmap(normalize is not None), vmin=vmin, vmax=vmax, label=_label_for(normalize))
    ax.set_title(f"{sub_id} session {session} -- mean across runs")
    return ax


def plot_mean_across_subjects(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    *,
    sub_ids: list[str] | None = None,
    group: str | None = None,
    subgroup: str | None = None,
    normalize: dict | None = None,
    clim: tuple[float, float] | None = None,
    cmap: Colormap | str | None = None,
) -> plt.Axes:
    """Grand-mean heatmap: each subject's own mean-across-runs grid, then
    averaged across subjects. Pass sub_ids explicitly, or filter by group/
    subgroup (metadata_df); with none given, uses every subject at this
    session."""
    if sub_ids is None:
        sub_ids = subjects_in_group(metadata_df, session, group=group, subgroup=subgroup)
    grid = mean_grid_across_subjects(runmap_df, baselines_df, sub_ids, session, normalize=normalize)
    vmin, vmax = clim if clim is not None else _auto_clim([grid], diverging=normalize is not None)
    _, ax = plt.subplots()
    _plot_heatmap(ax, grid, cmap=cmap or _default_cmap(normalize is not None), vmin=vmin, vmax=vmax, label=_label_for(normalize))
    label = ", ".join(filter(None, [group, subgroup])) or f"{len(sub_ids)} subjects"
    ax.set_title(f"session {session} -- mean across {label}")
    return ax


def plot_group_all_methods(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    *,
    sub_ids: list[str] | None = None,
    group: str | None = None,
    subgroup: str | None = None,
    cmap: Colormap | str | None = None,
) -> plt.Figure:
    """One row of panels for one subject group's mean map: raw, then percent/
    db/zscore normalization (scope='run', trials='all'). Each panel is scaled
    independently (unlike the other multi-panel functions here) since raw and
    normalized values are on different numeric scales, not comparable on one
    shared color axis."""
    if sub_ids is None:
        sub_ids = subjects_in_group(metadata_df, session, group=group, subgroup=subgroup)
    variants = [("raw", None)] + [(m, {"scope": "run", "trials": "all", "method": m}) for m in ("percent", "db", "zscore")]

    fig, axes = _multi_panel_figure(len(variants))
    for ax, (name, normalize) in zip(axes, variants):
        grid = mean_grid_across_subjects(runmap_df, baselines_df, sub_ids, session, normalize=normalize)
        vmin, vmax = _auto_clim([grid], diverging=normalize is not None)
        _plot_heatmap(ax, grid, cmap=cmap or _default_cmap(normalize is not None), vmin=vmin, vmax=vmax, label=_label_for(normalize))
        ax.set_title(name)
    subtitle = ", ".join(filter(None, [group, subgroup])) or f"{len(sub_ids)} subjects"
    fig.suptitle(f"session {session} -- {subtitle} (n={len(sub_ids)}) -- raw + normalization methods")
    fig.tight_layout()
    return fig


def plot_subjects_side_by_side(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    *,
    sub_ids: list[str] | None = None,
    group: str | None = None,
    subgroup: str | None = None,
    normalize: dict | None = None,
    clim: tuple[float, float] | None = None,
    cmap: Colormap | str | None = None,
) -> plt.Figure:
    """One heatmap per subject (each subject's own mean-across-runs grid), side
    by side, sharing one color scale by default. Pass sub_ids explicitly, or
    filter by group/subgroup (metadata_df)."""
    if sub_ids is None:
        sub_ids = subjects_in_group(metadata_df, session, group=group, subgroup=subgroup)
    grids = [mean_grid(runmap_df, baselines_df, sub_id, session, normalize=normalize) for sub_id in sub_ids]
    vmin, vmax = clim if clim is not None else _auto_clim(grids, diverging=normalize is not None)
    resolved_cmap = cmap or _default_cmap(normalize is not None)
    label = _label_for(normalize)

    fig, axes = _multi_panel_figure(len(sub_ids))
    for ax, sub_id, grid in zip(axes, sub_ids, grids):
        _plot_heatmap(ax, grid, cmap=resolved_cmap, vmin=vmin, vmax=vmax, label=label)
        ax.set_title(sub_id)
    subtitle = ", ".join(filter(None, [group, subgroup])) or f"{len(sub_ids)} subjects"
    fig.suptitle(f"session {session} -- {subtitle} side by side")
    fig.tight_layout()
    return fig


def plot_groups_side_by_side(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    categories: list[dict],
    *,
    normalize: dict | None = None,
    clim: tuple[float, float] | None = None,
    cmap: Colormap | str | None = None,
) -> plt.Figure:
    """One heatmap per category (each category's mean map across its matching
    subjects), side by side, panel titled with its sample size. categories is
    a list of {"label": str, "group": str | None, "subgroup": str | None},
    e.g. [{"label": "PD", "group": "PD"}, {"label": "protan", "subgroup": "protan"}]."""
    sub_id_lists = [subjects_in_group(metadata_df, session, group=cat.get("group"), subgroup=cat.get("subgroup")) for cat in categories]
    grids = [
        mean_grid_across_subjects(runmap_df, baselines_df, sub_ids, session, normalize=normalize) for sub_ids in sub_id_lists
    ]
    vmin, vmax = clim if clim is not None else _auto_clim(grids, diverging=normalize is not None)
    resolved_cmap = cmap or _default_cmap(normalize is not None)
    label = _label_for(normalize)

    fig, axes = _multi_panel_figure(len(categories))
    for ax, cat, sub_ids, grid in zip(axes, categories, sub_id_lists, grids):
        _plot_heatmap(ax, grid, cmap=resolved_cmap, vmin=vmin, vmax=vmax, label=label)
        ax.set_title(f"{cat['label']} (n={len(sub_ids)})")
    fig.suptitle(f"session {session} -- groups side by side")
    fig.tight_layout()
    return fig


def plot_interpolated_grid(
    grid: np.ndarray,
    shape: tuple[int, int],
    *,
    method: str = "linear",
    label: str = "value",
    diverging: bool = False,
    clim: tuple[float, float] | None = None,
    cmap: Colormap | str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of `grid` (indexed [red_idx, green_idx], any source: raw,
    normalized, single-subject, or group mean) resized to `shape` =
    (n_red, n_green) via interpolation.

    Same orientation convention as every other plot function here (red on x,
    green on y) -- just generalized to an arbitrary, including rectangular,
    resolution.
    """
    n_red, n_green = shape
    interp = interpolate_grid(grid, (n_red, n_green), method=method)
    vmin, vmax = clim if clim is not None else _auto_clim([interp], diverging=diverging)
    red_vals, green_vals = load_grid_axes()

    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(interp.T, origin="lower", cmap=cmap or _default_cmap(diverging), vmin=vmin, vmax=vmax, aspect="auto")
    xpos, xlabels = _nice_ticks(n_red, red_vals[0], red_vals[-1])
    ypos, ylabels = _nice_ticks(n_green, green_vals[0], green_vals[-1])
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_xlabel("red")
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel("green")
    plt.colorbar(im, ax=ax, label=label)
    if title:
        ax.set_title(title)
    return ax


def _boxplot(ax: plt.Axes, data: list[np.ndarray], labels: list[str], *, ylabel: str) -> None:
    """One box per array in data. A single uniform fill color is used throughout
    -- the x tick labels already carry category identity, so a static one-series
    chart needs no additional categorical hue."""
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, medianprops={"color": "black"})
    for patch in bp["boxes"]:
        patch.set_facecolor(DISTRIBUTION_COLOR)
        patch.set_alpha(0.7)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_ylabel(ylabel)


def _histogram(ax: plt.Axes, data: np.ndarray, *, xlabel: str, bins: int) -> None:
    ax.hist(data, bins=bins, color=DISTRIBUTION_COLOR, alpha=0.7, edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")


def plot_subject_boxplot(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    sub_id: str,
    session: int,
    *,
    normalize: dict | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Single box: every pixel of every run of one subject/session (400 values,
    or 300 for the ragged 3-run subjects)."""
    data = flatten_runs(runmap_df, baselines_df, sub_id, session, normalize=normalize)
    if ax is None:
        _, ax = plt.subplots()
    _boxplot(ax, [data], [sub_id], ylabel=_label_for(normalize))
    ax.set_title(f"{sub_id} session {session} -- all-run pixels (n={len(data)})")
    return ax


def plot_subject_mean_boxplot(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    sub_id: str,
    session: int,
    *,
    normalize: dict | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Single box: the 100 cells of one subject's mean-across-runs grid."""
    data = mean_grid(runmap_df, baselines_df, sub_id, session, normalize=normalize).ravel()
    if ax is None:
        _, ax = plt.subplots()
    _boxplot(ax, [data], [sub_id], ylabel=_label_for(normalize))
    ax.set_title(f"{sub_id} session {session} -- mean-grid pixels (n={len(data)})")
    return ax


def plot_subject_histogram(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    sub_id: str,
    session: int,
    *,
    normalize: dict | None = None,
    bins: int = 30,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of every pixel of every run of one subject/session."""
    data = flatten_runs(runmap_df, baselines_df, sub_id, session, normalize=normalize)
    if ax is None:
        _, ax = plt.subplots()
    _histogram(ax, data, xlabel=_label_for(normalize), bins=bins)
    ax.set_title(f"{sub_id} session {session} -- all-run pixels (n={len(data)})")
    return ax


def plot_subject_mean_histogram(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    sub_id: str,
    session: int,
    *,
    normalize: dict | None = None,
    bins: int = 30,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of the 100 cells of one subject's mean-across-runs grid."""
    data = mean_grid(runmap_df, baselines_df, sub_id, session, normalize=normalize).ravel()
    if ax is None:
        _, ax = plt.subplots()
    _histogram(ax, data, xlabel=_label_for(normalize), bins=bins)
    ax.set_title(f"{sub_id} session {session} -- mean-grid pixels (n={len(data)})")
    return ax


def plot_subjects_boxplot(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    *,
    sub_ids: list[str] | None = None,
    group: str | None = None,
    subgroup: str | None = None,
    normalize: dict | None = None,
) -> plt.Axes:
    """One box per subject (that subject's all-run pixel distribution), side by
    side. Pass sub_ids explicitly, or filter by group/subgroup (metadata_df)."""
    if sub_ids is None:
        sub_ids = subjects_in_group(metadata_df, session, group=group, subgroup=subgroup)
    data = [flatten_runs(runmap_df, baselines_df, sub_id, session, normalize=normalize) for sub_id in sub_ids]
    _, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(sub_ids)), 4.5))
    _boxplot(ax, data, sub_ids, ylabel=_label_for(normalize))
    subtitle = ", ".join(filter(None, [group, subgroup])) or f"{len(sub_ids)} subjects"
    ax.set_title(f"session {session} -- {subtitle} -- all-run pixels per subject")
    return ax


def plot_subjects_mean_boxplot(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    *,
    sub_ids: list[str] | None = None,
    group: str | None = None,
    subgroup: str | None = None,
    normalize: dict | None = None,
) -> plt.Axes:
    """One box per subject (that subject's mean-across-runs grid, 100 cells),
    side by side. Pass sub_ids explicitly, or filter by group/subgroup."""
    if sub_ids is None:
        sub_ids = subjects_in_group(metadata_df, session, group=group, subgroup=subgroup)
    data = [mean_grid(runmap_df, baselines_df, sub_id, session, normalize=normalize).ravel() for sub_id in sub_ids]
    _, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(sub_ids)), 4.5))
    _boxplot(ax, data, sub_ids, ylabel=_label_for(normalize))
    subtitle = ", ".join(filter(None, [group, subgroup])) or f"{len(sub_ids)} subjects"
    ax.set_title(f"session {session} -- {subtitle} -- mean-grid pixels per subject")
    return ax


def plot_group_pooled_boxplot(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    *,
    sub_ids: list[str] | None = None,
    group: str | None = None,
    subgroup: str | None = None,
    normalize: dict | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Single box: every pixel of every run of every subject in the group,
    pooled together (not averaged)."""
    if sub_ids is None:
        sub_ids = subjects_in_group(metadata_df, session, group=group, subgroup=subgroup)
    data = pooled_pixels(runmap_df, baselines_df, sub_ids, session, normalize=normalize)
    label = ", ".join(filter(None, [group, subgroup])) or f"{len(sub_ids)} subjects"
    if ax is None:
        _, ax = plt.subplots()
    _boxplot(ax, [data], [label], ylabel=_label_for(normalize))
    ax.set_title(f"session {session} -- {label} pooled pixels (n={len(data)}, {len(sub_ids)} subjects)")
    return ax


def plot_group_mean_boxplot(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    *,
    sub_ids: list[str] | None = None,
    group: str | None = None,
    subgroup: str | None = None,
    normalize: dict | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Single box: the 100 cells of the group's mean-of-subject-means grid
    (each subject's own mean-across-runs grid, then averaged across subjects)."""
    if sub_ids is None:
        sub_ids = subjects_in_group(metadata_df, session, group=group, subgroup=subgroup)
    data = mean_grid_across_subjects(runmap_df, baselines_df, sub_ids, session, normalize=normalize).ravel()
    label = ", ".join(filter(None, [group, subgroup])) or f"{len(sub_ids)} subjects"
    if ax is None:
        _, ax = plt.subplots()
    _boxplot(ax, [data], [label], ylabel=_label_for(normalize))
    ax.set_title(f"session {session} -- {label} mean-grid pixels (n={len(data)}, {len(sub_ids)} subjects)")
    return ax


def plot_groups_pooled_boxplot(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    categories: list[dict],
    *,
    normalize: dict | None = None,
) -> plt.Axes:
    """One box per category (that category's pooled all-run pixel
    distribution), side by side. categories as in plot_groups_side_by_side."""
    sub_id_lists = [subjects_in_group(metadata_df, session, group=cat.get("group"), subgroup=cat.get("subgroup")) for cat in categories]
    data = [pooled_pixels(runmap_df, baselines_df, sub_ids, session, normalize=normalize) for sub_ids in sub_id_lists]
    labels = [f"{cat['label']} (n={len(sub_ids)})" for cat, sub_ids in zip(categories, sub_id_lists)]
    _, ax = plt.subplots()
    _boxplot(ax, data, labels, ylabel=_label_for(normalize))
    ax.set_title(f"session {session} -- groups pooled pixels")
    return ax


def plot_groups_mean_boxplot(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    categories: list[dict],
    *,
    normalize: dict | None = None,
) -> plt.Axes:
    """One box per category (that category's mean-of-subject-means grid, 100
    cells), side by side. categories as in plot_groups_side_by_side."""
    sub_id_lists = [subjects_in_group(metadata_df, session, group=cat.get("group"), subgroup=cat.get("subgroup")) for cat in categories]
    data = [
        mean_grid_across_subjects(runmap_df, baselines_df, sub_ids, session, normalize=normalize).ravel() for sub_ids in sub_id_lists
    ]
    labels = [f"{cat['label']} (n={len(sub_ids)})" for cat, sub_ids in zip(categories, sub_id_lists)]
    _, ax = plt.subplots()
    _boxplot(ax, data, labels, ylabel=_label_for(normalize))
    ax.set_title(f"session {session} -- groups mean-grid pixels")
    return ax


def plot_groups_baseline_boxplot(
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    categories: list[dict],
    *,
    trials: str = "all",
) -> plt.Axes:
    """One box per category, its raw baseline trial values pooled across every
    run and subject in that category -- for comparing the baseline itself
    across groups, not a normalized response. Always raw: the baseline is the
    normalization's own denominator, so there is nothing to normalize it
    against. categories as in plot_groups_side_by_side."""
    sub_id_lists = [subjects_in_group(metadata_df, session, group=cat.get("group"), subgroup=cat.get("subgroup")) for cat in categories]
    data = [pooled_baseline_values(baselines_df, sub_ids, session, trials=trials) for sub_ids in sub_id_lists]
    labels = [f"{cat['label']} (n={len(sub_ids)})" for cat, sub_ids in zip(categories, sub_id_lists)]
    _, ax = plt.subplots()
    _boxplot(ax, data, labels, ylabel="baseline (raw)")
    ax.set_title(f"session {session} -- baseline values by group")
    return ax


MARKER_SHAPES = ["o", "s", "^", "D", "v", "P", "X"]


def plot_trough_scatter(troughs_df: pd.DataFrame, label_col: str, *, ax: plt.Axes | None = None) -> plt.Axes:
    """Scatter of trough (red, green) locations from a subject_troughs.csv or
    group_troughs.csv table (analysis.subject_troughs/group_troughs), one
    marker shape per distinct label_col value ('group' or 'label'). Shape,
    not color, carries category identity here -- at 4+ categories a scatter's
    all-pairs color comparisons exceed the categorical palette's CVD-safe cap
    (see the dataviz skill's palette notes), so shape sidesteps that entirely."""
    if ax is None:
        _, ax = plt.subplots()
    for shape, label in zip(MARKER_SHAPES, sorted(troughs_df[label_col].unique())):
        sub = troughs_df[troughs_df[label_col] == label]
        ax.scatter(sub["red"], sub["green"], marker=shape, s=80, alpha=0.7, facecolor=DISTRIBUTION_COLOR, edgecolor="black", label=label)
    red_vals, green_vals = load_grid_axes()
    ax.set_xlim(min(red_vals), max(red_vals))
    ax.set_ylim(min(green_vals), max(green_vals))
    ax.set_xlabel("red")
    ax.set_ylabel("green")
    ax.legend(title=label_col)
    ax.set_title("trough locations")
    return ax


def plot_troughs_boxplot(
    troughs_df: pd.DataFrame, value_col: str, label_col: str, *, ylabel: str | None = None, ax: plt.Axes | None = None
) -> plt.Axes:
    """One box per distinct label_col value in troughs_df (a subject_troughs
    table, or anything with that shape), from troughs_df[value_col] -- e.g.
    ramp_slope_red by subgroup (M6), or any other per-subject scalar feature
    against group/subgroup/label. NaN values are dropped per category (a
    fit column can be NaN for subjects where that particular fit failed)."""
    if ax is None:
        _, ax = plt.subplots()
    labels_present = sorted(troughs_df[label_col].unique())
    data = [troughs_df.loc[troughs_df[label_col] == label, value_col].dropna().to_numpy() for label in labels_present]
    labels = [f"{label} (n={len(d)})" for label, d in zip(labels_present, data)]
    _boxplot(ax, data, labels, ylabel=ylabel or value_col)
    ax.set_title(f"{value_col} by {label_col}")
    return ax


def plot_permutation_result(
    result: dict, panels: list[tuple[str, str]], *, title: str | None = None, cmap: Colormap | str | None = None
) -> plt.Figure:
    """One panel per (result_key, title) pair from a permutation.permutation_test_*
    result dict, sharing one z-score color scale across all panels."""
    grids = [result[key] for key, _ in panels]
    vmin, vmax = _auto_clim(grids, diverging=True)
    resolved_cmap = cmap or DIVERGING_BLUE_RED
    fig, axes = _multi_panel_figure(len(panels))
    for ax, (key, panel_title) in zip(axes, panels):
        _plot_heatmap(ax, result[key], cmap=resolved_cmap, vmin=vmin, vmax=vmax, label="z-score")
        ax.set_title(panel_title)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_permutation_test_size(result: dict, *, title: str | None = None, cmap: Colormap | str | None = None) -> plt.Figure:
    """Difference / uncorrected / cluster-size-corrected panels, from
    permutation.permutation_test_size's result dict."""
    return plot_permutation_result(
        result,
        [("zdiff", "difference (z)"), ("zthresh_uncorrected", "uncorrected"), ("zthresh_corrected", "cluster-size corrected")],
        title=title,
        cmap=cmap,
    )


def plot_permutation_test_weighted(result: dict, *, title: str | None = None, cmap: Colormap | str | None = None) -> plt.Figure:
    """Difference / uncorrected / size-corrected / weight-corrected panels,
    from permutation.permutation_test_weighted's result dict."""
    return plot_permutation_result(
        result,
        [
            ("zdiff", "difference (z)"),
            ("zthresh_uncorrected", "uncorrected"),
            ("zthresh_size_corrected", "size corrected"),
            ("zthresh_weight_corrected", "weight corrected"),
        ],
        title=title,
        cmap=cmap,
    )


def plot_permutation_test_directional(result: dict, *, title: str | None = None, cmap: Colormap | str | None = None) -> plt.Figure:
    """Difference / uncorrected / size- and weight-corrected positive and
    negative panels, from permutation.permutation_test_directional's result dict."""
    return plot_permutation_result(
        result,
        [
            ("zdiff", "difference (z)"),
            ("zthresh_uncorrected", "uncorrected"),
            ("zthresh_size_pos", "size corrected (+)"),
            ("zthresh_size_neg", "size corrected (-)"),
            ("zthresh_weight_pos", "weight corrected (+)"),
            ("zthresh_weight_neg", "weight corrected (-)"),
        ],
        title=title,
        cmap=cmap,
    )


def plot_permutation_null_histogram(null_values: np.ndarray, threshold: float, *, xlabel: str, ax: plt.Axes | None = None) -> plt.Axes:
    """Histogram of a null max-cluster-statistic distribution (e.g.
    result['null_sizes'] or result['null_weights']) with the corresponding
    significance threshold marked."""
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(null_values, bins=30, color=DISTRIBUTION_COLOR, alpha=0.7, edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold = {threshold:.1f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (permutations)")
    ax.legend()
    return ax


def plot_trough_locations(
    grid: np.ndarray, locations: dict[str, dict], *, cmap: Colormap | str | None = None, clim: tuple[float, float] | None = None, ax: plt.Axes | None = None
) -> plt.Axes:
    """Heatmap of grid with one marker per named location (each a dict with
    'red'/'green' keys, e.g. from analysis.trough_location or
    fit_trough_surface) -- for visually comparing trough-finding methods.
    Locations are converted from physical red/green values to the heatmap's
    pixel-index axes via interpolation, so a parametric fit's continuous,
    off-grid location overlays correctly alongside the grid argmin's
    on-grid one. Entries with a NaN location (failed fit) are skipped."""
    if ax is None:
        _, ax = plt.subplots()
    vmin, vmax = clim if clim is not None else _auto_clim([grid], diverging=True)
    _plot_heatmap(ax, grid, cmap=cmap or DIVERGING_BLUE_RED, vmin=vmin, vmax=vmax, label="value")

    red_vals, green_vals = load_grid_axes()
    for shape, (label, loc) in zip(MARKER_SHAPES, locations.items()):
        if np.isnan(loc["red"]) or np.isnan(loc["green"]):
            continue
        red_pos = np.interp(loc["red"], red_vals, range(len(red_vals)))
        green_pos = np.interp(loc["green"], green_vals, range(len(green_vals)))
        ax.scatter([red_pos], [green_pos], marker=shape, s=150, facecolor="white", edgecolor="black", linewidth=1.5, label=label)
    ax.legend()
    return ax


def plot_icc_map(icc: np.ndarray, *, title: str | None = None, cmap: Colormap | str | None = None, ax: plt.Axes | None = None) -> plt.Axes:
    """Heatmap of a [red_idx, green_idx] ICC map (reliability.icc_map), fixed
    to the [0, 1] ICC scale -- sequential (magnitude, unsigned), like the raw
    heatmaps. Title defaults to the mean/median ICC across the map, matching
    the template's own summary."""
    if ax is None:
        _, ax = plt.subplots()
    _plot_heatmap(ax, icc, cmap=cmap or SEQUENTIAL_BLUE, vmin=0.0, vmax=1.0, label="ICC")
    ax.set_title(title or f"mean ICC = {icc.mean():.2f}, median = {np.median(icc):.2f}")
    return ax


def plot_bland_altman(values1: np.ndarray, values2: np.ndarray, *, ax: plt.Axes | None = None) -> plt.Axes:
    """Bland-Altman plot: mean of the two measurements (x) vs. their
    difference (y), with the mean difference (bias) and +/-1.96 SD limits of
    agreement marked."""
    if ax is None:
        _, ax = plt.subplots()
    mean, diff = (values1 + values2) / 2, values1 - values2
    bias, sd = diff.mean(), diff.std()
    ax.scatter(mean, diff, color=DISTRIBUTION_COLOR, alpha=0.7, edgecolor="white")
    ax.axhline(bias, color="black", linewidth=1, label=f"bias = {bias:.2f}")
    ax.axhline(bias + 1.96 * sd, color="black", linestyle="--", linewidth=1, label=f"+/-1.96 SD = {1.96 * sd:.2f}")
    ax.axhline(bias - 1.96 * sd, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("mean of session 1, session 2")
    ax.set_ylabel("session 1 - session 2")
    ax.legend(fontsize=8)
    return ax


def plot_session_scatter(values1: np.ndarray, values2: np.ndarray, *, ax: plt.Axes | None = None) -> plt.Axes:
    """Session 1 vs. session 2 scatter, with the y=x identity line for reference."""
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(values1, values2, color=DISTRIBUTION_COLOR, alpha=0.7, edgecolor="white")
    lo, hi = min(values1.min(), values2.min()), max(values1.max(), values2.max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("session 1")
    ax.set_ylabel("session 2")
    return ax


def plot_example_points(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    sub_ids: list[str],
    points: list[dict],
    *,
    kind: str = "bland_altman",
    normalize: dict | None = DEFAULT_NORMALIZE,
    title: str | None = None,
) -> plt.Figure:
    """One panel per point in points (each a {"label", "red_idx", "green_idx"}
    dict, e.g. from reliability.example_points_fixed/example_points_informative):
    a Bland-Altman (kind='bland_altman') or session1-vs-session2 scatter
    (kind='scatter') of the paired session values across sub_ids at that
    grid cell."""
    plot_fn = {"bland_altman": plot_bland_altman, "scatter": plot_session_scatter}[kind]
    fig, axes = _multi_panel_figure(len(points))
    for ax, point in zip(axes, points):
        values1, values2 = session_pair_values(runmap_df, baselines_df, sub_ids, point["red_idx"], point["green_idx"], normalize=normalize)
        plot_fn(values1, values2, ax=ax)
        ax.set_title(f"{point['label']} (red_idx={point['red_idx']}, green_idx={point['green_idx']})")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
