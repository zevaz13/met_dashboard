"""Side-by-side EEG heatmap and behavioral click-density map, for visually
comparing where each modality's response is concentrated.

Both panels share the physical (red, green) grid, [red_idx, green_idx]
orientation (see overlap.py's module docstring for the orientation bug this
project fixed relative to the template code) -- red on x, green on y,
matching every other heatmap in this repo.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap

from overlap import DEFAULT_GREEN, DEFAULT_RED, behavioral_density_map

EEG_CMAP = "viridis"
DENSITY_CMAP = "Blues"
CLICK_COLOR = "white"  # readable against every value of EEG_CMAP (viridis), unlike any single fixed hue

# Same wrap-to-N-per-row convention as ssveps/scripts/plotting.py and
# beh/scripts/plotting.py's own (independent) copies of this helper.
MAX_PANEL_COLS = 5


def _multi_panel_figure(n: int, *, max_cols: int = MAX_PANEL_COLS, panel_size: float = 4.5) -> tuple[plt.Figure, list[plt.Axes]]:
    n_cols = min(n, max_cols)
    n_rows = -(-n // max_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_size * n_cols, panel_size * n_rows), squeeze=False)
    flat_axes = axes.ravel()
    for ax in flat_axes[n:]:
        ax.axis("off")
    return fig, list(flat_axes[:n])


def _auto_clim(grids: list[np.ndarray], *, diverging: bool) -> tuple[float, float]:
    values = np.concatenate([g.ravel() for g in grids])
    if diverging:
        vmax = float(np.abs(values).max())
        return -vmax, vmax
    return float(values.min()), float(values.max())


def _plot_grid(ax: plt.Axes, grid: np.ndarray, *, red: np.ndarray, green: np.ndarray, cmap: str, title: str) -> None:
    # grid is [red_idx, green_idx]; imshow needs [row, col] = [y, x], so
    # transpose and put the origin at the bottom to keep red on x, green on y.
    im = ax.imshow(grid.T, origin="lower", cmap=cmap, aspect="auto", extent=(red[0], red[-1], green[0], green[-1]))
    ax.set_xlabel("red")
    ax.set_ylabel("green")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_overlap(
    beh_df, eeg_grid: np.ndarray, sub_ids: list[str], *, red: np.ndarray = DEFAULT_RED, green: np.ndarray = DEFAULT_GREEN, title: str | None = None
) -> plt.Figure:
    """Two panels: the EEG response grid (eeg_grid, e.g. from
    analysis.mean_grid/mean_grid_across_subjects) and the behavioral
    click-density map for the same sub_ids (behavioral_density_map) --
    single participant (one-element sub_ids) or a pooled group."""
    B = behavioral_density_map(beh_df, sub_ids, red=red, green=green)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _plot_grid(axes[0], eeg_grid, red=red, green=green, cmap=EEG_CMAP, title="EEG response")
    _plot_grid(axes[1], B, red=red, green=green, cmap=DENSITY_CMAP, title=f"Behavioral clicks (n={int(B.sum())})")
    fig.suptitle(title or f"{len(sub_ids)} subject(s)")
    fig.tight_layout()
    return fig


def plot_grid_with_clicks(
    eeg_grid: np.ndarray,
    clicks_df,
    *,
    red: np.ndarray = DEFAULT_RED,
    green: np.ndarray = DEFAULT_GREEN,
    xlim: tuple[float, float] = (0, 3200),
    ylim: tuple[float, float] = (0, 2000),
    s: float = 20,
    alpha: float = 0.8,
    cmap: Colormap | str = EEG_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """The EEG response grid as a heatmap, with the actual behavioral click
    points scattered on top -- one combined view, complementing
    plot_overlap's side-by-side density comparison (M2, PLANssvepvsBeh.md).

    clicks_df needs 'red'/'green' columns -- pass a subject's own rows, a
    group's pooled rows, or an outlier-filtered subset of either (e.g.
    beh/scripts/features.py's subject_outliers/group_outliers) for the
    "repeated with outliers removed" version. xlim/ylim default to the EEG
    grid's own sampled range: a handful of subjects' clicks run past
    green=2000 (see PLANssvepvsBeh.md M2 for the list), shown clipped at
    the axis edge rather than expanding the view, for visual comparability
    across every subject/group plot. s/alpha (dashboard M2): marker size and
    opacity, smaller/more transparent than the notebook defaults gives
    room to shrink these when several panels share limited screen space.
    cmap defaults to EEG_CMAP (viridis); pass a project-specific ramp (e.g.
    ssveps/scripts/plotting.py's DIVERGING_BLUE_RED) to match that
    project's own normalization-method color convention. vmin/vmax default
    to None (per-panel autoscale, as before) -- plot_grids_with_clicks passes
    a shared pair so every panel in a multi-panel figure is comparable."""
    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(eeg_grid.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", extent=(red[0], red[-1], green[0], green[-1]))
    ax.scatter(clicks_df["red"], clicks_df["green"], color=CLICK_COLOR, s=s, alpha=alpha, edgecolor="black", linewidth=0.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("red")
    ax.set_ylabel("green")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_grids_with_clicks(
    eeg_grids: list[np.ndarray],
    clicks_dfs: list,
    titles: list[str],
    *,
    red: np.ndarray = DEFAULT_RED,
    green: np.ndarray = DEFAULT_GREEN,
    xlim: tuple[float, float] = (0, 3200),
    ylim: tuple[float, float] = (0, 2000),
    s: float = 20,
    alpha: float = 0.8,
    cmap: Colormap | str = EEG_CMAP,
    diverging: bool = False,
    suptitle: str | None = None,
) -> plt.Figure:
    """One plot_grid_with_clicks panel per (grid, clicks, title) triple,
    wrapped to MAX_PANEL_COLS per row with a shared color scale across every
    panel (dashboard M2's "show behavioral clicks" toggle) -- the
    click-overlay analog of ssveps/scripts/plotting.py's
    plot_groups_side_by_side/plot_subjects_side_by_side, which this mirrors
    (_multi_panel_figure, _auto_clim) since a group/subject-count-driven
    heatmap grid, unlike a single-panel scatter, has no colorblind-safety
    cap to enforce -- it wraps instead. diverging matches _auto_clim's
    convention: False (default) for a raw/sequential eeg_grid, True for a
    normalized (percent/db) one, same as ssveps' own raw-vs-normalized
    distinction."""
    if not (len(eeg_grids) == len(clicks_dfs) == len(titles)):
        raise ValueError(f"eeg_grids ({len(eeg_grids)}), clicks_dfs ({len(clicks_dfs)}), and titles ({len(titles)}) must be the same length")
    vmin, vmax = _auto_clim(eeg_grids, diverging=diverging)
    fig, axes = _multi_panel_figure(len(eeg_grids))
    for ax, grid, clicks, title in zip(axes, eeg_grids, clicks_dfs, titles):
        plot_grid_with_clicks(grid, clicks, red=red, green=green, xlim=xlim, ylim=ylim, s=s, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig
