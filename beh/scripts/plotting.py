"""Scatter plots of the manual/behavioral (red, green) match-point data.

Axes: red is always x, green is always y, matching ssveps' convention and
the physical stimulus this task shares with it. Default axis limits are
x=[0, 3200], y=[0, 2000] (the SSVEP stimulus grid's own range) for visual
comparability across plots -- some subjects' green values run past 2000
(the raw data does, see loader.load_behavioral), and are not clipped, just
plotted against a fixed default window; pass ylim= to see the full range for
a specific subject/group.

Color follows the dataviz skill's categorical rule: the first three
palette slots are the only ones validated for an all-pairs (scatter)
comparison, so per-session coloring (at most 3 sessions in this dataset)
uses them directly. Multi-subject/multi-group figures instead use one panel
per subject/group (faceting, the skill's own prescribed alternative past
three categories) with a single uniform point color -- identity comes from
the panel title, not hue, the same convention ssveps/scripts/plotting.py
uses for its boxplots.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from comparisons import group_points
from features import group_features, outlier_mask, subject_outliers, subject_pca_line
from loader import subjects_in_group

POINT_COLOR = "#2a78d6"
SESSION_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated all-pairs for scatter, up to 3 categories
SCATTER4_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]  # validated all-pairs for scatter, up to 4 (dashboard M2) --
# #4a3aa7 replaces FULL_PALETTE's natural 4th slot, #eda100, which fails the
# dataviz skill's normal-vision floor against #eb6834 (deltaE 13.7 < 15)
FIT_LINE_COLOR = "#52514e"  # secondary ink, not a categorical slot: this is a derived overlay, not a series
OUTLIER_COLOR = SESSION_COLORS[1]  # same validated all-pairs slot as session 2 -- outlier/inlier is a 2-category scatter too
# Full 8-hue categorical order (dataviz skill's default palette), for the centroid
# plots below -- those are a handful of large, legended, well-separated marks, not
# a dense point cloud, so past 3 categories identity is carried by CENTROID_MARKERS
# (shape) as well as color, not by hue alone (past slot 3 the palette isn't
# all-pairs colorblind-safe on its own -- see plot_feature_space/SESSION_COLORS).
FULL_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CENTROID_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]
XLIM = (0, 3200)
YLIM = (0, 2000)

# Every multi-panel figure here wraps to at most this many panels per row.
MAX_PANEL_COLS = 5


def _style_axes(ax: plt.Axes, *, xlim: tuple[float, float], ylim: tuple[float, float], title: str | None = None) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("red")
    ax.set_ylabel("green")
    if title:
        ax.set_title(title)


def _plot_centroid_series(ax: plt.Axes, label: str, points: np.ndarray, *, color: str, marker: str) -> None:
    """One category's centroid (mean of points, ±1 SD error bars) as a
    single marker -- shared by plot_group_centroids and
    plot_feature_group_centroids (M3)."""
    mean = points.mean(axis=0)
    sd = points.std(axis=0, ddof=1) if len(points) > 1 else (0.0, 0.0)
    ax.errorbar(
        mean[0], mean[1], xerr=sd[0], yerr=sd[1], fmt=marker, color=color, markersize=10, markeredgecolor="white", capsize=4, label=f"{label} (n={len(points)})"
    )


def _multi_panel_figure(n: int, *, max_cols: int = MAX_PANEL_COLS, panel_size: float = 3.5) -> tuple[plt.Figure, list[plt.Axes]]:
    n_cols = min(n, max_cols)
    n_rows = -(-n // max_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_size * n_cols, panel_size * n_rows), squeeze=False)
    flat_axes = axes.ravel()
    for ax in flat_axes[n:]:
        ax.axis("off")
    return fig, list(flat_axes[:n])


def plot_subject_session(
    df: pd.DataFrame, sub_id: str, session: int, *, xlim: tuple[float, float] = XLIM, ylim: tuple[float, float] = YLIM, ax: plt.Axes | None = None
) -> plt.Axes:
    """Scatter of one subject's clicks in one session."""
    sub = df[(df["sub_id"] == sub_id) & (df["session"] == session)]
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(sub["red"], sub["green"], color=POINT_COLOR, alpha=0.7, edgecolor="white")
    _style_axes(ax, xlim=xlim, ylim=ylim, title=f"{sub_id} session {session} (n={len(sub)})")
    return ax


def plot_subject_sessions(
    df: pd.DataFrame,
    sub_id: str,
    *,
    sessions: list[int] | None = None,
    xlim: tuple[float, float] = XLIM,
    ylim: tuple[float, float] = YLIM,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter of one subject's clicks, every session overlaid, one color
    per session (at most 3 sessions in this dataset -- see module docstring
    for why color, not shape, is safe here)."""
    sub = df[df["sub_id"] == sub_id]
    if sessions is None:
        sessions = sorted(sub["session"].unique())
    if ax is None:
        _, ax = plt.subplots()
    for session, color in zip(sessions, SESSION_COLORS):
        points = sub[sub["session"] == session]
        ax.scatter(points["red"], points["green"], color=color, alpha=0.7, edgecolor="white", label=f"session {session} (n={len(points)})")
    _style_axes(ax, xlim=xlim, ylim=ylim, title=sub_id)
    ax.legend(fontsize=8)
    return ax


def plot_subject_cloud(
    df: pd.DataFrame,
    sub_id: str,
    *,
    xlim: tuple[float, float] = XLIM,
    ylim: tuple[float, float] = YLIM,
    show_fit: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter of every click from every session of one subject, pooled into
    a single color -- the whole-subject point cloud, session distinctions
    dropped. show_fit=True overlays that subject's fitted PCA line
    (features.subject_pca_line), spanning the actual data extent along the
    first principal component (M2 -- see features.py)."""
    sub = df[df["sub_id"] == sub_id]
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(sub["red"], sub["green"], color=POINT_COLOR, alpha=0.7, edgecolor="white")
    if show_fit:
        p1, p2 = subject_pca_line(df, sub_id)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=FIT_LINE_COLOR, linewidth=2, linestyle="--")
    _style_axes(ax, xlim=xlim, ylim=ylim, title=f"{sub_id} (n={len(sub)}, all sessions)")
    return ax


def plot_subjects_pooled(
    df: pd.DataFrame,
    *,
    sub_ids: list[str] | None = None,
    group: str | None = None,
    subgroup: str | None = None,
    xlim: tuple[float, float] = XLIM,
    ylim: tuple[float, float] = YLIM,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter of every click from every subject/session matching
    sub_ids (explicit) or group/subgroup, pooled into one panel and one
    color -- "all the data from one group on one plot" (PLANbeh.md M1).
    Pass sub_ids to plot an arbitrary hand-picked set of subjects together
    instead of a group/subgroup filter."""
    if sub_ids is None:
        sub_ids = subjects_in_group(df, group=group, subgroup=subgroup)
    sub = df[df["sub_id"].isin(sub_ids)]
    label = ", ".join(filter(None, [group, subgroup])) or f"{len(sub_ids)} subjects"
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(sub["red"], sub["green"], color=POINT_COLOR, alpha=0.5, edgecolor="white")
    _style_axes(ax, xlim=xlim, ylim=ylim, title=f"{label} (n={len(sub_ids)} subjects, {len(sub)} clicks)")
    return ax


def plot_subjects_grid(
    df: pd.DataFrame,
    *,
    sub_ids: list[str] | None = None,
    group: str | None = None,
    subgroup: str | None = None,
    xlim: tuple[float, float] = XLIM,
    ylim: tuple[float, float] = YLIM,
    show_fit: bool = False,
    max_cols: int = MAX_PANEL_COLS,
) -> plt.Figure:
    """One panel per subject (that subject's whole point cloud, every
    session pooled), wrapped to at most max_cols per row -- "all on a
    grid (max 5 participants per row)" (PLANbeh.md M1). Pass sub_ids for an
    arbitrary hand-picked set of subjects side by side instead of a
    group/subgroup filter. show_fit=True overlays each subject's fitted PCA
    line (see plot_subject_cloud). max_cols defaults to MAX_PANEL_COLS (5);
    dashboard/pages/2_Behavioral.py's raw-clicks tab passes max_cols=4 for
    an even 4+4 layout at its 8-subject cap (dashboard M3)."""
    if sub_ids is None:
        sub_ids = subjects_in_group(df, group=group, subgroup=subgroup)
    fig, axes = _multi_panel_figure(len(sub_ids), max_cols=max_cols)
    for ax, sub_id in zip(axes, sub_ids):
        plot_subject_cloud(df, sub_id, xlim=xlim, ylim=ylim, show_fit=show_fit, ax=ax)
    subtitle = ", ".join(filter(None, [group, subgroup])) or f"{len(sub_ids)} subjects"
    fig.suptitle(subtitle)
    fig.tight_layout()
    return fig


def _draw_ellipse(ax: plt.Axes, pca: dict, *, n_std: float) -> None:
    """The outlier boundary itself: an ellipse centered at pca['mean'],
    semi-axes n_std standard deviations along pc1/pc2 (M4)."""
    angle = np.degrees(np.arctan2(pca["pc1"][1], pca["pc1"][0]))
    ellipse = Ellipse(
        pca["mean"], width=2 * n_std * np.sqrt(pca["along_var"]), height=2 * n_std * np.sqrt(pca["perp_var"]),
        angle=angle, facecolor="none", edgecolor=FIT_LINE_COLOR, linewidth=2, linestyle="--",
    )
    ax.add_patch(ellipse)


def plot_subject_outliers(
    df: pd.DataFrame,
    sub_id: str,
    *,
    n_std: float = 2.0,
    pca: dict | None = None,
    xlim: tuple[float, float] = XLIM,
    ylim: tuple[float, float] = YLIM,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """One subject's clicks, colored by whether they fall outside a fitted
    ellipse at n_std standard deviations along each principal axis (M4).

    pca=None (default): fit the ellipse to this subject's own points --
    the per-participant outlier check. pca=<a group_outliers result's
    "pca">: classify this subject's points against a shared group/subgroup
    ellipse instead -- the group-level check, where the same ellipse is
    drawn in every subject's panel and only which of their own points fall
    outside it changes."""
    if pca is None:
        result = subject_outliers(df, sub_id, n_std=n_std)
        pca, points, mask = result["pca"], result["points"], result["outlier_mask"]
    else:
        points = df.loc[df["sub_id"] == sub_id, ["red", "green"]].to_numpy()
        mask = outlier_mask(pca, points, n_std=n_std)

    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(points[~mask, 0], points[~mask, 1], color=POINT_COLOR, alpha=0.7, edgecolor="white", label="inlier")
    ax.scatter(points[mask, 0], points[mask, 1], color=OUTLIER_COLOR, alpha=0.9, edgecolor="white", label="outlier")
    _draw_ellipse(ax, pca, n_std=n_std)
    _style_axes(ax, xlim=xlim, ylim=ylim, title=f"{sub_id} ({int(mask.sum())}/{len(points)} outliers)")
    ax.legend(fontsize=6, loc="upper right")
    return ax


def plot_subjects_outliers_grid(
    df: pd.DataFrame,
    *,
    sub_ids: list[str] | None = None,
    group: str | None = None,
    subgroup: str | None = None,
    n_std: float = 2.0,
    shared_pca: dict | None = None,
    xlim: tuple[float, float] = XLIM,
    ylim: tuple[float, float] = YLIM,
) -> plt.Figure:
    """One panel per subject (plot_subject_outliers), wrapped to at most
    MAX_PANEL_COLS per row -- "all participants as a grid" (M4).

    shared_pca=None (default): each panel fits and draws that subject's own
    ellipse (the per-participant check). shared_pca=<a group_outliers
    result's "pca">: every panel draws the same shared group/subgroup
    ellipse instead, classifying each subject's own points against it (the
    group-level check) -- pass group_outliers(df, group=..., n_std=n_std)
    ["pca"] to match."""
    if sub_ids is None:
        sub_ids = subjects_in_group(df, group=group, subgroup=subgroup)
    fig, axes = _multi_panel_figure(len(sub_ids))
    for ax, sub_id in zip(axes, sub_ids):
        plot_subject_outliers(df, sub_id, n_std=n_std, pca=shared_pca, xlim=xlim, ylim=ylim, ax=ax)
    subtitle = ", ".join(filter(None, [group, subgroup])) or f"{len(sub_ids)} subjects"
    ellipse_note = "shared group ellipse" if shared_pca is not None else "per-subject ellipse"
    fig.suptitle(f"{subtitle} (n_std={n_std}, {ellipse_note})")
    fig.tight_layout()
    return fig


def plot_groups_side_by_side(
    df: pd.DataFrame, categories: list[dict], *, xlim: tuple[float, float] = XLIM, ylim: tuple[float, float] = YLIM
) -> plt.Figure:
    """One panel per category, each showing that category's pooled point
    cloud -- "plot HC group, next to PD next to CVD, next to protan, next
    to deutan" (PLANbeh.md M1). categories is a list of {"label", "group",
    "subgroup"} dicts, same shape as ssveps' plotting.plot_groups_side_by_side."""
    fig, axes = _multi_panel_figure(len(categories))
    for ax, cat in zip(axes, categories):
        plot_subjects_pooled(df, group=cat.get("group"), subgroup=cat.get("subgroup"), xlim=xlim, ylim=ylim, ax=ax)
        ax.set_title(cat["label"] + "\n" + ax.get_title())
    fig.tight_layout()
    return fig


def plot_feature_space(
    df: pd.DataFrame, categories: list[dict], *, x_feature: str = "orientation_deg", y_feature: str = "perp_var", ax: plt.Axes | None = None
) -> plt.Axes:
    """One subject-level point per category in shape-feature space (M2,
    features.subject_shape_features) -- x_feature/y_feature name one of its
    keys ('orientation_deg', 'along_var', 'perp_var'). categories is the
    same {"label", "group", "subgroup"} shape as plot_groups_side_by_side,
    capped at len(SCATTER4_COLORS): like plot_subject_sessions, this overlays
    series by hue on one panel, so it inherits the dataviz skill's all-pairs
    scatter color limit -- past that, call it multiple times or facet
    instead. (Dashboard M2: raised from 3 to 4 categories using
    SCATTER4_COLORS, a separately-validated all-pairs-safe 4-color set.)"""
    if len(categories) > len(SCATTER4_COLORS):
        raise ValueError(f"plot_feature_space supports at most {len(SCATTER4_COLORS)} categories (scatter all-pairs color limit), got {len(categories)}")
    if ax is None:
        _, ax = plt.subplots()
    for cat, color in zip(categories, SCATTER4_COLORS):
        feats = group_features(df, group=cat.get("group"), subgroup=cat.get("subgroup"))
        ax.scatter(feats[x_feature], feats[y_feature], color=color, alpha=0.8, edgecolor="white", label=f"{cat['label']} (n={len(feats)})")
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.legend(fontsize=8)
    return ax


def plot_subject_centroids(df: pd.DataFrame, categories: list[dict], *, xlim: tuple[float, float] = XLIM, ylim: tuple[float, float] = YLIM, ax: plt.Axes | None = None) -> plt.Axes:
    """One point per subject, at that subject's own mean (red, green) across
    every click/session (comparisons.group_points's unit='subject' points,
    plotted directly instead of collapsed into a single Hotelling T^2 stat)
    -- M3. categories is the same {"label", "group", "subgroup"} shape as
    plot_groups_side_by_side, capped at len(SESSION_COLORS) like
    plot_feature_space: every point here can sit next to any other, so it
    inherits the same all-pairs scatter color limit."""
    if len(categories) > len(SESSION_COLORS):
        raise ValueError(f"plot_subject_centroids supports at most {len(SESSION_COLORS)} categories (scatter all-pairs color limit), got {len(categories)}")
    if ax is None:
        _, ax = plt.subplots()
    for cat, color in zip(categories, SESSION_COLORS):
        points = group_points(df, group=cat.get("group"), subgroup=cat.get("subgroup"), unit="subject")
        ax.scatter(points[:, 0], points[:, 1], color=color, alpha=0.85, edgecolor="white", s=60, label=f"{cat['label']} (n={len(points)})")
    _style_axes(ax, xlim=xlim, ylim=ylim, title="Subject centroids")
    ax.legend(fontsize=8)
    return ax


def plot_group_centroids(df: pd.DataFrame, categories: list[dict], *, xlim: tuple[float, float] = XLIM, ylim: tuple[float, float] = YLIM, ax: plt.Axes | None = None) -> plt.Axes:
    """One marker per category: the centroid of its subjects' own (red,
    green) means, ±1 SD error bars across those subject centroids -- M3.
    No category-count cap (see FULL_PALETTE/CENTROID_MARKERS above):
    identity is carried by marker shape as well as color."""
    if ax is None:
        _, ax = plt.subplots()
    for i, cat in enumerate(categories):
        points = group_points(df, group=cat.get("group"), subgroup=cat.get("subgroup"), unit="subject")
        _plot_centroid_series(ax, cat["label"], points, color=FULL_PALETTE[i % len(FULL_PALETTE)], marker=CENTROID_MARKERS[i % len(CENTROID_MARKERS)])
    _style_axes(ax, xlim=xlim, ylim=ylim, title="Group centroids (mean ± 1 SD across subjects)")
    ax.legend(fontsize=8)
    return ax


def plot_feature_group_centroids(
    df: pd.DataFrame, categories: list[dict], *, x_feature: str = "orientation_deg", y_feature: str = "perp_var", ax: plt.Axes | None = None
) -> plt.Axes:
    """Same as plot_group_centroids, in shape-feature space instead of
    (red, green) -- M3, the feature analog of plot_feature_space the same
    way plot_group_centroids is the analog of plot_subject_centroids. One
    marker per category at the mean of its subjects' x_feature/y_feature
    values, ±1 SD error bars, no category-count cap."""
    if ax is None:
        _, ax = plt.subplots()
    for i, cat in enumerate(categories):
        feats = group_features(df, group=cat.get("group"), subgroup=cat.get("subgroup"))
        points = feats[[x_feature, y_feature]].to_numpy()
        _plot_centroid_series(ax, cat["label"], points, color=FULL_PALETTE[i % len(FULL_PALETTE)], marker=CENTROID_MARKERS[i % len(CENTROID_MARKERS)])
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f"Group centroids: {x_feature} vs {y_feature}")
    ax.legend(fontsize=8)
    return ax


def plot_subjects_cloud_overlay(
    df: pd.DataFrame,
    sub_ids: list[str],
    *,
    xlim: tuple[float, float] = XLIM,
    ylim: tuple[float, float] = YLIM,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Every click from several subjects (every session pooled per subject,
    like plot_subject_cloud), overlaid on one panel with one color per
    subject (dashboard M2's "multiple participants ... each with a different
    color" raw-click view) -- unlike plot_subjects_grid's one-panel-per-
    subject faceting. Up to len(SCATTER4_COLORS) (4) subjects, the same
    all-pairs scatter color limit as plot_feature_space."""
    if len(sub_ids) > len(SCATTER4_COLORS):
        raise ValueError(f"plot_subjects_cloud_overlay supports at most {len(SCATTER4_COLORS)} subjects, got {len(sub_ids)}")
    if ax is None:
        _, ax = plt.subplots()
    for sub_id, color in zip(sub_ids, SCATTER4_COLORS):
        sub = df[df["sub_id"] == sub_id]
        ax.scatter(sub["red"], sub["green"], color=color, alpha=0.7, edgecolor="white", label=f"{sub_id} (n={len(sub)})")
    _style_axes(ax, xlim=xlim, ylim=ylim, title="Participants (all sessions pooled)")
    ax.legend(fontsize=8)
    return ax
