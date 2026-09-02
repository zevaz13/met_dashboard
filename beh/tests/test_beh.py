"""Regression tests for the behavioral (manual point-matching) pipeline.

Run: uv run pytest beh/tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

# ssveps/scripts/ has its own loader.py/plotting.py -- if a combined run (e.g.
# bare `pytest` at the repo root) collected that test module first, sys.modules
# would already hold ssveps' versions under these bare names. Drop them so the
# imports below re-resolve against SCRIPTS, just inserted at sys.path[0],
# regardless of what ran before this module in the same pytest session.
for _name in ("loader", "plotting"):
    sys.modules.pop(_name, None)

import comparisons  # noqa: E402
import features  # noqa: E402
import loader  # noqa: E402
import plotting  # noqa: E402
import retest  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return loader.load_behavioral()


# --- loader ------------------------------------------------------------


def test_load_behavioral_has_expected_shape_and_columns(df):
    assert list(df.columns) == ["sub_id", "session", "click", "red", "green", "group", "subgroup", "date", "folder"]
    assert len(df) == 1590
    assert df["sub_id"].nunique() == 47


def test_part_type_maps_to_group_matching_ssveps_metadata(df):
    """Confirmed empirically against ssveps/files/metadata.csv for every
    overlapping subject before choosing this mapping -- pin it."""
    ssvep_meta = pd.read_csv(loader.SSVEP_METADATA_PATH, keep_default_na=False)
    overlap = ssvep_meta.drop_duplicates("sub_id").set_index("sub_id")["group"]
    checked = 0
    for sub_id, group in df[["sub_id", "group"]].drop_duplicates().itertuples(index=False):
        if sub_id in overlap.index:
            assert group == overlap[sub_id]
            checked += 1
    assert checked == 43  # every ssveps-known subject in this dataset


def test_subgroup_is_populated_for_every_cvd_subject(df):
    cvd = df[df["group"] == "CVD"]
    assert (cvd["subgroup"].isin(["protan", "deutan"])).all()


def test_subgroup_is_na_for_subjects_missing_from_ssvep_metadata(df):
    """MET013/MET014 (HC) and MET041/MET042 (PD) aren't in ssveps'
    metadata.csv -- they must not silently disappear or error, just get 'NA'."""
    extra = df[df["sub_id"].isin(["MET013", "MET014", "MET041", "MET042"])]
    assert len(extra) > 0
    assert (extra["subgroup"] == "NA").all()


def test_subjects_in_group_filters_correctly(df):
    protan = loader.subjects_in_group(df, subgroup="protan")
    assert set(protan) == set(df.loc[df["subgroup"] == "protan", "sub_id"].unique())
    assert len(loader.subjects_in_group(df)) == 47


# --- comparisons (Hotelling T^2) ----------------------------------------


def test_group_points_subject_unit_is_one_row_per_subject(df):
    sub_ids = loader.subjects_in_group(df, group="PD")
    points = comparisons.group_points(df, group="PD", unit="subject")
    assert points.shape == (len(sub_ids), 2)


def test_group_points_point_unit_is_one_row_per_click(df):
    n_clicks = len(df[df["group"] == "PD"])
    points = comparisons.group_points(df, group="PD", unit="point")
    assert points.shape == (n_clicks, 2)


def test_group_points_subject_mean_matches_manual_groupby(df):
    points = comparisons.group_points(df, group="PD", unit="subject")
    expected = df[df["group"] == "PD"].groupby("sub_id")[["red", "green"]].mean().sort_index().to_numpy()
    np.testing.assert_allclose(np.sort(points, axis=0), np.sort(expected, axis=0))


def test_group_points_rejects_unknown_unit(df):
    with pytest.raises(ValueError):
        comparisons.group_points(df, group="PD", unit="nonsense")


def test_compare_groups_detects_a_known_separation():
    """Two clearly separated synthetic clusters, one point per row (unit
    doesn't matter at n=1 click per subject) -- should reject the null hard."""
    rng = np.random.default_rng(0)
    n = 20
    synth = pd.DataFrame(
        {
            "sub_id": [f"A{i}" for i in range(n)] + [f"B{i}" for i in range(n)],
            "session": 1,
            "click": 1,
            "red": np.concatenate([rng.normal(500, 50, n), rng.normal(2500, 50, n)]),
            "green": np.concatenate([rng.normal(500, 50, n), rng.normal(1500, 50, n)]),
            "group": ["A"] * n + ["B"] * n,
            "subgroup": "NA",
        }
    )
    result = comparisons.compare_groups(synth, group1="A", group2="B", unit="subject")
    assert result["p_value"] < 1e-6
    assert result["n1"] == n and result["n2"] == n


def test_compare_groups_point_unit_pseudoreplicates_toward_significance(df):
    """Same two groups, both units -- pooling every click as if independent
    should push the p-value down relative to the subject-mean version (more
    'observations', same underlying separation), not the other way round."""
    subject_result = comparisons.compare_groups(df, group1="CTR", group2="PD", unit="subject")
    point_result = comparisons.compare_groups(df, group1="CTR", group2="PD", unit="point")
    assert point_result["n1"] > subject_result["n1"]
    assert point_result["p_value"] <= subject_result["p_value"]


# --- features (PCA shape, M2) --------------------------------------------


def test_subject_shape_features_returns_expected_keys(df):
    feat = features.subject_shape_features(df, "MET001")
    assert set(feat) == {"orientation_deg", "along_var", "perp_var", "n"}
    assert feat["along_var"] >= feat["perp_var"]
    assert feat["n"] == len(df[df["sub_id"] == "MET001"])


def test_subject_shape_features_orientation_is_folded_to_0_180(df):
    for sub_id in loader.subjects_in_group(df)[:10]:
        feat = features.subject_shape_features(df, sub_id)
        assert 0 <= feat["orientation_deg"] < 180


def test_subject_shape_features_raises_for_fewer_than_two_points():
    synth = pd.DataFrame({"sub_id": ["A1"], "red": [500], "green": [500]})
    with pytest.raises(ValueError):
        features.subject_shape_features(synth, "A1")


def test_subject_pca_line_endpoints_span_projected_extent(df):
    """p1/p2 are the subject's own most-extreme points projected onto the
    fitted line -- every point should project inside [p1, p2], none beyond."""
    p1, p2 = features.subject_pca_line(df, "MET001")
    points = df.loc[df["sub_id"] == "MET001", ["red", "green"]].to_numpy()
    direction = (p2 - p1) / np.linalg.norm(p2 - p1)
    proj = (points - p1) @ direction
    assert proj.min() >= -1e-6
    assert proj.max() <= np.linalg.norm(p2 - p1) + 1e-6


def test_group_features_one_row_per_subject(df):
    sub_ids = loader.subjects_in_group(df, subgroup="protan")
    feats = features.group_features(df, subgroup="protan")
    assert list(feats.index) == sub_ids
    assert set(feats.columns) == {"orientation_deg", "along_var", "perp_var", "n"}


def test_compare_shape_feature_detects_a_known_separation():
    """Group A: points scattered tightly off a horizontal line (small
    perp_var). Group B: same line, much noisier off-line scatter (large
    perp_var). compare_shape_feature on 'perp_var' should reject the null."""
    rng = np.random.default_rng(0)
    n_subjects, n_points = 10, 20
    rows = []
    for group, perp_sd in [("A", 5), ("B", 80)]:
        for i in range(n_subjects):
            along = rng.normal(0, 200, n_points)
            perp = rng.normal(0, perp_sd, n_points)
            rows.append(pd.DataFrame({"sub_id": f"{group}{i}", "red": 1000 + along, "green": 1000 + perp, "group": group, "subgroup": "NA"}))
    synth = pd.concat(rows, ignore_index=True)
    result = features.compare_shape_feature(synth, "perp_var", group1="A", group2="B")
    assert result["p_value"] < 0.001
    assert result["n1"] == n_subjects and result["n2"] == n_subjects


# --- plotting ------------------------------------------------------------


def test_plot_subject_session_sets_default_axis_limits(df):
    ax = plotting.plot_subject_session(df, "MET001", 1)
    assert ax.get_xlim() == plotting.XLIM
    assert ax.get_ylim() == plotting.YLIM
    assert ax.get_xlabel() == "red" and ax.get_ylabel() == "green"


def test_plot_subject_sessions_draws_one_series_per_session(df):
    ax = plotting.plot_subject_sessions(df, "MET001")
    n_sessions = df.loc[df["sub_id"] == "MET001", "session"].nunique()
    assert len(ax.collections) == n_sessions


def test_plot_subjects_grid_wraps_at_max_panel_cols(df):
    sub_ids = loader.subjects_in_group(df, subgroup="protan")  # 8 subjects
    fig = plotting.plot_subjects_grid(df, sub_ids=sub_ids)
    rows, cols = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert (rows, cols) == (2, plotting.MAX_PANEL_COLS)  # ceil(8/5) rows x 5 cols
    visible = [ax for ax in fig.axes if ax.axison]
    assert len(visible) == len(sub_ids)


def test_plot_subjects_grid_respects_max_cols_override(df):
    """dashboard M3: the raw-clicks tab wants an even 4+4 layout at 8
    subjects, not MAX_PANEL_COLS' default 5+3."""
    sub_ids = loader.subjects_in_group(df, subgroup="protan")  # 8 subjects
    fig = plotting.plot_subjects_grid(df, sub_ids=sub_ids, max_cols=4)
    rows, cols = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert (rows, cols) == (2, 4)


def test_plot_groups_side_by_side_one_panel_per_category(df):
    categories = [{"label": "HC", "group": "CTR"}, {"label": "PD", "group": "PD"}]
    fig = plotting.plot_groups_side_by_side(df, categories)
    assert len(fig.axes) == len(categories)


def test_plot_subject_cloud_show_fit_adds_a_line(df):
    ax = plotting.plot_subject_cloud(df, "MET001", show_fit=True)
    assert len(ax.lines) == 1


def test_plot_feature_space_rejects_more_categories_than_scatter4_colors(df):
    categories = [
        {"label": "HC", "group": "CTR"},
        {"label": "PD", "group": "PD"},
        {"label": "CVD", "group": "CVD"},
        {"label": "protan", "subgroup": "protan"},
        {"label": "deutan", "subgroup": "deutan"},
    ]
    with pytest.raises(ValueError):
        plotting.plot_feature_space(df, categories)


def test_plot_feature_space_one_series_per_category(df):
    categories = [{"label": "HC", "group": "CTR"}, {"label": "PD", "group": "PD"}]
    ax = plotting.plot_feature_space(df, categories)
    assert len(ax.collections) == len(categories)


def test_plot_subjects_cloud_overlay_one_series_per_subject(df):
    sub_ids = sorted(df["sub_id"].unique())[:3]
    ax = plotting.plot_subjects_cloud_overlay(df, sub_ids)
    assert len(ax.collections) == len(sub_ids)


def test_plot_subjects_cloud_overlay_rejects_more_subjects_than_scatter4_colors(df):
    sub_ids = sorted(df["sub_id"].unique())[: len(plotting.SCATTER4_COLORS) + 1]
    with pytest.raises(ValueError):
        plotting.plot_subjects_cloud_overlay(df, sub_ids)


# --- centroid plots (M3) --------------------------------------------------


def test_plot_subject_centroids_one_series_per_category(df):
    categories = [{"label": "HC", "group": "CTR"}, {"label": "PD", "group": "PD"}]
    ax = plotting.plot_subject_centroids(df, categories)
    assert len(ax.collections) == len(categories)


def test_plot_subject_centroids_rejects_more_categories_than_session_colors(df):
    categories = [
        {"label": "HC", "group": "CTR"},
        {"label": "PD", "group": "PD"},
        {"label": "CVD", "group": "CVD"},
        {"label": "protan", "subgroup": "protan"},
    ]
    with pytest.raises(ValueError):
        plotting.plot_subject_centroids(df, categories)


def test_plot_group_centroids_one_marker_per_category_beyond_the_scatter_cap(df):
    """Unlike plot_subject_centroids/plot_feature_space, this one has no
    category-count cap -- pin that 5 categories works."""
    categories = [
        {"label": "HC", "group": "CTR"},
        {"label": "PD", "group": "PD"},
        {"label": "CVD", "group": "CVD"},
        {"label": "protan", "subgroup": "protan"},
        {"label": "deutan", "subgroup": "deutan"},
    ]
    ax = plotting.plot_group_centroids(df, categories)
    _, labels = ax.get_legend_handles_labels()
    assert len(labels) == len(categories)


def test_plot_group_centroids_marker_is_at_the_mean_of_subject_centroids(df):
    categories = [{"label": "PD", "group": "PD"}]
    ax = plotting.plot_group_centroids(df, categories)
    expected = comparisons.group_points(df, group="PD", unit="subject").mean(axis=0)
    line = ax.lines[0]
    np.testing.assert_allclose(line.get_xydata()[0], expected)


def test_plot_feature_group_centroids_one_marker_per_category(df):
    categories = [
        {"label": "HC", "group": "CTR"},
        {"label": "protan", "subgroup": "protan"},
        {"label": "deutan", "subgroup": "deutan"},
    ]
    ax = plotting.plot_feature_group_centroids(df, categories)
    _, labels = ax.get_legend_handles_labels()
    assert len(labels) == len(categories)


# --- features: per-session features, outlier ellipse (M4) -----------------


def test_subject_session_features_one_row_per_subject_session(df):
    result = features.subject_session_features(df[df["sub_id"] == "MET001"])
    n_sessions = df.loc[df["sub_id"] == "MET001", "session"].nunique()
    assert len(result) == n_sessions
    assert set(result.columns) == {"sub_id", "session", "centroid_red", "centroid_green", "orientation_deg", "along_var", "perp_var", "n"}


def test_subject_session_features_centroid_matches_manual_mean(df):
    result = features.subject_session_features(df[df["sub_id"] == "MET001"])
    for _, row in result.iterrows():
        expected = df[(df["sub_id"] == "MET001") & (df["session"] == row["session"])][["red", "green"]].mean()
        assert row["centroid_red"] == pytest.approx(expected["red"])
        assert row["centroid_green"] == pytest.approx(expected["green"])


def test_subject_session_features_skips_sessions_with_fewer_than_two_points():
    synth = pd.DataFrame({"sub_id": ["A1", "A1", "A1"], "session": [1, 1, 2], "red": [500, 520, 500], "green": [500, 480, 500]})
    result = features.subject_session_features(synth)
    assert list(result["session"]) == [1]  # session 2 has only 1 point, dropped


def test_within_session_scatter_is_zero_for_identical_clicks():
    synth = pd.DataFrame({"sub_id": ["A1"] * 4, "session": [1, 1, 2, 2], "red": [500, 500, 700, 700], "green": [500, 500, 700, 700]})
    assert features.within_session_scatter(synth, "A1") == pytest.approx(0.0)


def test_within_session_scatter_averages_sessions_equally():
    """One session with 2 points at distance 10 from their centroid, one
    session with 10 points at distance 100 -- averaging per-session first
    (not pooling all points) should land near (10+100)/2, not be dragged
    toward 100 by the larger session's point count."""
    rng = np.random.default_rng(0)
    session1 = pd.DataFrame({"sub_id": "A1", "session": 1, "red": [490, 510], "green": [500, 500]})
    angles = rng.uniform(0, 2 * np.pi, 10)
    session2 = pd.DataFrame({"sub_id": "A1", "session": 2, "red": 700 + 100 * np.cos(angles), "green": 700 + 100 * np.sin(angles)})
    synth = pd.concat([session1, session2], ignore_index=True)
    # (10 + ~93.8) / 2 -- session2's RMS-to-its-own-sample-centroid isn't
    # exactly 100 at n=10 points, computed directly rather than assumed.
    assert features.within_session_scatter(synth, "A1") == pytest.approx(51.9, abs=0.5)


def test_outlier_mask_flags_points_beyond_n_std_along_each_axis():
    # A perfectly axis-aligned synthetic cloud: pc1 along red, pc2 along green,
    # so the ellipse boundary is exactly known in raw (red, green) units.
    rng = np.random.default_rng(0)
    red = rng.normal(0, 100, 200)
    green = rng.normal(0, 20, 200)
    points = np.column_stack([red, green])
    pca = features._points_pca(points)

    n_std = 2.0
    clearly_inside = np.array([[0.0, 0.0], [50.0, 5.0]])
    clearly_outside = np.array([[100.0 * n_std * 3, 0.0], [0.0, 20.0 * n_std * 3]])
    assert not features.outlier_mask(pca, clearly_inside, n_std=n_std).any()
    assert features.outlier_mask(pca, clearly_outside, n_std=n_std).all()


def test_subject_outliers_flags_a_planted_outlier():
    rng = np.random.default_rng(0)
    n = 40
    inliers = pd.DataFrame({"sub_id": "A1", "red": 1000 + rng.normal(0, 50, n), "green": 1000 + rng.normal(0, 50, n)})
    outlier = pd.DataFrame({"sub_id": "A1", "red": [1000 + 2000], "green": [1000 + 2000]})
    synth = pd.concat([inliers, outlier], ignore_index=True)
    result = features.subject_outliers(synth, "A1", n_std=2.0)
    assert result["outlier_mask"][-1]  # the planted point
    assert result["outlier_mask"].sum() < n  # most of the tight cluster stays inside


def test_group_outliers_table_has_one_row_per_click(df):
    result = features.group_outliers(df, group="PD", n_std=2.0)
    assert len(result["table"]) == len(df[df["group"] == "PD"])
    assert set(result["table"]["sub_id"]) == set(loader.subjects_in_group(df, group="PD"))
    assert result["table"]["is_outlier"].dtype == bool


# --- retest.py: cross-session feature reliability (M4) --------------------


def test_paired_subjects_requires_both_sessions(df):
    paired = retest.paired_subjects(df, group="CTR")
    session1 = set(loader.subjects_in_group(df[df["session"] == 1], group="CTR"))
    session2 = set(loader.subjects_in_group(df[df["session"] == 2], group="CTR"))
    assert set(paired) == session1 & session2
    assert len(paired) > 0


def test_reliability_table_returns_all_five_features(df):
    result = retest.reliability_table(df, group="CTR")
    assert set(result["feature"]) == {"centroid_red", "centroid_green", "along_var", "perp_var", "orientation_deg"}
    angle_row = result[result["feature"] == "orientation_deg"].iloc[0]
    assert angle_row["statistic"] == "circ_r"
    assert (result[result["feature"] != "orientation_deg"]["statistic"] == "icc").all()
    assert result["value"].between(-1, 1).all()


def test_reliability_table_raises_below_three_paired_subjects(df):
    with pytest.raises(ValueError):
        retest.reliability_table(df, group="HD")  # n=1 subject total


# --- plotting: outlier ellipse (M4) ----------------------------------------


def test_plot_subject_outliers_draws_two_series_and_an_ellipse(df):
    ax = plotting.plot_subject_outliers(df, "MET001")
    assert len(ax.collections) == 2  # inliers, outliers (one may be empty)
    assert len(ax.patches) == 1


def test_plot_subject_outliers_shared_pca_draws_the_same_ellipse_everywhere(df):
    shared = features.group_outliers(df, group="PD", n_std=2.0)["pca"]
    ax1 = plotting.plot_subject_outliers(df, "MET025", pca=shared)
    ax2 = plotting.plot_subject_outliers(df, "MET030", pca=shared)
    e1, e2 = ax1.patches[0], ax2.patches[0]
    assert (e1.width, e1.height, e1.angle) == (e2.width, e2.height, e2.angle)
    assert tuple(e1.center) == tuple(e2.center)


def test_plot_subjects_outliers_grid_wraps_at_max_panel_cols(df):
    sub_ids = loader.subjects_in_group(df, subgroup="protan")  # 8 subjects
    fig = plotting.plot_subjects_outliers_grid(df, sub_ids=sub_ids)
    rows, cols = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert (rows, cols) == (2, plotting.MAX_PANEL_COLS)
    visible = [ax for ax in fig.axes if ax.axison]
    assert len(visible) == len(sub_ids)
