"""Regression tests for the ssvepBeh (behavioral-vs-EEG overlap) pipeline.

Run: uv run pytest ssvepBeh/tests -q
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, path: Path):
    """Load a module directly by file path, bypassing sys.modules entirely
    -- beh/, ssveps/, standardizedScores/FM100/, and this project each have
    their own loader.py/plotting.py under the same bare names, so a bare
    `import loader` here would be ambiguous (see beh/README.md's Tests
    section). Used only for beh's loader.py, the one cross-project import
    ssvepBeh/scripts/overlap.py itself doesn't need (see its docstring)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


beh_loader = _load("beh_loader_for_ssvepbeh_tests", ROOT / "beh" / "scripts" / "loader.py")
template = _load("template_grid_mapping", ROOT / "ssvepBeh" / "templateCode" / "grid_mapping.py")

# overlap.py/plotting.py are this project's own modules -- import normally,
# dropping any stale loader/plotting cache first (overlap.py inserts
# ssveps/scripts onto sys.path itself and imports "analysis", a unique name,
# so it doesn't need the same defense).
sys.path.insert(0, str(SCRIPTS))
for _name in ("loader", "plotting"):
    sys.modules.pop(_name, None)
import overlap  # noqa: E402

# overlap's own import above inserted ssveps/scripts at sys.path[0] as a
# side effect (see its docstring) -- ssveps/scripts has its own plotting.py
# too, so re-assert this project's scripts/ first before resolving
# "plotting", or it would silently import ssveps' version instead.
sys.path.remove(str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS))
sys.modules.pop("plotting", None)
import plotting  # noqa: E402
import correlation  # noqa: E402 -- inserts beh/scripts and imports "features" (unique name), no further defense needed
import session_reliability  # noqa: E402 -- only imports overlap/correlation (unique names), no further defense needed

analysis = overlap.analysis  # ssveps/scripts/analysis.py, already imported by overlap


@pytest.fixture(scope="module")
def beh_df():
    return beh_loader.load_behavioral()


@pytest.fixture(scope="module")
def runmap_df():
    return analysis.load_runmap()


@pytest.fixture(scope="module")
def baselines_df():
    return analysis.load_baselines()


@pytest.fixture(scope="module")
def metadata_df():
    return analysis.load_metadata()


# --- orientation (the bug this project found and fixed) --------------------


def test_behavioral_density_map_orientation_matches_template_subs_not_outmat(beh_df):
    """The template's outMat = subs.T is [green_idx, red_idx] -- the same
    axis-swap bug ssveps/ already fixed once (docs/ssvep_summary.md finding
    2.1). Pin that this module's density map matches the template's
    untransposed subs (correct) and NOT outMat (swapped)."""
    sub_id = "MET001"
    clicks = beh_df.loc[beh_df["sub_id"] == sub_id, ["red", "green"]].to_numpy()
    idx, out_mat = template.closest_grid_indices(clicks)
    subs = np.zeros((10, 10), dtype=int)
    for red_idx, green_idx in idx:
        subs[red_idx, green_idx] += 1

    B = overlap.behavioral_density_map(beh_df, [sub_id])
    np.testing.assert_array_equal(B, subs)
    assert not np.array_equal(B, out_mat)  # outMat is the swapped orientation -- must NOT match


def test_behavioral_centroid_peaks_at_the_expected_grid_cell(beh_df):
    """A subject's mean click should land near the density map's peak cell
    -- an independent sanity check on orientation beyond the template
    comparison above."""
    sub_id = "MET001"
    clicks = beh_df.loc[beh_df["sub_id"] == sub_id, ["red", "green"]]
    expected_red_idx = int(np.argmin(np.abs(overlap.DEFAULT_RED - clicks["red"].mean())))
    expected_green_idx = int(np.argmin(np.abs(overlap.DEFAULT_GREEN - clicks["green"].mean())))

    B = overlap.behavioral_density_map(beh_df, [sub_id])
    peak_red_idx, peak_green_idx = np.unravel_index(np.argmax(B), B.shape)
    assert (peak_red_idx, peak_green_idx) == (expected_red_idx, expected_green_idx)


# --- behavioral_density_map --------------------------------------------------


def test_behavioral_density_map_sums_to_click_count(beh_df):
    sub_id = "MET001"
    n_clicks = len(beh_df[beh_df["sub_id"] == sub_id])
    B = overlap.behavioral_density_map(beh_df, [sub_id])
    assert B.shape == (10, 10)
    assert B.sum() == n_clicks


def test_behavioral_density_map_pools_multiple_subjects(beh_df):
    sub_ids = ["MET001", "MET002"]
    n_clicks = len(beh_df[beh_df["sub_id"].isin(sub_ids)])
    B = overlap.behavioral_density_map(beh_df, sub_ids)
    assert B.sum() == n_clicks


# --- weighted_overlap_test --------------------------------------------------


def test_weighted_overlap_test_obs_stat_matches_template_formula(beh_df, runmap_df, baselines_df):
    """obs_stat = sum(E * B/B.sum()) -- same formula as the template's
    permWeighted2Dshifts, applied to the same (correctly oriented) B/E."""
    sub_id, session = "MET001", 1
    B = overlap.behavioral_density_map(beh_df, [sub_id])
    E = analysis.mean_grid(runmap_df, baselines_df, sub_id, session, normalize=analysis.DEFAULT_NORMALIZE)

    result = overlap.weighted_overlap_test(B, E, n_perm=100, seed=0)
    expected_obs_stat = float(np.sum(E * (B / B.sum())))
    assert result["obs_stat"] == pytest.approx(expected_obs_stat)


def test_weighted_overlap_test_is_reproducible_under_a_seed(beh_df, runmap_df, baselines_df):
    sub_id, session = "MET001", 1
    B = overlap.behavioral_density_map(beh_df, [sub_id])
    E = analysis.mean_grid(runmap_df, baselines_df, sub_id, session, normalize=analysis.DEFAULT_NORMALIZE)
    r1 = overlap.weighted_overlap_test(B, E, n_perm=200, seed=42)
    r2 = overlap.weighted_overlap_test(B, E, n_perm=200, seed=42)
    np.testing.assert_array_equal(r1["null_stats"], r2["null_stats"])
    assert r1["p_value"] == r2["p_value"]


def test_weighted_overlap_test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        overlap.weighted_overlap_test(np.ones((10, 10)), np.ones((5, 5)))


def test_weighted_overlap_test_rejects_empty_behavioral_map():
    with pytest.raises(ValueError):
        overlap.weighted_overlap_test(np.zeros((10, 10)), np.ones((10, 10)))


def test_weighted_overlap_test_detects_a_known_overlap():
    """B and E both concentrated in the same corner -- obs_stat should sit
    at the low tail of the null (p small), since a real toroidal shift would
    almost always move E's low corner away from B's peak."""
    B = np.zeros((10, 10))
    B[0, 0] = 100  # every click in one cell
    E = np.ones((10, 10)) * 10
    E[0, 0] = 0  # EEG response is minimal exactly where clicks concentrate
    result = overlap.weighted_overlap_test(B, E, n_perm=2000, seed=0)
    assert result["obs_stat"] == 0
    assert result["p_value"] < 0.05


# --- subject_overlap / group_overlap / centroid_distance --------------------


def test_subject_overlap_runs_end_to_end(beh_df, runmap_df, baselines_df):
    result = overlap.subject_overlap(beh_df, runmap_df, baselines_df, "MET001", 1, n_perm=200, seed=0)
    assert {"p_value", "obs_stat", "null_stats"}.issubset(result)
    assert 0 <= result["p_value"] <= 1


def test_group_overlap_pools_subjects_and_reports_n(beh_df, runmap_df, baselines_df, metadata_df):
    result = overlap.group_overlap(beh_df, runmap_df, baselines_df, metadata_df, 1, group="CTR", n_perm=200, seed=0)
    expected_n = len(analysis.subjects_in_group(metadata_df, 1, group="CTR"))
    assert result["n_subjects"] == expected_n
    assert expected_n > 1


def test_group_overlap_sub_ids_overrides_group_filter(beh_df, runmap_df, baselines_df, metadata_df):
    sub_ids = ["MET001", "MET002"]
    result = overlap.group_overlap(beh_df, runmap_df, baselines_df, metadata_df, 1, sub_ids=sub_ids, n_perm=100, seed=0)
    assert result["n_subjects"] == len(sub_ids)


def test_centroid_distance_returns_expected_keys(beh_df, runmap_df, baselines_df):
    result = overlap.centroid_distance(beh_df, runmap_df, baselines_df, "MET001", 1)
    assert set(result) == {"beh_red", "beh_green", "eeg_red", "eeg_green", "distance"}
    assert result["distance"] >= 0


# --- click_value_test (a genuinely different null model from weighted_overlap_test) ----


def test_click_value_test_obs_mean_matches_manual_formula(beh_df, runmap_df, baselines_df):
    sub_id, session = "MET001", 1
    B = overlap.behavioral_density_map(beh_df, [sub_id])
    E = analysis.mean_grid(runmap_df, baselines_df, sub_id, session, normalize=analysis.DEFAULT_NORMALIZE)
    result = overlap.click_value_test(B, E, n_perm=100, seed=0)
    expected_obs_mean = float(np.sum(E * B) / B.sum())
    assert result["obs_mean"] == pytest.approx(expected_obs_mean)


def test_click_value_test_is_reproducible_under_a_seed(beh_df, runmap_df, baselines_df):
    sub_id, session = "MET001", 1
    B = overlap.behavioral_density_map(beh_df, [sub_id])
    E = analysis.mean_grid(runmap_df, baselines_df, sub_id, session, normalize=analysis.DEFAULT_NORMALIZE)
    r1 = overlap.click_value_test(B, E, n_perm=200, seed=42)
    r2 = overlap.click_value_test(B, E, n_perm=200, seed=42)
    np.testing.assert_array_equal(r1["null_means"], r2["null_means"])


def test_click_value_test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        overlap.click_value_test(np.ones((10, 10)), np.ones((5, 5)))


def test_click_value_test_rejects_empty_behavioral_map():
    with pytest.raises(ValueError):
        overlap.click_value_test(np.zeros((10, 10)), np.ones((10, 10)))


def test_click_value_test_detects_a_known_relationship():
    """All clicks in a cell where E is 0, everywhere else E is 10 -- the
    null (uniform random cells) should almost never draw all-zero, so
    p should be small. Different null-construction from
    test_weighted_overlap_test_detects_a_known_overlap (random cells vs.
    toroidal shift) on the same synthetic B/E, as a cross-check the two
    methods agree."""
    B = np.zeros((10, 10))
    B[0, 0] = 100
    E = np.ones((10, 10)) * 10
    E[0, 0] = 0
    result = overlap.click_value_test(B, E, n_perm=2000, seed=0)
    assert result["obs_mean"] == 0
    assert result["p_value"] < 0.05


def test_subject_click_value_test_runs_end_to_end(beh_df, runmap_df, baselines_df):
    result = overlap.subject_click_value_test(beh_df, runmap_df, baselines_df, "MET001", 1, n_perm=200, seed=0)
    assert {"p_value", "obs_mean", "null_means"}.issubset(result)


def test_group_click_value_test_reports_n(beh_df, runmap_df, baselines_df, metadata_df):
    result = overlap.group_click_value_test(beh_df, runmap_df, baselines_df, metadata_df, 1, group="CTR", n_perm=200, seed=0)
    expected_n = len(analysis.subjects_in_group(metadata_df, 1, group="CTR"))
    assert result["n_subjects"] == expected_n


# --- correlation (individual-differences convergent validity) --------------


def test_subject_features_table_has_expected_columns(beh_df):
    table = correlation.subject_features_table(beh_df, session=1)
    expected_cols = {"sub_id", "group", "subgroup", "beh_red", "beh_green", "orientation_deg", "along_var", "perp_var", "eeg_red", "eeg_green", "ramp_slope_red", "ramp_slope_green", "ramp_intercept"}
    assert expected_cols.issubset(table.columns)
    assert len(table) > 0
    assert table["sub_id"].is_unique


def test_subject_features_table_only_includes_subjects_in_both_datasets(beh_df):
    table = correlation.subject_features_table(beh_df, session=1)
    troughs = pd.read_csv(correlation.SUBJECT_TROUGHS_PATH)
    eeg_sub_ids = set(troughs.loc[troughs["session"] == 1, "sub_id"])
    beh_sub_ids = set(beh_df["sub_id"])
    assert set(table["sub_id"]) == beh_sub_ids & eeg_sub_ids


def test_feature_correlations_detects_a_known_relationship():
    """Synthetic table with a perfect linear relationship between one beh
    and one eeg feature -- Spearman correlation should find it (r close to
    1, p small), everything else built from independent noise shouldn't
    reliably.."""
    rng = np.random.default_rng(0)
    n = 30
    beh_red = rng.normal(0, 1, n)
    table = pd.DataFrame(
        {
            "group": ["CTR"] * n,
            "subgroup": ["NA"] * n,
            "beh_red": beh_red,
            "beh_green": rng.normal(0, 1, n),
            "eeg_red": beh_red * 2 + 100,  # perfectly monotonic in beh_red
            "eeg_green": rng.normal(0, 1, n),
        }
    )
    result = correlation.feature_correlations(table, beh_features=["beh_red", "beh_green"], eeg_features=["eeg_red", "eeg_green"])
    matched = result[(result["beh_feature"] == "beh_red") & (result["eeg_feature"] == "eeg_red")].iloc[0]
    assert matched["r"] == pytest.approx(1.0)
    assert matched["p_value"] < 1e-6


def test_feature_correlations_filters_by_group(beh_df):
    table = correlation.subject_features_table(beh_df, session=1)
    result = correlation.feature_correlations(table, group="CTR")
    assert (result["group"] == "CTR").all()
    assert result["n"].iloc[0] == (table["group"] == "CTR").sum()


# --- correct_multiple_comparisons -------------------------------------------


def test_correct_multiple_comparisons_flags_a_strong_signal_among_noise():
    """One p-value far below the rest -- Holm should still flag it as
    significant even after correcting for the other 9 tests."""
    result = pd.DataFrame({"p_value": [0.0001] + [0.5] * 9})
    corrected = correlation.correct_multiple_comparisons(result, method="holm")
    assert corrected["significant"].iloc[0]
    assert not corrected["significant"].iloc[1:].any()
    assert (corrected["p_corrected"] >= corrected["p_value"]).all()


def test_correct_multiple_comparisons_fdr_is_less_conservative_than_holm():
    result = pd.DataFrame({"p_value": [0.001, 0.01, 0.02, 0.03, 0.04, 0.5, 0.6, 0.7, 0.8, 0.9]})
    holm = correlation.correct_multiple_comparisons(result, method="holm")
    fdr = correlation.correct_multiple_comparisons(result, method="fdr_bh")
    assert fdr["significant"].sum() >= holm["significant"].sum()


def test_correct_multiple_comparisons_none_of_the_real_pooled_pairs_survive(beh_df):
    """Pins the actual finding this analysis produced: with 43 subjects and
    25 pairwise tests, nothing clears Holm or FDR correction -- the
    individual-differences correlation is real-looking but not yet
    statistically supported, per docs/ssvepbeh_reliability_gaps.md."""
    table = correlation.subject_features_table(beh_df, session=1)
    pooled = correlation.feature_correlations(table)
    corrected = correlation.correct_multiple_comparisons(pooled, method="fdr_bh")
    assert not corrected["significant"].any()


# --- session_reliability -----------------------------------------------------


def test_paired_subjects_matches_known_counts():
    """Pins the real paired-subject counts behind this project's reliability
    gap -- protan/deutan/CVD(combined) are all too thin to assess."""
    assert len(session_reliability.paired_subjects()) == 19
    assert len(session_reliability.paired_subjects(group="CTR")) == 13
    assert len(session_reliability.paired_subjects(group="PD")) == 4
    assert len(session_reliability.paired_subjects(group="CVD")) == 2
    assert len(session_reliability.paired_subjects(group="CVD", subgroup="protan")) == 2
    assert len(session_reliability.paired_subjects(group="CVD", subgroup="deutan")) == 0


def test_session_overlap_comparison_raises_below_minimum(beh_df, runmap_df, baselines_df, metadata_df):
    with pytest.raises(ValueError):
        session_reliability.session_overlap_comparison(beh_df, runmap_df, baselines_df, metadata_df, group="CVD", subgroup="deutan")


def test_session_correlation_comparison_raises_below_minimum(beh_df):
    with pytest.raises(ValueError):
        session_reliability.session_correlation_comparison(beh_df, group="CVD", subgroup="deutan")


def test_session_overlap_comparison_returns_one_row_per_session(beh_df, runmap_df, baselines_df, metadata_df):
    result = session_reliability.session_overlap_comparison(beh_df, runmap_df, baselines_df, metadata_df, group="CTR", n_perm=200, seed=0)
    assert list(result["session"]) == [1, 2]
    assert (result["n"] == 13).all()


def test_session_correlation_comparison_returns_both_sessions(beh_df):
    result = session_reliability.session_correlation_comparison(beh_df, group="CTR")
    assert set(result["session"]) == {1, 2}
    assert len(result) == 2 * len(correlation.DEFAULT_BEH_FEATURES) * len(correlation.DEFAULT_EEG_FEATURES)


# --- plotting -----------------------------------------------------------


def test_plot_overlap_returns_two_panels(beh_df, runmap_df, baselines_df):
    E = analysis.mean_grid(runmap_df, baselines_df, "MET001", 1, normalize=analysis.DEFAULT_NORMALIZE)
    fig = plotting.plot_overlap(beh_df, E, ["MET001"])
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == ["EEG response", "Behavioral clicks (n=30)"]


def test_plot_grid_with_clicks_draws_one_scatter_series_and_caps_the_view(beh_df, runmap_df, baselines_df):
    E = analysis.mean_grid(runmap_df, baselines_df, "MET001", 1, normalize=analysis.DEFAULT_NORMALIZE)
    clicks = beh_df[beh_df["sub_id"] == "MET001"]
    ax = plotting.plot_grid_with_clicks(E, clicks)
    assert len(ax.collections) == 1  # the click scatter (imshow isn't a "collection")
    assert ax.collections[0].get_offsets().shape == (len(clicks), 2)
    assert ax.get_xlim() == (0.0, 3200.0)
    assert ax.get_ylim() == (0.0, 2000.0)


def test_plot_grid_with_clicks_accepts_a_filtered_subset(beh_df, runmap_df, baselines_df):
    """Passing fewer rows (e.g. an outlier-filtered subset) must plot fewer
    points, not the full subject's clicks."""
    E = analysis.mean_grid(runmap_df, baselines_df, "MET001", 1, normalize=analysis.DEFAULT_NORMALIZE)
    clicks = beh_df[beh_df["sub_id"] == "MET001"]
    filtered = clicks.iloc[:5]
    ax = plotting.plot_grid_with_clicks(E, filtered)
    assert ax.collections[0].get_offsets().shape == (5, 2)


def test_plot_grid_with_clicks_respects_size_alpha_and_cmap(beh_df, runmap_df, baselines_df):
    E = analysis.mean_grid(runmap_df, baselines_df, "MET001", 1, normalize=analysis.DEFAULT_NORMALIZE)
    clicks = beh_df[beh_df["sub_id"] == "MET001"]
    ax = plotting.plot_grid_with_clicks(E, clicks, s=8, alpha=0.4, cmap="plasma")
    assert ax.collections[0].get_sizes()[0] == 8
    assert ax.collections[0].get_alpha() == 0.4
    assert ax.images[0].get_cmap().name == "plasma"


def test_plot_grid_with_clicks_vmin_vmax_override_autoscale(beh_df, runmap_df, baselines_df):
    E = analysis.mean_grid(runmap_df, baselines_df, "MET001", 1, normalize=analysis.DEFAULT_NORMALIZE)
    clicks = beh_df[beh_df["sub_id"] == "MET001"]
    ax = plotting.plot_grid_with_clicks(E, clicks, vmin=-5, vmax=5)
    assert ax.images[0].get_clim() == (-5, 5)


# --- multi-panel click overlay (dashboard M2) --------------------------------


def test_plot_grids_with_clicks_draws_one_panel_per_grid(beh_df, runmap_df, baselines_df):
    sub_ids = ["MET001", "MET002", "MET003"]
    grids = [analysis.mean_grid(runmap_df, baselines_df, sub_id, 1, normalize=analysis.DEFAULT_NORMALIZE) for sub_id in sub_ids]
    clicks_dfs = [beh_df[beh_df["sub_id"] == sub_id] for sub_id in sub_ids]
    fig = plotting.plot_grids_with_clicks(grids, clicks_dfs, sub_ids)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == sub_ids


def test_plot_grids_with_clicks_shares_one_color_scale_across_panels(beh_df, runmap_df, baselines_df):
    sub_ids = ["MET001", "MET002"]
    grids = [analysis.mean_grid(runmap_df, baselines_df, sub_id, 1, normalize=analysis.DEFAULT_NORMALIZE) for sub_id in sub_ids]
    clicks_dfs = [beh_df[beh_df["sub_id"] == sub_id] for sub_id in sub_ids]
    fig = plotting.plot_grids_with_clicks(grids, clicks_dfs, sub_ids, diverging=True)
    heatmap_axes = [ax for ax in fig.axes if ax.get_title()]
    clims = [ax.images[0].get_clim() for ax in heatmap_axes]
    assert clims[0] == clims[1]


def test_plot_grids_with_clicks_rejects_mismatched_lengths(beh_df, runmap_df, baselines_df):
    grid = analysis.mean_grid(runmap_df, baselines_df, "MET001", 1, normalize=analysis.DEFAULT_NORMALIZE)
    clicks = beh_df[beh_df["sub_id"] == "MET001"]
    with pytest.raises(ValueError):
        plotting.plot_grids_with_clicks([grid, grid], [clicks], ["a", "b"])
