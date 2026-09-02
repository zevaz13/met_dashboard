"""Regression tests for the SSVEP pipeline.

Run: uv run pytest ssveps/tests -q

Deliberately small. Each test pins an invariant that was either got wrong once
already, or would be silent if it broke.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

# beh/scripts/ has its own loader.py/plotting.py -- if a combined run (e.g.
# bare `pytest` at the repo root) collected that test module first, sys.modules
# would already hold beh's versions under these bare names. Drop them so the
# imports below re-resolve against SCRIPTS, just inserted at sys.path[0],
# regardless of what ran before this module in the same pytest session.
for _name in ("loader", "plotting"):
    sys.modules.pop(_name, None)

import analysis  # noqa: E402
import plotting  # noqa: E402
import permutation  # noqa: E402
import variance  # noqa: E402
import reliability  # noqa: E402
import pca  # noqa: E402
from loader import load_ssvep, to_rows  # noqa: E402

RAW_DIR = Path("/home/sebas/data/ssveps")


@pytest.fixture(scope="module")
def data():
    return analysis.load_runmap(), analysis.load_baselines(), analysis.load_metadata()


@pytest.fixture(scope="module")
def axes():
    return analysis.load_grid_axes()


# --- axis orientation ------------------------------------------------------
# The one that was wrong. ssveps/CTRdata.png is an independently produced
# reference for this exact group; its minimum sits at red 2133 / green 889.


def test_ctr_trough_matches_reference_image(data, axes):
    runmap, baselines, meta = data
    red_vals, green_vals = axes
    grid = analysis.group_grid(runmap, baselines, meta, 1, group="CTR", normalize=analysis.DEFAULT_NORMALIZE)
    loc = analysis.trough_location(grid, red_vals, green_vals)
    assert loc["red"] == pytest.approx(2133.3, abs=1)
    assert loc["green"] == pytest.approx(888.9, abs=1)


def test_grid_is_indexed_red_then_green(data, axes):
    """Axis 0 is red, axis 1 is green -- the naming every docstring assumes."""
    runmap, baselines, meta = data
    red_vals, green_vals = axes
    grid = analysis.group_grid(runmap, baselines, meta, 1, group="CTR", normalize=analysis.DEFAULT_NORMALIZE)
    red_idx, green_idx = np.unravel_index(np.argmin(grid), grid.shape)
    assert red_vals[red_idx] == pytest.approx(2133.3, abs=1)
    assert green_vals[green_idx] == pytest.approx(888.9, abs=1)
    # SSVEP amplitude falls off with red: averaging over green, the red=0 edge
    # must be the brightest. This is direction-sensitive, unlike a correlation.
    raw = analysis.group_grid(runmap, baselines, meta, 1, group="CTR")
    assert raw.mean(axis=1)[0] == raw.mean(axis=1).max()


def test_heatmap_puts_red_on_x(data):
    """_plot_heatmap transposes for display, so the image's x axis is red."""
    runmap, baselines, meta = data
    grid = analysis.group_grid(runmap, baselines, meta, 1, group="CTR", normalize=analysis.DEFAULT_NORMALIZE)
    ax = plotting.plot_mean_across_subjects(runmap, baselines, meta, 1, group="CTR",
                                            normalize=analysis.DEFAULT_NORMALIZE)
    displayed = ax.images[0].get_array()
    assert displayed.shape == grid.T.shape
    np.testing.assert_allclose(displayed, grid.T)
    assert ax.get_xlabel() == "red" and ax.get_ylabel() == "green"


def test_loader_reads_runmap_green_first():
    """runMap's own axes are (green, red, run) despite mapDIM saying otherwise."""
    d = load_ssvep(str(RAW_DIR / "MET000.mat"))
    _, runmap_rows, _ = to_rows(d, "MET000.mat")
    row = next(r for r in runmap_rows if r["red_idx"] == 6 and r["green_idx"] == 4 and r["run"] == 1)
    assert row["value"] == d["runMap"][4, 6, 0]


# --- normalization ---------------------------------------------------------
# Must stay identical to templateCode/ICCs/computeICC_gridMaps.m lines 27-32.


def test_percent_normalization_matches_matlab_template(data):
    runmap, baselines, _ = data
    raw = analysis.raw_grid(runmap, "MET000", 1, 1)
    base = analysis.baseline_values(baselines, "MET000", 1, scope="run", run=1, trials="all")
    got = analysis.normalized_grid(runmap, baselines, "MET000", 1, 1, scope="run", trials="all", method="percent")
    np.testing.assert_allclose(got, (raw - base.mean()) / base.mean())


def test_mean_grid_normalizes_per_run_then_averages(data):
    """Order matters: normalize each run, then average -- not the reverse."""
    runmap, baselines, _ = data
    per_run = [
        analysis.normalized_grid(runmap, baselines, "MET000", 1, run, **analysis.DEFAULT_NORMALIZE)
        for run in (1, 2, 3, 4)
    ]
    got = analysis.mean_grid(runmap, baselines, "MET000", 1, normalize=analysis.DEFAULT_NORMALIZE)
    np.testing.assert_allclose(got, np.mean(per_run, axis=0))


# --- ragged 3-run subjects -------------------------------------------------


def test_ragged_three_run_subject(data):
    runmap, baselines, _ = data
    assert sorted(runmap.query("sub_id == 'MET037' and session == 1")["run"].unique()) == [1, 2, 3]
    assert analysis.flatten_runs(runmap, baselines, "MET037", 1).shape == (300,)
    assert analysis.flatten_runs(runmap, baselines, "MET000", 1).shape == (400,)
    assert analysis.mean_grid(runmap, baselines, "MET037", 1).shape == (10, 10)


# --- trough surface fit ---


def test_ramp_gaussian_converges_on_every_subject(data, axes):
    """The reason it replaced the paraboloid: it converges on all 62 rows.
    Converging is not the same as locating a dip -- see at_bound below."""
    runmap, baselines, meta = data
    red_vals, green_vals = axes
    for row in meta.to_dict("records"):
        grid = analysis.mean_grid(runmap, baselines, row["sub_id"], row["session"],
                                  normalize=analysis.DEFAULT_NORMALIZE)
        assert not np.isnan(analysis.fit_ramp_gaussian(grid, red_vals, green_vals)["r_squared"])


def test_at_bound_fit_is_not_reported_valid(axes):
    """A monotonic ramp has no interior dip; the fit pegs at an edge and must
    say so. This is the protan case -- their trough sits beyond max(red)."""
    red_vals, green_vals = axes
    x, y = np.meshgrid(red_vals, green_vals, indexing="ij")
    grid = 2.0 - 0.0004 * x  # pure ramp, no dip anywhere in range
    fit = analysis.fit_ramp_gaussian(grid, red_vals, green_vals)
    assert fit["at_bound"]
    assert not fit["fit_valid"]


def test_ramp_gaussian_recovers_a_known_dip(axes):
    """Fit a surface we constructed, and check it finds the planted dip."""
    red_vals, green_vals = axes
    x, y = np.meshgrid(red_vals, green_vals, indexing="ij")
    truth = dict(x0=2000.0, y0=900.0, amp=0.5, sx=600.0, sy=400.0)
    grid = (1.5 - 0.0001 * x - 0.00005 * y
            - truth["amp"] * np.exp(-(((x - truth["x0"]) ** 2) / (2 * truth["sx"] ** 2)
                                      + ((y - truth["y0"]) ** 2) / (2 * truth["sy"] ** 2))))
    fit = analysis.fit_ramp_gaussian(grid, red_vals, green_vals)
    assert fit["fit_valid"]
    assert fit["red"] == pytest.approx(truth["x0"], abs=60)
    assert fit["green"] == pytest.approx(truth["y0"], abs=60)
    assert fit["amp"] == pytest.approx(truth["amp"], rel=0.1)
    assert fit["r_squared"] > 0.999


def test_fit_trough_surface_methods_share_a_schema(axes):
    """Every method returns the same keys, so subject_troughs' columns are stable."""
    red_vals, green_vals = axes
    x, y = np.meshgrid(red_vals, green_vals, indexing="ij")
    grid = (x - 1600.0) ** 2 / 1e6 + (y - 1000.0) ** 2 / 1e6
    keys = {"red", "green", "depth", "amp", "sigma_red", "sigma_green", "r_squared", "at_bound", "fit_valid"}
    for method in ("ramp_gaussian", "paraboloid", "gaussian"):
        assert set(analysis.fit_trough_surface(grid, red_vals, green_vals, method=method)) == keys


# --- rotated dip fit (M11) --------------------------------------------------


def test_rotated_gaussian_converges_on_every_subject(data, axes):
    runmap, baselines, meta = data
    red_vals, green_vals = axes
    for row in meta.to_dict("records"):
        grid = analysis.mean_grid(runmap, baselines, row["sub_id"], row["session"],
                                  normalize=analysis.DEFAULT_NORMALIZE)
        assert not np.isnan(analysis.fit_rotated_gaussian(grid, red_vals, green_vals)["r_squared"])


def test_rotated_gaussian_at_bound_on_pure_ramp(axes):
    """Same pure-ramp case as fit_ramp_gaussian's own at_bound test: no
    interior dip anywhere in range, so the fit must peg and say so."""
    red_vals, green_vals = axes
    x, y = np.meshgrid(red_vals, green_vals, indexing="ij")
    grid = 2.0 - 0.0004 * x
    fit = analysis.fit_rotated_gaussian(grid, red_vals, green_vals)
    assert fit["at_bound"]
    assert not fit["fit_valid"]


def test_rotated_gaussian_recovers_a_known_tilted_dip(axes):
    """The point of this model over fit_ramp_gaussian: plant a dip whose
    major axis is neither the red nor the green axis, and check the fit
    actually recovers the tilt, not just location/amplitude."""
    red_vals, green_vals = axes
    x, y = np.meshgrid(red_vals, green_vals, indexing="ij")
    truth = dict(x0=2000.0, y0=900.0, amp=0.5, sigma_major=700.0, sigma_minor=300.0, theta=np.radians(30.0))
    dx, dy = x - truth["x0"], y - truth["y0"]
    xr = dx * np.cos(truth["theta"]) + dy * np.sin(truth["theta"])
    yr = -dx * np.sin(truth["theta"]) + dy * np.cos(truth["theta"])
    grid = (1.5 - 0.0001 * x - 0.00005 * y
            - truth["amp"] * np.exp(-(xr**2 / (2 * truth["sigma_major"] ** 2) + yr**2 / (2 * truth["sigma_minor"] ** 2))))
    fit = analysis.fit_rotated_gaussian(grid, red_vals, green_vals)
    assert fit["fit_valid"]
    assert fit["red"] == pytest.approx(truth["x0"], abs=60)
    assert fit["green"] == pytest.approx(truth["y0"], abs=60)
    assert fit["amp"] == pytest.approx(truth["amp"], rel=0.1)
    assert fit["sigma_major"] == pytest.approx(truth["sigma_major"], rel=0.15)
    assert fit["sigma_minor"] == pytest.approx(truth["sigma_minor"], rel=0.15)
    assert fit["orientation_deg"] == pytest.approx(30.0, abs=5)
    assert fit["r_squared"] > 0.999


def test_rotated_gaussian_canonicalizes_major_as_the_longer_axis(axes):
    """An axis-aligned dip elongated along green (sigma_green > sigma_red,
    i.e. true tilt = 90deg) should still come back with sigma_major as the
    longer one and orientation folded into [0, 180), not left at 90+90."""
    red_vals, green_vals = axes
    x, y = np.meshgrid(red_vals, green_vals, indexing="ij")
    truth = dict(x0=2000.0, y0=900.0, amp=0.5, sx=250.0, sy=650.0)
    grid = (1.5 - 0.0001 * x - 0.00005 * y
            - truth["amp"] * np.exp(-(((x - truth["x0"]) ** 2) / (2 * truth["sx"] ** 2)
                                      + ((y - truth["y0"]) ** 2) / (2 * truth["sy"] ** 2))))
    fit = analysis.fit_rotated_gaussian(grid, red_vals, green_vals)
    assert fit["fit_valid"]
    assert fit["sigma_major"] > fit["sigma_minor"]
    assert fit["sigma_major"] == pytest.approx(truth["sy"], rel=0.15)
    assert fit["orientation_deg"] == pytest.approx(90.0, abs=5)


def test_subject_troughs_adds_rotated_columns_without_touching_existing_ones(data):
    """Purely additive: every fitted_*/ramp_* number stays exactly what
    build_troughs.py already persisted, and the new rotated_* columns show
    up alongside them."""
    runmap, baselines, meta = data
    one_subject = meta[(meta["sub_id"] == "MET000") & (meta["session"] == 1)]
    without_rotated_cols = {
        "sub_id", "session", "group", "subgroup", "red", "green", "depth", "red_idx", "green_idx",
        "fitted_red", "fitted_green", "fitted_depth", "fitted_amp", "fitted_sigma_red", "fitted_sigma_green",
        "fitted_r_squared", "fitted_at_bound", "fitted_valid",
        "ramp_intercept", "ramp_slope_red", "ramp_slope_green", "ramp_r_squared",
    }
    rotated_cols = {
        "rotated_red", "rotated_green", "rotated_depth", "rotated_amp", "rotated_sigma_major",
        "rotated_sigma_minor", "rotated_orientation_deg", "rotated_r_squared", "rotated_at_bound", "rotated_valid",
    }
    troughs = analysis.subject_troughs(runmap, baselines, one_subject)
    assert set(troughs.columns) == without_rotated_cols | rotated_cols


# --- ramp-only fit, extrapolation, bootstrap CI (M6) ------------------------


def test_fit_ramp_recovers_a_known_slope(axes):
    """Pure linear surface, no dip -- fit_ramp should recover it almost exactly
    (closed-form least squares on noiseless data)."""
    red_vals, green_vals = axes
    x, y = np.meshgrid(red_vals, green_vals, indexing="ij")
    grid = 2.0 - 0.0003 * x + 0.0001 * y
    fit = analysis.fit_ramp(grid, red_vals, green_vals)
    assert fit["slope_red"] == pytest.approx(-0.0003, abs=1e-9)
    assert fit["slope_green"] == pytest.approx(0.0001, abs=1e-9)
    assert fit["r_squared"] > 0.999


def test_fit_ramp_defined_even_when_ramp_gaussian_pegs(axes):
    """The whole point of fit_ramp for M6: it has no interior minimum to fail
    to find, so it stays usable on exactly the pure-ramp case where
    fit_ramp_gaussian pegs at_bound (see test_at_bound_fit_is_not_reported_valid)."""
    red_vals, green_vals = axes
    x, _ = np.meshgrid(red_vals, green_vals, indexing="ij")
    grid = 2.0 - 0.0004 * x
    assert analysis.fit_ramp_gaussian(grid, red_vals, green_vals)["at_bound"]
    fit = analysis.fit_ramp(grid, red_vals, green_vals)
    assert fit["slope_red"] == pytest.approx(-0.0004, abs=1e-9)
    assert not np.isnan(fit["r_squared"])


def test_extrapolate_ramp_crossing_solves_the_linear_equation():
    ramp = {"intercept": 10.0, "slope_red": -0.01, "slope_green": 0.0}
    red = analysis.extrapolate_ramp_crossing(ramp, target_depth=5.0, green_ref=0.0)
    assert red == pytest.approx(500.0)
    assert 10.0 + -0.01 * red == pytest.approx(5.0)


def test_bootstrap_ci_is_tight_around_a_constant():
    lo, hi = analysis.bootstrap_ci(lambda rng: 3.0, n_boot=200)
    assert lo == pytest.approx(3.0)
    assert hi == pytest.approx(3.0)


def test_bootstrap_ci_brackets_the_resampled_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng_master = np.random.default_rng(0)

    def replicate(rng):
        return rng.choice(values, size=len(values), replace=True).mean()

    lo, hi = analysis.bootstrap_ci(replicate, n_boot=2000, seed=0)
    assert lo < values.mean() < hi
    del rng_master  # unused, just documents replicate_fn's own rng is independent of any caller state


def test_subject_troughs_ramp_columns_are_never_nan(data):
    """Unlike fitted_*, the ramp-only columns must be defined for every
    subject regardless of fitted_at_bound/fitted_valid -- that's what lets
    ramp_slope_red be used for all 15 CVD subjects instead of just the ones
    with an interior trough (M6)."""
    runmap, baselines, meta = data
    subset = meta[meta["sub_id"].isin(["MET015", "MET016", "MET000"])]
    df = analysis.subject_troughs(runmap, baselines, subset)
    for col in ("ramp_intercept", "ramp_slope_red", "ramp_slope_green", "ramp_r_squared"):
        assert col in df.columns
        assert not df[col].isna().any()


def test_pooled_baseline_values_pools_across_subjects_and_runs(data):
    _, baselines, _ = data
    one = analysis.baseline_values(baselines, "MET000", 1, scope="session", trials="all")
    pooled = analysis.pooled_baseline_values(baselines, ["MET000"], 1)
    np.testing.assert_array_equal(pooled, one)
    two = analysis.pooled_baseline_values(baselines, ["MET000", "MET001"], 1)
    assert len(two) == len(one) + len(analysis.baseline_values(baselines, "MET001", 1, scope="session", trials="all"))


# --- permutation testing ---------------------------------------------------


def test_clusters_use_8_connectivity():
    """MATLAB bwconncomp's 2D default is 8-connected; scipy's is 4."""
    zmap = np.zeros((10, 10))
    zmap[2, 2] = zmap[3, 3] = 5.0  # diagonal neighbours: one cluster, not two
    assert len(permutation._clusters(zmap, 1.96)) == 1


def test_permutation_keeps_every_subject(data):
    """n1/n2 default to the full group sizes -- no subject is discarded."""
    runmap, baselines, meta = data
    result = permutation.permutation_test_size(runmap, baselines, meta, 1, group1="PD", group2="CTR",
                                               n_perm=50, seed=0)
    assert result["n1"] == len(analysis.subjects_in_group(meta, 1, group="PD"))
    assert result["n2"] == len(analysis.subjects_in_group(meta, 1, group="CTR"))


def test_permutation_is_reproducible_under_a_seed(data):
    runmap, baselines, meta = data
    kw = dict(group1="PD", group2="CTR", n_perm=50, seed=7)
    a = permutation.permutation_test_size(runmap, baselines, meta, 1, **kw)
    b = permutation.permutation_test_size(runmap, baselines, meta, 1, **kw)
    np.testing.assert_array_equal(a["zdiff"], b["zdiff"])


# --- gain/shape decomposition (M8) ------------------------------------------


def test_fit_gain_shape_recovers_a_known_gain():
    template = np.array([[1.0, 2.0], [3.0, 4.0]])
    grid = 0.5 * template + 0.1  # pure gain change, no shape change
    fit = analysis.fit_gain_shape(grid, template)
    assert fit["gain"] == pytest.approx(0.5, abs=1e-9)
    assert fit["intercept"] == pytest.approx(0.1, abs=1e-9)
    assert fit["r_squared"] > 0.999
    np.testing.assert_allclose(fit["residual"], 0.0, atol=1e-9)


def test_fit_gain_shape_isolates_a_localized_shape_change():
    template = np.ones((10, 10))
    grid = template.copy()
    grid[4, 5] -= 1.0  # one cell selectively deeper -- not explainable by a uniform gain
    fit = analysis.fit_gain_shape(grid, template)
    residual = fit["residual"]
    assert residual[4, 5] == pytest.approx(residual.min())
    assert abs(residual[4, 5]) > 5 * np.abs(np.delete(residual.ravel(), 4 * 10 + 5)).mean()


def test_trough_region_residual_separates_local_from_global():
    residual = np.zeros((10, 10))
    residual[4:7, 3:6] = -1.0  # a 3x3 deficit block
    result = analysis.trough_region_residual(residual, red_idx=5, green_idx=4, half_width=1)
    assert result["trough_region"] == pytest.approx(-1.0)
    assert result["rest_of_grid"] == pytest.approx(0.0)


# --- variance components (M7) -----------------------------------------------


def test_fit_components_recovers_known_within_and_between_sd():
    """Simulate subjects with a known between-subject SD and known
    within-subject (run) SD, and check the MixedLM recovers both."""
    rng = np.random.default_rng(0)
    true_within, true_between, true_mean = 0.15, 0.35, 1.0
    n_subjects, n_runs = 40, 4
    subject_means = rng.normal(true_mean, true_between, size=n_subjects)
    values_by_subject = {
        f"S{i}": subject_means[i] + rng.normal(0, true_within, size=n_runs) for i in range(n_subjects)
    }
    within_sd, between_sd = variance._fit_components(values_by_subject)
    assert within_sd == pytest.approx(true_within, rel=0.25)
    assert between_sd == pytest.approx(true_between, rel=0.25)


def test_variance_components_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    values_by_subject = {f"S{i}": rng.normal(1.0, 0.3, size=4) + rng.normal(0, 0.1) for i in range(15)}
    result = variance.variance_components(values_by_subject, n_boot=100, seed=0)
    assert result["within_ci"][0] <= result["within_sd"] <= result["within_ci"][1]
    assert result["between_ci"][0] <= result["between_sd"] <= result["between_ci"][1]
    assert result["n_subjects"] == 15
    assert result["n_boot_used"] <= 100


def test_within_subject_cv_is_scale_invariant():
    """CV = SD/|mean|, so scaling every value by a constant leaves it unchanged."""
    values_by_subject = {"S0": np.array([1.0, 1.2, 0.9, 1.1]), "S1": np.array([2.0, 2.4, 1.8, 2.2])}
    cv = variance.within_subject_cv(values_by_subject)
    assert cv["S0"] == pytest.approx(cv["S1"])


def test_group_run_values_matches_run_mean_values(data):
    runmap, baselines, meta = data
    values = variance.group_run_values(runmap, baselines, meta, 1, group="PD")
    expected_sub_ids = set(analysis.subjects_in_group(meta, 1, group="PD"))
    assert set(values) == expected_sub_ids
    for sub_id, vals in values.items():
        np.testing.assert_array_equal(
            vals, analysis.run_mean_values(runmap, baselines, sub_id, 1, normalize=analysis.DEFAULT_NORMALIZE)
        )


# --- feature reliability, minimum detectable effect (M9) --------------------


def test_feature_icc_matches_icc_grid_on_the_same_pixel(data):
    """feature_icc and icc_grid share _icc_a1 -- confirm the refactor didn't
    change icc_grid's own numbers by checking they agree on one pixel."""
    runmap, baselines, meta = data
    sub_ids = reliability.paired_subjects(meta, group="CTR")
    icc_df = reliability.icc_grid(runmap, baselines, sub_ids)
    row = icc_df.iloc[0]
    red_idx, green_idx = int(row["red_idx"]), int(row["green_idx"])
    values1, values2 = reliability.session_pair_values(runmap, baselines, sub_ids, red_idx, green_idx)
    result = reliability.feature_icc(values1, values2)
    assert result["icc"] == pytest.approx(row["icc"])
    assert result["pval"] == pytest.approx(row["pval"])


def test_feature_icc_is_high_for_a_near_identical_repeat():
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 1, size=20)
    values1 = truth + rng.normal(0, 0.01, size=20)
    values2 = truth + rng.normal(0, 0.01, size=20)
    result = reliability.feature_icc(values1, values2)
    assert result["icc"] > 0.99


def test_feature_icc_is_low_for_independent_noise():
    rng = np.random.default_rng(1)
    values1 = rng.normal(0, 1, size=30)
    values2 = rng.normal(0, 1, size=30)
    result = reliability.feature_icc(values1, values2)
    assert result["icc"] < 0.3


def test_minimum_detectable_effect_worsens_as_reliability_drops():
    """A noisier measure (lower ICC) needs a bigger true effect to detect at
    the same n and power -- MDE should increase monotonically as ICC falls."""
    mde_perfect = reliability.minimum_detectable_effect(10, 10, icc=1.0)
    mde_half = reliability.minimum_detectable_effect(10, 10, icc=0.5)
    mde_low = reliability.minimum_detectable_effect(10, 10, icc=0.1)
    assert mde_perfect < mde_half < mde_low
    # icc=1.0 is exactly the textbook two-sample MDE formula, no attenuation.
    assert mde_perfect == pytest.approx((norm.ppf(0.975) + norm.ppf(0.8)) * np.sqrt(1 / 10 + 1 / 10))


def test_minimum_detectable_effect_shrinks_with_more_subjects():
    mde_small_n = reliability.minimum_detectable_effect(6, 6, icc=0.7)
    mde_large_n = reliability.minimum_detectable_effect(30, 30, icc=0.7)
    assert mde_large_n < mde_small_n


# --- PCA (M10) ---------------------------------------------------------------


def test_fit_pca_recovers_a_known_low_rank_structure():
    """Data built from one strong direction plus tiny noise -- PC1 should
    capture almost all the variance and its scores should recover the
    generating signal (up to sign, which SVD leaves arbitrary)."""
    rng = np.random.default_rng(0)
    n_subjects, n_features = 40, 100
    direction = rng.normal(size=n_features)
    direction /= np.linalg.norm(direction)
    signal = rng.normal(0, 5, size=n_subjects)
    X = np.outer(signal, direction) + rng.normal(0, 0.05, size=(n_subjects, n_features))

    result = pca.fit_pca(X)
    assert result["explained_variance_ratio"][0] > 0.95
    correlation = abs(np.corrcoef(result["scores"][:, 0], signal)[0, 1])
    assert correlation > 0.99


def test_permutation_component_count_finds_nothing_in_pure_noise():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 100))
    result = pca.permutation_component_count(X, n_perm=200, seed=0)
    assert result["n_components_real"] == 0


def test_permutation_component_count_finds_an_injected_signal():
    rng = np.random.default_rng(2)
    n_subjects, n_features = 40, 100
    direction = rng.normal(size=n_features)
    direction /= np.linalg.norm(direction)
    signal = rng.normal(0, 8, size=n_subjects)
    X = np.outer(signal, direction) + rng.normal(0, 1, size=(n_subjects, n_features))
    result = pca.permutation_component_count(X, n_perm=200, seed=0)
    assert result["n_components_real"] >= 1


def test_pixel_matrix_matches_mean_grid(data):
    runmap, baselines, meta = data
    meta_out, X = pca.pixel_matrix(runmap, baselines, meta, 1)
    assert X.shape == (len(meta_out), 100)
    row = meta_out.iloc[0]
    expected = analysis.mean_grid(runmap, baselines, row["sub_id"], 1, normalize=analysis.DEFAULT_NORMALIZE).ravel()
    np.testing.assert_allclose(X[0], expected)


# --- figure layout ---------------------------------------------------------


@pytest.mark.parametrize("n,expected_rows,expected_cols", [(3, 1, 3), (5, 1, 5), (6, 2, 5), (21, 5, 5)])
def test_panels_wrap_at_max_panel_cols(n, expected_rows, expected_cols):
    fig, axs = plotting._multi_panel_figure(n)
    rows, cols = axs[0].get_subplotspec().get_gridspec().get_geometry()
    assert (rows, cols) == (expected_rows, expected_cols)
    assert len(axs) == n
