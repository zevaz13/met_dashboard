"""Regression tests for the FM100 pipeline.

Run: uv run pytest standardizedScores/FM100/tests -q
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

# beh/scripts/ and ssveps/scripts/ each have their own loader.py/plotting.py
# under the same bare names -- if a combined pytest session collected one of
# those test modules first, sys.modules would hold their versions. Drop them
# so the imports below re-resolve against SCRIPTS (see beh/README.md's Tests
# section for the same issue there).
for _name in ("loader", "plotting", "scores", "comparisons"):
    sys.modules.pop(_name, None)

import comparisons  # noqa: E402
import loader  # noqa: E402
import plotting  # noqa: E402
import scores  # noqa: E402

TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "templateCode" / "FM100.py"


def _load_template():
    spec = importlib.util.spec_from_file_location("template_fm100", TEMPLATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def df():
    return loader.load_fm100_raw()


@pytest.fixture(scope="module")
def template():
    return _load_template()


# --- loader ----------------------------------------------------------------


def test_load_fm100_raw_has_expected_shape_and_columns(df):
    assert list(df.columns) == ["sub_id", "session", "group", "subgroup", "sex", "date", "caps"]
    assert len(df) == 69  # 70 raw rows minus the byte-identical duplicate MET000 row


def test_duplicate_first_row_is_dropped(df):
    """The raw file's first line is a byte-identical duplicate of the
    second (MET000) row -- pin that only one MET000/session-1 row survives."""
    assert len(df[(df["sub_id"] == "MET000") & (df["session"] == 1)]) == 1


def test_session_suffix_parsing(df):
    met000_sessions = sorted(df.loc[df["sub_id"] == "MET000", "session"])
    assert met000_sessions == [1, 2, 3]  # MET000, MET000b, MET000c


def test_no_duplicate_subject_session_pairs(df):
    """Every (sub_id, session) pair must be unique -- a raw Reference field
    missing its b/c suffix (as MET039's repeat visit briefly was) collapses
    two real sessions into one, which _subject_profile then rejects."""
    counts = df.groupby(["sub_id", "session"]).size()
    dupes = counts[counts > 1]
    assert dupes.empty, f"duplicate (sub_id, session) rows: {dupes.to_dict()}"


def test_caps_are_a_permutation_of_1_to_85(df):
    for caps in df["caps"]:
        assert caps.shape == (85,)
        np.testing.assert_array_equal(np.sort(caps), np.arange(1, 86))


def test_group_matches_ssveps_metadata_for_every_overlapping_subject(df):
    ssvep_meta = pd.read_csv(loader.SSVEP_METADATA_PATH, keep_default_na=False)
    overlap = ssvep_meta.drop_duplicates("sub_id").set_index("sub_id")["group"]
    checked = 0
    for sub_id, group in df[["sub_id", "group"]].drop_duplicates().itertuples(index=False):
        if sub_id in overlap.index:
            assert group == overlap[sub_id]
            checked += 1
    assert checked > 0


def test_met047_is_unknown_and_met021_keeps_its_ssvep_label(df):
    """MET047 has no SSVEP/behavioral data at all (new, unallocated).
    MET021 is labeled CTR in ssveps/files/metadata.csv -- trusted as-is
    here, per project decision, despite being flagged as a possibly
    different (non-red-green) deficiency type."""
    met047 = df[df["sub_id"] == "MET047"]
    assert len(met047) == 1
    assert (met047["group"] == "UNKNOWN").all() and (met047["subgroup"] == "NA").all()

    met021 = df[df["sub_id"] == "MET021"]
    assert len(met021) > 0
    assert (met021["group"] == "CTR").all()


def test_group_falls_back_to_behavioral_data_when_absent_from_ssvep_metadata(df):
    """MET013, MET014, MET041, MET042 have no SSVEP record (metadata.csv)
    but do have behavioral data -- group should be resolved from there
    instead of defaulting to UNKNOWN. MET047 has neither and stays
    UNKNOWN (see test_met047_is_unknown_and_met021_keeps_its_ssvep_label)."""
    expected = {"MET013": "CTR", "MET014": "CTR", "MET041": "PD", "MET042": "PD"}
    for sub_id, expected_group in expected.items():
        rows = df[df["sub_id"] == sub_id]
        assert len(rows) > 0
        assert (rows["group"] == expected_group).all()
        assert (rows["subgroup"] == "NA").all()


def test_subjects_in_group_filters_correctly(df):
    protan = loader.subjects_in_group(df, subgroup="protan")
    assert set(protan) == set(df.loc[df["subgroup"] == "protan", "sub_id"].unique())
    assert len(loader.subjects_in_group(df)) == df["sub_id"].nunique()


# --- scores: pinned against templateCode/FM100.py on every real row --------


def test_err_vals_matches_template_on_every_subject(df, template):
    for caps in df["caps"]:
        _, _, expected = template.FM100_TESwhole(caps)
        np.testing.assert_allclose(scores.err_vals(caps), expected)


def test_tes_matches_template_on_every_subject(df, template):
    for caps in df["caps"]:
        expected_tes, expected_sqrt, _ = template.FM100_TESwhole(caps)
        result = scores.tes(caps)
        assert result["TES"] == pytest.approx(expected_tes)
        assert result["SqrtTES"] == pytest.approx(expected_sqrt)


def test_pes_matches_template_on_every_subject(df, template):
    for caps in df["caps"]:
        expected = template.compute_PES(caps)
        result = scores.pes(caps)
        for key in ["PES_RG", "PES_BY", "PES_RG_sqrt", "PES_BY_sqrt"]:
            assert result[key] == pytest.approx(expected[key])


def test_tes_trays_matches_template_on_every_subject(df, template):
    for caps in df["caps"]:
        expected = template.compute_TES_trays(caps)
        result = scores.tes_trays(caps)
        np.testing.assert_allclose(result["TES_tray"], expected["TES_tray"])
        assert result["TES_whole"] == pytest.approx(expected["TES_whole"])


def test_vks_matches_template_on_every_subject(df, template):
    for caps in df["caps"]:
        expected = template.compute_VKS_metrics(caps, silent=True)
        result = scores.vks(caps)
        assert result["VKS_Angle"] == pytest.approx(expected["Angle"])
        assert result["VKS_MajRad"] == pytest.approx(expected["MajRad"])
        assert result["VKS_MinRad"] == pytest.approx(expected["MinRad"])
        assert result["VKS_Sindex"] == pytest.approx(expected["Sindex"])
        assert result["VKS_Cindex"] == pytest.approx(expected["Cindex"])


def test_err_vals_rejects_wrong_length():
    with pytest.raises(ValueError):
        scores.err_vals(np.arange(1, 84))


def test_vks_rejects_non_permutation():
    bad_caps = np.arange(1, 86)
    bad_caps[0] = 2  # cap 2 repeated, cap 1 missing
    with pytest.raises(ValueError):
        scores.vks(bad_caps)


def test_build_scores_one_row_per_subject_session(df):
    result = scores.build_scores(df)
    assert len(result) == len(df)
    assert {"sub_id", "session", "group", "subgroup", "TES", "PES_RG", "PES_BY", "VKS_Angle"}.issubset(result.columns)


# --- plotting ----------------------------------------------------------------


def test_plot_subject_fm100_linear_draws_one_line_per_session(df):
    ax = plotting.plot_subject_fm100(df, "MET001", kind="linear")
    n_sessions = df.loc[df["sub_id"] == "MET001", "session"].nunique()
    assert len(ax.lines) == n_sessions
    assert ax.get_xlim() == (1.0, 85.0)


def test_plot_subject_fm100_radial_uses_a_polar_axes(df):
    ax = plotting.plot_subject_fm100(df, "MET001", kind="radial")
    assert ax.name == "polar"


def test_new_axes_radial_grows_clockwise():
    """PLANScores.md M3: fm100radialTemplate.png's cap numbers grow
    clockwise, not matplotlib's polar default (counterclockwise) --
    set once in _new_axes so the data line, cap-wheel ring, and any angle
    ticks (all plotted via the same CAP_ANGLES array on the same axes)
    stay mutually consistent for free."""
    assert plotting._new_axes("radial").get_theta_direction() == -1


def test_new_axes_linear_is_not_polar():
    assert plotting._new_axes("linear").name != "polar"


def test_plot_subject_fm100_radial_closes_the_circle(df):
    """CAP_ANGLES uses endpoint=False -- without closing, cap 85 and cap 1
    wouldn't connect, leaving a visible gap at the seam."""
    ax = plotting.plot_subject_fm100(df, "MET001", kind="radial", sessions=[1])
    theta, r = ax.lines[0].get_xydata().T
    assert theta[-1] == pytest.approx(theta[0] + 2 * np.pi)
    assert r[-1] == pytest.approx(r[0])


def test_plot_group_fm100_radial_closes_the_circle(df):
    categories = [{"label": "HC", "group": "CTR"}]
    ax = plotting.plot_group_fm100(df, categories, kind="radial")
    theta, r = ax.lines[0].get_xydata().T
    assert theta[-1] == pytest.approx(theta[0] + 2 * np.pi)
    assert r[-1] == pytest.approx(r[0])


def test_plot_subject_fm100_rejects_unknown_kind(df):
    with pytest.raises(ValueError):
        plotting.plot_subject_fm100(df, "MET001", kind="nonsense")


def test_plot_group_fm100_draws_one_line_and_band_per_category(df):
    categories = [{"label": "HC", "group": "CTR"}, {"label": "PD", "group": "PD"}]
    ax = plotting.plot_group_fm100(df, categories)
    assert len(ax.lines) == len(categories)
    assert len(ax.collections) == len(categories)  # fill_between bands


def test_plot_group_fm100_rejects_more_categories_than_full_palette(df):
    categories = [{"label": str(i), "group": "CTR"} for i in range(len(plotting.FULL_PALETTE) + 1)]
    with pytest.raises(ValueError):
        plotting.plot_group_fm100(df, categories)


def test_plot_group_fm100_averages_multi_session_subjects_before_group_stats(df):
    """A subject with 2 sessions must count once toward the group mean/SD,
    not twice -- pin against a hand-picked pair (one 1-session, one
    2-session subject)."""
    session_counts = df.groupby("sub_id")["session"].nunique()
    pair = [session_counts[session_counts == 1].index[0], session_counts[session_counts == 2].index[0]]
    profiles = plotting.group_profiles(df, sub_ids=pair)
    assert profiles.shape == (2, 85)


def test_smooth_circular_is_a_noop_for_window_1():
    values = np.arange(85, dtype=float)
    np.testing.assert_array_equal(plotting._smooth_circular(values, 1), values)


def test_smooth_circular_preserves_length():
    values = np.arange(85, dtype=float)
    for window in [2, 3, 5, 8]:
        assert plotting._smooth_circular(values, window).shape == (85,)


# --- radial cap-number labels (M2) -------------------------------------------


def test_cap_label_starts_at_85_then_continues_1_through_84():
    expected = [85] + list(range(1, 85))
    assert [plotting._cap_label(i) for i in range(85)] == expected


def test_apply_cap_labels_uses_the_requested_step(df):
    ax = plotting.plot_subject_fm100(df, "MET001", kind="radial", sessions=[1], label_mode="cap")
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == [str(plotting._cap_label(i)) for i in range(0, 85, plotting.RADIAL_TICK_STEP)]
    assert labels[0] == "85"


def test_apply_cap_labels_linear_uses_the_requested_step(df):
    """dashboard M3: linear plots get the same 85,1,2,...,84 cap-numbering
    convention radial already uses, unconditionally."""
    ax = plotting.plot_subject_fm100(df, "MET001", kind="linear", sessions=[1])
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == [str(plotting._cap_label(i)) for i in range(0, 85, plotting.RADIAL_TICK_STEP)]
    assert labels[0] == "85"


def test_plot_subject_fm100_rejects_unknown_label_mode(df):
    with pytest.raises(ValueError):
        plotting.plot_subject_fm100(df, "MET001", label_mode="nonsense")


def test_plot_group_fm100_cap_labels_only_applied_for_radial(df):
    """label_mode='cap' on a linear plot should be silently ignored, not
    error -- only the radial axes has angle ticks to relabel."""
    categories = [{"label": "HC", "group": "CTR"}]
    ax = plotting.plot_group_fm100(df, categories, kind="linear", label_mode="cap")
    assert ax.get_xlim() == (1.0, 85.0)


# --- multi-subject / group-vs-subject overlays (dashboard M2) ---------------


def test_plot_subjects_fm100_draws_one_line_per_subject(df):
    sub_ids = sorted(df["sub_id"].unique())[:3]
    ax = plotting.plot_subjects_fm100(df, sub_ids)
    assert len(ax.lines) == len(sub_ids)


def test_plot_subjects_fm100_rejects_more_subjects_than_subject_colors(df):
    sub_ids = sorted(df["sub_id"].unique())[: len(plotting.SUBJECT_COLORS) + 1]
    with pytest.raises(ValueError):
        plotting.plot_subjects_fm100(df, sub_ids)


def test_plot_subjects_fm100_uses_session_1_only(df):
    """A subject with session 1 and 2 must contribute one line, session 1's,
    not one per session (unlike plot_subject_fm100's own overlay)."""
    session_counts = df.groupby("sub_id")["session"].nunique()
    two_session_subject = session_counts[session_counts == 2].index[0]
    ax = plotting.plot_subjects_fm100(df, [two_session_subject])
    expected = plotting._subject_profile(df, two_session_subject, 1, window=1)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), expected)


def test_plot_subjects_fm100_show_cap_wheel_draws_the_ring(df):
    """PLANdashboard.md M2: the dashboard needs show_cap_wheel on this
    function too (previously only on plot_subject_fm100/plot_group_fm100)."""
    sub_ids = sorted(df["sub_id"].unique())[:2]
    ax = plotting.plot_subjects_fm100(df, sub_ids, kind="radial", show_cap_wheel=True)
    wheel = ax.collections[-1]
    assert wheel.get_offsets().shape[0] == plotting.N_CAPS


def test_plot_group_vs_subjects_fm100_draws_group_bands_plus_dashed_subject_lines(df):
    categories = [{"label": "HC", "group": "CTR"}, {"label": "PD", "group": "PD"}]
    sub_ids = sorted(df["sub_id"].unique())[:2]
    ax = plotting.plot_group_vs_subjects_fm100(df, categories, sub_ids)
    assert len(ax.lines) == len(categories) + len(sub_ids)
    subject_lines = ax.lines[len(categories):]
    assert all(line.get_linestyle() == "--" for line in subject_lines)


def test_plot_group_vs_subjects_fm100_rejects_more_subjects_than_subject_colors(df):
    categories = [{"label": "HC", "group": "CTR"}]
    sub_ids = sorted(df["sub_id"].unique())[: len(plotting.SUBJECT_COLORS) + 1]
    with pytest.raises(ValueError):
        plotting.plot_group_vs_subjects_fm100(df, categories, sub_ids)


# --- comparisons.py: group comparisons and offset (M2) -----------------------


def test_subject_pooled_scores_one_row_per_subject(df):
    pooled = comparisons.subject_pooled_scores(df)
    assert len(pooled) == df["sub_id"].nunique()
    assert set(comparisons.FEATURES).issubset(pooled.columns)
    assert pooled["VKS_Angle"].between(0, 180, inclusive="left").all()


def test_subject_pooled_scores_linear_mean_matches_manual_average(df):
    """Non-angle features: plain mean across a subject's sessions."""
    multi_session = df.groupby("sub_id")["session"].nunique()
    sub_id = multi_session[multi_session > 1].index[0]
    pooled = comparisons.subject_pooled_scores(df)
    row = pooled[pooled["sub_id"] == sub_id].iloc[0]
    expected_tes = scores.build_scores(df[df["sub_id"] == sub_id])["TES"].mean()
    assert row["TES"] == pytest.approx(expected_tes)


def test_group_pooled_scores_filters_to_the_requested_group(df):
    result = comparisons.group_pooled_scores(df, group="PD")
    assert set(result["sub_id"]) == set(loader.subjects_in_group(df, group="PD"))


def test_compare_fm100_feature_detects_a_known_separation(df):
    """CTR vs CVD on TES is a large, well-established real separation
    (docs/findings.md section 1: CVD's TES is roughly 4x CTR's) --
    compare_fm100_feature (built on subject_pooled_scores, which needs a
    raw caps-shaped df, not a pre-scored one) should find it easily."""
    result = comparisons.compare_fm100_feature(df, "TES", group1="CTR", group2="CVD")
    assert result["p_value"] < 0.001
    assert result["n1"] == len(loader.subjects_in_group(df, group="CTR"))
    assert result["n2"] == len(loader.subjects_in_group(df, group="CVD"))


def test_estimate_offset_recovers_a_planted_constant():
    rng = np.random.default_rng(0)
    n_subjects, n_caps = 15, 85
    base = rng.normal(50, 5, (n_subjects, n_caps))
    shifted = base + rng.normal(0, 1, (n_subjects, n_caps)) + 30.0  # +30 constant, plus a little noise
    result = comparisons.estimate_offset(base, shifted, n_boot=500, seed=0)
    assert result["offset"] == pytest.approx(30.0, abs=1.0)
    assert result["ci_lower"] < 30.0 < result["ci_upper"]
    assert result["p_value"] < 0.01
    assert result["r_squared"] > 0.9  # a near-pure constant shift, little per-position structure left


def test_estimate_offset_ci_includes_zero_when_there_is_no_offset():
    rng = np.random.default_rng(0)
    n_subjects, n_caps = 15, 85
    base = rng.normal(50, 5, (n_subjects, n_caps))
    same = rng.normal(50, 5, (n_subjects, n_caps))
    result = comparisons.estimate_offset(base, same, n_boot=500, seed=0)
    assert result["ci_lower"] < 0 < result["ci_upper"]
    assert result["p_value"] > 0.05


# --- multiple-comparisons correction & outlier flagging (M3) ---------------


def test_correct_multiple_comparisons_adds_expected_columns():
    result = pd.DataFrame({"p_value": [0.001, 0.02, 0.5]})
    corrected = comparisons.correct_multiple_comparisons(result)
    assert list(corrected.columns) == ["p_value", "p_corrected", "significant"]
    assert (corrected["p_corrected"] >= corrected["p_value"]).all()  # Holm never lowers a p-value
    assert (corrected["p_corrected"] <= 1.0).all()


def test_correct_multiple_comparisons_can_flip_significance():
    """Even the smallest p-value in a family shouldn't survive correction if
    it isn't small enough relative to the family size -- pins the same
    'some uncorrected-significant rows don't survive' behavior
    02_group_comparisons.ipynb reports on real data (CTR vs PD's own six
    p-values, smallest first)."""
    result = pd.DataFrame({"p_value": [0.008648, 0.010223, 0.016728, 0.018148, 0.018159, 0.842472]})
    corrected = comparisons.correct_multiple_comparisons(result)
    assert result.loc[0, "p_value"] < 0.05  # uncorrected-significant
    assert not corrected.loc[0, "significant"]  # doesn't survive Holm at this family size


def test_tukey_outlier_mask_flags_a_clear_outlier():
    values = np.array([10, 11, 9, 10.5, 9.5, 100])
    mask = comparisons.tukey_outlier_mask(values)
    assert mask.tolist() == [False, False, False, False, False, True]


def test_tukey_outlier_mask_flags_nothing_when_evenly_spread():
    values = np.linspace(0, 1, 20)
    assert not comparisons.tukey_outlier_mask(values).any()


def test_subject_feature_outliers_one_row_per_subject_with_n_flagged(df):
    result = comparisons.subject_feature_outliers(df, group="CTR")
    assert len(result) == len(loader.subjects_in_group(df, group="CTR"))
    assert set(comparisons.FEATURES).issubset(result.columns)
    assert (result["n_flagged"] == result[comparisons.FEATURES].sum(axis=1)).all()
    assert result["n_flagged"].max() <= len(comparisons.FEATURES)


# --- radial cap wheel & feature boxplots (M3) -------------------------------


def test_show_cap_wheel_draws_n_caps_dots_and_blanks_radial_ticks(df):
    ax = plotting.plot_subject_fm100(df, "MET001", kind="radial", sessions=[1], show_cap_wheel=True)
    wheel = ax.collections[-1]  # the cap-color scatter, added after the profile line
    assert wheel.get_offsets().shape[0] == scores.N_CAPS
    assert all(label.get_text() == "" for label in ax.get_yticklabels())


def test_show_cap_wheel_overrides_label_mode_cap(df):
    """Both set: the ring (which already carries every cap's number) wins,
    _apply_cap_labels' every-Nth-cap ticks are never applied."""
    ax = plotting.plot_subject_fm100(df, "MET001", kind="radial", sessions=[1], show_cap_wheel=True, label_mode="cap")
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels != [str(plotting._cap_label(i)) for i in range(0, 85, plotting.RADIAL_TICK_STEP)]


def test_plot_feature_boxplot_draws_one_box_per_category(df):
    categories = [{"label": "CTR", "group": "CTR"}, {"label": "PD", "group": "PD"}]
    ax = plotting.plot_feature_boxplot(df, "TES", categories)
    assert len(ax.get_xticks()) == len(categories)


def test_plot_feature_boxplot_scatters_every_subject(df):
    categories = [{"label": "PD", "group": "PD"}]
    ax = plotting.plot_feature_boxplot(df, "TES", categories)
    n_pd = len(loader.subjects_in_group(df, group="PD"))
    n_scattered = sum(c.get_offsets().shape[0] for c in ax.collections)
    assert n_scattered == n_pd


def test_plot_feature_boxplot_rejects_more_categories_than_full_palette(df):
    categories = [{"label": str(i), "group": "CTR"} for i in range(len(plotting.FULL_PALETTE) + 1)]
    with pytest.raises(ValueError):
        plotting.plot_feature_boxplot(df, "TES", categories)


def test_plot_feature_boxplots_grid_has_one_panel_per_feature(df):
    categories = [{"label": "CTR", "group": "CTR"}, {"label": "PD", "group": "PD"}]
    fig = plotting.plot_feature_boxplots_grid(df, categories)
    on_axes = [ax for ax in fig.axes if ax.has_data()]
    assert len(on_axes) == len(comparisons.FEATURES)


def test_plot_group_vs_subjects_fm100_wheel_does_not_change_the_rendered_hole_size(df):
    """Same regression as test_show_cap_wheel_does_not_change_the_rendered_hole_size,
    for plot_group_vs_subjects_fm100's own manual hole/wheel sequence
    (it doesn't call _finish_radial_axes directly -- see its own comment)."""
    categories = [{"label": "HC", "group": "CTR"}]
    with_wheel = plotting.plot_group_vs_subjects_fm100(df, categories, ["MET020"], kind="radial", show_cap_wheel=True)
    without_wheel = plotting.plot_group_vs_subjects_fm100(df, categories, ["MET020"], kind="radial", show_cap_wheel=False)
    assert _rendered_hole_fraction(with_wheel) == pytest.approx(_rendered_hole_fraction(without_wheel))


def test_show_cap_wheel_does_not_clip_a_subject_overlay_exceeding_the_group_band(df):
    """_draw_cap_wheel calls ax.set_ylim, which disables further
    autoscaling -- if it ran before plot_group_vs_subjects_fm100's dashed
    subject overlays were added, any subject reaching past the group
    bands' own r-range would get silently clipped at the frozen limit."""
    categories = [{"label": "HC", "group": "CTR"}]
    ax = plotting.plot_group_vs_subjects_fm100(df, categories, ["MET020"], kind="radial", show_cap_wheel=True)
    subject_line = ax.lines[-1]  # the dashed MET020 overlay, added last
    assert subject_line.get_label() == "MET020 (individual)"
    r_max_plotted = subject_line.get_xydata()[:, 1].max()
    assert r_max_plotted <= ax.get_ylim()[1]


# --- radial hole & linear cap colors (M3) -----------------------------------


def test_apply_radial_hole_sets_rorigin_proportional_to_data_max():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(plotting.CAP_ANGLES, np.full(plotting.CAP_ANGLES.shape, 10.0))
    plotting._apply_radial_hole(ax)
    # ax.get_ylim()[1] rather than the plotted 10.0 directly -- matplotlib's
    # autoscale adds its own margin, so the two aren't exactly equal.
    assert ax.get_rorigin() == pytest.approx(-plotting.RADIAL_HOLE_FRAC * ax.get_ylim()[1])


def test_apply_radial_hole_recomputes_on_a_second_call_with_more_data():
    """Safe to call twice (plot_group_vs_subjects_fm100 does, after adding
    subject overlays) -- the second call must reflect the larger range,
    not the first call's now-stale one."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(plotting.CAP_ANGLES, np.full(plotting.CAP_ANGLES.shape, 10.0))
    plotting._apply_radial_hole(ax)
    first_rorigin = ax.get_rorigin()
    ax.plot(plotting.CAP_ANGLES, np.full(plotting.CAP_ANGLES.shape, 20.0))
    plotting._apply_radial_hole(ax)
    assert ax.get_rorigin() == pytest.approx(-plotting.RADIAL_HOLE_FRAC * ax.get_ylim()[1])
    assert ax.get_rorigin() < first_rorigin  # more negative -- a bigger hole for the bigger range


def test_plot_subject_fm100_radial_applies_the_hole_by_default(df):
    """MET038 has several caps at err_vals=0 (PLANScores.md M3) -- confirms
    the hole is actually wired into plot_subject_fm100's default radial
    path, not just present in _apply_radial_hole's own unit test above."""
    ax = plotting.plot_subject_fm100(df, "MET038", kind="radial", sessions=[1], window=1)
    assert ax.get_rorigin() < 0


def _rendered_hole_fraction(ax: plt.Axes) -> float:
    """matplotlib's polar rendering computes the visible hole as
    |rorigin| / (ylim[1] - rorigin) -- not rorigin alone -- since
    originViewLim locks only y0 (to rorigin); y1 stays live and tracks
    ax.viewLim.y1. See _finish_radial_axes's docstring."""
    rorigin = ax.get_rorigin()
    return abs(rorigin) / (ax.get_ylim()[1] - rorigin)


def test_show_cap_wheel_does_not_change_the_rendered_hole_size(df):
    """Regression test: _draw_cap_wheel calls ax.set_ylim, inflating the
    r-range by ~1.5x. _apply_radial_hole must run *after* the wheel (or
    'cap' labels), not before, or the same absolute rorigin renders a
    visibly smaller hole once the wheel's set_ylim has run -- caught by
    code review, verified against matplotlib's actual polar transform
    (originViewLim locks only y0=rorigin; y1 stays live)."""
    with_wheel = plotting.plot_subject_fm100(df, "MET038", kind="radial", sessions=[1], window=5, show_cap_wheel=True)
    without_wheel = plotting.plot_subject_fm100(df, "MET038", kind="radial", sessions=[1], window=5)
    assert _rendered_hole_fraction(with_wheel) == pytest.approx(_rendered_hole_fraction(without_wheel))


def test_cap_color_matches_cap_wheel_ordering():
    """_cap_color(cap_label) must agree with _draw_cap_wheel's own
    position_index-based coloring -- they're independent code paths (kept
    separate to avoid touching already-verified wheel code) that need to
    produce the same color for the same cap."""
    for position_index in range(plotting.N_CAPS):
        label = plotting._cap_label(position_index)
        expected = plotting.CAP_WHEEL_CMAP(position_index / plotting.N_CAPS)
        assert plotting._cap_color(label) == expected


def test_draw_cap_colors_linear_adds_one_scatter_per_sampled_cap():
    fig, ax = plt.subplots()
    ax.plot(plotting.CAP_POSITIONS, np.ones_like(plotting.CAP_POSITIONS, dtype=float))
    n_before = len(ax.collections)
    plotting._draw_cap_colors_linear(ax)
    assert len(ax.collections) == n_before + 1
    n_expected = len(np.arange(1, 86, plotting.RADIAL_TICK_STEP))
    assert ax.collections[-1].get_offsets().shape[0] == n_expected


def test_show_cap_colors_places_dots_below_the_data_range(df):
    ax = plotting.plot_subject_fm100(df, "MET038", kind="linear", sessions=[1], show_cap_colors=True)
    dots_y = ax.collections[-1].get_offsets()[:, 1]
    line_y = ax.lines[0].get_ydata()
    assert dots_y.max() < line_y.min()
