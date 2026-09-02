"""Load tidy SSVEP files and compute baseline-normalized grids.

Baseline trial order (confirmed): trials 1-2 are pre-grid, 3-4 are post-grid.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

# Data lives outside this repo's git history (per-subject data is never
# committed) -- MET_DASHBOARD_DATA_DIR overrides where it's found, defaulting
# to <repo_root>/data (ssveps/scripts/ -> repo root is two levels up).
DATA_DIR = os.environ.get(
    "MET_DASHBOARD_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data"),
)
FILES_DIR = os.path.join(DATA_DIR, "ssveps", "files")

TRIAL_SUBSETS = {"all": (1, 2, 3, 4), "first2": (1, 2), "last2": (3, 4)}

# Standard cross-subject-comparable normalization (percent change from
# baseline) -- raw SSVEP amplitude varies a lot subject-to-subject, so
# anything comparing across subjects/groups (trough depth, permutation
# testing) defaults to this rather than raw values.
DEFAULT_NORMALIZE = {"scope": "run", "trials": "all", "method": "percent"}


def load_runmap() -> pd.DataFrame:
    return pd.read_csv(os.path.join(FILES_DIR, "runmap.csv"))


def load_baselines() -> pd.DataFrame:
    return pd.read_csv(os.path.join(FILES_DIR, "baselines.csv"))


def load_metadata() -> pd.DataFrame:
    # keep_default_na=False: subgroup's literal "NA" value must not become NaN
    return pd.read_csv(os.path.join(FILES_DIR, "metadata.csv"), keep_default_na=False)


def load_grid_axes() -> tuple[list[float], list[float]]:
    with open(os.path.join(FILES_DIR, "grid.json")) as f:
        grid = json.load(f)
    return grid["redArray"], grid["greenArray"]


def raw_grid(runmap_df: pd.DataFrame, sub_id: str, session: int, run: int) -> np.ndarray:
    """10x10 (red_idx, green_idx) grid of raw runMap values for one run."""
    sub = runmap_df.query("sub_id == @sub_id and session == @session and run == @run")
    return sub.pivot(index="red_idx", columns="green_idx", values="value").sort_index().sort_index(axis=1).to_numpy()


def mean_raw_grid(runmap_df: pd.DataFrame, sub_id: str, session: int) -> np.ndarray:
    """Mean raw grid across all runs of a session."""
    sub = runmap_df.query("sub_id == @sub_id and session == @session")
    pivoted = sub.groupby(["red_idx", "green_idx"])["value"].mean().unstack()
    return pivoted.sort_index().sort_index(axis=1).to_numpy()


def baseline_values(
    baselines_df: pd.DataFrame, sub_id: str, session: int, *, scope: str = "run", run: int | None = None, trials: str = "all"
) -> np.ndarray:
    """Selected baseline trial values: one run's own trials (scope='run'), or
    pooled across every run of the session (scope='session')."""
    trial_nums = TRIAL_SUBSETS[trials]
    sub = baselines_df.query("sub_id == @sub_id and session == @session and trial in @trial_nums")
    if scope == "run":
        if run is None:
            raise ValueError("scope='run' requires a run")
        sub = sub.query("run == @run")
    elif scope != "session":
        raise ValueError(f"unknown scope: {scope}")
    return sub["value"].to_numpy()


def normalize_grid(raw: np.ndarray, baseline_vals: np.ndarray, *, method: str = "percent") -> np.ndarray:
    """Normalize a raw grid against a scalar baseline derived from baseline_vals."""
    base_mean = baseline_vals.mean()
    if method == "percent":
        return (raw - base_mean) / base_mean
    if method == "db":
        return 10 * np.log10(raw / base_mean)
    if method == "zscore":
        return (raw - base_mean) / baseline_vals.std()
    raise ValueError(f"unknown method: {method}")


def normalized_grid(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    sub_id: str,
    session: int,
    run: int,
    *,
    scope: str = "run",
    trials: str = "all",
    method: str = "percent",
) -> np.ndarray:
    raw = raw_grid(runmap_df, sub_id, session, run)
    bvals = baseline_values(baselines_df, sub_id, session, scope=scope, run=run, trials=trials)
    return normalize_grid(raw, bvals, method=method)


def run_grids(
    runmap_df: pd.DataFrame, baselines_df: pd.DataFrame, sub_id: str, session: int, normalize: dict | None
) -> list[np.ndarray]:
    """Each run's grid (raw or normalized) for one subject/session, in run
    order. Public (not just an internal helper for mean_grid/flatten_runs)
    because a run-level bootstrap -- resample which runs go into the mean,
    with replacement -- needs the individual run grids directly rather than
    their average; see the ramp-crossing extrapolation CI in
    08_cvd_gamut.ipynb."""
    runs = sorted(runmap_df.query("sub_id == @sub_id and session == @session")["run"].unique())
    if normalize is None:
        return [raw_grid(runmap_df, sub_id, session, run) for run in runs]
    return [normalized_grid(runmap_df, baselines_df, sub_id, session, run, **normalize) for run in runs]


def mean_grid(
    runmap_df: pd.DataFrame, baselines_df: pd.DataFrame, sub_id: str, session: int, *, normalize: dict | None = None
) -> np.ndarray:
    """Mean grid across all runs of one subject/session, raw or normalized
    (normalize, if given, is a scope/trials/method dict for normalized_grid;
    each run is normalized individually, then the runs are averaged)."""
    if normalize is None:
        return mean_raw_grid(runmap_df, sub_id, session)
    return np.mean(run_grids(runmap_df, baselines_df, sub_id, session, normalize), axis=0)


def flatten_runs(
    runmap_df: pd.DataFrame, baselines_df: pd.DataFrame, sub_id: str, session: int, *, normalize: dict | None = None
) -> np.ndarray:
    """Every pixel of every run of one subject/session, concatenated into one
    1D array (400 values for 4-run subjects, 300 for the ragged 3-run ones)."""
    grids = run_grids(runmap_df, baselines_df, sub_id, session, normalize)
    return np.concatenate([g.ravel() for g in grids])


def pooled_pixels(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    sub_ids: list[str],
    session: int,
    *,
    normalize: dict | None = None,
) -> np.ndarray:
    """flatten_runs for every subject in sub_ids, concatenated into one pooled
    1D array (a group's entire raw pixel distribution, not averaged)."""
    return np.concatenate([flatten_runs(runmap_df, baselines_df, sub_id, session, normalize=normalize) for sub_id in sub_ids])


def run_mean_values(
    runmap_df: pd.DataFrame, baselines_df: pd.DataFrame, sub_id: str, session: int, *, normalize: dict | None = None
) -> np.ndarray:
    """Each run's overall response level for one subject/session -- the mean
    of that run's grid (100 cells, raw or normalized), one scalar per run.
    This is the unit the within/between-subject variance decomposition (M7,
    docs/ssvep_analyses.md proposal 3, variance.py) operates on: run-to-run
    spread in this value is "within-subject" variance, and subject-to-subject
    spread in each subject's own mean of these values is "between-subject"
    variance."""
    return np.array([g.mean() for g in run_grids(runmap_df, baselines_df, sub_id, session, normalize)])


def pooled_baseline_values(
    baselines_df: pd.DataFrame, sub_ids: list[str], session: int, *, trials: str = "all"
) -> np.ndarray:
    """Every baseline trial value (scope='session', i.e. pooled across every
    run) for every subject in sub_ids, concatenated -- the baseline-value
    analogue of pooled_pixels, for comparing the raw baseline itself across
    groups (baselines are the normalization's own denominator, so this is
    always raw, never normalized)."""
    return np.concatenate(
        [baseline_values(baselines_df, sub_id, session, scope="session", trials=trials) for sub_id in sub_ids]
    )


def subjects_in_group(
    metadata_df: pd.DataFrame, session: int, *, group: str | None = None, subgroup: str | None = None
) -> list[str]:
    """Subject IDs at a given session, optionally filtered by group and/or subgroup."""
    sub = metadata_df.query("session == @session")
    if group is not None:
        sub = sub.query("group == @group")
    if subgroup is not None:
        sub = sub.query("subgroup == @subgroup")
    return sorted(sub["sub_id"].unique())


def mean_grid_across_subjects(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    sub_ids: list[str],
    session: int,
    *,
    normalize: dict | None = None,
) -> np.ndarray:
    """Mean of each subject's own mean-across-runs grid (each subject weighted equally)."""
    grids = [mean_grid(runmap_df, baselines_df, sub_id, session, normalize=normalize) for sub_id in sub_ids]
    return np.mean(grids, axis=0)


def group_grid(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    session: int,
    *,
    group: str | None = None,
    subgroup: str | None = None,
    normalize: dict | None = None,
) -> np.ndarray:
    """Aggregated mean grid across every subject at this session matching
    group/subgroup (every subject at that session if neither is given).
    Convenience wrapper composing subjects_in_group + mean_grid_across_subjects
    for reuse outside of plotting (e.g. further stats, group difference maps)."""
    sub_ids = subjects_in_group(metadata_df, session, group=group, subgroup=subgroup)
    return mean_grid_across_subjects(runmap_df, baselines_df, sub_ids, session, normalize=normalize)


def interpolate_grid(grid: np.ndarray, shape: tuple[int, int], *, method: str = "linear") -> np.ndarray:
    """Resize a (red_idx, green_idx) grid to a new (n_red, n_green) resolution."""
    n_red, n_green = grid.shape
    interpolator = RegularGridInterpolator((np.arange(n_red), np.arange(n_green)), grid, method=method)
    new_red_n, new_green_n = shape
    red_q, green_q = np.linspace(0, n_red - 1, new_red_n), np.linspace(0, n_green - 1, new_green_n)
    query_red, query_green = np.meshgrid(red_q, green_q, indexing="ij")
    return interpolator(np.stack([query_red.ravel(), query_green.ravel()], axis=-1)).reshape(new_red_n, new_green_n)


def trough_location(grid: np.ndarray, red_vals: list[float], green_vals: list[float]) -> dict:
    """Location and depth of a grid's minimum: physical red/green values, their
    red_idx/green_idx grid positions, and the depth (the minimum value itself).
    Native grid resolution only -- no interpolation (a finer, noise-resistant
    localization via a parametric surface fit is planned separately for M4)."""
    red_idx, green_idx = np.unravel_index(np.argmin(grid), grid.shape)
    return {
        "red": red_vals[red_idx],
        "green": green_vals[green_idx],
        "depth": grid[red_idx, green_idx],
        "red_idx": red_idx,
        "green_idx": green_idx,
    }


def fit_paraboloid(grid: np.ndarray, red_vals: list[float], green_vals: list[float]) -> dict:
    """Fit z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f to the grid via linear
    least squares (closed-form, no initial guess needed), then locate the
    fitted surface's own minimum analytically (solving the gradient = 0),
    which need not land on a grid point -- a noise-robust alternative to
    trough_location's argmin on the coarse 10x10 grid.

    fit_valid is False if the critical point isn't a genuine minimum (the
    Hessian isn't positive-definite -- the fit found a saddle or a maximum
    instead) or falls outside the sampled red/green range (extrapolation);
    depth/red/green are still returned in that case for inspection, but
    shouldn't be treated as a real trough location."""
    x, y = np.meshgrid(red_vals, green_vals, indexing="ij")
    x, y, z = x.ravel(), y.ravel(), grid.ravel()
    design = np.column_stack([x**2, y**2, x * y, x, y, np.ones_like(x)])
    (a, b, c, d, e, f), *_ = np.linalg.lstsq(design, z, rcond=None)

    hessian = np.array([[2 * a, c], [c, 2 * b]])
    is_minimum = bool(np.all(np.linalg.eigvalsh(hessian) > 0))
    if is_minimum:
        x_min, y_min = np.linalg.solve(hessian, [-d, -e])
    else:
        x_min, y_min = np.nan, np.nan
    depth = a * x_min**2 + b * y_min**2 + c * x_min * y_min + d * x_min + e * y_min + f

    z_pred = design @ [a, b, c, d, e, f]
    r_squared = 1 - np.sum((z - z_pred) ** 2) / np.sum((z - z.mean()) ** 2)
    in_bounds = min(red_vals) <= x_min <= max(red_vals) and min(green_vals) <= y_min <= max(green_vals)

    return {
        "red": x_min,
        "green": y_min,
        "depth": depth,
        "r_squared": r_squared,
        "fit_valid": is_minimum and in_bounds,
    }


def fit_gaussian(grid: np.ndarray, red_vals: list[float], green_vals: list[float]) -> dict:
    """Fit an inverted 2D Gaussian dip z = f0 - amp*exp(-((x-x0)^2/(2*sx^2) +
    (y-y0)^2/(2*sy^2))) via nonlinear least squares (scipy.optimize.curve_fit),
    initialized from trough_location's grid argmin. More flexible than
    fit_paraboloid for a sharply localized trough, but can fail to converge
    on flat/noisy data -- fit_valid is False if curve_fit doesn't converge,
    if the fitted amplitude isn't positive (not actually a dip), or if the
    fitted center falls outside the sampled red/green range."""
    x, y = np.meshgrid(red_vals, green_vals, indexing="ij")
    x, y, z = x.ravel(), y.ravel(), grid.ravel()

    def model(xy, f0, amp, x0, y0, sx, sy):
        xx, yy = xy
        return f0 - amp * np.exp(-(((xx - x0) ** 2) / (2 * sx**2) + ((yy - y0) ** 2) / (2 * sy**2)))

    seed = trough_location(grid, red_vals, green_vals)
    p0 = [z.mean(), z.mean() - seed["depth"], seed["red"], seed["green"], (max(red_vals) - min(red_vals)) / 4, (max(green_vals) - min(green_vals)) / 4]
    try:
        (f0, amp, x0, y0, sx, sy), _ = curve_fit(model, (x, y), z, p0=p0, maxfev=5000)
    except RuntimeError:
        return {"red": np.nan, "green": np.nan, "depth": np.nan, "r_squared": np.nan, "fit_valid": False}

    depth = f0 - amp
    z_pred = model((x, y), f0, amp, x0, y0, sx, sy)
    r_squared = 1 - np.sum((z - z_pred) ** 2) / np.sum((z - z.mean()) ** 2)
    in_bounds = min(red_vals) <= x0 <= max(red_vals) and min(green_vals) <= y0 <= max(green_vals)

    return {"red": x0, "green": y0, "depth": depth, "r_squared": r_squared, "fit_valid": bool(amp > 0 and in_bounds)}


def fit_ramp_gaussian(grid: np.ndarray, red_vals: list[float], green_vals: list[float], *, min_snr: float = 2.0) -> dict:
    """Fit a linear ramp plus one localized Gaussian dip:

        z = c0 + c1*x + c2*y - amp * exp(-((x-x0)^2/(2*sx^2) + (y-y0)^2/(2*sy^2)))

    This is the shape the data actually has -- SSVEP amplitude falls off
    monotonically with red intensity, and the isoluminant trough is a
    localized dip sitting on top of that ramp. fit_paraboloid and fit_gaussian
    each ask a single term to represent both the ramp and the dip, which is
    why they fail on ~40% of subjects (a quadratic fitted to a ramp+dip
    surface often has no interior minimum at all).

    Bounds encode the failure modes directly instead of testing for them
    afterwards: amp >= 0 (it must be a dip, not a bump), the centre must lie
    inside the sampled range (no extrapolation), and the widths are bounded
    below (no degenerate one-pixel spikes) and above (no dip so wide it is
    just re-fitting the ramp).

    Two separate quality flags come back, because they mean different things:

    - at_bound -- some parameter is pegged against its bound, so the fit is
      reporting the edge of what the data can express rather than a located
      dip. This is common and physiologically meaningful here: most protan
      subjects peg fitted_red at max(red_vals), i.e. their isoluminant point
      lies beyond the sampled red range. When the centre and the width peg
      together the "dip" has degenerated into re-fitting the ramp, and amp is
      then the ramp's extent, not a trough depth.
    - fit_valid -- the dip is deep enough to be real (amp above min_snr times
      the residual SD) AND not at_bound. Only use red/green/amp/sigma_* from
      rows where fit_valid is True.

    Returns red/green (the dip centre), depth (the fitted surface's value at
    that centre), amp (dip depth relative to the local ramp -- the
    scale-free "how deep is the trough" measure), sigma_red/sigma_green (dip
    width along each axis), r_squared, at_bound and fit_valid.
    """
    x, y = np.meshgrid(red_vals, green_vals, indexing="ij")
    x, y, z = x.ravel(), y.ravel(), grid.ravel()
    red_span, green_span = np.ptp(red_vals), np.ptp(green_vals)

    def model(xy, c0, c1, c2, amp, x0, y0, sx, sy):
        xx, yy = xy
        return c0 + c1 * xx + c2 * yy - amp * np.exp(-(((xx - x0) ** 2) / (2 * sx**2) + ((yy - y0) ** 2) / (2 * sy**2)))

    seed = trough_location(grid, red_vals, green_vals)
    p0 = [z.mean(), 0.0, 0.0, max(z.mean() - seed["depth"], 1e-9), seed["red"], seed["green"], red_span / 4, green_span / 4]
    lower = [-np.inf, -np.inf, -np.inf, 0.0, min(red_vals), min(green_vals), red_span / 20, green_span / 20]
    upper = [np.inf, np.inf, np.inf, np.inf, max(red_vals), max(green_vals), red_span, green_span]

    try:
        params, _ = curve_fit(model, (x, y), z, p0=np.clip(p0, lower, upper), bounds=(lower, upper), maxfev=20000)
    except RuntimeError:
        nan = float("nan")
        return {"red": nan, "green": nan, "depth": nan, "amp": nan, "sigma_red": nan,
                "sigma_green": nan, "r_squared": nan, "at_bound": False, "fit_valid": False}

    c0, c1, c2, amp, x0, y0, sx, sy = params
    residuals = z - model((x, y), *params)
    r_squared = 1 - np.sum(residuals**2) / np.sum((z - z.mean()) ** 2)
    depth = c0 + c1 * x0 + c2 * y0 - amp

    # Relative tolerance on each span, so this does not depend on the units.
    tol_red, tol_green = red_span * 1e-3, green_span * 1e-3
    at_bound = bool(
        min(abs(x0 - min(red_vals)), abs(x0 - max(red_vals))) < tol_red
        or min(abs(y0 - min(green_vals)), abs(y0 - max(green_vals))) < tol_green
        or abs(sx - red_span) < tol_red
        or abs(sy - green_span) < tol_green
    )

    return {
        "red": x0,
        "green": y0,
        "depth": depth,
        "amp": amp,
        "sigma_red": sx,
        "sigma_green": sy,
        "r_squared": r_squared,
        "at_bound": at_bound,
        "fit_valid": bool(amp > min_snr * residuals.std() and not at_bound),
    }


def fit_rotated_gaussian(grid: np.ndarray, red_vals: list[float], green_vals: list[float], *, min_snr: float = 2.0) -> dict:
    """Fit a linear ramp plus one rotated, anisotropic Gaussian dip:

        z = c0 + c1*x + c2*y - amp * exp(-(xr^2/(2*sigma_major^2) + yr^2/(2*sigma_minor^2)))
        xr =  (x - x0) * cos(theta) + (y - y0) * sin(theta)
        yr = -(x - x0) * sin(theta) + (y - y0) * cos(theta)

    fit_ramp_gaussian's dip is axis-aligned (sigma_red, sigma_green
    independently, no tilt). M11's grid-shape-metric brainstorm
    (PLANssveps.md) asks whether a subject's dip is actually tilted relative
    to the red/green grid axes -- something an axis-aligned model can never
    express. Purely additive: does not touch fit_ramp_gaussian or any of its
    bounds/behavior/persisted columns.

    Fit in native (red, green) units, deliberately not rescaled to
    comparable variance first -- the same choice beh/scripts/features.py's
    subject_shape_features makes on the behavioral click cloud, and for the
    same reason: the physical stimulus space itself is what's meaningful
    here. This also keeps orientation_deg on the same footing as beh's own
    orientation_deg, for a later cross-modality comparison via
    ssvep_beh_fm100/scripts/type_axis.py.

    theta is bounded to [0, pi) during the fit; sigma_major/sigma_minor are
    canonicalized after fitting (swapped, with theta rotated 90deg, if the
    fit put the larger sigma second) so "major" always means "the longer
    axis" and orientation_deg is folded into [0, 180) -- mirroring beh's own
    orientation_deg convention (an ellipse's major axis has no intrinsic
    direction).

    Same two quality flags as fit_ramp_gaussian, same meaning: at_bound (a
    parameter pegged against its bound) and fit_valid (amp above min_snr
    times the residual SD, and not at_bound). One additional caveat specific
    to this model: orientation_deg is poorly identified as sigma_major
    approaches sigma_minor (a nearly circular dip has no well-defined tilt)
    -- read it cautiously when sigma_major/sigma_minor is close to 1.
    """
    x, y = np.meshgrid(red_vals, green_vals, indexing="ij")
    x, y, z = x.ravel(), y.ravel(), grid.ravel()
    red_span, green_span = np.ptp(red_vals), np.ptp(green_vals)
    sigma_lower = min(red_span, green_span) / 20
    sigma_upper = float(np.hypot(red_span, green_span))

    def model(xy, c0, c1, c2, amp, x0, y0, sigma_major, sigma_minor, theta):
        xx, yy = xy
        dx, dy = xx - x0, yy - y0
        xr = dx * np.cos(theta) + dy * np.sin(theta)
        yr = -dx * np.sin(theta) + dy * np.cos(theta)
        return c0 + c1 * xx + c2 * yy - amp * np.exp(-(xr**2 / (2 * sigma_major**2) + yr**2 / (2 * sigma_minor**2)))

    seed = trough_location(grid, red_vals, green_vals)
    p0 = [z.mean(), 0.0, 0.0, max(z.mean() - seed["depth"], 1e-9), seed["red"], seed["green"], red_span / 4, green_span / 4, 0.0]
    lower = [-np.inf, -np.inf, -np.inf, 0.0, min(red_vals), min(green_vals), sigma_lower, sigma_lower, 0.0]
    upper = [np.inf, np.inf, np.inf, np.inf, max(red_vals), max(green_vals), sigma_upper, sigma_upper, np.pi]

    try:
        params, _ = curve_fit(model, (x, y), z, p0=np.clip(p0, lower, upper), bounds=(lower, upper), maxfev=20000)
    except RuntimeError:
        nan = float("nan")
        return {"red": nan, "green": nan, "depth": nan, "amp": nan, "sigma_major": nan,
                "sigma_minor": nan, "orientation_deg": nan, "r_squared": nan, "at_bound": False, "fit_valid": False}

    c0, c1, c2, amp, x0, y0, sigma_major, sigma_minor, theta = params
    residuals = z - model((x, y), *params)
    r_squared = 1 - np.sum(residuals**2) / np.sum((z - z.mean()) ** 2)
    depth = c0 + c1 * x0 + c2 * y0 - amp

    if sigma_minor > sigma_major:
        sigma_major, sigma_minor = sigma_minor, sigma_major
        theta += np.pi / 2
    orientation_deg = float(np.degrees(theta) % 180)

    tol_red, tol_green = red_span * 1e-3, green_span * 1e-3
    tol_sigma = (sigma_upper - sigma_lower) * 1e-3
    at_bound = bool(
        min(abs(x0 - min(red_vals)), abs(x0 - max(red_vals))) < tol_red
        or min(abs(y0 - min(green_vals)), abs(y0 - max(green_vals))) < tol_green
        or min(abs(sigma_major - sigma_lower), abs(sigma_major - sigma_upper)) < tol_sigma
        or min(abs(sigma_minor - sigma_lower), abs(sigma_minor - sigma_upper)) < tol_sigma
    )

    return {
        "red": x0,
        "green": y0,
        "depth": depth,
        "amp": amp,
        "sigma_major": sigma_major,
        "sigma_minor": sigma_minor,
        "orientation_deg": orientation_deg,
        "r_squared": r_squared,
        "at_bound": at_bound,
        "fit_valid": bool(amp > min_snr * residuals.std() and not at_bound),
    }


def fit_ramp(grid: np.ndarray, red_vals: list[float], green_vals: list[float]) -> dict:
    """Fit z = c0 + c1*x + c2*y -- the ramp term alone, no dip -- via linear
    least squares. Where fit_ramp_gaussian pegs a subject's dip against the
    sampled-range bound (at_bound=True), the dip has degenerated into
    re-fitting the ramp anyway (see fit_ramp_gaussian's docstring), so a
    dedicated ramp-only fit gives the same information more directly and
    without a nonlinear solve: slope_red is a continuous measure of "how fast
    does the response fall off with red" that every subject has, whether or
    not their trough itself was ever sampled (M6, docs/ssvep_analyses.md
    proposal 2)."""
    x, y = np.meshgrid(red_vals, green_vals, indexing="ij")
    x, y, z = x.ravel(), y.ravel(), grid.ravel()
    design = np.column_stack([np.ones_like(x), x, y])
    (c0, c1, c2), *_ = np.linalg.lstsq(design, z, rcond=None)
    z_pred = design @ [c0, c1, c2]
    r_squared = 1 - np.sum((z - z_pred) ** 2) / np.sum((z - z.mean()) ** 2)
    return {"intercept": c0, "slope_red": c1, "slope_green": c2, "r_squared": r_squared}


def extrapolate_ramp_crossing(ramp: dict, target_depth: float, green_ref: float) -> float:
    """The red value at which a fitted ramp (fit_ramp's output) would reach
    target_depth at green=green_ref: solves target_depth = intercept +
    slope_red*red + slope_green*green_ref for red.

    For a pegged subject this deliberately does not use that subject's own
    fitted trough depth/green (fit_ramp_gaussian already reports those as
    unreliable when at_bound) -- pass a target_depth/green_ref derived from
    subjects whose trough *was* actually located (e.g. the median fitted_depth
    and fitted_green among fit_valid subjects of the same subgroup), so the
    result estimates "how far beyond the sampled range would this subject's
    ramp need to extend to reach a typical trough" rather than reproducing the
    pegged fit's own boundary value. Always beyond the sampled range for a
    pegged subject by construction -- this is extrapolation, not a
    measurement, and should be labelled as such wherever reported."""
    return (target_depth - ramp["intercept"] - ramp["slope_green"] * green_ref) / ramp["slope_red"]


def bootstrap_ci(replicate_fn, *, n_boot: int = 2000, ci: float = 0.95, seed: int | None = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval. replicate_fn(rng) computes
    and returns one resampled statistic (using rng for its own resampling,
    e.g. rng.integers to pick which runs/subjects to resample with
    replacement); called n_boot times. NaN replicates (e.g. a degenerate
    resample) are dropped before taking percentiles. Generic over what's being
    resampled -- used both for a group proportion's CI (resample subjects) and
    for a per-subject fitted statistic's CI (resample runs)."""
    rng = np.random.default_rng(seed)
    replicates = np.array([replicate_fn(rng) for _ in range(n_boot)])
    replicates = replicates[~np.isnan(replicates)]
    alpha = (1 - ci) / 2
    return float(np.quantile(replicates, alpha)), float(np.quantile(replicates, 1 - alpha))


def fit_gain_shape(grid: np.ndarray, template: np.ndarray) -> dict:
    """Per-subject decomposition of a response grid against a reference
    template (typically the CTR group mean grid): grid ~= gain*template +
    intercept, fit by linear least squares over the 100 cell values (M8,
    docs/ssvep_analyses.md proposal 4).

    `gain` is a uniform scaling of the template's whole shape -- gain < 1
    means a smaller response everywhere, same relative shape as the
    template. `residual` (grid - fitted, same shape as grid) is what's left
    after removing that uniform scaling: near zero everywhere means the
    subject's surface really is just a scaled/shifted copy of the template;
    a residual concentrated specifically near the template's own trough
    (see trough_region_residual) means there's a trough-specific effect
    beyond a uniform gain change, which a gain number alone cannot show."""
    x, y = template.ravel(), grid.ravel()
    design = np.column_stack([x, np.ones_like(x)])
    (gain, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = gain * x + intercept
    r_squared = 1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    return {
        "gain": float(gain),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "residual": (y - fitted).reshape(grid.shape),
    }


def trough_region_residual(residual: np.ndarray, red_idx: int, green_idx: int, *, half_width: int = 1) -> dict:
    """Mean residual (fit_gain_shape's output) inside a (2*half_width+1)^2
    window centered on (red_idx, green_idx) -- typically the reference
    template's own trough location, not the subject's -- versus the mean
    residual over the rest of the grid. These are the two numbers proposal 4
    compares to ask whether a deficit is trough-specific or spread evenly
    across the whole grid (which fit_gain_shape's uniform gain already
    explains)."""
    red_lo, red_hi = max(0, red_idx - half_width), min(residual.shape[0], red_idx + half_width + 1)
    green_lo, green_hi = max(0, green_idx - half_width), min(residual.shape[1], green_idx + half_width + 1)
    mask = np.zeros_like(residual, dtype=bool)
    mask[red_lo:red_hi, green_lo:green_hi] = True
    return {"trough_region": float(residual[mask].mean()), "rest_of_grid": float(residual[~mask].mean())}


# Default surface-fit model. ramp_gaussian fits every subject-session in this
# dataset (62/62) where paraboloid manages 37/62, at a higher r-squared, and it
# is the only one of the three that also reports the dip's width.
DEFAULT_SURFACE_METHOD = "ramp_gaussian"


def fit_trough_surface(grid: np.ndarray, red_vals: list[float], green_vals: list[float], *, method: str = DEFAULT_SURFACE_METHOD) -> dict:
    """fit_ramp_gaussian (default), fit_paraboloid or fit_gaussian, by method.

    Only ramp_gaussian reports amp/sigma_red/sigma_green; the other two return
    NaN for those keys so every caller sees the same set of columns."""
    if method == "ramp_gaussian":
        return fit_ramp_gaussian(grid, red_vals, green_vals)
    if method == "paraboloid":
        fit = fit_paraboloid(grid, red_vals, green_vals)
    elif method == "gaussian":
        fit = fit_gaussian(grid, red_vals, green_vals)
    else:
        raise ValueError(f"unknown method: {method}")
    return {**fit, "amp": float("nan"), "sigma_red": float("nan"), "sigma_green": float("nan"), "at_bound": False}


def subject_troughs(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    normalize: dict | None = DEFAULT_NORMALIZE,
    surface_method: str = DEFAULT_SURFACE_METHOD,
) -> pd.DataFrame:
    """One row per (sub_id, session) in metadata_df: the minimum location and
    depth of that subject's mean-across-runs grid, both from the raw grid
    argmin (red/green/depth/red_idx/green_idx, trough_location) and from a
    parametric surface fit (fitted_*, fit_trough_surface -- the noise-robust
    alternative). With the default surface_method='ramp_gaussian' the fit also
    yields fitted_amp (dip depth relative to the local ramp) and
    fitted_sigma_red/fitted_sigma_green (dip width along each axis), which the
    argmin cannot provide at all. normalize=None for raw depth, otherwise a
    scope/trials/method dict as elsewhere.

    Also adds a separate ramp-only fit (fit_ramp, M6): ramp_intercept,
    ramp_slope_red, ramp_slope_green, ramp_r_squared. Unlike fitted_red/
    fitted_green/fitted_amp, these are defined -- and meaningful -- for every
    subject regardless of fitted_at_bound/fitted_valid, since a ramp has no
    interior minimum to fail to find. This is what makes CVD subjects whose
    trough lies beyond the sampled red range still usable: ramp_slope_red is
    the continuous measure to compare across all of them (see
    docs/ssvep_analyses.md proposal 2 and 08_cvd_gamut.ipynb).

    Also adds a rotated-dip fit (fit_rotated_gaussian, M11): rotated_red,
    rotated_green, rotated_depth, rotated_amp, rotated_sigma_major,
    rotated_sigma_minor, rotated_orientation_deg, rotated_r_squared,
    rotated_at_bound, rotated_valid -- the axis-aligned fitted_* columns'
    tilted counterpart, added purely alongside them (fitted_*/ramp_* are
    unchanged by this addition)."""
    red_vals, green_vals = load_grid_axes()
    rows = []
    for meta in metadata_df.to_dict("records"):
        grid = mean_grid(runmap_df, baselines_df, meta["sub_id"], meta["session"], normalize=normalize)
        loc = trough_location(grid, red_vals, green_vals)
        fit = fit_trough_surface(grid, red_vals, green_vals, method=surface_method)
        ramp = fit_ramp(grid, red_vals, green_vals)
        rotated = fit_rotated_gaussian(grid, red_vals, green_vals)
        rows.append(
            {
                "sub_id": meta["sub_id"],
                "session": meta["session"],
                "group": meta["group"],
                "subgroup": meta["subgroup"],
                **loc,
                "fitted_red": fit["red"],
                "fitted_green": fit["green"],
                "fitted_depth": fit["depth"],
                "fitted_amp": fit["amp"],
                "fitted_sigma_red": fit["sigma_red"],
                "fitted_sigma_green": fit["sigma_green"],
                "fitted_r_squared": fit["r_squared"],
                "fitted_at_bound": fit["at_bound"],
                "fitted_valid": fit["fit_valid"],
                "ramp_intercept": ramp["intercept"],
                "ramp_slope_red": ramp["slope_red"],
                "ramp_slope_green": ramp["slope_green"],
                "ramp_r_squared": ramp["r_squared"],
                "rotated_red": rotated["red"],
                "rotated_green": rotated["green"],
                "rotated_depth": rotated["depth"],
                "rotated_amp": rotated["amp"],
                "rotated_sigma_major": rotated["sigma_major"],
                "rotated_sigma_minor": rotated["sigma_minor"],
                "rotated_orientation_deg": rotated["orientation_deg"],
                "rotated_r_squared": rotated["r_squared"],
                "rotated_at_bound": rotated["at_bound"],
                "rotated_valid": rotated["fit_valid"],
            }
        )
    return pd.DataFrame(rows)


def group_troughs(
    runmap_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    sessions: list[int],
    categories: list[dict],
    *,
    normalize: dict | None = DEFAULT_NORMALIZE,
) -> pd.DataFrame:
    """One row per (session, category) with at least one subject: the minimum
    location and depth of that category's mean-of-subject-means grid.
    categories as in plotting.plot_groups_side_by_side -- categories with no
    subjects at a given session (e.g. deutan at session 2) are skipped."""
    red_vals, green_vals = load_grid_axes()
    rows = []
    for session in sessions:
        for cat in categories:
            sub_ids = subjects_in_group(metadata_df, session, group=cat.get("group"), subgroup=cat.get("subgroup"))
            if not sub_ids:
                continue
            grid = mean_grid_across_subjects(runmap_df, baselines_df, sub_ids, session, normalize=normalize)
            loc = trough_location(grid, red_vals, green_vals)
            rows.append({"label": cat["label"], "session": session, "n": len(sub_ids), **loc})
    return pd.DataFrame(rows)
