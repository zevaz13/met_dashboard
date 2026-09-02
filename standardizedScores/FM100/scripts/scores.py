"""FM100 scoring: Total Error Score (TES), Partial Error Scores (PES,
red-green/blue-yellow), tray-level TES, and Vingrys-King-Smith (VKS) ellipse
metrics -- refactored from templateCode/FM100.py.

TES, PES, and the tray scores all derive from the same per-cap circular hue
error (err_vals): TES sums it over all 85 caps, PES sums two fixed subsets
of it (the red-green vs. blue-yellow axis caps), and tes_trays recomputes an
analogous but distinct local version per 22-cap tray (different wrap
boundaries -- each tray's error is relative to its own neighbors, not the
whole circle). err_vals is computed once and shared by tes/pes rather than
recomputed, since the template repeated the identical loop for each.

The PES red-green/blue-yellow index groups (RG_INDICES/BY_INDICES) and the
VKS ellipse's U-V lookup table are transcribed verbatim from the template,
not re-derived -- getting either wrong would silently corrupt every
downstream score with no test oracle to catch it. tests/test_fm100.py pins
this module's output against the original template functions on real data.
"""

import numpy as np
import pandas as pd

N_CAPS = 85


def _circ_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum((a - b) % N_CAPS, (b - a) % N_CAPS)


def _wrap_error(padded: np.ndarray) -> np.ndarray:
    """dist(prev, curr) + dist(next, curr) - 2 for each interior position of
    an already wrap-padded sequence (length n+2, one extra cap glued on
    each end) -- the FM100 TES scoring rule. Shared by err_vals (whole
    85-cap circle) and tes_trays (each 22-cap tray, its own local wrap)."""
    prev, curr, nxt = padded[:-2], padded[1:-1], padded[2:]
    return _circ_dist(prev, curr) + _circ_dist(nxt, curr) - 2


def err_vals(caps: np.ndarray) -> np.ndarray:
    """Per-cap circular hue error (length 85): dist(prev, curr) +
    dist(next, curr) - 2 for each cap, wrapping around the full 85-cap
    circle. The standard FM100 TES scoring rule -- 0 for a perfectly ordered
    neighbourhood, larger for a cap placed out of hue order."""
    caps = np.asarray(caps, dtype=int)
    if caps.size != N_CAPS:
        raise ValueError(f"caps must have {N_CAPS} entries, got {caps.size}")
    padded = np.concatenate(([caps[-1]], caps, [caps[0]]))
    return _wrap_error(padded)


def tes(caps: np.ndarray) -> dict:
    """Total Error Score and its square root."""
    total = float(err_vals(caps).sum())
    return {"TES": total, "SqrtTES": float(np.sqrt(total))}


# Tray-position groups for the red-green / blue-yellow axes, and the
# real_idxs reindexing step -- transcribed verbatim from
# templateCode/FM100.py's compute_PES (see module docstring).
_RG_INDICES = np.r_[13:34, 55:76] - 1
_BY_INDICES = np.r_[1:13, 34:55, 76:85] - 1
_REAL_IDXS = np.r_[85, np.arange(1, 85)] - 1
_RG_IDXS = np.where(np.isin(_REAL_IDXS, _RG_INDICES))[0]
_BY_IDXS = np.where(np.isin(_REAL_IDXS, _BY_INDICES))[0]


def pes(caps: np.ndarray) -> dict:
    """Partial Error Scores: red-green and blue-yellow axis subsets of
    err_vals, and their square roots."""
    err = err_vals(caps)
    rg = float(err[_RG_IDXS].sum())
    by = float(err[_BY_IDXS].sum())
    return {"PES_RG": rg, "PES_BY": by, "PES_RG_sqrt": float(np.sqrt(rg)), "PES_BY_sqrt": float(np.sqrt(by))}


# Each tray spans 22 cap positions (0-indexed slice bounds) with its own
# wrap boundary caps (the cap just outside each tray's ends), distinct from
# the whole-circle wrap err_vals uses.
_TRAY_BOUNDS = [(0, 22), (22, 43), (43, 64), (64, 85)]
_TRAY_EXT = [(84, 22), (21, 43), (42, 64), (63, 85)]


def tes_trays(caps: np.ndarray) -> dict:
    """TES computed separately within each of the 4 trays (22-23 caps
    each), using that tray's own local wrap boundaries rather than the
    whole 85-cap circle -- captures whether errors cluster in one tray."""
    caps = np.asarray(caps, dtype=int)
    tes_tray = np.zeros(4)
    for t, (start, end) in enumerate(_TRAY_BOUNDS):
        left, right = _TRAY_EXT[t]
        padded = np.concatenate(([left], caps[start:end], [right]))
        tes_tray[t] = _wrap_error(padded).sum()
    whole = float(tes_tray.sum())
    return {"TES_tray": tes_tray, "TES_tray_sqrt": np.sqrt(tes_tray), "TES_whole": whole, "TES_whole_sqrt": float(np.sqrt(whole))}


# Vingrys-King-Smith U-V coordinates for caps 0-85 (cap 0 duplicates cap 85,
# closing the circle) -- transcribed verbatim from templateCode/FM100.py.
_VKS_UV = np.array(
    [
        [43.57, 4.76], [43.18, 8.03], [44.37, 11.34], [44.07, 13.62], [44.95, 16.04],
        [44.11, 18.52], [42.92, 20.64], [42.02, 22.49], [42.28, 25.15], [40.96, 27.78],
        [37.68, 29.55], [37.11, 32.95], [35.41, 35.94], [33.38, 38.03], [30.88, 39.59],
        [28.99, 43.07], [25.00, 44.12], [22.87, 46.44], [18.86, 45.87], [15.47, 44.97],
        [13.01, 42.12], [10.91, 42.85], [8.49, 41.35], [3.11, 41.70], [0.68, 39.23],
        [-1.70, 39.23], [-4.14, 36.66], [-6.57, 32.41], [-8.53, 33.19], [-10.98, 31.47],
        [-15.07, 27.89], [-17.13, 26.31], [-19.39, 23.82], [-21.93, 22.52], [-23.40, 20.14],
        [-25.32, 17.76], [-25.10, 13.29], [-26.58, 11.87], [-27.35, 9.52], [-28.41, 7.26],
        [-29.54, 5.10], [-30.37, 2.63], [-31.07, 0.10], [-31.72, -2.42], [-31.44, -5.13],
        [-32.26, -8.16], [-29.86, -9.51], [-31.13, -10.59], [-31.04, -14.30], [-29.10, -17.32],
        [-29.67, -19.59], [-28.61, -22.65], [-27.76, -26.66], [-26.31, -29.24], [-23.16, -31.24],
        [-21.31, -32.92], [-19.15, -33.17], [-16.00, -34.90], [-14.10, -35.21], [-12.47, -35.84],
        [-10.55, -37.74], [-8.49, -34.78], [-7.21, -35.44], [-5.16, -37.08], [-3.00, -35.95],
        [-0.31, -33.94], [1.55, -34.50], [3.68, -30.63], [5.88, -31.18], [8.46, -29.46],
        [9.75, -29.46], [12.24, -27.35], [15.61, -25.68], [19.63, -24.79], [21.20, -22.83],
        [25.60, -20.51], [26.94, -18.40], [29.39, -16.29], [32.93, -12.30], [34.96, -11.57],
        [38.24, -8.88], [39.06, -6.81], [39.51, -3.03], [40.90, -1.50], [42.80, 0.60],
        [43.57, 4.76],
    ]
)
_VKS_NORMAL_RADIUS = 2.525249  # normative "unit circle" radius, for Cindex


def vks(caps: np.ndarray) -> dict:
    """Vingrys-King-Smith confusion-ellipse metrics: fitted ellipse angle,
    major/minor radii, total error radius, selectivity index (Sindex,
    elongation) and confusion index (Cindex, size relative to a normative
    subject)."""
    caps = np.asarray(caps, dtype=int)
    if caps.size != N_CAPS:
        raise ValueError(f"caps must have {N_CAPS} entries, got {caps.size}")
    if not np.array_equal(np.sort(caps), np.arange(1, N_CAPS + 1)):
        raise ValueError("caps must contain each of 1..85 exactly once")

    caps0 = np.concatenate(([0], caps))
    du = _VKS_UV[caps0[1:], 0] - _VKS_UV[caps0[:-1], 0]
    dv = _VKS_UV[caps0[1:], 1] - _VKS_UV[caps0[:-1], 1]

    u2, v2, uv = np.sum(du**2), np.sum(dv**2), np.sum(du * dv)
    d = u2 - v2
    a0 = np.pi / 4 if d == 0 else 0.5 * np.arctan2(2 * uv, d)
    a1 = a0 + np.pi / 2

    def _inertia(angle):
        return u2 * np.sin(angle) ** 2 + v2 * np.cos(angle) ** 2 - 2 * uv * np.sin(angle) * np.cos(angle)

    i0, i1 = _inertia(a0), _inertia(a1)
    if i1 > i0:
        a0, a1, i0, i1 = a1, a0, i1, i0

    n = len(caps0) - 2
    r0, r1 = np.sqrt(i0 / n), np.sqrt(i1 / n)  # major, minor radius
    total_err = np.sqrt(r0**2 + r1**2)

    return {
        "VKS_Angle": float(np.degrees(a1)),
        "VKS_MajRad": float(r0),
        "VKS_MinRad": float(r1),
        "VKS_TotErr": float(total_err),
        "VKS_Sindex": float(r0 / r1),
        "VKS_Cindex": float(r0 / _VKS_NORMAL_RADIUS),
    }


def score_row(caps: np.ndarray) -> dict:
    """All scores for one (subject, session)'s caps: TES, PES, tray TES,
    VKS, merged into one flat dict."""
    result = {}
    result.update(tes(caps))
    result.update(pes(caps))
    trays = tes_trays(caps)
    result.update({f"TES_tray{i + 1}": v for i, v in enumerate(trays["TES_tray"])})
    result.update(vks(caps))
    return result


def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    """score_row for every (subject, session) row in a loader.load_fm100_raw
    table, joined with its identifying columns. Computed live, not
    persisted -- 69 rows, cheap to recompute on every load."""
    rows = [{"sub_id": r.sub_id, "session": r.session, "group": r.group, "subgroup": r.subgroup, **score_row(r.caps)} for r in df.itertuples()]
    return pd.DataFrame(rows)
