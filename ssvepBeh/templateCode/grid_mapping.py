import numpy as np

# Default canonical grid (from MATLAB)
DEFAULT_RED = np.array([0, 355.6, 711.1, 1066.7, 1422.2, 1777.8, 2133.3, 2488.9, 2844.4, 3200.0])
DEFAULT_GREEN = np.array([0, 222.2, 444.4, 666.7, 888.9, 1111.1, 1333.3, 1555.6, 1777.8, 2000.0])

def closest_grid_indices(points, red=DEFAULT_RED, green=DEFAULT_GREEN):
    """
    Map scattered behavioral points onto the EEG grid.

    Parameters
    ----------
    points : array-like, shape (N, 2)
        Behavioral points in continuous space (columns: Red, Green).

    red : array-like, shape (10,), optional
        Grid coordinates along the red axis.
        Defaults to canonical grid.

    green : array-like, shape (10,), optional
        Grid coordinates along the green axis.
        Defaults to canonical grid.

    Returns
    -------
    idx : ndarray, shape (N, 2)
        Closest grid indices for each point (ix, iy).

    outMat : ndarray, shape (10, 10)
        Subscription matrix: count of points per grid cell (transposed to match MATLAB).
    """

    points = np.asarray(points)
    x = points[:, 0]
    y = points[:, 1]

    # Compute closest red index for each point
    ix = np.argmin(np.abs(red.reshape(1, -1) - x.reshape(-1, 1)), axis=1)

    # Compute closest green index for each point
    iy = np.argmin(np.abs(green.reshape(1, -1) - y.reshape(-1, 1)), axis=1)

    idx = np.column_stack([ix, iy])

    # Build subscription matrix
    subs = np.zeros((10, 10), dtype=int)
    for k in range(len(idx)):
        subs[idx[k, 0], idx[k, 1]] += 1

    # Match MATLAB orientation
    outMat = subs.T

    return idx, outMat


def permWeighted2Dshifts(B, E, nPerm=5000):
    """
    Weighted spatial overlap test using 2D toroidal circular shifts.
    
    Parameters
    ----------
    B : ndarray, shape (10, 10)
        Behavioral count map.
    E : ndarray, shape (10, 10)
        EEG map.
    nPerm : int
        Number of permutations.

    Returns
    -------
    p_value : float
        One-sided p-value (observed LOWER than null).
    obs_stat : float
        Observed weighted overlap: sum(E * B_norm).
    null_stats : ndarray, shape (nPerm,)
        Null distribution of weighted overlaps.
    """

    B = np.asarray(B, dtype=float)
    E = np.asarray(E, dtype=float)

    totalB = B.sum()
    if totalB == 0:
        raise ValueError("Behavioral map B contains no visits.")

    # 1. Normalize behavioral map into a probability distribution
    B_norm = B / totalB

    # 2. Observed statistic
    obs_stat = np.sum(E * B_norm)

    # 3. Null distribution via toroidal circular shifts
    null_stats = np.zeros(nPerm)

    for p in range(nPerm):
        shiftX = np.random.randint(0, 100)
        shiftY = np.random.randint(0, 100)

        B_perm = np.roll(B_norm, shift=(shiftX, shiftY), axis=(0, 1))
        null_stats[p] = np.sum(E * B_perm)

    # 4. One-sided p-value: observed should be LOWER
    p_value = np.mean(null_stats <= obs_stat)

    return p_value, obs_stat, null_stats
