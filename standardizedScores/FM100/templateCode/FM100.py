import os
import pandas as pd
import numpy as np


FM100_DATA_DIR = r""

# ---------------------------------------------------------
# 1. Load participants metadata
# ---------------------------------------------------------

def load_participant_info():
    """
    Loads partINFO.csv which contains SubID, Group, Subgroup.
    """
    path = os.path.join(FM100_DATA_DIR, "partINFO.csv")
    return pd.read_csv(path)

# ---------------------------------------------------------
# 1.b Load participants FM100 results
# ---------------------------------------------------------

def load_FM100_raw():
    path = f"{FM100_DATA_DIR}/repeatedSessionsPY.txt"
    df = pd.read_csv(
        path,
        header=None,          # do NOT treat any row as header
        skiprows=1,           # skip the first line entirely
        skip_blank_lines=True,
        encoding="utf-8-sig"  # handles BOM if present
    )
    return df

# ---------------------------------------------------------
# 2. based on raw FM100 data, parse sessions
# ---------------------------------------------------------

def parse_session_and_id(raw_id):
    """
    Extract validID and session number from FM100 Reference field.
    MET000  -> session 1
    MET000b -> session 2
    MET000c -> session 3
    """
    if raw_id.endswith("b"):
        return raw_id[:-1], 2
    elif raw_id.endswith("c"):
        return raw_id[:-1], 3
    else:
        return raw_id, 1
# ---------------------------------------------------------
# 3. based on metadata, parse group and subgroup
# ---------------------------------------------------------

def determine_group(valid_id, info_table):
    """
    Look up Group and Subgroup from partINFO.csv.
    """
    row = info_table[info_table["SubID"] == valid_id]

    if row.empty:
        return "UNKNOWN", "NA"

    group = row.iloc[0]["Group"]
    subgroup = row.iloc[0]["Subgroup"]
    return group, subgroup

# ---------------------------------------------------------
# 4. wrapper to get info from metadata, based on raw data
# ---------------------------------------------------------

def identify_participant(raw_id, info_table):
    """
    Given a raw FM100 Reference ID (e.g., MET016b),
    return ValidID, Session, Group, Subgroup.
    """
    valid_id, session = parse_session_and_id(raw_id)
    group, subgroup = determine_group(valid_id, info_table)

    return {
        "ValidID": valid_id,
        "Session": session,
        "Group": group,
        "Subgroup": subgroup
    }


# ---------------------------------------------------------
# 5. Compute FM100 TES-related metrics
# ---------------------------------------------------------

def FM100_TESwhole(arranged_caps):
    """
    Compute TES and sqrt(TES) for the FM100 test using the same logic
    as the original software and your MATLAB implementation.

    Parameters
    ----------
    arranged_caps : array-like of length 85
        The ordered cap positions (1–85) after rearrangement.

    Returns
    -------
    TES : float
        Total Error Score.
    SRTES : float
        Square root of TES.
    err_vals : np.ndarray
        Error value for each of the 85 caps.
    """

    caps = np.asarray(arranged_caps)

    if caps.size != 85:
        raise ValueError("Input must be a vector of 85 cap positions.")

    # Circular hue distance function
    def circ_hue_dist(a, b):
        return np.minimum((a - b) % 85, (b - a) % 85)

    # Pad for circular wraparound: [last, ..., first]
    tray_caps = np.concatenate(([caps[-1]], caps, [caps[0]]))

    err_vals = np.zeros(85, dtype=float)
    tray_error_sum = 0.0

    # Loop exactly like MATLAB: i = 2:(len-1)
    for i in range(1, len(tray_caps) - 1):
        prev_cap = tray_caps[i - 1]
        curr_cap = tray_caps[i]
        next_cap = tray_caps[i + 1]

        err = circ_hue_dist(prev_cap, curr_cap) + circ_hue_dist(next_cap, curr_cap) - 2
        tray_error_sum += err
        err_vals[i - 1] = err

    TES = float(tray_error_sum)
    SRTES = np.sqrt(TES)

    return TES, SRTES, err_vals

# ---------------------------------------------------------
# 6. Vingrys–King–Smith (VKS) Metrics
# ---------------------------------------------------------

def compute_VKS_metrics(arranged_caps, silent=True):
    """
    Compute Vingrys–King–Smith ellipse metrics for FM100 Hue Test.

    Parameters
    ----------
    arranged_caps : array-like of length 85
        Cap order (1–85)
    silent : bool
        If False, prints the metrics.

    Returns
    -------
    dict with:
        Angle, MajRad, MinRad, TotErr, Sindex, Cindex
    """

    caps = np.asarray(arranged_caps, dtype=int)

    # Validate input
    if caps.size != 85:
        raise ValueError("Input must be a vector of 85 elements.")
    if not np.array_equal(np.sort(caps), np.arange(1, 86)):
        raise ValueError("Input must contain integers 1–85 without repetition.")

    # Vingrys U-V coordinates (caps 0–85)
    dataVKS = np.array([
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
        [43.57, 4.76]
    ])

    # Prepend cap 0
    caps0 = np.concatenate(([0], caps))

    # Compute DU, DV
    DU = dataVKS[caps0[1:], 0] - dataVKS[caps0[:-1], 0]
    DV = dataVKS[caps0[1:], 1] - dataVKS[caps0[:-1], 1]

    U2 = np.sum(DU**2)
    V2 = np.sum(DV**2)
    UV = np.sum(DU * DV)
    D = U2 - V2

    # Angle
    if D == 0:
        A0 = np.pi / 4
    else:
        A0 = 0.5 * np.arctan2(2 * UV, D)

    A1 = A0 + np.pi/2

    I0 = U2 * np.sin(A0)**2 + V2 * np.cos(A0)**2 - 2 * UV * np.sin(A0) * np.cos(A0)
    I1 = U2 * np.sin(A1)**2 + V2 * np.cos(A1)**2 - 2 * UV * np.sin(A1) * np.cos(A1)

    # Ensure A0 = major axis
    if I1 > I0:
        A0, A1 = A1, A0
        I0, I1 = I1, I0

    n = len(caps0) - 2
    R0 = np.sqrt(I0 / n)  # major radius
    R1 = np.sqrt(I1 / n)  # minor radius
    R = np.sqrt(R0**2 + R1**2)
    R2 = 2.525249

    result = {
        "Angle": np.degrees(A1),
        "MajRad": R0,
        "MinRad": R1,
        "TotErr": R,
        "Sindex": R0 / R1,
        "Cindex": R0 / R2
    }

    if not silent:
        print(result)

    return result

# ---------------------------------------------------------
# 7. PES (Red–Green, Blue–Yellow)
# ---------------------------------------------------------

def compute_PES(arranged_caps):
    caps = np.asarray(arranged_caps, dtype=int)

    # RG and BY index groups (MATLAB 1-indexed → Python 0-indexed)
    RG_indices = np.r_[13:34, 55:76] - 1
    BY_indices = np.r_[1:13, 34:55, 76:85] - 1

    real_idxs = np.r_[85, np.arange(1, 85)] - 1

    RG_idxs = np.where(np.isin(real_idxs, RG_indices))[0]
    BY_idxs = np.where(np.isin(real_idxs, BY_indices))[0]

    # Circular hue distance
    def circ(a, b):
        return min((a - b) % 85, (b - a) % 85)

    tray = np.concatenate(([caps[-1]], caps, [caps[0]]))
    err = np.zeros(85)

    for i in range(1, len(tray) - 1):
        prev, curr, nxt = tray[i-1], tray[i], tray[i+1]
        err[i-1] = circ(prev, curr) + circ(nxt, curr) - 2

    RG = np.sum(err[RG_idxs])
    BY = np.sum(err[BY_idxs])

    return {
        "PES_RG": RG,
        "PES_BY": BY,
        "PES_RG_sqrt": np.sqrt(RG),
        "PES_BY_sqrt": np.sqrt(BY)
    }

# ---------------------------------------------------------
# 8. Tray‑level TES
# ---------------------------------------------------------

def compute_TES_trays(arranged_caps):
    caps = np.asarray(arranged_caps, dtype=int)

    tray_bounds = [(0, 22), (22, 43), (43, 64), (64, 85)]
    tray_ext = [(84, 22), (21, 43), (42, 64), (63, 85)]

    def circ(a, b):
        return min((a - b) % 85, (b - a) % 85)

    TEStray = np.zeros(4)
    errstr = []

    for t, (start, end) in enumerate(tray_bounds):
        tray_caps = caps[start:end]
        left, right = tray_ext[t]
        tray_caps = np.concatenate(([left], tray_caps, [right]))

        tray_err = 0
        for i in range(1, len(tray_caps) - 1):
            prev, curr, nxt = tray_caps[i-1], tray_caps[i], tray_caps[i+1]
            e = circ(prev, curr) + circ(nxt, curr) - 2
            tray_err += e
            errstr.append(e)

        TEStray[t] = tray_err

    TESwhole = np.sum(TEStray)

    return {
        "TES_tray": TEStray,
        "TES_tray_sqrt": np.sqrt(TEStray),
        "TES_whole": TESwhole,
        "TES_whole_sqrt": np.sqrt(TESwhole),
        "err_vals": np.array(errstr)
    }

# ---------------------------------------------------------
# 8. Build and save a complete table
# ---------------------------------------------------------

def build_FM100_scores(save_csv=True):
    """
    Build full FM100 scoring table for all participants.

    Columns (in order):
    SubID, Session, Group, Subgroup,
    TES, SqrtTES,
    PES_RG, PES_BY, PES_RG_sqrt, PES_BY_sqrt,
    TES_tray1, TES_tray2, TES_tray3, TES_tray4,
    VKS_Angle, VKS_MajRad, VKS_MinRad, VKS_Sindex, VKS_Cindex

    Returns
    -------
    df_scores : pandas.DataFrame
    """

    # --- Load data ---
    FM100 = load_FM100_raw()
    info = load_participant_info()

    rows = []

    for _, row in FM100.iterrows():
        # ID and caps from raw file
        raw_id = row.iloc[3]                 # Reference
        caps   = row.iloc[16:101].to_numpy() # 85 caps

        # Participant metadata
        meta = identify_participant(raw_id, info)

        # TES
        TES, SRTES, _ = FM100_TESwhole(caps)

        # PES
        pes = compute_PES(caps)

        # Tray TES
        trays = compute_TES_trays(caps)

        # VKS
        vks = compute_VKS_metrics(caps)

        rows.append({
            "SubID": meta["ValidID"],
            "Session": meta["Session"],
            "Group": meta["Group"],
            "Subgroup": meta["Subgroup"],
            "TES": TES,
            "SqrtTES": SRTES,
            "PES_RG": pes["PES_RG"],
            "PES_BY": pes["PES_BY"],
            "PES_RG_sqrt": pes["PES_RG_sqrt"],
            "PES_BY_sqrt": pes["PES_BY_sqrt"],
            "TES_tray1": trays["TES_tray"][0],
            "TES_tray2": trays["TES_tray"][1],
            "TES_tray3": trays["TES_tray"][2],
            "TES_tray4": trays["TES_tray"][3],
            "VKS_Angle": vks["Angle"],
            "VKS_MajRad": vks["MajRad"],
            "VKS_MinRad": vks["MinRad"],
            "VKS_Sindex": vks["Sindex"],
            "VKS_Cindex": vks["Cindex"],
        })

    # DataFrame with explicit column order
    col_order = [
        "SubID", "Session", "Group", "Subgroup",
        "TES", "SqrtTES",
        "PES_RG", "PES_BY", "PES_RG_sqrt", "PES_BY_sqrt",
        "TES_tray1", "TES_tray2", "TES_tray3", "TES_tray4",
        "VKS_Angle", "VKS_MajRad", "VKS_MinRad", "VKS_Sindex", "VKS_Cindex",
    ]

    df_scores = pd.DataFrame(rows)[col_order]

    if save_csv:
        # FM100.py is in: .../metamers/src/metamers/scores/FM100.py
        module_dir = os.path.dirname(os.path.abspath(__file__))      # .../src/metamers/scores
        metamers_pkg_dir = os.path.dirname(module_dir)               # .../src/metamers
        src_dir = os.path.dirname(metamers_pkg_dir)                  # .../src
        project_root = os.path.dirname(src_dir)                      # .../metamers

        out_dir = os.path.join(project_root, "standardScores")
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, "FM100_scores.csv")
        df_scores.to_csv(out_path, index=False)
        print(f"FM100 scores saved to: {out_path}")


    return df_scores