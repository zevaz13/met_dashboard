"""Load SSVEP grid experiment .mat files, and write the derived tidy CSVs."""

import scipy.io as sio


def load_ssvep(path: str) -> dict:
    """Load a MET*.mat file, dropping MATLAB header keys."""
    raw = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def to_rows(d: dict, filename: str) -> tuple[dict, list[dict], list[dict]]:
    """Convert one loaded .mat dict into (metadata_row, runmap_rows, baseline_rows).

    red_idx/green_idx are 0-based positions into redArray/greenArray. run/trial
    are 1-based.

    runMap's first axis is GREEN and its second is RED, despite mapDIM
    reporting "RED_GREEN_RUN". Confirmed three ways: the MATLAB templates plot
    it as imagesc(redArray, greenArray, runMap_slice), which puts the array's
    rows on the green axis; the independent reference image ssveps/CTRdata.png
    has its minimum at red 2133 / green 889, which only matches if the second
    axis is red; and decoding that image and correlating against this data
    gives r=0.92 in this orientation vs r=0.70 transposed. We unpack it in the
    file's own order and emit truthful red_idx/green_idx, so every grid
    downstream is genuinely indexed [red_idx, green_idx].
    """
    sub_id, session = d["SubID"], int(d["session"])
    metadata_row = {"filename": filename, "sub_id": sub_id, "session": session, "group": d["group"], "subgroup": d["subgroup"]}

    n_green, n_red, n_runs = d["runMap"].shape
    runmap_rows = [
        {
            "sub_id": sub_id,
            "session": session,
            "run": run + 1,
            "red_idx": red_idx,
            "green_idx": green_idx,
            "value": d["runMap"][green_idx, red_idx, run],
        }
        for red_idx in range(n_red)
        for green_idx in range(n_green)
        for run in range(n_runs)
    ]

    n_trials, n_runs = d["baselines"].shape
    baseline_rows = [
        {"sub_id": sub_id, "session": session, "run": run + 1, "trial": trial + 1, "value": d["baselines"][trial, run]}
        for trial in range(n_trials)
        for run in range(n_runs)
    ]

    return metadata_row, runmap_rows, baseline_rows


# --- derived-CSV writing, shared by build_derived.py and update_derived.py ---
# Both scripts write the same three files, so they share one writer: identical
# column order, identical row order, identical float formatting. Without this
# they disagree on ~9.5k of 25k rows' text (same values, different rendering)
# and on row order, producing a huge spurious diff whenever the other script runs.

CSV_COLUMNS = {
    "metadata": ["filename", "sub_id", "session", "group", "subgroup"],
    "runmap": ["sub_id", "session", "run", "red_idx", "green_idx", "value"],
    "baselines": ["sub_id", "session", "run", "trial", "value"],
}

CSV_SORT_KEYS = {
    "metadata": ["sub_id", "session"],
    "runmap": ["sub_id", "session", "run", "red_idx", "green_idx"],
    "baselines": ["sub_id", "session", "run", "trial"],
}


def write_derived_csv(path: str, df, kind: str) -> None:
    """Write one derived CSV in the canonical column/row order.

    float_format="%.17g": pandas' default to_csv formatting does not always
    round-trip float64 exactly; 17 significant digits guarantees it does.

    infer_objects() first, because float_format is ignored on object-dtype
    columns -- and concatenating onto an empty DataFrame (update_derived.py's
    first-run path) leaves `value` as object, which silently defeats it.
    """
    df.reindex(columns=CSV_COLUMNS[kind]).infer_objects().sort_values(CSV_SORT_KEYS[kind]).to_csv(
        path, index=False, float_format="%.17g"
    )
