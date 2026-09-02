"""Full from-scratch rebuild of the derived files from all raw SSVEP .mat files.

Reads every METxxx[b].mat under RAW_DIR and overwrites ssveps/files/:
  - metadata.csv  one row per file: filename, sub_id, session, group, subgroup
  - grid.json     shared grid constants (redArray, greenArray, baseDIM, mapDIM)
  - runmap.csv    tidy: sub_id, session, run, red_idx, green_idx, value
  - baselines.csv tidy: sub_id, session, run, trial, value

Writes through loader.write_derived_csv, the same writer update_derived.py uses,
so the two scripts produce byte-identical files for the same data.

This always regenerates group/subgroup from the raw files, so it will wipe any
hand-edit made directly to metadata.csv (e.g. a manual group/subgroup
correction). For day-to-day updates that preserve such edits, use
update_derived.py instead -- this script is for an initial or intentional
full reset only.
"""

import glob
import json
import os

import pandas as pd

from loader import load_ssvep, to_rows, write_derived_csv

RAW_DIR = "/home/sebas/data/ssveps"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "files")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.mat")))

    metadata_rows, runmap_rows, baselines_rows = [], [], []
    grid = None

    for path in files:
        d = load_ssvep(path)
        metadata_row, run_rows, baseline_rows = to_rows(d, os.path.basename(path))
        metadata_rows.append(metadata_row)
        runmap_rows.extend(run_rows)
        baselines_rows.extend(baseline_rows)

        if grid is None:
            grid = {
                "baseDIM": d["baseDIM"],
                "mapDIM": d["mapDIM"],
                "redArray": d["redArray"].tolist(),
                "greenArray": d["greenArray"].tolist(),
            }

    write_derived_csv(os.path.join(OUT_DIR, "metadata.csv"), pd.DataFrame(metadata_rows), "metadata")
    write_derived_csv(os.path.join(OUT_DIR, "runmap.csv"), pd.DataFrame(runmap_rows), "runmap")
    write_derived_csv(os.path.join(OUT_DIR, "baselines.csv"), pd.DataFrame(baselines_rows), "baselines")
    with open(os.path.join(OUT_DIR, "grid.json"), "w") as f:
        json.dump(grid, f, indent=2)

    print(f"{len(files)} files -> metadata.csv ({len(metadata_rows)} rows), "
          f"runmap.csv ({len(runmap_rows)} rows), "
          f"baselines.csv ({len(baselines_rows)} rows), grid.json")


if __name__ == "__main__":
    main()
