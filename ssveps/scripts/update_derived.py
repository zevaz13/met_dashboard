"""Incrementally add or refresh derived files as raw .mat files are added.

For each raw .mat file under RAW_DIR:
  - new (sub_id, session): added to metadata.csv, runmap.csv, baselines.csv.
  - existing (sub_id, session): asks whether to overwrite its runmap.csv and
    baselines.csv rows with the current raw file's data. metadata.csv's
    group/subgroup is set once, at first creation, and is never touched here --
    edit metadata.csv by hand to change it (e.g. correcting a group label).

Run: uv run python scripts/update_derived.py
"""

import glob
import json
import os

import pandas as pd

from loader import CSV_COLUMNS, load_ssvep, to_rows, write_derived_csv

RAW_DIR = "/home/sebas/data/ssveps"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "files")


def _read_or_empty(path: str, columns: list[str]) -> pd.DataFrame:
    # keep_default_na=False: subgroup's literal "NA" value must not become NaN
    return pd.read_csv(path, keep_default_na=False) if os.path.exists(path) else pd.DataFrame(columns=columns)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    metadata_path = os.path.join(OUT_DIR, "metadata.csv")
    runmap_path = os.path.join(OUT_DIR, "runmap.csv")
    baselines_path = os.path.join(OUT_DIR, "baselines.csv")
    grid_path = os.path.join(OUT_DIR, "grid.json")

    metadata_df = _read_or_empty(metadata_path, CSV_COLUMNS["metadata"])
    runmap_df = _read_or_empty(runmap_path, CSV_COLUMNS["runmap"])
    baselines_df = _read_or_empty(baselines_path, CSV_COLUMNS["baselines"])
    grid = json.load(open(grid_path)) if os.path.exists(grid_path) else None

    existing_keys = set(zip(metadata_df["sub_id"], metadata_df["session"]))
    new_metadata_rows, new_runmap_rows, new_baselines_rows, overwrite_keys = [], [], [], []

    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.mat"))):
        d = load_ssvep(path)
        metadata_row, run_rows, baseline_rows = to_rows(d, os.path.basename(path))
        key = (metadata_row["sub_id"], metadata_row["session"])

        if grid is None:
            grid = {
                "baseDIM": d["baseDIM"],
                "mapDIM": d["mapDIM"],
                "redArray": d["redArray"].tolist(),
                "greenArray": d["greenArray"].tolist(),
            }
        elif d["redArray"].tolist() != grid["redArray"] or d["greenArray"].tolist() != grid["greenArray"]:
            print(f"WARNING: {path} has different redArray/greenArray than grid.json -- grid.json left unchanged")

        if key not in existing_keys:
            new_metadata_rows.append(metadata_row)
            new_runmap_rows.extend(run_rows)
            new_baselines_rows.extend(baseline_rows)
            print(f"added {key[0]} session {key[1]}")
        else:
            answer = input(f"{key[0]} session {key[1]} already exists -- overwrite its runmap/baselines data? [y/N] ")
            if answer.strip().lower() == "y":
                overwrite_keys.append(key)
                new_runmap_rows.extend(run_rows)
                new_baselines_rows.extend(baseline_rows)
                print(f"overwritten {key[0]} session {key[1]} (metadata.csv group/subgroup left untouched)")
            else:
                print(f"skipped {key[0]} session {key[1]}")

    metadata_df = pd.concat([metadata_df, pd.DataFrame(new_metadata_rows)], ignore_index=True)

    if overwrite_keys:
        overwrite_index = pd.MultiIndex.from_tuples(overwrite_keys)
        runmap_df = runmap_df[~runmap_df.set_index(["sub_id", "session"]).index.isin(overwrite_index)]
        baselines_df = baselines_df[~baselines_df.set_index(["sub_id", "session"]).index.isin(overwrite_index)]
    runmap_df = pd.concat([runmap_df, pd.DataFrame(new_runmap_rows)], ignore_index=True)
    baselines_df = pd.concat([baselines_df, pd.DataFrame(new_baselines_rows)], ignore_index=True)

    write_derived_csv(metadata_path, metadata_df, "metadata")
    write_derived_csv(runmap_path, runmap_df, "runmap")
    write_derived_csv(baselines_path, baselines_df, "baselines")
    with open(grid_path, "w") as f:
        json.dump(grid, f, indent=2)

    print(f"added {len(new_metadata_rows)}, overwrote {len(overwrite_keys)} -- "
          f"metadata.csv now has {len(metadata_df)} rows")


if __name__ == "__main__":
    main()
