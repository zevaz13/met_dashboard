"""Builds ssveps/files/subject_troughs.csv and group_troughs.csv from the tidy
runmap/baselines/metadata CSVs.

Straight recompute from the other derived files (no hand-edits to preserve),
so this is safe to rerun anytime after runmap.csv/metadata.csv change.
"""

import os

from analysis import FILES_DIR, group_troughs, load_baselines, load_metadata, load_runmap, subject_troughs

# Same categories used throughout ssveps/notebooks/03_group_comparisons.ipynb
# and 04_distributions.ipynb.
CATEGORIES = [
    {"label": "PD", "group": "PD"},
    {"label": "HC", "group": "CTR"},
    {"label": "protan", "subgroup": "protan"},
    {"label": "deutan", "subgroup": "deutan"},
]


def main() -> None:
    runmap_df = load_runmap()
    baselines_df = load_baselines()
    metadata_df = load_metadata()

    subject_df = subject_troughs(runmap_df, baselines_df, metadata_df)
    subject_df.to_csv(os.path.join(FILES_DIR, "subject_troughs.csv"), index=False)

    group_df = group_troughs(runmap_df, baselines_df, metadata_df, sessions=[1, 2], categories=CATEGORIES)
    group_df.to_csv(os.path.join(FILES_DIR, "group_troughs.csv"), index=False)

    print(f"subject_troughs.csv ({len(subject_df)} rows), group_troughs.csv ({len(group_df)} rows)")


if __name__ == "__main__":
    main()
