import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import streamlit as st

from _pagesetup import CATEGORY_OPTIONS, sidebar_mode_header, use_scripts

mods = use_scripts("ssveps/scripts", "analysis", "permutation", "plotting")
analysis, permutation, plotting = mods["analysis"], mods["permutation"], mods["plotting"]
beh_mods = use_scripts("beh/scripts", "loader")
beh_loader = beh_mods["loader"]
ssvepbeh_mods = use_scripts("ssvepBeh/scripts", "plotting")
overlap_plotting = ssvepbeh_mods["plotting"]

st.set_page_config(page_title="SSVEP", page_icon="🎨", layout="wide")
st.title("SSVEP (EEG grid response)")

INDIVIDUALS_COLOR = "#f1c232"
SESSION = 1  # every subject has a session-1 row (matches every ssveps/ notebook)
OVERLAY_SIZE, OVERLAY_ALPHA = 8, 0.4  # smaller/more transparent than the notebooks' defaults -- less screen space here

NORMALIZE_OPTIONS = {
    "percent change (default)": analysis.DEFAULT_NORMALIZE,
    "db": {"scope": "run", "trials": "all", "method": "db"},
    "raw": None,
}
# One resolved cmap per normalization, always -- not a partial override with
# a None fallback. percent/raw match plot_groups_side_by_side's own
# cmap or _default_cmap(normalize is not None) default exactly (no visual
# change); db is viridis (PLANdashboard.md M2, was DIVERGING_GREEN_RED).
# Resolving all three here means the click-overlay views below can pass
# this cmap straight through instead of falling back to a hardcoded
# ramp when it's unset, which was the actual bug: overlap_plotting.EEG_CMAP
# (viridis) firing for percent/raw regardless of the sidebar selection.
CMAP_OVERRIDES = {
    "percent change (default)": plotting.DIVERGING_BLUE_RED,
    "db": overlap_plotting.EEG_CMAP,
    "raw": plotting.SEQUENTIAL_BLUE,
}


@st.cache_data
def load_data():
    return analysis.load_runmap(), analysis.load_baselines(), analysis.load_metadata()


@st.cache_data
def load_behavioral():
    return beh_loader.load_behavioral()


runmap_df, baselines_df, metadata_df = load_data()
beh_df = load_behavioral()


@st.cache_data(show_spinner="Running permutation test...")
def run_permutation(_runmap_df, _baselines_df, _metadata_df, session, group1, subgroup1, group2, subgroup2, normalize, n_perm):
    return permutation.permutation_test_weighted(
        _runmap_df, _baselines_df, _metadata_df, session,
        group1=group1, subgroup1=subgroup1, group2=group2, subgroup2=subgroup2,
        normalize=normalize, n_perm=n_perm, seed=0,
    )


mode = st.segmented_control("View", ["Groups", "Individuals"], default="Groups")

if mode == "Groups":
    sidebar_mode_header("Groups (up to 4)")
    selected_labels = st.sidebar.multiselect(
        "Categories", list(CATEGORY_OPTIONS), default=["HC", "protan", "deutan"], max_selections=4
    )
    categories = [{"label": label, **CATEGORY_OPTIONS[label]} for label in selected_labels]

    st.sidebar.header("Normalization")
    normalize_label = st.sidebar.radio("Normalization", list(NORMALIZE_OPTIONS), index=0, key="groups_normalize")
    normalize = NORMALIZE_OPTIONS[normalize_label]
    cmap = CMAP_OVERRIDES[normalize_label]

    st.sidebar.header("Behavioral overlay")
    show_clicks = st.sidebar.checkbox("Show behavioral clicks on the maps", key="groups_show_clicks")

    st.sidebar.header("Significance test")
    n_perm = st.sidebar.slider("Permutations", 100, 1000, 300, step=100)

    st.subheader("Mean response grids")
    if categories:
        if show_clicks:
            sub_id_lists = [analysis.subjects_in_group(metadata_df, SESSION, group=cat.get("group"), subgroup=cat.get("subgroup")) for cat in categories]
            grids = [analysis.mean_grid_across_subjects(runmap_df, baselines_df, sub_ids, SESSION, normalize=normalize) for sub_ids in sub_id_lists]
            clicks_dfs = [beh_df[beh_df["sub_id"].isin(sub_ids)] for sub_ids in sub_id_lists]
            titles = [f"{cat['label']} (n={len(sub_ids)})" for cat, sub_ids in zip(categories, sub_id_lists)]
            fig = overlap_plotting.plot_grids_with_clicks(
                grids, clicks_dfs, titles, s=OVERLAY_SIZE, alpha=OVERLAY_ALPHA, cmap=cmap,
                diverging=normalize is not None, suptitle=f"session {SESSION} -- groups side by side, with behavioral clicks",
            )
        else:
            fig = plotting.plot_groups_side_by_side(runmap_df, baselines_df, metadata_df, SESSION, categories, normalize=normalize, cmap=cmap)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Select at least one category in the sidebar.")

    st.subheader("Individual subjects within a group")
    subject_category = st.selectbox("Category to break down by subject", ["(none)"] + selected_labels)
    if subject_category != "(none)":
        cat = {"label": subject_category, **CATEGORY_OPTIONS[subject_category]}
        fig = plotting.plot_subjects_side_by_side(
            runmap_df, baselines_df, metadata_df, SESSION, group=cat.get("group"), subgroup=cat.get("subgroup"), normalize=normalize, cmap=cmap
        )
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Pairwise cluster-permutation significance")
    st.caption("Cluster-weight-corrected two-tailed permutation test (Maris & Oostenveld style) between each pair's mean grid.")
    if len(categories) >= 2:
        rows = []
        for cat1, cat2 in itertools.combinations(categories, 2):
            result = run_permutation(
                runmap_df, baselines_df, metadata_df, SESSION,
                cat1.get("group"), cat1.get("subgroup"), cat2.get("group"), cat2.get("subgroup"),
                normalize, n_perm,
            )
            cluster_pvals = [c["pvalue"] for c in result["cluster_results"]]
            min_p = round(min(cluster_pvals), 4) if cluster_pvals else None
            rows.append(
                {
                    "pair": f"{cat1['label']} vs {cat2['label']}",
                    "min cluster p_value": min_p if min_p is not None else "no significant cluster",
                    "n1": result["n1"],
                    "n2": result["n2"],
                }
            )
        st.table(rows)
    else:
        st.info("Select at least two categories to compare.")

else:
    sidebar_mode_header("Individual participants", INDIVIDUALS_COLOR)
    subjects = st.sidebar.multiselect(
        "Subjects (any group)", sorted(metadata_df.loc[metadata_df["session"] == SESSION, "sub_id"].unique()), key="individuals_tab_subjects"
    )

    st.sidebar.header("Normalization")
    normalize_label = st.sidebar.radio("Normalization", list(NORMALIZE_OPTIONS), index=0, key="individuals_normalize")
    normalize = NORMALIZE_OPTIONS[normalize_label]
    cmap = CMAP_OVERRIDES[normalize_label]

    st.sidebar.header("Behavioral overlay")
    show_clicks = st.sidebar.checkbox("Show behavioral clicks on the maps", key="individuals_show_clicks")

    st.subheader("Selected participants, side by side")
    if subjects:
        if show_clicks:
            grids = [analysis.mean_grid(runmap_df, baselines_df, sub_id, SESSION, normalize=normalize) for sub_id in subjects]
            clicks_dfs = [beh_df[beh_df["sub_id"] == sub_id] for sub_id in subjects]
            fig = overlap_plotting.plot_grids_with_clicks(
                grids, clicks_dfs, subjects, s=OVERLAY_SIZE, alpha=OVERLAY_ALPHA, cmap=cmap,
                diverging=normalize is not None, suptitle=f"session {SESSION} -- selected participants, with behavioral clicks",
            )
        else:
            fig = plotting.plot_subjects_side_by_side(runmap_df, baselines_df, metadata_df, SESSION, sub_ids=subjects, normalize=normalize, cmap=cmap)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Select one or more participants in the sidebar -- not limited to a single group.")
