import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import streamlit as st

from _pagesetup import CATEGORY_OPTIONS, sidebar_mode_header, use_scripts

mods = use_scripts("beh/scripts", "loader", "features", "comparisons", "plotting")
loader, features, comparisons, plotting = mods["loader"], mods["features"], mods["comparisons"], mods["plotting"]

st.set_page_config(page_title="Behavioral", page_icon="🎨", layout="wide")
st.title("Behavioral (manual match)")

SHAPE_FEATURES_COLOR = "#5f8158"
RAW_CLICKS_MAX_SUBJECTS = 8
RAW_CLICKS_GRID_COLS = 4  # even 4+4 layout at the 8-subject cap (dashboard M3)

SHAPE_FEATURES = ["orientation_deg", "along_var", "perp_var"]
FEATURE_DESCRIPTIONS = {
    "orientation_deg": "Which way a subject's line of clicks points (0-180 degrees). "
    "This is the strongest known protan/deutan split in this dataset.",
    "along_var": "How spread out the clicks are *along* that line -- a wide vs. narrow match range.",
    "perp_var": "How tightly the clicks cluster *off* that line -- match consistency/noise, independent of where the line points.",
}


@st.cache_data
def load_data():
    return loader.load_behavioral()


df = load_data()

mode = st.segmented_control("View", ["Raw clicks", "Shape features"], default="Raw clicks")

if mode == "Raw clicks":
    sidebar_mode_header("Groups (up to 4)")
    selected_labels = st.sidebar.multiselect(
        "Categories", list(CATEGORY_OPTIONS), default=["HC", "protan", "deutan"], max_selections=4
    )
    categories = [{"label": label, **CATEGORY_OPTIONS[label]} for label in selected_labels]

    st.sidebar.header(f"Participants (up to {RAW_CLICKS_MAX_SUBJECTS})")
    raw_subjects = st.sidebar.multiselect(
        "Subjects", sorted(df["sub_id"].unique()), max_selections=RAW_CLICKS_MAX_SUBJECTS, key="raw_tab_subjects"
    )

    st.subheader("Groups side by side")
    if categories:
        fig = plotting.plot_groups_side_by_side(df, categories)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Select at least one category in the sidebar.")

    overlay_allowed = len(raw_subjects) <= len(plotting.SCATTER4_COLORS)
    subject_view = st.radio("Participant view", ["Side by side", "Overlaid (one plot)"], horizontal=True, key="raw_tab_view")
    if not overlay_allowed and subject_view == "Overlaid (one plot)":
        st.caption(f"Overlaid view needs {len(plotting.SCATTER4_COLORS)} or fewer participants to keep each one visually distinguishable -- showing side by side instead.")
    if raw_subjects:
        if subject_view == "Side by side" or not overlay_allowed:
            fig = plotting.plot_subjects_grid(df, sub_ids=raw_subjects, max_cols=RAW_CLICKS_GRID_COLS)
            st.pyplot(fig)
            plt.close(fig)
        else:
            ax = plotting.plot_subjects_cloud_overlay(df, raw_subjects)
            st.pyplot(ax.figure)
            plt.close(ax.figure)
    else:
        st.info("Select one or more participants above to see their raw clicks.")

else:
    sidebar_mode_header("Groups (up to 4)", SHAPE_FEATURES_COLOR)
    feature_labels = st.sidebar.multiselect(
        "Categories", list(CATEGORY_OPTIONS), default=["HC", "protan", "deutan"], max_selections=4, key="feature_tab_categories"
    )
    feature_categories = [{"label": label, **CATEGORY_OPTIONS[label]} for label in feature_labels]

    x_feature = st.sidebar.selectbox("Shape feature (x-axis)", SHAPE_FEATURES, index=0)
    y_feature = st.sidebar.selectbox("Shape feature (y-axis)", SHAPE_FEATURES, index=2)

    st.sidebar.header("Single subject (optional)")
    subject = st.sidebar.selectbox("Subject", ["(none)"] + sorted(df["sub_id"].unique()), key="feature_tab_subject")

    st.subheader("What these features mean")
    for feat in SHAPE_FEATURES:
        st.markdown(f"- **{feat}** -- {FEATURE_DESCRIPTIONS[feat]}")

    st.subheader("Shape-feature space (PCA line per subject)")
    if feature_categories:
        ax = plotting.plot_feature_space(df, feature_categories, x_feature=x_feature, y_feature=y_feature)
        st.pyplot(ax.figure)
        plt.close(ax.figure)
    else:
        st.info("Select at least one category in the sidebar.")

    if subject != "(none)":
        st.subheader(f"{subject}'s click cloud, with fitted PCA line")
        ax = plotting.plot_subject_cloud(df, subject, show_fit=True)
        st.pyplot(ax.figure)
        plt.close(ax.figure)

    st.subheader("Pairwise comparisons")
    if len(feature_categories) >= 2:
        location_rows, shape_rows = [], []
        for cat1, cat2 in itertools.combinations(feature_categories, 2):
            pair_label = f"{cat1['label']} vs {cat2['label']}"
            location = comparisons.compare_groups(
                df,
                group1=cat1.get("group"),
                subgroup1=cat1.get("subgroup"),
                group2=cat2.get("group"),
                subgroup2=cat2.get("subgroup"),
            )
            location_rows.append({"pair": pair_label, "p_value": round(location["p_value"], 4), "n1": location["n1"], "n2": location["n2"]})
            shape = features.compare_shape_feature(
                df,
                x_feature,
                group1=cat1.get("group"),
                subgroup1=cat1.get("subgroup"),
                group2=cat2.get("group"),
                subgroup2=cat2.get("subgroup"),
            )
            shape_rows.append({"pair": pair_label, "p_value": round(shape["p_value"], 4), "n1": shape["n1"], "n2": shape["n2"]})

        st.write("Click location (Hotelling T²)")
        st.table(location_rows)
        st.write(f"{x_feature} (Mann-Whitney U)")
        st.table(shape_rows)
    else:
        st.info("Select at least two categories to compare.")
