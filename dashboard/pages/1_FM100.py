import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import streamlit as st

from _pagesetup import CATEGORY_OPTIONS, sidebar_mode_header, use_scripts

mods = use_scripts("standardizedScores/FM100/scripts", "loader", "scores", "plotting", "comparisons")
loader, plotting, comparisons = mods["loader"], mods["plotting"], mods["comparisons"]

st.set_page_config(page_title="FM100", page_icon="🎨", layout="wide")
st.title("FM100")

GROUP_COLOR = "#e34948"  # this repo's established red swatch (FULL_PALETTE's last slot, hue/'s HueR mnemonic)
PARTICIPANT_COLOR = "#4682b4"

# Cap colors default on for both styles (PLANdashboard.md M2) -- radial's
# show_cap_wheel already carries every cap's number, making the old
# angle/cap tick-label toggle a no-op, so it's gone rather than left dead.
def _cap_display_kwargs(kind: str) -> dict:
    return {"show_cap_wheel": kind == "radial", "show_cap_colors": kind == "linear"}


@st.cache_data
def load_data():
    return loader.load_fm100_raw()


df = load_data()

mode = st.segmented_control("View", ["Group", "Participant"], default="Group")

if mode == "Group":
    sidebar_mode_header("Group", GROUP_COLOR)
    selected_labels = st.sidebar.multiselect("Categories", list(CATEGORY_OPTIONS), default=["HC", "protan", "deutan"])
    categories = [{"label": label, **CATEGORY_OPTIONS[label]} for label in selected_labels]

    st.sidebar.header("Display")
    kind = st.sidebar.radio("Plot style", ["linear", "radial"], index=0, key="group_kind")
    window = st.sidebar.slider("Smoothing window", 1, 10, 1, key="group_window")

    st.sidebar.header("Compare to participant(s) (optional)")
    compare_subjects = st.sidebar.multiselect(
        "Participants to overlay", sorted(df["sub_id"].unique()), max_selections=len(plotting.SUBJECT_COLORS)
    )

    st.subheader("Group error profiles")
    if categories:
        if compare_subjects:
            ax = plotting.plot_group_vs_subjects_fm100(df, categories, compare_subjects, kind=kind, window=window, **_cap_display_kwargs(kind))
        else:
            ax = plotting.plot_group_fm100(df, categories, kind=kind, window=window, **_cap_display_kwargs(kind))
        st.pyplot(ax.figure)
        plt.close(ax.figure)
    else:
        st.info("Select at least one category in the sidebar.")

    st.subheader("Pairwise comparisons")
    feature = st.selectbox("Feature", comparisons.FEATURES, index=0)
    if len(categories) >= 2:
        rows = []
        for cat1, cat2 in itertools.combinations(categories, 2):
            result = comparisons.compare_fm100_feature(
                df,
                feature,
                group1=cat1.get("group"),
                subgroup1=cat1.get("subgroup"),
                group2=cat2.get("group"),
                subgroup2=cat2.get("subgroup"),
            )
            rows.append(
                {
                    "pair": f"{cat1['label']} vs {cat2['label']}",
                    "p_value": round(result["p_value"], 4),
                    "n1": result["n1"],
                    "n2": result["n2"],
                }
            )
        st.table(rows)
    else:
        st.info("Select at least two categories to compare.")

else:
    sidebar_mode_header("Participant", PARTICIPANT_COLOR)
    subjects = st.sidebar.multiselect(
        "Subject(s)", sorted(df["sub_id"].unique()), max_selections=len(plotting.SUBJECT_COLORS), key="participant_tab_subjects"
    )

    st.sidebar.header("Display")
    kind = st.sidebar.radio("Plot style", ["linear", "radial"], index=0, key="participant_kind")
    window = st.sidebar.slider("Smoothing window", 1, 10, 1, key="participant_window")

    if not subjects:
        st.info("Select one or more participants in the sidebar.")
    elif len(subjects) == 1:
        st.subheader(f"{subjects[0]}'s own profile (every session)")
        ax = plotting.plot_subject_fm100(df, subjects[0], kind=kind, window=window, **_cap_display_kwargs(kind))
        st.pyplot(ax.figure)
        plt.close(ax.figure)
    else:
        st.subheader("Participants (session 1 only)")
        ax = plotting.plot_subjects_fm100(df, subjects, kind=kind, window=window, **_cap_display_kwargs(kind))
        st.pyplot(ax.figure)
        plt.close(ax.figure)
