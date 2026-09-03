import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from _pagesetup import GROUP_DISPLAY, sidebar_mode_header, use_scripts

fm100_mods = use_scripts("standardizedScores/FM100/scripts", "loader")
fm100_loader = fm100_mods["loader"]
beh_mods = use_scripts("beh/scripts", "loader")
beh_loader = beh_mods["loader"]
ssvep_mods = use_scripts("ssveps/scripts", "analysis")
ssvep_analysis = ssvep_mods["analysis"]
roster_mods = use_scripts("participants/scripts", "roster")
roster = roster_mods["roster"]

st.set_page_config(page_title="Participants", page_icon="🎨", layout="wide")
st.title("Participants")

PARTICIPANTS_COLOR = "#4682b4"
GROUPS_COLOR = "#e34948"


@st.cache_data
def load_roster():
    fm100_df = fm100_loader.load_fm100_raw()
    beh_df = beh_loader.load_behavioral()
    ssvep_meta_df = ssvep_analysis.load_metadata()
    return roster.build_roster(fm100_df, beh_df, ssvep_meta_df)


def _filter_options(roster_df: pd.DataFrame) -> dict:
    """Label -> {"group": g} or {"subgroup": sg}, built from whatever
    values actually appear in the roster (unlike the other pages' fixed
    CATEGORY_OPTIONS, this must also surface HD/UNKNOWN participants)."""
    options = {GROUP_DISPLAY.get(g, g): {"group": g} for g in sorted(roster_df["group"].unique())}
    for subgroup in sorted(v for v in roster_df["subgroup"].unique() if v != "NA"):
        options[subgroup] = {"subgroup": subgroup}
    return options


def _display(roster_df: pd.DataFrame) -> pd.DataFrame:
    out = roster_df.copy()
    out["group"] = out["group"].map(lambda g: GROUP_DISPLAY.get(g, g))
    return out


roster_df = load_roster()

mode = st.segmented_control("View", ["Participants", "Groups"], default="Participants")

if mode == "Participants":
    sidebar_mode_header("Participants", PARTICIPANTS_COLOR)
    filter_options = _filter_options(roster_df)
    selected_labels = st.sidebar.multiselect("Filter by group / subgroup", list(filter_options))

    if selected_labels:
        mask = pd.Series(False, index=roster_df.index)
        for label in selected_labels:
            criterion = filter_options[label]
            key, value = next(iter(criterion.items()))
            mask |= roster_df[key] == value
        filtered = roster_df[mask]
    else:
        filtered = roster_df

    st.subheader(f"{len(filtered)} participant(s)")
    st.dataframe(_display(filtered), hide_index=True, width="stretch")

else:
    sidebar_mode_header("Groups", GROUPS_COLOR)
    summary = roster.build_group_summary(roster_df)
    st.subheader("Group summary")
    st.dataframe(_display(summary), hide_index=True, width="stretch")
