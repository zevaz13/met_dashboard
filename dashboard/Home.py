from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
GITHUB_REPO = "https://github.com/zevaz13/met_dashboard"
GITHUB_BLOB = f"{GITHUB_REPO}/blob/main"

st.set_page_config(page_title="CVD study dashboard", page_icon="🎨", layout="wide")

st.title("CVD study dashboard")
st.write("Interactive views over the color-vision-deficiency analysis in this repo.")
st.image(str(REPO_ROOT / "dashboard" / "color-wheel.jpg"), caption="The color space this whole study is about.")

st.subheader("Pick a line of analysis")
cols = st.columns(3)
pages = [
    ("pages/1_FM100.py", "FM100", "The clinical hue-ordering test: each participant arranges 85 caps in hue order."),
    ("pages/2_Behavioral.py", "Behavioral", "The manual red/green match task: where does isoluminance look like it sits?"),
    ("pages/3_SSVEP.py", "SSVEP", "The EEG grid response to the same red/green stimulus space."),
]
for col, (page, label, description) in zip(cols, pages):
    with col:
        st.page_link(page, label=label, use_container_width=True)
        st.caption(description)

st.divider()
st.write("References")
st.page_link(GITHUB_REPO, label="Repository", icon="🔗")
st.page_link(f"{GITHUB_BLOB}/README.md", label="Project README", icon="📄")
