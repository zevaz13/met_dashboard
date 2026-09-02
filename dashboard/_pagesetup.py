"""Import isolation for the per-project scripts/ modules this dashboard
reuses.

beh/scripts, ssveps/scripts, and standardizedScores/FM100/scripts each have
their own loader.py/plotting.py/etc, independently, by convention (see
beh/README.md's Tests section) -- each is written to be imported with its
own directory on sys.path via a bare `import loader`. That's fine for a
notebook or a single pytest file, but this dashboard is one long-running
process with pages for all three projects, so a second project's
same-named module would otherwise be served from sys.modules' cache
instead of being read from disk.
"""

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent

# Shared across every page's category selector -- all three projects use the
# same group/subgroup labels (see e.g. beh/README.md's group field), so this
# is dashboard-layer presentation config, not a per-project analysis choice.
CATEGORY_OPTIONS = {
    "HC": {"group": "CTR"},
    "PD": {"group": "PD"},
    "CVD": {"group": "CVD"},
    "protan": {"subgroup": "protan"},
    "deutan": {"subgroup": "deutan"},
}


def sidebar_mode_header(text: str, color: str | None = None) -> None:
    """A sidebar section header for the active st.segmented_control mode
    (dashboard M3) -- plain st.sidebar.header when color is None (the
    "keep the current scheme" side of a page's mode split), or tinted
    `color` as that mode's UI accent (st.sidebar.header itself takes no
    color parameter, hence the raw markdown)."""
    if color is None:
        st.sidebar.header(text)
    else:
        st.sidebar.markdown(f"<h3 style='color:{color}'>{text}</h3>", unsafe_allow_html=True)


def use_scripts(rel_path: str, *names: str) -> dict:
    """Import `names` (e.g. "loader", "plotting") from
    <repo_root>/<rel_path>, dropping any already-cached module of the same
    name first and putting that directory first on sys.path. Returns
    {name: module}."""
    for name in names:
        sys.modules.pop(name, None)
    scripts_dir = str(REPO_ROOT / rel_path)
    if scripts_dir in sys.path:
        sys.path.remove(scripts_dir)
    sys.path.insert(0, scripts_dir)
    return {name: __import__(name) for name in names}
