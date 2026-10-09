import os
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "dashboard"))

from _pagesetup import use_scripts  # noqa: E402

# dataintake/scripts has same-named errors.py and _domain_loaders.py
mods = use_scripts("dataremoval/scripts", "errors", "_domain_loaders", "delete", "inventory")
delete, inventory, RowNotFoundError = mods["delete"], mods["inventory"], mods["errors"].RowNotFoundError

DATA_DIR = Path(os.environ.get("MET_DASHBOARD_DATA_DIR", REPO_ROOT / "data"))
DELETED_DIR = Path(os.environ.get("MET_DASHBOARD_DELETED_DIR", REPO_ROOT / "deletedData"))

DELETE_FN = {"ssvep": delete.delete_ssvep, "beh": delete.delete_beh, "fm100": delete.delete_fm100}

st.set_page_config(page_title="Data Removal", page_icon="🗑️", layout="wide")
st.title("Data Removal")
st.caption(f"Removes from {DATA_DIR}, backing up to {DELETED_DIR} first. This has no git history to fall back on.")


def _delete_one(row: dict, *, dry_run: bool) -> Path | None:
    return DELETE_FN[row["domain"]](DATA_DIR, DELETED_DIR, row["sub_id"], row["session"], dry_run=dry_run)


subjects = inventory.list_subjects(DATA_DIR)
if not subjects:
    st.info("No subjects found in the data store.")
    st.stop()

sub_id = st.selectbox("Subject", subjects)
rows = inventory.rows_for(DATA_DIR, sub_id)

if not rows:
    st.info(f"No data found for {sub_id}.")
    st.stop()

st.subheader(f"Existing data for {sub_id}")
selected = []
for i, row in enumerate(sorted(rows, key=lambda r: (r["domain"], r["session"]))):
    cols = st.columns([1, 1, 3])
    checked = cols[0].checkbox("", key=f"row_{sub_id}_{i}")
    cols[1].write(row["domain"])
    cols[2].write(f"session {row['session']}")
    if checked:
        selected.append({**row, "sub_id": sub_id})

confirm = st.text_input(f"Type the subject ID ({sub_id}) to confirm deletion")
can_delete = bool(selected) and confirm == sub_id

if st.button("Delete selected", disabled=not can_delete):
    try:
        # Validate every selected row still exists before removing any of
        # them -- otherwise one row's conflict could leave an earlier row in
        # the same batch already deleted for real, with no way to undo it as
        # a single unit.
        for row in selected:
            _delete_one(row, dry_run=True)
        backup_paths = [_delete_one(row, dry_run=False) for row in selected]
    except RowNotFoundError as exc:
        st.error(f"{exc} -- nothing for {sub_id} was deleted.")
    except Exception as exc:
        st.error(f"Delete failed, nothing for {sub_id} was deleted: {exc}")
    else:
        st.success(f"Deleted {len(selected)} row(s) for {sub_id}. Backups: " + ", ".join(str(p) for p in backup_paths))
        st.cache_data.clear()
        st.rerun()
elif not confirm:
    st.caption("Select at least one row and type the subject ID above to enable deletion.")
elif confirm != sub_id:
    st.caption(f"Typed ID doesn't match {sub_id}.")
