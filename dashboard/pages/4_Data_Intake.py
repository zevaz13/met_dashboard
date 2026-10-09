import os
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "dashboard"))

from _pagesetup import use_scripts  # noqa: E402

# dataremoval/scripts has same-named errors.py and _domain_loaders.py
mods = use_scripts("dataintake/scripts", "errors", "_domain_loaders", "detect", "commit", "formats")
commit, detect, formats, DuplicateKeyError = mods["commit"], mods["detect"], mods["formats"], mods["errors"].DuplicateKeyError

UPDATE_DIR = Path(os.environ.get("MET_DASHBOARD_UPDATE_DIR", REPO_ROOT / "updateData"))
DATA_DIR = Path(os.environ.get("MET_DASHBOARD_DATA_DIR", REPO_ROOT / "data"))

GROUP_OPTIONS = ["CTR", "CVD", "PD", "HD", "UNKNOWN"]
SUBGROUP_OPTIONS = ["NA", "protan", "deutan"]

st.set_page_config(page_title="Data Intake", page_icon="📥", layout="wide")
st.title("Data Intake")
st.caption(f"Watching {UPDATE_DIR}")
with st.expander("What can I submit?"):
    st.markdown(formats.ACCEPTED_FORMATS_MD)


def scan_batches():
    """Group every file in UPDATE_DIR by its content-derived sub_id. Returns
    (batches, unrecognized) -- unrecognized files never enter a batch, so
    they never block a subject's commit."""
    batches: dict[str, list[dict]] = {}
    unrecognized = []
    if not UPDATE_DIR.exists():
        return batches, unrecognized
    for path in sorted(UPDATE_DIR.iterdir()):
        if not path.is_file():
            continue
        domain, sub_id, meta = detect.detect_file(path)
        if domain is None:
            unrecognized.append((path, meta.get("reason", "unrecognized")))
            continue
        batches.setdefault(sub_id, []).append({"domain": domain, "path": path, "meta": meta})
    return batches, unrecognized


def _commit_one(f: dict, override: dict, data_dir: Path, *, dry_run: bool) -> None:
    if f["domain"] == "beh":
        commit.commit_beh(
            f["path"], data_dir, sub_id=override["sub_id"], group=override["group"], session=override["session"], overwrite=override["overwrite"],
            dry_run=dry_run,
        )
    elif f["domain"] == "fm100":
        commit.commit_fm100(
            f["path"], data_dir, sub_id=override["sub_id"], session=override["session"], overwrite=override["overwrite"], dry_run=dry_run
        )
    elif f["domain"] == "ssvep":
        commit.commit_ssvep(
            f["path"],
            data_dir,
            sub_id=override["sub_id"],
            group=override["group"],
            subgroup=override["subgroup"],
            session=override["session"],
            overwrite=override["overwrite"],
            dry_run=dry_run,
        )


batches, unrecognized = scan_batches()

if not batches and not unrecognized:
    st.info("No files waiting in the intake directory.")

for sub_id, files in sorted(batches.items()):
    st.subheader(f"Subject: {sub_id}")
    row_overrides = []

    for i, f in enumerate(files):
        meta = f["meta"]
        cols = st.columns([1, 2, 2, 2, 2, 2, 2])
        cols[0].write(f["domain"])
        cols[1].write(f["path"].name)
        if meta.get("filename_mismatch"):
            cols[1].caption(f"filename suggested {meta['filename_mismatch']}")

        sub_id_override = cols[2].text_input("sub_id", value=sub_id, key=f"subid_{sub_id}_{i}")

        group_override = None
        if f["domain"] != "fm100":
            default_group = meta.get("group", "UNKNOWN")
            default_index = GROUP_OPTIONS.index(default_group) if default_group in GROUP_OPTIONS else GROUP_OPTIONS.index("UNKNOWN")
            group_override = cols[3].selectbox("group", GROUP_OPTIONS, index=default_index, key=f"group_{sub_id}_{i}")
        else:
            cols[3].caption("fm100 has no group field of its own")

        subgroup_override = None
        if f["domain"] == "ssvep":
            default_subgroup = meta.get("subgroup", "NA")
            default_index = SUBGROUP_OPTIONS.index(default_subgroup) if default_subgroup in SUBGROUP_OPTIONS else 0
            subgroup_override = cols[4].selectbox("subgroup", SUBGROUP_OPTIONS, index=default_index, key=f"subgroup_{sub_id}_{i}")

        # Session: ssvep/beh already carry a real detected value to confirm or
        # correct; fm100's raw line carries none, so its default is the actual
        # next-free slot for whatever sub_id is currently entered above.
        session_override = None
        if f["domain"] == "ssvep":
            session_override = int(cols[5].number_input("session", min_value=1, step=1, value=int(meta["session"]), key=f"session_{sub_id}_{i}"))
        elif f["domain"] == "beh":
            sessions = meta.get("sessions", [])
            if len(sessions) == 1:
                session_override = int(cols[5].number_input("session", min_value=1, step=1, value=int(sessions[0]), key=f"session_{sub_id}_{i}"))
            else:
                cols[5].caption(f"sessions in file: {sessions}")
        elif f["domain"] == "fm100":
            suggested = commit.suggest_fm100_session(DATA_DIR, sub_id_override)
            session_override = int(cols[5].number_input("session", min_value=1, max_value=3, step=1, value=suggested, key=f"session_{sub_id}_{i}"))

        overwrite = cols[6].checkbox("overwrite if exists", key=f"overwrite_{sub_id}_{i}")
        row_overrides.append(
            {
                "sub_id": sub_id_override,
                "group": group_override,
                "subgroup": subgroup_override,
                "session": session_override,
                "overwrite": overwrite,
            }
        )

    if st.button(f"Commit {sub_id}", key=f"commit_{sub_id}"):
        try:
            # Validate every file in the batch before writing any of them --
            # otherwise an earlier file's write already lands for real by the
            # time a later file's conflict is discovered, and retrying would
            # re-commit that earlier file on top of its own prior write.
            for f, override in zip(files, row_overrides):
                _commit_one(f, override, DATA_DIR, dry_run=True)
            for f, override in zip(files, row_overrides):
                _commit_one(f, override, DATA_DIR, dry_run=False)
        except DuplicateKeyError as exc:
            st.error(f"{exc} -- tick 'overwrite if exists' above to replace it. Nothing for {sub_id} was written.")
        except Exception as exc:
            st.error(f"Commit failed, nothing for {sub_id} was written: {exc}")
        else:
            for f in files:
                f["path"].unlink()
            st.success(f"Committed and cleared {len(files)} file(s) for {sub_id}.")
            st.cache_data.clear()
            st.rerun()

if unrecognized:
    st.subheader("Unrecognized files")
    st.caption("Left in place -- not guessed at.")
    for path, reason in unrecognized:
        st.write(f"**{path.name}**: {reason}")
