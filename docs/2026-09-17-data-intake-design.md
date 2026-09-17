# M1 Data Intake — Design

Status: approved, not yet implemented.
Relates to: PLAN/PLAN.md M1 (Data Loading).

## Problem

New subject data (ssvep, behavioral, FM100 — any subset) is dropped into
`updateData/`. The dashboard needs to detect what each file is, figure out
which subject and group it belongs to, let the researcher confirm or correct
that, write it into the real data stores under `data/`, and clear
`updateData/` once done. Datasets can be partial: not every subject has all
three domains.

The three domains do not share one raw-file shape:

- **ssvep**: one `.mat` file per (subject, session). `ssveps/scripts/loader.py`
  already knows how to read it (`load_ssvep`, `to_rows`) and write the derived
  CSVs (`write_derived_csv`). `ssveps/scripts/update_derived.py` already does
  incremental intake for this domain, but it's interactive (`input()`-driven)
  and hardcoded to `RAW_DIR = "/home/sebas/data/ssveps"`, an absolute path
  outside this repo — a leftover from vendoring. We reuse its underlying
  `loader.py` functions, not the script itself.
- **behavioral**: one flat master file, `data/manualTest/behavioral_table.csv`
  — one row per click. New data is *appended rows*, not a new file. Group is
  derived from the file's own `PartType` column via `PART_TYPE_GROUP` in
  `beh/scripts/loader.py`.
- **fm100**: one flat master file, `data/standardizedScores/repeatedSessionsPY.txt`
  — one row per (subject, session), positional (no header), 102
  comma-separated fields. New data is an *appended line*. Repeat sessions for
  the same subject are encoded by suffixing the reference field (`MET000` →
  session 1, `MET000b` → session 2, `MET000c` → session 3), per
  `_parse_session_and_id` in `standardizedScores/FM100/scripts/loader.py`.

Group/subgroup is only durably stored in `ssveps/files/metadata.csv`; beh and
fm100 loaders fall back to it (or to beh's own `PartType`) at load time. There
is no separate "group override" store in the current data model, and this
design doesn't add one — a group correction is written back into whichever
field is authoritative for that domain (see Commit below).

## Sample fixtures already in `updateData/`

- `MET999_beh.csv` — matches the master beh CSV's header exactly.
- `MET999_FM100.txt` — one raw line, 102 comma-separated fields, matching the
  master fm100 file's row shape.
- `MET999.mat` — a real ssvep capture. Its embedded `SubID` is `MET000`, not
  `MET999` — the filename and content disagree. This is intentional and is
  kept as-is: it's the fixture that exercises filename/content mismatch
  handling, not a data error to fix. Content is always authoritative for
  sub_id; the filename is only ever a display hint.

## Architecture

New module `dataintake/`, mirroring the existing `participants/` split
(thin page, tested logic underneath):

- `dataintake/scripts/detect.py` — scans the intake directory, identifies
  each file's domain by content (see Detection below), and groups files into
  per-subject batches keyed by the sub_id found inside each file.
- `dataintake/scripts/commit.py` — `commit_ssvep`, `commit_beh`,
  `commit_fm100`: one committer per domain, each writing into that domain's
  real store using its existing format knowledge (imported from
  `ssveps/scripts/loader.py`, `beh/scripts/loader.py`,
  `standardizedScores/FM100/scripts/loader.py` — schemas and constants are
  not redefined here).
- `dataintake/tests/` — pytest, using the `MET999_*` files copied into
  `dataintake/tests/fixtures/` (tests don't mutate the live `updateData/`
  working files).
- `dashboard/pages/4_Data_Intake.py` — thin Streamlit page calling into the
  above; no parsing/business logic lives in the page itself.

New env var `MET_DASHBOARD_UPDATE_DIR`, mirroring `MET_DASHBOARD_DATA_DIR`,
defaulting to `<repo_root>/updateData`.

`ssveps/scripts/update_derived.py` and `build_derived.py` are untouched —
they remain for the original researcher's own external raw-data workflow,
orthogonal to this dashboard-native intake path.

## Detection

Content-first; extension only narrows candidates, never decides alone:

- `.mat` → must load via `scipy.io.loadmat` and contain the full expected key
  set (`SubID`, `session`, `group`, `subgroup`, `runMap`, `baselines`,
  `redArray`, `greenArray`). Missing any of these → rejected as malformed,
  not guessed at.
- `.csv`/`.txt` whose first line matches the header
  `SubID,Red,Green,RunNumber,session,PartType,Date,FolderOrg` → beh.
- `.csv`/`.txt` with no such header and exactly 102 comma-separated fields →
  fm100.
- Anything else → "unrecognized": shown in the UI, left in place, does not
  block other files in its batch.

Sub_id used for batching/display is always the content-embedded value. If a
file's name implies a different sub_id than its content, the UI shows both
and flags the mismatch rather than silently picking one.

## Per-subject commit flow

Files are grouped into a batch by content-derived sub_id. Per file, the UI
shows: domain, detected sub_id (editable — manual override supported),
detected group/subgroup (always shown for confirmation, editable), and
validity. A batch's "Commit" action is enabled only once every file in it is
valid — invalid files must be fixed or removed first.

On commit, per domain:

- **ssvep**: build metadata/runmap/baseline rows via
  `ssveps/scripts/loader.to_rows`, using the (possibly overridden) sub_id and
  group. If `(sub_id, session)` already exists in `metadata.csv`, require an
  explicit overwrite confirmation (default off) before replacing its
  runmap/baseline rows — same semantics as `update_derived.py`'s y/n prompt,
  as a checkbox instead of stdin. `metadata.csv` is written through the
  shared `write_derived_csv` writer so formatting stays byte-identical to the
  existing rebuild/update scripts.
- **beh**: reverse-map the confirmed group to `PartType`, append rows to
  `behavioral_table.csv`. Existing `(sub_id, session)` triggers the same
  overwrite confirmation.
- **fm100**: count existing sessions for this sub_id in
  `repeatedSessionsPY.txt` to pick the reference-field suffix (none/`b`/`c`),
  then append the 102-field row. Targeting an already-existing session number
  triggers the same overwrite confirmation.

**Atomicity**: if every file in a batch commits successfully, all are deleted
from `updateData/` together. If any file's commit throws despite passing
validation, nothing in the batch is deleted — the error is surfaced and the
batch stays in place for retry. A subject with only some domains present
(e.g. beh only) is deleted as soon as that smaller batch is fully committed —
it does not wait for the missing domains to ever show up.

## Error handling

- Unrecognized file: left in `updateData/`, shown as unrecognized, never
  blocks sibling files.
- Malformed file (fails a domain's structural check): flagged invalid with a
  reason, left in place.
- Duplicate `(sub_id, session)` in the target store: requires an explicit,
  default-off overwrite confirmation — never silently overwritten.
- Unexpected commit failure: whole batch left untouched (see Atomicity).

## Testing

`dataintake/tests/` covers: domain detection for each sample type, the
filename/content mismatch path (`MET999.mat`), rejection of malformed files,
fm100 session-suffix assignment against a small synthetic fixture, beh's
`PartType` round-trip, and overwrite-confirmation behavior for existing keys
in all three domains.

The Streamlit page itself is not unit tested (no other page in this repo is)
— verified manually via `uv run streamlit run dashboard/Home.py` before
calling the feature done.

## Out of scope (M1)

- M2 (deleting data from `data/`) — separate milestone, not addressed here.
- Any change to `ssveps/scripts/update_derived.py`/`build_derived.py`.
- A persisted "group override" store — corrections are written back into the
  domain's own authoritative field, not a new file.
