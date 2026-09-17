# M2 Data Removal — Design

Status: implemented (`dataremoval/`, `dashboard/pages/5_Data_Removal.py`).
Relates to: PLAN/PLAN.md M2 (Data removing). Counterpart to M1
(docs/2026-09-17-data-intake-design.md).

## Problem

M1 added a way to get new subject data into `data/`. M2 is the inverse: a way
to remove it, at the granularity of one domain+session at a time (e.g. "just
MET039's ssvep session 2," not necessarily all of MET039's data). This is
higher risk than intake: `data/` is entirely gitignored (`data/**` in
`.gitignore`, only READMEs excepted), so anything removed there has no git
history to fall back on.

## Scope

Touches the same three "live" stores M1 writes to:
`data/ssveps/files/{metadata,runmap,baselines}.csv`,
`data/manualTest/behavioral_table.csv`,
`data/standardizedScores/repeatedSessionsPY.txt`.

Out of scope, matching M1's own precedent: `ssveps/files/subject_troughs.csv`
and `group_troughs.csv` are precomputed and not read by any dashboard page
(only by external notebooks via `ssveps/scripts/build_troughs.py`) — neither
M1 nor M2 touches them. `grid.json` is shared, not per-subject, and is never
deleted.

## Architecture

New module `dataremoval/`, mirroring `dataintake/`'s structure. Deliberately
does **not** import from `dataintake/` — this codebase's own convention
(see `beh/scripts/loader.py`'s comment on `PART_TYPE_GROUP`) is to duplicate
small per-domain format knowledge across projects rather than centralize it,
so each project stays independently understandable and modifiable.

- `dataremoval/scripts/_domain_loaders.py` — the same small loader-isolation
  helper as `dataintake/scripts/_domain_loaders.py` (avoids the same
  same-named-`loader.py`-across-projects collision problem), independently
  duplicated.
- `dataremoval/scripts/inventory.py`:
  - `list_subjects(data_dir) -> list[str]`: union of sub_ids across all three
    stores, computed directly here (not via `participants/scripts/roster.py`,
    same reasoning as above).
  - `rows_for(data_dir, sub_id) -> list[dict]`: every existing
    `(domain, session)` row for that subject across all three stores, e.g.
    `[{"domain": "ssvep", "session": 2}, {"domain": "beh", "session": 1}, ...]`.
- `dataremoval/scripts/delete.py`:
  - `delete_ssvep(data_dir, sub_id, session) -> Path`: writes a backup, then
    removes the matching `metadata.csv` row and its `runmap.csv`/
    `baselines.csv` rows. Returns the backup path.
  - `delete_beh(data_dir, sub_id, session) -> Path`: writes a backup, then
    removes the matching `behavioral_table.csv` rows.
  - `delete_fm100(data_dir, sub_id, session) -> Path`: writes a backup, then
    removes the matching line from `repeatedSessionsPY.txt`.
  - `restore_ssvep(data_dir, backup_path)`, `restore_beh(...)`,
    `restore_fm100(...)`: exact inverses — re-insert a backup file's contents
    directly into the real store. Not implemented via `dataintake`'s
    `commit_*` functions, since restoring must reproduce the exact original
    data, not re-run override/dedup logic that has nothing to do with
    restoring.
- `dashboard/pages/5_Data_Removal.py` — thin page: subject dropdown (from
  `list_subjects`) → checkbox per existing `(domain, session)` row (from
  `rows_for`) → a text input that must exactly match the selected subject ID
  → "Delete selected" button, disabled until the ID matches and at least one
  row is checked. No restore UI in M2 — `restore_*` exist and are tested,
  not wired to a button yet.

New directory `deletedData/` (repo-root sibling to `data/`/`updateData/`),
gitignored the same way (`deletedData/*` except a `README.md` placeholder,
mirroring `updateData/`'s `.gitignore` entry).

## Backup format

One file per deleted `(domain, session)`, named
`<timestamp>_<sub_id>_<domain>_session<N>.<ext>`:

- **fm100** (`.txt`): the exact raw 102-field line being removed, no header —
  the same shape `dataintake`'s own fm100 detection recognizes.
- **beh** (`.csv`): the exact rows being removed, header included — the same
  shape `dataintake`'s own beh detection recognizes.
- **ssvep** (`.json`): `{"metadata_row": {...}, "runmap_rows": [...],
  "baseline_rows": [...]}` — the exact tidy rows being removed.

Bonus, not required for restore: because the fm100/beh backups happen to be
in `dataintake`'s own accepted shape, a human could alternatively drop one
back into `updateData/` and use the normal intake page instead of calling
`restore_*` directly.

## Delete flow (per row)

1. Look up the exact row(s)/line to remove from the real store.
2. Write them to a new backup file in `deletedData/` (this must succeed
   before anything is removed from the real store).
3. Remove those rows/line from the real store.

**Multi-row atomicity**: when more than one `(domain, session)` checkbox is
selected at once, the page validates that every selected row still exists
(re-reading the store immediately before deleting, in case something else
changed it) before deleting any of them — the same two-pass
validate-then-apply pattern `dataintake` uses for commits, adapted for
deletes: a `dry_run` flag on each `delete_*` function checks existence and
writes nothing; the page runs a dry-run pass over every selected row, and
only if all pass does it run a second real pass.

## Error handling

- A row that no longer exists when the real pass runs (e.g. deleted by
  another action in between): the dry-run pass catches this and the whole
  batch is aborted with a clear message — nothing is removed.
- A backup write failure (e.g. disk full, permissions): the corresponding
  `delete_*` call raises before touching the real store, aborting that row
  (and, via the dry-run pass, ideally the whole batch before anything real
  happens).
- The subject-ID confirmation text input is a UI-layer guard only — the
  underlying `delete_*` functions don't require it, so tests can call them
  directly.

## Testing

`dataremoval/tests/` covers, for all three domains: `inventory.rows_for`
correctly enumerating existing sessions, `delete_*` writing a correct backup
and removing the correct (and only the correct) rows, `restore_*`
round-tripping a deleted row back to identical data, and the dry-run/real
two-pass pattern not leaving partial state when one row in a multi-row
selection no longer exists.

The Streamlit page itself is not unit tested (matching every other page in
this repo) — verified manually via `uv run streamlit run dashboard/Home.py`
before calling the feature done, using copies of real data/updateData (never
the live directories), same practice as M1's manual verification.

## Out of scope (M2)

- A "Restore" UI button — `restore_*` functions exist and are tested, but
  nothing in the dashboard calls them yet.
- Automatic cleanup/retention of `deletedData/` — backups accumulate; a
  human clears them manually when confident they're no longer needed.
- `subject_troughs.csv`/`group_troughs.csv` and any other file not written
  by M1's intake path.
