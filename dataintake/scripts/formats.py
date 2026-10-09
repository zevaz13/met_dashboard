"""What the intake accepts, as one markdown table. Rendered on the Data
Intake page; updateData/README.md carries the same table -- update both."""

ACCEPTED_FORMATS_MD = """\
| Data | File | Recognized by | Required | Defaulted if missing (editable on the page) |
|---|---|---|---|---|
| Behavioral | `.csv` | header has `Red` and `Green` (any case) | `Red`, `Green` | `SubID` = filename, `RunNumber` = row order, `session` = 1, group = UNKNOWN, `Date` = file date, `FolderOrg` = filename |
| Behavioral | `.json` (task export) | `metadata.mode` is `behavioral`, columns include `TrialNumber`, `Red`, `Green` | those three | sub_id = filename, session = 1, group = UNKNOWN |
| SSVEP | `.mat` | contains `SubID`, `session`, `group`, `subgroup`, `runMap`, `baselines`, `redArray`, `greenArray` | all of those | none |
| FM100 | `.csv`/`.txt` raw export | no header, exactly 102 comma-separated fields | the raw line | session = next free slot (1-3) |

Behavioral files hold one subject each (all `SubID` values the same). Files are
recognized by content, not name; anything that doesn't match is listed as
unrecognized and left in place.
"""
