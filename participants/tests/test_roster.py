"""Regression tests for the cross-modality participant roster.

Run: uv run pytest participants/tests -q
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

sys.modules.pop("roster", None)

import roster  # noqa: E402


def _import_module(path: Path, name: str):
    """Load a same-named sibling module (loader.py exists in three
    different projects) under a unique name. FM100/beh/ssveps's own test
    files dodge the sys.modules collision by popping the cache and relying
    on import order (they only ever need their own project's loader); this
    file needs all three loaded at once, so that trick alone isn't enough."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fm100_df():
    fm100_loader = _import_module(REPO_ROOT / "standardizedScores/FM100/scripts/loader.py", "fm100_loader_for_roster_tests")
    return fm100_loader.load_fm100_raw()


@pytest.fixture(scope="module")
def beh_df():
    beh_loader = _import_module(REPO_ROOT / "beh/scripts/loader.py", "beh_loader_for_roster_tests")
    return beh_loader.load_behavioral()


@pytest.fixture(scope="module")
def ssvep_meta_df():
    ssvep_analysis = _import_module(REPO_ROOT / "ssveps/scripts/analysis.py", "ssvep_analysis_for_roster_tests")
    return ssvep_analysis.load_metadata()


@pytest.fixture(scope="module")
def roster_df(fm100_df, beh_df, ssvep_meta_df):
    return roster.build_roster(fm100_df, beh_df, ssvep_meta_df)


# --- build_roster ------------------------------------------------------


def test_roster_has_expected_columns(roster_df):
    assert list(roster_df.columns) == ["sub_id", "group", "subgroup", "fm100_sessions", "beh_sessions", "ssvep_sessions"]


def test_roster_has_one_row_per_subject_across_all_three_sources(roster_df, fm100_df, beh_df, ssvep_meta_df):
    expected = set(fm100_df["sub_id"]) | set(beh_df["sub_id"]) | set(ssvep_meta_df["sub_id"])
    assert set(roster_df["sub_id"]) == expected
    assert roster_df["sub_id"].is_unique


def test_group_resolves_from_ssvep_metadata_when_present(roster_df):
    """MET021 is a real case documented in FM100's own tests: labeled CTR
    in ssveps/files/metadata.csv, trusted as-is."""
    row = roster_df[roster_df["sub_id"] == "MET021"].iloc[0]
    assert row["group"] == "CTR"


def test_group_falls_back_to_behavioral_data_when_absent_from_ssvep_metadata(roster_df):
    """MET013/MET014/MET041/MET042 have no SSVEP session at all, but do
    have behavioral data -- same fallback FM100's own loader applies."""
    expected = {"MET013": "CTR", "MET014": "CTR", "MET041": "PD", "MET042": "PD"}
    for sub_id, expected_group in expected.items():
        row = roster_df[roster_df["sub_id"] == sub_id].iloc[0]
        assert row["group"] == expected_group
        assert row["subgroup"] == "NA"


def test_group_is_unknown_when_absent_from_both_ssvep_and_behavioral():
    """MET047 has an SSVEP file (ssveps' own metadata.csv already records
    it as UNKNOWN) but no behavioral data to resolve it further."""
    fm100_df = pd.DataFrame({"sub_id": ["MET047"], "session": [1]})
    beh_df = pd.DataFrame(columns=["sub_id", "session", "group", "subgroup"])
    ssvep_meta_df = pd.DataFrame({"sub_id": ["MET047"], "session": [1], "group": ["UNKNOWN"], "subgroup": ["NA"]})
    row = roster.build_roster(fm100_df, beh_df, ssvep_meta_df).iloc[0]
    assert row["group"] == "UNKNOWN"
    assert row["subgroup"] == "NA"
    assert row["beh_sessions"] == 0


def test_session_counts_match_each_source_independently(roster_df):
    """MET000 has a different session count in each modality (3 FM100
    visits, 3 behavioral, 2 SSVEP) -- pins that each count is read from
    its own source, not conflated with another."""
    row = roster_df[roster_df["sub_id"] == "MET000"].iloc[0]
    assert row["fm100_sessions"] == 3
    assert row["beh_sessions"] == 3
    assert row["ssvep_sessions"] == 2


def test_build_roster_includes_a_subject_with_no_fm100_data():
    """Real data has no such subject today (FM100 happens to be a
    superset of the other two), so this pins the union logic itself with
    synthetic frames -- a subject tested only on SSVEP must still appear,
    with zero everywhere else."""
    fm100_df = pd.DataFrame({"sub_id": ["MET100"], "session": [1]})
    beh_df = pd.DataFrame(columns=["sub_id", "session", "group", "subgroup"])
    ssvep_meta_df = pd.DataFrame({"sub_id": ["MET100", "MET101"], "session": [1, 1], "group": ["CTR", "PD"], "subgroup": ["NA", "NA"]})
    result = roster.build_roster(fm100_df, beh_df, ssvep_meta_df)
    assert set(result["sub_id"]) == {"MET100", "MET101"}
    row = result[result["sub_id"] == "MET101"].iloc[0]
    assert row["fm100_sessions"] == 0
    assert row["group"] == "PD"


# --- build_group_summary -------------------------------------------------


def test_group_summary_has_one_row_per_group_subgroup_pair(roster_df):
    summary = roster.build_group_summary(roster_df)
    assert set(zip(summary["group"], summary["subgroup"])) == set(
        roster_df[["group", "subgroup"]].drop_duplicates().itertuples(index=False, name=None)
    )


def test_group_summary_n_participants_matches_roster_group_size(roster_df):
    summary = roster.build_group_summary(roster_df)
    ctr_row = summary[(summary["group"] == "CTR") & (summary["subgroup"] == "NA")].iloc[0]
    expected = len(roster_df[(roster_df["group"] == "CTR") & (roster_df["subgroup"] == "NA")])
    assert ctr_row["n_participants"] == expected


def test_group_summary_total_sessions_sums_across_participants():
    roster_df = pd.DataFrame(
        {
            "sub_id": ["A", "B"],
            "group": ["CTR", "CTR"],
            "subgroup": ["NA", "NA"],
            "fm100_sessions": [2, 1],
            "beh_sessions": [0, 3],
            "ssvep_sessions": [1, 1],
        }
    )
    row = roster.build_group_summary(roster_df).iloc[0]
    assert row["n_participants"] == 2
    assert row["fm100_total_sessions"] == 3
    assert row["beh_total_sessions"] == 3
    assert row["ssvep_total_sessions"] == 2


def test_group_summary_coverage_counts_subjects_with_at_least_one_session():
    roster_df = pd.DataFrame(
        {
            "sub_id": ["A", "B", "C"],
            "group": ["PD", "PD", "PD"],
            "subgroup": ["NA", "NA", "NA"],
            "fm100_sessions": [1, 0, 2],
            "beh_sessions": [0, 0, 1],
            "ssvep_sessions": [1, 1, 0],
        }
    )
    row = roster.build_group_summary(roster_df).iloc[0]
    assert row["fm100_n_with_data"] == 2  # A and C
    assert row["beh_n_with_data"] == 1  # C only
    assert row["ssvep_n_with_data"] == 2  # A and B
