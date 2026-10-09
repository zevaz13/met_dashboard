"""Regression tests for cross-project loader.py isolation.

Independently duplicated from dataintake/tests/test_domain_loaders.py --
see dataremoval/scripts/_domain_loaders.py's docstring for why.

Run: uv run pytest dataremoval/tests -q
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

sys.modules.pop("_domain_loaders", None)

from _domain_loaders import load_domain_loader  # noqa: E402


def test_loads_beh_loader_from_its_own_file():
    loader = load_domain_loader("beh")
    assert Path(loader.__file__).resolve() == Path(__file__).resolve().parents[2] / "beh/scripts/loader.py"
    assert loader.PART_TYPE_GROUP == {0: "UNKNOWN", 1: "CTR", 2: "CVD", 3: "PD", 4: "HD"}


def test_switching_domains_reloads_from_the_correct_file_each_time():
    beh_loader = load_domain_loader("beh")
    fm100_loader = load_domain_loader("fm100")
    ssvep_loader = load_domain_loader("ssvep")

    assert Path(beh_loader.__file__).resolve() == Path(__file__).resolve().parents[2] / "beh/scripts/loader.py"
    assert Path(fm100_loader.__file__).resolve() == (
        Path(__file__).resolve().parents[2] / "standardizedScores/FM100/scripts/loader.py"
    )
    assert Path(ssvep_loader.__file__).resolve() == Path(__file__).resolve().parents[2] / "ssveps/scripts/loader.py"
    assert hasattr(fm100_loader, "REFERENCE_COL")
    assert hasattr(ssvep_loader, "load_ssvep")
