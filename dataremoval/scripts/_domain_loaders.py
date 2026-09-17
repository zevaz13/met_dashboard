"""Load one domain's project-local loader.py fresh, isolated from the
others -- beh/, ssveps/, and standardizedScores/FM100/ each have their own
loader.py, and importing more than one bare `loader` in the same process
means the second import serves the first's cached module.

Independently duplicated from dataintake/scripts/_domain_loaders.py rather
than imported from it -- dataremoval and dataintake are separate, single
-purpose modules (add vs. remove), matching this codebase's own convention
of duplicating small per-domain format knowledge across projects (see
beh/scripts/loader.py's PART_TYPE_GROUP comment) rather than centralizing it.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_DOMAIN_DIRS = {
    "ssvep": "ssveps/scripts",
    "beh": "beh/scripts",
    "fm100": "standardizedScores/FM100/scripts",
}


def load_domain_loader(domain: str):
    """Import and return that domain's loader module, re-read from disk."""
    sys.modules.pop("loader", None)
    scripts_dir = str(REPO_ROOT / _DOMAIN_DIRS[domain])
    if scripts_dir in sys.path:
        sys.path.remove(scripts_dir)
    sys.path.insert(0, scripts_dir)
    import loader

    return loader
