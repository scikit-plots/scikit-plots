"""
Synchronize the standalone HF proxy page-feedback runtime mirror.

The authoritative generic feedback backend lives in ``_sphinx_feedback``.
The HF Space proxy is deployed as a standalone Docker context, so it carries a
byte-identical mirror of the server-only contracts/service package.  This helper
is deterministic and idempotent; CI compares the mirror against the source.
"""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve()
PROXY = HERE.parents[1]
EXT_ROOT = PROXY.parents[1]
SOURCE = EXT_ROOT / "_sphinx_feedback"
DEST = PROXY / "_page_feedback"
TOP_LEVEL_FILES = ("__init__.py", "_contracts.py", "_network.py")
SERVICE_FILES = (
    "__init__.py",
    "_bitbucket.py",
    "_config.py",
    "_core.py",
    "_custom.py",
    "_forge.py",
    "_git.py",
    "_github.py",
    "_gitlab.py",
    "_sqlite.py",
    "_webhook.py",
    "app.py",
)


def sync() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "_service").mkdir(parents=True, exist_ok=True)
    for name in TOP_LEVEL_FILES:
        shutil.copyfile(SOURCE / name, DEST / name)
    for name in SERVICE_FILES:
        shutil.copyfile(SOURCE / "_service" / name, DEST / "_service" / name)


if __name__ == "__main__":
    sync()
