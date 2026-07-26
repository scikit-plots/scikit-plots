"""CYTHON-LOAD-002 probe: raw native-byte staging must refuse symlinked final
paths, keep the content dir private (0700), and publish atomically.

Exit 0 = all security properties hold.
"""
from __future__ import annotations

import os
import stat
import sys
import tempfile
from hashlib import sha256
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

# Dev/AI probe bootstrap: make ``scikitplot.cython`` importable whether this
# file is run from its shipped location (_templates/probe/) or copied elsewhere.
try:
    import scikitplot.cython  # noqa: F401
except ImportError:
    _here = Path(__file__).resolve()
    for _cand in _here.parents:
        if (_cand / "scikitplot" / "__init__.py").exists():
            sys.path.insert(0, str(_cand))
            break

from scikitplot.cython._loader import _stage_artifact_bytes_atomically  # noqa: E402

DATA = b"ELF_FAKE\x00\x01\x02" + b"\x00" * 64


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        h = sha256(DATA).hexdigest()

        # 1) Atomic write + verify, no temp leftovers.
        out_dir = root / "scikitplot_cython_import" / h[:16]
        out_dir.mkdir(parents=True, mode=0o700)
        out_path = out_dir / f"mod{EXTENSION_SUFFIXES[0]}"
        _stage_artifact_bytes_atomically(out_dir, out_path, DATA, expected_sha256=h)
        wrote_ok = out_path.read_bytes() == DATA and not list(out_dir.glob(".artifact-*"))
        print(f"atomic write + verify: {'OK' if wrote_ok else 'FAIL'}")
        ok = ok and wrote_ok

        # 2) Private dir perms (POSIX).
        if os.name == "posix":
            mode = stat.S_IMODE(os.lstat(out_dir).st_mode)
            priv = (mode & 0o077) == 0
            print(f"content dir private (0700): {'OK' if priv else 'FAIL'} ({oct(mode)})")
            ok = ok and priv

        # 3) Symlinked final path refused; victim untouched.
        if hasattr(os, "symlink"):
            d2 = root / "scikitplot_cython_import" / (h[:15] + "e")
            d2.mkdir(parents=True, mode=0o700)
            victim = root / "victim.bin"
            victim.write_bytes(b"secret")
            link = d2 / f"mod{EXTENSION_SUFFIXES[0]}"
            try:
                os.symlink(victim, link)
                refused = False
                try:
                    _stage_artifact_bytes_atomically(d2, link, DATA, expected_sha256=h)
                except OSError:
                    refused = True
                untouched = victim.read_bytes() == b"secret"
                print(f"symlink refused: {'OK' if refused else 'FAIL'}; "
                      f"victim untouched: {'OK' if untouched else 'FAIL'}")
                ok = ok and refused and untouched
            except (OSError, NotImplementedError):
                print("symlink not permitted here; skipping that check")

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
