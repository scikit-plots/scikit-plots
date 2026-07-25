"""CYTHON-CACHE-004 repro: an interrupted export must preserve the prior
export (transactional stage + atomic swap), never leave a partial dest.
"""
import sys
import tempfile
from pathlib import Path
from unittest import mock

# Dev/AI probe bootstrap.
try:
    import scikitplot.cython  # noqa: F401
except ImportError:
    _here = Path(__file__).resolve()
    for _cand in _here.parents:
        if (_cand / "scikitplot" / "__init__.py").exists():
            sys.path.insert(0, str(_cand))
            break
from scikitplot.cython._public import export_cached
import shutil

with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "cache"; root.mkdir()
    key = "abc123"
    entry = root / key; entry.mkdir()
    (entry / "artifact.so").write_bytes(b"NEW-GOOD-ARTIFACT")
    (entry / "meta.json").write_text('{"kind":"module"}')

    dest = Path(td) / "dest"
    # Pre-existing GOOD export at destination
    (dest / key).mkdir(parents=True)
    (dest / key / "artifact.so").write_bytes(b"OLD-GOOD-ARTIFACT")
    (dest / key / "meta.json").write_text('{"kind":"module"}')

    # Simulate copytree failing PART-WAY (after rmtree already deleted the old export)
    real_copytree = shutil.copytree
    def failing_copytree(s, d, *a, **k):
        # create the dir + one file then fail (partial copy)
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "artifact.so").write_bytes(b"PARTIAL")
        raise OSError("disk full mid-copy")
    with mock.patch("scikitplot.cython._public.shutil.copytree", failing_copytree):
        try:
            export_cached(key, dest_dir=dest, cache_dir=root)
        except OSError as e:
            pass
    # What's left at dest/key?
    d = dest / key
    if d.exists():
        files = sorted(p.name for p in d.iterdir())
        art = (d / "artifact.so").read_bytes() if (d/"artifact.so").exists() else b"<none>"
        meta_ok = (d / "meta.json").exists()
        print(f"dest after failed export: files={files}, artifact={art!r}, meta_present={meta_ok}")
        if art == b"PARTIAL" or not meta_ok:
            print("RESULT: BUG — destination left CORRUPT (old export destroyed, new incomplete)")
        else:
            print("RESULT: destination intact")
    else:
        print("RESULT: dest/key missing entirely (old export destroyed)")
