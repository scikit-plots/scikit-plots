"""CYTHON-PIN-001 repro: a corrupt pins.json must raise PinRegistryError
(not silently return empty) and pin() must refuse to clobber it.
"""
import sys
import tempfile
from pathlib import Path

# Dev/AI probe bootstrap.
try:
    import scikitplot.cython  # noqa: F401
except ImportError:
    _here = Path(__file__).resolve()
    for _cand in _here.parents:
        if (_cand / "scikitplot" / "__init__.py").exists():
            sys.path.insert(0, str(_cand))
            break
from scikitplot.cython._pins import pin, list_pins, PinRegistryError
from scikitplot.cython._cache import make_cache_key

with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "cache"; root.mkdir()
    key = make_cache_key({"p": "1"})
    pin(key, alias="zz_test", cache_dir=root)
    # Corrupt pins.json
    (root / "pins.json").write_text("NOT JSON <<<", encoding="utf-8")
    try:
        list_pins(root)
        print("RESULT: silently returned (BUG still present)")
    except PinRegistryError as e:
        print(f"RESULT: PinRegistryError raised (FIXED): {str(e)[:55]}")
    # And pin() must NOT clobber a corrupt registry
    try:
        pin(make_cache_key({"p":"2"}), alias="another", cache_dir=root)
        print("RESULT: pin() overwrote corrupt registry (BUG)")
    except PinRegistryError:
        print("RESULT: pin() refuses to clobber corrupt registry (FIXED)")
