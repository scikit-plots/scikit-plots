"""CYTHON-PKG-001 repro: malformed package names (traversal, separators,
keywords, leading digits) must be rejected before any build path is made.
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
from scikitplot.cython._builder import build_extension_package_from_code_result

# Try a malformed package_name with traversal — should be REJECTED before building
for bad in ["../evil", "foo/bar", "1foo", "foo..bar", "foo.class", "foo bar"]:
    with tempfile.TemporaryDirectory() as td:
        try:
            build_extension_package_from_code_result(
                {"m": "def f(): return 1"},
                package_name=bad,
                cache_dir=Path(td),
            )
            print(f"  package_name={bad!r:14} -> ACCEPTED (BUG: no validation)")
        except ValueError as e:
            print(f"  package_name={bad!r:14} -> ValueError (validated)")
        except Exception as e:
            # Reached builder internals with a bad name => not validated up front
            print(f"  package_name={bad!r:14} -> {type(e).__name__}: {str(e)[:45]} (reached build)")
