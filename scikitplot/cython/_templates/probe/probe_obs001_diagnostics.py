"""CYTHON-OBS-001 probe: bounded capture + typed BuildDiagnostic.

Checks:
1. output capture is a bounded buffer (tail-retaining);
2. a Cythonize failure attaches a typed BuildDiagnostic (phase/module/versions);
3. the exception message is preserved (backward compatible).

Exit 0 = observability behaves correctly.
"""
from __future__ import annotations

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

from scikitplot.cython._budget import BoundedBuffer, BuildDiagnostic  # noqa: E402
from scikitplot.cython._builder import build_extension_module_result  # noqa: E402

_KW = dict(
    source_path=None, use_cache=True, force_rebuild=False, verbose=-1,
    annotate=False, view_annotate=False, numpy_support=False, numpy_required=False,
    include_dirs=None, library_dirs=None, libraries=None, define_macros=None,
    extra_compile_args=None, extra_link_args=None, compiler_directives=None,
)


def main() -> int:
    ok = True

    b = BoundedBuffer(max_bytes=100)
    for i in range(1000):
        b.write(f"x{i}\n")
    bounded = b.truncated and len(b.getvalue()) < 200
    print(f"bounded capture: {'OK' if bounded else 'FAIL'}")
    ok = ok and bounded

    with tempfile.TemporaryDirectory() as td:
        try:
            build_extension_module_result(
                module_name="obs_probe", code="def f(:\n bad\n",
                cache_dir=Path(td), **_KW,
            )
            attached = False
            preserved = False
        except RuntimeError as e:
            d = getattr(e, "diagnostic", None)
            attached = isinstance(d, BuildDiagnostic) and d.phase == "cythonize" and (
                "cython" in d.tool_versions
            )
            preserved = "Cythonize failed" in str(e)
    print(f"typed diagnostic attached: {'OK' if attached else 'FAIL'}")
    print(f"message preserved: {'OK' if preserved else 'FAIL'}")
    ok = ok and attached and preserved

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
