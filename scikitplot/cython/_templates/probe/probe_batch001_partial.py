"""CYTHON-BATCH-001 probe: batch build returns a structured partial report.

Checks (mocks cython_import_result — no compiler needed):
1. fail-fast raises BatchBuildError carrying committed items + resume token;
2. collect attempts every item and records failures;
3. resume via `only` builds the remaining items.

Exit 0 = batch policy + partial reporting behave correctly.
"""
from __future__ import annotations

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

import scikitplot.cython._public as P  # noqa: E402
from scikitplot.cython import BatchBuildError  # noqa: E402


def _fake(fail):
    def _inner(f, **kw):
        if f.stem in fail:
            raise ValueError(f"boom-{f.stem}")
        return f"built-{f.stem}"
    return _inner


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for s in ("a", "b", "c"):
            (d / f"{s}.pyx").write_text("x")

        with mock.patch.object(P, "cython_import_result", _fake({"b"})):
            try:
                P.cython_import_all_result(d, collect=False)
                ff = False
            except BatchBuildError as e:
                ff = list(e.result.committed) == ["a"] and e.resume_token == ("c",)
            print(f"fail-fast partial + resume token: {'OK' if ff else 'FAIL'}")
            ok = ok and ff

            r = P.cython_import_all_result(d, collect=True)
            col = list(r.successes) == ["a", "c"] and [f.name for f in r.failures] == ["b"]
            print(f"collect records failures: {'OK' if col else 'FAIL'}")
            ok = ok and col

        with mock.patch.object(P, "cython_import_result", _fake(set())):
            r2 = P.cython_import_all_result(d, collect=True, only=["c"])
            res = list(r2.successes) == ["c"]
            print(f"resume via only: {'OK' if res else 'FAIL'}")
            ok = ok and res

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
