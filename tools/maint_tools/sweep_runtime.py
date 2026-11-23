#!/usr/bin/env python3
"""
Import every public submodule of a package in a fresh process and report what it does.

Why this exists
---------------
Static checks cannot see what importing a module does: which optional
dependency it really needs, what it warns or logs, how long it takes,
which heavy libraries it pulls in, whether every name in ``__all__`` can be
read, and whether the public API names the same idea the same way. This
tool imports each public submodule (``scikitplot.corpus``, ...) in its own
child interpreter, so one import cannot hide another's cost, and reports:

* ``error``: the import failed (``ModuleNotFoundError: No module named
  'seaborn'`` on a base install is a packaging finding);
* ``seconds``, ``warnings`` (``warnings.warn``), ``logged`` (log records of
  level WARNING or above) and ``heavy`` (large libraries now imported);
* ``all`` (the length of ``__all__``, ``null`` when undefined),
  ``undocumented`` (public callables without a docstring) and
  ``lazy_failures`` (names whose access raises something other than
  ``AttributeError``, usually a missing optional dependency);
* naming clusters: one idea spelled several ways across the public API
  (``seed`` and ``random_state``, ``workers`` and ``n_jobs``).

Usage
-----
::

    python -B tools/maint_tools/sweep_runtime.py                     # scikitplot, this Python
    python -B tools/maint_tools/sweep_runtime.py --python VENV/bin/python --api --json out.json
    python -B tools/maint_tools/sweep_runtime.py --module corpus --module annoy
    python -B tools/maint_tools/sweep_runtime.py --pythonpath OVERLAY_DIR   # a tree before the installed one

``--package NAME`` sweeps another package; ``--heavy NAME`` (repeatable)
adds a library to the ``heavy`` list; ``--api`` also runs
``python -m <package> api <module>`` for each module (scikit-plots' listing
command) and records its exit status; ``--timeout SECONDS`` bounds each
child (default 300).

Notes
-----
**Times vary.** ``seconds`` depends on the machine and on what is already
in the operating system's file cache; compare runs on one machine. All
other fields are deterministic for one environment.

**Developer notes.** The tool never imports the swept package itself; only
the children do (lesson CI-12). Standard library only; Python 3.8 and newer
(Rule 137).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

#: Version of the JSON this tool writes.
FORMAT = 1

#: Marks the line of a child's standard output that holds its result.
MARK = "@@sweep_runtime@@"

#: Libraries whose import is worth knowing about (top-level module names);
#: ``--heavy NAME`` adds more.
HEAVY = (
    "torch", "tensorflow", "keras", "jax", "transformers", "scipy", "pandas",
    "matplotlib", "seaborn", "PIL", "lxml", "pdfminer", "mlflow", "spacy", "nltk",
    "gensim", "langchain_core", "pyspark", "cv2",
)

#: Idea -> the parameter names that spell it (a cluster is reported when
#: the public API uses two or more of them).
CLUSTERS: Dict[str, Sequence[str]] = {
    "random seed": ("seed", "random_state", "rng", "random_seed"),
    "parallel workers": ("n_jobs", "workers", "max_workers", "num_workers", "n_workers", "n_threads"),
    "verbosity": ("verbose", "verbosity", "quiet", "log_level", "silent"),
    "input file": ("path", "input_path", "file_path", "filepath", "filename", "fname", "src", "source_path"),
    "output location": ("out", "output", "output_path", "out_path", "dest", "output_dir", "out_dir", "save_path", "to_file"),
    "result count": ("top_k", "k", "n_results", "limit", "max_results", "top_n"),
    "time limit": ("timeout", "timeout_s", "timeout_seconds", "max_seconds", "deadline"),
    "size limit": ("max_bytes", "max_file_bytes", "max_size", "max_content_bytes", "max_response_bytes", "max_total_bytes"),
    "backend choice": ("backend", "engine", "impl", "implementation", "prefer_backend"),
    "colour map": ("cmap", "colormap", "palette", "color_map"),
    "class labels": ("labels", "class_names", "classes", "target_names"),
    "score cutoff": ("threshold", "cutoff", "score_cutoff", "min_score", "score_threshold"),
    "language": ("language", "lang", "locale", "default_language"),
    "overwrite": ("overwrite", "force", "exist_ok"),
}

#: Lists the public submodules of a package without importing it.
_DISCOVER = r"""
import importlib.util, json, pkgutil, sys
spec = importlib.util.find_spec(sys.argv[1])
if spec is None or not spec.submodule_search_locations:
    print("@@sweep_runtime@@" + json.dumps({"error": "not a package: " + sys.argv[1]}))
else:
    names = sorted(m.name for m in pkgutil.iter_modules(spec.submodule_search_locations)
                   if not m.name.startswith("_") and m.name not in ("tests", "conftest"))
    print("@@sweep_runtime@@" + json.dumps({"modules": names}))
"""

#: Imports one module and describes it; argv: module, heavy (JSON list).
_CHILD = r"""
import importlib, inspect, io, json, logging, sys, time, warnings
module_name, heavy = sys.argv[1], set(json.loads(sys.argv[2]))
before = set(sys.modules)
stream = io.StringIO()
handler = logging.StreamHandler(stream)
handler.setLevel(logging.WARNING)
handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
logging.getLogger().addHandler(handler)
result = {"error": None}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    start = time.perf_counter()
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        module = None
        result["error"] = (type(exc).__name__ + ": " + str(exc))[:300]
    result["seconds"] = round(time.perf_counter() - start, 3)
result["warnings"] = sorted({(w.category.__name__ + ": " + str(w.message))[:300] for w in caught})
result["logged"] = sorted({line[:300] for line in stream.getvalue().splitlines() if line.strip()})
result["heavy"] = sorted({n.split(".")[0] for n in set(sys.modules) - before} & heavy)
names = getattr(module, "__all__", None) if module is not None else None
result["all"] = None if names is None else len(names)
undocumented, lazy, params = [], [], {}
for name in names or ():
    try:
        obj = getattr(module, name)
    except AttributeError:
        continue
    except Exception as exc:
        lazy.append(name + ": " + type(exc).__name__)
        continue
    if not callable(obj):
        continue
    if not (getattr(obj, "__doc__", None) or "").strip():
        undocumented.append(name)
    try:
        target = obj.__init__ if inspect.isclass(obj) else obj
        params[name] = [p for p in inspect.signature(target).parameters if p not in ("self", "cls")]
    except (TypeError, ValueError):
        pass
result.update(undocumented=sorted(undocumented), lazy_failures=sorted(lazy), params=params)
print("@@sweep_runtime@@" + json.dumps(result, sort_keys=True))
"""


def _child(python: str, code: str, args: Sequence[str], env: Dict[str, str], timeout: float) -> Dict[str, Any]:
    """Run a child interpreter; return the JSON it marked, or an error record."""
    try:
        done = subprocess.run(  # noqa: S603 - the caller's interpreter, no shell
            [python, "-B", "-c", code, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(Path.home()),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"TimeoutError: no result within {timeout:g} s"}
    marked = [line[len(MARK) :] for line in done.stdout.splitlines() if line.startswith(MARK)]
    if not marked:
        tail = done.stderr.strip().splitlines()[-1:] or [f"exit status {done.returncode}"]
        return {"error": ("ChildProcessError: " + tail[0])[:300]}
    return json.loads(marked[-1])


def clusters(modules: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Return each idea the public API spells two or more ways.

    Parameters
    ----------
    modules : dict
        Module name to its record (``params``: public name -> parameters).

    Returns
    -------
    dict
        Idea -> spelling -> sorted ``module.name`` users, for the ideas of
        ``CLUSTERS`` with at least two spellings in use.
    """
    found: Dict[str, Dict[str, set]] = {}
    for module, record in sorted(modules.items()):
        for public, parameters in sorted((record.get("params") or {}).items()):
            for idea, spellings in CLUSTERS.items():
                for parameter in parameters:
                    if parameter in spellings:
                        found.setdefault(idea, {}).setdefault(parameter, set()).add(f"{module}.{public}")
    return {
        idea: {name: sorted(users) for name, users in sorted(by_name.items())}
        for idea, by_name in sorted(found.items())
        if len(by_name) > 1
    }


def sweep(
    python: str,
    package: str,
    *,
    modules: Sequence[str] = (),
    api: bool = False,
    timeout: float = 300.0,
    pythonpath: Sequence[str] = (),
    heavy: Sequence[str] = HEAVY,
) -> Dict[str, Any]:
    """
    Import each public submodule of *package* in a fresh child and describe it.

    Parameters
    ----------
    python : str
        Interpreter whose installation is swept.
    package : str
        Top-level package name.
    modules : sequence of str, optional
        Submodules to sweep (names below *package*); empty discovers every
        public one (no leading underscore, not ``tests``).
    api : bool, optional
        Also run ``python -m <package> api <package>.<module>``. Default:
        ``False``.
    timeout : float, optional
        Seconds per child. Default: 300.
    pythonpath : sequence of str, optional
        Folders put before the children's ``PYTHONPATH``.
    heavy : sequence of str, optional
        Top-level modules reported when an import loads them. Default:
        ``HEAVY``.

    Returns
    -------
    dict
        ``format``, ``package``, ``modules`` (name -> record, without the
        raw parameter lists) and ``clusters`` (:func:`clusters`).

    Raises
    ------
    ValueError
        If *package* cannot be found by the interpreter or *timeout* is not
        a number > 0.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"sweep_runtime: timeout must be a number > 0; got {timeout!r}.")
    env = dict(os.environ)
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([*map(str, pythonpath), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    names = list(modules)
    if not names:
        found = _child(python, _DISCOVER, [package], env, timeout)
        if "modules" not in found:
            raise ValueError(f"sweep_runtime: {found.get('error')}")
        names = found["modules"]
    records: Dict[str, Dict[str, Any]] = {}
    for name in sorted(set(names)):
        record = _child(python, _CHILD, [f"{package}.{name}", json.dumps(sorted(set(heavy)))], env, timeout)
        if api:
            try:
                done = subprocess.run(  # noqa: S603 - the caller's interpreter, no shell
                    [python, "-B", "-m", package, "api", f"{package}.{name}"],
                    capture_output=True, text=True, timeout=timeout, env=env, cwd=str(Path.home()), check=False,
                )
                record["api_status"] = done.returncode
                lines = (done.stderr or done.stdout).strip().splitlines()
                record["api_error"] = lines[0][:300] if done.returncode and lines else None
            except subprocess.TimeoutExpired:
                record["api_status"], record["api_error"] = None, f"no result within {timeout:g} s"
        records[name] = record
    grouped = clusters(records)
    for record in records.values():
        record.pop("params", None)
    return {"format": FORMAT, "package": package, "modules": records, "clusters": grouped}


def report(result: Dict[str, Any]) -> List[str]:
    """
    Return the human summary of a sweep: one line per module, then clusters.

    Parameters
    ----------
    result : dict
        From :func:`sweep`.

    Returns
    -------
    list of str
        Text lines.
    """
    lines = []
    for name, r in result["modules"].items():
        state = r.get("error") or "ok"
        api = "" if "api_status" not in r else f" api={r['api_status']}"
        lines.append(
            f"{name:20} {state[:70]:70} {r.get('seconds', 0) or 0:7.3f}s all={r.get('all')}"
            f" undocumented={len(r.get('undocumented') or [])} lazy={len(r.get('lazy_failures') or [])}"
            f" warned={len(r.get('warnings') or []) + len(r.get('logged') or [])}{api}"
            f" heavy={','.join(r.get('heavy') or []) or '-'}"
        )
    for idea, by_name in result["clusters"].items():
        spelled = "; ".join(f"{n} x{len(users)} ({', '.join(users[:2])})" for n, users in by_name.items())
        lines.append(f"cluster {idea}: {spelled}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--package", default="scikitplot")
    parser.add_argument("--module", action="append", default=[], metavar="NAME")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0, metavar="SECONDS")
    parser.add_argument("--pythonpath", action="append", default=[], metavar="DIR")
    parser.add_argument("--heavy", action="append", default=[], metavar="NAME")
    parser.add_argument("--json", type=Path, default=None, metavar="OUT")
    args = parser.parse_args(argv)
    try:
        result = sweep(
            args.python, args.package, modules=args.module, api=args.api,
            timeout=args.timeout, pythonpath=args.pythonpath, heavy=(*HEAVY, *args.heavy),
        )
    except ValueError as exc:
        parser.error(str(exc))
    for line in report(result):
        print(line)  # noqa: T201
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
