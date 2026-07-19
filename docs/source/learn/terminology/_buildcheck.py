#!/usr/bin/env python3
"""
Self-contained build + validate + idempotency (+ optional sync) for THIS folder.

Drop this file into any ``learn/`` subfolder next to its ``build_*.py`` generator
and ``*_content.py`` corpus, then run it from inside that folder:

    python3 _buildcheck.py [--sync]

Every course carries an identical copy — it discovers the generator and corpus in
its own directory, so no folder argument is needed and the same file works whether
the course is FLAT (lesson pages at the top level) or NESTED (lesson pages in
per-section subfolders). All paths are taken relative to this folder, and ``:doc:``
targets resolve relative to each file's own directory, so both layouts validate
with one code path. A folder with no ``build_*.py`` (a hand-maintained static
folder such as ``cheatsheet``/``glossary``/``resources``) runs in validate-only
mode.

What it does, each run:
  * runs the generator (if present);
  * validates every ``:doc:`` target (resolved per file directory), every
    ``:ref:`` target, and every section underline (skipping 3-space-indented
    code-block lines);
  * counts whichever of CONTENT / MINDMAP / GLOSS the corpus defines;
  * runs a full-vs-stub lesson census (per section for nested trees);
  * rebuilds once and compares a sha256 of every ``.rst`` for byte-identical
    idempotency;
  * with ``--sync``, mirrors this folder into
    ``/mnt/user-data/outputs/learn/<name>`` and rezips the whole ``learn/`` tree.

Prints two or three compact lines.
"""
import hashlib
import posixpath
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = HERE.name
OUT_LEARN = Path("/mnt/user-data/outputs/learn")
ZIP = Path("/mnt/user-data/outputs/scikit-plots-learn.zip")

# Sibling hub anchors a cross-course :ref: may target. Kept identical across every
# copy so inter-course links never false-positive, regardless of which folder the
# check runs in.
DEFINED_ANCHORS = {
    "terminology-index",
    "time-series-index",
    "deep-learning-index",
    "data-preparation-and-analysis-index",
    "bayesian-data-analysis-index",
    "data-analytics-index",
    "cheatsheet-index",
    "glossary-index",
    "external-learning-resources-index",
    "hands-on-index",
}

STUB_MARKER = "Lesson in progress"


def rsts():
    return sorted(HERE.rglob("*.rst"))


def relstem(p: Path) -> str:
    return p.relative_to(HERE).with_suffix("").as_posix()


def generator():
    return next(iter(HERE.glob("build_*.py")), None)


def build():
    gen = generator()
    if gen is None:
        return None
    r = subprocess.run([sys.executable, gen.name], cwd=HERE,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        raise SystemExit(f"generator failed: {gen.name}")
    out = r.stdout.strip().splitlines()
    return out[-1] if out else "(built)"


def _mask_code_blocks(text: str) -> str:
    """
    Blank the body lines of literal / code blocks so example roles inside them
    (e.g. an RST tutorial showing ``:ref:`genindex```) are not mistaken for real
    references. Mirrors docutils: a ``.. code-block::`` / ``code`` / ``sourcecode``
    / ``parsed-literal`` / ``literalinclude`` directive, or a line ending in ``::``
    that is not itself a directive, introduces a literal block whose body is every
    following blank-or-more-indented line. Only the :doc:/:ref: checks use this;
    anchor collection and the underline check keep the original text.
    """
    lit_dir = re.compile(
        r"^\s*\.\.\s+(?:code-block|code|sourcecode|parsed-literal|literalinclude)::")
    lines = text.split("\n")
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()
        intro = bool(lit_dir.match(line)) or (s.endswith("::") and not s.startswith(".."))
        out.append(line)                               # keep the introducer line
        i += 1
        if not intro:
            continue
        indent = len(line) - len(line.lstrip())
        while i < n and lines[i].strip() == "":        # blank separator lines
            out.append(lines[i])
            i += 1
        while i < n:                                    # the indented literal body
            body = lines[i]
            if body.strip() == "":
                out.append("")
                i += 1
                continue
            if len(body) - len(body.lstrip()) <= indent:   # dedent ends the block
                break
            out.append("")                              # blank a body line
            i += 1
    masked = "\n".join(out)
    # also blank inline-literal spans (double backticks) so example roles inside
    # them (e.g. ``:ref:`Label```) are not read as real references
    masked = re.sub(r"``.+?``", lambda m: "``" + " " * (len(m.group(0)) - 4) + "``",
                    masked)
    return masked


def validate():
    docs = {relstem(p) for p in rsts()}
    anchors = set(DEFINED_ANCHORS)
    for p in rsts():
        for m in re.finditer(r'^\s*\.\. _([^:\n]+):\s*$', p.read_text(encoding="utf-8"), re.M):
            anchors.add(m.group(1))

    bad_doc = bad_ref = bad_underline = 0
    for p in rsts():
        text = p.read_text(encoding="utf-8")
        checkable = _mask_code_blocks(text)            # example roles in code masked
        base = p.parent.relative_to(HERE).as_posix()  # "" at the top level
        for m in re.finditer(r':doc:`[^<`]*<([^>]+)>`', checkable):
            target = m.group(1).split("#")[0]
            if not target:
                continue
            if target.startswith("/"):
                resolved = posixpath.normpath(target.lstrip("/"))
            else:
                resolved = posixpath.normpath(posixpath.join(base, target))
            if resolved not in docs:
                bad_doc += 1
        for m in re.finditer(r':ref:`[^<`]*<([^>]+)>`|:ref:`([\w-]+)`', checkable):
            target = m.group(1) or m.group(2)
            if target and target not in anchors:
                bad_ref += 1
        lines = text.split("\n")
        for i in range(len(lines) - 1):
            head, under = lines[i], lines[i + 1]
            if head.startswith("   "):                 # skip code-block content
                continue
            if (under and len(set(under)) == 1 and under[0] in "=-~^\"'`#*+.:_"
                    and head and not head.startswith("..")):
                if len(under) < len(head):
                    bad_underline += 1
    return bad_doc, bad_ref, bad_underline


def corpus_counts():
    cm = next(iter(HERE.glob("*_content.py")), None)
    if cm is None:
        return {}
    sys.path.insert(0, str(HERE))
    try:
        mod = __import__(cm.stem)
    finally:
        del sys.path[0]
    return {a: len(getattr(mod, a)) for a in ("CONTENT", "MINDMAP", "GLOSS")
            if hasattr(mod, a)}


def census():
    groups = {}
    for p in rsts():
        if p.name == "index.rst" or not re.match(r"\d+-", p.name):
            continue
        rel = p.relative_to(HERE)
        group = rel.parts[0] if len(rel.parts) > 1 else "."
        full = STUB_MARKER not in p.read_text(encoding="utf-8")
        g = groups.setdefault(group, [0, 0])
        g[0] += 1
        if full:
            g[1] += 1
    total = sum(v[0] for v in groups.values())
    full = sum(v[1] for v in groups.values())
    return total, full, groups


def tree_sha() -> str:
    h = hashlib.sha256()
    for p in rsts():
        h.update(relstem(p).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def sync_and_zip():
    import shutil
    import zipfile
    dst = OUT_LEARN / NAME
    if dst.exists():
        for p in dst.rglob("*"):                       # defensive file-by-file
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(HERE, dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    tmp = ZIP.with_suffix(".tmp.zip")
    if tmp.exists():
        tmp.unlink()
    subprocess.run(["zip", "-rq", str(tmp), "learn", "-x", "*__pycache__*",
                    "-x", "*.pyc"], cwd=OUT_LEARN.parent, check=True)
    tmp.replace(ZIP)
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
    files = sum(1 for n in names if not n.endswith("/"))
    pyc = sum(1 for n in names if n.endswith(".pyc") or "__pycache__" in n)
    return files, pyc


def main():
    do_sync = "--sync" in sys.argv[1:]
    gen = generator()

    build_line = build()
    bd, br, bu = validate()
    counts = corpus_counts()
    total, full, groups = census()

    if gen is not None:
        sha1 = tree_sha()
        build()                                        # second build: idempotency
        sha2 = tree_sha()
        idem = f"idem {'OK' if sha1 == sha2 else 'DIFFERS'} {sha1[:12]}"
    else:
        idem = "static (no generator)"

    print(f"[{NAME}] {build_line or 'static folder — validate only'}")
    parts = [f"validate: doc {bd} | ref {br} | underline {bu}"]
    if counts:
        parts.append(" ".join(f"{k.lower()} {v}" for k, v in counts.items()))
    if total:
        parts.append(f"lessons {full}/{total} full")
    parts.append(idem)
    print("  ||  ".join(parts))
    if len(groups) > 1:
        print("  sections: " + "  ".join(
            f"{g}={v[1]}/{v[0]}" for g, v in sorted(groups.items())))
    if do_sync:
        files, pyc = sync_and_zip()
        print(f"synced+zipped: {files} files, pyc {pyc}")


if __name__ == "__main__":
    main()
