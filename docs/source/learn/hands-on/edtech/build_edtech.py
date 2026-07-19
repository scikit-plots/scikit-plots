#!/usr/bin/env python3
"""
Deterministic, idempotent generator for the BilimEdtech Labs CC BY 4.0 mirror.

Mode B (verbatim mirror) generator for ``learn/hands-on/edtech/``. It is a pure
function of its inputs — ``edtech_manifest.py`` plus the raw upstream RST cached
under ``_sources_cache/`` — and writes byte-identical output on every run.

What it does
------------
1. Clears every generated ``*.rst`` under this folder (the generator owns them).
2. Emits ``LICENSE_edtech.rst`` — the full CC BY 4.0 legal text (verbatim, in a
   literal block) with a mirror-scope header.
3. For each path in ``VERBATIM``: reads ``_sources_cache/<path>.rst.txt``,
   optionally strips the trailing ``Search`` section (root only), reproduces it
   **byte-for-byte**, and appends a per-page *Source & license* admonition whose
   ``:doc:`` link to ``LICENSE_edtech`` is depth-correct for that page.
4. Collects every ``toctree`` target across the emitted pages and auto-writes a
   clearly-labelled **stub** for any target not yet in ``VERBATIM`` — so the tree
   is always buildable and the full upstream structure is visible while the
   mirror is filled in batches.

Determinism: inputs sorted, no RNG, no timestamps. Verify with ``_buildcheck.py``.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import edtech_manifest as M  # noqa: E402

CACHE = HERE / "_sources_cache"


# --- helpers -------------------------------------------------------------
def source_url(doc: str) -> str:
    """Upstream .html URL for a mirrored doc path (deterministic)."""
    return f"{M.SITE_BASE}/{doc}.html"


def lic_rel(doc: str) -> str:
    """Relative :doc: target from ``doc`` to the root ``LICENSE_edtech``."""
    return "../" * doc.count("/") + "LICENSE_edtech"


def read_cache(doc: str) -> str:
    return (CACHE / f"{doc}.rst.txt").read_text(encoding="utf-8")


def write(doc: str, text: str) -> None:
    f = HERE / f"{doc}.rst"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")


def strip_search(text: str) -> str:
    """Remove a trailing ``Search`` section (heading + ``:ref:`search```)."""
    lines = text.split("\n")
    for i in range(len(lines) - 1):
        if lines[i].strip() == "Search" and set(lines[i + 1].strip()) == {"="}:
            return "\n".join(lines[:i]).rstrip() + "\n"
    return text


def rewrite_prolog(text: str, doc: str) -> str:
    """
    Repoint the upstream site-wide prolog include at the mirror-local copy.

    Upstream pages carry ``.. include:: /includes/prolog.inc`` — a leading-slash,
    Sphinx-source-root-relative path. That file is not published under _sources,
    and in scikit-plots ``/includes/`` is a different tree, so we rewrite it to a
    path (relative to this ``doc``'s depth) that resolves to the mirror-local
    ``includes/prolog.inc`` emitted from cache. Documented CC BY change notice.
    """
    rel = "../" * doc.count("/")
    return text.replace(".. include:: /includes/prolog.inc",
                        f".. include:: {rel}includes/prolog.inc")


def attribution(doc: str) -> str:
    """Per-page CC BY attribution admonition (appended after verbatim body)."""
    return (
        ".. admonition:: Source & license\n"
        "   :class: note\n"
        "\n"
        "   Reproduced **verbatim, without modification** from\n"
        f"   `{M.COPYRIGHT} <{M.ROOT_SOURCE}>`__,\n"
        "   licensed under\n"
        "   `Creative Commons Attribution 4.0 International (CC BY 4.0) "
        f"<{M.LICENSE_DEED}>`__.\n"
        "\n"
        "   Source page:\n"
        f"   {source_url(doc)}\n"
        "\n"
        f"   See :doc:`LICENSE <{lic_rel(doc)}>` for the full license text.\n"
    )


def _mask_code_blocks(text: str) -> str:
    """
    Blank literal / code-block bodies.

    So example directives inside them (e.g. an
    RST tutorial showing a ``.. toctree::``) are not parsed as real structure.
    Same rule the validator uses. Real page-level toctrees are never indented under
    a code block, so they are unaffected.
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
        out.append(line)
        i += 1
        if not intro:
            continue
        indent = len(line) - len(line.lstrip())
        while i < n and lines[i].strip() == "":
            out.append(lines[i])
            i += 1
        while i < n:
            body = lines[i]
            if body.strip() == "":
                out.append("")
                i += 1
                continue
            if len(body) - len(body.lstrip()) <= indent:
                break
            out.append("")
            i += 1
    masked = "\n".join(out)
    masked = re.sub(r"``.+?``", lambda m: "``" + " " * (len(m.group(0)) - 4) + "``",
                    masked)
    return masked


def toctree_targets(doc: str, text: str):
    """
    Yield toctree entry doc paths in ``text``, resolved relative to ``doc``.

    Example toctrees inside code blocks are ignored (see ``_mask_code_blocks``).
    """
    text = _mask_code_blocks(text)
    base = doc.rsplit("/", 1)[0] if "/" in doc else ""
    lines = text.split("\n")
    i, n = 0, len(lines)
    while i < n:
        m = re.match(r"^(\s*)\.\.\s+toctree::\s*$", lines[i])
        if not m:
            i += 1
            continue
        indent = len(m.group(1))
        i += 1
        while i < n:
            ln = lines[i]
            if ln.strip() == "":
                i += 1
                continue
            cur = len(ln) - len(ln.lstrip())
            if cur <= indent:                 # dedent -> block ended
                break
            s = ln.strip()
            if s.startswith(":") or s.startswith(".."):   # toctree option or comment
                i += 1
                continue
            entry = s
            if entry.endswith(">") and "<" in entry:   # "Title <path>" form
                entry = entry[entry.index("<") + 1:-1].strip()
            if entry.startswith("/"):
                resolved = entry.lstrip("/")
            elif base:
                resolved = f"{base}/{entry}"
            else:
                resolved = entry
            yield resolved
            i += 1


def stub_title(doc: str) -> str:
    if doc in M.STUB_TITLES:
        return M.STUB_TITLES[doc]
    seg = doc.split("/")[-2] if doc.endswith("/index") else doc.split("/")[-1]
    return seg.replace("-", " ").replace("_", " ").title()


def stub_rst(doc: str) -> str:
    title = stub_title(doc)
    bar = "=" * max(len(title), 3)
    return (
        f"{title}\n{bar}\n"
        "\n"
        ".. note::\n"
        "\n"
        "   **Verbatim mirror pending.** This page is a placeholder in the\n"
        "   scikit-plots mirror of the BilimEdtech Labs site. The upstream page\n"
        f"   `{title} <{source_url(doc)}>`__ will be reproduced here verbatim,\n"
        "   without modification, under CC BY 4.0. Until then, refer to the\n"
        "   upstream page directly.\n"
        "\n"
        f"   See :doc:`LICENSE <{lic_rel(doc)}>` for license details and\n"
        "   ``EDTECH.md`` for mirror status and how pages are added.\n"
    )


def license_rst() -> str:
    cc = read_cache_license()
    body = "\n".join(("    " + ln) if ln.strip() else "" for ln in cc.split("\n"))
    bar = "=" * 70
    title = "License: BilimEdtech Labs mirror (CC BY 4.0)"
    return (
        ":orphan:\n"
        "\n"
        ".. _edtech-license:\n"
        "\n"
        f"{bar}\n{title}\n{bar}\n"
        "\n"
        "All educational content in this folder is reproduced **without\n"
        f"modification** from {M.SITE_BASE} ({M.COPYRIGHT}) under the Creative\n"
        "Commons Attribution 4.0 International (CC BY 4.0) license:\n"
        f"{M.LICENSE_DEED}\n"
        "\n"
        f"The full legal text ({M.LICENSE_LEGAL}) follows verbatim.\n"
        "\n"
        "::\n"
        "\n"
        f"{body}\n"
    )


def read_cache_license() -> str:
    return (CACHE / "LICENSE.txt").read_text(encoding="utf-8")


def write_placeholder(path: str, upstream_url: str) -> None:
    """
    Provide the image at ``path``.

    If ``fetch_edtech.py`` downloaded the real
    upstream original into ``_sources_cache/_images/``, copy that; otherwise write
    a deterministic, labelled placeholder so the real Sphinx build stays free of
    missing-image warnings. The mirrored RST already references ``path``.
    """
    import shutil
    target = HERE / path
    target.parent.mkdir(parents=True, exist_ok=True)
    real = CACHE / "_images" / path.rsplit("/", 1)[-1]
    if real.exists():                       # real original fetched by the crawler
        shutil.copyfile(real, target)
        return
    from PIL import Image, ImageDraw
    w, h = 640, 360
    img = Image.new("RGB", (w, h), (238, 238, 238))
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, w - 3, h - 3], outline=(170, 170, 170), width=2)
    for i, ln in enumerate((
        "screenshot placeholder",
        path.rsplit("/", 1)[-1],
        "",
        "upstream original (not fetched):",
        upstream_url,
        "",
        "reproduced under CC BY 4.0 - see LICENSE_edtech",
    )):
        d.text((28, 120 + i * 22), ln, fill=(80, 80, 80))
    f = HERE / path
    f.parent.mkdir(parents=True, exist_ok=True)
    ext = path.rsplit(".", 1)[-1].lower()
    img.save(f, "JPEG" if ext in ("jpg", "jpeg") else "PNG")


# --- build ---------------------------------------------------------------
def main() -> None:
    for p in HERE.rglob("*.rst"):             # generator owns all .rst here
        p.unlink()

    write("LICENSE_edtech", license_rst())

    emitted = set()
    targets = set()
    for doc in sorted(M.VERBATIM):
        text = read_cache(doc)
        if doc in M.DROP_SEARCH:
            text = strip_search(text)
        text = rewrite_prolog(text, doc)
        write(doc, text.rstrip() + "\n\n" + attribution(doc))
        emitted.add(doc)
        targets.update(toctree_targets(doc, text))

    stubs = sorted(targets - emitted)
    for doc in stubs:
        write(doc, stub_rst(doc))
        emitted.add(doc)

    # Include fragments: files pulled into pages via ``.. include:: <name>`` that
    # have no HTML page of their own. Emit verbatim from cache, WITHOUT the
    # attribution footer (they are inlined into pages that already carry it).
    # Paths carry the ``.rst`` extension; cache is ``<path>.txt``. These are not
    # toctree docs, so scikit-plots' conf.py must exclude them from the build
    # (see EDTECH.md) to avoid "not in any toctree" warnings.
    includes = sorted(getattr(M, "INCLUDES", ()))
    for rel in includes:
        dst = HERE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text((CACHE / f"{rel}.txt").read_text(encoding="utf-8").rstrip()
                       + "\n", encoding="utf-8")

    for path, url in sorted(getattr(M, "PLACEHOLDER_IMAGES", {}).items()):
        write_placeholder(path, url)

    for d in sorted(HERE.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if d.is_dir() and not any(d.iterdir()):    # prune dirs emptied by removed stubs
            d.rmdir()

    total_rst = len(emitted) + 1 + len(includes)   # + LICENSE_edtech + fragments
    imgs = len(getattr(M, "PLACEHOLDER_IMAGES", {}))
    print(f"edtech mirror: {len(M.VERBATIM)} verbatim, {len(stubs)} stub, "
          f"{len(includes)} include-frag, "
          f"{total_rst} rst (incl LICENSE), {imgs} placeholder png")


if __name__ == "__main__":
    main()
