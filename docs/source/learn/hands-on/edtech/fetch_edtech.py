#!/usr/bin/env python3
"""
fetch_edtech.py — populate the edtech source cache directly from the upstream
BilimEdtech Labs site (https://labs.bilimedtech.com), so ``build_edtech.py`` can
produce the complete verbatim mirror in one deterministic pass.

WHY THIS EXISTS
    The in-notebook fetcher can only retrieve URLs that were previously surfaced,
    which makes mirroring ~200 pages one-at-a-time slow. Run from a machine with
    ordinary network access to the site, this crawler pulls every page source,
    every ``.. include::`` fragment, and every referenced image, writing them into
    ``_sources_cache/`` in the exact layout the generator already expects.

WHAT IT DOES  (all read-only against the site; only writes under _sources_cache/)
    1. Starts at the root document and walks every ``.. toctree::`` to discover
       all reachable pages (orphan pages outside the nav are intentionally skipped
       — this matches the mirror's scope).
    2. For each page, fetches  _sources/<path>.rst.txt  ->  _sources_cache/<path>.rst.txt
    3. Detects ``.. include:: <name>`` directives, resolves them relative to the
       including page, and fetches those fragments too (they have no HTML page but
       ARE published under _sources). Fragments keep their ``.rst`` extension:
       _sources/<frag>.txt  ->  _sources_cache/<frag>.txt
    4. Detects ``.. image::`` / ``.. figure::`` / ``.. |x| image::`` targets and
       downloads the binary from  _images/<basename>  ->  _sources_cache/_images/<basename>
    5. Writes  _sources_cache/_discovered.json  = {pages, fragments, images}
       which edtech_manifest.py merges into VERBATIM / INCLUDES / PLACEHOLDER_IMAGES.

DESIGN NOTES
    * Zero third-party dependencies (uses urllib) so it runs anywhere Python 3 does.
    * Idempotent: already-cached files are skipped unless --force is given.
    * Fails safe: a page/fragment/image that will not fetch is recorded in the
      failure report and skipped; the crawl continues. Missing pages simply remain
      stubs in the build, so you can hand-mirror just those.
    * Polite: a short delay between requests and a descriptive User-Agent.

USAGE
    python3 fetch_edtech.py                 # crawl + fetch everything new
    python3 fetch_edtech.py --force         # refetch even cached files
    python3 fetch_edtech.py --dry-run       # discover + report, fetch nothing
    python3 fetch_edtech.py --delay 0.5     # seconds between requests (default 0.3)

Then, from this directory:
    python3 _buildcheck.py                  # regenerates + validates the full mirror
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# --- Configuration --------------------------------------------------------
SITE_BASE = "https://labs.bilimedtech.com"
ROOT_DOC = "index"                       # site root document (a Sphinx docname)
HERE = Path(__file__).resolve().parent
CACHE = HERE / "_sources_cache"
IMAGE_CACHE = CACHE / "_images"
DISCOVERED = CACHE / "_discovered.json"
USER_AGENT = ("scikit-plots-edtech-mirror/1.0 (CC BY 4.0 verbatim mirror; "
              "contact: scikit-plots maintainers)")
RETRIES = 3
TIMEOUT = 30

# Implicit Sphinx targets and self-references that are never real documents.
SKIP_TARGETS = {"genindex", "modindex", "search", "self"}
UNDERLINE = set("=-~^\"'`#*+.:_")


# --- HTTP helpers ---------------------------------------------------------
def _fetch(url: str, binary: bool, delay: float) -> bytes | str | None:
    """Fetch a URL with retries. Returns bytes/str, or None on hard failure."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
            time.sleep(delay)
            return data if binary else data.decode("utf-8")
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code == 404:            # not found — no point retrying
                break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = str(e)
        time.sleep(delay * attempt)      # simple backoff
    print(f"    ! fetch failed ({last}): {url}", file=sys.stderr)
    return None


# --- RST parsing (self-contained; mirrors build_edtech.py's masker) -------
_LIT_DIR = re.compile(
    r"^\s*\.\.\s+(?:code-block|code|sourcecode|parsed-literal|literalinclude)::")


def mask_code_blocks(text: str) -> str:
    """
    Blank literal/code-block bodies and inline literals so example directives
    inside them (e.g. a tutorial showing a ``.. toctree::`` or ``.. include::``)
    are not parsed as real structure. Identical rule to build_edtech.py: a
    ``::``-ending line introduces a literal block only when it is NOT itself a
    directive (does not start with ``..``); explicit code directives are matched
    by name, so a real ``.. toctree::`` is never masked.
    """
    lines = text.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()
        intro = bool(_LIT_DIR.match(line)) or (s.endswith("::") and not s.startswith(".."))
        out.append(line)
        i += 1
        if not intro:
            continue
        indent = len(line) - len(line.lstrip())
        while i < n and lines[i].strip() == "":
            out.append(lines[i])
            i += 1
        while i < n:                      # blank the more-indented block body
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
    masked = re.sub(r"``.+?``", lambda mm: "``" + " " * (len(mm.group(0)) - 4) + "``",
                    masked)
    return masked


def _resolve(base_dir: str, target: str) -> str:
    """Resolve a doc/include/image target relative to a page's directory."""
    return os.path.normpath(os.path.join(base_dir, target)).replace(os.sep, "/")


def toctree_targets(text: str, base_dir: str) -> list[str]:
    """Return resolved docnames listed in this page's toctree directives."""
    masked = mask_code_blocks(text).split("\n")
    targets: list[str] = []
    i, n = 0, len(masked)
    while i < n:
        if re.match(r"^\s*\.\.\s+toctree::", masked[i]):
            indent = len(masked[i]) - len(masked[i].lstrip())
            i += 1
            # skip option lines and the blank line(s) before entries
            while i < n and (masked[i].strip() == "" or masked[i].lstrip().startswith(":")):
                i += 1
            while i < n:
                ln = masked[i]
                if ln.strip() == "":
                    i += 1
                    continue
                cur = len(ln) - len(ln.lstrip())
                if cur <= indent:
                    break
                s = ln.strip()
                if not s.startswith(":") and not s.startswith(".."):
                    m = re.match(r"^.*<([^>]+)>\s*$", s)   # 'Title <path>' form
                    entry = m.group(1) if m else s
                    if entry and not entry.startswith(("http://", "https://")):
                        entry = entry.rstrip("/")
                        if entry not in SKIP_TARGETS:
                            targets.append(_resolve(base_dir, entry))
                i += 1
        else:
            i += 1
    return targets


def find_includes(text: str, base_dir: str) -> list[str]:
    """
    Return resolved fragment paths (with .rst) for real ``.. include::``.

    Leading-slash targets (e.g. ``/includes/prolog.inc``) are Sphinx-source-root
    relative site infrastructure, not per-page fragments — they are skipped here
    and repointed to a mirror-local copy by the generator (see build_edtech.py).
    """
    frags: list[str] = []
    for line in mask_code_blocks(text).split("\n"):
        m = re.match(r"^\s*\.\.\s+include::\s+(\S+)\s*$", line)
        if m and not m.group(1).startswith("/"):
            frags.append(_resolve(base_dir, m.group(1)))
    return frags


def find_images(text: str, base_dir: str) -> list[tuple[str, str]]:
    """Return (mirror_path, image_basename) for real image/figure directives."""
    imgs: list[tuple[str, str]] = []
    pats = (r"^\s*\.\.\s+(?:image|figure)::\s+(\S+)\s*$",
            r"^\s*\.\.\s+\|[^|]+\|\s+image::\s+(\S+)\s*$")
    for line in mask_code_blocks(text).split("\n"):
        for pat in pats:
            m = re.match(pat, line)
            if m:
                target = m.group(1)
                mirror = _resolve(base_dir, target)
                imgs.append((mirror, os.path.basename(mirror)))
    return imgs


# --- Fetch bookkeeping ----------------------------------------------------
def src_url(source_path: str) -> str:
    """_sources URL for a source file path (already ending in .rst)."""
    return f"{SITE_BASE}/_sources/{source_path}.txt"


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the edtech mirror sources.")
    ap.add_argument("--force", action="store_true", help="refetch cached files")
    ap.add_argument("--dry-run", action="store_true", help="discover only; no writes")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between requests")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    pages: list[str] = []                 # docnames (no extension)
    fragments: list[str] = []             # fragment paths (with .rst)
    images: dict[str, str] = {}           # mirror_path -> source url
    seen_pages: set[str] = set()
    seen_frag: set[str] = set()
    failures: list[str] = []

    queue = [ROOT_DOC]
    print(f"[fetch] crawling from {SITE_BASE} (root: {ROOT_DOC!r})")

    while queue:
        doc = queue.pop(0)
        if doc in seen_pages:
            continue
        seen_pages.add(doc)
        base_dir = os.path.dirname(doc)
        cache_file = CACHE / f"{doc}.rst.txt"

        if cache_file.exists() and not args.force:
            text = cache_file.read_text(encoding="utf-8")
            print(f"  = page (cached)  {doc}")
        else:
            text = _fetch(src_url(f"{doc}.rst"), binary=False, delay=args.delay)
            if text is None:
                failures.append(f"page   {doc}")
                continue
            if not args.dry_run:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(text, encoding="utf-8")
            print(f"  + page  {doc}")
        pages.append(doc)

        # discover children / fragments / images from this page
        for child in toctree_targets(text, base_dir):
            if child not in seen_pages:
                queue.append(child)
        for frag in find_includes(text, base_dir):
            if frag not in seen_frag:
                seen_frag.add(frag)
                fragments.append(frag)
        for mirror, _ in find_images(text, base_dir):
            images.setdefault(mirror, f"{SITE_BASE}/_images/{os.path.basename(mirror)}")

    # fetch fragments (and scan them for nested includes/images, just in case)
    fi = 0
    while fi < len(fragments):
        frag = fragments[fi]
        fi += 1
        base_dir = os.path.dirname(frag)
        cache_file = CACHE / f"{frag}.txt"
        if cache_file.exists() and not args.force:
            ftext = cache_file.read_text(encoding="utf-8")
            print(f"  = frag (cached)  {frag}")
        else:
            ftext = _fetch(src_url(frag), binary=False, delay=args.delay)
            if ftext is None:
                failures.append(f"frag   {frag}")
                continue
            if not args.dry_run:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(ftext, encoding="utf-8")
            print(f"  + frag  {frag}")
        for nested in find_includes(ftext, base_dir):
            if nested not in seen_frag:
                seen_frag.add(nested)
                fragments.append(nested)
        for mirror, _ in find_images(ftext, base_dir):
            images.setdefault(mirror, f"{SITE_BASE}/_images/{os.path.basename(mirror)}")

    # fetch images (binary)
    for mirror, url in sorted(images.items()):
        dest = IMAGE_CACHE / os.path.basename(mirror)
        if dest.exists() and not args.force:
            print(f"  = img (cached)  {os.path.basename(mirror)}")
            continue
        if args.dry_run:
            continue
        data = _fetch(url, binary=True, delay=args.delay)
        if data is None:
            failures.append(f"image  {mirror}  ({url})")
            continue
        dest.write_bytes(data)
        print(f"  + img  {os.path.basename(mirror)}")

    # write the discovery manifest for edtech_manifest.py to merge
    result = {
        "pages": sorted(set(pages) - set(fragments)),
        "fragments": sorted(set(fragments)),
        "images": {k: v for k, v in sorted(images.items())},
    }
    if not args.dry_run:
        DISCOVERED.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")

    print("\n[fetch] summary")
    print(f"    pages:     {len(result['pages'])}")
    print(f"    fragments: {len(result['fragments'])}")
    print(f"    images:    {len(result['images'])}")
    if failures:
        print(f"    FAILURES ({len(failures)}) — hand-mirror these:")
        for f in failures:
            print(f"      - {f}")
    else:
        print("    no failures")
    if not args.dry_run:
        print(f"\n[fetch] wrote {DISCOVERED.relative_to(HERE)}")
        print("[fetch] next: run  python3 _buildcheck.py  to build + validate the mirror")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
