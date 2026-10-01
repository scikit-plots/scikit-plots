"""Live entity-engine probe: offset alignment, round trip, interchangeability.

This is the lane that was recorded as ``UNAVAILABLE`` for two rounds because no
entity engine was installed. Both are now present, so the claims that were held
open can finally be measured rather than asserted.

Three properties, each over fuzzed text rather than a fixed sentence:

``ALIGN``
    ``span.text == text[span.start:span.end]`` for every span from every
    engine. This is the whole reason :mod:`scikitplot.cleanprompt._nltk`
    carries offsets through tokenization instead of recovering them by search,
    and the fuzzer feeds it precisely the input where search drifts — tabs,
    newlines, directional quotes, accents, astral-plane emoji, repeated names.

``RT``
    Redact then restore returns the original byte for byte, under every engine
    mode, including ``both`` where two engines disagree and the span resolver
    has to merge them.

``SWAP``
    A vault written under one engine restores under the other. That is what the
    canonical vocabulary buys, and without it a placeholder would mean
    different things on two machines.
"""

import random
import sys
import time

import pathlib as _pathlib
# The repository root, from this file's own location: evidence/ -> _maintenance/
# -> cleanprompt/ -> maintenances/ -> root. Never a hard-coded checkout path,
# which silently probed a stale tree once the checkout moved (CP-053).
_ROOT = str(_pathlib.Path(__file__).resolve().parents[4])
sys.path.insert(0, _ROOT)

from scikitplot.cleanprompt import (  # noqa: E402
    DEFAULT_POLICY,
    Redactor,
    build_detectors,
    default_registry,
    describe_engines,
    restore,
)
from scikitplot.cleanprompt._engines import CANONICAL_LABELS  # noqa: E402

fails = []


def check(cid, desc, ok, detail=""):
    print("%-9s %-4s %s%s" % (cid, "PASS" if ok else "FAIL", desc, ("  -> " + detail) if detail and not ok else ""))
    if not ok:
        fails.append(cid)


LIVE = [n for n in ("spacy", "nltk") if describe_engines()["engines"][n]["usable"]]

print("=" * 78)
print("LIVE ENTITY ENGINES")
print("=" * 78)
for name, engine in describe_engines()["engines"].items():
    print("  %-6s %-14s %s" % (name, engine["status"], engine["detail"]))
print("  usable:", ", ".join(LIVE) or "none")
print()

if not LIVE:
    print("no engine usable; nothing to measure")
    sys.exit(0)

# Fragments chosen for the ways an offset can drift: whitespace a sentence
# tokenizer normalises, quotes Treebank rewrites, multi-byte letters, astral
# emoji, and names that repeat so that a find-the-text implementation would
# collapse every occurrence onto the first.
NAMES = ["Ada Lovelace", "Charles Babbage", "Zoë François", "Mustafa Kemal Atatürk", "Renée Dupont"]
PLACES = ["London", "Montréal", "Turkey", "New York City", "İstanbul"]
ORGS = ["the Analytical Society", "Acme Corporation", "the Republic of Turkey"]
NOISE = [
    "\t", "\n", "\n\n", "  ", " — ", " … ",
    '"quoted"', "“curly”", "‘single’", "(parenthetical)", "[1]",
    "🚀", "🎉", "naïve café", "ada@example.com", "4242 4242 4242 4242",
    "2024-01-15", "v1.26.4", "<<angle>>", "[PERSON-1]",
]

rng = random.Random(20260920)
# 120 documents per mode: spaCy dominates the cost at roughly 4 ms a
# document, and the invariants under test are structural rather than
# statistical — a misaligned span shows up in the first dozen, not the
# five hundredth. The 3000-document scale run lives in probe_negative.py,
# where no model has to load.
N = 120

print("=" * 78)
print("OFFSET ALIGNMENT AND ROUND TRIP OVER %d FUZZED DOCUMENTS" % N)
print("=" * 78)

modes = LIVE + (["both"] if len(LIVE) > 1 else []) + ["none"]
totals = {}
t0 = time.monotonic()

for mode in modes:
    detectors = build_detectors(mode=mode)
    registry = default_registry()
    for detector in detectors:
        registry.add(detector)
    bad_align = bad_rt = bad_label = 0
    spans_seen = 0
    for _ in range(N):
        parts = []
        for _ in range(rng.randint(1, 12)):
            bucket = rng.random()
            if bucket < 0.25:
                parts.append(rng.choice(NAMES))
            elif bucket < 0.40:
                parts.append(rng.choice(PLACES))
            elif bucket < 0.50:
                parts.append(rng.choice(ORGS))
            else:
                parts.append(rng.choice(NOISE))
        text = " ".join(parts)

        for detector in detectors:
            for span in detector.detect(text, DEFAULT_POLICY):
                spans_seen += 1
                if span.text != text[span.start : span.end]:
                    bad_align += 1
                if not (0 <= span.start < span.end <= len(text)):
                    bad_align += 1
                if span.kind not in CANONICAL_LABELS:
                    bad_label += 1

        result = Redactor(registry=registry).redact(text)
        if restore(result.text, result.vault).text != text:
            bad_rt += 1

    totals[mode] = spans_seen
    check("ALIGN", "%-5s span.text == text[start:end]  (%d spans)" % (mode, spans_seen), bad_align == 0, "%d misaligned" % bad_align)
    check("RT", "%-5s redact/restore is exact" % mode, bad_rt == 0, "%d failures" % bad_rt)
    check("LABEL", "%-5s every label is canonical" % mode, bad_label == 0, "%d uncanonical" % bad_label)

print("           (%d documents x %d modes in %.1fs)" % (N, len(modes), time.monotonic() - t0))

if len(LIVE) > 1:
    print()
    print("=" * 78)
    print("ENGINE INTERCHANGEABILITY")
    print("=" * 78)
    sample = "Ada Lovelace wrote to ada@example.com from London about Acme Corporation."
    vaults = {}
    for name in LIVE:
        registry = default_registry()
        registry.add(build_detectors(mode=name)[0])
        vaults[name] = Redactor(registry=registry).redact(sample)

    # A placeholder issued by one engine must be meaningful to the other: the
    # kinds come from one vocabulary, so the same value carries the same
    # category whichever engine happened to be installed.
    kinds = {name: sorted({e.kind for e in res.entries}) for name, res in vaults.items()}
    for name, found in kinds.items():
        print("  %-6s -> %s" % (name, ", ".join(found)))
    shared = set.intersection(*(set(v) for v in kinds.values()))
    check("SWAP-001", "engines agree on at least one category", bool(shared), str(kinds))
    check("SWAP-002", "no engine invents a category outside the vocabulary",
          all(k in CANONICAL_LABELS or k in ("EMAIL", "URL", "PHONE") for v in kinds.values() for k in v),
          str(kinds))
    for name, res in vaults.items():
        check("SWAP-003", "%-6s vault restores exactly" % name,
              restore(res.text, res.vault).text == sample)

print()
print("TOTAL FAILURES:", len(fails), fails if fails else "")
sys.exit(1 if fails else 0)
