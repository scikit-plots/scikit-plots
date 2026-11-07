"""
Fingerprint cleanprompt's output over a fixed corpus, to prove "unchanged".

Usage::

    python -B maintenances/cleanprompt/_maintenance/tools/output_digest.py [--check | --record]

Without a flag it prints the digest. ``--check`` exits 1 when the digest
differs from the recorded baseline; ``--record`` writes the baseline (only
after a deliberate change in default output, named in the round's records).

Notes
-----
**User notes.** An opt-in change (a new pack, an action, a set key) must not
change what anyone gets by default. This runs a fixed, seeded corpus of
documents (prose, Markdown, ``.env``, JSON and CSV, placeholder and surrogate
styles, every built-in pack) through the library and hashes every output in
order, so one hash says whether any default output moved.

**Developer notes.** Deterministic: a seeded :class:`random.Random`, a fixed
list of values, and no clock or environment input. The baseline lives in
``evidence/output_digest.json`` with the corpus size and seed it was
computed with, so a check with other parameters is refused rather than
compared. The corpus is the one used for every round since CP-108 (seed 115,
1500 documents); changing it changes the baseline and must be recorded. An
exception during encoding is part of the output (``ERR <type>``), so a
change that makes a document fail also changes the hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

#: The recorded baseline, relative to this subsystem's ``_maintenance``.
BASELINE = Path("evidence/output_digest.json")

#: Values that the default policy hides, and text it must leave.
_HOT = [
    "ada@example.com", "+1 555 010 4477", "4242 4242 4242 4242",
    "https://example.com/a?b=1", "192.168.1.10", "GB82 WEST 1234 5698 7654 32",
    "AKIAIOSFODNN7EXAMPLE", "123-45-6789", "Ann Lee", "Marion Holt",
    "[EMAIL-1]", "[email_1]",
]  # fmt: skip
_COLD = [
    "hello", "world", "2024-01-15", "note", "\n", "x\u200by", "caf\u00e9",
    "Acme Ltd", "password: s3cr3t!",
]  # fmt: skip
_FORMATS = ("text", "markdown", "env", "json", "csv")


def _root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "scikitplot" / "cleanprompt" / "_engine.py").is_file():
            return candidate
    msg = "could not locate the repository root"
    raise SystemExit(msg)


def corpus(documents: int, seed: int):
    """
    Yield ``(text, fmt, style)`` for each document of the fixed corpus.

    Parameters
    ----------
    documents : int
        How many documents.
    seed : int
        The random seed.

    Yields
    ------
    tuple of str
        The document, its format name and the tag style.
    """
    rng = random.Random(seed)  # noqa: S311 - a reproducible corpus, not a secret
    for index in range(documents):
        words = [
            rng.choice(_HOT if rng.random() < 0.4 else _COLD)  # noqa: PLR2004
            for _ in range(rng.randint(0, 16))
        ]
        text = " ".join(words)
        style = "surrogate" if index % 3 == 0 else "placeholder"
        fmt = _FORMATS[index % 5]
        if fmt == "json":
            text = json.dumps({"note": text, "email": rng.choice(_HOT)})
        elif fmt == "csv":
            cells = (rng.choice(_HOT), rng.choice(_HOT), text)
            text = "name,email,note\n" + ",".join(json.dumps(c) for c in cells) + "\n"
        elif fmt == "env":
            text = "NOTE=" + text.replace("\n", " ") + f"\nDB_PASSWORD=pw{index}\n"
        yield text, fmt, style


def digest(documents: int = 1500, seed: int = 115) -> str:
    """
    Return the SHA-256 of every output of the corpus, in order.

    Parameters
    ----------
    documents : int, default=1500
        Corpus size.
    seed : int, default=115
        Corpus seed.

    Returns
    -------
    str
        Hex digest.
    """
    sys.path.insert(0, str(_root()))
    from scikitplot.cleanprompt._plan import (  # ruff: ignore[import-outside-top-level]
        FluentCleanPrompt,
    )
    from scikitplot.cleanprompt._runtime import (  # ruff: ignore[import-outside-top-level]
        Cleaner,
    )

    hasher = hashlib.sha256()
    for text, fmt, style in corpus(documents, seed):
        cleaner = Cleaner(FluentCleanPrompt().style(style).packs("all").build())
        try:
            out = cleaner.encode_text(text, fmt, name="s." + fmt).text
        except Exception as error:  # noqa: BLE001 - the failure is the output
            out = "ERR " + type(error).__name__
        hasher.update(out.encode("utf-8") + b"\0")
    return hasher.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Print, check or record the digest."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="exit 1 if changed")
    mode.add_argument("--record", action="store_true", help="write the baseline")
    parser.add_argument("--documents", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=115)
    args = parser.parse_args(argv)
    if args.documents < 1:
        parser.error("--documents must be at least 1")
    baseline = Path(__file__).resolve().parent.parent / BASELINE
    value = digest(args.documents, args.seed)
    record = {"documents": args.documents, "seed": args.seed, "sha256": value}
    if args.record:
        baseline.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        sys.stdout.write(f"recorded {value}\n")
        return 0
    sys.stdout.write(f"{value}\n")
    if not args.check:
        return 0
    if not baseline.is_file():
        sys.stderr.write(f"no baseline at {BASELINE}; run --record once\n")
        return 1
    expected = json.loads(baseline.read_text(encoding="utf-8"))
    if (expected["documents"], expected["seed"]) != (args.documents, args.seed):
        sys.stderr.write(
            f"baseline was computed with {expected['documents']} documents and seed "
            f"{expected['seed']}; check with those\n"
        )
        return 1
    if expected["sha256"] != value:
        sys.stderr.write(f"default output changed: baseline {expected['sha256']}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
