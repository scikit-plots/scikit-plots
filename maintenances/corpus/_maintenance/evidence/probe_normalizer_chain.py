"""
Digest ``NormalizationPipeline`` results over many chains and texts (CX-12, D14).

Run from the repository root:
``python -B maintenances/corpus/_maintenance/evidence/probe_normalizer_chain.py [ROOT] [--texts DIR]``.
``ROOT`` defaults to this checkout. ``--texts`` adds every paragraph of the
``.md``/``.txt``/``.rst`` files under ``DIR``. Writes the number of results
and a SHA-256 digest of every normalised document (its full ``to_dict()``),
for every ordered chain of one to three steps drawn from the built-in
text-only steps, a step that sets another field, a custom step and a subclass
that overrides ``normalize_doc``. Run it on two checkouts: equal digests
mean equal documents for every chain and text.
"""

import argparse
import hashlib
import itertools
import json
import logging
import pathlib
import sys

#: Fixed texts: markup, repeated lines, mixed case, composed and decomposed
#: characters, runs of whitespace, empty lines.
TEXTS = [
    "Hello   World\n\nHello   World\n<b>Bold</b> &amp; plain",
    "  leading and trailing  ",
    "e\u0301 vs \u00e9 \ufb01 ligature \u00a0 nbsp",
    "<p>one</p>\n<p>one</p>\n\n\n<p>two</p>",
    "MiXeD Case \u0130stanbul stra\u00dfe",
    "line\r\nline\r\nother\tline",
    "x",
]


def _say(line: str) -> None:
    sys.stdout.write(line + "\n")


def _paragraphs(directory):
    for path in sorted(pathlib.Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".rst"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            yield from (p for p in text.split("\n\n") if p.strip())


def main() -> int:
    """Run every chain over every text and print the digest."""
    parser = argparse.ArgumentParser()
    default_root = pathlib.Path(__file__).resolve().parents[4]
    parser.add_argument("root", nargs="?", type=pathlib.Path, default=default_root)
    parser.add_argument("--texts", type=pathlib.Path, default=None)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    logging.disable(logging.WARNING)
    from scikitplot import corpus  # noqa: PLC0415

    class Tagging(corpus.NormalizerBase):
        """A custom step that sets another field and reads the text so far."""

        def normalize_doc(self, doc):
            seen = doc.normalized_text if doc.normalized_text is not None else doc.text
            return doc.replace(metadata={**doc.metadata, "seen": len(seen)})

    class Shouting(corpus.LowercaseNormalizer):
        """A subclass that overrides ``normalize_doc``: never composed."""

        def normalize_doc(self, doc):
            return doc.replace(normalized_text=self._get_source_text(doc).upper())

    steps = {
        "unicode": corpus.UnicodeNormalizer(form="NFKC"),
        "space": corpus.WhitespaceNormalizer(),
        "lines": corpus.WhitespaceNormalizer(collapse_newlines=False, strip=False),
        "html": corpus.HTMLStripNormalizer(),
        "lower": corpus.LowercaseNormalizer(locale_aware=True),
        "dedup": corpus.DedupLinesNormalizer(),
        "tag": Tagging(),
        "shout": Shouting(),
    }
    chains = [
        chain
        for size in (1, 2, 3)
        for chain in itertools.permutations(sorted(steps), size)
    ]
    real = list(_paragraphs(args.texts)) if args.texts else []
    digest = hashlib.sha256()
    results = 0
    for chain in chains:
        pipeline = corpus.NormalizationPipeline([steps[name] for name in chain])
        # Every chain on the fixed texts; real paragraphs on a slice per chain.
        offset = sum(map(len, chain)) % 97
        texts = TEXTS + real[offset::97]
        docs = [corpus.CorpusDocument.create("p.md", i, t) for i, t in enumerate(texts)]
        seeded = docs[0].replace(normalized_text="  Already   SET  ")
        for doc in [*pipeline.normalize_batch(docs), pipeline.normalize_doc(seeded)]:
            record = json.dumps(doc.to_dict(), sort_keys=True, default=str)
            digest.update(record.encode("utf-8", "surrogatepass") + b"\n")
            results += 1
    _say(f"root={args.root} python={sys.version.split()[0]}")
    _say(f"chains {len(chains)} results {results} (real paragraphs {len(real)})")
    _say(f"digest {digest.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
