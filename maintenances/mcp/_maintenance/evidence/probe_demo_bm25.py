"""
Digest and time ``InMemoryBm25Retriever`` searches (finding CX-13).

Run from the repository root:
``python -B maintenances/mcp/_maintenance/evidence/probe_demo_bm25.py [ROOT] [--texts DIR]``.
``ROOT`` defaults to this checkout. ``--texts`` indexes every paragraph of the
``.md``/``.txt``/``.rst`` files under ``DIR`` (the first 20,000) instead of
the generated documents. Writes the number of hits, a SHA-256 digest of every
hit (document id and the score's exact bits, ``float.hex``) and the time per
search, for several ``k1``/``b``/``title_weight`` settings. Run it on two
checkouts: equal digests mean equal scores and order for every search.
"""

import argparse
import hashlib
import logging
import pathlib
import sys
import time

WORDS = [
    "memory",
    "recall",
    "graph",
    "index",
    "vector",
    "search",
    "rank",
    "fusion",
    "token",
    "chunk",
    "cache",
    "agent",
    "tool",
    "plan",
    "audit",
    "score",
    "model",
    "text",
    "query",
    "term",
    "the",
    "of",
    "and",
    "a",
    "to",
    "in",
]
SETTINGS = [
    {},
    {"k1": 0.5, "b": 0.0},
    {"k1": 2.0, "b": 1.0, "title_weight": 0.0},
    {"k1": 1.2, "b": 0.3, "title_weight": 5.5},
]
MAX_DOCS = 20_000


def _say(line: str) -> None:
    sys.stdout.write(line + "\n")


def _generated(count: int = 3000):
    for i in range(count):
        stream = hashlib.shake_256(f"mcp-r2:{i}".encode()).digest(2 + 60)
        size = 1 + stream[0] % 60
        words = [WORDS[stream[2 + j] % len(WORDS)] for j in range(size)]
        yield " ".join(words)


def _paragraphs(directory):
    for path in sorted(pathlib.Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".rst"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            yield from (p for p in text.split("\n\n") if p.strip())


def main() -> int:
    """Build each index, run every query, print the digest and the timings."""
    parser = argparse.ArgumentParser()
    default_root = pathlib.Path(__file__).resolve().parents[4]
    parser.add_argument("root", nargs="?", type=pathlib.Path, default=default_root)
    parser.add_argument("--texts", type=pathlib.Path, default=None)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    logging.disable(logging.WARNING)
    from scikitplot.mcp import DemoDocument, InMemoryBm25Retriever  # noqa: PLC0415

    texts = list(_paragraphs(args.texts)) if args.texts else list(_generated())
    texts = [t[:20_000] for t in texts[:MAX_DOCS]]
    documents = [
        DemoDocument(
            doc_id=f"d{i}",
            text=text,
            source_uri=f"doc-{i % 7}.md",
            title=" ".join(text.split()[:3]) if i % 3 == 0 else "",
        )
        for i, text in enumerate(texts)
    ]
    queries = [
        " ".join(text.split()[:5]) for text in texts[:: max(1, len(texts) // 150)]
    ]
    queries += [" ".join(WORDS[i : i + 1 + i % 4]) for i in range(len(WORDS))]
    queries += ["memory memory graph", "zzz-unknown", "The OF and", "token.", "  ", ""]
    digest = hashlib.sha256()
    hits = 0
    seconds = 0.0
    searches = 0
    for setting in SETTINGS:
        index = InMemoryBm25Retriever(documents, **setting)
        for query in queries:
            for k in (1, 5, 50):
                start = time.perf_counter()
                found = index.search(query, k)
                seconds += time.perf_counter() - start
                searches += 1
                for hit in found:
                    digest.update(f"{hit.doc_id} {hit.score.hex()}\n".encode())
                    hits += 1
                digest.update(b"--\n")
    _say(f"root={args.root} python={sys.version.split()[0]}")
    _say(f"documents {len(documents)} searches {searches} hits {hits}")
    _say(f"digest {digest.hexdigest()}")
    _say(f"search {seconds / searches * 1e3:.3f} ms each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
