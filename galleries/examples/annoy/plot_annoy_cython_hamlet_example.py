# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Approximate Nearest Neighbors with Annoy — A Hamlet Example
===========================================================

An example showing the :py:class:`~scikitplot.annoy._annoy.Index` class.
"""

# %%
# This notebook demonstrates the **scikitplot Annoy** library using Shakespeare's *Hamlet*
# as a real-world text corpus. We will:
#
# 1. **Parse** Hamlet into passages (speech blocks per character per scene)
# 2. **Embed** passages into dense vectors using TF-IDF → truncated SVD
# 3. **Build** Annoy indexes across multiple `dtype` / `metric` combinations
#    — including **float16** for half-precision (TensorFlow/PyTorch style)
# 4. **Query** for the most similar passages and compare results
# 5. **Benchmark** disk usage and build time across dtypes
#
# Why Annoy?
#
# | Feature | Value |
# |---|---|
# | Algorithm | Random-projection forest + priority search |
# | Index types | `int8` → `uint64` (any item-ID width) |
# | Data types | `float16`, `float32`, `float64`, `float128` |
# | Metrics | `angular`, `euclidean`, `manhattan`, `dot`, `hamming` |
# | Thread-safety | Concurrent reads after build (GIL released) |
# | Memory-map | `mmap`-based — indexes larger than RAM |
#
# **float16** stores each dimension in 2 bytes instead of 4 (float32) or 8 (float64).
# For large-scale embeddings (millions of items × hundreds of dimensions),
# this halves disk and memory usage with minimal accuracy loss — the same
# trade-off used by PyTorch's `torch.float16` and TensorFlow's `tf.float16`.

import os
import re
import sys
import time
import math
import random
import tempfile
import textwrap
from pathlib import Path
from collections import Counter, defaultdict
from pprint import pprint

# The scikitplot Annoy wrapper
# After building scikitplot with the fixed annoylib.h:
#   pip install -e .   # or:  pip install scikit-plots
from scikitplot.annoy._annoy import Index

print(f"Python {sys.version}")
print(f"Index class: {Index}")

# %%
## 1. Load & Parse Hamlet
#
# We use the full text of *Hamlet* (public domain, ~1600) and segment it into
# **speech blocks** — each block is one character's consecutive lines within a scene.
# This gives us ~300–600 passages, each with enough text for meaningful similarity.
#
# ```
# download_hamlet()
#         │
#         ├─ success → use Gutenberg full text
#         │
#         └─ failure → use embedded excerpt
#                  │
#                  ▼
#              raw_text
# ```

# ---------------------------------------------------------------------------
# Option A: Download from Project Gutenberg (if network available)
# Option B: Use bundled excerpt below
# ---------------------------------------------------------------------------
HAMLET_URL = "https://www.gutenberg.org/cache/epub/1524/pg1524.txt"

def download_hamlet(url: str = HAMLET_URL) -> str:
    """Attempt download; return raw text or empty string on failure."""
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.read().decode("utf-8-sig")
    except Exception as exc:
        print(f"Download failed ({exc}); using built-in excerpt.")
        return ""

raw_text = download_hamlet()

# Fallback: a representative excerpt covering Act I & Act III soliloquies
if not raw_text:
    raw_text = textwrap.dedent("""\
    The Project Gutenberg eBook of Hamlet

    This eBook is for the use of anyone anywhere in the United States and
    most other parts of the world at no cost and with almost no restrictions
    whatsoever. You may copy it, give it away or re-use it under the terms
    of the Project Gutenberg License included with this eBook or online
    at www.gutenberg.org. If you are not located in the United States,
    you will have to check the laws of the country where you are located
    before using this eBook.

    Title: Hamlet

    Author: William Shakespeare

    Release date: November 1, 1998 [eBook #1524]
                    Most recently updated: September 19, 2025

    Language: English

    Other information and formats: www.gutenberg.org/ebooks/1524

    Credits: Dianne Bean


    *** START OF THE PROJECT GUTENBERG EBOOK HAMLET ***

    THE TRAGEDY OF HAMLET, PRINCE OF DENMARK

    by William Shakespeare

    Contents

    ACT I
    Scene I. Elsinore. A platform before the Castle
    Scene II. Elsinore. A room of state in the Castle
    Scene III. A room in Polonius’s house
    Scene IV. The platform
    Scene V. A more remote part of the Castle

    ACT II
    Scene I. A room in Polonius’s house
    Scene II. A room in the Castle

    ACT III
    Scene I. A room in the Castle
    Scene II. A hall in the Castle
    Scene III. A room in the Castle
    Scene IV. Another room in the Castle

    ACT IV
    Scene I. A room in the Castle
    Scene II. Another room in the Castle
    Scene III. Another room in the Castle
    Scene IV. A plain in Denmark
    Scene V. Elsinore. A room in the Castle
    Scene VI. Another room in the Castle
    Scene VII. Another room in the Castle

    ACT V
    Scene I. A churchyard
    Scene II. A hall in the Castle

    Dramatis Personæ

    HAMLET, Prince of Denmark
    CLAUDIUS, King of Denmark, Hamlet’s uncle
    The GHOST of the late king, Hamlet’s father
    GERTRUDE, the Queen, Hamlet’s mother, now wife of Claudius
    POLONIUS, Lord Chamberlain
    LAERTES, Son to Polonius
    OPHELIA, Daughter to Polonius
    HORATIO, Friend to Hamlet
    FORTINBRAS, Prince of Norway
    VOLTEMAND, Courtier
    CORNELIUS, Courtier
    ROSENCRANTZ, Courtier
    GUILDENSTERN, Courtier
    MARCELLUS, Officer
    BARNARDO, Officer
    FRANCISCO, a Soldier
    OSRIC, Courtier
    REYNALDO, Servant to Polonius
    Players
    A Gentleman, Courtier
    A Priest
    Two Clowns, Grave-diggers
    A Captain
    English Ambassadors.
    Lords, Ladies, Officers, Soldiers, Sailors, Messengers, and Attendants

    SCENE. Elsinore.

    ACT I
    SCENE I. Elsinore. A platform before the Castle.

    Enter Francisco and Barnardo, two sentinels.

    BERNARDO. Who's there?
    FRANCISCO. Nay, answer me. Stand and unfold yourself.
    BERNARDO. Long live the King!
    FRANCISCO. Bernardo?
    BERNARDO. He.
    FRANCISCO. You come most carefully upon your hour.
    BERNARDO. ’Tis now struck twelve. Get thee to bed, Francisco.
    FRANCISCO. For this relief much thanks. 'Tis bitter cold, And I am sick at heart.
    BERNARDO. Have you had quiet guard?
    FRANCISCO. Not a mouse stirring.
    BERNARDO. Well, good night. If you do meet Horatio and Marcellus,
    The rivals of my watch, bid them make haste.
    FRANCISCO. I think I hear them. Stand, ho! Who is there?

    HORATIO. Friends to this ground.
    MARCELLUS. And liegemen to the Dane.
    FRANCISCO. Give you good night.
    MARCELLUS. O, farewell, honest soldier. Who hath reliev'd you?
    FRANCISCO. Bernardo has my place. Give you good night.

    [_Exit._]

    MARCELLUS. Holla, Bernardo!
    BERNARDO. Say, what, is Horatio there?
    HORATIO. A piece of him.
    BERNARDO. Welcome, Horatio. Welcome, good Marcellus.
    MARCELLUS. What, has this thing appear'd again to-night?
    BERNARDO. I have seen nothing.
    MARCELLUS. Horatio says 'tis but our fantasy.

    ACT I. SCENE II. Elsinore. A room of State in the Castle.

    KING. Though yet of Hamlet our dear brother's death
    The memory be green, and that it us befitted
    To bear our hearts in grief, and our whole kingdom
    To be contracted in one brow of woe,
    Yet so far hath discretion fought with nature
    That we with wisest sorrow think on him
    Together with remembrance of ourselves.

    HAMLET. A little more than kin, and less than kind.
    KING. How is it that the clouds still hang on you?
    HAMLET. Not so, my lord. I am too much i' the sun.

    QUEEN. Good Hamlet, cast thy nighted colour off,
    And let thine eye look like a friend on Denmark.
    Do not for ever with thy vailed lids
    Seek for thy noble father in the dust.
    Thou know'st 'tis common. All that lives must die,
    Passing through nature to eternity.
    HAMLET. Ay, madam, it is common.

    HAMLET. O that this too too solid flesh would melt,
    Thaw, and resolve itself into a dew!
    Or that the Everlasting had not fix'd
    His canon 'gainst self-slaughter! O God! God!
    How weary, stale, flat, and unprofitable
    Seem to me all the uses of this world!
    Fie on't! ah, fie! 'Tis an unweeded garden
    That grows to seed. Things rank and gross in nature
    Possess it merely. That it should come to this!

    SCENE III. A room in Polonius's house.

    LAERTES. My necessaries are embark'd. Farewell.
    And, sister, as the winds give benefit
    And convoy is assistant, do not sleep
    But let me hear from you.
    OPHELIA. Do you doubt that?
    LAERTES. For Hamlet, and the trifling of his favour,
    Hold it a fashion, and a toy in blood;
    A violet in the youth of primy nature,
    Forward, not permanent, sweet, not lasting,
    The perfume and suppliance of a minute;
    No more.
    OPHELIA. No more but so?
    LAERTES. Think it no more.

    POLONIUS. Yet here, Laertes? Aboard, aboard, for shame!
    The wind sits in the shoulder of your sail,
    And you are stay'd for. There- my blessing with thee!
    And these few precepts in thy memory
    Look thou character. Give thy thoughts no tongue,
    Nor any unproportion'd thought his act.
    Be thou familiar, but by no means vulgar;
    Those friends thou hast, and their adoption tried,
    Grapple them unto thy soul with hoops of steel;
    But do not dull thy palm with entertainment
    Of each new-hatch'd, unfledg'd comrade. Beware
    Of entrance to a quarrel; but being in,
    Bear't that the opposed may beware of thee.
    Give every man thy ear, but few thy voice;
    Take each man's censure, but reserve thy judgment.
    This above all- to thine own self be true,
    And it must follow, as the night the day,
    Thou canst not then be false to any man.

    ACT I. SCENE IV. The platform.

    HAMLET. The air bites shrewdly; it is very cold.
    HORATIO. It is a nipping and an eager air.
    HAMLET. What hour now?
    HORATIO. I think it lacks of twelve.
    MARCELLUS. No, it is struck.
    HORATIO. Indeed? I heard it not.

    HAMLET. Angels and ministers of grace defend us!
    Be thou a spirit of health or goblin damn'd,
    Bring with thee airs from heaven or blasts from hell,
    Be thy intents wicked or charitable,
    Thou com'st in such a questionable shape
    That I will speak to thee.

    ACT I. SCENE V. A more remote part of the Castle.

    GHOST. I am thy father's spirit,
    Doom'd for a certain term to walk the night,
    And for the day confin'd to fast in fires,
    Till the foul crimes done in my days of nature
    Are burnt and purg'd away. But that I am forbid
    To tell the secrets of my prison house,
    I could a tale unfold whose lightest word
    Would harrow up thy soul, freeze thy young blood,
    Make thy two eyes, like stars, start from their spheres.

    HAMLET. O all you host of heaven! O earth! What else?
    And shall I couple hell? O, fie! Hold, hold, my heart,
    And you, my sinews, grow not instant old,
    But bear me stiffly up. Remember thee!
    Ay, thou poor ghost, while memory holds a seat
    In this distracted globe. Remember thee!

    ACT III. SCENE I. A room in the Castle.

    HAMLET. To be, or not to be- that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune
    Or to take arms against a sea of troubles,
    And by opposing end them. To die- to sleep-
    No more; and by a sleep to say we end
    The heartache, and the thousand natural shocks
    That flesh is heir to. 'Tis a consummation
    Devoutly to be wish'd. To die- to sleep.
    To sleep- perchance to dream: ay, there's the rub!
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause. There's the respect
    That makes calamity of so long life.

    HAMLET. Get thee to a nunnery. Why wouldst thou be a
    breeder of sinners? I am myself indifferent honest,
    but yet I could accuse me of such things that it
    were better my mother had not borne me.

    OPHELIA. O, what a noble mind is here o'erthrown!
    The courtier's, soldier's, scholar's, eye, tongue, sword;
    The expectancy and rose of the fair state,
    The glass of fashion and the mould of form,
    The observed of all observers, quite, quite down!

    ACT III. SCENE II. A hall in the Castle.

    HAMLET. Speak the speech, I pray you, as I pronounced it to
    you, trippingly on the tongue. But if you mouth it,
    as many of your players do, I had as lief the
    town crier spoke my lines. Nor do not saw the air
    too much with your hand, thus, but use all gently;
    for in the very torrent, tempest, and, as I may say,
    the whirlwind of passion, you must acquire and beget
    a temperance that may give it smoothness.

    HAMLET. Suit the action to the word, the word to
    the action, with this special observance, that you
    o'erstep not the modesty of nature. For anything so
    overdone is from the purpose of playing, whose end,
    both at the first and now, was and is, to hold,
    as 'twere, the mirror up to nature.

    ACT III. SCENE III. A room in the Castle.

    KING. O, my offence is rank, it smells to heaven;
    It hath the primal eldest curse upon't,
    A brother's murder. Pray can I not,
    Though inclination be as sharp as will.
    My stronger guilt defeats my strong intent,
    And, like a man to double business bound,
    I stand in pause where I shall first begin,
    And both neglect.

    ACT V. SCENE I. A churchyard.

    HAMLET. Alas, poor Yorick! I knew him, Horatio. A fellow
    of infinite jest, of most excellent fancy. He hath
    borne me on his back a thousand times; and now, how
    abhorred in my imagination it is! My gorge rises at
    it. Here hung those lips that I have kissed I know
    not how oft. Where be your gibes now? Your
    gambols? Your songs? Your flashes of merriment
    that were wont to set the table on a roar?

    ACT V. SCENE II. A hall in the Castle.

    HAMLET. There's a divinity that shapes our ends,
    Rough-hew them how we will.
    HORATIO. That is most certain.

    HAMLET. Not a whit, we defy augury. There's a special
    providence in the fall of a sparrow. If it be now,
    'tis not to come; if it be not to come, it will be
    now; if it be not now, yet it will come. The
    readiness is all.
    """)

def strip_gutenberg_boilerplate(text: str) -> str:
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start != -1:
        text = text[start:]
        text = text[text.find("\n")+1:]
    if end != -1:
        text = text[:end]
    return text

raw_text = strip_gutenberg_boilerplate(raw_text)
print(f"Raw text length: {len(raw_text):,} chars")
print(raw_text[:200])

# %%
# ```
# raw_text
#    ↓
# clean_text()
#    ↓
# tokenize()
#    ↓
# chunk_text()
#    ↓
# vectorize()
#    ↓
# Annoy index
# ```

# ---------------------------------------------------------------------------
# Parse into passages: (character, act_scene, text)
# ---------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    """Normalize quotes and whitespace."""
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def parse_hamlet_passages(text: str) -> list[dict]:
    """
    Extract speech blocks from Hamlet text.

    Parameters
    ----------
    text : str
        Raw Hamlet text (Gutenberg or sample excerpt).

    Returns
    -------
    list of dict
        Each dict has keys: 'id', 'character', 'scene', 'text'.
    """
    passages = []
    current_scene = "Unknown"
    current_char = None
    current_lines = []

    # Scene header pattern: optional ACT prefix, SCENE number, optional description
    # Pattern: "ACT I. SCENE II." or "SCENE II."
    scene_pat = re.compile(
        r"^\s*(ACT\s+[IVX]+\.?\s*)?SCENE\s+[IVX]+\.?.*$",
        re.IGNORECASE,
    )
    # Character line pattern: uppercase name, period, optional initial text
    # Pattern: "CHARACTER_NAME." at start of line (all-caps name)
    char_pat = re.compile(r"^\s*([A-Z][A-Z '’\-]+)\.\s*(.*)$")

    def flush():
        """Flush accumulated lines into a passage."""
        nonlocal passages, current_char, current_scene, current_lines
        if current_char and current_lines:
            # joined = " ".join(current_lines).strip()
            joined = normalize_text(" ".join(current_lines))
            if len(joined) > 20:  # skip trivially short entries
                passages.append({
                    "id": len(passages),
                    "character": current_char.title(),
                    "scene": current_scene,
                    "text": joined,
                })

    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue

        # Detect scene headers
        if scene_pat.match(line):
            flush()
            current_char = None
            current_lines = []
            # Simplified scene label: "ACT I. SCENE II"
            # Extract "Act I Scene II" style label
            parts = [p.strip() for p in line.strip().split(".") if p.strip()]
            current_scene = ". ".join(parts[:2]) if len(parts) >= 2 else parts[0]

            # current_scene = re.sub(r"\s+", " ", line.strip().split(".")[0] + "." +
            #                        (line.strip().split(".")[1] if "." in line else "")).strip()
            # current_scene = line.strip().split(".")[0].strip()
            # parts = [p.strip() for p in line.strip().split(".") if p.strip()]
            # current_scene = ". ".join(parts[:2]) if len(parts) >= 2 else parts[0] if parts else "Unknown"
            continue

        # Detect character speech
        m = char_pat.match(line)
        if m:
            flush()
            current_char = m.group(1).strip()
            # ignore empty first line
            first_line = m.group(2).strip()
            current_lines = [first_line] if first_line else []
        elif current_char and line.strip():
            current_lines.append(line.strip())

    flush()  # final passage
    return passages


def merge_passages(passages: list[dict]) -> list[dict]:
    """
    Merge consecutive passages by the same character in the same scene.

    Returns a new list of merged passages with longer text blocks.
    """
    merged = []
    buffer = None

    for p in passages:
        if buffer is None:
            buffer = p.copy()
            continue

        if p["character"] == buffer["character"] and p["scene"] == buffer["scene"]:
            buffer["text"] += " " + p["text"]
        else:
            merged.append(buffer)
            buffer = p.copy()

    if buffer:
        merged.append(buffer)

    # Reassign IDs
    for i, p in enumerate(merged):
        p["id"] = i

    return merged

# Filter out non-character speakers
INVALID_SPEAKERS = {"All", "Both", "Danes", "Captain"}
passages = parse_hamlet_passages(raw_text)
passages = [p for p in passages if p["character"] not in INVALID_SPEAKERS]

print(f"Extracted {len(passages)} passages")
print(f"Characters: {sorted(set(p['character'] for p in passages))}")
print()

# Show a few examples
for p in passages[:7]:
    print(f"  [{p['id']:3d}] {p['character']:12s} | {p['scene']}")
    print(f"        {p['text'][:90]}...")
    print()

# %%
# ## 2. Build TF-IDF Embeddings (Pure Python)
#
# We create a simple **TF-IDF → SVD** embedding pipeline without external ML
# dependencies. This produces a dense vector per passage that captures its
# semantic content relative to the corpus.
#
# For production use, you would replace this with sentence-transformers,
# OpenAI embeddings, or any other embedding model.

# ---------------------------------------------------------------------------
# Tokenizer and TF-IDF (stdlib only)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase alpha tokens, drop short words."""
    return [
        w for w in re.findall(r"[a-z']+", text.lower())
        if len(w) > 2
    ]


def build_vocabulary(docs: list[list[str]], max_vocab: int = 2000) -> dict[str, int]:
    """
    Build word→index mapping from the top-frequency terms.

    Parameters
    ----------
    docs : list of token lists
    max_vocab : int
        Maximum vocabulary size.

    Returns
    -------
    dict mapping word → column index
    """
    freq = Counter()
    for doc in docs:
        freq.update(set(doc))  # document frequency (not term frequency)
    # Keep most common words (they appear in many documents → useful for IDF)
    most_common = freq.most_common(max_vocab)
    return {word: idx for idx, (word, _) in enumerate(most_common)}


def tfidf_vectors(
    docs: list[list[str]],
    vocab: dict[str, int],
) -> list[list[float]]:
    """
    Compute TF-IDF vectors for each document.

    Parameters
    ----------
    docs : list of token lists
    vocab : word → index mapping

    Returns
    -------
    list of float vectors, one per document, dimension = len(vocab)
    """
    n_docs = len(docs)
    n_terms = len(vocab)

    # Document frequency
    df = Counter()
    for doc in docs:
        for w in set(doc):
            if w in vocab:
                df[w] += 1

    vectors = []
    for doc in docs:
        tf = Counter(doc)
        vec = [0.0] * n_terms
        for word, count in tf.items():
            if word in vocab:
                idx = vocab[word]
                # TF: log(1 + count), IDF: log(N / df)
                idf = math.log(n_docs / max(df.get(word, 1), 1))
                vec[idx] = math.log(1 + count) * idf
        # L2-normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vec = [x / norm for x in vec]
        vectors.append(vec)

    return vectors


def reduce_dimensionality(
    vectors: list[list[float]],
    target_dim: int = 64,
    seed: int = 42,
) -> list[list[float]]:
    """
    Random projection to reduce dimensionality.

    This is a lightweight alternative to truncated SVD that preserves
    approximate distances (Johnson-Lindenstrauss lemma).

    Parameters
    ----------
    vectors : list of float vectors (n_docs × original_dim)
    target_dim : int
        Output dimension.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of float vectors (n_docs × target_dim)
    """
    rng = random.Random(seed)
    orig_dim = len(vectors[0])
    # Generate random projection matrix (Gaussian)
    scale = 1.0 / math.sqrt(target_dim)
    proj = [
        [rng.gauss(0.0, scale) for _ in range(target_dim)]
        for _ in range(orig_dim)
    ]

    result = []
    for vec in vectors:
        reduced = [0.0] * target_dim
        for i, val in enumerate(vec):
            if val != 0.0:
                for j in range(target_dim):
                    reduced[j] += val * proj[i][j]
        # L2-normalize for angular distance
        norm = math.sqrt(sum(x * x for x in reduced)) or 1.0
        result.append([x / norm for x in reduced])

    return result


# Merge passages first
passages = merge_passages(passages)

# Build the pipeline
tokenized = [tokenize(p["text"]) for p in passages]
vocab = build_vocabulary(tokenized, max_vocab=1500)
tfidf_vecs = tfidf_vectors(tokenized, vocab)

EMBED_DIM = 64  # Final embedding dimension
embeddings = reduce_dimensionality(tfidf_vecs, target_dim=EMBED_DIM, seed=42)

print(f"Vocabulary size : {len(vocab)}")
print(f"TF-IDF dimension: {len(tfidf_vecs[0])}")
print(f"Final embedding : {len(embeddings[0])}-d")
print(f"Num passages    : {len(embeddings)}")

# %%
# ## 3. Build Annoy Indexes — Core API Walkthrough
#
# The `Index` class is the main entry point. Here's the lifecycle:
#
# ```python
# idx = Index(f=64, metric='angular', dtype='float32')  # 1. Create
# idx.add_item(0, vector)                                # 2. Add items
# idx.build(n_trees=10)                                  # 3. Build forest
# neighbors = idx.get_nns_by_item(0, 5)                  # 4. Query
# idx.save('index.ann')                                  # 5. Persist
# ```

# ---------------------------------------------------------------------------
# 3a. Basic Index — angular metric, float32 (default)
# ---------------------------------------------------------------------------

idx = Index(
    f=EMBED_DIM,         # embedding dimension
    metric='angular',    # cosine-similarity based distance
    dtype='float32',     # 32-bit float storage (default)
    seed=42,             # reproducible builds
)

# Add all passage embeddings
for i, vec in enumerate(embeddings):
    idx.add_item(i, vec)

print(f"Items added: {idx.get_n_items()}")

# Build the index forest
# More trees → better recall, more memory, slower build
idx.build(n_trees=10)

print(f"Trees built: {idx.get_n_trees()}")
print(f"Index ready for queries!")

# %%

# ---------------------------------------------------------------------------
# 3b. Query: Find passages most similar to "To be or not to be"
# ---------------------------------------------------------------------------

# Find passage ID by exact text substring
def find_passage_by_text(passages: list[dict], query: str) -> int:
    """
    Return the passage ID containing the query string (case-insensitive).

    Parameters
    ----------
    passages : list of dict
        Each passage has 'id', 'character', 'scene', 'text'.
    query : str
        Substring to search for.

    Returns
    -------
    int
        ID of the passage containing the query.

    Raises
    ------
    ValueError
        If the query string is not found.
    """
    query_lower = query.lower()
    for p in passages:
        if query_lower in p["text"].lower():
            return p["id"]
    raise ValueError(f"Query string not found in any passage: '{query}'")


# Fallback: find passage by token overlap
def find_passage_by_text_fallback(passages: list[dict], query: str) -> int:
    """
    Try exact match first; fall back to token overlap scoring.

    Parameters
    ----------
    passages : list of dict
    query : str

    Returns
    -------
    int
        Best-matching passage ID.
    """
    try:
        return find_passage_by_text(passages, query)
    except ValueError:
        query_tokens = set(tokenize(query))
        best_id, best_score = 0, 0
        for p in passages:
            score = sum(1 for w in query_tokens if w in set(tokenize(p["text"])))
            if score > best_score:
                best_score = score
                best_id = p["id"]
        return best_id


def display_neighbors(idx, passages, query_id, n=5):
    """
    Query the index and display results.

    Parameters
    ----------
    idx : Index
        Built Annoy index.
    passages : list of dict
        Passage metadata.
    query_id : int
        Item ID to query.
    n : int
        Number of neighbors.
    """
    neighbors, distances = idx.get_nns_by_item(
        query_id, n,
        search_k=-1,               # auto search effort
        include_distances=True,     # also return distance values
    )

    query_p = passages[query_id]
    print(f"Query: [{query_id}] {query_p['character']} ({query_p['scene']})")
    print(f"  \"{query_p['text'][:80]}...\"")
    print()
    print(f"{'Rank':<5} {'ID':<5} {'Dist':>8} {'Character':<14} {'Passage excerpt'}")
    print("-" * 90)

    for rank, (nid, dist) in enumerate(zip(neighbors, distances), 1):
        p = passages[nid]
        marker = " ← query" if nid == query_id else ""
        print(
            f"{rank:<5d} {nid:<5d} {dist:>8.4f} {p['character']:<14s} "
            f"{p['text'][:50]}...{marker}"
        )

# %%
# Find the canonical soliloquy "To be or not to be"
# tobe_id = find_passage_by_text(passages, "To be or not to be")
# Now the soliloquy should be found reliably
tobe_id = find_passage_by_text_fallback(
    passages,
    "To be, or not to be"
)
print("=" * 90)
print("QUERY: 'To be, or not to be' soliloquy")
print("=" * 90)
# Display top 5 nearest neighbors in the embedding space
display_neighbors(idx, passages, tobe_id, n=5)

# %%

# ---------------------------------------------------------------------------
# 3c. Query by vector (not by item ID)
# ---------------------------------------------------------------------------

# Create a synthetic query: average of two Hamlet passages
# This simulates querying with a new embedding not in the index.
def average_embeddings(ids: list[int], embeddings: list[list[float]]) -> list[float]:
    avg_vec = [0.0] * len(embeddings[0])
    for i in ids:
        for d, val in enumerate(embeddings[i]):
            avg_vec[d] += val
    avg_vec = [x / len(ids) for x in avg_vec]
    norm = math.sqrt(sum(x*x for x in avg_vec)) or 1.0
    return [x / norm for x in avg_vec]


# yorick_id = find_passage_by_text(passages, "Alas poor Yorick")
yorick_id = find_passage_by_text_fallback(passages, "Alas poor Yorick")
# Average the "To be" and "Yorick" embeddings
# query_vec = [
#     (embeddings[tobe_id][d] + embeddings[yorick_id][d]) / 2.0
#     for d in range(EMBED_DIM)
# ]
query_vec = average_embeddings([tobe_id, yorick_id], embeddings)

# Normalize
qnorm = math.sqrt(sum(x * x for x in query_vec)) or 1.0
query_vec = [x / qnorm for x in query_vec]

# Query by vector
neighbors, distances = idx.get_nns_by_vector(
    query_vec, 5, include_distances=True
)

print("=" * 90)
print("QUERY BY VECTOR: average of 'To be' + 'Alas poor Yorick'")
print("=" * 90)
print()
print(f"{'Rank':<5} {'ID':<5} {'Dist':>8} {'Character':<14} {'Passage excerpt'}")
print("-" * 90)
for rank, (nid, dist) in enumerate(zip(neighbors, distances), 1):
    p = passages[nid]
    print(
        f"{rank:<5d} {nid:<5d} {dist:>8.4f} {p['character']:<14s} "
        f"{p['text'][:55]}..."
    )

# %%
# ## 4. float16 — Half-Precision for Fast & Compact Indexes
#
# The `float16` dtype stores each embedding dimension in **2 bytes** instead of
# 4 (float32) or 8 (float64). This is the same half-precision format used by:
#
# - **PyTorch**: `torch.float16` / `torch.half`
# - **TensorFlow**: `tf.float16`
# - **NumPy**: `np.float16`
#
# ### Trade-offs
#
# | Aspect | float16 | float32 | float64 |
# |---|---|---|---|
# | Bytes/dimension | 2 | 4 | 8 |
# | Precision (decimal digits) | ~3.3 | ~7.2 | ~15.9 |
# | Range | ±65504 | ±3.4×10³⁸ | ±1.8×10³⁰⁸ |
# | Typical use case | Storage, inference | Training, general | Scientific computing |
# | Disk for 1M × 128-d | ~244 MB | ~488 MB | ~976 MB |
#
# For ANN search, the **distance ranking** (which neighbor is closest) is far
# more important than the exact distance value. float16 preserves ranking
# for well-separated vectors while halving storage.



# %%

# ---------------------------------------------------------------------------
# 4a. Build float16 index — same data, half the storage
# ---------------------------------------------------------------------------

idx_f16 = Index(
    f=EMBED_DIM,
    metric='angular',
    dtype='float16',     # ← half precision
    seed=42,
)

for i, vec in enumerate(embeddings):
    idx_f16.add_item(i, vec)

idx_f16.build(n_trees=10)

print(f"float16 index built: {idx_f16.get_n_items()} items, {idx_f16.get_n_trees()} trees")

# Query the same passage
print()
print("=" * 90)
print("float16 QUERY: 'To be, or not to be'")
print("=" * 90)
display_neighbors(idx_f16, passages, tobe_id, n=5)

# %%

# ---------------------------------------------------------------------------
# 4b. Compare float16 vs float32 results
# ---------------------------------------------------------------------------

def compare_dtypes(passages, embeddings, query_id, n=10):
    """
    Build indexes with different dtypes and compare neighbor rankings.

    Parameters
    ----------
    passages : list of dict
    embeddings : list of float vectors
    query_id : int
    n : int
        Number of neighbors to retrieve.

    Returns
    -------
    dict mapping dtype → list of neighbor IDs
    """
    results = {}

    for dtype in ['float16', 'float32', 'float64']:
        t0 = time.perf_counter()
        ix = Index(f=EMBED_DIM, metric='angular', dtype=dtype, seed=42)
        for i, vec in enumerate(embeddings):
            ix.add_item(i, vec)
        ix.build(n_trees=10)
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        neighbors, distances = ix.get_nns_by_item(query_id, n, include_distances=True)
        query_time = time.perf_counter() - t0

        results[dtype] = {
            'neighbors': neighbors,
            'distances': distances,
            'build_ms': build_time * 1000,
            'query_us': query_time * 1e6,
        }

        # Save to temp file to measure disk size
        with tempfile.NamedTemporaryFile(suffix='.ann', delete=False) as tmp:
            ix.save(tmp.name)
            results[dtype]['disk_bytes'] = os.path.getsize(tmp.name)
            os.unlink(tmp.name)

    return results


results = compare_dtypes(passages, embeddings, tobe_id, n=10)

# Display comparison table
print(f"{'dtype':<10} {'Build(ms)':>10} {'Query(μs)':>10} {'Disk(KB)':>10} {'Top-5 IDs'}")
print("-" * 70)
for dtype, r in results.items():
    top5 = str(r['neighbors'][:5])
    print(
        f"{dtype:<10} {r['build_ms']:>10.1f} {r['query_us']:>10.1f} "
        f"{r['disk_bytes']/1024:>10.1f} {top5}"
    )

# Compute ranking agreement
f32_set = set(results['float32']['neighbors'])
f16_set = set(results['float16']['neighbors'])
f64_set = set(results['float64']['neighbors'])

print()
print(f"Overlap float16 ∩ float32 (top-10): {len(f16_set & f32_set)}/10")
print(f"Overlap float16 ∩ float64 (top-10): {len(f16_set & f64_set)}/10")
print(f"Overlap float32 ∩ float64 (top-10): {len(f32_set & f64_set)}/10")

# %%
# ## 5. Multiple Distance Metrics
#
# Annoy supports several metrics, each suited to different data:
#
# | Metric | Formula | Best for |
# |---|---|---|
# | `angular` | 1 − cos(u, v) | Normalized embeddings, text similarity |
# | `euclidean` | ‖u − v‖₂ | Spatial data, un-normalized vectors |
# | `manhattan` | ‖u − v‖₁ | Sparse features, robust to outliers |
# | `dot` | −u·v (negated) | Recommendation scores, un-normalized |
# | `hamming` | popcount(u ⊕ v) | Binary features, hash codes |

# ---------------------------------------------------------------------------
# 5a. Compare metrics on the same query
# ---------------------------------------------------------------------------

for metric in ['angular', 'euclidean', 'manhattan', 'dot']:
    ix = Index(f=EMBED_DIM, metric=metric, dtype='float32', seed=42)
    for i, vec in enumerate(embeddings):
        ix.add_item(i, vec)
    ix.build(n_trees=10)

    neighbors, distances = ix.get_nns_by_item(tobe_id, 5, include_distances=True)

    print(f"\n{'='*60}")
    print(f"Metric: {metric}")
    print(f"{'='*60}")
    for rank, (nid, dist) in enumerate(zip(neighbors, distances), 1):
        p = passages[nid]
        print(f"  {rank}. [{nid:3d}] d={dist:.4f}  {p['character']:<12s} {p['text'][:45]}...")

# %%
# ## 6. Extended Index Types
#
# The `index_dtype` parameter controls the width of item IDs:
#
# | `index_dtype` | Max items | Memory/ID | Use case |
# |---|---|---|---|
# | `int8` | 127 | 1 byte | Tiny prototype indexes |
# | `int16` | 32,767 | 2 bytes | Small datasets |
# | `int32` | ~2.1B | 4 bytes | Standard (default) |
# | `int64` | ~9.2×10¹⁸ | 8 bytes | Large-scale production |
# | `uint8` | 255 | 1 byte | Labels, categories |
# | `uint16` | 65,535 | 2 bytes | Medium catalogs |
# | `uint32` | ~4.3B | 4 bytes | Very large datasets |
# | `uint64` | ~9.2×10¹⁸ | 8 bytes | Maximum capacity |

# ---------------------------------------------------------------------------
# 6a. int8 index — tiny item IDs for small datasets (max 127 items)
# ---------------------------------------------------------------------------

# Use only first 100 passages (within int8 range)
n_small = min(100, len(passages))

idx_i8 = Index(
    f=EMBED_DIM,
    metric='angular',
    dtype='float16',       # half-precision data
    index_dtype='int8',    # tiny item IDs (1 byte each)
    seed=42,
)

for i in range(n_small):
    idx_i8.add_item(i, embeddings[i])

idx_i8.build(n_trees=5)

# Query
query_id_small = min(tobe_id, n_small - 1)
neighbors = idx_i8.get_nns_by_item(query_id_small, 5)

print(f"int8 + float16 index: {idx_i8.get_n_items()} items")
print(f"Neighbors of [{query_id_small}]: {neighbors}")
for nid in neighbors:
    p = passages[nid]
    print(f"  [{nid:3d}] {p['character']:<12s} {p['text'][:60]}...")

# %%
# ## 7. Hamming Distance — Binary Embeddings
#
# Hamming distance counts the number of differing bits between two binary vectors.
# This is useful for locality-sensitive hashing (LSH) or binary feature vectors.

# ---------------------------------------------------------------------------
# 7a. Convert embeddings to binary (threshold at median)
# ---------------------------------------------------------------------------

def binarize_embeddings(
    embeddings: list[list[float]],
) -> list[list[float]]:
    """
    Convert float embeddings to binary (0.0/1.0) via median thresholding.

    Each dimension is independently thresholded at its median value
    across all documents. This preserves relative ordering information.

    Parameters
    ----------
    embeddings : list of float vectors

    Returns
    -------
    list of binary (0.0/1.0) vectors
    """
    n_dims = len(embeddings[0])
    # Compute median per dimension
    medians = []
    for d in range(n_dims):
        vals = sorted(e[d] for e in embeddings)
        mid = len(vals) // 2
        medians.append(vals[mid])

    return [
        [1.0 if e[d] >= medians[d] else 0.0 for d in range(n_dims)]
        for e in embeddings
    ]


binary_embs = binarize_embeddings(embeddings)

# Build Hamming index
idx_hamming = Index(
    f=EMBED_DIM,
    metric='hamming',
    dtype='float32',  # Hamming requires float/double external type
    seed=42,
)

for i, bvec in enumerate(binary_embs):
    idx_hamming.add_item(i, bvec)

idx_hamming.build(n_trees=10)

neighbors, distances = idx_hamming.get_nns_by_item(
    tobe_id, 5, include_distances=True
)

print(f"Hamming index built: {idx_hamming.get_n_items()} items")
print()
print(f"{'Rank':<5} {'ID':<5} {'Hamming':>8} {'Character':<14} {'Passage excerpt'}")
print("-" * 80)
for rank, (nid, dist) in enumerate(zip(neighbors, distances), 1):
    p = passages[nid]
    print(
        f"{rank:<5d} {nid:<5d} {dist:>8.0f} {p['character']:<14s} "
        f"{p['text'][:48]}..."
    )

# %%
# ## 8. Save, Load & Memory-Mapped Queries
#
# Annoy indexes can be saved to disk and memory-mapped for:
# - Zero-copy loading (instant startup)
# - Sharing across processes
# - Indexes larger than available RAM

# ---------------------------------------------------------------------------
# 8a. Save and reload
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmpdir:
    # Save indexes in different dtypes
    for dtype in ['float16', 'float32', 'float64']:
        ix = Index(f=EMBED_DIM, metric='angular', dtype=dtype, seed=42)
        for i, vec in enumerate(embeddings):
            ix.add_item(i, vec)
        ix.build(n_trees=10)

        path = os.path.join(tmpdir, f"hamlet_{dtype}.ann")
        ix.save(path)

        size_kb = os.path.getsize(path) / 1024
        print(f"{dtype:<10} saved: {size_kb:>8.1f} KB  ({path})")

    print()

    # Reload and query — the loaded index is memory-mapped
    reload_path = os.path.join(tmpdir, "hamlet_float16.ann")
    idx_loaded = Index(f=EMBED_DIM, metric='angular', dtype='float16')
    idx_loaded.load(reload_path, prefault=False)  # mmap, no prefault

    neighbors = idx_loaded.get_nns_by_item(tobe_id, 5)
    print(f"Loaded float16 index from disk, queried [{tobe_id}]:")
    for nid in neighbors:
        p = passages[nid]
        print(f"  [{nid:3d}] {p['character']:<12s} {p['text'][:55]}...")

# %%
# ## 9. float128 — Extended Precision for Scientific Workloads
#
# For scenarios requiring maximum numerical fidelity (scientific computing,
# high-dimensional geometry, accumulation-sensitive distance computations),
# `float128` provides 128-bit storage:
#
# - On GCC/Clang (x86): native `__float128` (33 decimal digits)
# - On MSVC / other: falls back to `long double` (≥80-bit extended)
#
# This is the counterpart to float16's "trade accuracy for size" —
# float128 trades size for accuracy.

# ---------------------------------------------------------------------------
# 9a. float128 index (if supported on this platform)
# ---------------------------------------------------------------------------

try:
    idx_f128 = Index(
        f=EMBED_DIM,
        metric='euclidean',
        dtype='float128',    # ← extended precision
        seed=42,
    )

    for i, vec in enumerate(embeddings):
        idx_f128.add_item(i, vec)

    idx_f128.build(n_trees=10)

    neighbors_128, dists_128 = idx_f128.get_nns_by_item(
        tobe_id, 5, include_distances=True
    )

    print("float128 euclidean index — results:")
    for rank, (nid, dist) in enumerate(zip(neighbors_128, dists_128), 1):
        p = passages[nid]
        print(f"  {rank}. [{nid:3d}] d={dist:.10f}  {p['character']:<12s} {p['text'][:40]}...")

except Exception as exc:
    print(f"float128 not available on this platform: {exc}")
    print("This is expected on some ARM / MSVC builds.")

# %%
# ## 10. Putting It All Together — Character Similarity Analysis
#
# Let's use Annoy to find which *characters* speak most similarly,
# by averaging their passage embeddings and querying.

# ---------------------------------------------------------------------------
# 10a. Character-level embeddings (mean of passage embeddings)
# ---------------------------------------------------------------------------

char_embs = defaultdict(lambda: [0.0] * EMBED_DIM)
char_counts = Counter()

for p in passages:
    char_name = p['character']
    char_counts[char_name] += 1
    for d in range(EMBED_DIM):
        char_embs[char_name][d] += embeddings[p['id']][d]

# Normalize: divide by count, then L2-normalize
char_list = []
char_vecs = []
for char_name, count in char_counts.most_common():
    if count < 2:  # skip characters with too few lines
        continue
    vec = [char_embs[char_name][d] / count for d in range(EMBED_DIM)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    vec = [x / norm for x in vec]
    char_list.append(char_name)
    char_vecs.append(vec)

print(f"Characters with 2+ passages: {len(char_list)}")
for name in char_list:
    print(f"  {name:<15s} ({char_counts[name]} passages)")

# Build character-level index (float16 for efficiency)
idx_chars = Index(
    f=EMBED_DIM,
    metric='angular',
    dtype='float16',
    index_dtype='int8',   # few characters → tiny IDs
    seed=42,
)

for i, vec in enumerate(char_vecs):
    idx_chars.add_item(i, vec)

idx_chars.build(n_trees=10)

# Find who speaks most like Hamlet
if 'Hamlet' in char_list:
    hamlet_idx = char_list.index('Hamlet')
    n_query = min(len(char_list), 5)
    neighbors, distances = idx_chars.get_nns_by_item(
        hamlet_idx, n_query, include_distances=True
    )

    print(f"\nCharacters most similar to HAMLET in speech:")
    print(f"{'Rank':<5} {'Character':<15} {'Distance':>10} {'Passages':>10}")
    print("-" * 45)
    for rank, (nid, dist) in enumerate(zip(neighbors, distances), 1):
        name = char_list[nid]
        marker = " ←" if name == 'Hamlet' else ""
        print(f"{rank:<5d} {name:<15s} {dist:>10.4f} {char_counts[name]:>10d}{marker}")

# %%
# ## 11. API Quick Reference
#
# ```python
# from scikitplot.annoy._annoy import Index
#
# # ── Construction ──────────────────────────────────────────────
# idx = Index(
#     f=128,                  # embedding dimension
#     metric='angular',       # angular|euclidean|manhattan|dot|hamming
#     dtype='float16',        # float16|float32|float64|float128
#     index_dtype='int32',    # int8|int16|int32|int64|uint8|uint16|uint32|uint64
#     seed=42,                # reproducible builds
# )
#
# # ── Adding items ──────────────────────────────────────────────
# idx.add_item(item_id, vector)    # vector: list[float] of length f
#
# # ── Building ──────────────────────────────────────────────────
# idx.build(n_trees=10)            # more trees → better recall
#
# # ── Querying ──────────────────────────────────────────────────
# ids = idx.get_nns_by_item(id, n=10)                    # by item ID
# ids = idx.get_nns_by_vector(vec, n=10)                  # by vector
# ids, dists = idx.get_nns_by_item(id, 10,                # with distances
#                                   include_distances=True)
#
# # ── Inspection ────────────────────────────────────────────────
# idx.get_n_items()                # number of items
# idx.get_n_trees()                # number of trees
# idx.get_item(id)                 # retrieve stored vector
# idx.get_distance(i, j)           # distance between two items
#
# # ── Persistence ───────────────────────────────────────────────
# idx.save('index.ann')            # save to disk
# idx.load('index.ann')            # load (mmap-based, instant)
# idx.unload()                     # release memory
#
# # ── Context manager ───────────────────────────────────────────
# with Index(f=128, metric='angular') as idx:
#     idx.add_item(0, vec)
#     idx.build(10)
#     result = idx.get_nns_by_item(0, 5)
# # auto-cleanup on exit
# ```

# %%
# ## 12. Key Takeaways
#
# **float16 for production ANN search:**
# - Halves disk and memory usage vs float32
# - Preserves neighbor ranking for well-separated vectors
# - Compatible with PyTorch/TensorFlow half-precision workflows
# - Platform-agnostic: F16C hardware acceleration on x86, software fallback elsewhere
#
# **float128 for scientific computing:**
# - Maximum numerical precision for sensitive distance computations
# - Native `__float128` on GCC/Clang, `long double` fallback on MSVC
#
# **Extended index types (`int8` → `uint64`):**
# - Match ID width to your dataset size
# - Smaller IDs = less memory per tree node
#
# **Platform compatibility:**
# - All features work on x86, x64, ARM (Apple Silicon, Android, RPi)
# - Windows (MSVC), macOS (Clang), Linux (GCC) all supported
# - The fixed `annoylib.h` handles all platform differences via compile-time detection

# %%
#
# .. tags::
#
#    model-workflow: vector-db
#    level: beginner
#    purpose: showcase
