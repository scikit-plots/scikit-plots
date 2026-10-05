"""
Cleanprompt Advanced: Invariants, Encryption, and the Lossy Model Hop
=====================================================================

.. currentmodule:: scikitplot.cleanprompt

Everything in the two earlier examples is a consequence of a small number of
properties that the pipeline is built to hold. This example measures them
instead of describing them, and then looks at the two places where the design
has to deal with something it does not control: a disk that keeps what it is
given, and a language model that rewrites what it is given.

.. code-block:: text

    detect ── assign ── rewrite ──▶ [ the model ] ──▶ restore
      │         │          │              │              │
      │         │          │              │              └─ lenient, and reports
      │         │          │              │                 every repair
      │         │          │              └─ not controlled: the one hop where
      │         │          │                 the text can come back different
      │         │          └─ one pass over the ORIGINAL text, never over a
      │         │             partially rewritten one
      │         └─ one stable label per distinct value
      └─ detectors are pure: they read, they never rewrite

The encryption section runs with nothing optional installed. The ``fernet``
section reports a specific ``SKIP`` when ``cryptography`` is absent.

Every value below is reserved and cannot reach anybody.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import base64
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_WORKSPACE = tempfile.TemporaryDirectory(prefix="scikitplot-cleanprompt-advanced-")
_HOME = Path(_WORKSPACE.name)
_VAULT = _HOME / "vault.json"

os.environ["CLEANPROMPT_VAULT"] = str(_VAULT)


def _cli() -> list[str]:
    executable = shutil.which("scikitplot")
    if executable:
        return [executable, "cleanprompt"]
    return [sys.executable, "-m", "scikitplot.cleanprompt"]


CLI = _cli()


def run(*arguments: str, limit: int = 0, env: dict | None = None):
    """Run one CLI command and show both streams, with paths masked."""
    environment = dict(os.environ)
    if env:
        environment.update(env)
    completed = subprocess.run(
        [*CLI, *arguments],
        input="",
        capture_output=True,
        text=True,
        env=environment,
    )
    shown = " ".join(
        ('"{0}"'.format(one) if " " in one else one).replace(
            _WORKSPACE.name, "<workspace>"
        )
        for one in arguments
    )
    print("$ cleanprompt", shown)
    for label, stream in (("out", completed.stdout), ("err", completed.stderr)):
        if not stream:
            continue
        lines = stream.replace(_WORKSPACE.name, "<workspace>").rstrip("\n").split("\n")
        if limit and len(lines) > limit:
            lines = lines[:limit] + ["... ({0} more)".format(len(lines) - limit)]
        print("  {0} │ {1}".format(label, "\n      │ ".join(lines)))
    print("  exit │", completed.returncode)
    return completed


# %%
# 1. The invariants, measured
# ---------------------------
# Nine properties carry the design.  Six of them can be checked on randomised
# documents in a few lines, which is what the submodule's own probes do at
# larger scale; the numbers below are small so the example stays fast.

from scikitplot.cleanprompt import (  # noqa: E402
    DEFAULT_POLICY,
    Redactor,
    restore,
)

FRAGMENTS = [
    "mail ada@example.com",
    "call +1 555 0142",
    "host 192.0.2.10",
    "see https://example.com/report",
    "card 4242 4242 4242 4242",
    "and nothing sensitive at all",
    "IPv6 2001:db8::1",
    "then ada@example.com again",
]


def documents(count: int, seed: int = 20260922) -> list[str]:
    """Build reproducible documents out of the fragments above."""
    generator = random.Random(seed)
    built = []
    for _ in range(count):
        chosen = generator.sample(FRAGMENTS, generator.randint(2, len(FRAGMENTS)))
        built.append(". ".join(chosen) + ".")
    return built


CORPUS = documents(60)

print("documents:", len(CORPUS))

# %%
# **I1 — round trip.** Restoring the redacted text returns the original,
# byte for byte.

failures = [text for text in CORPUS
            if restore(Redactor().redact(text).text, Redactor().redact(text).vault).text != text]
print("I1 round trip          failures:", len(failures))
assert not failures

# %%
# **I2 — no leakage.** No value that was removed appears anywhere in the
# redacted text. This is the property the whole submodule exists for.

leaks = []
for text in CORPUS:
    outcome = Redactor().redact(text)
    for entry in outcome.entries:
        if entry.original in outcome.text:
            leaks.append((entry.label, entry.original))
print("I2 no leakage          failures:", len(leaks))
assert not leaks

# %%
# **I3 — determinism.** The same input and the same policy give the same
# output, including the labels, in this process and in the next one. Nothing
# depends on :func:`hash`, which Python randomises per process.

unstable = [
    text
    for text in CORPUS
    if Redactor().redact(text).text != Redactor().redact(text).text
]
print("I3 determinism         failures:", len(unstable))
assert not unstable

# %%
# **I5 — disjointness.** The spans that survive overlap arbitration never
# overlap each other. This is what lets the rewrite be a single pass over the
# original text, which in turn is what makes the offsets meaningful: a pass
# that rewrote as it went would invalidate every span after the first.

from scikitplot.cleanprompt import default_registry, resolve_spans  # noqa: E402

registry = default_registry()

overlapping = []
for text in CORPUS:
    kept = resolve_spans(registry.detect_all(text, DEFAULT_POLICY), DEFAULT_POLICY, text)
    bounds = [(span.start, span.end) for span in kept]
    if any(left[1] > right[0] for left, right in zip(bounds, bounds[1:])):
        overlapping.append(text)
print("I5 disjointness        failures:", len(overlapping))
assert not overlapping

# %%
# **I8 — spans index the original text.** Every surviving span, sliced out of
# the text it was found in, is the value the vault holds. An implementation
# that recovered offsets by searching would collapse every repeat of a value
# onto the first occurrence; this asserts that none does.

misaligned = []
for text in CORPUS:
    kept = resolve_spans(registry.detect_all(text, DEFAULT_POLICY), DEFAULT_POLICY, text)
    for span in kept:
        if text[span.start : span.end] != span.text:
            misaligned.append((text, span))
print("I8 offset alignment    failures:", len(misaligned))
assert not misaligned

# %%
# **I6 — idempotence.** Running the pipeline on its own output changes
# nothing. Without it, a placeholder would be detected as a value and wrapped
# again, and the vault would stop describing the text.

not_idempotent = []
for text in CORPUS:
    once = Redactor().redact(text).text
    twice = Redactor().redact(once).text
    if once != twice:
        not_idempotent.append(text)
print("I6 idempotence         failures:", len(not_idempotent))
assert not not_idempotent

# %%
# **I9 — value identity.** One distinct value gets exactly one label, however
# many times it occurs, and two distinct values never share one.

collisions = []
for text in CORPUS:
    outcome = Redactor().redact(text)
    labels = [entry.label for entry in outcome.entries]
    values = [entry.original for entry in outcome.entries]
    if len(set(labels)) != len(labels) or len(set(values)) != len(values):
        collisions.append(text)
print("I9 value identity      failures:", len(collisions))
assert not collisions

# %%
# 2. Overlap arbitration, chosen rather than inherited
# ----------------------------------------------------
# When two detectors claim characters, one has to win, and the choice is a
# policy field rather than registration order.  There is deliberately **no**
# "first wins" strategy: that would make the result depend on the order
# detectors happened to be added, which is exactly the hidden coupling that
# makes a redaction unreproducible.

from scikitplot.cleanprompt import OverlapError, OverlapStrategy  # noqa: E402

CLASH = "Mail ada@example.com about it"

for strategy in OverlapStrategy:
    policy = DEFAULT_POLICY.evolve(overlap=strategy)
    try:
        outcome = Redactor(policy=policy).redact(CLASH, extra_terms=["ada@example.com"])
        print("{0:<14} {1}".format(strategy.value, outcome.text))
    except OverlapError as exc:
        print("{0:<14} refused: {1}".format(strategy.value, str(exc)[:70]))

# %%
# ``LONGEST_WINS`` is the default and the safe direction: covering more of the
# text cannot disclose more than covering less.  ``STRICT`` belongs in a
# pipeline that must *prove* its detector set is disjoint — it turns a silent
# arbitration into a failed build.

# %%
# 3. Encrypting the vault, without a compiled dependency
# ------------------------------------------------------
# ``forget --force`` unlinks the vault, and on a journalling filesystem or an
# SSD that is not the same as destroying it.  For values that must not be
# recoverable, encrypt the vault with a passphrase you hold: the file on disk
# is then useless without it, whatever the filesystem did with the blocks.
#
# The default cipher needs **nothing installed**.  A vault that could only be
# opened where a compiled dependency happens to be present would have traded
# one risk for another, so the portable construction is built from the standard
# library alone:
#
# .. code-block:: text
#
#     passphrase
#        └─ scrypt  (PBKDF2-HMAC-SHA256 where the interpreter cannot run scrypt)
#              └─ 64 key bytes, split into an encryption key and a MAC key
#                    ├─ HMAC-SHA256 counter mode  → keystream → ciphertext
#                    └─ HMAC-SHA256 over cipher name ‖ label ‖ nonce ‖ ciphertext
#
# Encrypt-then-MAC, with the **label bound into the tag**, so a token cannot be
# moved from one placeholder to another.

from scikitplot.cleanprompt._vaultcrypt import (  # noqa: E402
    CIPHERS,
    DEFAULT_CIPHER,
    decrypt_mapping,
    encrypt_mapping,
    new_passphrase,
    resolve_cipher,
)

print("ciphers        :", CIPHERS)
print("default        :", DEFAULT_CIPHER)
print("'auto' resolves:", resolve_cipher("auto"), "(Fernet when available, else portable)")

# %%
# Generate a passphrase with the command line rather than inventing one.  The
# alphabet excludes the characters people confuse when reading aloud:
#
# .. code-block:: bash
#
#     cleanprompt doctor --new-key
#     export CLEANPROMPT_VAULT_KEY="$(cleanprompt doctor --new-key)"
#
# This example uses a fixed one so the output is reproducible.  Never do that
# with a real vault.

SAMPLE_KEY = "SAMPLE-KEY-4TXQ2-DEMO"

generated_passphrase = new_passphrase()
print(
    "a generated passphrase has the shape {0}, from an alphabet with no "
    "look-alike characters".format(
        "-".join("X" * len(group) for group in generated_passphrase.split("-"))
    )
)

# %%
# The properties the construction has to deliver, each checked rather than
# claimed.

entries, params = encrypt_mapping({"[EMAIL-1]": "ada@example.com"}, SAMPLE_KEY.encode())

print("kdf       :", params["name"], "n={0} r={1} p={2}".format(params["n"], params["r"], params["p"]))
print("token     :", entries["[EMAIL-1]"][:44], "...")
print("round trip:", decrypt_mapping(entries, SAMPLE_KEY.encode(), params))

# %%
# Two encryptions of the same value under the same key must differ — that is
# the nonce doing its job — and every way of interfering with a token must be
# refused rather than absorbed.

first, _ = encrypt_mapping({"[A-1]": "same"}, SAMPLE_KEY.encode())
second, _ = encrypt_mapping({"[A-1]": "same"}, SAMPLE_KEY.encode())
print("two encryptions differ:", first != second)

raw = bytearray(base64.b64decode(entries["[EMAIL-1]"]))
raw[-1] ^= 0x01
flipped = {"[EMAIL-1]": base64.b64encode(bytes(raw)).decode("ascii")}

ATTACKS = [
    ("wrong passphrase", entries, b"NOT-THE-KEY"),
    ("flipped bit", flipped, SAMPLE_KEY.encode()),
    ("token moved to another label", {"[EMAIL-2]": entries["[EMAIL-1]"]}, SAMPLE_KEY.encode()),
    ("truncated token", {"[EMAIL-1]": entries["[EMAIL-1]"][:20]}, SAMPLE_KEY.encode()),
]

for name, payload, passphrase in ATTACKS:
    try:
        decrypt_mapping(payload, passphrase, params)
        raise AssertionError("{0} was accepted".format(name))
    except AssertionError:
        raise
    except Exception as exc:  # noqa: BLE001 - the refusal is the assertion
        print("{0:<30} refused: {1}".format(name, str(exc)[:56]))

# %%
# What this does **not** claim is that the construction is secure; no example
# could establish that.  What is shown is that it behaves as described and
# fails closed when it is interfered with.

# %%
# The same thing from the command line.  ``CLEANPROMPT_VAULT_KEY`` supplies the
# passphrase non-interactively; without it the command prompts.

run("forget", "--force")
run(
    "encode",
    "--quiet",
    "--vault-mode",
    "overwrite",
    "--encrypt",
    "Mail ada@example.com",
    env={"CLEANPROMPT_VAULT_KEY": SAMPLE_KEY},
)

document = json.loads(_VAULT.read_text(encoding="utf-8"))
print()
print("encrypted :", document["encrypted"])
print("cipher    :", document["cipher"])
print("on disk   :", list(document["entries"].values())[0][:44], "...")
print("no value in the file:", "ada@example.com" not in _VAULT.read_text(encoding="utf-8"))
assert "ada@example.com" not in _VAULT.read_text(encoding="utf-8")

# %%
# With the passphrase, ``decode`` works as usual.  Without it, the vault is a
# file of unusable tokens.

run("decode", "I wrote to [EMAIL-1].", env={"CLEANPROMPT_VAULT_KEY": SAMPLE_KEY})
run("decode", "I wrote to [EMAIL-1].", env={"CLEANPROMPT_VAULT_KEY": "WRONG-KEY"})

# %%
# ``--cipher fernet`` uses the reviewed AES backend from ``cryptography`` when
# that tier is installed.  It is not the default, because the default has to
# work everywhere.
#
# The two ciphers do **not** share a key format, and this is worth seeing
# rather than reading: a portable passphrase is a human-readable string of any
# length, while a Fernet key is exactly 32 bytes in url-safe base64.  Handing
# one to the other is refused with the command that generates the right kind.

from scikitplot.cleanprompt import probe  # noqa: E402

crypto = probe("crypto")

if not crypto.available:
    print("[SKIP] --cipher fernet: {0}".format(crypto.detail))
else:
    print("crypto tier:", crypto.detail)

    # The portable passphrase, offered to Fernet: refused, with a remedy.
    run(
        "encode",
        "--quiet",
        "--vault-mode",
        "overwrite",
        "--encrypt",
        "--cipher",
        "fernet",
        "Mail bob@example.com",
        env={"CLEANPROMPT_VAULT_KEY": SAMPLE_KEY},
    )

    # Generate the right kind of key and complete the round trip.
    generated = subprocess.run(
        [*CLI, "doctor", "--new-key", "--cipher", "fernet"],
        input="",
        capture_output=True,
        text=True,
        check=True,
    )
    fernet_key = generated.stdout.strip()
    # Print the shape, never the bytes: a key in a documentation build is still
    # a key, and a gallery page is published.
    print(
        "generated Fernet key: {0} characters, url-safe base64, ends '{1}'".format(
            len(fernet_key), fernet_key[-1]
        )
    )

    run(
        "encode",
        "--quiet",
        "--vault-mode",
        "overwrite",
        "--encrypt",
        "--cipher",
        "fernet",
        "Mail bob@example.com",
        env={"CLEANPROMPT_VAULT_KEY": fernet_key},
    )

    written = json.loads(_VAULT.read_text(encoding="utf-8"))
    print("cipher on disk:", written["cipher"])
    assert written["encrypted"] and written["cipher"] != "cleanprompt-hmac-v1"

    run("decode", "I wrote to [EMAIL-1].", env={"CLEANPROMPT_VAULT_KEY": fernet_key})

# %%
# 4. The hop that is not controlled
# ---------------------------------
# Every invariant above holds across parts this submodule owns.  Restoration is
# the exception: between ``encode`` and ``decode`` the text passes through a
# language model, and the original design treated that hop as if it returned
# the bytes it was given.
#
# It does not.  Measured against realistic replies, a label comes back in eight
# shapes, and exact matching restored the first and missed the rest — while
# reporting ``restored 0 placeholder(s)`` at exit status ``0``.

from scikitplot.cleanprompt import encode  # noqa: E402

issued = encode("Mail ada@example.com")

SHAPES = [
    ("as issued", "[EMAIL-1]"),
    ("lower-cased", "[email-1]"),
    ("underscore for the hyphen", "[EMAIL_1]"),
    ("space for the hyphen", "[EMAIL 1]"),
    ("Unicode dash", "[EMAIL‑1]"),
    ("brackets escaped for Markdown", r"\[EMAIL-1\]"),
    ("separator escaped for Markdown", r"[EMAIL\_1]"),
    ("wrapped across a line", "[EMAIL-\n1]"),
    ("bold", "**[EMAIL-1]**"),
    ("in a code span", "`[EMAIL-1]`"),
]

for name, spelling in SHAPES:
    outcome = restore("I wrote to {0}.".format(spelling), issued.handle.vault)
    print(
        "{0:<32} restored={1!s:<6} repaired={2}".format(
            name, "ada@example.com" in outcome.text, bool(outcome.repaired)
        )
    )
    assert "ada@example.com" in outcome.text

# %%
# The leniency is bounded, and what bounds it is not the pattern.  A lenient
# match is acted on **only** when it resolves to a label the vault actually
# holds, so ordinary prose comes back byte-identical and is not reported as an
# unknown placeholder.
#
# Without that rule, a leniency wide enough to catch the shapes above would
# rewrite the model's own writing — a worse failure than the one it fixes.

for prose in ("[note 2]", "[1]", "[figure 3]", "[Table 1]", "[ISO 8601]"):
    source = "See {0} for details.".format(prose)
    outcome = restore(source, issued.handle.vault)
    print("{0:<12} unchanged={1!s:<6} unknown={2}".format(prose, outcome.text == source, outcome.unknown))
    assert outcome.text == source

# %%
# Two escape hatches, for callers who need the other behaviour:
#
# ``lenient=False`` (``--exact``)
#     restore only what was issued verbatim, and repair nothing.
# ``strict=True`` (``--strict``)
#     turn a placeholder the vault does not hold into an error rather than
#     leaving it in the text.

exact = restore("I wrote to [email-1].", issued.handle.vault, lenient=False)
print("lenient=False  restored:", len(exact.restored), " unknown:", exact.unknown)

# %%
# And the report that makes a silent failure loud.  A restoration that resolved
# nothing while the vault is non-empty says so, in words that name the next
# thing to check.

from scikitplot.cleanprompt import restoration_note  # noqa: E402

print(restoration_note(restore("Nothing to restore here.", issued.handle.vault)))

# %%
# 5. Preventing the rewrite rather than repairing it
# --------------------------------------------------
# Repair is the mitigation.  The prevention is to stop issuing tokens that
# invite rewriting: ``--style surrogate`` puts an ordinary-looking value where
# a bracket label would go, and a model has no reason to reformat a name.
#
# Use both. They fail in different ways, which is the point of having two.

from scikitplot.cleanprompt import TagStyle  # noqa: E402

surrogate = DEFAULT_POLICY.evolve(tag_style=TagStyle(style="surrogate"))
prose = Redactor(policy=surrogate).redact("Mail ada@example.com from 192.0.2.10")

print("surrogate :", prose.text)
print("no bracket token where the address was:", "[EMAIL" not in prose.text)
print("credential kinds keep their placeholder:", "[IPV4-1]" in prose.text)

# %%
# 6. Cleanup
# ----------

run("forget", "--force")

os.environ.pop("CLEANPROMPT_VAULT", None)
os.environ.pop("CLEANPROMPT_VAULT_KEY", None)
_WORKSPACE.cleanup()

print("Temporary workspace cleaned:", not Path(_WORKSPACE.name).exists())

# %%
#
# .. tags::
#
#    model-workflow: cleanprompt
#    plot-type: text
#    level: advanced
#    purpose: showcase
