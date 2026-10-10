..
  docs/source/user_guide/cleanprompt/security_and_limits.rst

.. currentmodule:: scikitplot.cleanprompt

.. _cleanprompt-security:

======================================================================
What it protects, and what it does not
======================================================================

CleanPrompt is a boundary for **detected values**. This page says exactly what
that covers, so you can decide what else your situation needs.

What is guaranteed
------------------

For every value an active detector finds, or that you name with ``hide``:

* it does not appear in the redacted text, and restoration puts back exactly
  what was written (:ref:`cleanprompt-how-it-works` lists the nine core
  runtime invariants the probes measure);
* it is never written to a log record — counts, kinds and labels are, values
  are not;
* it stays in the vault, which is a separate object from the redacted text so
  that no serializer emits both by accident, and which must not be sent with
  the prompt;
* a request that cannot be met fails loudly rather than running with less than
  you asked for: an entity engine that is not ready, a tier that is missing, a
  limit that is exceeded.

What is not guaranteed
----------------------

**Values no active detector recognises.** A zero-result inspection is not
proof that no sensitive information exists. It can mean a relevant detector is
disabled or not installed. Run ``doctor`` to see the blind spots, ``inspect``
to see what one input would expose, and use ``hide`` / ``--hide`` for values
you know about — the suggested terms derived from the input that ``inspect``
offers are candidates for exactly that, chosen by you, never redacted
automatically.

**Names without an entity engine.** No regular expression recognises a
person's name. Without a ready engine, names, organisations and places are a
reported blind spot (:ref:`cleanprompt-entity-detection`). With one, recall is
statistical: an engine can miss a name.

**Disguises other than compatibility forms.** The detection view finds values
written with invisible format characters, full-width or other single-character
compatibility forms, Unicode spaces and dashes, and field names written that
way in CSV/TSV headers, JSON keys and email headers. It does not yet cover an
invisible character inside a shell or ``.env`` variable name or a code
identifier, or inside a JSON string value; it does not fold look-alike letters
from another script (a Cyrillic letter that looks Latin is a different
string), a value broken up with ordinary spaces, or a value reworded.

**Address shapes outside the patterns' intent.** Each structural pattern
states what it accepts. The ``EMAIL`` pattern follows the RFC 5322 dot-atom
local part with an ASCII domain, and it is bounded by word boundaries. An
address with a non-ASCII letter in it — ``josé@example.com``, an
internationalised domain such as ``exämple.com``, or a look-alike letter from
another script at its start — is therefore not recognised **at all**, not
partly. Hide such addresses with ``hide`` or a custom pattern until the
library covers them; this gap is tracked for a future release.

**Meaning carried by what remains.** Redaction removes values; it does not
anonymise. The surrounding text, the placeholders' categories and the
structure of a document can still identify someone to a reader who knows the
context. Reversible pseudonymisation is not anonymisation.

**Prompt injection.** Keeping values out of a model's input does not make the
model's instructions safe. CleanPrompt does not detect or neutralise
instructions hidden in text you send, and it does not judge what a model does.
Tool calls are restricted to the value kinds you allow
(:ref:`cleanprompt-agents`), which limits what an injected instruction can
extract, not whether it is obeyed.

Things you are trusted with
---------------------------

**The vault.** It holds the removed values. On the command line its default
location is outside the working directory, created ``0700``/``0600`` on POSIX
systems. Encrypt it (``--encrypt``) when the values must not rest in clear
text, and delete it (``forget --force``) when a conversation is over.
Deleting a file is not physical erasure on journalling filesystems, SSDs or
backups.

**Exported handles.** :meth:`Handle.export` output contains the values. Store
it as you would a password.

**Pack files.** A custom pattern runs on every document; validation cannot
prove it finishes quickly, and a pattern with nested repetition can
backtrack catastrophically. Load pack files only from people you would accept
code from (:ref:`cleanprompt-pack-trust`).

**The web page.** It has no authentication. It binds loopback by default,
refuses other addresses unless acknowledged, and refuses debug mode off
loopback; a shared deployment needs authentication, TLS and a production
server in front of it (:ref:`cleanprompt-web-and-containers`).

**The portable cipher.** It is a standard composition built from the Python
standard library, written out step by step so it can be reviewed. Its tests
prove that it refuses a wrong passphrase, a flipped bit, a swapped label and a
truncated token; no test suite proves a construction secure. Fernet
(``--cipher fernet``) is the better primitive where ``cryptography`` can be
installed.
