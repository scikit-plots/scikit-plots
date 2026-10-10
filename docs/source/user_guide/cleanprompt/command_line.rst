..
  docs/source/user_guide/cleanprompt/command_line.rst

.. currentmodule:: scikitplot.cleanprompt

.. _cleanprompt-command-line:

======================================================================
The command line
======================================================================

Two entry points reach the same CleanPrompt command surface:

.. prompt:: bash $

   python -m scikitplot.cleanprompt doctor
   scikitplot cleanprompt doctor

The base command needs no optional package: it is always runnable. Optional
features say what they need when you ask for them.

The commands
------------

Grouped by what you are trying to do. Aliases are conveniences; automation
should prefer the canonical names.

.. list-table::
   :header-rows: 1
   :widths: 19 25 56

   * - Command
     - Category
     - Purpose
   * - ``encode``
     - prompt boundary
     - text in, a prompt you can paste into any chat out; aliases:
       ``clean``, ``prompt``
   * - ``decode``
     - prompt boundary
     - put the values back into the model's answer; alias: ``restore``
   * - ``redact``
     - explicit pipeline
     - redact text and write a named or default vault, with a report
   * - ``forget``
     - vault lifecycle
     - preview or delete a vault; alias: ``clear-vault``
   * - ``roundtrip``
     - learning/debugging
     - the whole loop in one command, nothing written; alias: ``demo``
   * - ``inspect``
     - diagnosis
     - dry run: what would be redacted, and what would not
   * - ``scan``
     - automation
     - exit non-zero when sensitive values are present (a CI gate)
   * - ``doctor``
     - diagnosis
     - what is active, what is blind, and what to install; alias:
       ``capabilities``
   * - ``kinds``
     - discovery
     - list the built-in detection kinds
   * - ``cli``
     - interactive
     - paste-and-go terminal session; aliases: ``session``, ``repl``
   * - ``flask``
     - interactive
     - local browser interface; aliases: ``web``, ``serve``
   * - ``packs``
     - structured data
     - list, show or check pack and format definitions
   * - ``batch``
     - structured data
     - encode a folder or zip into a safe copy, or decode one back
   * - ``ask``
     - model boundary
     - send one guarded prompt through any command-line model
   * - ``mcp``
     - agent boundary
     - serve guarded tools to an MCP agent over standard input and output
   * - ``plan``
     - reproducibility
     - save, show or verify a fingerprinted team plan
   * - ``skill``
     - agent setup
     - print or install the agent instructions
   * - ``docker``
     - deployment
     - write container files for the web interface

Use command-specific help for the exact options; both supported frontends
(``argparse`` and ``click``) render it from one internal definition, so the
two accept exactly the same command lines:

.. prompt:: bash $

   python -m scikitplot.cleanprompt encode --help

Giving it text
--------------

Every command that reads text — ``encode``, ``decode``, ``redact``,
``inspect``, ``scan``, ``roundtrip`` — takes it three ways:

.. code-block:: bash

   cleanprompt inspect "Mail ada@example.com"       # 1. as arguments
   cleanprompt inspect --in prompt.txt              # 2. from a file
   cleanprompt inspect < prompt.txt                 # 3. from standard input

Giving both arguments and ``--in`` is refused rather than silently preferring
one, because the ignored input is exactly the text you cared about. Run with
neither at a terminal and the command says, on standard error, that it is
reading standard input and that Ctrl-D on a blank line ends the paste.

For prose with parentheses, quotes, ``$`` or backticks, use a heredoc with a
**quoted** delimiter, which stops the shell from interpreting anything:

.. code-block:: bash

   cleanprompt encode <<'END'
   Ada (our lead) asked: "is $PATH set?" — ada@example.com
   END

The option grammar
------------------

* ``--`` ends the options: everything after it is text, even a token that
  starts with a dash. ``cleanprompt inspect -- --this-is-text``.
* A text argument that starts with a dash is refused unless ``--`` came
  first, so a mistyped option is never redacted as if it were your prompt.
* Long options are never abbreviated: ``--form`` is an error, not
  ``--format``. An abbreviation that works today breaks when a later version
  adds an option with the same prefix.
* A value that starts with a dash is written attached: ``--hide=-secret``.
* Short aliases: ``-h/--help``, ``-V/--version``, ``-i/--in``, ``-o/--out``,
  ``-f/--format``, ``-k/--kinds``, ``-q/--quiet``.

The vault on the command line
-----------------------------

Where it lives
^^^^^^^^^^^^^^

A vault holds the removed values **in clear text** unless encrypted. Its
default location is deliberately *not* the working directory — this tool runs
inside checkouts, and a vault beside the input file is one ``git add .`` from
committing the values you were trying not to send. The path is resolved in
this order:

1. ``--vault PATH``
2. ``$CLEANPROMPT_VAULT``
3. ``$XDG_STATE_HOME/cleanprompt/vault.json``
4. ``~/.local/state/cleanprompt/vault.json`` (``%LOCALAPPDATA%\cleanprompt\``
   on Windows)

On POSIX systems the directory is created ``0700`` and the file ``0600``, at
creation. Every write names the vault on standard error, and ``doctor``
reports the resolved path as ``configuration.vault_path``.

Append or overwrite
^^^^^^^^^^^^^^^^^^^

``encode`` defaults to ``--vault-mode append``
    a conversation: a value keeps the label it was first given, and a reply
    quoting an earlier turn still restores.

``redact`` defaults to ``--vault-mode overwrite``
    a scripted pass over a document: each run starts clean.

A vault written by an old format without a label index cannot be appended to
— that would renumber from ``[EMAIL-1]`` while the old vault already uses the
label for another value. The refusal costs one ``--vault-mode overwrite``; it
still restores normally.

Two commands that share a vault file hold a lock on it from read to write,
and the file is replaced atomically, never truncated.

Ending a conversation
^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $

   python -m scikitplot.cleanprompt forget
   python -m scikitplot.cleanprompt forget --force

The first is a dry run naming how many values of which categories the vault
holds — never a value. The second deletes it, even if it is damaged.
Unlinking a file is not a promise of physical erasure on journalling
filesystems, SSDs, or backups.

Encryption
^^^^^^^^^^

For values that must not be stored as clear text, encrypt the vault. The
portable cipher needs nothing installed; Fernet is the optional
``cryptography``-backed alternative:

.. prompt:: bash $

   python -m scikitplot.cleanprompt doctor --new-key
   python -m scikitplot.cleanprompt encode --encrypt "Mail ada@example.com"
   python -m scikitplot.cleanprompt encode --encrypt --cipher fernet \
       "Mail ada@example.com"

The key comes from ``CLEANPROMPT_VAULT_KEY``, otherwise from a hidden prompt
at a terminal — which keeps it out of your shell history. ``doctor --new-key``
prints a standard-library passphrase. Keep the passphrase outside the vault;
losing it makes the vault unrecoverable.

The portable cipher is a standard construction from ``hashlib``, ``hmac`` and
``secrets`` (scrypt or PBKDF2 key derivation, HMAC-SHA256 in counter mode,
encrypt-then-MAC with the label inside the tag) whose steps are written out
in the module so they can be reviewed; it is not a reviewed implementation of
AES. Fernet is the better primitive where ``cryptography`` can be installed.

Reading a reply
---------------

``decode`` repairs a bounded set of rewrites a model makes to placeholders —
case, separator, Markdown escaping, line wrapping — and reports each repair.
``--exact`` restores only the exact issued spelling; ``--strict`` refuses a
reply containing a placeholder the vault never issued.

Exit status
-----------

Exit status is part of the scripting contract:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Status
     - Meaning
   * - ``0``
     - command completed successfully
   * - ``1``
     - handled runtime error, such as an unreadable file or unusable vault
   * - ``2``
     - command-line usage error
   * - ``3``
     - ``scan`` found more sensitive values than ``--max-findings`` allows
   * - ``69``
     - a requested capability is unavailable: a missing tier, or an entity
       engine that is installed but not ready (no model, no data)
   * - ``130``
     - interrupted
   * - ``141``
     - downstream pipe closed early

A CI gate
---------

.. code-block:: bash

   cleanprompt scan --profile strict --in prompts/system.txt
   # exit 3 when anything is found; --max-findings N tolerates N while you
   # ratchet an existing codebase down

``--format json`` is the machine-readable form for every reporting command.
