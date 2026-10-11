..
  docs/source/user_guide/cleanprompt/agents_and_models.rst

.. currentmodule:: scikitplot.cleanprompt

.. _cleanprompt-agents:

======================================================================
Models, tools and agents
======================================================================

:class:`Guard` is the one path from your data to any model. It has three
properties worth relying on:

* **encode, check, then call.** Text is encoded, then checked independently
  for any value the conversation holds; a leak raises :class:`LeakError`
  *before* your function is called.
* **no network of its own.** The guard owns no client, socket or key. You pass
  the function that talks to the model, so it works with every provider.
* **one vault per guard.** A value hidden once is hidden everywhere in that
  conversation, however it is written.

A gate in front of any client
-----------------------------

.. prompt:: python >>>

   from scikitplot.cleanprompt import FluentCleanPrompt

   guard = FluentCleanPrompt().profile("strict").guard()
   reply = guard.ask(
       "Mail ada@example.com",
       call=lambda safe: "Queued " + safe.split()[-1],
   )
   reply
   # 'Queued ada@example.com'

``call`` receives only the safe text and returns the model's answer, which the
guard decodes. :meth:`Guard.chat` does the same for a list of chat messages,
and :meth:`Guard.aask` / :meth:`Guard.achat` are the asynchronous forms. For
the lower-level steps, :meth:`Guard.outgoing` encodes and checks one text and
:meth:`Guard.incoming` decodes one reply.

Streamed replies
^^^^^^^^^^^^^^^^

A placeholder can arrive split across chunks. :meth:`Guard.stream` returns a
:class:`StreamDecoder` whose output, for every possible chunking, equals
decoding the whole reply at once:

.. prompt:: python >>>

   decoder = guard.stream()
   text = "".join(decoder.feed(chunk) for chunk in model_chunks) + decoder.flush()

Tool calls are an exit
^^^^^^^^^^^^^^^^^^^^^^

When a model asks to call a tool, its arguments may contain placeholders.
Restoring *every* value into a tool call would hand any tool — or an injected
instruction — every value in the conversation. So arguments are decoded only
through :meth:`Guard.decode_tool_arguments`, with the kinds that tool may
receive:

.. prompt:: python >>>

   args = guard.decode_tool_arguments(call.arguments, allow=["EMAIL"])

A call naming a kind it was not allowed is refused. :meth:`Guard.chat` returns
tool calls still encoded, so nothing is restored by accident.

Tool results are documents
^^^^^^^^^^^^^^^^^^^^^^^^^^

A tool's JSON result is guarded as one document, in the form it will be sent:
keys, numbers and strings, with prose inside JSON strings read as prose.
:meth:`Guard.encode_object` does that; :meth:`Guard.decode_object` reverses
it.

Command-line models
-------------------

For a model you run from a terminal, ``ask`` performs the same boundary
without a shell and keeps the values in memory:

.. prompt:: bash $

   python -m scikitplot.cleanprompt ask \
       --via "ollama run llama3" \
       "Summarize the request from ada@example.com"

The command is run directly (never through a shell), receives the safe text
on standard input, and is stopped if CleanPrompt exits early. ``--show-sent``
prints what was actually sent; ``--timeout`` bounds the wait.

Agents without code: MCP
------------------------

For agents that speak the Model Context Protocol, ``mcp`` serves guarded file
tools over standard input and output, confined to the roots you name:

.. prompt:: bash $

   python -m scikitplot.cleanprompt mcp --root ./project

Whatever a tool returns, the model reads — so no MCP tool returns a removed
value. There is deliberately no "decode" tool: reading goes through the gate,
and writing restores values on disk, where the model never sees them. Paths
are shown relative to a root.

``skill`` prints, or installs with ``--write``, instructions that make an AI
coding assistant use the same boundary; ``plan`` lets a team pin the packs and
rules those integrations are expected to use (:ref:`cleanprompt-files-and-packs`).

Logging without leaking
-----------------------

Nothing that was removed is ever logged. Log records carry counts, kinds,
labels, offsets, durations and capability decisions — what happened, never
what was found — and a filter on the whole ``scikitplot.cleanprompt`` logger
namespace scrubs held values as a second line of defence. Nothing is emitted
until you configure logging (``--log-level``, ``--log-format``, or
:func:`configure_logging`).

What a guard does not do
------------------------

A guard keeps the values it detects out of what the model sees. It does not
make a model's instructions safe: it is not a defence against prompt
injection, and it does not judge what a model does with the text it is given.
See :ref:`cleanprompt-security`.
