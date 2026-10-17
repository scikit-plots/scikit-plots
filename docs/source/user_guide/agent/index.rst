.. currentmodule:: scikitplot.agent

.. _agent-index:
.. _agent-user-guide:

Agents: bounded, audited tool use
=================================

:mod:`scikitplot.agent` runs a goal through a loop in which a **planner**
proposes each next step and the **runtime** alone decides whether it runs. The
planner is yours: a language-model adapter or a plain rule. The runtime gives
every run the same guarantees:

* every action passes five gates, in order: **Validation** (the tool exists
  and the arguments match its schema), **Permissions** (the policy allows the
  tool and its kind of effect), **Scope** (the policy grants the scopes the tool
  needs), **Budget** (one more call stays within the limits) and **Approval** (a
  person approved this exact action, when the policy asks for it);
* a denied action is not an error: it is the step's outcome, and the planner
  reads it at its next decision;
* every run ends: decisions are counted against ``max_steps``, and tool
  calls, cost, tokens and running time against their own limits;
* every decision, denial, approval and tool call is written to a hash-chained
  audit trail;
* the state is saved after every step, so a run paused for approval, or
  stopped by a crash, resumes where it was.

The package uses only the Python standard library and imports no other
``scikitplot`` submodule: retrieval, MCP and redaction are tools you pass in.
A tool that retrieves is scoped twice, by two contracts: the Scope gate
decides whether the tool may run at all (``Tool.scopes`` against
``Policy.granted_scopes``), and the retrieval contract decides what it may
read: a tool over :mod:`scikitplot.corpus` passes a ``filter=`` to
``RetrievalIndex.search``, and an index bound to a ``SecurityFilter`` applies
it to every query; the MCP server takes the same filter (``--corpus-filter``).

A first run
-----------

A tool is a Python function with a JSON Schema of its keyword arguments. The
planner receives a copy of the run's state and the tools' descriptions, and
returns a :class:`Decision`.

.. code-block:: python

   from scikitplot.agent import Agent, Decision, Limits, Policy, Tool, verify_audit

   PAGES = {"spans": "A chunk's text is a slice of its source.", "ids": "64 hex digits."}

   lookup = Tool(
       "lookup",
       lambda topic: PAGES.get(topic, ""),
       description="Return the page about a topic.",
       parameters={
           "type": "object",
           "properties": {"topic": {"type": "string", "enum": sorted(PAGES)}},
           "required": ["topic"],
           "additionalProperties": False,
       },
       scopes={"docs:read"},
   )

   def planner(state, tools):
       if not state.steps:
           return Decision.act("lookup", {"topic": "spans"})
       return Decision.finish(state.steps[-1].result)

   agent = Agent(
       [lookup],
       planner,
       policy=Policy(granted_scopes={"docs:read"}),
       limits=Limits(max_steps=4, max_tool_calls=2),
   )
   state = agent.run("what is a chunk's text?")
   assert state.status.value == "completed"
   assert state.answer == "A chunk's text is a slice of its source."
   assert verify_audit(agent.audit.records) == []

Without ``granted_scopes={"docs:read"}`` the same run completes too, but its
first step is ``denied`` with ``scope_denied`` and the tool never runs.
``Policy()`` is least privilege: ``read`` tools that need no scope, and
nothing else.

Limits and how a run ends
-------------------------

:class:`Limits` holds the stopping conditions. Each is checked before the
call it bounds, so a call already running is not interrupted; a tool's own
timeout bounds one call.

=====================  =====================================================
Limit                  When it is reached
=====================  =====================================================
``max_steps``          No further decision is asked for (``stopped``,
                       ``max_steps``). Required: it is what ends every run.
``max_tool_calls``     A call beyond it is denied at the Budget gate.
``max_cost``           A call whose ``Tool.cost`` would pass it is denied; at
                       or past it no further decision is asked for. The
                       planner's reported cost counts too.
``max_tokens``         No further decision is asked for. The planner must then
                       report :class:`Usage` with every decision, or the run
                       fails (``usage_not_reported``): a limit that cannot be
                       enforced is refused, not ignored.
``max_seconds``        Running time, summed over the run's sessions; time
                       spent waiting for an approval does not count.
=====================  =====================================================

A run ends ``completed`` (the planner finished and the ``done`` criterion, if
given, accepted the answer), ``stopped`` (a limit; ``stop_reason`` names it),
``failed`` (the planner or the ``done`` criterion raised or returned something
invalid), or pauses ``awaiting_approval``.

Approval, pause and resume
--------------------------

``write`` and ``external`` tools run only under a policy that allows their
effect, and then, by default, only with a person's approval of each call. An
approval is bound to the digest of the exact action (run, step, tool,
arguments, effect and scopes) and of the run's history up to the request; at
resume the digest is recomputed from the tools as they are then and every
gate runs again, so an approval of an action that has since changed
authorises nothing (``approval_stale``). Each run's first audit record holds a
random nonce, so two runs never share a history, and a run id already in the
store or the audit trail is refused. Resuming a copy older than the stored
state is refused, and so is resuming a paused copy that the audit trail shows
was resumed already: an approval runs its action once. (A caller with neither
a store nor a shared audit sink holds every copy itself, and replays what it
resumes twice.)

.. code-block:: python

   from scikitplot.agent import Approval, MemoryStore

   written = []
   save = Tool(
       "save",
       lambda text: written.append(text) or "saved",
       parameters={
           "type": "object",
           "properties": {"text": {"type": "string"}},
           "required": ["text"],
           "additionalProperties": False,
       },
       effect="write",
   )

   def writer(state, tools):
       if not state.steps:
           return Decision.act("save", {"text": "the report"})
       return Decision.finish(state.steps[-1].outcome.value)

   store = MemoryStore()
   agent = Agent([save], writer, policy=Policy(allow_effects={"read", "write"}), store=store)
   paused = agent.run("save the report")
   assert paused.status.value == "awaiting_approval" and written == []

   request = paused.pending  # what is asked, and why
   approval = Approval(request.digest, granted=True, approver="allen")
   done = agent.resume(paused.run_id, approvals=[approval])  # read back from the store
   assert written == ["the report"] and done.answer == "executed"

An ``approver=`` callable answers at once instead (an automated policy, a
prompt to a person); an answer of ``None`` pauses the run. A refusal
(``granted=False``) is the step's outcome, ``approval_denied``.

If a run stops while a tool call is in flight (the process is killed), the
saved state says so; the call was charged against the limits when it started.
At resume an idempotent call (every ``read`` tool, by default) starts again,
through every gate, at most three times in all; a call of a non-idempotent
tool is not repeated, because whether it took effect is unknown: the step is
``interrupted``.

The audit trail
---------------

Each record names its run, a sequence number from 0, the event and its data,
and the digest of the record before it. :func:`verify_audit` finds a record
that was changed, removed or reordered. ``JsonlAuditSink(path)`` appends one
line per record (``fsync=True`` to survive a machine crash), and
:func:`read_audit` reads the file back. The chain cannot show a trail cut
after its last record, but the state can: it keeps ``audit_head``, the last
record's number and digest, and a resume whose trail does not hold that
record is refused with :class:`StateError`.

Measured
--------

On the corpus user guide (``recursive (500)`` chunks, BM25 search) with 82
tasks, each the first half of a guide sentence to be found, three rule
planners using ``search`` (cost 1) and ``read`` (cost 0.25); the benchmark is
``maintenances/agent/_maintenance/evidence/bench_r5a.py``, its output
``bench_r5a.out`` beside it:

==========  ========  =========  ==========  =====
Planner     Success   Decisions  Tool calls  Cost
==========  ========  =========  ==========  =====
one shot    0.91      2.00       1.00        1.00
verify      0.94      7.52       6.52        2.38
verify (3)  0.94      4.02       3.02        1.51
==========  ========  =========  ==========  =====

``one shot`` answers the first search hit, which is exactly plain search's
recall@1. ``verify`` reads the hits in order and answers the first whose text
holds the query; reading at most three gives the same success at two thirds
of the cost. In every run the audit chain verified and every tool call had
its audit record. Without the ``corpus:read`` scope every action of every run
was denied and no tool ran; with a ``write`` tool and no approver every run
paused before writing.

Writing a planner
-----------------

A planner is any callable ``planner(state, tools) -> Decision``:

* ``state`` is a copy of the :class:`RunState`, rebuilt from its saved form:
  ``state.goal``, ``state.steps`` (each with its decision, outcome, result
  or error) and ``state.observations()`` (the same as plain JSON, ready for a
  prompt); changing it changes nothing;
* ``tools`` is :meth:`Toolset.describe`: names, descriptions, JSON Schemas,
  effects, scopes and costs, ready for a model's tool-calling interface;
* return :meth:`Decision.act` or :meth:`Decision.finish`, with
  ``usage=Usage(tokens, cost)`` when a model produced it.

A planner that raises, or returns anything else, fails the run with
``planner_failed`` or ``invalid_decision``.
