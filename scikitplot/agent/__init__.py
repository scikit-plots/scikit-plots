# scikitplot/agent/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Bounded, audited tool use: a planner proposes, the runtime disposes.

:class:`Agent` runs a goal with a caller-supplied planner (a model adapter or
a rule) and a :class:`Toolset`. The planner reads the run's state and returns
a :class:`Decision`; the runtime alone decides whether it runs. Every action
passes five gates, in order -- Validation, Permissions, Scope, Budget,
Approval -- and every decision, denial, approval and tool call is written to
a hash-chained audit trail. The state is saved after every step, so a run
paused for a person's approval, or stopped by a crash, resumes where it was.

.. code-block:: python

    from scikitplot.agent import Agent, Limits, Policy, Tool, verify_audit

    search = Tool(
        "search",
        lambda query: [h.doc_id for h in index.search(query)],
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
        scopes={"corpus:read"},
    )
    agent = Agent(
        [search],
        my_planner,
        policy=Policy(granted_scopes={"corpus:read"}),
        limits=Limits(max_steps=8, max_tool_calls=4),
    )
    state = agent.run("which chunker keeps spans?")
    state.status, state.stop_reason, state.answer
    verify_audit(agent.audit.records)  # [] when the trail is whole

Notes
-----
**User.** Least privilege is the default: ``Policy()`` lets ``read`` tools that
need no scope run, and nothing else; ``write`` and ``external`` tools need a
policy that allows them, and then a person's approval of each call. A denied
action is not an error: it is the step's outcome, and the planner reads it at
its next decision.

**Developer.** Standard library only, and no other ``scikitplot`` submodule is
imported (invariant I9): retrieval, MCP and redaction are tools a caller
passes in. Every limit is checked before the call it bounds (Rule 127); a
limit that cannot be enforced is refused rather than ignored. The invariants
I1-I9 are listed in ``maintenances/agent/todo/todo.md``.
"""

from __future__ import annotations

from . import (
    _audit,
    _canonical,
    _errors,
    _loop,
    _policy,
    _schema,
    _state,
    _store,
    _tools,
)
from ._audit import *  # noqa: F403
from ._canonical import *  # noqa: F403
from ._errors import *  # noqa: F403
from ._loop import *  # noqa: F403
from ._policy import *  # noqa: F403
from ._schema import *  # noqa: F403
from ._state import *  # noqa: F403
from ._store import *  # noqa: F403
from ._tools import *  # noqa: F403

__all__: list[str] = []
__all__ += _audit.__all__
__all__ += _canonical.__all__
__all__ += _errors.__all__
__all__ += _loop.__all__
__all__ += _policy.__all__
__all__ += _schema.__all__
__all__ += _state.__all__
__all__ += _store.__all__
__all__ += _tools.__all__
