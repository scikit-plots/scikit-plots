"""
The agent user guide runs as written (Rule 54) and names real contracts.
"""

from __future__ import annotations

import re
from pathlib import Path

from scikitplot import agent

HERE = Path(__file__).parent
BLOCK = re.compile(r"^\.\. code-block:: python\n\n((?:   .*\n|\n)+)", re.MULTILINE)


def _text() -> str:
    return (HERE / "index.rst").read_text(encoding="utf-8")


def test_every_example_runs_in_order() -> None:
    blocks = BLOCK.findall(_text())
    assert len(blocks) == 2  # noqa: PLR2004
    namespace: dict = {}
    for block in blocks:
        source = "\n".join(line[3:] for line in block.splitlines())
        exec(compile(source, "index.rst", "exec"), namespace)  # noqa: S102


def test_named_codes_and_statuses_exist() -> None:
    text = _text()
    values = (
        {c.value for c in agent.ErrorCode}
        | {s.value for s in agent.RunStatus}
        | {s.value for s in agent.StopReason}
        | {o.value for o in agent.StepOutcome}
    )
    # Field names of the public dataclasses are not vocabulary values.
    fields = set().union(
        *(
            cls.__dataclass_fields__
            for cls in (agent.RunState, agent.Limits, agent.Policy)
        )
    )
    named = set(re.findall(r"``([a-z_]+)``", text))
    vocabulary = {n for n in named if "_" in n} - fields
    assert vocabulary <= values, sorted(vocabulary - values)


def test_the_limits_table_names_every_limit() -> None:
    text = _text()
    for name in agent.Limits.__dataclass_fields__:
        assert f"``{name}``" in text
