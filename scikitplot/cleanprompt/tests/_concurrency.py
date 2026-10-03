"""
A shared harness for ``CP-067``: many threads, one vault owner.

Notes
-----
**Developer notes.** The interpreter's switch interval is shrunk for the
duration so threads interleave inside an encode, where the race lived; with
the default interval the race still occurs, only rarely. Every value is
unique per thread and turn except a small shared set, so a label issued to
two different values shows up as a round-trip failure or a label collision.
"""

from __future__ import annotations

import sys
import threading
from typing import Callable


def hammer(
    encode: Callable[[str], str],
    decode: Callable[[str], str],
    threads: int = 8,
    turns: int = 30,
) -> list:
    """
    Encode from many threads at once; return every inconsistency found.

    Parameters
    ----------
    encode, decode : callable
        The owner's encode and decode, each ``str -> str``.
    threads, turns : int
        How many threads, and how many texts each encodes.

    Returns
    -------
    list
        Empty when every text round-trips and no label names two values.
    """
    results: dict = {}
    errors: list = []

    def work(worker: int) -> None:
        try:
            for turn in range(turns):
                text = f"mail u{worker}_{turn}@example.com and s{turn % 5}@example.com"
                results[(worker, turn)] = (text, encode(text))
        except Exception as error:  # noqa: BLE001 - reported to the test
            errors.append(repr(error))

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        pool = [threading.Thread(target=work, args=(n,)) for n in range(threads)]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()
    finally:
        sys.setswitchinterval(previous)
    problems = list(errors)
    owner_of: dict = {}
    for text, safe in results.values():
        if decode(safe) != text:
            problems.append(("round trip", text, safe))
        values = text.replace(" and ", " ").split()[1:]
        labels = safe.replace(" and ", " ").split()[1:]
        for value, label in zip(values, labels):
            if owner_of.setdefault(label, value) != value:
                problems.append(("one label, two values", label))
    return problems
