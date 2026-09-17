"""
Conversion and text-format regressions (slices S-28, S-30).

Two silent losses. TOML conversion dropped ``None`` from lists as well as from
mappings: ``[1, None, 2]`` became ``[1, 2]``, so every later position shifted
and a caller reading ``xs[1]`` got a different value. And the ``text`` format
interpolated values straight into ``key: value`` lines, so a value containing a
newline produced additional lines indistinguishable from real keys.

See Also
--------
scikitplot._cli.output.emit
"""

import io
import json

import pytest

from .. import output
from ..context import Context


def _render(data, fmt):
    ctx = Context(stdout=io.StringIO(), stderr=io.StringIO(), fmt=fmt)
    output.emit(ctx, data)
    return ctx.stdout.getvalue()


# -- S-28: a lossy conversion is refused or reported ----------------------


def test_a_none_inside_a_list_is_refused():
    """Dropping an element shifts every later position, so it is not permitted."""
    with pytest.raises(ValueError) as excinfo:
        output._toml_safe({"xs": [1, None, 2]})
    assert "xs" in str(excinfo.value)


def test_the_refusal_names_the_position():
    """A caller must be able to find the offending element."""
    with pytest.raises(ValueError) as excinfo:
        output._toml_safe({"outer": {"xs": ["a", None]}})
    message = str(excinfo.value)
    assert "outer.xs" in message and "1" in message


def test_dropping_a_mapping_key_is_still_allowed():
    """An absent key denotes unset; that is documented and position-free."""
    assert output._toml_safe({"a": 1, "b": None}) == {"a": 1}


def test_lists_without_none_are_unchanged():
    """The ordinary path is untouched."""
    assert output._toml_safe({"xs": [1, 2, 3]})["xs"] == [1, 2, 3]


def test_nested_lists_are_checked_too():
    """The rule holds at every depth, not only the top level."""
    with pytest.raises(ValueError):
        output._toml_safe({"xs": [[1, None]]})


# -- S-30: the text format cannot be forged -------------------------------


def test_a_newline_in_a_value_cannot_forge_a_key():
    """The defect L05 recorded: a value producing an apparent second field."""
    text = _render({"note": "first line\nkey: injected", "next": 1}, "text")
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 2
    assert not any(line.startswith("key:") for line in lines)


def test_the_escaped_value_is_still_readable():
    """Escaping must not make the value unrecognisable."""
    text = _render({"note": "first\nsecond"}, "text")
    assert "first" in text and "second" in text


def test_a_tab_or_carriage_return_is_escaped_too():
    """Any character that breaks the line contract is escaped, not just \\n."""
    text = _render({"note": "a\tb\rc"}, "text")
    assert len(text.splitlines()) == 1


def test_nested_data_is_not_rendered_as_a_python_repr():
    """``{'available': True}`` is not the documented key: value shape."""
    text = _render({"capabilities": {"click": {"available": True}}}, "text")
    assert "'available': True" not in text


def test_nested_data_is_reachable_in_the_text_output():
    """Flattening keeps the information; it does not drop it."""
    text = _render({"capabilities": {"click": {"available": True}}}, "text")
    assert "capabilities.click.available" in text
    assert "True" in text


def test_ordinary_values_are_unchanged():
    """A plain scalar renders exactly as before."""
    assert _render({"status": "ok"}, "text").strip() == "status: ok"


def test_structured_formats_are_unaffected():
    """Escaping belongs to the human format only."""
    data = {"note": "first\nsecond"}
    assert json.loads(_render(data, "json")) == data
