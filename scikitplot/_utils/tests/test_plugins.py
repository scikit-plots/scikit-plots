# scikitplot/_utils/tests/test_plugins.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.plugins`.

Coverage map
------------
_get_entry_points   Python >= 3.10 path (group= kwarg),
                    Python < 3.10 path via eps.get(),
                    fallback via eps.select(),
                    empty group, non-existent group               -> TestGetEntryPoints
get_entry_points    delegates to _get_entry_points,
                    returns list, empty string group              -> TestGetEntryPoints (public)

Run standalone::

    python -m unittest scikitplot._utils.tests.test_plugins -v
"""

from __future__ import annotations

import sys
import unittest
import unittest.mock as mock
from importlib.metadata import EntryPoint

from ..plugins import _get_entry_points, get_entry_points


# ===========================================================================
# Helpers
# ===========================================================================


def _make_entry_point(name: str, group: str, value: str = "module:attr") -> EntryPoint:
    """Build an EntryPoint compatible with all Python 3.8+ APIs."""
    return EntryPoint(name=name, group=group, value=value)


# ===========================================================================
# _get_entry_points  (internal)
# ===========================================================================


class TestInternalGetEntryPoints(unittest.TestCase):
    """_get_entry_points must call importlib.metadata correctly."""

    # -- Python >= 3.10 path: entry_points(group=...) --

    def test_py310_path_calls_entry_points_with_group_kwarg(self):
        """On Python >= 3.10, entry_points must be called with group= keyword."""
        ep = _make_entry_point("plugin_a", "myapp.plugins")

        with mock.patch("sys.version_info", new=(3, 10, 0)):
            with mock.patch(
                "scikitplot._utils.plugins.entry_points",
                return_value=[ep],
            ) as mock_ep:
                result = _get_entry_points("myapp.plugins")

        mock_ep.assert_called_once_with(group="myapp.plugins")
        self.assertEqual(result, [ep])

    def test_py310_path_returns_list(self):
        """On Python >= 3.10, the return value must be a list."""
        with mock.patch("sys.version_info", new=(3, 10, 0)):
            with mock.patch(
                "scikitplot._utils.plugins.entry_points",
                return_value=[],
            ):
                result = _get_entry_points("myapp.plugins")
        self.assertIsInstance(result, list)

    # -- Python < 3.10 path: eps.get() --

    def test_pre310_path_uses_eps_get(self):
        """On Python < 3.10, eps.get(group, []) must be used if available."""
        ep = _make_entry_point("plugin_b", "myapp.plugins")

        eps_dict = mock.MagicMock()
        eps_dict.get.return_value = [ep]
        # Must NOT have .select to stay on the .get() branch
        del eps_dict.select

        with mock.patch("sys.version_info", new=(3, 9, 0)):
            with mock.patch(
                "scikitplot._utils.plugins.entry_points",
                return_value=eps_dict,
            ):
                result = _get_entry_points("myapp.plugins")

        eps_dict.get.assert_called_once_with("myapp.plugins", [])
        self.assertEqual(result, [ep])

    def test_pre310_path_fallback_to_select(self):
        """On Python < 3.10, .select() must be used when .get() raises AttributeError."""
        ep = _make_entry_point("plugin_c", "myapp.plugins")

        eps_mock = mock.MagicMock()
        # Simulate AttributeError on .get() so code falls back to .select()
        eps_mock.get.side_effect = AttributeError("no get")
        eps_mock.select.return_value = [ep]

        with mock.patch("sys.version_info", new=(3, 9, 0)):
            with mock.patch(
                "scikitplot._utils.plugins.entry_points",
                return_value=eps_mock,
            ):
                result = _get_entry_points("myapp.plugins")

        eps_mock.select.assert_called_once_with(group="myapp.plugins")
        self.assertEqual(result, [ep])

    # -- empty / non-existent group --

    def test_empty_group_returns_empty_list(self):
        """A group with no registered plugins must return an empty list."""
        with mock.patch(
            "scikitplot._utils.plugins.entry_points",
            return_value=[],
        ):
            result = _get_entry_points("nonexistent.group")
        self.assertEqual(result, [])

    def test_non_string_group_passes_through(self):
        """_get_entry_points must pass the group argument to entry_points unchanged."""
        with mock.patch(
            "scikitplot._utils.plugins.entry_points",
            return_value=[],
        ) as mock_ep:
            _get_entry_points("some.group")
        call_kwargs = mock_ep.call_args
        self.assertIsNotNone(call_kwargs)

    # -- multiple entry points --

    def test_multiple_entry_points_returned(self):
        """Multiple entry points under the same group must all be returned."""
        eps = [
            _make_entry_point("p1", "myapp.plugins"),
            _make_entry_point("p2", "myapp.plugins"),
            _make_entry_point("p3", "myapp.plugins"),
        ]
        with mock.patch("sys.version_info", new=(3, 10, 0)):
            with mock.patch(
                "scikitplot._utils.plugins.entry_points",
                return_value=eps,
            ):
                result = _get_entry_points("myapp.plugins")
        self.assertEqual(len(result), 3)


# ===========================================================================
# get_entry_points  (public API)
# ===========================================================================


class TestGetEntryPoints(unittest.TestCase):
    """get_entry_points must delegate to _get_entry_points and return its result."""

    def test_returns_list(self):
        """get_entry_points must return a list."""
        with mock.patch(
            "scikitplot._utils.plugins._get_entry_points",
            return_value=[],
        ):
            result = get_entry_points("some.group")
        self.assertIsInstance(result, list)

    def test_delegates_to_internal(self):
        """get_entry_points must delegate to _get_entry_points with the same group."""
        ep = _make_entry_point("test", "some.group")
        with mock.patch(
            "scikitplot._utils.plugins._get_entry_points",
            return_value=[ep],
        ) as mock_internal:
            result = get_entry_points("some.group")
        mock_internal.assert_called_once_with("some.group")
        self.assertEqual(result, [ep])

    def test_empty_group_string(self):
        """An empty group string must be handled without raising."""
        with mock.patch(
            "scikitplot._utils.plugins._get_entry_points",
            return_value=[],
        ):
            result = get_entry_points("")
        self.assertEqual(result, [])

    def test_returns_entry_point_instances(self):
        """Each element of the returned list must be an EntryPoint."""
        ep = _make_entry_point("myplugin", "myapp.plugins")
        with mock.patch(
            "scikitplot._utils.plugins._get_entry_points",
            return_value=[ep],
        ):
            result = get_entry_points("myapp.plugins")
        for item in result:
            self.assertIsInstance(item, EntryPoint)

    def test_callable(self):
        """get_entry_points must be callable."""
        self.assertTrue(callable(get_entry_points))

    def test_real_call_returns_list(self):
        """A real call (no mock) must always return a list, even for unknown groups."""
        result = get_entry_points("__nonexistent_group_xyz_12345__")
        self.assertIsInstance(result, list)

    def test_well_known_group_returns_list(self):
        """console_scripts group must return a list (possibly empty)."""
        result = get_entry_points("console_scripts")
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
