# scikitplot/impute/tests/test__privacy.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for :mod:`scikitplot.impute._privacy`.

Covers
------
PrivateIndexMixin
  * _set_internal_index / _get_internal_index — round-trip, overwrite, not-set
  * _set_index — always raises with informative message
  * _get_index — public (fitted/not-fitted), private, external, arbitrary mode

OutsourcedIndexMixin._store_index
  * public   — in-memory, no file written, optional path as metadata
  * private  — in-memory, public access blocked, runtime access works
  * external — file written, string/pathlib/PathNamer/auto-generated paths,
               on_disk_path skip-save branch, on_disk_path differs branch
  * invalid mode — ValueError with mode name in message

OutsourcedIndexMixin._get_index_for_runtime
  * public/private — returns in-memory index
  * external loader — correct path, data round-trips
  * external no index_path_ — RuntimeError (transform before fit)
  * external file gone — RuntimeError with path in message
  * external no loader — ValueError
  * unknown mode — ValueError

OutsourcedIndexMixin.delete_external_index
  * removes file, no-op when absent/None, second delete raises OSError

Regressions
  * get_index public not-fitted raises AttributeError, never TypeError
    (was broken when check_is_fitted required BaseEstimator inheritance)
"""

import os
import tempfile
import unittest
from pathlib import Path

from scikitplot.impute._privacy import OutsourcedIndexMixin, PrivateIndexMixin
from scikitplot.utils._path import PathNamer


# ---------------------------------------------------------------------------
# Minimal concrete estimators
# ---------------------------------------------------------------------------

class _PrivateEstimator(PrivateIndexMixin):
    """Only inherits PrivateIndexMixin — no sklearn BaseEstimator."""
    def __init__(self, index_access="private"):
        self.index_access = index_access


class _OutsourcedEstimator(OutsourcedIndexMixin):
    """Supports all three index_access modes."""
    def __init__(self, index_access="external"):
        self.index_access = index_access


# ---------------------------------------------------------------------------
# Minimal saveable index stub
# ---------------------------------------------------------------------------

class _FakeIndex:
    """Minimal index with save/load and optional on_disk_path."""
    def __init__(self, tag="ok", on_disk_path=None):
        self.tag = tag
        self.on_disk_path = on_disk_path

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as fh:
            pickle.dump({"tag": self.tag}, fh)

    @classmethod
    def load_from(cls, path: str) -> "_FakeIndex":
        import pickle
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return cls(tag=data["tag"])


# ===========================================================================
# PrivateIndexMixin — internal storage
# ===========================================================================

class TestPrivateIndexMixinInternalStorage(unittest.TestCase):

    def test_set_then_get_returns_same_object(self):
        est = _PrivateEstimator()
        idx = _FakeIndex("alpha")
        est._set_internal_index(idx)
        self.assertIs(est._get_internal_index(), idx)

    def test_get_before_set_raises_attribute_error(self):
        est = _PrivateEstimator()
        with self.assertRaises(AttributeError):
            est._get_internal_index()

    def test_get_before_set_message_mentions_fit(self):
        est = _PrivateEstimator()
        with self.assertRaises(AttributeError) as ctx:
            est._get_internal_index()
        self.assertIn("fit", str(ctx.exception).lower())

    def test_set_index_via_setter_raises(self):
        est = _PrivateEstimator()
        with self.assertRaises(AttributeError):
            est._set_index("train_index_", _FakeIndex())

    def test_set_index_error_contains_attribute_name(self):
        est = _PrivateEstimator()
        with self.assertRaises(AttributeError) as ctx:
            est._set_index("train_index_", _FakeIndex())
        self.assertIn("train_index_", str(ctx.exception))

    def test_overwrite_replaces_internal_index(self):
        est = _PrivateEstimator()
        est._set_internal_index(_FakeIndex("v1"))
        est._set_internal_index(_FakeIndex("v2"))
        self.assertEqual(est._get_internal_index().tag, "v2")

    def test_store_does_not_leak_to_plain_attribute(self):
        est = _PrivateEstimator()
        est._set_internal_index(_FakeIndex())
        self.assertFalse(hasattr(est, "_annoy_index"))
        self.assertFalse(hasattr(est, "_index"))

    def test_two_instances_independent_stores(self):
        a, b = _PrivateEstimator(), _PrivateEstimator()
        a._set_internal_index(_FakeIndex("a"))
        b._set_internal_index(_FakeIndex("b"))
        self.assertEqual(a._get_internal_index().tag, "a")
        self.assertEqual(b._get_internal_index().tag, "b")


# ===========================================================================
# PrivateIndexMixin — _get_index policy
# ===========================================================================

class TestPrivateIndexMixinGetIndex(unittest.TestCase):

    def test_public_mode_returns_index(self):
        est = _PrivateEstimator(index_access="public")
        idx = _FakeIndex("pub")
        est._set_internal_index(idx)
        self.assertIs(est._get_index("train_index_"), idx)

    def test_public_mode_not_fitted_raises_attribute_error(self):
        est = _PrivateEstimator(index_access="public")
        with self.assertRaises(AttributeError):
            est._get_index("train_index_")

    def test_public_mode_not_fitted_never_raises_type_error(self):
        """Regression: check_is_fitted raised TypeError for non-BaseEstimator."""
        est = _PrivateEstimator(index_access="public")
        raised = None
        try:
            est._get_index("train_index_")
        except (AttributeError, TypeError) as exc:
            raised = exc
        self.assertIsInstance(raised, AttributeError,
                              f"Expected AttributeError, got {type(raised).__name__}")

    def test_private_mode_raises_attribute_error(self):
        est = _PrivateEstimator(index_access="private")
        est._set_internal_index(_FakeIndex())
        with self.assertRaises(AttributeError):
            est._get_index("train_index_")

    def test_private_mode_error_mentions_public(self):
        est = _PrivateEstimator(index_access="private")
        est._set_internal_index(_FakeIndex())
        with self.assertRaises(AttributeError) as ctx:
            est._get_index("train_index_")
        self.assertIn("public", str(ctx.exception))

    def test_external_mode_raises_attribute_error(self):
        est = _PrivateEstimator(index_access="external")
        est._set_internal_index(_FakeIndex())
        with self.assertRaises(AttributeError):
            est._get_index("train_index_")

    def test_arbitrary_non_public_mode_raises(self):
        est = _PrivateEstimator(index_access="restricted")
        est._set_internal_index(_FakeIndex())
        with self.assertRaises(AttributeError):
            est._get_index("train_index_")

    def test_custom_public_name_appears_in_error(self):
        est = _PrivateEstimator(index_access="private")
        est._set_internal_index(_FakeIndex())
        with self.assertRaises(AttributeError) as ctx:
            est._get_index("my_special_index_")
        self.assertIn("my_special_index_", str(ctx.exception))


# ===========================================================================
# OutsourcedIndexMixin._store_index — public mode
# ===========================================================================

class TestOutsourcedStorePublic(unittest.TestCase):

    def _est(self):
        return _OutsourcedEstimator(index_access="public")

    def test_store_sets_internal_index(self):
        est = self._est()
        idx = _FakeIndex("pub")
        est._store_index(idx, public_name="train_index_", index_path=None)
        self.assertIs(est._get_internal_index(), idx)

    def test_store_records_index_path_none(self):
        est = self._est()
        est._store_index(_FakeIndex(), index_path=None)
        self.assertIsNone(est.index_path_)

    def test_store_explicit_path_recorded_as_metadata(self):
        est = self._est()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "pub.annoy")
            est._store_index(_FakeIndex(), index_path=p)
            self.assertEqual(est.index_path_, p)

    def test_store_created_at_iso8601(self):
        est = self._est()
        est._store_index(_FakeIndex(), index_path=None)
        self.assertIn("T", est.index_created_at_)

    def test_store_does_not_write_file(self):
        est = self._est()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "should_not_exist.annoy")
            est._store_index(_FakeIndex(), index_path=p)
            self.assertFalse(os.path.exists(p))

    def test_get_index_succeeds_after_store(self):
        est = self._est()
        idx = _FakeIndex("ok")
        est._store_index(idx)
        self.assertIs(est._get_index("train_index_"), idx)


# ===========================================================================
# OutsourcedIndexMixin._store_index — private mode
# ===========================================================================

class TestOutsourcedStorePrivate(unittest.TestCase):

    def _est(self):
        return _OutsourcedEstimator(index_access="private")

    def test_store_sets_internal_index(self):
        est = self._est()
        idx = _FakeIndex("priv")
        est._store_index(idx, index_path=None)
        self.assertIs(est._get_internal_index(), idx)

    def test_get_index_raises(self):
        est = self._est()
        est._store_index(_FakeIndex(), index_path=None)
        with self.assertRaises(AttributeError):
            est._get_index("train_index_")

    def test_get_index_for_runtime_returns_index(self):
        est = self._est()
        idx = _FakeIndex("runtime")
        est._store_index(idx)
        self.assertIs(est._get_index_for_runtime("train_index_"), idx)


# ===========================================================================
# OutsourcedIndexMixin._store_index — external mode
# ===========================================================================

class TestOutsourcedStoreExternal(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.td = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def _p(self, name="idx.annoy"):
        return os.path.join(self.td, name)

    def _store(self, path, tag="ext", **kw):
        est = _OutsourcedEstimator(index_access="external")
        idx = _FakeIndex(tag, **kw)
        est._store_index(idx, public_name="train_index_", index_path=path)
        return est, idx

    # --- file written ---

    def test_writes_file(self):
        est, _ = self._store(self._p())
        self.assertTrue(os.path.exists(est.index_path_))

    def test_records_index_path(self):
        path = self._p("my.annoy")
        est, _ = self._store(path)
        self.assertEqual(est.index_path_, str(path))

    def test_records_created_at(self):
        est, _ = self._store(self._p("t.annoy"))
        self.assertIn("T", est.index_created_at_)

    def test_does_not_set_internal_index(self):
        est, _ = self._store(self._p())
        with self.assertRaises(AttributeError):
            est._get_internal_index()

    # --- path type variants ---

    def test_auto_path_when_index_path_none(self):
        est = _OutsourcedEstimator(index_access="external")
        est._store_index(_FakeIndex("auto"), index_path=None)
        self.assertIsNotNone(est.index_path_)
        self.assertTrue(os.path.exists(est.index_path_))
        try:
            os.remove(est.index_path_)
        except OSError:
            pass

    def test_pathlib_creates_intermediate_dirs(self):
        path = Path(self.td) / "sub" / "deep" / "idx.annoy"
        est = _OutsourcedEstimator(index_access="external")
        est._store_index(_FakeIndex("lib"), index_path=path)
        self.assertTrue(os.path.exists(est.index_path_))

    def test_pathnamer_input(self):
        namer = PathNamer(prefix="tst_", ext=".annoy", directory=self.td)
        est = _OutsourcedEstimator(index_access="external")
        est._store_index(_FakeIndex("named"), index_path=namer)
        self.assertIsNotNone(est.index_path_)
        self.assertTrue(os.path.exists(est.index_path_))

    # --- on_disk_path skip-save branch ---

    def test_skip_save_when_on_disk_path_matches_target(self):
        """File already at target path (on-disk build) → save() not called."""
        path = self._p("same.annoy")
        idx = _FakeIndex("skip")
        idx.save(path)
        mtime_before = os.path.getmtime(path)
        idx.on_disk_path = path

        est = _OutsourcedEstimator(index_access="external")
        est._store_index(idx, index_path=path)

        self.assertEqual(os.path.getmtime(path), mtime_before)
        self.assertEqual(est.index_path_, str(path))

    def test_save_when_on_disk_path_differs(self):
        source = self._p("source.annoy")
        target = self._p("target.annoy")
        idx = _FakeIndex("save_me")
        idx.save(source)
        idx.on_disk_path = source

        est = _OutsourcedEstimator(index_access="external")
        est._store_index(idx, index_path=target)

        self.assertTrue(os.path.exists(target))
        self.assertEqual(est.index_path_, str(target))

    # --- invalid mode ---

    def test_invalid_mode_raises_value_error(self):
        est = _OutsourcedEstimator(index_access="bad_mode")
        with self.assertRaises(ValueError):
            est._store_index(_FakeIndex(), index_path=None)

    def test_invalid_mode_error_mentions_mode(self):
        est = _OutsourcedEstimator(index_access="bad_mode")
        with self.assertRaises(ValueError) as ctx:
            est._store_index(_FakeIndex(), index_path=None)
        self.assertIn("bad_mode", str(ctx.exception))


# ===========================================================================
# OutsourcedIndexMixin._get_index_for_runtime
# ===========================================================================

class TestOutsourcedGetIndexForRuntime(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.td = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def _p(self, name="rt.annoy"):
        return os.path.join(self.td, name)

    def test_public_returns_in_memory_index(self):
        est = _OutsourcedEstimator(index_access="public")
        idx = _FakeIndex("mem")
        est._set_internal_index(idx)
        self.assertIs(est._get_index_for_runtime("train_index_"), idx)

    def test_private_returns_in_memory_index(self):
        est = _OutsourcedEstimator(index_access="private")
        idx = _FakeIndex("priv")
        est._set_internal_index(idx)
        self.assertIs(est._get_index_for_runtime("train_index_"), idx)

    def test_external_loader_called_with_correct_path(self):
        path = self._p()
        est = _OutsourcedEstimator(index_access="external")
        est._store_index(_FakeIndex("disk"), index_path=path)
        calls = []
        def loader(p):
            calls.append(p)
            return _FakeIndex.load_from(p)
        result = est._get_index_for_runtime("train_index_", loader=loader)
        self.assertEqual(calls, [str(path)])
        self.assertEqual(result.tag, "disk")

    def test_external_data_round_trips(self):
        path = self._p("rtrip.annoy")
        est = _OutsourcedEstimator(index_access="external")
        est._store_index(_FakeIndex("roundtrip_tag"), index_path=path)
        result = est._get_index_for_runtime(loader=_FakeIndex.load_from)
        self.assertEqual(result.tag, "roundtrip_tag")

    def test_external_no_index_path_attr_raises_runtime_error(self):
        est = _OutsourcedEstimator(index_access="external")
        with self.assertRaises(RuntimeError):
            est._get_index_for_runtime("train_index_", loader=lambda p: None)

    def test_external_file_deleted_raises_runtime_error(self):
        path = self._p("gone.annoy")
        est = _OutsourcedEstimator(index_access="external")
        est._store_index(_FakeIndex(), index_path=path)
        os.remove(est.index_path_)
        with self.assertRaises(RuntimeError):
            est._get_index_for_runtime("train_index_", loader=lambda p: None)

    def test_external_missing_file_error_mentions_path(self):
        path = self._p("msg.annoy")
        est = _OutsourcedEstimator(index_access="external")
        est._store_index(_FakeIndex(), index_path=path)
        os.remove(est.index_path_)
        with self.assertRaises(RuntimeError) as ctx:
            est._get_index_for_runtime("train_index_", loader=lambda p: None)
        self.assertIn("msg.annoy", str(ctx.exception))

    def test_external_no_loader_raises_value_error(self):
        path = self._p("nol.annoy")
        est = _OutsourcedEstimator(index_access="external")
        est._store_index(_FakeIndex(), index_path=path)
        with self.assertRaises(ValueError):
            est._get_index_for_runtime("train_index_", loader=None)

    def test_unknown_mode_raises_value_error(self):
        est = _OutsourcedEstimator(index_access="unknown_mode")
        with self.assertRaises(ValueError):
            est._get_index_for_runtime("train_index_", loader=lambda p: None)


# ===========================================================================
# OutsourcedIndexMixin.delete_external_index
# ===========================================================================

class TestDeleteExternalIndex(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.td = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_delete_removes_file(self):
        path = os.path.join(self.td, "del.annoy")
        est = _OutsourcedEstimator(index_access="external")
        est._store_index(_FakeIndex(), index_path=path)
        self.assertTrue(os.path.exists(path))
        est.delete_external_index()
        self.assertFalse(os.path.exists(path))

    def test_delete_no_index_path_attr_does_nothing(self):
        est = _OutsourcedEstimator(index_access="public")
        est.delete_external_index()  # must not raise

    def test_delete_index_path_none_does_nothing(self):
        est = _OutsourcedEstimator(index_access="public")
        est._store_index(_FakeIndex(), index_path=None)
        est.delete_external_index()  # must not raise

    def test_delete_twice_raises_os_error(self):
        path = os.path.join(self.td, "twice.annoy")
        est = _OutsourcedEstimator(index_access="external")
        est._store_index(_FakeIndex(), index_path=path)
        est.delete_external_index()
        with self.assertRaises(OSError):
            est.delete_external_index()


# ===========================================================================
# Regressions
# ===========================================================================

class TestPrivacyRegressions(unittest.TestCase):

    def test_get_index_public_not_fitted_raises_attribute_error_not_type_error(self):
        """check_is_fitted previously raised TypeError for non-BaseEstimator classes."""
        est = _PrivateEstimator(index_access="public")
        raised = None
        try:
            est._get_index("train_index_")
        except (AttributeError, TypeError) as exc:
            raised = exc
        self.assertIsNotNone(raised)
        self.assertIsInstance(raised, AttributeError,
                              f"Expected AttributeError, got {type(raised).__name__}: {raised}")

    def test_outsourced_get_index_public_not_fitted_raises_attribute_error(self):
        est = _OutsourcedEstimator(index_access="public")
        raised = None
        try:
            est._get_index("train_index_")
        except (AttributeError, TypeError) as exc:
            raised = exc
        self.assertIsInstance(raised, AttributeError,
                              f"Expected AttributeError, got {type(raised).__name__}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
