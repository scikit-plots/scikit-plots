# ruff: noqa: F401
# pylint: disable=unused-import

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import datetime as _dt
import itertools
import os
from pathlib import Path

from sklearn.utils.validation import check_is_fitted

from ..utils._path import PathNamer, make_path

_counter = itertools.count()


class PrivateIndexMixin:
    """
    Mixin to control public access to a fitted index object.

    This mixin is designed to prevent **accidental** access or misuse of
    a fitted ANN index (for example an Annoy, Voyager or HNSW index) through
    the public estimator API. It reduces visibility but it is *not* a
    hard security boundary: any code running in the same Python process
    can still inspect objects via introspection.

    The actual index is stored in a name-mangled internal attribute on
    this mixin, **not** on the estimator itself (for example there is
    no ``estimator._annoy_index`` attribute). This makes it harder to
    depend on estimator-specific attribute names in user code.

    Estimators using this mixin are expected to:

    - Define an ``index_access`` init parameter with values
      ``'public'`` or ``'private'``.
    - Call :meth:`_set_internal_index` during :meth:`fit` once the
      index has been built.
    - Call :meth:`_get_internal_index` inside :meth:`transform` (or
      similar runtime methods) to use the fitted index.
    - Expose a public read-only property (for example
      :attr:`train_index_`) that internally delegates to
      :meth:`_get_index` so that the ``index_access`` policy is
      consistently enforced.
    """

    # ---------------- internal storage helpers ---------------- #
    def _set_internal_index(self, index):
        """
        Store the fitted index in a name-mangled attribute on this mixin.

        This avoids creating estimator-specific attributes like
        `_annoy_index` or `_voyager_index`.
        """
        # obj._ClassName__attr
        # Name-mangled attribute: `_PrivateIndexMixin__index_store`
        object.__setattr__(self, "_PrivateIndexMixin__index_store", index)

    def _get_internal_index(self):
        """
        Return the internal index instance or raise AttributeError
        if it has not been set (i.e. `fit` has not been called).
        """  # noqa: D205
        try:
            return object.__getattribute__(self, "_PrivateIndexMixin__index_store")
        except AttributeError as exc:  # not fitted
            raise AttributeError(
                "This estimator does not have a fitted index yet. "
                "Call `fit` before using it."
            ) from exc

    # ---------------- public-policy helper ---------------- #
    def _set_index(
        self,
        public_name: str = "train_index_",
        value=None,
    ):
        """
        Disallow external code from overwriting the internal index.

        The index should only be created during `fit`.
        """
        raise AttributeError(
            f"Direct assignment to '{public_name}' is not supported. "
            "The index is built internally during `fit`."
        )

    def _get_index(
        self,
        public_name: str = "train_index_",
    ):
        """
        Return the internal index or raise if `index_access` is not ``'public'``.

        Parameters
        ----------
        public_name : str, default="train_index_"
            Public attribute name for error messages
            (e.g. ``'train_index_'``).
        """
        mode = getattr(self, "index_access", "private")
        if mode == "public":
            # Use a standard sklearn-style fitted check, but on a
            # mixin-specific internal attribute.
            check_is_fitted(self, "_PrivateIndexMixin__index_store")
            return self._get_internal_index()

        raise AttributeError(
            f"Access to '{public_name}' is disabled because this estimator was "
            f"instantiated with `index_access='{mode}'`. Re-create it with "
            "`index_access='public'` if you need direct access to the "
            "underlying index object."
        )


class OutsourcedIndexMixin(PrivateIndexMixin):
    """
    Extension of :class:`PrivateIndexMixin` that adds an ``'external'``
    mode for :attr:`index_access`.

    When ``index_access='external'``, the fitted ANN index is stored
    on disk and the estimator keeps only:

    * the file path (:attr:`index_path_`), and
    * lightweight metadata (for example :attr:`index_created_at_`).

    This is useful when you:

    * want to pickle or ship the estimator without embedding the index
      in the pickle payload,
    * store the ANN index on a separate, access-controlled volume
      (for example a shared mount or object storage),
    * keep the Python object small while the ANN index itself remains
      large.

    Estimators using this mixin should:

    - Continue defining an ``index_access`` init parameter with values
      ``'public'``, ``'private'`` or ``'external'``.
    - Call :meth:`_store_index` in :meth:`fit` once the index has been
      built.
    - Call :meth:`_get_index_for_runtime` inside :meth:`transform` (or
      similar runtime methods) to obtain an index object suitable for
      querying.

    Public API access still goes through
    :meth:`PrivateIndexMixin._get_index`, so an attribute such as
    :attr:`train_index_` remains available only in
    ``index_access='public'`` mode.
    """  # noqa: D205

    def _store_index(
        self,
        index,
        *,
        public_name: str = "train_index_",
        index_path: str | os.PathLike | PathNamer | None = None,
    ):
        """
        Store the fitted index according to ``index_access``.

        * ``'public'`` / ``'private'``: keep the index in memory via
          :meth:`_set_internal_index`.
        * ``'external'``: persist the index to disk using ``index.save()``
          and keep only the file path and metadata on the estimator.

        Parameters
        ----------
        index : object
            The fitted ANN index. Must implement ``save(path)`` when
            used with ``index_access='external'``.

        public_name : str, default="train_index_"
            Public attribute name used only for error messages
            (for example ``'train_index_'``).

        index_path : str or None, optional
            Target file path when ``index_access='external'``. If this
            is ``None``, an OS-friendly unique file name from
            :func:`~scikitplot.utils._path.make_path` will be generated,
            otherwise in the current working directory.
        """
        # Default to external if not set by the estimator
        mode = getattr(self, "index_access", "external")
        on_disk_path = getattr(index, "on_disk_path", None)

        if mode not in {"public", "private", "external"}:
            raise ValueError(
                f"{self.__class__.__name__}: invalid `index_access={mode!r}`. "
                "Expected one of {'public', 'private', 'external'}."
            )

        # Metadata: ISO 8601 creation time (for logs / audit)
        self.index_created_at_ = _dt.datetime.now(_dt.timezone.utc).isoformat()

        # -------------------------------------------------------------
        # public / private  => normal in-memory behaviour
        # -------------------------------------------------------------
        if mode in {"public", "private"}:
            # Standard behaviour: keep the index in memory.
            self._set_internal_index(index)
            # If caller passed a path we still record it as metadata,
            # but we do not require it.
            self.index_path_ = index_path  # str(index_path)
            return

        # -------------------------------------------------------------
        # external  => index kept on disk, only path stored on estimator
        # -------------------------------------------------------------
        if mode == "external":
            # OS-friendly file name if path not provided
            if index_path is None:
                index_path = make_path(
                    prefix="OutsourcedIndexMixin",
                )
            elif isinstance(index_path, PathNamer):
                index_path = index_path.make_path()
            else:
                index_path = Path(index_path).resolve()

            index_path.parent.mkdir(parents=True, exist_ok=True)

            # 3) Persist index and remember the path
            should_save = True
            if on_disk_path is not None:
                try:
                    should_save = Path(on_disk_path).resolve() != index_path
                except Exception:
                    should_save = str(on_disk_path) != str(index_path)

            if should_save:
                # Persist the index to disk and keep only metadata on the estimator.
                index.save(str(index_path))

            # Estimator metadata
            self.index_path_ = str(index_path)

            # Do NOT keep the index in the internal store. For runtime use,
            # call `_get_index_for_runtime` with an appropriate loader.
            return

        raise ValueError(
            f"{self.__class__.__name__}: unsupported `index_access={mode!r}`. "
            "Expected one of {'public', 'private', 'external'}."
        )

    def _get_index_for_runtime(
        self,
        public_name: str = "train_index_",
        loader: callable | None = None,
    ):
        """
        Return an index object usable at runtime (e.g. in ``transform``).

        Behaviour
        ---------
        * ``index_access in {'public', 'private'}``:
          return the in-memory index stored via :meth:`_set_internal_index`.
          In this case the ``loader`` argument is ignored.
        * ``index_access == 'external'``:
          load the index from :attr:`index_path_` using the provided
          ``loader(path) -> index`` callable.

        Parameters
        ----------
        public_name : str, default="train_index_"
            Public attribute name used only for error messages.

        loader : Callable or None
            A callable of the form ``loader(path) -> index`` used to
            reconstruct the index from disk when
            ``index_access='external'``. It is ignored for
            ``'public'`` and ``'private'`` modes, and required for
            ``'external'``.

        Returns
        -------
        index : object
            An ANN index object suitable for querying.
        """
        mode = getattr(self, "index_access", "external")

        if mode in {"public", "private"}:
            return self._get_internal_index()

        if mode == "external":
            if not hasattr(self, "index_path_"):
                raise RuntimeError(
                    f"{self.__class__.__name__}: `index_access='external'` "
                    "was requested but no `index_path_` metadata was found. "
                    "Did you call `fit` before `transform`?"
                )
            path = self.index_path_
            if not os.path.exists(path):
                raise RuntimeError(
                    f"{self.__class__.__name__}: external index file not found at "
                    f"{path!r}. The file may have been moved or deleted. "
                    "Please re-fit the imputer or provide a valid `index_store_path`."
                )
            if loader is None:
                raise ValueError(
                    f"{self.__class__.__name__}: `loader` must be provided "
                    "when using `index_access='external'`."
                )
            return loader(path)

        raise ValueError(
            f"{self.__class__.__name__}: unsupported `index_access={mode!r}` "
            "in `_get_index_for_runtime`."
        )

    def delete_external_index(self):
        """
        Delete the external index file referenced by :attr:`index_path_`.

        This helper removes the file on disk if :attr:`index_path_` is
        set. It does **not** modify :attr:`index_path_` itself, so
        subsequent calls that rely on the file (for example
        :meth:`_get_index_for_runtime` in ``index_access='external'``
        mode) will fail until the estimator is re-fitted.

        Any :class:`OSError` raised by :func:`os.remove` will propagate
        to the caller.
        """
        if getattr(self, "index_path_", None):
            os.remove(self.index_path_)
