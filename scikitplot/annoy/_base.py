"""
High-level public Index class.

This class is intentionally small and composed from mixins to keep the
API modular and extensible (future Series/Frame-style layers).

.. seealso::
    * :py:obj:`~scikitplot.annoy.Index.from_low_level`
    * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
"""

from __future__ import annotations

# import uuid  # f"annoy-{uuid.uuid4().hex}.annoy"
from ..cexternals._annoy import Annoy
from ._mixins import ManifestMixin, ObjectIOMixin, PickleMixin, VectorOpsMixin


class Index(VectorOpsMixin, ObjectIOMixin, ManifestMixin, PickleMixin):
    """
    High-level Pythonic Annoy wrapper with picklable (or pickle-able).

    Minimal modify spotify/annoy low-level C-API to extend Python API.

    .. seealso::
        * :py:obj:`~scikitplot.annoy.Index.from_low_level`
        * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
    """

    @classmethod
    def from_low_level(
        cls,
        obj: Annoy,
        prefault: bool = False,
    ) -> "Index":  # noqa: UP037
        """
        Convert a low-level Annoy instance into a high-level Index.

        Instance by round-tripping through serialize/deserialize.
        """
        inst = cls(int(obj.f), str(obj.metric))
        byte = obj.serialize()

        ok = inst.deserialize(byte, prefault=prefault)
        if not ok:
            raise RuntimeError("deserialize failed during conversion")

        # Preserve path metadata if exposed on the low-level object
        path = getattr(obj, "on_disk_path", None)
        if path:
            inst.on_disk_path = str(path)

        return inst
