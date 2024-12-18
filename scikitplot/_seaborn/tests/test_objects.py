from .._core.plot import Plot
from .._core.moves import Move
from .._core.scales import Scale
from .._marks.base import Mark
from .._stats.base import Stat

from .. import objects as seaborn_objects


def test_objects_namespace():

    for name in dir(seaborn_objects):
        if not name.startswith("__"):
            obj = getattr(seaborn_objects, name)
            assert issubclass(obj, (Plot, Mark, Stat, Move, Scale))
