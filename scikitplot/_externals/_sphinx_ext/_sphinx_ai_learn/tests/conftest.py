"""Keep extension tests independent of the compiled scikitplot package."""
from pathlib import Path
import sys

_HERE = Path(__file__).resolve()

# ``_sphinx_ext`` as a top-level package, whichever checkout this is.
sys.path.insert(0, str(_HERE.parents[3]))
# ``_learn_site``, the tests' own helper. Added here so it is importable under
# every pytest import mode, not only the one that happens to put a test file's
# directory on the path.
sys.path.insert(0, str(_HERE.parent))
