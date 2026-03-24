# scikitplot/utils/__init__.py

"""
Various utilities to help with development.
"""

# scikitplot/
# │
# └── utils/                   ← 15 utility submodules
#     ├── __init__.py
#     ├── _missing.py          ← sklearn-adapted NaN helpers
#     ├── _file.py             ← humansize, humansize_vector
#     ├── _path.py             ← PathNamer, make_path, get_path, remove_path
#     ├── _time.py             ← Timer context manager
#     ├── _inspect.py          ← 3 signature-inspection helpers
#     ├── _serialize.py        ← Matplotlib→JSON serializers
#     ├── _toml.py             ← Multi-backend TOML read/write
#     ├── _show_versions.py    ← System/dep introspection
#     ├── _encode.py           ← sklearn _unique encoding (adapted)
#     ├── _fixes.py            ← sklearn compat shims
#     ├── _matplotlib.py       ← safe_tight_layout, save decorators
#     ├── _pil.py              ← PIL font/image helpers
#     ├── _dotenv.py           ← .env loader
#     ├── _env.py              ← env-var resolver
#     ├── _huggingface.py      ← HF Hub login helper
#     ├── _mlflow.py           ← MLflow gateway config loader
#     ├── _streamlit.py        ← Streamlit secrets helpers
#     └── tests/               ← 3 existing, 7 new created below

from .. import logger  # noqa: F401
from ._path import PathNamer, get_path, make_path, remove_path  # noqa: F401
from ._time import Timer  # noqa: F401
