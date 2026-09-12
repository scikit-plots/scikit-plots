# `_brand` ownership family

Owned: package exposure in `__init__.py`, logo geometry/palette/presets/wordmark/save behavior in `_logo.py`, banner case/figlet/borders/cache/batch/CLI behavior in `_banner.py`, and focused tests.

External boundaries: Matplotlib/NumPy, system `figlet` and installed fonts, remote font hosting/network/cache permissions, and root-package aliases/console entrypoints. A mocked figlet lane is not live font evidence.
