# Copied: Mlflow
# See: from ._compat.optional_deps import LazyImport

"""Utility to lazy load modules."""

import importlib as _importlib
import sys as _sys
import types as _types


class LazyLoader(_types.ModuleType):
    """
    Class for module lazy loading.

    This class helps lazily load modules at package level, which avoids pulling in large
    dependencies like `tensorflow` or `torch`. This class is mirrored from wandb's LazyLoader:
    https://github.com/wandb/wandb/blob/79b2d4b73e3a9e4488e503c3131ff74d151df689/wandb/sdk/lib/lazyloader.py#L9
    """

    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals

        self._module = None
        super().__init__(str(name))

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        if self._module:
            # If already loaded, return the loaded module.
            return self._module

        # Import the target module and insert it into the parent's namespace
        module = _importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module
        _sys.modules[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the `LazyLoader`,
        # lookups are efficient (`__getattr__` is only called on lookups that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):  # noqa: D105
        module = self._load()
        return getattr(module, item)

    def __dir__(self):  # noqa: D105
        module = self._load()
        return dir(module)

    def __repr__(self):  # noqa: D105
        if not self._module:
            return f"<module '{self.__name__} (Not loaded yet)'>"
        return repr(self._module)
