"""
This test script is adopted from:
https://github.com/numpy/numpy/blob/main/numpy/tests/test_public_api.py
"""

import importlib
import pkgutil
import types
import warnings

# from importlib import import_module
import numpy as np
import pytest

import scikitplot

from ..conftest import xp_available_backends


def test_dir_testing():
    """
    Assert that output of dir has only one "testing/tester"
    attribute without duplicate
    """
    assert len(dir(scikitplot)) == len(set(dir(scikitplot)))


# Historically SciPy has not used leading underscores for private submodules
# much.  This has resulted in lots of things that look like public modules
# (i.e. things that can be imported as `import scipy.somesubmodule.somefile`),
# but were never intended to be public.  The PUBLIC_MODULES list contains
# modules that are either public because they were meant to be, or because they
# contain public functions/objects that aren't present in any other namespace
# for whatever reason and therefore should be treated as public.
PUBLIC_MODULES = [
    "scikitplot." + s
    for s in [
        "experimental",
        "cbook",
        "sp_logging",
        "api",
        "api.plotters",
        "api.decomposition",
        "api.estimators",
        "api.metrics",
        "api.utils",
        "api.utils.validation",
        "kds",
        "misc",
        "misc.font",
        "misc.helper",
        "misc.plot_colortable",
        "modelplotpy",
        "probscale",
        "probscale.algo",
        "probscale.formatters",
        "probscale.probscale",
        "probscale.transforms",
        "probscale.validate",
        "probscale.viz",
        "stats",
        "typing",
        "visualkeras",
        "visualkeras.graph",
        "visualkeras.layer_utils",
        "visualkeras.layered",
        "visualkeras.utils",
    ]
]
# The PRIVATE_BUT_PRESENT_MODULES list contains modules that lacked underscores
# in their name and hence looked public, but weren't meant to be. All these
# namespace were deprecated in the 1.8.0 release - see "clear split between
# public and private API" in the 1.8.0 release notes.
# These private modules support will be removed in SciPy v2.0.0, as the
# deprecation messages emitted by each of these modules say.
PRIVATE_BUT_PRESENT_MODULES = ["scikitplot." + s for s in ["_build_utils"]]


def is_unexpected(name):
    """Check if this needs to be considered."""
    if "._" in name or ".tests" in name or ".setup" in name:
        return False

    if name in PUBLIC_MODULES:
        return False

    if name in PRIVATE_BUT_PRESENT_MODULES:
        return False

    return True


SKIP_LIST = ["scikitplot.conftest", "scikitplot.version"]


# XXX: this test does more than it says on the tin - in using `pkgutil.walk_packages`,
# it will raise if it encounters any exceptions which are not handled by `ignore_errors`
# while attempting to import each discovered package.
# For now, `ignore_errors` only ignores what is necessary, but this could be expanded -
# for example, to all errors from private modules or git subpackages - if desired.
@pytest.mark.thread_unsafe
def test_all_modules_are_expected():
    """
    Test that we don't add anything that looks like a new public module by
    accident.  Check is based on filenames.
    """

    def ignore_errors(name):
        # if versions of other array libraries are installed which are incompatible
        # with the installed NumPy version, there can be errors on importing
        # `array_api_compat`. This should only raise if SciPy is configured with
        # that library as an available backend.
        backends = {"cupy", "torch", "dask.array"}
        for backend in backends:
            path = f"array_api_compat.{backend}"
            if path in name and backend not in xp_available_backends:
                return
        raise

    modnames = []

    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning, "scikitplot._build_utils")
        for _, modname, _ in pkgutil.walk_packages(
            path=scikitplot.__path__,
            prefix=scikitplot.__name__ + ".",
            onerror=ignore_errors,
        ):
            if is_unexpected(modname) and modname not in SKIP_LIST:
                # We have a name that is new.  If that's on purpose, add it to
                # PUBLIC_MODULES.  We don't expect to have to add anything to
                # PRIVATE_BUT_PRESENT_MODULES.  Use an underscore in the name!
                modnames.append(modname)

    if modnames:
        raise AssertionError(f"Found unexpected modules: {modnames}")


# Stuff that clearly shouldn't be in the API and is detected by the next test
# below
SKIP_LIST_2 = [
    # seaborn artifact
    "scikitplot.algorithms",
    "scikitplot.axisgrid",
    "scikitplot.categorical",
    "scikitplot.cm",
    "scikitplot.colors",
    "scikitplot.distributions",
    "scikitplot.external",
    "scikitplot.matrix",
    "scikitplot.miscplot",
    "scikitplot.palettes",
    "scikitplot.rcmod",
    "scikitplot.regression",
    "scikitplot.relational",
    "scikitplot.utils",
    "scikitplot.widgets",
    # api artifact
    "scikitplot.decomposition",
    "scikitplot.estimators",
    "scikitplot.metrics",
    "scikitplot.plotters",
    # root artifact
    "scikitplot.version",
]


def test_all_modules_are_expected_2():
    """
    Method checking all objects. The pkgutil-based method in
    `test_all_modules_are_expected` does not catch imports into a namespace,
    only filenames.
    """

    def find_unexpected_members(mod_name):
        members = []
        module = importlib.import_module(mod_name)
        if hasattr(module, "__all__"):
            objnames = module.__all__
        else:
            objnames = dir(module)

        for objname in objnames:
            if not objname.startswith("_"):
                fullobjname = mod_name + "." + objname
                if isinstance(getattr(module, objname), types.ModuleType):
                    if is_unexpected(fullobjname) and fullobjname not in SKIP_LIST_2:
                        members.append(fullobjname)

        return members

    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning, "scikitplot._build_utils")
        unexpected_members = find_unexpected_members("scikitplot")

    for modname in PUBLIC_MODULES:
        unexpected_members.extend(find_unexpected_members(modname))

    if unexpected_members:
        raise AssertionError(
            f"Found unexpected object(s) that look like modules: {unexpected_members}"
        )


def test_api_importable():
    """
    Check that all submodules listed higher up in this file can be imported
    Note that if a PRIVATE_BUT_PRESENT_MODULES entry goes missing, it may
    simply need to be removed from the list (deprecation may or may not be
    needed - apply common sense).
    """

    def check_importable(module_name):
        try:
            importlib.import_module(module_name)
        except (ImportError, AttributeError):
            return False

        return True

    module_names = []
    for module_name in PUBLIC_MODULES:
        if not check_importable(module_name):
            module_names.append(module_name)

    if module_names:
        raise AssertionError(
            f"Modules in the public API that cannot be imported: {module_names}"
        )

    with warnings.catch_warnings(record=True):
        warnings.filterwarnings("always", category=DeprecationWarning)
        warnings.filterwarnings("always", category=ImportWarning)
        for module_name in PRIVATE_BUT_PRESENT_MODULES:
            if not check_importable(module_name):
                module_names.append(module_name)

    if module_names:
        raise AssertionError(
            "Modules that are not really public but looked "
            "public and can not be imported: "
            f"{module_names}"
        )


@pytest.mark.thread_unsafe
@pytest.mark.parametrize(
    ("module_name", "correct_module"),
    [
        ("scikitplot.kds._deciles", None)
        #    ('scikitplot.kds._deciles.', None),
    ],
)
def test_private_but_present_deprecation(module_name, correct_module):
    # gh-18279, gh-17572, gh-17771 noted that deprecation warnings
    # for imports from private modules
    # were misleading. Check that this is resolved.
    module = importlib.import_module(module_name)
    if correct_module is None:
        import_name = f'scikitplot.{module_name.split(".")[1]}'
    else:
        import_name = f'scikitplot.{module_name.split(".")[1]}.{correct_module}'

    correct_import = importlib.import_module(import_name)

    # Attributes that were formerly in `module_name` can still be imported from
    # `module_name`, albeit with a deprecation warning.
    for attr_name in module.__all__:
        # ensure attribute is present where the warning is pointing
        assert (
            getattr(correct_import, attr_name, None) is not None
        ), f"{getattr(correct_import, attr_name, None)}"
        message = f"Please import `{attr_name}` from the `{import_name}`..."
        with pytest.deprecated_call(match=message):
            getattr(module, attr_name)

    # Attributes that were not in `module_name` get an error notifying the user
    # that the attribute is not in `module_name` and that `module_name` is deprecated.
    message = f"`{module_name}` is deprecated..."
    with pytest.raises(AttributeError, match=message):
        module.ekki
