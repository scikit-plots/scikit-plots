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
        "api",
        "api.plotters",
        "api.decomposition",
        "api.estimators",
        "api.metrics",
        "config",
        "config.cbook",
        "doremi",
        "doremi.composer",
        "doremi.config",
        "doremi.envelopes",
        "doremi.note",
        "doremi.note_io",
        "doremi.note_utils",
        "doremi.synthesis",
        "doremi.waveform_playback",
        "doremi.waveform_viz",
        "entities",
        "entities.file_info",
        "experimental",
        "kds",
        "llm_provider",
        "llm_provider.chat_provider",
        "llm_provider.clint_provider",
        "llm_provider.model_registry",
        "misc",
        "misc.plot_colortable",
        "modelplotpy",
        "pipeline",
        "pipeline.pipeline",
        "probscale",
        "probscale.algo",
        "probscale.formatters",
        "probscale.probscale",
        "probscale.transforms",
        "probscale.validate",
        "probscale.viz",
        "snsx",
        "snsx.dataset",
        "snsx.evaluation",
        "snsx.feature",
        "snsx.model",
        "snsx.representation",
        "snsx.target",
        "snsx.training",
        "sphinxext",
        "sphinxext.figmpl_directive",
        "sphinxext.mathmpl",
        "sphinxext.plot_directive",
        "sphinxext.roles",
        "stats",
        "typing",
        "ui_app",
        "utils",
        "utils.arguments_utils",
        "utils.cli_args",
        "utils.download_cloud_file_chunk",
        "utils.env_manager",
        "utils.exception_utils",
        "utils.file_utils",
        "utils.git_utils",
        "utils.lazy_load",
        "utils.logging_utils",
        "utils.mime_type_utils",
        "utils.os",
        "utils.plot_serializer",
        "utils.plugins",
        "utils.process",
        "utils.request_utils",
        "utils.string_utils",
        "utils.time",
        "utils.timeout",
        "utils.uri",
        "utils.utils_dot_env",
        "utils.utils_env",
        "utils.utils_file",
        "utils.utils_huggingface",
        "utils.utils_mlflow",
        "utils.utils_params",
        "utils.utils_path",
        "utils.utils_pil",
        "utils.utils_plot_mpl",
        "utils.utils_st_secrets",
        "utils.utils_stream",
        "utils.utils_toml",
        "utils.validation",
        "utils.yaml_utils",
        "visualkeras",
        "visualkeras.graph",
        "visualkeras.layer_utils",
        "visualkeras.layered",
        "visualkeras.utils",
        # py
        "cli",
        "conftest",
        "environment_variables",
        "exceptions",
        "ml_package_versions",
        "sp_logging",
        "version",
    ]
]
# The PRIVATE_BUT_PRESENT_MODULES list contains modules that lacked underscores
# in their name and hence looked public, but weren't meant to be. All these
# namespace were deprecated in the 1.8.0 release - see "clear split between
# public and private API" in the 1.8.0 release notes.
# These private modules support will be removed in SciPy v2.0.0, as the
# deprecation messages emitted by each of these modules say.
PRIVATE_BUT_PRESENT_MODULES = [
    "scikitplot." + s
    for s in [
        "_build_utils",
    ]
]


def is_unexpected(name):
    """Check if this needs to be considered."""
    if "._" in name or ".tests" in name or ".setup" in name:
        return False

    if name in PUBLIC_MODULES:
        return False

    if name in PRIVATE_BUT_PRESENT_MODULES:
        return False

    return True


# Skip public modules
SKIP_LIST = [
    # root artifact
    "scikitplot.conftest",
    "scikitplot._clv",
    # optional
    "scikitplot.ui_app.gradio",
    "scikitplot.ui_app.gradio.template_gr_app",
    "scikitplot.ui_app.gradio.template_gr_b_doremi_ui",
    "scikitplot.ui_app.gradio.template_gr_i_doremi_ui",
    "scikitplot.ui_app.streamlit",
    "scikitplot.ui_app.streamlit.build_app",
    "scikitplot.ui_app.streamlit.catalog",
    "scikitplot.ui_app.streamlit.run_app",
    "scikitplot.ui_app.streamlit.template_st_app",
    "scikitplot.ui_app.streamlit.template_st_chat_ui",
    "scikitplot.ui_app.streamlit.template_st_config",
    "scikitplot.ui_app.streamlit.template_st_data_visualizer_ui",
    "scikitplot.ui_app.streamlit.template_st_dataset_loader_ui",
    "scikitplot.ui_app.streamlit.template_st_login_ui",
]


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


# Skip unexpected object(s) that look like modules
# Stuff that clearly shouldn't be in the API and is detected by the next test
# below
SKIP_LIST_2 = [
    # root artifact
    "scikitplot._clv",
    # api artifact
    "scikitplot.decomposition",
    "scikitplot.estimators",
    "scikitplot.metrics",
    "scikitplot.plotters",
] + [
    'scikitplot.logger', 'scikitplot.doremi.envelopes.np', 'scikitplot.llm_provider.chat_provider.argparse', 'scikitplot.llm_provider.chat_provider.json',
    'scikitplot.llm_provider.chat_provider.logger', 'scikitplot.llm_provider.clint_provider.functools', 'scikitplot.llm_provider.clint_provider.importlib',
    'scikitplot.llm_provider.clint_provider.logger', 'scikitplot.llm_provider.clint_provider.requests', 'scikitplot.pipeline.pipeline.pipeline',
    'scikitplot.sphinxext.figmpl_directive.directives', 'scikitplot.sphinxext.figmpl_directive.matplotlib', 'scikitplot.sphinxext.figmpl_directive.nodes',
    'scikitplot.sphinxext.figmpl_directive.os', 'scikitplot.sphinxext.figmpl_directive.shutil', 'scikitplot.sphinxext.mathmpl.directives', 'scikitplot.sphinxext.mathmpl.hashlib',
    'scikitplot.sphinxext.mathmpl.mathtext', 'scikitplot.sphinxext.mathmpl.mpl', 'scikitplot.sphinxext.mathmpl.nodes', 'scikitplot.sphinxext.mathmpl.sphinx',
    'scikitplot.sphinxext.plot_directive.cbook', 'scikitplot.sphinxext.plot_directive.contextlib', 'scikitplot.sphinxext.plot_directive.directives',
    'scikitplot.sphinxext.plot_directive.doctest', 'scikitplot.sphinxext.plot_directive.itertools', 'scikitplot.sphinxext.plot_directive.jinja2',
    'scikitplot.sphinxext.plot_directive.matplotlib', 'scikitplot.sphinxext.plot_directive.os', 'scikitplot.sphinxext.plot_directive.plt', 'scikitplot.sphinxext.plot_directive.re',
    'scikitplot.sphinxext.plot_directive.shutil', 'scikitplot.sphinxext.plot_directive.sys', 'scikitplot.sphinxext.plot_directive.textwrap',
    'scikitplot.sphinxext.plot_directive.traceback', 'scikitplot.sphinxext.roles.matplotlib', 'scikitplot.sphinxext.roles.nodes', 'scikitplot.ui_app.logger',
    'scikitplot.ui_app.gradio.template_gr_app.gr', 'scikitplot.ui_app.gradio.template_gr_b_doremi_ui.atexit', 'scikitplot.ui_app.gradio.template_gr_b_doremi_ui.base64',
    'scikitplot.ui_app.gradio.template_gr_b_doremi_ui.doremi', 'scikitplot.ui_app.gradio.template_gr_b_doremi_ui.gr', 'scikitplot.ui_app.gradio.template_gr_b_doremi_ui.os',
    'scikitplot.ui_app.gradio.template_gr_b_doremi_ui.shutil', 'scikitplot.ui_app.gradio.template_gr_b_doremi_ui.tempfile', 'scikitplot.ui_app.gradio.template_gr_b_doremi_ui.uuid',
    'scikitplot.ui_app.gradio.template_gr_i_doremi_ui.doremi', 'scikitplot.ui_app.gradio.template_gr_i_doremi_ui.gr', 'scikitplot.ui_app.streamlit.build_app.os',
    'scikitplot.ui_app.streamlit.build_app.platform', 'scikitplot.ui_app.streamlit.build_app.shutil', 'scikitplot.ui_app.streamlit.build_app.subprocess',
    'scikitplot.ui_app.streamlit.run_app.argparse', 'scikitplot.ui_app.streamlit.run_app.logger', 'scikitplot.ui_app.streamlit.run_app.os',
    'scikitplot.ui_app.streamlit.run_app.subprocess', 'scikitplot.ui_app.streamlit.run_app.sys', 'scikitplot.ui_app.streamlit.template_st_app.importlib',
    'scikitplot.ui_app.streamlit.template_st_app.os', 'scikitplot.ui_app.streamlit.template_st_app.st', 'scikitplot.ui_app.streamlit.template_st_app.template_st_chat_ui',
    'scikitplot.ui_app.streamlit.template_st_app.template_st_data_visualizer_ui', 'scikitplot.ui_app.streamlit.template_st_app.template_st_dataset_loader_ui',
    'scikitplot.ui_app.streamlit.template_st_app.template_st_login_ui', 'scikitplot.ui_app.streamlit.template_st_chat_ui.chat_provider',
    'scikitplot.ui_app.streamlit.template_st_chat_ui.logger', 'scikitplot.ui_app.streamlit.template_st_chat_ui.st', 'scikitplot.ui_app.streamlit.template_st_config.st',
    'scikitplot.ui_app.streamlit.template_st_data_visualizer_ui.logger', 'scikitplot.ui_app.streamlit.template_st_data_visualizer_ui.pd',
    'scikitplot.ui_app.streamlit.template_st_data_visualizer_ui.plt', 'scikitplot.ui_app.streamlit.template_st_data_visualizer_ui.st',
    'scikitplot.ui_app.streamlit.template_st_dataset_loader_ui.logger', 'scikitplot.ui_app.streamlit.template_st_dataset_loader_ui.pd',
    'scikitplot.ui_app.streamlit.template_st_dataset_loader_ui.st', 'scikitplot.ui_app.streamlit.template_st_dataset_loader_ui.textwrap',
    'scikitplot.ui_app.streamlit.template_st_dataset_loader_ui.traceback', 'scikitplot.ui_app.streamlit.template_st_dataset_loader_ui.uuid',
    'scikitplot.ui_app.streamlit.template_st_login_ui.st', 'scikitplot.utils.inspect', 'scikitplot.utils.socket', 'scikitplot.utils.subprocess',
    'scikitplot.utils.uuid', 'scikitplot.utils.arguments_utils.inspect', 'scikitplot.utils.cli_args.click', 'scikitplot.utils.cli_args.warnings',
    'scikitplot.utils.download_cloud_file_chunk.argparse', 'scikitplot.utils.download_cloud_file_chunk.importlib', 'scikitplot.utils.download_cloud_file_chunk.json',
    'scikitplot.utils.download_cloud_file_chunk.os', 'scikitplot.utils.download_cloud_file_chunk.sys', 'scikitplot.utils.exception_utils.sys',
    'scikitplot.utils.exception_utils.traceback', 'scikitplot.utils.file_utils.atexit', 'scikitplot.utils.file_utils.codecs', 'scikitplot.utils.file_utils.concurrent',
    'scikitplot.utils.file_utils.contextlib', 'scikitplot.utils.file_utils.fnmatch', 'scikitplot.utils.file_utils.gzip', 'scikitplot.utils.file_utils.json',
    'scikitplot.utils.file_utils.logger', 'scikitplot.utils.file_utils.math', 'scikitplot.utils.file_utils.os', 'scikitplot.utils.file_utils.pathlib',
    'scikitplot.utils.file_utils.posixpath', 'scikitplot.utils.file_utils.requests', 'scikitplot.utils.file_utils.shutil', 'scikitplot.utils.file_utils.stat',
    'scikitplot.utils.file_utils.sys', 'scikitplot.utils.file_utils.tarfile', 'scikitplot.utils.file_utils.tempfile', 'scikitplot.utils.file_utils.time',
    'scikitplot.utils.file_utils.urllib', 'scikitplot.utils.git_utils.logging', 'scikitplot.utils.git_utils.os', 'scikitplot.utils.lazy_load.importlib',
    'scikitplot.utils.lazy_load.sys', 'scikitplot.utils.lazy_load.types', 'scikitplot.utils.logging_utils.contextlib', 'scikitplot.utils.logging_utils.logging',
    'scikitplot.utils.logging_utils.re', 'scikitplot.utils.logging_utils.sys', 'scikitplot.utils.mime_type_utils.os', 'scikitplot.utils.mime_type_utils.pathlib',
    'scikitplot.utils.os.os', 'scikitplot.utils.plot_serializer.cab', 'scikitplot.utils.plot_serializer.json', 'scikitplot.utils.plot_serializer.mpl',
    'scikitplot.utils.plot_serializer.np', 'scikitplot.utils.plot_serializer.os', 'scikitplot.utils.plot_serializer.plt', 'scikitplot.utils.plugins.importlib',
    'scikitplot.utils.plugins.sys', 'scikitplot.utils.process.functools', 'scikitplot.utils.process.os', 'scikitplot.utils.process.subprocess', 'scikitplot.utils.process.sys',
    'scikitplot.utils.request_utils.os', 'scikitplot.utils.request_utils.random', 'scikitplot.utils.request_utils.requests', 'scikitplot.utils.request_utils.urllib3',
    'scikitplot.utils.string_utils.re', 'scikitplot.utils.string_utils.shlex', 'scikitplot.utils.time.datetime', 'scikitplot.utils.time.time', 'scikitplot.utils.timeout.signal',
    'scikitplot.utils.uri.os', 'scikitplot.utils.uri.pathlib', 'scikitplot.utils.uri.posixpath', 'scikitplot.utils.uri.re', 'scikitplot.utils.uri.urllib',
    'scikitplot.utils.uri.uuid', 'scikitplot.utils.utils_dot_env.logger', 'scikitplot.utils.utils_dot_env.os', 'scikitplot.utils.utils_dot_env.pathlib',
    'scikitplot.utils.utils_env.os', 'scikitplot.utils.utils_huggingface.logger', 'scikitplot.utils.utils_huggingface.os', 'scikitplot.utils.utils_mlflow.logger',
    'scikitplot.utils.utils_mlflow.os', 'scikitplot.utils.utils_path.os', 'scikitplot.utils.utils_path.re', 'scikitplot.utils.utils_path.shutil',
    'scikitplot.utils.utils_plot_mpl.contextlib', 'scikitplot.utils.utils_plot_mpl.functools', 'scikitplot.utils.utils_plot_mpl.logger', 'scikitplot.utils.utils_plot_mpl.mpl',
    'scikitplot.utils.utils_plot_mpl.np', 'scikitplot.utils.utils_plot_mpl.plt', 'scikitplot.utils.utils_plot_mpl.warnings', 'scikitplot.utils.utils_st_secrets.logger',
    'scikitplot.utils.utils_st_secrets.os', 'scikitplot.utils.utils_toml.logger', 'scikitplot.utils.utils_toml.os', 'scikitplot.utils.utils_toml.pathlib',
    'scikitplot.utils.utils_toml.toml', 'scikitplot.utils.utils_toml.tomllib', 'scikitplot.utils.validation.warnings', 'scikitplot.utils.yaml_utils.codecs',
    'scikitplot.utils.yaml_utils.json', 'scikitplot.utils.yaml_utils.os', 'scikitplot.utils.yaml_utils.pathlib', 'scikitplot.utils.yaml_utils.shutil',
    'scikitplot.utils.yaml_utils.tempfile', 'scikitplot.utils.yaml_utils.yaml', 'scikitplot.cli.cli_args', 'scikitplot.cli.click', 'scikitplot.cli.contextlib',
    'scikitplot.cli.json', 'scikitplot.cli.os', 'scikitplot.cli.re', 'scikitplot.cli.scikitplot', 'scikitplot.cli.sys', 'scikitplot.cli.warnings', 'scikitplot.conftest.gc',
    'scikitplot.conftest.hypothesis', 'scikitplot.conftest.json', 'scikitplot.conftest.np', 'scikitplot.conftest.np_testing', 'scikitplot.conftest.os', 'scikitplot.conftest.pd',
    'scikitplot.conftest.pytest', 'scikitplot.conftest.tempfile', 'scikitplot.conftest.warnings', 'scikitplot.environment_variables.os',
    'scikitplot.environment_variables.tempfile', 'scikitplot.exceptions.json', 'scikitplot.ui_app.gradio', 'scikitplot.ui_app.streamlit', 'scikitplot.conftest.pytest_run_parallel',
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
