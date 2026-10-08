# libs/_tools/registry.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Packaging facts for the partial distributions.

Two files describe a partial distribution and each fact lives in exactly one:

* ``scikitplot/_distributions.py`` says which files it **owns** (and is also
  read at run time, for install hints and ``scikitplot doctor``);
* this module says how it is **packaged**: its dependencies, extras, licence
  and build inputs.

Everything else is inherited from the repository's root ``pyproject.toml`` or
derived by rule, so there is nothing here to keep in step by hand.

Notes
-----
**Developer.** The dependency rule is the same for every distribution, so that
none of them is a special case:

* ``dependencies`` are what the part imports at import time, plus the core
  distribution. Their version specifiers are **inherited by name** from the
  root ``[project.dependencies]``, so a partial distribution and the full one
  can never declare conflicting ranges for the same project.
* ``extras`` re-export root ``[project.optional-dependencies]`` groups **under
  the same name**, so ``scikit-plots[mcp]`` and ``scikit-plots-mcp[mcp]`` mean
  the same thing.
* ``siblings`` are extras that pull in another partial distribution, for the
  optional integrations between parts (``scikitplot.mcp`` can search a
  ``scikitplot.corpus`` index when one is installed).

A requirement the root does not declare is written out in ``requires`` with
the reason beside it.
"""

from __future__ import annotations

import re
from typing import NamedTuple

__all__ = [
    "BUILD_REQUIRES",
    "CLASSIFIER_PREFIXES",
    "NAME_PREFIX",
    "PACKAGES",
    "Extension",
    "Package",
    "by_distribution",
    "directory_of",
    "python_satisfies",
]

#: Every partial distribution is named ``<prefix><directory>``; the directory
#: under ``libs/`` is therefore derived from the name and never stated twice.
NAME_PREFIX = "scikit-plots-"

#: Build requirement of every partial distribution.
#:
#: The floor is the first setuptools release that builds these projects:
#: 61.0.0 introduced the ``[project]`` table (PEP 621) but fails with a
#: ``TypeError`` when a field is declared ``dynamic``, which the licence is
#: (see ``generate.render_setup``); 61.2.0 is the first release verified to
#: work. There is no ceiling: nothing here uses an interface setuptools has
#: announced it will remove.
BUILD_REQUIRES: tuple[str, ...] = ("setuptools>=61.2.0",)

#: Root classifiers a partial distribution inherits, selected by prefix. The
#: rest of the root list describes the full distribution only (its compiled
#: languages, its plotting framework).
CLASSIFIER_PREFIXES: tuple[str, ...] = (
    "Development Status ::",
    "Intended Audience ::",
    "Natural Language ::",
    "Operating System ::",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development ::",
    "Typing ::",
)


class Extension(NamedTuple):
    """
    One compiled extension module of a partial distribution.

    Parameters
    ----------
    name : str
        Dotted module name, e.g. ``"scikitplot.cexternals._annoy.annoylib"``.
    sources : tuple of str
        Source files, as POSIX paths relative to the lib directory. A
        ``.pyx`` source is translated to C++ by Cython before it is compiled.
    include_dirs : tuple of str
        Header directories, as POSIX paths relative to the lib directory.
    cxx_standard : str
        C++ language standard the sources need, e.g. ``"c++17"``.
    templates : tuple of str
        Tempita templates (``*.in``) rendered next to themselves, without the
        ``.in`` suffix, before the extension is built.
    define_macros : tuple of (str, str)
        Preprocessor macros defined for every source of the extension.
    threads_macro : str or None
        Preprocessor macro that compiles in the extension's multithreaded
        code, when it has any. It is defined, and the platform's thread
        library is linked, only when the builder asks for it
        (``SKPLT_BUILD_THREADS=1``, see the generated ``setup.py``); a target
        without threads refuses the request. ``None`` for an extension with
        no such code.

    Notes
    -----
    **Developer.** These values restate what the full distribution's Meson
    build does for the same extension (``scikitplot/**/meson.build`` and the
    root ``meson.build``), so that the partial build compiles the same code
    the same way. The one deliberate difference is that no CPU-specific flag
    is passed: a wheel must run on machines other than the one that built it.
    """

    name: str
    sources: tuple[str, ...]
    include_dirs: tuple[str, ...]
    cxx_standard: str
    templates: tuple[str, ...] = ()
    define_macros: tuple[tuple[str, str], ...] = ()
    threads_macro: str | None = None


class Package(NamedTuple):
    """
    How one partial distribution is packaged.

    Parameters
    ----------
    distribution : str
        Canonical project name; must be an entry of
        ``scikitplot._distributions.DISTRIBUTIONS``.
    license : str
        SPDX licence expression covering every file the distribution ships.
    inherit : tuple of str
        Project names whose requirement lines are copied from the root
        ``[project.dependencies]``.
    requires : tuple of str
        Requirements the root does not declare, written in full. The project
        declares requirements without version ranges wherever it can; write a
        lower bound only where a measured failure needs it, and say what was
        measured.
    test_python : str or None
        Python versions the test suite itself runs on, as ``">=X.Y"``, when
        that is narrower than ``requires-python``. Below it the modules are
        still imported and the commands still run; only the suite is reported
        as skipped, with this reason. Say what was measured.
    test_ignore : tuple of str
        Test files or directories the verification leaves out, as paths below
        ``scikitplot/`` with ``/`` separators. Only for tests that *state*
        they cannot run from an installed distribution (they need the
        repository checkout); say which statement that is.
    test_gated : tuple of (str, str)
        Test files or directories that need a newer Python than the rest of
        the suite, as ``(path, ">=X.Y")`` with the path written like a
        ``test_ignore`` entry. Below the floor the verification leaves the
        path out and runs everything else; at or above it the path runs like
        any other. For tests of an optional tier whose own code needs that
        Python; say what was measured.
    lowest_constraints : tuple of str
        Constraints applied only by the verification's lowest-version run,
        never written into the distribution. Each one records a *measured*
        case where two floors the root declares cannot be used together, so
        that the run tests the oldest environment that can work instead of
        one that cannot. It changes what is tested, not what is declared.
    extras : tuple of str
        Root ``[project.optional-dependencies]`` groups re-exported under the
        same name.
    own_extras : tuple of (str, tuple of str)
        Extras that belong to this distribution alone: name to requirements,
        written in full and version-free. For optional needs of single
        features that the root does not group (the Sphinx extensions each
        import different third-party packages). There is deliberately no
        catch-all extra: the lists differ too much in weight (a theme helper
        beside a web service) for one name to be a sensible default.
    siblings : tuple of (str, tuple of str)
        Extra name to the partial distributions it pulls in.
    keywords : tuple of str
        Keywords for the package index.
    classifiers : tuple of str
        Classifiers added to the inherited ones.
    scripts : bool
        Whether the root ``[project.scripts]`` are installed by this
        distribution. True for the core only: a console script is a file, and
        a file has one owner.
    extensions : tuple of Extension
        Compiled extension modules. Empty for a pure-Python distribution.
    build_inherit : tuple of str
        Project names whose requirement lines are copied from the root
        ``[build-system].requires`` into this distribution's build
        requirements, in addition to :data:`BUILD_REQUIRES`.
    requires_python : str or None
        Overrides the root ``requires-python`` when the part supports a
        narrower range, in the form ``">=X.Y"``. ``None`` inherits. An
        override states what was measured: ``libs/_tools verify`` installs
        and tests the part on every Python the range admits, and checks that
        an installer refuses it on one the range excludes.
    python_gated : tuple of (str, str)
        Modules that belong to an optional tier needing a newer Python than
        the distribution itself, each with that tier's floor in the form
        ``">=X.Y"``. Below the floor such a module cannot be imported, by
        design; ``libs/_tools verify`` reports it as gated instead of as a
        failure. At or above the floor it is checked like any other module.
    test_extras : tuple of str
        The distribution's own extras that its test suite needs installed.
    test_extras_python : str or None
        The Pythons on which ``test_extras`` can be installed, in the form
        ``">=X.Y"``, when that is narrower than ``requires_python``. On an
        older Python the part is still installed and checked; only its test
        suite is reported as skipped, with this as the reason.
    test_requires : tuple of str
        Test tooling its test suite needs beyond ``pytest``. These are not
        dependencies of the distribution and are not part of its metadata;
        only ``libs/_tools verify`` installs them.
    example : str
        A short usage example for the package description. It is executed by
        ``libs/_tools verify`` in an environment that has only this
        distribution, so it cannot drift from what the package does.
    example_language : {"python", "shell"}
        How ``example`` is run and highlighted.
    """

    distribution: str
    license: str
    inherit: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    extras: tuple[str, ...] = ()
    siblings: tuple[tuple[str, tuple[str, ...]], ...] = ()
    keywords: tuple[str, ...] = ()
    classifiers: tuple[str, ...] = ()
    scripts: bool = False
    extensions: tuple[Extension, ...] = ()
    build_inherit: tuple[str, ...] = ()
    requires_python: str | None = None
    python_gated: tuple[tuple[str, str], ...] = ()
    test_extras: tuple[str, ...] = ()
    test_extras_python: str | None = None
    test_requires: tuple[str, ...] = ()
    example: str = ""
    example_language: str = "python"
    lowest_constraints: tuple[str, ...] = ()
    test_ignore: tuple[str, ...] = ()
    test_python: str | None = None
    own_extras: tuple[tuple[str, tuple[str, ...]], ...] = ()
    test_gated: tuple[tuple[str, str], ...] = ()


#: Macros the root ``meson.build`` defines for every C++ extension
#: (``cython_cpp_flags``): heap types with a mutable ``__module__``, and the
#: C-header compatibility define.
_CYTHON_CPP_MACROS: tuple[tuple[str, str], ...] = (
    ("CYTHON_USE_TYPE_SPECS", "1"),
    ("__STDC_VERSION__", "0"),
)

#: Macro of the vendored Annoy header (``annoylib.h``) that selects
#: ``AnnoyIndexMultiThreadedBuildPolicy``: ``build(n_trees, n_jobs=N)`` then
#: can build the trees on ``N`` threads. Without it the header selects the
#: single-threaded policy and ``n_jobs`` is accepted and has no effect.
#:
#: Two switches, on purpose:
#:
#: * build time, here: ``SKPLT_BUILD_THREADS=1`` (the Meson build of the full
#:   distribution has ``-Dannoy-threads=true``; the defaults are equal and
#:   ``test_generate.py`` checks that). It decides what a wheel *can* do.
#: * run time: ``SKPLT_ANNOY_THREADS`` = ``auto``, ``single`` or ``multi``
#:   (``scikitplot/annoy/_threads.py``). It decides what a build *does*, so
#:   one wheel compiled with threads serves every user.
#:
#: Measured on Linux x86_64 (2 CPUs, CPython 3.10, GCC 13), 60 000 vectors of
#: 64 dimensions, 24 trees, same seed, the compiled module called directly:
#:
#: ===================  =========  ==========================================
#: wheel                n_jobs     build time; saved file
#: ===================  =========  ==========================================
#: without the macro    1, 2, -1   2.0 to 2.2 s; one file, always
#: with the macro       1, -1      2.0 to 2.2 s; that same file, byte for byte
#: with the macro       2          1.1 s; other trees, and node order varies
#:                                 from run to run
#: ===================  =========  ==========================================
#:
#: Both annoy suites pass on a wheel built with the macro. Two findings of
#: the first measurement are settled (``tasks/todo.md``, round 5):
#: ANNOY-MT-002, files that differed in a few bytes, was uninitialised
#: padding copied from the stack into every split node; ANNOY-MT-003,
#: ``n_jobs=-1`` running on one thread, is now stated by the run-time rule
#: (``auto``: one thread; ``multi``: every CPU) instead of left to chance.
_ANNOY_THREADS_MACRO = "ANNOYLIB_MULTITHREADED_BUILD"

PACKAGES: tuple[Package, ...] = (
    Package(
        distribution="scikit-plots-skinny",
        license="BSD-3-Clause",
        keywords=("scikit-plots", "cli", "logging"),
        scripts=True,
        # The click frontend, and the YAML and TOML writers of `--format`.
        test_requires=("click", "pyyaml", "tomli-w"),
        example=(
            "import scikitplot as sp\n"
            "\n"
            'sp.get_logger().info("scikit-plots %s", sp.__version__)\n'
        ),
    ),
    Package(
        distribution="scikit-plots-rank-bm25",
        # scikitplot/rank_bm25 is adapted from dorianbrown/rank_bm25; its
        # sources carry an Apache-2.0 notice.
        license="BSD-3-Clause AND Apache-2.0",
        inherit=("numpy",),
        keywords=("scikit-plots", "bm25", "ranking", "information retrieval", "search"),
        example=(
            "from scikitplot.rank_bm25 import BM25Okapi\n"
            "\n"
            'corpus = ["Hello there good man!", "It is quite windy in London"]\n'
            'bm25 = BM25Okapi([doc.split(" ") for doc in corpus])\n'
            'print(bm25.get_top_n("windy London".split(" "), corpus, n=1))\n'
        ),
    ),
    Package(
        distribution="scikit-plots-corpus",
        license="BSD-3-Clause",
        inherit=("numpy",),
        # Measured: on Python 3.8 ``import scikitplot.corpus`` raises TypeError
        # (scikitplot/corpus/_types.py subscripts the built-in ``dict`` at
        # import time), and part of its test suite does not parse.
        requires_python=">=3.9",
        extras=("corpus",),
        siblings=(("annoy", ("scikit-plots-annoy",)),),
        # Readers and downloaders whose tests exercise the real libraries.
        test_requires=("pandas", "pillow", "requests"),
        keywords=(
            "scikit-plots",
            "corpus",
            "rag",
            "chunking",
            "embeddings",
            "retrieval",
        ),
        example="import scikitplot.corpus as corpus\n",
    ),
    Package(
        distribution="scikit-plots-annoy",
        # scikitplot/cexternals/_annoy is Spotify's Annoy (Apache-2.0); its
        # LICENSE file ships inside the package.
        license="BSD-3-Clause AND Apache-2.0",
        inherit=("numpy", "scikit-learn"),
        requires=(
            # Imported at import time by scikitplot/annoy/_mixins/_pickle.py
            # (``Self``), and not declared by the root project.
            "typing_extensions",
        ),
        # scikitplot/annoy/_mixins/_ndarray.py and _pickle.py import
        # ``typing.TypeAlias`` and evaluate ``X | Y`` on types at import time;
        # both need Python 3.10. Measured: the extensions build on 3.8, and
        # ``import scikitplot.annoy`` then fails.
        requires_python=">=3.10",
        # The root declares ``numpy>=2.0.0`` (Python >= 3.9) and
        # ``scikit-learn>=1.3.0rc1``. Those two floors cannot be used
        # together. Measured on Python 3.10 with numpy 2.0.0:
        #   scikit-learn 1.3.0rc1  installs; import fails ("numpy.dtype size
        #                          changed, may indicate binary incompatibility")
        #   scikit-learn 1.3.0     installs; import fails (cannot import
        #                          ``ComplexWarning`` from ``numpy.core.numeric``)
        #   1.3.2, 1.4.0, 1.4.1    refused by the installer (they require
        #                          ``numpy<2.0``)
        #   scikit-learn 1.4.2     installs and imports
        # so 1.4.2 is the oldest scikit-learn this part can run with, and the
        # lowest-version run starts there. Remove this when the root's two
        # floors agree.
        lowest_constraints=("scikit-learn>=1.4.2",),
        keywords=(
            "scikit-plots",
            "annoy",
            "nearest neighbors",
            "ann",
            "vector search",
        ),
        classifiers=("Programming Language :: C++", "Programming Language :: Cython"),
        extensions=(
            # The CPython C-API binding (scikitplot/cexternals/_annoy/meson.build).
            Extension(
                name="scikitplot.cexternals._annoy.annoylib",
                sources=("scikitplot/cexternals/_annoy/src/annoymodule.cc",),
                include_dirs=("scikitplot/cexternals/_annoy/src",),
                cxx_standard="c++17",
                define_macros=_CYTHON_CPP_MACROS,
                threads_macro=_ANNOY_THREADS_MACRO,
            ),
            # The Cython binding, generated from a Tempita template
            # (scikitplot/annoy/_annoy/meson.build). Its ``cdef extern``
            # blocks name headers relative to its own directory, which is why
            # that directory is the include path.
            Extension(
                name="scikitplot.annoy._annoy.annoylib",
                sources=("scikitplot/annoy/_annoy/annoylib.pyx",),
                include_dirs=("scikitplot/annoy/_annoy",),
                cxx_standard="c++17",
                templates=(
                    "scikitplot/annoy/_annoy/annoylib.pxd.in",
                    "scikitplot/annoy/_annoy/annoylib.pyx.in",
                ),
                define_macros=_CYTHON_CPP_MACROS,
                threads_macro=_ANNOY_THREADS_MACRO,
            ),
        ),
        # Cython translates the binding and provides Tempita.
        build_inherit=("cython",),
        example=(
            "from scikitplot.annoy import Index\n"
            "\n"
            'index = Index(3, "angular")\n'
            "index.add_item(0, [1.0, 0.0, 0.0])\n"
            "index.add_item(1, [0.0, 1.0, 0.0])\n"
            "index.build(10)\n"
            "print(index.get_nns_by_item(0, 2))\n"
        ),
    ),
    Package(
        distribution="scikit-plots-sphinx-ext",
        # SPDX identifiers found in the files of the tree: BSD-3-Clause,
        # Apache-2.0 and MIT.
        license="BSD-3-Clause AND Apache-2.0 AND MIT",
        # Every extension needs Sphinx; nothing else is needed to import the
        # package or any module in it (measured: all modules import with
        # Sphinx alone). What single extensions use beyond that (a theme, an
        # HTTP client, ...) stays the choice of the project that enables them.
        requires=("sphinx",),
        keywords=("scikit-plots", "sphinx", "documentation", "extension"),
        classifiers=("Framework :: Sphinx :: Extension",),
        # One extra per extension that imports something beyond Sphinx, named
        # after the extension directory (``_sphinx_collection`` is
        # ``collection``). Each list is what the extension's modules import,
        # found by reading their import statements; an extension that is not
        # listed needs Sphinx only. ``proxy`` is the hosted service under
        # ``_sphinx_ai_assistant/_hf_spaces_proxy``, not a Sphinx extension.
        # The model server under ``_hf_spaces_model`` (torch, transformers,
        # gradio) is deployed from its own requirements and has no extra.
        own_extras=(
            ("ai-assistant", ("beautifulsoup4", "httpx", "markdownify")),
            ("collection", ("pyyaml", "sphinx-design")),
            ("feedback", ("httpx",)),
            ("gallery-grid", ("sphinx-design",)),
            ("gallery-jupyterlite", ("sphinx-gallery",)),
            ("llm", ("sphinx-markdown-builder",)),
            ("youtube-gallery", ("defusedxml", "pyyaml")),
            (
                "proxy",
                (
                    "cryptography",
                    "fastapi",
                    "httpx",
                    "huggingface_hub",
                    "pandas",
                    "redis",
                    "starlette",
                    # The release tooling reads TOML; before Python 3.11 the
                    # parser is this package instead of the standard library.
                    'tomli; python_version < "3.11"',
                    "typing_extensions",
                ),
            ),
        ),
        # Measured on the installed wheel, with every test requirement
        # installed (Linux x86_64):
        #
        # * every module imports on Python 3.8 to 3.15;
        # * on 3.9, with the hosted services gated below: 4958 passed, 80
        #   skipped, 5 xfailed;
        # * on 3.8 the suite does not run as a whole: 225 failed, 8 errors.
        #   125 of them are the tests' own use of the Sphinx test fixtures
        #   (``'PosixPath' object has no attribute 'makedirs'`` and
        #   ``'copytree'``): the newest Sphinx for Python 3.8 hands out its
        #   own path class, the tests are written for the ``pathlib`` paths
        #   of later releases. The others were Python 3.9 API in tests and,
        #   in four extensions, in the code (``str.removeprefix``); the code
        #   is fixed, see ``tasks/todo.md`` (round 4).
        test_python=">=3.9",
        # The hosted services (``_hf_spaces_proxy`` and the feedback service)
        # create ``asyncio`` locks and events in constructors. Python 3.9
        # binds those to the current event loop at creation and refuses when
        # there is none ("There is no current event loop in thread
        # 'MainThread'": 46 tests), and two tests use the built-in ``anext``
        # (3.10). The proxy is deployed on Python 3.11 (its Dockerfile). The
        # Sphinx extensions do not go through any of this.
        test_gated=tuple(
            ("_externals/_sphinx_ext/" + path, ">=3.10")
            for path in (
                "_sphinx_ai_assistant/tests/_hf_spaces_proxy",
                "_sphinx_feedback/tests/test_app.py",
                "_sphinx_feedback/tests/test_service.py",
            )
        ),
        # Each entry was added for a collection error, a failure or a skipped
        # module it removed (``markdownify``: 644 tests were skipped with
        # "build-time conversion needs markdownify").
        test_requires=(
            "pytest-asyncio",
            "pytest-regressions",
            "beautifulsoup4",
            "cryptography",
            "defusedxml",
            "fastapi",
            "httpx",
            "huggingface_hub",
            "markdownify",
            "pydata-sphinx-theme",
            "python-multipart",
            "pyyaml",
            "sphinx-design",
            "starlette",
            'tomli; python_version < "3.11"',
            "typing_extensions",
        ),
        # These modules import ``tests/_paths.py``, which raises "AI-assistant
        # tests require a repository/workspace root containing both
        # 'scikitplot/' and 'maintenances/'": they test the repository, and
        # an installed distribution has no ``maintenances/`` directory.
        test_ignore=tuple(
            "_externals/_sphinx_ext/_sphinx_ai_assistant/tests/" + path
            for path in (
                "_architecture/test_test_layout.py",
                "_hf_spaces_proxy/_utils/test__provider_artifact_lifecycle.py",
                "_hf_spaces_proxy/ci/test_run_redis_chaos.py",
                "_integration/test_bounded_remote_response.py",
                "_integration/test_logging_privacy.py",
                "_integration/test_release_security_boundary.py",
                "_integration/test_stub_mirror_proxy.py",
                "_integration/test_stub_mirror_security_inspector.py",
            )
        ),
        example=(
            "# In the conf.py of a Sphinx project, name an extension by its module:\n"
            "extensions = [\n"
            '    "scikitplot._externals._sphinx_ext._sphinx_gallery_grid",\n'
            "]\n"
            "\n"
            "# The extensions that are installed:\n"
            "from scikitplot._externals import _sphinx_ext\n"
            "\n"
            "print(sorted(_sphinx_ext._PRIVATE_SUBMODULES))\n"
        ),
    ),
    Package(
        distribution="scikit-plots-mcp",
        license="BSD-3-Clause",
        extras=("mcp",),
        siblings=(("corpus", ("scikit-plots-corpus",)),),
        # The protocol server is the part's "Tier S", which the part itself
        # defines as Python 3.10 and later (scikitplot/mcp/_capabilities.py).
        python_gated=(("scikitplot.mcp._server", ">=3.10"),),
        # The server tier is tested against the real SDK, asynchronously.
        test_extras=("mcp",),
        # The SDK (``mcp>=2``) exists for Python 3.10 and later only; below
        # that the SDK-free retrieval tier is what the part offers
        # (scikitplot/mcp/__init__.py).
        test_extras_python=">=3.10",
        test_requires=("pytest-asyncio",),
        keywords=(
            "scikit-plots",
            "mcp",
            "model context protocol",
            "documentation",
            "llm",
        ),
        example="scikitplot mcp --help\n",
        example_language="shell",
    ),
    Package(
        distribution="scikit-plots-cleanprompt",
        # scikitplot/cleanprompt ships LICENSE_cleanprompt (MIT).
        license="BSD-3-Clause AND MIT",
        extras=(
            "cleanprompt",
            "cleanprompt-ner",
            "cleanprompt-nltk",
            "cleanprompt-web",
            "cleanprompt-crypto",
        ),
        siblings=(("corpus", ("scikit-plots-corpus",)),),
        keywords=("scikit-plots", "pii", "redaction", "prompt", "llm", "privacy"),
        example="scikitplot cleanprompt --help\n",
        example_language="shell",
    ),
    Package(
        distribution="scikit-plots-cython",
        license="BSD-3-Clause",
        extras=("cython",),
        # Stated by the part itself (scikitplot/cython/_profiles.py): its
        # dataclasses use ``slots=True``, which needs Python 3.10. Measured:
        # ``import scikitplot.cython`` fails on 3.8 and 3.9.
        requires_python=">=3.10",
        # The builder tests compile real extension modules.
        test_extras=("cython",),
        keywords=("scikit-plots", "cython", "pybind11", "build"),
        example="import scikitplot.cython as skcython\n",
    ),
    Package(
        distribution="scikit-plots-mlflow",
        license="BSD-3-Clause",
        extras=("mlflow",),
        # Stated by the part itself (scikitplot/mlflow/_project.py): it reads
        # TOML with the standard library's ``tomllib``, which needs Python
        # 3.11. Measured: ``import scikitplot.mlflow`` fails on 3.8 to 3.10.
        requires_python=">=3.11",
        # Project files are read in YAML as well as TOML.
        test_requires=("pyyaml",),
        keywords=("scikit-plots", "mlflow", "mlops", "experiment tracking"),
        example="python -m scikitplot.mlflow --help\n",
        example_language="shell",
    ),
)


_PYTHON_FLOOR = re.compile(r"^>=(\d+)\.(\d+)$")
_PYTHON_VERSION = re.compile(r"^(\d+)\.(\d+)(?:\.\d+)*$")


def python_satisfies(specifier: str, version: str) -> bool:
    """
    Return whether a Python version satisfies a ``">=X.Y"`` requirement.

    Parameters
    ----------
    specifier : str
        A lower bound in the form ``">=X.Y"``, the only form this project
        uses for ``requires-python``.
    version : str
        A Python version, ``"X.Y"`` or ``"X.Y.Z"``.

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If ``specifier`` is not of the form ``">=X.Y"``, or ``version`` does
        not start with ``X.Y``. Any other form would need a full version
        specifier parser, and silently mis-reading one would decide whether a
        distribution is tested at all.
    """
    floor = _PYTHON_FLOOR.match(specifier.replace(" ", ""))
    if floor is None:
        raise ValueError(f"requires-python {specifier!r} is not of the form '>=X.Y'")
    release = _PYTHON_VERSION.match(version)
    if release is None:
        raise ValueError(f"{version!r} is not a Python version of the form 'X.Y'")
    return (int(release.group(1)), int(release.group(2))) >= (
        int(floor.group(1)),
        int(floor.group(2)),
    )


def directory_of(distribution: str) -> str:
    """
    Return the directory under ``libs/`` for a distribution name.

    Parameters
    ----------
    distribution : str
        Canonical project name starting with :data:`NAME_PREFIX`.

    Returns
    -------
    str
        The directory name, e.g. ``"rank-bm25"`` for
        ``"scikit-plots-rank-bm25"``.

    Raises
    ------
    ValueError
        If the name does not start with :data:`NAME_PREFIX` or nothing
        follows the prefix.
    """
    if not distribution.startswith(NAME_PREFIX) or len(distribution) == len(
        NAME_PREFIX
    ):
        raise ValueError(
            f"{distribution!r} is not of the form {NAME_PREFIX + '<directory>'!r}"
        )
    return distribution[len(NAME_PREFIX) :]


def by_distribution() -> dict[str, Package]:
    """
    Return the packages keyed by distribution name.

    Returns
    -------
    dict of str to Package
        In declaration order.

    Raises
    ------
    ValueError
        If a distribution is declared twice.
    """
    packages: dict[str, Package] = {}
    for package in PACKAGES:
        if package.distribution in packages:
            raise ValueError(f"{package.distribution} is declared twice in PACKAGES")
        packages[package.distribution] = package
    return packages
