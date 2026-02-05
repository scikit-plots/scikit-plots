# pylint: disable=pointless-string-statement

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the APIs reference documentation."""

import importlib
import inspect
import textwrap


def _get_guide(*refs, is_developer=False, title=None, capitalize=True):
    """
    Build a readable RST-style reference sentence for the User/Developer guide.

    Parameters
    ----------
    *refs : str
        One or more section reference names usable in :ref:`...` directives.
    is_developer : bool, default=False
        If True, refers to the Developer Guide; otherwise, the User Guide.
    title : str, optional
        Custom title for the guide (overrides automatic User/Developer title).
    capitalize : bool, default=True
        Whether to capitalize the guide title ("User guide" vs "user guide").

    Returns
    -------
    str
        A formatted sentence linking to the specified guide sections.
    """

    # Defensive validation — help catch silent issues
    if not refs:
        raise ValueError("At least one reference name must be provided.")
    if not all(isinstance(r, str) and r.strip() for r in refs):
        raise TypeError("All references must be non-empty strings.")

    # Handle custom or automatic guide name
    guide_name = title or ("Developer" if is_developer else "User")
    if not capitalize:
        guide_name = guide_name.lower()

    # Smart handling of plural/singular language
    num_refs = len(refs)
    if num_refs == 1:
        ref_desc = f":ref:`{refs[0]}` section"
    elif num_refs == 2:
        ref_desc = f":ref:`{refs[0]}` and :ref:`{refs[1]}` sections"
    else:
        # Oxford comma and final conjunction
        joined = ", ".join(f":ref:`{r}`" for r in refs[:-1])
        ref_desc = f"{joined}, and :ref:`{refs[-1]}` sections"

    # Optionally include punctuation and clarity markers
    return f"**{guide_name} guide.** See the {ref_desc} for further details."


def _get_submodule(
    module_name: str,
    submodule_name: str,
    *,
    include_docstring: bool = True,
    strict: bool = True,
    format: str = "rst",
) -> str:
    """
    Dynamically import a submodule and generate its Sphinx documentation directive.

    This is designed for use with Sphinx's autosummary/autodoc extensions,
    ensuring the current module context is properly restored after import.

    Parameters
    ----------
    module_name : str
        Parent module path (e.g. "sklearn.feature_extraction").
    submodule_name : str
        Submodule name (e.g. "image").
    include_docstring : bool, default=True
        Whether to include the submodule's top-level docstring in the output.
    strict : bool, default=False
        If True, re-raises all exceptions; otherwise, returns a warning comment
        when the submodule fails to import.
    format : {"rst", "markdown"}, default="rst"
        Output format. If "markdown", generates markdown-style references.

    Returns
    -------
    str
        A formatted text block for documentation inclusion.

    Examples
    --------
    >>> # Basic (RST, with docstring)
    >>> print(_get_submodule("sklearn.feature_extraction", "image"))

    >>> # Markdown, no docstring
    >>> print(_get_submodule("sklearn.feature_extraction", "image", include_docstring=False, format="markdown"))

    >>> # Graceful failure
    >>> print(_get_submodule("sklearn.nonexistent", "fake", strict=False))
    """
    full_name = f"{module_name}.{submodule_name}"
    # 1️⃣ Verify the submodule spec exists (exact, no guessing)
    spec = importlib.util.find_spec(full_name)
    if spec is None:
        msg = f"Submodule '{full_name}' not found (not importable)."
        if strict:
            raise ModuleNotFoundError(msg)
        return f".. warning:: {msg}"
    try:
        # 2️⃣ Import only after spec is verified
        # submod = __import__(full_name, fromlist=[""])
        submod = importlib.import_module(full_name)

        # 3️⃣ Verify the module actually corresponds to a real file
        if not spec.origin or spec.origin == "built-in":
            raise ValueError(f"Submodule '{full_name}' has no source file (built-in or namespace).")

        # 4️⃣ Verify docstring existence (exact presence check)
        doc = inspect.getdoc(submod) if include_docstring else None
        doc = textwrap.indent(doc or "No module docstring found.", "   ")
    except ImportError as e:
        if strict:
            raise ImportError(
                f"Failed to import submodule '{submodule_name}' from module '{module_name}'. "
                "Ensure the module is installed and available."
            ) from e
        msg = (
            f".. warning:: Failed to import submodule `{full_name}` — {type(e).__name__}: {e}"
            if format == "rst"
            else f"> **Warning:** Could not import `{full_name}` ({e})."
        )
        # return msg
    except ValueError as e:
        pass
    # Extract docstring if desired
    # doc = (submod.__doc__ or "").strip() if include_docstring else ""
    # if not doc:
    #     doc = (
    #         "*No module docstring found.*"
    #         if format == "markdown"
    #         else ".. note:: No module docstring found."
    #     )
    # # Choose formatting style
    # if format == "rst":
    #     parts = [
    #         f".. automodule:: {full_name}",
    #         f".. currentmodule:: {module_name}",
    #         # "",
    #         # f"   {doc.replace(chr(10), chr(10) + '   ')}",  # indent docstring for valid RST nesting
    #     ]
    #     return "\n".join(parts)
    # else:  # markdown mode
    #     return f"### `{full_name}`\n\n{doc}\n\n_Current module_: `{module_name}`\n"

    # 5️⃣ Output canonical Sphinx directive (no guessing)
    # .. automodule:: scikitplot.experimental._doremi
    # .. currentmodule:: scikitplot.experimental
    lines = [
        f".. automodule:: {full_name}",
        f".. currentmodule:: {module_name}\n",
        # f"{doc}\n"
    ]
    return "\n\n".join(lines)


"""
CONFIGURING APIS_REFERENCE
=========================

APIS_REFERENCE: dict[str, dict[str, any]]

APIS_REFERENCE maps each module name to a dictionary that consists of the following
components:

short_summary (required)
    The text to be printed on the index page; it has nothing to do the APIs reference
    page of each module.
description (required, `None` if not needed)
    The additional description for the module to be placed under the module
    docstring, before the sections start.
sections (required)
    A list of sections, each of which consists of:
    - title (required, `None` if not needed): the section title, commonly it should
      not be `None` except for the first section of a module,
    - description (optional): the optional additional description for the section,
    - autosummary (required): an autosummary block, assuming current module is the
      current module name.

Essentially, the rendered page would look like the following:

|---------------------------------------------------------------------------------|
|     {{ module_name }}                                                           |
|     =================                                                           |
|     {{ module_docstring }}                                                      |
|     {{ description }}                                                           |
|                                                                                 |
|     {{ section_title_1 }}   <-------------- Optional if one wants the first     |
|     ---------------------                   section to directly follow          |
|     {{ section_description_1 }}             without a second-level heading.     |
|     {{ section_autosummary_1 }}                                                 |
|                                                                                 |
|     {{ section_title_2 }}                                                       |
|     ---------------------                                                       |
|     {{ section_description_2 }}                                                 |
|     {{ section_autosummary_2 }}                                                 |
|                                                                                 |
|     More sections...                                                            |
|---------------------------------------------------------------------------------|

Hooks will be automatically generated for each module and each section. For a module,
e.g., `sklearn.feature_extraction`, the hook would be `feature_extraction_ref`; for a
section, e.g., "From text" under `sklearn.feature_extraction`, the hook would be
`feature_extraction_ref-from-text`. However, note that a better way is to refer using
the :mod: directive, e.g., :mod:`sklearn.feature_extraction` for the module and
:mod:`sklearn.feature_extraction.text` for the section. Only in case that a section
is not a particular submodule does the hook become useful, e.g., the "Loaders" section
under `sklearn.datasets`.
"""

APIS_REFERENCE: dict[str, dict[str, any]] = {
    "scikitplot": {
        "short_summary": "Settings and information visualization tools.",
        # "description": None,
        "sections": [
            {
                "title": (
                    "Version Tracking"
                ),
                # "description": None,
                "autosummary": [
                    # "__git_hash__",
                    "__version__",
                    "__version_iso_8601__",
                    "online_help",
                ],
            },
            {
                "title": (
                    "Configure global settings and get information about the working environment."
                ),
                # "description": None,
                "autosummary": [
                    "config_context",
                    "get_config",
                    "set_config",
                    "show_config",
                    "show_versions",
                ],
            },
            {
                "title": (
                    "Module-Level Globals"
                ),
                # "description": None,
                "autosummary": [
                    "get_logger",
                    "logger",
                ],
            },
        ],
    },
    "scikitplot.annoy": {
        "short_summary": "ANNoy (Approximate Nearest Neighbors Oh Yeah)",
        "description": _get_guide("annoy-index"),
        "sections": [
            {
                "title": "ANNoy (Approximate Nearest Neighbors Oh Yeah)",
                "description": None,
                "autosummary": [
                    # "get_include",
                    "Annoy",
                    "AnnoyIndex",
                    "Index",
                ],
            },
            {
                "title": "ANNoy :py:class:`~.scikitplot.annoy.Index` Mixins",
                "description": None,
                "autosummary": [
                    "CompressMode",
                    "IndexIOMixin",
                    "MetaMixin",
                    "NDArrayMixin",
                    "PickleMixin",
                    "PickleMode",
                    "PlottingMixin",
                    "VectorOpsMixin",
                ],
            },
        ],
    },
    "scikitplot.api": {
        "short_summary": "Functional Api for Visualizations (function-based, implicit)",
        "description": _get_guide("api-index"),
        "sections": [
            {
                "title": "Plot a PCA representation",
                "description": (
                    _get_submodule("scikitplot.api", "decomposition")
                    + "\n\n" +
                    _get_guide("decomposition-index")
                ),
                "autosummary": [
                    "decomposition.plot_pca_2d_projection",
                    "decomposition.plot_pca_component_variance",
                ],
            },
            {
                "title": "Plot Estimators (model) object instances",
                "description": (
                    _get_submodule("scikitplot.api", "estimators")
                    + "\n\n" +
                    _get_guide("estimators-index")
                ),
                "autosummary": [
                    # Regressor estimators
                    # Classifier estimators
                    "estimators.plot_feature_importances",
                    # Classifier models scalability
                    "estimators.plot_learning_curve",
                    # Cluster estimators
                    "estimators.plot_elbow",
                ],
            },
            {
                "title": "Plot model evaluation metrics",
                "description": (
                    _get_submodule("scikitplot.api", "metrics")
                    + "\n\n" +
                    _get_guide("metrics-index")
                ),
                "autosummary": [
                    # Regression metrics
                    "metrics.plot_residuals_distribution",
                    # Classification metrics
                    "metrics.plot_classifier_eval",
                    "metrics.plot_confusion_matrix",
                    "metrics.plot_precision_recall",
                    "metrics.plot_roc",
                    # Classification models scalability
                    "metrics.plot_calibration",
                    # Clustering metrics
                    "metrics.plot_silhouette",
                ],
            },
            {
                "title": "API Development Utilities",
                "description": (
                    _get_guide("developers-guide-index", is_developer=True)
                    # _get_submodule("scikitplot.api", "utils")
                    # + "\n\n" +
                    # _get_guide("utils-index")
                ),
                "autosummary": [
                    # _helper
                    "_utils.validate_labels",
                    "_utils.cumulative_gain_curve",
                    "_utils.binary_ks_curve",
                    # validation
                    "_utils.validate_plotting_kwargs",
                ],
            },
        ],
    },
    "scikitplot.cexperimental": {
        "short_summary": "C-Experimental Python modules based on C/CPP.",
        "description": _get_guide("cexperimental-index"),
        "sections": [
            {
                "title": "Cython Bindings samples",
                "description": (
                    _get_submodule("scikitplot.cexperimental", "_cy_cexperimental")
                    + "\n\n" +
                    _get_guide("cexperimental-index")
                ),
                "autosummary": [
                    # "_cy_cexperimental",
                    "_cy_cexperimental.expit",
                    "_cy_cexperimental.log_expit",
                    "_cy_cexperimental.logit",
                ],
            },
            {
                "title": "Pybind11 Bindings samples",
                "description": (
                    _get_submodule("scikitplot.cexperimental", "_py_cexperimental")
                    + "\n\n" +
                    _get_guide("cexperimental-index")
                ),
                "autosummary": [
                    # "_py_cexperimental",
                    "_py_cexperimental.py_print",
                ],
            },
            {
                "title": "Python samples",
                "description": (
                    _get_submodule("scikitplot.cexperimental", "_logsumexp")
                    + "\n\n" +
                    _get_guide("cexperimental-index")
                ),
                "autosummary": [
                    # "_logsumexp",
                    "_logsumexp.sigmoid",
                    "_logsumexp.softmax",
                    "_logsumexp.logsumexp",
                    "_logsumexp.log_softmax",
                ],
            },
        ],
    },
    "scikitplot.cexternals": {
        "short_summary": "C-Externals Python modules based on C/CPP.",
        "description": _get_guide("cexternals-index"),
        "sections": [
            {
                "title": "Spotify ANNoy as submodule.",
                "description": (
                    _get_submodule("scikitplot.cexternals", "_annoy")
                    + "\n\n" +
                    _get_guide("cexternals-annoy-index")
                ),
                "autosummary": [
                    "_annoy",
                    "_annoy.annoylib",
                    "_annoy.Annoy",
                    "_annoy.AnnoyIndex",
                ],
            },
            {
                "title": "astropy stats as submodule.",
                "description": (
                    _get_submodule("scikitplot.cexternals", "_astropy")
                    + "\n\n" +
                    _get_guide("cexternals-astropy-index")
                ),
                "autosummary": [
                    "_astropy",
                    "_astropy.stats",
                ],
            },
            {
                "title": "NumPy f2py as submodule.",
                "description": (
                    _get_submodule("scikitplot.cexternals", "_f2py")
                    + "\n\n" +
                    _get_guide("cexternals-f2py-index")
                ),
                "autosummary": [
                    "_f2py",
                    "_f2py.get_include",
                ],
            },
            {
                "title": "NumCpp header's as submodule.",
                "description": (
                    _get_submodule("scikitplot.cexternals", "_numcpp")
                    + "\n\n" +
                    _get_guide("cexternals-numcpp-index")
                ),
                "autosummary": [
                    "_numcpp",
                ],
            },
        ],
    },
    "scikitplot.config": {
        "short_summary": "config.",
        "description": None,
        "sections": [
            {
                "title": "Research and Citation Resources",
                "description": "Cite your source automatically (e.g., .bib, .cff, APA, MLA, IEEE, AMA, Chicago, and Harvard)",
                "autosummary": [
                    "__bibtex__",
                    "__citation__",
                ],
            },
        ],
    },
    "scikitplot.cython": {
        "short_summary": "A small runtime Cython devkit with caching, pinning, GC, and templates.",
        "description": _get_guide("cython-index"),
        "sections": [
            {
                "title": "Cython Realtime Inplace PKG/MOD Generation",
                # "description": (
                #     _get_submodule("scikitplot.cython", "__init__")
                # ),
                "autosummary": [
                    # Public compilation/import API
                    "compile_and_load",
                    "compile_and_load_result",
                    "cython_import",
                    "cython_import_result",
                    "cython_import_all",
                    "build_package_from_code",
                    "build_package_from_code_result",
                    "build_package_from_paths",
                    "build_package_from_paths_result",
                    "import_cached",
                    "import_cached_result",
                    "import_cached_by_name",
                    "import_cached_package",
                    "import_cached_package_result",
                    "register_cached_artifact_path",
                    "register_cached_artifact_bytes",
                    "import_artifact_path",
                    "import_artifact_bytes",
                    "export_cached",
                ],
            },
            {
                "title": "Cache management",
                # "description": (
                #     _get_submodule("scikitplot.cython", "__init__")
                # ),
                "autosummary": [
                    # Cache management
                    "get_cache_dir",
                    "list_cached",
                    "list_cached_packages",
                    "cache_stats",
                    "gc_cache",
                    "purge_cache",
                ],
            },
            {
                "title": "Pins/aliases",
                # "description": (
                #     _get_submodule("scikitplot.cython", "__init__")
                # ),
                "autosummary": [
                    # Pins/aliases
                    "pin",
                    "unpin",
                    "list_pins",
                    "import_pinned",
                    "import_pinned_result",
                ],
            },
            {
                "title": "Prereqs",
                # "description": (
                #     _get_submodule("scikitplot.cython", "__init__")
                # ),
                "autosummary": [
                    # Prereqs
                    "check_build_prereqs",
                ],
            },
            {
                "title": "Results",
                # "description": (
                #     _get_submodule("scikitplot.cython", "__init__")
                # ),
                "autosummary": [
                    # Results
                    "BuildResult",
                    "PackageBuildResult",
                    "CacheStats",
                    "CacheGCResult",
                    "CacheEntry",
                    "PackageCacheEntry",
                ],
            },
            {
                "title": "Templates / workflows",
                # "description": (
                #     _get_submodule("scikitplot.cython", "__init__")
                # ),
                "autosummary": [
                    # Templates / workflows
                    "TemplateInfo",
                    "template_root",
                    "list_templates",
                    "get_template_path",
                    "read_template",
                    "read_template_info",
                    "load_template_metadata",
                    "compile_template",
                    "compile_template_result",
                    "list_package_examples",
                    "get_package_example_path",
                    "load_package_example_metadata",
                    "build_package_example_result",
                    "build_package_example",
                    "list_workflows",
                    "get_workflow_path",
                    "workflow_cli_template_path",
                    "copy_workflow",
                    "generate_sphinx_template_docs",
                ],
            },
        ],
    },
    "scikitplot.datasets": {
        "short_summary": "Utilities to load popular datasets and artificial data generators.",
        # "description": _get_guide("datasets-index"),
        "sections": [
            {
                "title": "Load an example dataset",
                "description": (
                    _get_submodule("scikitplot.datasets", "_load_dataset")
                    # + "\n\n" +
                    # _get_guide("datasets-index")
                ),
                "autosummary": [
                    "get_data_home",
                    "get_dataset_names",
                    "load_dataset",
                ],
            },
        ],
    },
    "scikitplot.decile": {
        "short_summary": "Predictive model insights",
        "description": _get_guide("decile-index"),
        "sections": [
            {
                "title": "Key To DataScience: kds",
                "description": (
                    _get_submodule("scikitplot.decile", "kds")
                    + "\n\n" +
                    _get_guide("decile-kds-index")
                ),
                "autosummary": [
                    "kds.print_labels",
                    "kds.decile_table",
                    "kds.plot_cumulative_gain",
                    "kds.plot_lift",
                    "kds.plot_lift_decile_wise",
                    "kds.plot_ks_statistic",
                    "kds.report",
                ],
            },
            {
                "title": "ModelPlotPy Initializer object",
                "description": (
                    _get_submodule("scikitplot.decile", "modelplotpy")
                    # + "\n\n" +
                    # _get_guide("decile-modelplotpy-index", "decile-modelplotpy-financial-index")
                ),
                "autosummary": [
                    # Initialize modelplotpy object
                    "modelplotpy.ModelPlotPy",
                ],
            },
            {
                "title": "ModelPlotPy Initializer object",
                "description": (
                    _get_submodule("scikitplot.decile", "_decile_modelplotpy")
                    + "\n\n" +
                    _get_guide("decile-modelplotpy-index", "decile-modelplotpy-financial-index")
                ),
                "autosummary": [
                    # Initialize modelplotpy object
                    "ModelPlotPy",
                ],
            },
            {
                "title": "(Cumulative) Gains, Lift and Response Plots",
                # "description": None,
                "description": _get_guide("decile-modelplotpy-index"),
                "autosummary": [
                    # (Cumulative) Gains, Lift and Response Plots
                    "plot_response",
                    "plot_cumresponse",
                    "plot_cumlift",
                    "plot_cumgains",
                    "plot_all",
                ],
            },
            {
                "title": "Business-savvy Financial Insight Plots",
                "description": _get_guide("decile-modelplotpy-financial-index"),
                "autosummary": [
                    # Business-savvy financial plots
                    "plot_costsrevs",
                    "plot_profit",
                    "plot_roi",
                ],
            },
        ],
    },
    "scikitplot.exceptions": {
        "short_summary": "Exceptions and warnings.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "ModuleDeprecationWarning",
                    "ScikitplotException",
                    "VisibleDeprecationWarning",
                ],
            },
        ],
    },
    "scikitplot.experimental": {
        "short_summary": "Experimental tools/modules.",
        "description": _get_guide("experimental-index"),
        "sections": [
            {
                # "title": None,
                # "description": None,
                "description": (
                    _get_submodule("scikitplot.experimental", "__init__")
                    # + "\n\n" +
                    # _get_guide("experimental-index")
                ),
                "autosummary": [
                    "enable_ann_imputer",
                ],
            },
            {
                "title": "sklearn's pipeline.",
                "description": (
                    _get_submodule("scikitplot.experimental", "pipeline")
                    + "\n\n" +
                    _get_guide("pipeline-index")
                ),
                "autosummary": [
                    # "pipeline",
                    "pipeline.pipeline",
                ],
            },
            {
                "title": "Musical note handling, synthesis, and notation.",
                "description": (
                    _get_submodule("scikitplot.experimental", "_doremi")
                    + "\n\n" +
                    _get_guide("doremi-index")
                ),
                "autosummary": [
                    # "_doremi",
                    "_doremi.ENVELOPES",
                    "_doremi.compose_as_waveform",
                    "_doremi.play_waveform",
                    "_doremi.plot_waveform",
                    "_doremi.save_waveform",
                    "_doremi.save_waveform_as_mp3",
                    "_doremi.sheet_to_note",
                    "_doremi.sheet_converter",
                    "_doremi.serialize_sheet",
                    "_doremi.export_sheet",
                ],
            },
            {
                "title": "Large Language Models.",
                "description": (
                    _get_submodule("scikitplot.experimental", "_llm_provider")
                    + "\n\n" +
                    _get_guide("llm_provider-index")
                ),
                "autosummary": [
                    # "_llm_provider",
                    "_llm_provider.LLM_PROVIDER_CONFIG_MAP",
                    "_llm_provider.LLM_PROVIDER_ENV_CONNECTOR_MAP",
                    "_llm_provider.get_response",
                    "_llm_provider.load_mlflow_gateway_config",
                ],
            },
        ],
    },
    "scikitplot.externals": {
        "short_summary": "External Python modules.",
        "description": _get_guide("externals-index"),
        "sections": [
            {
                "title": "data-apis array_api_compat as submodule.",
                "description": (
                    _get_submodule("scikitplot.externals", "array_api_compat")
                    + "\n\n" +
                    _get_guide("externals-array_api_compat-index")
                ),
                "autosummary": [
                    "array_api_compat",
                ],
            },
            {
                "title": "data-apis array_api_extra as submodule.",
                "description": (
                    _get_submodule("scikitplot.externals", "array_api_extra")
                    + "\n\n" +
                    _get_guide("externals-array_api_extra-index")
                ),
                "autosummary": [
                    "array_api_extra",
                ],
            },
            {
                "title": "Real probability scales for matplotlib.",
                "description": (
                    _get_submodule("scikitplot.externals", "_probscale")
                    + "\n\n" +
                    _get_guide("externals-probscale-index")
                ),
                "autosummary": [
                    # "_probscale",
                    "_probscale.ProbScale",
                    "_probscale.probplot",
                    "_probscale.plot_pos",
                    "_probscale.fit_line",
                ],
            },
            {
                "title": "Seaborn as submodule.",
                "description": (
                    _get_submodule("scikitplot.externals", "_seaborn")
                    + "\n\n" +
                    _get_guide("externals-seaborn-index")
                ),
                "autosummary": [
                    "_seaborn",
                ],
            },
            {
                "title": "Matplotlib Sphinxext Ext.",
                "description": (
                    _get_submodule("scikitplot.externals", "_sphinxext")
                    + "\n\n" +
                    _get_guide("externals-sphinxext-index")
                ),
                "autosummary": [
                    "_sphinxext",
                    "_sphinxext.figmpl_directive",
                    "_sphinxext.mathmpl",
                    "_sphinxext.plot_directive",
                    "_sphinxext.roles",
                    "_sphinxext.sphinx_tabs_patch",
                ],
            },
            {
                "title": "Tweedie Family.",
                "description": (
                    _get_submodule("scikitplot.externals", "_tweedie")
                    + "\n\n" +
                    _get_guide("externals-tweedie-index")
                ),
                "autosummary": [
                    "_tweedie",
                    "_tweedie.tweedie",
                    "_tweedie.tweedie_gen",
                ],
            },
        ],
    },
    "scikitplot.impute": {
        "short_summary": "Missing value imputation.",
        "description": _get_guide("impute-index"),
        "sections": [
            {
                "title": "Approximate K-nearest-neighbours (KNN) imputation.",
                "description": (
                    _get_submodule("scikitplot.impute", "_ann")
                    + "\n\n" +
                    _get_guide("ann_imputer-index")
                ),
                "autosummary": [
                    "_ann.ANNImputer",
                ],
            },
        ],
    },
    "scikitplot.logging": {
        "short_summary": "Scikit-plots Logging.",
        "description": _get_guide("logging-index"),
        "sections": [
            {
                "title": "Logging Levels",
                "description": None,
                "autosummary": [
                    "CRITICAL",
                    "DEBUG",
                    "ERROR",
                    "FATAL",
                    "INFO",
                    "NOTSET",
                    "WARN",
                    "WARNING",
                ],
            },
            {
                "title": "Logging Helpers",
                "description": None,
                "autosummary": [
                    "AlwaysStdErrHandler",
                    "GoogleLogFormatter",
                    "getEffectiveLevel",
                    "get_logger",
                    "get_verbosity",
                    "setLevel",
                    "set_verbosity",
                ],
            },
            {
                "title": "Log the Message",
                "description": "Log the message at a predefined logging level.",
                "autosummary": [
                    "log",
                    "log_every_n",
                    "log_first_n",
                    "log_if",
                    "vlog",
                    "debug",
                    "info",
                    "warning",
                    "warn",
                    "error",
                    "error_log",
                    "exception",
                    "TaskLevelStatusMessage",
                    "critical",
                    "fatal",
                ],
            },
        ],
    },
    "scikitplot.memmap": {
        "short_summary": "Cross-platform memory mapping module.",
        "description": _get_guide("memmap-index"),
        "sections": [
            {
                "title": "MemMap: file-backed or anonymous memory mapping.",
                "description": (
                    # _get_submodule("scikitplot.memmap", "_memmap.mem_map")
                    # + "\n\n" +
                    _get_guide("memmap-index")
                ),
                "autosummary": [
                    'MemoryMap',
                    'mmap_region',
                ],
            },
        ],
    },
    "scikitplot.mlflow": {
        "short_summary": "Missing value imputation.",
        "description": _get_guide("mlflow-index"),
        "sections": [
            {
                "title": "Mlflow Session Helper",
                "description": (
                    _get_submodule("scikitplot.mlflow", "_session")
                    + "\n\n" +
                    _get_guide("mlflow-index")
                ),
                "autosummary": [
                    "MlflowHandle",
                    "session",
                    "session_from_file",
                    "session_from_toml",
                ],
            },
            {
                "title": "project config helpers",
                "description": (
                    _get_submodule("scikitplot.mlflow", "_project")
                    + "\n\n" +
                    _get_guide("mlflow-index")
                ),
                "autosummary": [
                    "DEFAULT_PROJECT_MARKERS",
                    "find_project_root",
                    "load_project_config",
                    "load_project_config_toml",
                    "dump_project_config_yaml",
                ],
            },
            {
                "title": "config",
                "description": (
                    _get_submodule("scikitplot.mlflow", "_config")
                    + "\n\n" +
                    _get_guide("mlflow-index")
                ),
                "autosummary": [
                    "ServerConfig",
                    "SessionConfig",
                    "ProjectConfig",
                ],
            },
            {
                "title": "errors",
                "description": None,
                "autosummary": [
                    "MlflowIntegrationError",
                    "MlflowNotInstalledError",
                    "MlflowCliIncompatibleError",
                    "MlflowServerStartError",
                ],
            },
            {
                "title": "workflow helpers",
                "description": (
                    _get_submodule("scikitplot.mlflow", "_workflow")
                    + "\n\n" +
                    _get_guide("mlflow-index")
                ),
                "autosummary": [
                    "workflow",
                    "builtin_config_path",
                    "export_builtin_config",
                    "default_project_paths",
                ],
            },
        ],
    },
    "scikitplot.nc": {
        "short_summary": "High-performance Numerical Functions",
        "description": _get_guide("nc-index"),
        "sections": [
            {
                "title": "NumCpp Library Header",
                "description": None,
                "autosummary": [
                    "get_include",
                ],
            },
            {
                "title": "Linear Algebra Functions",
                "description": None,
                "autosummary": [
                    "dot",
                ],
            },
        ],
    },
    "scikitplot.preprocessing": {
        "short_summary": "Extended sklearn preprocessing.",
        "description": _get_guide("preprocessing-index"),
        "sections": [
            {
                "title": "Extended sklearn feature preprocessing.",
                # "description": (
                #     _get_submodule("scikitplot.preprocessing", "_get_dummies")
                #     + "\n\n" +
                #     _get_guide("get_dummies-index")
                # ),
                "autosummary": [
                    "DummyCodeEncoder",
                    "GetDummies",
                ],
            },
        ],
    },
    "scikitplot.random": {
        "short_summary": "Random Number Generation (Numpy-Like :class:`~numpy.random.Generator`).",
        "description": _get_guide("random-index"),
        "sections": [
            {
                "title": "Random Number Generation (Numpy-Like :class:`~numpy.random.Generator`).",
                "description": (
                    # _get_submodule("scikitplot.random", "_kiss.kiss_random")
                    # + "\n\n" +
                    _get_guide("random-index")
                ),
                "autosummary": [
                    'Kiss32Random',
                    'Kiss64Random',
                    'KissRandom',
                    'KissSeedSequence',
                    'KissBitGenerator',
                    'KissGenerator',
                    'KissRandomState',
                    'default_rng',
                    'kiss_context',
                ],
            },
            {
                "title": "``KissGenerator`` Distribution Methods",
                "description": (
                    # _get_submodule("scikitplot.random", "__init__")
                    # + "\n\n" +
                    _get_guide("random-index")
                ),
                "autosummary": [
                    # KissGenerator Distribution Methods
                    'choice',
                    'integers',
                    'normal',
                    'permutation',
                    'random',
                    'shuffle',
                    'uniform',
                ],
            },
        ],
    },
    "scikitplot.seaborn": {
        "short_summary": "Extended Seaborn",
        "description": _get_guide("seaborn-index"),
        "sections": [
            {
                "title": ".scikitplot via Seaborn",
                # "description": None,
                # "description": (
                #     _get_submodule("scikitplot.seaborn", "_auc")
                #     + "\n\n" +
                #     _get_guide("auc-index")
                # ),
                "description": _get_guide("aucplot-index", "evalplot-index"),
                "autosummary": [
                    "aucplot",
                    "evalplot",
                ],
            },
            {
                "title": ".kds to SeabornX",
                "description": (
                    _get_submodule("scikitplot.seaborn", "_decile")
                    + "\n\n" +
                    _get_guide("decileplot-index")
                ),
                "autosummary": [
                    "decileplot",
                    "print_labels",
                ],
            },
        ],
    },
    "scikitplot.stats": {
        "short_summary": "Statistical tools for data visualization and interpretation.",
        "description": _get_guide("stats-index"),
        "sections": [
            {
                "title": "Astrostatistics: Bayesian Blocks for Time Series Analysis",
                "description": (
                    # _get_submodule("scikitplot.cexternals", "_astropy")
                    # + "\n\n" +
                    _get_guide("astrostatistics-index")
                ),
                "autosummary": [
                    "Events",
                    "FitnessFunc",
                    "PointMeasures",
                    "RegularEvents",
                    "bayesian_blocks",
                ],
            },
            {
                "title": "Astrostatistics Tools",
                "description": (
                    _get_submodule("scikitplot.cexternals._astropy.stats", "funcs")
                    + "\n\n" +
                    _get_guide("astrostatistics-index")
                ),
                "autosummary": [
                    "binned_binom_proportion",
                    "binom_conf_interval",
                    "bootstrap",
                    "cdf_from_intervals",
                    "fold_intervals",
                    "gaussian_fwhm_to_sigma",
                    "gaussian_sigma_to_fwhm",
                    "histogram_intervals",
                    "interval_overlap_length",
                    "kuiper",
                    "kuiper_false_positive_probability",
                    "kuiper_two",
                    "mad_std",
                    "median_absolute_deviation",
                    "poisson_conf_interval",
                    "signal_to_noise_oir_ccd",
                ],
            },
            {
                "title": "Astrostatistics: Selecting the bin width of histograms",
                # "description": (
                #     _get_submodule("scikitplot.cexternals._astropy.stats", "histogram")
                #     + "\n\n" +
                #     _get_guide("astrostatistics-index")
                # ),
                "autosummary": [
                    "calculate_bin_edges",
                    "freedman_bin_width",
                    "histogram",
                    "knuth_bin_width",
                    "scott_bin_width",
                ],
            },
            {
                "title": "Astrostatistics: Model Selection",
                "description": (
                    _get_submodule("scikitplot.cexternals._astropy.stats", "info_theory")
                    + "\n\n" +
                    _get_guide("astrostatistics-index")
                ),
                "autosummary": [
                    "akaike_info_criterion",
                    "akaike_info_criterion_lsq",
                    "bayesian_info_criterion",
                    "bayesian_info_criterion_lsq",
                ],
            },
            {
                "title": "Discrete Distributions Tools",
                "description": (
                    _get_submodule("scikitplot.externals._tweedie", "_tweedie_dist")
                    + "\n\n" +
                    _get_guide("tweedie-dist-index")
                ),
                "autosummary": [
                    "tweedie_gen",
                    "tweedie",
                ],
            },
        ],
    },
    "scikitplot.utils": {
      "short_summary": "Development Utilities",
      "description": (
        _get_guide("developers-guide-index", is_developer=True)
      ),
      "sections": [
        {
          "title": "Time Utilities",
          "description": _get_submodule("scikitplot.utils", "_time"),
          "autosummary": [
            '_time.Timer',
          ],
        },
        {
          "title": "File/Folder Utilities",
          "description": _get_submodule("scikitplot.utils", "_path"),
          "autosummary": [
            '_path.PathNamer',
            '_path.make_path',
          ],
        },
        # {
        #   "title": "Optimal matplotlib operations",
        #   "description": _get_submodule("scikitplot.utils", "_figures"),
        #   "autosummary": [
        #     '_figures.save_figure',
        #   ],
        # },
        # {
        #   "title": "Optimal mathematical operations",
        #   "description": _get_submodule("scikitplot.utils", "_helpers"),
        #   "autosummary": [
        #     '_helpers.validate_labels',
        #     '_helpers.cumulative_gain_curve',
        #     '_helpers.binary_ks_curve',
        #   ],
        # },
        # {
        #   "title": "Input and parameter validation",
        #   "description": _get_submodule("scikitplot.utils", "validation"),
        #   "autosummary": [
        #     'validation.validate_plotting_kwargs',
        #   ],
        # },
      ],
    },
    # "scikitplot.utils": {
    #   "short_summary": "Development Utilities",
    #   "description": (
    #     _get_guide("developers-guide-index", is_developer=True)
    #   ),
    #   "sections": [
    #     {
    #       "title": "Optimal matplotlib operations",
    #       "description": _get_submodule("scikitplot.utils", "_figures"),
    #       "autosummary": [
    #         '_figures.save_figure',
    #       ],
    #     },
    #     {
    #       "title": "Optimal mathematical operations",
    #       "description": _get_submodule("scikitplot.utils", "_helpers"),
    #       "autosummary": [
    #         '_helpers.validate_labels',
    #         '_helpers.cumulative_gain_curve',
    #         '_helpers.binary_ks_curve',
    #       ],
    #     },
    #     {
    #       "title": "Input and parameter validation",
    #       "description": _get_submodule("scikitplot.utils", "validation"),
    #       "autosummary": [
    #         'validation.validate_plotting_kwargs',
    #       ],
    #     },
    #   ],
    # },
    "scikitplot.visualkeras": {
        "short_summary": (
            "Visualization of Neural Network Architectures Keras "
            "(either standalone or included in tensorflow)."
        ),
        "description": _get_guide("visualkeras-index"),
        "sections": [
            {
                "title": "Graphical Visualization",
                "description": _get_submodule("scikitplot.visualkeras", "_graph"),
                "autosummary": [
                    "graph_view",
                ],
            },
            {
                "title": "Layered Visualization",
                "description": _get_submodule("scikitplot.visualkeras", "_layered"),
                "autosummary": [
                    "layered_view",
                ],
            },
            {
                "title": "Visualization Helper",
                "description": None,
                "autosummary": [
                    "SpacingDummyLayer",
                ],
            },
        ],
    },
}


"""
CONFIGURING DEPRECATED_APIS_REFERENCE
====================================

DEPRECATED_APIS_REFERENCE: dict[str, list[str]]

DEPRECATED_APIS_REFERENCE maps each deprecation target version to a corresponding
autosummary block. It will be placed at the bottom of the APIs index page under the
"Recently deprecated" section. Essentially, the rendered section would look like the
following:

|------------------------------------------|
|     To be removed in {{ version_1 }}     |
|     --------------------------------     |
|     {{ autosummary_1 }}                  |
|                                          |
|     To be removed in {{ version_2 }}     |
|     --------------------------------     |
|     {{ autosummary_2 }}                  |
|                                          |
|     More versions...                     |
|------------------------------------------|

Note that the autosummary here assumes that the current module is `sklearn`, i.e., if
`sklearn.utils.Memory` is deprecated, one should put `utils.Memory` in the "entries"
slot of the autosummary block.

Example:

DEPRECATED_APIS_REFERENCE = {
    "0.24": [
        "model_selection.fit_grid_point",
        "utils.safe_indexing",
    ],
}
"""

DEPRECATED_APIS_REFERENCE: dict[str, list[str]] = {
    "0.5": [
        "_factory_api",  # Visualizations (object-based, explicit)
        "api.metrics.plot_roc_curve",
        "api.metrics.plot_precision_recall_curve",
    ]
}  # type: ignore
