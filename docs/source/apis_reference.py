"""Configuration for the APIs reference documentation."""


def _get_guide(*refs, is_developer=False):
    """
    Get the rst to refer to user/developer guide.

    `refs` is several references that can be used in the :ref:`...` directive.
    """
    if len(refs) == 1:
        ref_desc = f":ref:`{refs[0]}` section"
    elif len(refs) == 2:
        ref_desc = f":ref:`{refs[0]}` and :ref:`{refs[1]}` sections"
    else:
        ref_desc = ", ".join(f":ref:`{ref}`" for ref in refs[:-1])
        ref_desc += f", and :ref:`{refs[-1]}` sections"

    guide_name = "Developer" if is_developer else "User"
    return f"**{guide_name} guide.** See the {ref_desc} for further details."


def _get_submodule(module_name, submodule_name):
    """
    Get the submodule docstring and automatically add the hook.

    `module_name` is e.g. `sklearn.feature_extraction`, and `submodule_name` is e.g.
    `image`, so we get the docstring and hook for `sklearn.feature_extraction.image`
    submodule. `module_name` is used to reset the current module because autosummary
    automatically changes the current module.
    """
    lines = [
        f".. automodule:: {module_name}.{submodule_name}",
        f".. currentmodule:: {module_name}",
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
        "description": None,
        "sections": [
            {
                "title": (
                    "Configure global settings and get information about the working environment."
                ),
                "autosummary": [
                    "config_context",
                    "get_config",
                    "set_config",
                    "show_config",
                    "show_versions",
                    "online_help",
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
                    + "\n\n"
                    + _get_guide("decomposition-index")
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
                    + "\n\n"
                    + _get_guide("estimators-index")
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
                    + "\n\n"
                    + _get_guide("metrics-index")
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
                    # + "\n\n"
                    # + _get_guide("utils-index")
                ),
                "autosummary": [
                    # _helper
                    "_utils.validate_labels",
                    "_utils.cumulative_gain_curve",
                    "_utils.binary_ks_curve",
                    # validation
                    "_utils.validate_plotting_kwargs",
                    # figures
                    "_utils.save_figure",
                    "_utils.save_plot",
                ],
            },
        ],
    },
    "scikitplot.experimental": {
        "short_summary": "Experimental Python module based on C/CPP.",
        "description": _get_guide("experimental-index"),
        "sections": [
            {
                "title": "Cython Bindings samples",
                # "description": (
                #   _get_submodule("scikitplot.kds", "_kds")
                #   + "\n\n"
                #   + _get_guide("kds-index")
                # ),
                "autosummary": [
                    "_cy_experimental.expit",
                    "_cy_experimental.log_expit",
                    "_cy_experimental.logit",
                ],
            },
            {
                "title": "Pybind11 Bindings samples",
                # "description": (
                #   _get_submodule("scikitplot.kds", "_kds")
                #   + "\n\n"
                #   + _get_guide("kds-index")
                # ),
                "autosummary": [
                    "_py_experimental.py_print",
                ],
            },
            {
                "title": "Python samples",
                # "description": (
                #   _get_submodule("scikitplot.kds", "_kds")
                #   + "\n\n"
                #   + _get_guide("kds-index")
                # ),
                "autosummary": [
                    "_logsumexp.sigmoid",
                    "_logsumexp.softmax",
                    "_logsumexp.logsumexp",
                    "_logsumexp.log_softmax",
                ],
            },
        ],
    },
    "scikitplot.kds": {
        "short_summary": "KeyToDataScience",
        "description": _get_guide("kds-index"),
        "sections": [
            {
                "title": "Key To DataScience",
                # "description": (
                #   _get_submodule("scikitplot.kds", "_kds")
                #   + "\n\n"
                #   + _get_guide("kds-index")
                # ),
                "autosummary": [
                    "print_labels",
                    "decile_table",
                    "plot_cumulative_gain",
                    "plot_lift",
                    "plot_lift_decile_wise",
                    "plot_ks_statistic",
                    "report",
                ],
            },
        ],
    },
    "scikitplot.modelplotpy": {
        "short_summary": "Predictive model insights",
        "description": _get_guide("modelplotpy-index", "modelplotpy_financial-index"),
        "sections": [
            {
                "title": "Initializer ModelPlotPy object",
                # "description": (
                #   _get_submodule("scikitplot.modelplotpy", "_modelplotpy")
                #   + "\n\n"
                #   + _get_guide("modelplotpy-index")
                # ),
                "autosummary": [
                    # Initialize modelplotpy object
                    "ModelPlotPy",
                ],
            },
            {
                "title": "Gains, Lift and (cumulative) Response Plots",
                "description": None,
                "autosummary": [
                    # Gains, Lift and (cumulative) Response plots
                    "plot_response",
                    "plot_cumresponse",
                    "plot_cumlift",
                    "plot_cumgains",
                    "plot_all",
                ],
            },
            {
                "title": "Business-savvy Financial Insight Plots",
                "description": None,
                "autosummary": [
                    # Business-savvy financial plots
                    "plot_costsrevs",
                    "plot_profit",
                    "plot_roi",
                ],
            },
        ],
    },
    "scikitplot.probscale": {
        "short_summary": "Real probability scales for matplotlib.",
        "description": _get_guide("probscale-index"),
        "sections": [
            {
                "title": "probscale",
                # "description": (
                #   _get_submodule("scikitplot.probscale", "Probscale")
                #   + "\n\n"
                #   + _get_guide("probscale-index")
                # ),
                "autosummary": [
                    "ProbScale",
                    "probplot",
                    "plot_pos",
                    "fit_line",
                ],
            },
        ],
    },
    "scikitplot.sp_logging": {
        "short_summary": "Scikit-plots Logging.",
        "description": _get_guide("sp_logging-index"),
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
                "title": "Functional Interface - get_logger",
                # "description": (
                #   _get_submodule("scikitplot.sp_logging", "SpLogger")
                #   + "\n\n"
                #   + _get_guide("sp_logging-index")
                # ),
                "autosummary": [
                    "AlwaysStdErrHandler",
                    "GoogleLogFormatter",
                    "critical",
                    "debug",
                    "error",
                    "error_log",
                    "fatal",
                    "getEffectiveLevel",
                    "get_logger",
                    "log_if",
                    "setLevel",
                    "vlog",
                    "warn",
                    "warning",
                ],
            },
            {
                "title": "Class Interface - SpLogger",
                "description": None,
                "autosummary": [
                    "SpLogger",
                    "sp_logger",
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
                # "description": _get_submodule("scikitplot._astropy.stats", "bayesian_blocks"),
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
                "description": _get_submodule("scikitplot._astropy.stats", "funcs"),
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
                # "description": _get_submodule("scikitplot._astropy.stats", "histogram"),
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
                "description": _get_submodule(
                    "scikitplot._astropy.stats", "info_theory"
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
                "description": _get_submodule("scikitplot._tweedie", "_tweedie_dist"),
                "autosummary": [
                    "tweedie_gen",
                    "tweedie",
                ],
            },
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
            "Visualization of Neural Network Architectures Keras (either standalone or included in tensorflow)."
        ),
        "description": _get_guide("visualkeras-index"),
        "sections": [
            {
                "title": "Graphical Visualization",
                "description": _get_submodule("scikitplot.visualkeras", "graph"),
                "autosummary": [
                    "graph_view",
                ],
            },
            {
                "title": "Layered Visualization",
                "description": _get_submodule("scikitplot.visualkeras", "layered"),
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
