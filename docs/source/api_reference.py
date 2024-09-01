"""
Configuration for the API reference documentation.
"""


def _get_guide(*refs, is_developer=False):
    """Get the rst to refer to user/developer guide.

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
    """Get the submodule docstring and automatically add the hook.

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
CONFIGURING API_REFERENCE
=========================

API_REFERENCE maps each module name to a dictionary that consists of the following
components:

short_summary (required)
    The text to be printed on the index page; it has nothing to do the API reference
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

API_REFERENCE = {
    "scikitplot": {
        "short_summary": "Settings and information visualization tools.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "show_versions",
                ],
            },
        ],
    },
    "scikitplot.cluster": {
        "short_summary": "Clustering.",
        "description": _get_guide("clustering"),
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "plot_elbow",
                ],
            },
        ],
    },
    "scikitplot.deciles": {
        "short_summary": "Deciles.",
        "description": _get_guide("deciles"),
        "sections": [
            {
                "title": None,
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
    "scikitplot.decomposition": {
        "short_summary": "Decomposition.",
        "description": _get_guide("decomposition"),
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "plot_pca_component_variance", 
                    "plot_pca_2d_projection",
                ],
            },
        ],
    },
    "scikitplot.estimators": {
        "short_summary": "Visualizations for model’s decision-making process.",
        "description": _get_guide("estimators"),
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "plot_feature_importances",
                    "plot_learning_curve",
                ],
            },
        ],
    },
    "scikitplot.metrics": {
        "short_summary": "Visualizations for model performance metrics.",
        "description": _get_guide("metrics"),
        "sections": [
            {
                "title": "Model selection interface",
                "description": _get_guide("metrics"),
                "autosummary": [
                    "plot_calibration_curve",
                ],
            },
            {
                "title": "Classification metrics",
                "description": _get_guide("metrics"),
                "autosummary": [
                    "plot_classifier_eval",
                    "plot_confusion_matrix",
                    "plot_roc",
                    "plot_precision_recall",
                ],
            },
            {
                "title": "Clustering metrics",
                "description": _get_guide("metrics"),
                "autosummary": [
                    "plot_silhouette",
                ],
            },
        ],
    },
    "scikitplot.utils": {
        "short_summary": "Utilities.",
        "description": _get_guide("developers-utils", is_developer=True),
        "sections": [
            {
                "title": "Optimal mathematical operations",
                "description": _get_submodule("scikitplot.utils", "helpers"),
                "autosummary": [
                    "helpers.combine_and_save_figures",
                    "helpers.validate_labels",
                    "helpers.cumulative_gain_curve",
                    "helpers.binary_ks_curve",
                    "helpers.sigmoid",
                    "helpers.softmax",
                ],
            },
        ],
    },
}


"""
CONFIGURING DEPRECATED_API_REFERENCE
====================================

DEPRECATED_API_REFERENCE maps each deprecation target version to a corresponding
autosummary block. It will be placed at the bottom of the API index page under the
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

DEPRECATED_API_REFERENCE = {
    "0.24": [
        "model_selection.fit_grid_point",
        "utils.safe_indexing",
    ],
}
"""

DEPRECATED_API_REFERENCE = {
    "0.4": [
        "metrics.plot_roc_curve",
        "metrics.plot_precision_recall_curve",
        "classifiers",
        "clustering",
        "plotters",
    ]
}  # type: ignore