# scikitplot/cython/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
A small runtime inplace Cython devkit with caching, pinning, GC, and templates.

.. seealso::
  * https://doc.sagemath.org/html/en/reference/misc/sage/misc/cython.html
"""

from __future__ import annotations

from ._cache import CacheEntry, PackageCacheEntry
from ._public import (
    build_package_from_code,
    build_package_from_code_result,
    build_package_from_paths,
    build_package_from_paths_result,
    cache_stats,
    check_build_prereqs,
    compile_and_load,
    compile_and_load_result,
    cython_import,
    cython_import_all,
    cython_import_result,
    export_cached,
    gc_cache,
    get_cache_dir,
    import_artifact_bytes,
    import_artifact_path,
    import_cached,
    import_cached_by_name,
    import_cached_package,
    import_cached_package_result,
    import_cached_result,
    import_pinned,
    import_pinned_result,
    list_cached,
    list_cached_packages,
    list_pins,
    pin,
    purge_cache,
    register_cached_artifact_bytes,
    register_cached_artifact_path,
    unpin,
)
from ._result import (
    BuildResult,
    CacheGCResult,
    CacheStats,
    PackageBuildResult,
)
from ._templates_api import (
    TemplateInfo,
    build_package_example,
    build_package_example_result,
    compile_template,
    compile_template_result,
    copy_workflow,
    generate_sphinx_template_docs,
    get_package_example_path,
    get_template_path,
    get_workflow_path,
    list_package_examples,
    list_templates,
    list_workflows,
    load_package_example_metadata,
    load_template_metadata,
    read_template,
    read_template_info,
    template_root,
    workflow_cli_template_path,
)

__all__ = [  # noqa: RUF022
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
    # Cache management
    "get_cache_dir",
    "list_cached",
    "list_cached_packages",
    "cache_stats",
    "gc_cache",
    "purge_cache",
    # Pins/aliases
    "pin",
    "unpin",
    "list_pins",
    "import_pinned",
    "import_pinned_result",
    # Prereqs
    "check_build_prereqs",
    # Results
    "BuildResult",
    "PackageBuildResult",
    "CacheStats",
    "CacheGCResult",
    "CacheEntry",
    "PackageCacheEntry",
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
]
