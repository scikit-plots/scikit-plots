.. _cython-packages-templates-workflows:

Packages, templates, and workflows
==================================

For multiple extension modules, use ``build_package_from_code`` or
``build_package_from_paths`` instead of independently compiling modules that
need to share one package namespace.

Templates
---------

The template API validates metadata and exposes shipped examples through
``list_templates``, ``read_template``, ``compile_template`` and package-example
helpers. Template files are examples of build inputs, not privileged code:
review them before compiling after local modification.

Workflows
---------

``list_workflows`` and ``copy_workflow`` provide reproducible starting assets.
Copying a workflow intentionally creates user-owned files; future package
updates do not mutate that copy automatically.

See :doc:`_templates/templates_index` for the detailed shipped template
reference.
