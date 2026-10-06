# libs/_tools/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tooling for the partial distributions of scikit-plots (``libs/<name>``).

Run it from the repository root::

    python -m libs._tools list
    python -m libs._tools generate
    python -m libs._tools check
    python -m libs._tools build
    python -m libs._tools verify

Modules
-------
staging
    Which files a distribution owns, and how they reach the build.
registry
    How each distribution is packaged.
generate
    Writes the packaging files from the two registries and the root project.
verify
    Builds every distribution and proves they install, alone and together.

Notes
-----
**Developer.** Nothing here imports ``scikitplot``. The tooling reads
``scikitplot/_distributions.py`` by file path, so it runs in an environment
where the package is not installed and cannot be built.
"""
