..
    .. contents::
       :local:
       :depth: 2

======================================================================
How to Add ``scikit-plots`` to conda-forge Guidelines
======================================================================

This guide provides a **comprehensive, beginner-friendly walkthrough** for adding ``scikit-plots`` to conda-forge, including best practices, important notes, and maintenance instructions.

Prerequisites
=============

Ensure you have:

- A GitHub account.
- ``scikit-plots`` published one of:
   - ``github`` like `GitHub <https://github.com/scikit-plots/scikit-plots>`_
   - ``github release`` like `GitHub Release <https://github.com/scikit-plots/scikit-plots/releases>`_
   - ``source distribution`` like `PyPI <https://pypi.org/project/scikit-plots>`_
   - ``source distribution`` like `Anaconda.org <https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots>`_
- Installed tools: ``git``, ``conda``, and ``grayskull``.

.. important::
   Your package must be available via **a valid source distribution** like GitHub, PyPI, or Anaconda.org before it can be added to conda-forge.
   But can be use direct ``github repo``, read below more info.

.. hint::
   Install ``grayskull`` easily:

   .. code-block:: bash

      conda install -c conda-forge grayskull

   or:

   .. code-block:: bash

      pip install grayskull

.. _choosing-a-source-distribution:

Choosing a Source Distribution
==============================

When updating the recipe (`meta.yaml`), you must choose **one** of the valid source distribution types. Conda-forge supports four typical options:

1. **GitHub archive (tagged release)**

   .. code-block:: yaml

      source:
        url: https://github.com/scikit-plots/scikit-plots/archive/v{{ version }}.tar.gz
        sha256: <fill-this-sha>

   .. important::
      - Ensure the tag exists on GitHub.
      - Use a specific release version to avoid build inconsistencies.

2. **PyPI source distribution**

   .. code-block:: yaml

      source:
        url: https://pypi.org/packages/source/s/scikit-plots/scikit_plots-{{ version }}.tar.gz
        sha256: <fill-this-sha>

   .. note::
      This is the most common and preferred method when available on PyPI.

3. **Anaconda.org source (e.g., staging wheels)**

   .. code-block:: yaml

      source:
        url: https://pypi.anaconda.org/<channel>/simple/scikit-plots/{{ version }}/scikit_plots-{{ version }}.tar.gz
        sha256: <fill-this-sha>

   .. hint::
      Replace ``<channel>`` with your actual Anaconda channel, such as ``scikit-plots-wheels-staging-nightly``.

4. **Direct Git repository source**

   .. code-block:: yaml

      source:
        git_url: https://github.com/scikit-plots/scikit-plots.git
        git_rev: {{ tag }}  # 🔒 Use a tag or commit hash for reproducibility

   .. important::
      - Always use a **tag** or **commit hash**, not ``main`` or ``master``.
      - This method is less common and should only be used when source tarballs are not available.

.. hint::
   Use `grayskull pypi scikit-plots` to auto-generate a `meta.yaml` based on your latest PyPI release, then modify the `source` field if needed.

.. _publishing-to-pypi:

Publishing to PyPI
==================

If you use PyPI:

Before proceeding, confirm that ``scikit-plots`` is correctly published on PyPI.

.. important::
   **Conda-forge** builds the package **from PyPI releases**, not from GitHub directly (unless you configure otherwise in your recipe).

.. hint::
   If you need help publishing, check out the official
   `PyPI packaging tutorial <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_.

.. _creating-the-recipe-with-grayskull:

Creating the Recipe with Grayskull
==================================

Use ``grayskull`` to generate the initial conda recipe:

.. code-block:: bash

   grayskull pypi --strict-conda-forge scikit-plots

This generates a ``recipes/scikit-plots/`` folder containing a ``meta.yaml``.

.. note::
   ``grayskull`` will automatically pull metadata from your PyPI release — **but you must review it**.

.. hint::
   After generation:
   - Check license information.
   - Verify dependencies (`requirements` section).
   - Correct any missing classifiers or Python version constraints.

.. code-block:: jinja

   {% set name = "scikit-plots" %}
   {% set version = "0.4.0rc4" %}
   {% set tag = "v" ~ version %}

   package:
     name: {{ name|lower }}
     version: {{ version }}

   source:
     # Choose one valid source:

     # GitHub
     git_url: https://github.com/scikit-plots/scikit-plots.git
     git_rev: {{ tag }}       # 🔒 use a tag or commit hash for reproducibility

     # GitHub archive
     # url: https://github.com/{{ name }}/{{ name }}/archive/{{ tag }}.tar.gz

     # PyPI source
     # url: https://pypi.org/packages/source/{{ name[0] }}/{{ name }}/{{ name | replace("-", "_") }}-{{ version }}.tar.gz

     # Anaconda source (less common)
     # url: https://pypi.anaconda.org/scikit-plots-wheels-staging-nightly/simple/{{ name }}/{{ version }}/{{ name | replace("-", "_") }}-{{ version }}.tar.gz

     url: https://pypi.org/packages/source/{{ name[0] }}/{{ name }}/scikit_plots-{{ version }}.tar.gz
     sha256: cd6c8a3d11cfe0b9cc3e4ecc95399efe16ea242ddb4c02505031c6271f8876f8

.. note::
   You **must include** the correct ``sha256`` checksum for the chosen source archive.
   To compute it:

   .. code-block:: bash

      ## wget https://github.com/scikit-plots/scikit-plots/archive/refs/tags/{{ tag }}.tar.gz
      wget https://github.com/scikit-plots/scikit-plots/archive/refs/tags/v0.4.0rc4.tar.gz

      ## openssl sha256 {{ tag }}.tar.gz
      openssl sha256 v0.4.0rc4.tar.gz

.. _submitting-to-staged-recipes:

Submitting to Staged-Recipes
============================

1. Fork the `conda-forge/staged-recipes <https://github.com/conda-forge/staged-recipes>`_ repository.
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/<your-github-username>/staged-recipes.git
      cd staged-recipes/recipes/

3. Create a new branch:

   .. code-block:: bash

      git checkout -b add-scikit-plots

4. Add your recipe directory and license file.

5. Commit and push your changes:

   .. code-block:: bash

      git add scikit-plots/
      git commit -m "Add scikit-plots recipe"
      git push origin add-scikit-plots

6. Open a Pull Request (PR) to ``conda-forge/staged-recipes``.

.. important::
   Carefully fill out the PR checklist in the pull request description!

   - Confirm the recipe builds locally with ``conda build`` (optional but strongly recommended).
   - Confirm your metadata is accurate.
   - Ping reviewers if needed (`@conda-forge/help-python`).

.. hint::
   Your PR title should follow the format: ``Add package: scikit-plots``.

.. _post-merge-feedstock-creation:

Post-Merge: Feedstock Creation
==============================

After your PR is merged:

- A new feedstock repository will be created, for example: `scikit-plots-feedstock <https://github.com/conda-forge/scikit-plots-feedstock>`_.
- You will be added as a **maintainer**.
- CI (Continuous Integration) will build and upload the package across all platforms.

.. important::
   Carefully watch the CI builds!
   Build errors may still appear even after the staged-recipes PR is merged.

.. hint::
   Add the feedstock repo to your GitHub notifications (watch → participating) to stay informed!

.. _maintaining-the-package:

Maintaining the Package
=======================

Future updates for ``scikit-plots``:

- Publish a new release to PyPI.
- The **conda-forge bot** will open a PR automatically to update the feedstock recipe.

.. hint::
   Regularly check your feedstock repository for pending bot PRs!

Manual Updates
==============

If the bot **fails** to update your package or if you need to make manual changes:

1. Fork and clone your feedstock repository:

   .. code-block:: bash

      git clone https://github.com/<your-github-username>/scikit-plots-feedstock.git
      cd scikit-plots-feedstock/recipe/

2. Update ``meta.yaml`` manually (or use ``grayskull`` again).

3. Commit and push your changes:

   .. code-block:: bash

      git add recipe/meta.yaml
      git commit -m "Update scikit-plots to version X.Y.Z"
      git push origin update-scikit-plots

4. Open a pull request back to the feedstock repository.

5. Wait for CI builds to pass, then merge.

.. important::
   Always **let CI finish** before merging. Merging broken recipes can cause major issues across conda-forge!

Congratulations 🎉
==================

After the update PR is merged:

- The new version of ``scikit-plots`` will be built and uploaded automatically.
- Users will be able to install the latest version with:

.. code-block:: bash

   conda install -c conda-forge scikit-plots

You are now officially maintaining a package on conda-forge!

Quick Reference Summary
=======================

Follow these steps to successfully add and maintain ``scikit-plots`` on conda-forge.

Steps Overview
--------------

1. :ref:`Choose source distribution <choosing-a-source-distribution>`
2. :ref:`Publish to PyPI (if needed) <publishing-to-pypi>`
3. :ref:`Generate recipe with grayskull <creating-the-recipe-with-grayskull>`
4. :ref:`Submit PR to staged-recipes <submitting-to-staged-recipes>`
5. :ref:`Wait for review & merge <post-merge-feedstock-creation>`
6. :ref:`Feedstock repository created <post-merge-feedstock-creation>`
7. :ref:`Maintain future updates (bot/manual) <maintaining-the-package>`

.. hint::
   You can click on any action to jump directly to the detailed explanation!
