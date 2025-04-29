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
- ``scikit-plots`` published on `PyPI <https://pypi.org/project/scikit-plots/>`_.
- Installed tools: ``git``, ``conda``, and ``grayskull``.

.. important::
   Your package **must** be on PyPI before you can add it to conda-forge!

.. hint::
   Install ``grayskull`` easily:

   .. code-block:: bash

      conda install -c conda-forge grayskull

   or:

   .. code-block:: bash

      pip install grayskull

Publishing to PyPI
==================

Before proceeding, confirm that ``scikit-plots`` is correctly published on PyPI.

.. important::
   **Conda-forge** builds the package **from PyPI releases**, **not** from GitHub source directly.

.. hint::
   If you need help publishing, check out the official `PyPI packaging tutorial <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_.

Creating the Recipe with Grayskull
==================================

Use ``grayskull`` to generate the initial conda recipe:

.. code-block:: bash

   grayskull pypi --strict-conda-forge scikit-plots

This generates a ``recipes/scikit-plots/`` folder containing a ``meta.yaml``.

.. note::
   ``grayskull`` will automatically pull metadata from your PyPI release — **but you must review it**.

.. suggestion::
   After generation:
   - Check license information.
   - Verify dependencies (`requirements` section).
   - Correct any missing classifiers or Python version constraints.

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

Maintaining the Package
=======================

Future updates for ``scikit-plots``:

- Publish a new release to PyPI.
- The **conda-forge bot** will open a PR automatically to update the feedstock recipe.

.. suggestion::
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

+------+----------------------------------------------------+
| Step | Action                                              |
+======+====================================================+
| 1    | `Publish to PyPI <#publishing-to-pypi>`__            |
+------+----------------------------------------------------+
| 2    | `Fork & clone staged-recipes <#submitting-to-staged-recipes>`__ |
+------+----------------------------------------------------+
| 3    | `Generate recipe with grayskull <#creating-the-recipe-with-grayskull>`__ |
+------+----------------------------------------------------+
| 4    | `Submit PR to staged-recipes <#submitting-to-staged-recipes>`__ |
+------+----------------------------------------------------+
| 5    | `Wait for review & merge <#post-merge-feedstock-creation>`__ |
+------+----------------------------------------------------+
| 6    | `Feedstock repository created <#post-merge-feedstock-creation>`__ |
+------+----------------------------------------------------+
| 7    | `Maintain future updates (bot/manual) <#maintaining-the-package>`__ |
+------+----------------------------------------------------+

.. hint::
   You can click on any action to jump directly to the detailed explanation!
