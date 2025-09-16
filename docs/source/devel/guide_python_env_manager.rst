:orphan:
..
  # This file is included. So it needs to be marked as orphan to suppress warnings.
  .. include:: guide_python_env_manager.rst
      :start-after: :orphan:

.. https://www.tutorialspoint.com/compilers/online-restructure-editor.htm

.. _python_env_manager:

======================================================================
📦 Python Env Manager Guidelines
======================================================================

.. 🧊

.. seealso::

   🔎 Run the latest scikit-plots container — with full or partial preinstallation — interactively:

   * `"Scikit-plots Runtime Docker Images" <https://hub.docker.com/r/scikitplot/scikit-plots>`_

   * https://github.com/scikit-plots/scikit-plots


📦 Conda/Anaconda Environment Guidelines
---------------------------------------

.. seealso::

   * `"Conda Documentation" <https://docs.conda.io/en/latest/>`_

   * `"Installing Conda" <https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html>`_

   * `"Installing Anaconda" <https://www.anaconda.com/docs/getting-started/anaconda/install>`_

   * `"Installing Miniconda" <https://www.anaconda.com/docs/getting-started/miniconda/install>`_

   * `"Installing Miniforge" <https://conda-forge.org/download/>`_

   * `"Installing Mamba" <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_

   * `"Installing Micromamba" <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_

🛠 The Main Tools (And What They're Good At)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#table-directives

..
    =====  =====  =======
    A      B      A and B
    =====  =====  =======
    False  False  False
    True   False  False
    False  True   False
    True   True   True
    =====  =====  =======

..
    +-------------+------------------------------------------------------------+----------------------------+---------------------------------------------------------------+
    | Tool        | Create Environment                                         | ⚠️ Activate Environment    | Critical Notes                                                |
    +=============+============================================================+============================+===============================================================+
    | conda       | conda create -n py311 python=3.11                          | conda activate py311       | Full Anaconda distribution with many preinstalled packages.   |
    +-------------+------------------------------------------------------------+----------------------------+---------------------------------------------------------------+
    +-------------+------------------------------------------------------------+----------------------------+---------------------------------------------------------------+
    | miniconda   | conda create -n py311 python=3.11                          | conda activate py311       | Minimal installer. Lightweight. Recommended for most users.   |
    +-------------+------------------------------------------------------------+----------------------------+---------------------------------------------------------------+
    +-------------+------------------------------------------------------------+----------------------------+---------------------------------------------------------------+
    | mamba       | mamba create -n py311 python=3.11                          | conda activate py311       | Drop-in faster alternative to `conda`. Needs conda to run.    |
    +-------------+------------------------------------------------------------+----------------------------+---------------------------------------------------------------+
    +-------------+------------------------------------------------------------+----------------------------+---------------------------------------------------------------+
    | micromamba  | micromamba create -n py311 python=3.11                     | micromamba activate py311  | No Python or conda needed. Very fast. Shell hook required.    |
    +-------------+------------------------------------------------------------+----------------------------+---------------------------------------------------------------+

.. :widths: 12, 40, 20, 40
.. csv-table:: **🛠 The Main Tools (And What They're Good At)**
   :header: "Tool", "Create Environment", "⚠️ Activate Environment", "Critical Notes"

   "**conda**", "conda create -n py311 python=3.11", "conda activate py311", "Full Anaconda distribution with **many preinstalled packages**."
   "**miniconda**", "conda create -n py311 python=3.11", "conda activate py311", "Minimal installer. Lightweight. **Recommended for most users**."
   "**mamba**", "mamba create -n py311 python=3.11", "conda activate py311", "Drop-in **faster alternative to conda**. Needs conda to run."
   "**micromamba**", "micromamba create -n py311 python=3.11", "micromamba activate py311", "**No Python or conda needed**. Very fast. Shell hook required."


⚖️ Feature Comparison (In Plain Words)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    +-------------------------+---------------+--------------+-----------------+------------------+----------------------------------------------------------+
    | Feature                 | conda         | miniconda    | mamba           | micromamba       | Critical Notes                                           |
    +=========================+===============+==============+=================+==================+==========================================================+
    | Size                    | Large         | Medium       | Large           | Very Small       | micromamba is ideal for minimal or embedded environments |
    +-------------------------+---------------+--------------+-----------------+------------------+----------------------------------------------------------+
    | Language                | Python        | Python       | C++             | C++              | mamba/micromamba are faster due to compiled language     |
    +-------------------------+---------------+--------------+-----------------+------------------+----------------------------------------------------------+
    | Installation            | GUI/CLI       | CLI          | via conda/pip   | Static binary    | micromamba: no Python needed, portable                   |
    +-------------------------+---------------+--------------+-----------------+------------------+----------------------------------------------------------+
    | Speed                   | Slow          | Slow         | Fast            | Very Fast        | mamba/micromamba offer drastically faster performance    |
    +-------------------------+---------------+--------------+-----------------+------------------+----------------------------------------------------------+
    | Conda env creation      | ✅ Support   | ✅ Support   | ✅ Support     | ✅ Support       | ✅ All support full conda environments                   |
    +-------------------------+---------------+--------------+-----------------+------------------+----------------------------------------------------------+
    | Scripting / Docker      | ⚠️ Heavy     | ⚠️ Moderate  | ⚠️ Heavy       | ✅ Lightweight   | micromamba is best suited for Docker and automation      |
    +-------------------------+---------------+--------------+-----------------+------------------+----------------------------------------------------------+
    | Activation support      | ✅ Support   | ✅ Support   | ✅ Support     | ✅ (via hook)    | micromamba requires manual shell integration             |
    +-------------------------+---------------+--------------+-----------------+------------------+----------------------------------------------------------+
    | Use in CI/CD            | ⚠️ Slower    | ⚠️ Moderate  | ✅ Fast        | ✅ Best          | micromamba is ideal for CI/CD pipelines                  |
    +-------------------------+---------------+--------------+-----------------+------------------+----------------------------------------------------------+
    | GPU support             | ✅ Support   | ✅ Support   | ✅ Support     | ✅  Support      | ✅ All support CUDA & GPU packages                       |
    +-------------------------+---------------+--------------+-----------------+------------------+----------------------------------------------------------+
    | Platform support        | All           | All          | All             | All              | micromamba supports Linux/macOS/Windows (via shell)      |
    +-------------------------+---------------+--------------+-----------------+------------------+----------------------------------------------------------+

.. :widths: 22, 10, 10, 10, 12, 36
.. csv-table:: **⚖️ Feature Comparison (In Plain Words)**
   :header: "Feature", "conda", "miniconda", "mamba", "micromamba", "Critical Notes"

   "**Size**", "Large", "Medium", "Large", "Very Small", "micromamba is ideal for **minimal** or **embedded** environments"
   "**Language**", "Python", "Python", "C++", "C++", "mamba/micromamba are **faster due** to compiled language"
   "**Installation**", "GUI/CLI", "CLI", "via conda/pip", "Static binary", "🚀 micromamba: **no Python** needed, **portable**"
   "**Speed**", "Slow", "Slow", "Fast", "Very Fast", "mamba/micromamba offer drastically **faster performance**"
   "**Conda env creation**", "✅ Support", "✅ Support", "✅ Support", "✅ Support", "✅ All support full **conda environments**"
   "**Scripting / Docker**", "⚠️ Heavy", "⚠️ Moderate", "⚠️ Heavy", "✅ Lightweight", "micromamba is best suited for **Docker and automation**"
   "**Activation support**", "✅ Support", "✅ Support", "✅ Support", "✅ (via hook)", "micromamba requires **manual shell** integration"
   "**Use in CI/CD**", "⚠️ Slower", "⚠️ Moderate", "✅ Fast", "✅ Best", "micromamba is ideal for **CI/CD pipelines**"
   "**GPU support**", "✅ Support", "✅ Support", "✅ Support", "✅ Support", "✅ All support **CUDA & GPU** packages"
   "**Platform support**", "All", "All", "All", "All", "micromamba supports **Linux/macOS/Windows** (via shell)"



💡 When to Use What
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Use conda if you want the full Anaconda experience on local machines. Full-featured but heavy. Best for GUI users or those needing full Anaconda.
- Use miniconda for lighter installs with more manual control. Lightweight, clean base. Ideal for custom setups.
- Use mamba for speed, especially when resolving complex dependencies interactively. Same usage as conda, but much faster. Great for large envs.
- Use micromamba in containers, CI/CD, or when minimal overhead is critical. Tiny single binary, perfect for Docker, CI/CD, and scripting.

🛠 How to Use It Tips
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create the environment with only default (base) packages:

.. prompt:: bash

  # (conda or mamba) Create New Env and install ``scikit-plots``
  mamba create -n py311 python=3.11 ipykernel -y

.. prompt:: bash

  # (conda or mamba) Create New Env and install ``scikit-plots``
  conda env create -f environment.yml

.. prompt:: bash

  conda activate py311

Create the environment with both base and extended packages:

.. prompt:: bash

  conda create --name py311 --file environment.yml --group extended

Add the optional extended packages later to an existing environment:

.. prompt:: bash

  conda install --name py311 --file environment.yml --group extended


📦 Pipenv Environment Guidelines
---------------------------------------

.. seealso::

   * `"pipenv" <https://pypi.org/project/pipenv/>`_

   * https://github.com/scikit-plots/scikit-plots/tree/main/docker/env_pipenv
