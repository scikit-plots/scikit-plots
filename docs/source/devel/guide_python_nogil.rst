.. https://overbits.herokuapp.com/rsteditor/
.. https://rsted.info.ucl.ac.be/

.. _python_nogil:

Python Implementations and Free-Threading (No-GIL) Support
==========================================================

:download:`"Download This Page as .pdf" <./python_nogil.pdf>`.

.. warning::
   Free-threaded (No-GIL) mode is **experimental** and only available in
   special builds of CPython (starting with version 3.13). Compatibility with
   third-party packages, particularly C extensions, is limited.

Overview
--------

Python can be executed using multiple interpreters:

* **CPython** - the reference implementation. It now offers an **experimental free-threaded (No-GIL) mode**.
* **PyPy** - a JIT-compiled alternative optimized for long-running pure Python code, still using the GIL.

Comparison Table
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - Feature
     - CPython
     - PyPy
     - Notes
   * - Default Interpreter
     - ✅ Yes
     - ❌ No
     - Widely distributed
   * - JIT Compilation
     - ❌ No
     - ✅ Yes
     - Improves pure Python execution speed
   * - Free-Threading (No-GIL)
     - ✅ Experimental (3.13+)
     - ❌ No
     - Available only in special CPython builds
   * - C Extension Support
     - ✅ Full (legacy CPython API)
     - ⚠ Partial (via CFFI)
     - Free-threaded mode requires API updates
   * - Memory Model
     - Reference counting + GIL
     - Tracing GC + JIT
     - Affects concurrency behavior

Timeline of No-GIL Development
------------------------------

.. list-table::
   :header-rows: 1

   * - Year
     - Event
   * - 2021
     - PEP 703 proposed “No GIL” CPython fork
   * - 2023
     - PEP 703 accepted for experimental inclusion in CPython 3.13
   * - 2024
     - CPython 3.13 released with optional ``--disable-gil`` build
   * - 2025 (planned)
     - CPython 3.14 expands free-threading and extension support
   * - 2026+ (planned)
     - CPython 3.15+ may stabilize No-GIL execution

Future CPython Feature Matrix
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 40

   * - Version
     - GIL Support
     - Free-Threaded Mode
     - Status
     - Notes
   * - 3.12
     - ✅ Yes
     - ❌ No
     - Stable
     - Traditional GIL model
   * - 3.13
     - ✅ Yes
     - ✅ Yes
     - Experimental
     - Requires ``python-freethreading`` build
   * - 3.14
     - ✅ Yes
     - ✅ Yes
     - Experimental / Early Adoption
     - Improved extension support
   * - 3.15+
     - ✅ Yes
     - ✅ Yes
     - Future
     - No-GIL may become standard

Tabbed View (CPython vs PyPy)
-----------------------------

.. tabs::

   .. tab:: CPython

      CPython provides **experimental free-threaded execution**.

      **Useful Links:**

      * `CPython Repository <https://github.com/python/cpython>`_
      * `Free-Threading HOWTO <https://docs.python.org/3/howto/free-threading-python.html>`_
      * `Installing Free-Threaded CPython <https://py-free-threading.github.io/installing-cpython/>`_

      **Installation Example:**

      .. code-block:: bash

         mamba create -n nogil -c conda-forge python-freethreading

      or:

      .. code-block:: bash

         conda create -n nogil --override-channels -c conda-forge python-freethreading

   .. tab:: PyPy

      PyPy uses JIT compilation but **does not support No-GIL execution**.

      **Useful Links:**

      * `PyPy Repository <https://github.com/pypy/pypy>`_
      * `CPython vs PyPy Differences <https://doc.pypy.org/en/latest/cpython_differences.html>`_

..
  Flowchart: Choosing the Right Python Interpreter
  ------------------------------------------------
  .. mermaid::
    flowchart TD
        A[Do you need true multi-threading without the GIL?] -->|Yes| B[Use CPython 3.13+ Free-Threaded Build]
        A -->|No| C[Is your code performance-critical pure Python?]
        C -->|Yes| D[Use PyPy for JIT optimization]
        C -->|No| E[Use Standard CPython]
        B --> F[Are you using C extensions?]
        F -->|Yes, legacy API| G[Update extensions for No-GIL]
        F -->|No or using CFFI| H[Compatible]

Critical and Essential Knowledge
--------------------------------

.. important::
    Misunderstanding these points may lead to performance or correctness issues:

    * ``python-freethreading`` is a **special CPython build with GIL disabled**.
    * Free-threaded mode is **experimental**; not the default in any official release.
    * Most C extensions are **incompatible** with No-GIL and require updates.
    * Free-threaded execution **does not automatically improve performance**.
    * Memory and object lifecycle semantics differ; race conditions are possible.
    * Standard GIL-enabled CPython will remain available; No-GIL is optional.
    * Explicit installation and thread-safe coding practices are required.

.. caution::
   Code that runs correctly under GIL may be **unsafe under No-GIL**.

Risks vs Benefits Matrix
------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Category
     - Benefits
     - Risks
   * - Multi-threaded Python
     - True parallelism across CPU cores
     - Race conditions if code is not thread-safe
   * - Performance
     - Potential speed-up in CPU-bound multi-threaded code
     - May degrade single-threaded performance
   * - C Extension Compatibility
     - Can write No-GIL-safe extensions
     - Legacy extensions may crash or misbehave
   * - Future-Proofing
     - Prepares code for upcoming GIL-free CPython
     - Still experimental; behavior may change

Migration Checklist
-------------------

1. Install Free-Threaded Python.
2. Audit all C extensions.
3. Refactor shared mutable state.
4. Run multi-threaded tests.
5. Verify third-party library compatibility.
6. Monitor performance carefully.
7. Document interpreter requirements.

Who Should Not Use Free-Threaded Python Yet
-------------------------------------------

* Projects heavily dependent on legacy C extensions.
* Applications stable under standard CPython.
* Teams unfamiliar with thread safety.
* Environments requiring strict stability (e.g., production servers).


.. prompt:: bash

    python -VV

.. prompt:: python >>>

    import os, platform, subprocess, sys
    print("Implementation\t:", platform.python_implementation(), sys.implementation.name)
    print("Version\t\t:", sys.version)
    print("Compiler\t:", platform.python_compiler())

.. prompt:: python >>>

    # Python prints -VV, by os
    os.system("python -VV")
    with os.popen("python -VV") as f: output=f.read().strip()
    print(output)

.. prompt:: python >>>

    # Python prints -VV, by subprocess
    output = subprocess.Popen(["python", "-VV"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).communicate()[0].strip()
    output = subprocess.run(["python", "-VV"], capture_output=True, text=True).stdout.strip()
    output = subprocess.check_output("python -VV", shell=True, text=True).strip()
    print(output)
