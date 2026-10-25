.. _env-targets:

======================================================================
Targets: host, other platforms, browser, mobile
======================================================================

``target`` says where the environment will run, which decides how it can be
built.

=====================  ===========================================================
Target                 What is built
=====================  ===========================================================
``host``               an environment for this machine (default); imports are
                       verified, ``run`` works
``linux-aarch64``,     a cross-build for another Conda platform (any Conda
``osx-arm64``,         subdir): micromamba or mamba solve for it, nothing in it
``win-64`` ...         runs here, PyPI packages are refused. Copy it to the same
                       path on the target machine (or into an image).
``emscripten-wasm32``  a browser environment for JupyterLite / Xeus kernels, from
                       emscripten-forge; pure-Python PyPI packages are added
``pyodide``            plan only: the plan's artifact ``micropip.py`` installs the
                       PyPI packages inside Pyodide
``android``, ``ios``   refused, with the reason below
=====================  ===========================================================

Cross-building
----------------------------------------------------------------------

.. code-block:: console

   $ scikitplot env create -p /opt/envs/arm --target linux-aarch64 \
         --conda-package python=3.12 --conda-package numpy

The files keep the build prefix, so the environment works when copied to the
same path on an ``aarch64`` machine. Imports cannot be verified here; the
plan says so in a note.

Browser
----------------------------------------------------------------------

For JupyterLite with a Xeus kernel, build with ``--target
emscripten-wasm32`` (default channels ``https://prefix.dev/emscripten-forge-4x``
and ``https://prefix.dev/conda-forge``), or let the JupyterLite add-on
``scikitplot._env._xeus._addon.XeusAddon`` build each ``environment.yml`` the
same way. For Pyodide, the plan prints a ready script:

.. code-block:: python

   from scikitplot import _env

   plan = _env.plan(name="lite", target="pyodide", packages=["six", "attrs"])
   script = dict(plan.artifacts)["micropip.py"]
   assert "await micropip.install(['six', 'attrs'])" in script

Phones and tablets
----------------------------------------------------------------------

An Android or iOS app embeds its own Python, built together with the app by
a mobile packaging toolchain, so there is no environment to create on a host
and the builder refuses rather than produce something that cannot run
there. The browser targets run on phones and tablets in the browser.
