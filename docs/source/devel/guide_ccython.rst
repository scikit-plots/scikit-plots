.. _building-c-or-cython-extensions:

*********************************
C or Cython Extensions Guidelines
*********************************

.. admonition:: Template

   Template for further usage, template belong to `astropy`.

Astropy supports using C extensions for wrapping C libraries and Cython for
speeding up computationally-intensive calculations. Both Cython and C extension
building can be customized using the ``get_extensions`` function of the
``setup_package.py`` file. If defined, this function must return a list of
``setuptools.Extension`` objects. The creation process is left to the
subpackage designer, and can be customized however is relevant for the
extensions in the subpackage.

While C extensions must always be defined through the ``get_extensions``
mechanism, Cython files (ending in ``.pyx``) are automatically located
by `extension-helpers <https://extension-helpers.readthedocs.io/>`_ and
loaded in separate extensions if they are not in ``get_extensions``. For
Cython extensions located in this way, headers for numpy C functions are
included in the build, but no other external headers are included. ``.pyx``
files present in the extensions returned by ``get_extensions`` are not
included in the list of automatically generated extensions.

.. note::

    If a ``setuptools.Extension`` object is provided for Cython
    source files using the ``get_extensions`` mechanism, it is very
    important that the ``.pyx`` files be given as the ``source``, rather than the
    ``.c`` files generated by Cython.

Using Numpy C headers
=====================

If your C or Cython extensions uses `numpy` at the C level, you probably
need access to the numpy C headers.  When doing this, you should use
``numpy.get_include()`` to specify the include directory to use, for example::

    from setuptools import Extension
    import numpy

    def get_extensions():
        return Extension(name='myextension', sources=['myext.c'],
                         include_dirs=[numpy.get_include()])


Installing C header files
=========================

If your C extension needs to be linked from other third-party C code,
you probably want to install its header files along side the Python module.

    1) Create an ``include`` directory inside of your package for
       all of the header files.

    2) Use the ``[tool.setuptools.package_data]`` section in your ``pyproject.toml``
       file to include those header files in the package. For example, the
       `astropy.wcs` package has the following entries in the
       ``[tool.setuptools.package_data]`` section::

           [tool.setuptools.package_data]
           ...
           "astropy.wcs" = ["include/*/*.h"]
           ...

Preventing importing at build time
==================================

It is important to make sure that ``setup_package.py`` files do not trigger an
import of the package they are in - so they should be able to be executed without
relying on imports to other parts of the package.

Speed up your builds with ccache
================================

`ccache <https://en.wikipedia.org/wiki/Ccache>`_ is a tool that caches
compiled sources so that they don't have to be recompiled (so long as they are
unchanged) even if the outputs have been deleted.  This means that if you
switch branches or clean your source checkout you can save a lot of time by
avoiding the majority of re-compiles from scratch.

Because installation and configuration of ccache varies from platform to
platform, please consult the ccache documentation and/or Google to set up
ccache on your system--this is strongly encouraged for anyone doing significant
development of Astropy or scientific programming in general.
