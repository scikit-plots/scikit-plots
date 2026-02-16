"""
Cython: Realtime compile_and_load (.pyx)
======================================================================

.. currentmodule:: scikitplot.cython

An example showing the :py:mod:`~scikitplot.cython` submodule..
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause


# %%
# cython examples
# ---------------

import scikitplot as sp
sp.__version_iso_8601__

# %%

from scikitplot import cython

print(cython.__doc__)

# %%

from scikitplot.cython import compile_and_load
m = compile_and_load("def f(int n):\n    return n*n")
m.f(10)

# %%

cython.list_templates()

# %%

cython.get_template_path(cython.list_templates()[0])

# %%

print(cython.read_template(cython.list_templates()[0]))

# %%

cython.get_cache_dir()

# %%

cython.list_cached()

# %%

cython.list_cached()[0].build_dir

# %%

cython.list_cached()[0].artifact_path

# %%

from scikitplot.cython import compile_template

m = compile_template(cython.list_templates()[2])
m = compile_template(cython.list_templates()[1])
m = compile_template(cython.list_templates()[0])
m.square(12)

# %%

m = compile_template('module_cython/t01_counter')
m

# %%

cython.purge_cache()

# %%

cython.list_cached()


# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: model building
#    plot-type: cython
#    domain: cython
#    level: beginner
#    purpose: showcase
