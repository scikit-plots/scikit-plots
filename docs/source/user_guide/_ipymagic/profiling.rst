.. _ipymagic-profiling:

======================================================================
Profiling
======================================================================

``%scikitplot_profile`` runs a statement, ``%%scikitplot_profile`` a cell,
under ``cProfile`` and shows the functions that took the time:

.. code-block:: ipython

   %scikitplot_profile -s ncalls -l 5 sorted(range(10_000), reverse=True)

.. code-block:: ipython

   %%scikitplot_profile -l 10
   def fib(n):
       return n if n < 2 else fib(n - 1) + fib(n - 2)
   fib(18)

=====================  ============================================================
Option                 Meaning
=====================  ============================================================
``-s KEY``             sort by ``cumulative`` (default), ``tottime`` or ``ncalls``
``-l N``               rows of the table (default 25)
``-D FILE``            also write the profile to FILE (``pstats``, snakeviz)
``--snakeviz``         open the profile in snakeviz (see below)
``--stop-snakeviz``    stop the snakeviz servers of this session
=====================  ============================================================

The result is a ``ProfileResult``: ``.rows`` (``ncalls``, ``tottime``,
``cumtime``, ``location``), ``.total_calls``, ``.to_dict()``. Code that
raises is profiled up to the error; the table is shown, then the error.

Only one profiler can record a process: when another one is active (an
outer profile, ``python -m cProfile``, a profiling IDE) the magic refuses
instead of silently stopping it.

snakeviz
----------------------------------------------------------------------

``--snakeviz`` starts `snakeviz <https://jiffyclub.github.io/snakeviz/>`_ on
127.0.0.1 (never on every network interface) and shows it in the notebook.
That works when the browser runs on the kernel's machine, as with a local
Jupyter. On a remote kernel, keep the table, or write the profile with
``-D`` and open the file with snakeviz where you are. Servers stop with
``--stop-snakeviz``, when the extension is unloaded (``%unload_ext
scikitplot``) and when the kernel exits.
