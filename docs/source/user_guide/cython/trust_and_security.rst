.. _cython-trust-and-security:

Trust and security
==================

The trust boundary is the central concept of this submodule.

.. code-block:: text

   source -> validation -> toolchain -> native compiler -> shared library
                                                     |
                                                     v
                                             current Python process

:class:`SecurityPolicy` validates build inputs such as paths, macro names and
compiler arguments. ``DEFAULT_SECURITY_POLICY`` is the normal starting point;
``RELAXED_SECURITY_POLICY`` deliberately weakens restrictions and should be a
conscious application decision.

A policy cannot make malicious C/C++/Cython safe. Native code can access process
memory and operating-system capabilities. Run untrusted compilation in an
external sandbox/container/VM designed for hostile code, not inside your main
application process.

Build budgets
-------------

:class:`BuildBudget` can bound compile wall-clock time and retained compiler
output. This protects resource use and diagnostics size; a timeout does not
terminate every possible external compiler descendant on every platform, so use
process-level isolation for adversarial workloads.
