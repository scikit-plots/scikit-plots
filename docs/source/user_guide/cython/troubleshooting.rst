.. _cython-troubleshooting:

Troubleshooting
===============

Compiler or header missing
--------------------------

Start with capability/prerequisite inspection. Installing Cython alone is not
enough to guarantee a native compiler and compatible Python headers.

Security rejection
------------------

Treat a :class:`SecurityError` as a policy decision. Do not switch globally to
the relaxed policy simply to make one build pass; narrow the requested build or
isolate it appropriately.

Timeout or excessive compiler output
-------------------------------------

Adjust :class:`BuildBudget` only after checking whether the build is genuinely
large or stuck. The bounded diagnostic tail is intentionally not the full log.

Repository snapshot notes
-------------------------

The maintenance snapshot records open release-evidence issues involving a
checked-in ``__pycache__`` and order-dependent setuptools/distutils test
pollution. Those are maintenance/release facts, not a reason for applications
to bypass the cache or security model.
