.. _logging-python-api:

Python API
==========

The public facade re-exports common stdlib logging types/constants together with
Scikit-Plots helpers.

Normal application surface:

* ``get_logger``;
* ``setLevel`` / ``getEffectiveLevel``;
* ``set_verbosity`` / ``get_verbosity``;
* ``debug``, ``info``, ``warning``, ``error``, ``critical``;
* ``exception``;
* ``log_if``, ``log_every_n``, ``log_first_n``; and
* ``GoogleLogFormatter`` / ``AlwaysStdErrHandler``.

Do not build new code around undocumented private names even when they currently
appear in ``dir(scikitplot.logging)``.
