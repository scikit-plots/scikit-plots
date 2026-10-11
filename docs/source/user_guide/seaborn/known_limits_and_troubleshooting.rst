.. _seaborn-known-limits-and-troubleshooting:

Known limits and troubleshooting
================================

The maintenance review records open compatibility/model/decile findings
(``SBN-COMPAT-001``, ``SBN-MODEL-001``, ``SBN-DEC-001`` and
``SBN-DEC-002``). The consequence for users is simple: pin and test the actual
plotting stack you deploy, and do not treat gallery rendering as release proof
for every kind/data combination.

If a plot fails after a seaborn upgrade, determine whether the wrapper relies on
an upstream-private object before changing your data. If values look wrong but
the figure renders, validate the intermediate model/decile data before changing
style settings.
