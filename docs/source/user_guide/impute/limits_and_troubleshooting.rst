.. _impute-limits-and-troubleshooting:

Limits and troubleshooting
==========================

Optional backend missing
   Install/build the selected Annoy or Voyager capability; do not silently
   reinterpret the request as exact KNN.

Unexpected imputed values
   Verify feature scaling, metric, missing-value representation, neighbor count,
   weighting and initial strategy before tuning index search effort.

Current review caveats
   The maintenance review records a broad Annoy fallback behavior that needs a
   clearer ownership decision and an unused module-scope pandas import. It also
   notes that local mirror tests for ``_base.py`` currently exercise sklearn
   behavior more than local ownership. Treat release claims for the experimental
   path separately from the stable standard imputers.
