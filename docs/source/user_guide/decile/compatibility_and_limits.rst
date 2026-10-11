.. _decile-compatibility-and-limits:

Compatibility and limits
========================

``scikitplot.decile`` currently has three user-visible surfaces: current
ModelPlotPy, legacy ``modelplotpy``, and KDS. Test the surface you deploy rather
than treating one passing example as evidence for all three.

Decile/ntile boundaries can change when there are ties or small samples. Avoid
reading tiny changes in neighboring buckets as a stable business effect without
uncertainty analysis.

The maintenance snapshot records open forwarding issues in KDS and a global RNG
side effect in the legacy ModelPlotPy path. These limitations are documented so
users can choose safer defaults; they are not silently normalized by the guide.
