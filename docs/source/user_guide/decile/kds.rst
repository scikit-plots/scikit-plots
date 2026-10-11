.. _decile-kds:

KDS functions
=============

The ``scikitplot.decile.kds`` namespace provides array-oriented decile
statistics and plots.

.. autosummary::
   :nosignatures:

   scikitplot.decile.kds.decile_table
   scikitplot.decile.kds.plot_lift
   scikitplot.decile.kds.plot_lift_decile_wise
   scikitplot.decile.kds.plot_cumulative_gain
   scikitplot.decile.kds.plot_ks_statistic
   scikitplot.decile.kds.report
   scikitplot.decile.kds.print_labels

``decile_table`` is the shared statistical core. Plot/report helpers should be
interpreted from the same positive-class definition and decile count.

.. warning::

   The repository maintenance review records a current KDS contract issue where
   some documented class/positive-label/precision arguments are dropped by
   internal calls. Until that finding is closed, validate non-default
   ``class_index``, ``pos_label`` and precision-sensitive workflows with focused
   tests rather than assuming every wrapper forwards every option.
