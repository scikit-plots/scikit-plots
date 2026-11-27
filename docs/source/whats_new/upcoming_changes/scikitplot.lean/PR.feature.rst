- :class:`~scikitplot.lean.LeanIndex` finds vacuous predicates: ``Prop``
  definitions whose body is ``True``, ``False`` and connectives only, or
  uses none of their parameters, so that a theorem stated with them can
  hold without saying anything about its subject. Discovery lists them
  (``claims.vacuous_predicates``, with the reason), and a dossier marks
  each one its statement unfolds to and reviews them in the check
  ``vacuous_definitions``. A new gallery example audits an AI-assisted
  proof: vacuous statements, ``sorry`` behind notation, axioms,
  ``native_decide``, text Lean refuses and the CI gate.
