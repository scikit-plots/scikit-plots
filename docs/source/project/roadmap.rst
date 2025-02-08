.. |ss| raw:: html

   <strike>

.. |se| raw:: html

   </strike>

.. _roadmap-index:

Roadmap
=======

Purpose of this document
------------------------
This document list general directions that core contributors are interested
to see developed in scikit-plots. The fact that an item is listed here is in
no way a promise that it will happen, as resources are limited. Rather, it
is an indication that help is welcomed on this topic.

Statement of purpose: Scikit-plots in 2024
------------------------------------------
Since 2017 after the inception of Scikit-plot now Scikit-plots,
much has changed in the world of machine learning.


Architectural / general goals
-----------------------------
The list is numbered not as an indication of the order of priority, but to
make referring to specific points easier. Please add new entries only at the
bottom. Note that the crossed out entries are already done, and we try to keep
the document up to date as we work on these issues.


#. Improved handling of Pandas DataFrames

   * document current handling

#. Improved handling of categorical features

   * Handling mixtures of categorical and continuous variables

#. More didactic documentation

   * More and more options have been added to scikit-plots. As a result, the
     documentation is crowded which makes it hard for beginners to get the big
     picture. Some work could be done in prioritizing the information.

#. Passing around information that is not (X, y): Feature properties

   * Per-feature handling (e.g. "is this a nominal / ordinal / English language
     text?")

#. Passing around information that is not (X, y): Target information

   * We have no way to handle a mixture of categorical and continuous targets.

#. Make it easier for external users to write Scikit-plots-compatible
   components

   * More self-sufficient running of scikit-plots-contrib or a similar resource
