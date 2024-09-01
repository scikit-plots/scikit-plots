.. _contributing:

.. currentmodule:: scikitplot

============
Contributing
============

.. warning::

    Not Implemented...

**Reserved.**

This project is a community effort, and everyone is welcome to
contribute. It is hosted on https://github.com/scikit-plots/scikit-plots.

In case you experience issues using this package, do not hesitate to submit a
ticket to the
`GitHub issue tracker
<https://github.com/scikit-plots/scikit-plots/issues>`_. You are also
welcome to post feature requests or pull requests.

Ways to contribute
==================

There are many ways to contribute to scikit-plots, with the most common ones
being contribution of code or documentation to the project. Improving the
documentation is no less important than improving the library itself.  If you
find a typo in the documentation, or have made improvements, do not hesitate to
send an email to the mailing list or preferably submit a GitHub pull request.
Full documentation can be found under the doc/ directory.

Automated Contributions Policy
==============================

Submitting a bug report or a feature request
============================================

We use GitHub issues to track all bugs and feature requests; feel free to open
an issue if you have found a bug or wish to see a feature implemented.

.. _filing_bugs:

How to make a good bug report
-----------------------------


Contributing code
=================


How to contribute
-----------------

The preferred way to contribute to scikit-plots is to fork the `main
repository <https://github.com/scikit-plots/scikit-plots/>`__ on GitHub,
then submit a "pull request" (PR).

In the first few steps, we explain how to locally install scikit-plots, and
how to set up your git repository:

1. `Create an account <https://github.com/join>`_ on
   GitHub if you do not already have one.

2. Fork the `project repository
   <https://github.com/scikit-plots/scikit-plots>`__: click on the 'Fork'
   button near the top of the page. This creates a copy of the code under your
   account on the GitHub user account. For more details on how to fork a
   repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

3. Clone your fork of the scikit-plots repo from your GitHub account to your
   local disk:

    .. code-block:: text

        git clone git@github.com:YourLogin/scikit-plots.git  # add --depth 1 if your connection is slow
        cd scikitplots

4. Follow steps 2-6 in install_bleeding_edge to build scikit-plots in
   development mode and return to this document.

5. Install the development dependencies:

    .. code-block:: text

        pip install pytest pytest-cov ruff mypy numpydoc black==24.3.0

.. _upstream:

6. Add the ``upstream`` remote. This saves a reference to the main
   scikit-plots repository, which you can use to keep your repository
   synchronized with the latest changes:

   .. code-block:: text

        git remote add upstream git@github.com:scikit-plots/scikit-plots.git

7. Check that the `upstream` and `origin` remote aliases are configured correctly
   by running `git remote -v` which should display:

   .. code-block:: text

        origin	git@github.com:YourLogin/scikit-plots.git (fetch)
        origin	git@github.com:YourLogin/scikit-plots.git (push)
        upstream	git@github.com:scikit-plots/scikit-plots.git (fetch)
        upstream	git@github.com:scikit-plots/scikit-plots.git (push)

You should now have a working installation of scikit-plots, and your git repository
properly configured. It could be useful to run some test to verify your installation.

The next steps now describe the process of modifying code and submitting a PR:

8. Synchronize your ``main`` branch with the ``upstream/main`` branch,
   more details on `GitHub Docs <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork>`_:

   .. code-block:: text

        git checkout main
        git fetch upstream
        git merge upstream/main

9. Create a feature branch to hold your development changes:

   .. code-block:: text

        git checkout -b my_feature

   and start making changes. Always use a feature branch. It's good
   practice to never work on the ``main`` branch!

10. (**Optional**) Install `pre-commit <https://pre-commit.com/#install>`_ to
    run code style checks before each commit:

    .. code-block:: text

        pip install pre-commit
        pre-commit install

    pre-commit checks can be disabled for a particular commit with
    `git commit -n`.

11. Develop the feature on your feature branch on your computer, using Git to
    do the version control. When you're done editing, add changed files using
    ``git add`` and then ``git commit``:

    .. code-block:: text

        git add modified_files
        git commit

    to record your changes in Git, then push the changes to your GitHub
    account with:

    .. code-block:: text

        git push -u origin my_feature

12. Follow `these
    <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
    instructions to create a pull request from your fork. This will send an
    email to the committers. You may want to consider sending an email to the
    mailing list for more visibility.

It is often helpful to keep your local feature branch synchronized with the
latest changes of the main scikit-plots repository:

.. code-block:: text

        git fetch upstream
        git merge upstream/main

Subsequently, you might need to solve the conflicts. You can refer to the
`Git documentation related to resolving merge conflict using the command
line
<https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/>`_.

.. topic:: Learning Git

    The `Git documentation <https://git-scm.com/documentation>`_ and
    http://try.github.io are excellent resources to get started with git,
    and understanding all of the commands shown here.

.. _pr_checklist:

Pull request checklist
----------------------

Before a PR can be merged, it needs to be approved by two core developers.
An incomplete contribution -- where you expect to do more work before receiving
a full review -- should be marked as a `draft pull request
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request>`__
and changed to "ready for review" when it matures. Draft PRs may be useful to:
indicate you are working on something to avoid duplicated work, request
broad review of functionality or API, or seek collaborators. Draft PRs often
benefit from the inclusion of a `task list
<https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments>`_ in
the PR description.

In order to ease the reviewing process, we recommend that your contribution
complies with the following rules before marking a PR as "ready for review". The
**bolded** ones are especially important:

1. **Give your pull request a helpful title** that summarizes what your
   contribution does. This title will often become the commit message once
   merged so it should summarize your contribution for posterity. In some
   cases "Fix <ISSUE TITLE>" is enough. "Fix #<ISSUE NUMBER>" is never a
   good title.

2. **Make sure your code passes the tests**. The whole test suite can be run
   with `pytest`, but it is usually not recommended since it takes a long
   time. It is often enough to only run the test related to your changes:
   for example, if you changed something in
   `sklearn/linear_model/_logistic.py`, running the following commands will
   usually be enough:

   - `pytest sklearn/linear_model/_logistic.py` to make sure the doctest
     examples are correct
   - `pytest sklearn/linear_model/tests/test_logistic.py` to run the tests
     specific to the file
   - `pytest sklearn/linear_model` to test the whole
     :mod:`~sklearn.linear_model` module
   - `pytest doc/modules/linear_model.rst` to make sure the user guide
     examples are correct.
   - `pytest sklearn/tests/test_common.py -k LogisticRegression` to run all our
     estimator checks (specifically for `LogisticRegression`, if that's the
     estimator you changed).

   There may be other failing tests, but they will be caught by the CI so
   you don't need to run the whole test suite locally. For guidelines on how
   to use ``pytest`` efficiently.

3. **Make sure your code is properly commented and documented**, and **make
   sure the documentation renders properly**. To build the documentation, please
   refer to our :ref:`contribute_documentation` guidelines. The CI will also
   build the docs: please refer to :ref:`generated_doc_CI`.

4. **Tests are necessary for enhancements to be
   accepted**. Bug-fixes or new features should be provided with
   `non-regression tests
   <https://en.wikipedia.org/wiki/Non-regression_testing>`_. These tests
   verify the correct behavior of the fix or feature. In this manner, further
   modifications on the code base are granted to be consistent with the
   desired behavior. In the case of bug fixes, at the time of the PR, the
   non-regression tests should fail for the code base in the ``main`` branch
   and pass for the PR code.

5. Follow the coding-guidelines.

6. When applicable, use the validation tools and scripts in the :mod:`sklearn.utils`
   module. A list of utility routines available for developers can be found in the
   :ref:`developers-utils` page.

7. Often pull requests resolve one or more other issues (or pull requests).
   If merging your pull request means that some other issues/PRs should
   be closed, you should `use keywords to create link to them
   <https://github.com/blog/1506-closing-issues-via-pull-requests/>`_
   (e.g., ``Fixes #1234``; multiple issues/PRs are allowed as long as each
   one is preceded by a keyword). Upon merging, those issues/PRs will
   automatically be closed by GitHub. If your pull request is simply
   related to some other issues/PRs, or it only partially resolves the target
   issue, create a link to them without using the keywords (e.g., ``Towards #1234``).

8. PRs should often substantiate the change, through benchmarks of
   performance and efficiency (see :ref:`monitoring_performances`) or through
   examples of usage. Examples also illustrate the features and intricacies of
   the library to users. Have a look at other examples in the `examples/
   <https://github.com/scikit-plots/scikit-plots/tree/main/examples>`_
   directory for reference. Examples should demonstrate why the new
   functionality is useful in practice and, if possible, compare it to other
   methods available in scikit-plots.

9. New features have some maintenance overhead. We expect PR authors
   to take part in the maintenance for the code they submit, at least
   initially. New features need to be illustrated with narrative
   documentation in the user guide, with small code snippets.
   If relevant, please also add references in the literature, with PDF links
   when possible.

10. The user guide should also include expected time and space complexity
    of the algorithm and scalability, e.g. "this algorithm can scale to a
    large number of samples > 100000, but does not scale in dimensionality:
    `n_features` is expected to be lower than 100".

Continuous Integration (CI)
---------------------------

.. _commit_markers:

Commit message markers
^^^^^^^^^^^^^^^^^^^^^^

Please note that if one of the following markers appear in the latest commit
message, the following actions are taken.

====================== ===================
Commit Message Marker  Action Taken by CI
====================== ===================
[ci skip]              CI is skipped completely
[cd build]             CD is run (wheels and source distribution are built)
[cd build gh]          CD is run only for GitHub Actions
[cd build cirrus]      CD is run only for Cirrus CI
[lint skip]            Azure pipeline skips linting
[scipy-dev]            Build & test with our dependencies (numpy, scipy, etc.) development builds
[free-threaded]        Build & test with CPython 3.13 free-threaded
[pyodide]              Build & test with Pyodide
[azure parallel]       Run Azure CI jobs in parallel
[cirrus arm]           Run Cirrus CI ARM test
[float32]              Run float32 tests by setting `SKLEARN_RUN_FLOAT32_TESTS=1`.
[doc skip]             Docs are not built
[doc quick]            Docs built, but excludes example gallery plots
[doc build]            Docs built including example gallery plots (very long)
====================== ===================

Note that, by default, the documentation is built but only the examples
that are directly modified by the pull request are executed.

.. _stalled_pull_request:

Stalled pull requests
---------------------

As contributing a feature can be a lengthy process, some
pull requests appear inactive but unfinished. In such a case, taking
them over is a great service for the project. A good etiquette to take over is:

* **Determine if a PR is stalled**

  * A pull request may have the label "stalled" or "help wanted" if we
    have already identified it as a candidate for other contributors.

  * To decide whether an inactive PR is stalled, ask the contributor if
    she/he plans to continue working on the PR in the near future.
    Failure to respond within 2 weeks with an activity that moves the PR
    forward suggests that the PR is stalled and will result in tagging
    that PR with "help wanted".

    Note that if a PR has received earlier comments on the contribution
    that have had no reply in a month, it is safe to assume that the PR
    is stalled and to shorten the wait time to one day.

    After a sprint, follow-up for un-merged PRs opened during sprint will
    be communicated to participants at the sprint, and those PRs will be
    tagged "sprint". PRs tagged with "sprint" can be reassigned or
    declared stalled by sprint leaders.

* **Taking over a stalled PR**: To take over a PR, it is important to
  comment on the stalled PR that you are taking over and to link from the
  new PR to the old one. The new PR should be created by pulling from the
  old one.

Stalled and Unclaimed Issues
----------------------------

Generally speaking, issues which are up for grabs will have a
`"help wanted" <https://github.com/scikit-plots/scikit-plots/labels/help%20wanted>`_.
tag. However, not all issues which need contributors will have this tag,
as the "help wanted" tag is not always up-to-date with the state
of the issue. Contributors can find issues which are still up for grabs
using the following guidelines:

* First, to **determine if an issue is claimed**:

  * Check for linked pull requests
  * Check the conversation to see if anyone has said that they're working on
    creating a pull request

* If a contributor comments on an issue to say they are working on it,
  a pull request is expected within 2 weeks (new contributor) or 4 weeks
  (contributor or core dev), unless an larger time frame is explicitly given.
  Beyond that time, another contributor can take the issue and make a
  pull request for it. We encourage contributors to comment directly on the
  stalled or unclaimed issue to let community members know that they will be
  working on it.

* If the issue is linked to a :ref:`stalled pull request <stalled_pull_request>`,
  we recommend that contributors follow the procedure
  described in the :ref:`stalled_pull_request`
  section rather than working directly on the issue.

.. _new_contributors:

Issues for New Contributors
---------------------------

New contributors should look for the following tags when looking for issues.  We
strongly recommend that new contributors tackle "easy" issues first: this helps
the contributor become familiar with the contribution workflow, and for the core
devs to become acquainted with the contributor; besides which, we frequently
underestimate how easy an issue is to solve!

- **Good first issue tag**

  A great way to start contributing to scikit-plots is to pick an item from
  the list of `good first issues
  <https://github.com/scikit-plots/scikit-plots/labels/good%20first%20issue>`_
  in the issue tracker. Resolving these issues allow you to start contributing
  to the project without much prior knowledge. If you have already contributed
  to scikit-plots, you should look at Easy issues instead.

- **Easy tag**

  If you have already contributed to scikit-plots, another great way to contribute
  to scikit-plots is to pick an item from the list of `Easy issues
  <https://github.com/scikit-plots/scikit-plots/labels/Easy>`_ in the issue
  tracker. Your assistance in this area will be greatly appreciated by the
  more experienced developers as it helps free up their time to concentrate on
  other issues.

- **Help wanted tag**

  We often use the help wanted tag to mark issues regardless of difficulty.
  Additionally, we use the help wanted tag to mark Pull Requests which have been
  abandoned by their original contributor and are available for someone to pick up where
  the original contributor left off. The list of issues with the help wanted tag can be
  found `here <https://github.com/scikit-plots/scikit-plots/labels/help%20wanted>`_.
  Note that not all issues which need contributors will have this tag.

.. _contribute_documentation:

Documentation
=============

We are glad to accept any sort of documentation:


.. _building_documentation:

Building the documentation
--------------------------



.. _generated_doc_CI:

Generated documentation on GitHub Actions
-----------------------------------------


.. _testing_coverage:

Testing and improving test coverage
===================================


.. _monitoring_performances:

Monitoring performance
======================

.. _issue_tracker_tags:

Issue Tracker Tags
==================

All issues and pull requests on the
`GitHub issue tracker <https://github.com/scikit-plots/scikit-plots/issues>`_
should have (at least) one of the following tags:

:Bug:
    Something is happening that clearly shouldn't happen.
    Wrong results as well as unexpected errors from estimators go here.

:Enhancement:
    Improving performance, usability, consistency.

:Documentation:
    Missing, incorrect or sub-standard documentations and examples.

:New Feature:
    Feature requests and pull requests implementing a new feature.

There are four other tags to help new contributors:

:Good first issue:
    This issue is ideal for a first contribution to scikit-plots. Ask for help
    if the formulation is unclear. If you have already contributed to
    scikit-plots, look at Easy issues instead.

:Easy:
    This issue can be tackled without much prior experience.

:Moderate:
    Might need some knowledge of machine learning or the package,
    but is still approachable for someone new to the project.

:Help wanted:
    This tag marks an issue which currently lacks a contributor or a
    PR that needs another contributor to take over the work. These
    issues can range in difficulty, and may not be approachable
    for new contributors. Note that not all issues which need
    contributors will have this tag.

.. _backwards-compatibility:

Maintaining backwards compatibility
===================================

.. _contributing_deprecation:

Deprecation
-----------

If any publicly accessible class, function, method, attribute or parameter is renamed,
we still support the old one for two releases and issue a deprecation warning when it is
called, passed, or accessed.

.. _code_review:

Code Review Guidelines
======================

Reviewing code contributed to the project as PRs is a crucial component of
scikit-plots development. We encourage anyone to start reviewing code of other
developers. The code review process is often highly educational for everybody
involved. This is particularly appropriate if it is a feature you would like to
use, and so can respond critically about whether the PR meets your needs. While
each pull request needs to be signed off by two core developers, you can speed
up this process by providing your feedback.

.. _communication:


Reading the existing code base
==============================

Reading and digesting an existing code base is always a difficult exercise
that takes time and experience to master. Even though we try to write simple
code in general, understanding the code can seem overwhelming at first,
given the sheer size of the project. Here is a list of tips that may help
make this task easier and faster (in no particular order).