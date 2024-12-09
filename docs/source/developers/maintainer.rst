Maintainer Information
======================

.. warning::

    Not Implemented...

**Reserved.**

Releasing
---------

This section is about preparing a major/minor release, a release candidate (RC), or a
bug-fix release. We follow `PEP440 <https://www.python.org/dev/peps/pep-0440/>`_ for
the version scheme and to indicate different types of releases. Our convention is to
follow the "major.minor.micro" scheme, although in practice there is no fundamental
difference between major and minor releases and micro releases are bug-fix releases.

We adopted the following release schedule:

- Major/Minor releases every 6 months, usually in May and November. These releases
  are numbered ``X.Y.0`` and are preceded by one or more release candidates ``X.Y.0rcN``.
- Bug-fix releases are done as needed between major/minor releases and only apply to
  the last stable version. These releases are numbered ``X.Y.Z``.

.. rubric:: Preparation

- Confirm that all blockers tagged for the milestone have been resolved, and that other
  issues tagged for the milestone can be postponed.

- Make sure the deprecations, FIXMEs, and TODOs tagged for the release have been taken
  care of.

- For major/minor final releases, make sure that a *Release Highlights* page has been
  done as a runnable example and check that its HTML rendering looks correct. It should
  be linked from the what's new file for the new version of scikit-learn.

- Ensure that the changelog and commits correspond, and that the changelog is reasonably
  well curated. In particular, make sure that the changelog entries are labeled and
  ordered within each section. The order of the labels should be ``|MajorFeature|``,
  ``|Feature|``, ``|Efficiency|``, ``|Enhancement|``, ``|Fix|``, and ``|API|``.

.. rubric:: Permissions

- The release manager must be a **maintainer** of the
  https://github.com/scikit-plots/scikit-plots repository to be able to publish on
  `PyPI <https://pypi.org/>`_ and `Test PyPI <https://test.pypi.org/>`_
  (via a manual trigger of a dedicated Github Actions workflow).

- The release manager must be a **maintainer** of the
  https://github.com/conda-forge/scikit-plots-feedstock repository to be able to publish
  on ``conda-forge``. This can be changed by editing the ``recipe/meta.yaml`` file in the
  first release pull request.

Reference Steps
^^^^^^^^^^^^^^^