.. _quickstart_contributing:

======================================================================
Quickstart Contributing Guidelines
======================================================================

.. _contributing_environment:

Creating a development environment
==================================

To make and test code changes and build the documentation locally you will need to
create a development environment. If you run into problems at any stage do not hesitate
to `TEMPLATE ask for help <https://scikit-plots.github.io/dev/help.html>`_.

Set up GitHub and Git
---------------------

scikit-plots is hosted on `GitHub <https://github.com/scikit-plots/scikit-plots>`_, and to
contribute, you will need a `GitHub account
<https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github>`_.

We use `Git <https://git-scm.com/>`_ for version control and to allow many people to
work together on the project. See the `GitHub quickstart instructions
<https://docs.github.com/en/get-started/quickstart/set-up-git>`__ for installing and
configuring git, as well as the :ref:`git-resources` page.

If you are new to contributing to projects through forking on GitHub, see the
`GitHub documentation for contributing to projects
<https://docs.github.com/en/get-started/quickstart/contributing-to-projects>`_.


.. important::

  If you dont want to install any packages on your local you can use
  ``Setup Docker Pre-Installed Env`` via :ref:`docker-index`.


Install a C compiler if needed
------------------------------

How to do this will depend on your platform.

**Windows**

You will need `Build Tools for Visual Studio
<https://visualstudio.microsoft.com/downloads/?q=build+tools>`_.

.. note::
    You DO NOT need to install Visual Studio.
    You only need "Build Tools for Visual Studio" found by
    scrolling down to "All downloads" -> "Tools for Visual Studio" -> "Build Tools
    for Visual Studio".

Alternative options include:

- Install the necessary components on the command line using `vs_BuildTools.exe
  <https://learn.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio?source=recommendations&view=vs-2022>`_.
- Use the `WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

**MacOS**

Install the Developer Tools using ``xcode-select --install``. There is no need to
install the full Xcode application and this command will install only the command line
tools and developer utilities.

Further details and related information can be found at
https://devguide.python.org/setup/#macos.

**Linux**

For Linux-based installations, you won't have to install any additional components.

.. _contributing_forking:

Create a clone of scikit-plots
==================================

If you have not done so already, you will need your own copy of ``scikit-plots`` to
build it and/or contribute to the source. scikit-plots is hosted in the
`scikit-plots GitHub repository <https://www.github.com/scikit-plots/scikit-plots>`_
and you need to make a clone.

First, create a `GitHub Fork
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_
by going to the `scikit-plots project page <https://github.com/scikit-plots/scikit-plots>`_
and hitting the ``Fork`` button.

Next, `clone <https://git-scm.com/docs/git-clone>`__ your GitHub fork to your machine:

.. code-block:: shell

    ## Forked repo: https://github.com/scikit-plots/scikit-plots.git
    git clone https://github.com/YOUR-USER-NAME/scikit-plots.git
    cd scikit-plots

Initialize and Fetch Submodules (Not Needed Every Time):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    ## Or use `git config ...` to add `scikit-plots` in git safe dirs
    bash docker/script/safe_dirs.sh  # add safe directories for git

    ## Initialize and clone any missing submodules, set up the working tree
    ## Almost always used after cloning a repo with submodules.
    git submodule update --init --recursive

    ## Update submodules to the latest commit on their configured remote branch
    ## Used when you want your submodules to move to their latest remote commit.
    git submodule update --remote --recursive # (not needed every time)

Adding and Fetching Upstream Remote:

.. code-block:: shell

    git remote add upstream https://github.com/scikit-plots/scikit-plots.git
    git fetch upstream --tags

This creates the directory ``scikit-plots`` and connects your repository to the upstream
(main project) `scikit-plots <https://github.com/scikit-plots/scikit-plots>`__ repository.

You can see the remote repositories as follows::

    >>> git remote --verbose

You will see something like::

    >>> origin  https://github.com/YOUR-USER-NAME/scikit-plots (fetch)
    >>> origin  https://github.com/YOUR-USER-NAME/scikit-plots (push)
    >>> upstream        https://github.com/scikit-plots/scikit-plots.git (fetch)
    >>> upstream        https://github.com/scikit-plots/scikit-plots.git (push)

.. _create_isolated_env:

Create an isolated development environment
============================================

A key requirement is to have an isolated Python environment, meaning that it is
isolated from both your system Python and any other Python environments you may have
for doing other work. This is important because the development environment will often
be unstable and possibly broken at times, and you don't want to break your other work.

There are many good options for doing this, including a number of virtual environment
managers (e.g., the Python standard library `venv <https://docs.python.org/3/library/venv.html>`_
module). Users who have a preference for a particular virtual environment manager are
encouraged to use it!

For this quickstart guide we use the `conda <https://docs.conda.io/en/latest/>`_ package
manager provided by `miniforge <https://github.com/conda-forge/miniforge>`_. This is a
popular choice and generally works well, especially for newcomers. It is easy to install
and use on all platforms and it makes it easy to install different Python versions which
can be useful for testing.

Install miniforge and conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you do not already have ``conda`` installed, `download and install miniforge
<https://github.com/conda-forge/miniforge/blob/main/README.md>`_. The details depend on
your system but the end result is to provide a ``conda`` executable that you can use
to create and manage isolated Python environments.

Now create and activate an ``py311`` conda environment using the following::

   >>> # (Optionally) Can be run on `conda base` or `venv` env
   >>> mamba create -n py311 python=3.11 ipykernel graphviz -y
   >>> conda activate py311

Note the ``graphviz`` package is required for building the documentation.

Install the development version of scikit-plots
-----------------------------------------------

Now you can install the development version of ``scikit-plots`` into your new environment. This
will install the latest version of ``scikit-plots`` from your local git repo, along with
all the dependencies needed to build and fully test ``scikit-plots``::

   >>> ## (Optionally) setup one line scikit-plots development version
   >>> make dev

   >>> ## Setup scikit-plots lib dep
   >>> # pip install -r ./requirements/all.txt
   >>> # pip install -r ./requirements/cpu.txt
   >>> pip install -r ./requirements/build.txt

   >>> ## Setup scikit-plots lib dep
   >>> pip install --no-build-isolation --no-cache-dir -e . -v

   >>> ## (Optionally) It is also possible to include optional dependencies:
   >>> ## cpu refer tensorflow-cpu, keras, transformers
   >>> python -m pip install --no-build-isolation --no-cache-dir -e .[build,dev,test,doc] -v

.. _contributing_check_build:

Checking the build scikit-plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At this point you should be able to ``import scikitplot`` from your locally built version::

   >>> python -c 'import scikitplot; scikitplot.show_config()'

   >>> ## Checking the build without root
   >>> bash -c "cd ~ && python -c 'import scikitplot; scikitplot.show_config()'"

Next you may want to try running some or all of the ``scikitplot`` unit tests.
Running the full test suite can take a few minutes, so you may want to start with a
single sub-package (e.g. :ref:`kds-index`)::


   >>> ## python -m pytest -p vscode_pytest --collect-only --rootdir=.
   >>> ## run a sub set of the test suite
   >>> pytest scikitplot/kds

   >>> ## or the whole suite
   >>> pytest

Details on running and writing tests can be found in the :ref:`testing-guidelines`
section.

.. _contributing_pre_commit:

Install pre-commit
------------------

This is optional, but *highly recommended*. `Pre-commit <https://pre-commit.com/>`_ is a
tool that runs a number of `Continuous Integration (CI) <contributing_ci>`_ checks
(e.g. code formatting) on your code before you commit it. If you skip this step then it
is likely that one or more of those CI checks will fail when you make a pull request,
resulting in lost time (yours and CI resources).

(Recommended) Installation is straightforward. From the root of the ``scikit-plots`` repository, run::

    >>> ## It triggered when committing `git commit ...` if pass then next pushing changes
    >>> pre-commit install

(Optionally) Manually one-by-one testing (not needed every time)::

    >>> ## (Optionally) Manually one-by-one testing:
    >>> ## If the test is successful one by one
    >>> ## but there is an error in the entire test, check the inconsistency
    >>> # pre-commit run ruff-format --verbose # (with `--all-files` option not recommended)
    >>> pre-commit run ruff
    >>> pre-commit run black

(Optionally) Update and reinstall pre-commit hooks (not needed every time)::

    >>> ## (Optionally) Update and reinstall pre-commit hooks (not needed every time), If Needed
    >>> pre-commit autoupdate  # (not needed every time)
    >>> pre-commit clean && pre-commit install
    >>> pre-commit run # (with `--all-files` option not recommended)

Now all of the styling checks will be run each time you commit changes, ensuring that
the CI formatting checks for your :ref:`pull request <quickstart-pull-request>` will
pass.

.. tip:: To learn more about pre-commit, see the :ref:`pre-commit` section.

.. _contributing_pull_request:

Creating and submitting a pull request
======================================

You can contribute bug fixes, new features, and documentation updates by submitting a
GitHub pull request (PR). This section will guide you through the process. We encourage
you to `ask for help <https://github.com/orgs/scikit-plots/discussions>`_ if you get stuck.
The ``scikit-plots`` community is welcoming and friendly and will help you!

If you are new to the ``scikit-plots`` Project and interested to submit a large patch
(e.g., a new big feature or significant refactoring), we encourage you to first
discuss your ideas on GitHub to increase the chance of your PR
being accepted.

Creating a branch
-----------------

Your local ``main`` branch should always reflect the current state of ``scikit-plots`` repository.
First ensure it's up-to-date with the ``main`` ``scikit-plots`` repository::

    >>> # git checkout main
    >>> git switch main

(Recommended) Use Fast-forward only:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fast-forward only: Updates your branch only if it can be fast-forwarded (i.e., no local commits that diverge from upstream/main).
- If divergence exists: Git refuses to pull, and you must manually resolve the situation (e.g., using git rebase or git merge).
- Use case: When you want a clean history and are sure your local branch is either up-to-date or strictly behind upstream/main.

Use when you have no local commits diverging from upstream/main::

    >>> ## When you just want to update and have no local commits diverging from upstream.
    >>> git pull upstream main --ff-only

::

    ## Example Scenario Before Fast-forward only:

    A---B---C---D---E---F  (upstream/main)
             \
              (local main)

    ## ✅ Works: After git pull upstream main --ff-only, local main becomes:

    A---B---C---D---E---F  (local main = upstream/main)

    ## ❌ Fails (Diverging History): You've made local commits (X and Y), but upstream has new commits (D, E, F):
    ## Because the histories have diverged, Git refuses to merge since a fast-forward isn't possible.

    A---B---C---D---E---F  (upstream/main)
             \
              X---Y  (local main)  ❌ (Fast-forward not possible)

(Optionally) Rebases:
^^^^^^^^^^^^^^^^^^^^^^

- Rebases your local commits on top of the latest upstream/main
- Rewrites history by replaying your commits on top of upstream/main, making the history linear
- Use case: When you want to keep a clean history while incorporating upstream changes without a merge commit.
- Since no local commits exist, both commands do the same thing: fast-forward the branch.

Use when you have local commits and want to apply them on top of upstream/main while keeping a linear history::

    >>> ## When you do have local commits and want to apply them cleanly on top of the latest upstream changes.
    >>> git pull upstream main --rebase

::

    ## Example Scenario Before Rebasing:

    A---B---C---D---E---F  (upstream/main)
             \
              X---Y  (local main)

    ## ✅ Works: After git pull upstream main --rebase, local main becomes:
    ## ✅ Rewrites history, replaying local commits on top of upstream/main:
    ## Your commits (X and Y) are reapplied on top of F, creating new commits (X' and Y' with new hashes).

    A---B---C---D---E---F---X'---Y'  (rebased local main)

..
    >>> ## Download Updates Only: forked a repository and want to fetch the latest changes
    >>> ## from the original repository (upstream) into your local copy.
    >>> git fetch upstream main

    >>> ## merge or rebase to apply those updates.
    >>> ## Does not modify existing commits.
    >>> git merge upstream/main
    >>> ## Rewrites commit history. Avoids unnecessary merge commits.
    >>> # git rebase upstream/main

To view the commit history in Git, you can use the following commands::

    >>> git log --pretty=format:"%h - %an, %ar : %s" -n 9
    >>> git log --oneline --graph --decorate --all -n 9

Now create a development branch for making your changes. For example::

    >>> git switch -c subpackage-bug-fix

Delete the branch locally::

    >>> # git checkout main && git pull
    >>> git switch main
    >>> git branch -d subpackage-bug-fix

This changes your working branch from ``main`` to the ``subpackage-bug-fix`` branch.
Keep any changes in this branch specific to one bug or feature so it is clear what the
branch brings to ``scikit-plots``. You can have many feature branches and switch in between them
using the `git switch <https://git-scm.com/docs/git-switch>`_ command.

Using a descriptive branch name can help you stay organized. For example
```kds-commented-header``` might be a good name for a branch that fixes the
commented header issue `#1 <https://github.com/scikit-plots/scikit-plots/issues/1>`_ in
the ``kds`` sub-package.

When you want to update the feature branch with changes in main after
you created the branch, check the section on
:ref:`updating a PR <contributing_update_pr>`.

.. _contributing_commit_code:

Making code or documentation changes
============================================

Now comes the fun part where you use your favorite editor or IDE to make changes to the
code or documentation! At a high level this breaks into a few parts:

- **Make changes**: Make the changes you want to make. This could be fixing a bug,
  adding a new feature, or updating the documentation.
- **Test changes**: For code changes, ensure that they work as expected following the
  process outlined in the :ref:`testing-guidelines` section.
- **Build documentation**: If you are updating the documentation, you will want to
  :ref:`build the documentation <builddocs>` to ensure that it looks good.
- **Add a changelog entry**: For most code changes you will need to
  :ref:`add-changelog`.

.. tip:: For more information and examples see :ref:`edit-flow` section.

You can see a summary of the changes you've currently made by running:

.. code-block:: shell

    git status

You can then commit your all your changes to your local repository with an explanatory
`commit message <https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html>`_:

.. code-block:: shell

    git add files-that-you-changed ...
    git commit -m "your commit message goes here"

.. Important:: Never merge changes from ``upstream/main`` into your feature branch. If
   changes in ``main`` require changes to our code you must :ref:`rebase`.

.. _contributing_push_code:

Pushing your changes
--------------------

When you want your changes to appear publicly on your GitHub page, push your
forked feature branch's commits::

    >>> git push origin --set-upstream subpackage-bug-fix

Here ``origin`` is the default name given to your fork on GitHub.

Now your code is on GitHub, but it is not visible to the ``scikit-plots`` maintainers. For that
to happen, a pull request needs to be submitted on GitHub.

The first time you push to a new branch on GitHub, you will see a message like below
with a useful link to create a pull request::

  >>> remote: Create a pull request for 'subpackage-bug-fix' on GitHub by visiting:
  >>> remote:      https://github.com/YOUR-USER-NAME/scikit-plots/pull/new/subpackage-bug-fix


.. _quickstart_pull_request:

Making a pull request
---------------------

If everything looks good, you are ready to make a pull request (PR). A PR is how
code from your local repository becomes available to the GitHub community to review and
merged into project to appear the in the next release.

Most of the time you can just follow the link that ``git`` provided when you pushed
your branch and create the PR. If you don't have that link (and for a few more details),
you can follow the :ref:`pull-request` instructions.

Follow the instructions in the PR template and fill it out as completely as possible.

If your PR is still a work in progress then instead of clicking "Create pull request",
click on the small down arrow next to it and select "`Create draft pull request
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests>`__".
In addition, if your commits are not ready for CI testing, you
should include ``[ci skip]`` the last commit message – but note that code formatting
checks and documentation building will still be done. Formatting and style errors *should*
already have been fixed before committing if you have locally
:ref:`installed pre-commit<contributing_pre_commit>`; but if you have not,
you can use the :ref:`pre-commit_bot` to fix them automatically in the PR.

Once submitted (and marked as ready), this request goes to the ``scikit-plots`` maintainers and
they will review the PR.

.. _contributing_update_pr:

Updating your pull request
--------------------------

Based on the review you get on your pull request, you will probably need to make
some adjustments. You can follow the :ref:`code committing steps <contributing_commit_code>`
again to address any feedback and update your pull request::

    >>> git push origin subpackage-bug-fix

Any ``git push`` will automatically update your pull request with your branch's changes
and restart the `Continuous Integration <contributing_ci>`_ checks.

.. Important:: At this point please read (or at least skim) the sections :ref:`revise
    and push`, :ref:`rebase`, and :ref:`squash-if-necessary`. The information here
    covers situations that happen on occasion and can be cause trouble. As always if
    you have questions, ask for help from the maintainer reviewing your PR.

..
    git branch -d subpackage-bug-fix
    git fetch origin  # Get the latest changes from remote
    git reset --hard origin/<branch-name>  # Reset local branch to match remote

Tips for a successful pull request
----------------------------------

If you have made it to this point and submitted a pull request, one of the core
maintainers will take a look. To make the process as smooth and efficient as possible,
here are some tips:

- **Reference any existing open issue** to `link to that issue
  <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests>`_ and close the
  issue if the PR is merged.
- **Ensure you have appropriate tests**.
- **Keep your pull requests as simple as possible** -- larger PRs take longer to review.
- **When practical, limit the scope of a PR to one sub-package** -- this means fewer
  required reviewers and a faster review process.
- **Ensure that CI is in a green state** -- any required failures should be addressed.

.. _contributing_ci: https://pandas.pydata.org/pandas-docs/stable/development/contributing_codebase.html#contributing-ci
