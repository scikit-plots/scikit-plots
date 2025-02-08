.. _spep_000:

==============================
SPEP000: SPEP and its workflow
==============================

:Author: Adrin Jalali
:Status: Accepted
:Type: Process
:Created: 2020-02-13

Abstract
########

This SPEP specifies details related to SPEP submission, review, and acceptance
process.

Motivation
##########

Without a predefined workflow, the discussions around a SPEP can be long and
consume a lot of energy for both the author(s) and the reviewers. The lack of a
known workflow also results in the SPEPs to take months (if not years) before
it is merged as ``Under Review``. The purpose of this SPEP is to lubricate and
ease the process of working on a SPEP, and make it a more enjoyable and
productive experience. This SPEP borrows the process used in PEPs and NEPs
which means there will be no ``Under Review`` status.


What is a SPEP?
###############

SPEP stands for Scikit-plots Enhancement Proposal, inspired from Python PEPs or
Numpy NEPs. A SPEP is a design document providing information to the
scikit-plots community, or describing a new feature for scikit-plots or its
processes or environment. The SPEP should provide a concise technical
specification of the proposed solution, and a rationale for the feature.

We intend SPEPs to be the primary mechanisms for proposing major new features,
for collecting community input on an issue, and for documenting the design
decisions that have gone into scikit-plots. The SPEP author is responsible for
building consensus within the community and documenting dissenting opinions.

Because the SPEPs are maintained as text files in a versioned repository, their
revision history is the historical record of the feature proposal.

SPEP Audience
#############

The typical primary audience for SPEPs are the core developers of
``scikit-plots`` and technical committee, as well as contributors to the
project. However, these documents also serve the purpose of documenting the
changes and decisions to help users understand the changes and why they are
made. The SPEPs are available under `Scikit-plots enhancement proposals
<https://scikit-plots.github.io/dev/devel/SEP/index.html>`_.

SPEP Types
##########

There are three kinds of SPEPs:

1. A Standards Track SPEP describes a new feature or implementation for
scikit-plots.

2. An Informational SPEP describes a scikit-plots design issue, or provides
general guidelines or information to the scikit-plots community, but does not
propose a new feature. Informational SPEPs do not necessarily represent a
scikit-plots community consensus or recommendation, so users and implementers
are free to ignore Informational SPEPs or follow their advice. For instance, an
informational SPEP could be one explaining how people can write a third party
estimator, one to explain the usual process of adding a package to the contrib
org, or what our inclusion criteria are for scikit-plots and
scikit-plots-extra.

3. A Process SPEP describes a process surrounding scikit-plots, or proposes a
change to (or an event in) a process. Process SPEPs are like Standards Track
SPEPs but apply to areas other than the scikit-plots library itself. They may
propose an implementation, but not to scikit-plotsâ€™s codebase; they require
community consensus. Examples include procedures, guidelines, changes to the
decision-making process and the governance document, and changes to the tools
or environment used in scikit-plots development. Any meta-SPEP is also
considered a Process SPEP.


SPEP Workflow
#############

A SPEP starts with an idea, which usually is discussed in an issue or a pull
request on the main repo before submitting a SPEP. It is generally a good idea
for the author of the SPEP to gauge the viability and the interest of the
community before working on a SPEP, mostly to save author's time.

A SPEP must have one or more champions: people who write the SPEP following the
SPEP template, shepherd the discussions around it, and seek consensus in the
community.

The proposal should be submitted as a draft SPEP via a GitHub pull request to a
``spepXXX`` directory with the name ``proposal.rst`` where ``XXX`` is an
appropriately assigned three-digit number (e.g., ``spep000/proposal.rst``). The
draft must use the `SPEP â€” Template and Instructions
<https://github.com/scikit-plots/scikit-plots/blob/main/docs/source/devel/SPEP/template.rst>`_
file.

Once the PR for the SPEP is created, a post should be made to the mailing list
containing the sections up to â€œBackward compatibilityâ€, with the purpose of
limiting discussion there to usage and impact. Discussion on the pull request
will have a broader scope, also including details of implementation.

The first draft of the SPEP needs to be approved by at least one core developer
before being merged. Merging the draft does not mean it is accepted or is ready
for the vote. To this end, the SPEP draft is reviewed for structure,
formatting, and other errors. Approval criteria are:

- The draft is sound and complete. The ideas must make technical sense.
- The initial PR reviewer(s) should not consider whether the SPEP seems likely
  to be accepted.
- The title of the SPEP draft accurately describes its content.

Reviewers are generally quite lenient about this initial review, expecting that
problems will be corrected by the further reviewing process. **Note**: Approval
of the SPEP draft is no guarantee that there are no embarrassing mistakes!
Ideally they're avoided, but they can also be fixed later in separate PRs. Once
approved by at least one core developer, the SPEP draft can be merged.
Additional PRs may be made by the champions to update or expand the SPEP, or by
maintainers to set its status, discussion URL, etc.

Standards Track SPEPs (see bellow) consist of two parts, a design document and
a reference implementation. It is generally recommended that at least a
prototype implementation be co-developed with the SPEP, as ideas that sound
good in principle sometimes turn out to be impractical when subjected to the
test of implementation. Often it makes sense for the prototype implementation
to be made available as PR to the scikit-plots repo (making sure to
appropriately mark the PR as a WIP).

Review and Resolution
---------------------

SPEPs are discussed on the mailing list or the PRs modifying the SPEP. The
possible paths of the status of SPEPs are as follows:

.. image:: pep-0001-process_flow.png
   :alt: SPEP process flow diagram

All SPEPs should be created with the ``Draft`` status.

Eventually, after discussion, there may be a consensus that the SPEP should be
accepted â€“ see the next section for details. At this point the status becomes
``Accepted``.

Once a SPEP has been ``Accepted``, the reference implementation must be
completed. When the reference implementation is complete and incorporated into
the main source code repository, the status will be changed to ``Final``. Since
most SPEPs deal with a part of scikit-plots's API, another way of viewing a
SPEP as ``Final`` is when its corresponding API interface is considered stable.

To allow gathering of additional design and interface feedback before
committing to long term stability for a feature or API, a SPEP may also be
marked as ``Provisional``. This is short for "Provisionally Accepted", and
indicates that the proposal has been accepted for inclusion in the reference
implementation, but additional user feedback is needed before the full design
can be considered ``Final``. Unlike regular accepted SPEPs, provisionally
accepted SPEPs may still be ``Rejected`` or ``Withdrawn`` even after the
related changes have been included in a scikit-plots release.

Wherever possible, it is considered preferable to reduce the scope of a
proposal to avoid the need to rely on the ``Provisional`` status (e.g. by
deferring some features to later SPEPs), as this status can lead to version
compatibility challenges in the wider scikit-plots ecosystem.

A SPEP can also be assigned status ``Deferred``. The SPEP author or a core
developer can assign the SPEP this status when no progress is being made on the
SPEP.

A SPEP can also be ``Rejected``. Perhaps after all is said and done it was not
a good idea. It is still important to have a record of this fact. The
``Withdrawn`` status is similar; it means that the SPEP author themselves has
decided that the SPEP is actually a bad idea, or has accepted that a competing
proposal is a better alternative.

When a SPEP is ``Accepted``, ``Rejected``, or ``Withdrawn``, the SPEP should be
updated accordingly. In addition to updating the status field, at the very
least the ``Resolution`` header should be added with a link to the relevant
thread in the mailing list archives or where the discussion happened.

SPEPs can also be ``Superseded`` by a different SPEP, rendering the original
obsolete. The ``Replaced-By`` and ``Replaces`` headers should be added to the
original and new SPEPs respectively.

``Process`` SPEPs may also have a status of ``Active`` if they are never meant
to be completed, e.g. SPEP 1 (this SPEP).

How a SPEP becomes Accepted
---------------------------

A SPEP is ``Accepted`` by the voting mechanism defined in the `governance model
<https://scikit-plots.github.io/dev/project/governance.html>`_. We
need a concrete way to tell whether consensus has been reached. When you think
a SPEP is ready to accept, create a PR changing the status of the SPEP to
``Accepted``, then send an email to the scikit-plots mailing list with a
subject like:

    [VOTE] Proposal to accept SPEP #<number>: <title>

In the body of your email, you should:

- link to the latest version of the SPEP, and a link to the PR accepting the
  SPEP.

- briefly describe any major points of contention and how they were resolved,

- include a sentence like: â€œThe vote will be closed in a month i.e. on
  <the_date>.â€

Generally the SPEP author will be the one to send this email, but anyone can do
it; the important thing is to make sure that everyone knows when a SPEP is on
the verge of acceptance, and give them a final chance to respond.

In general, the goal is to make sure that the community has consensus, not
provide a rigid policy for people to try to game. When in doubt, err on the
side of asking for more feedback and looking for opportunities to compromise.

If the final comment and voting period passes with the required majority, then
the SPEP can officially be marked ``Accepted``. The ``Resolution`` header
should link to the PR accepting the SPEP.

If the vote does not achieve a required majority, then the SPEP remains in
``Draft`` state, discussion continues as normal, and it can be proposed for
acceptance again later once the objections are resolved.

In unusual cases, with the request of the author, the scikit-plots technical
committee may be asked to decide whether a controversial SPEP is ``Accepted``,
put back to ``Draft`` with additional recommendation to try again to reach
consensus or definitely ``Rejected``. Please refer to the governance doc for
more details.

Maintenance
-----------

In general, Standards track SPEPs are no longer modified after they have
reached the ``Final`` state as the code and project documentation are
considered the ultimate reference for the implemented feature. However,
finalized Standards track SPEPs may be updated as needed.

Process SPEPs may be updated over time to reflect changes to development
practices and other details. The precise process followed in these cases will
depend on the nature and purpose of the SPEP being updated.

Format and Template
-------------------

SPEPs are UTF-8 encoded text files using the `reStructuredText
<http://docutils.sourceforge.net/rst.html>`_ format. Please see the `SPEP â€”
Template and Instructions
<https://github.com/scikit-plots/scikit-plots/blob/main/docs/source/devel/SPEP/template.rst>`_
file and the `reStructuredTextPrimer
<https://www.sphinx-doc.org/en/stable/rest.html>`_ for more information. We use
`Sphinx <https://www.sphinx-doc.org/en/stable/>`_ to convert SPEPs to HTML for
viewing on the web.

Header Preamble
---------------

Each SPEP must begin with a header preamble. The headers must appear in the
following order. Headers marked with * are optional. All other headers are
required::

      :Author: <list of authors' real names and optionally, email addresses>
      :Status: <Draft | Active | Accepted | Deferred | Rejected |
               Withdrawn | Final | Superseded>
      :Type: <Standards Track | Informational | Process>
      :Created: <date created on, in yyyy-mm-dd format>
    * :Requires: <spep numbers>
    * :scikit-plots-Version: <version number>
    * :Replaces: <spep number>
    * :Replaced-By: <spep number>
    * :Resolution: <url>

The Author header lists the names, and optionally the email addresses of all
the authors of the SPEP. The format of the Author header value must be

    Random J. User <address@dom.ain>

if the email address is included, and just

    Random J. User

if the address is not given. If there are multiple authors, each should be on a
separate line.

Copyright
---------

This document has been placed in the public domain [1]_.

References and Footnotes
------------------------

.. [1] Open Publication License: https://www.opencontent.org/openpub/
