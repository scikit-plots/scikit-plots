:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-015:
.. _da-3-prep-prep-015:

========================================================================
Open Data and Openness in Data Ethics
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`⚖️ Bias & Data Ethics` :bdg-info:`Lesson 015`

◀ :doc:`Previous <014-data-privacy-in-data-ethics>` · :doc:`Next <016-databases-and-relational-database-concepts>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


The case for openness
-----------------------

Privacy argues for closing data down; **openness** argues, in the right cases,
for opening it up. **Open data** is data that is freely available for anyone to
access, use, and share. The idea rests on a genuine public good: data — especially
data gathered with public money or of public importance — can create more value
when many people can use it than when it is locked away, powering research,
transparency, innovation, and accountability.

What open data enables
------------------------

- **Research and innovation** — open datasets let researchers, entrepreneurs,
  and analysts build on each other's work rather than each collecting from
  scratch. Much of science and many products rest on shared data.
- **Transparency and accountability** — open government data (budgets, outcomes,
  performance) lets citizens and journalists hold institutions to account, which
  is why the public-service sector often carries an obligation to publish.
- **A common resource** — freely available data is infrastructure, like public
  roads: broadly useful precisely because it is not fenced off.

For data to be genuinely open, it typically must be not only *free of charge*
but *usably* available — in accessible formats, with documentation, under
licences that permit reuse. Data that is technically public but trapped in
unusable form is open in name only.

The tension with privacy
--------------------------

Openness and privacy pull in opposite directions, and the conflict is real, not
resolvable by slogan. Open data about *institutions* (how a government spends,
how a company performs) serves accountability. Open data about *individuals*
threatens privacy — and the danger is that "anonymised" open datasets can be
re-identified, exactly the failure mode from the privacy lesson, now at public
scale and irreversible once released. The governing principle: openness is a
virtue for data about *institutions and the aggregate*; personal data requires
privacy protection first, and openness only after genuine, robust
de-identification — if at all.

Open data in the analyst's work
---------------------------------

Openness cuts two ways for a working analyst. As a *consumer*, open data is a
valuable source — government statistics, public research data, open civic
datasets — to be evaluated with the same ROCCC rigour as any other source
(open does not mean reliable). As a *producer*, sharing methods and
non-sensitive data openly makes analysis reproducible and trustworthy, the
transparency the foundations valued — while sharing anything derived from
personal data demands the privacy safeguards of the previous lesson.

The caveat
------------

"Open" is not an unqualified good, and neither is "closed". Some data should be
open (public accountability), some must stay protected (personal privacy), and
much sits in a contested middle where reasonable people weigh public benefit
against individual risk differently. The ethical stance is not a blanket
preference either way but a *case-by-case* judgement: what is the benefit of
openness here, who bears the risk, and can the risk be genuinely mitigated? This
closes the bias-and-ethics stage; the next turns to the concrete systems where
organisational data lives — relational databases.

.. hint::

   - :doc:`Data Ethics in Data Analysis <013-data-ethics-in-data-analysis>`
   - :doc:`Data Privacy in Data Ethics <014-data-privacy-in-data-ethics>`
   - :doc:`How Data Is Generated and Collected <001-how-data-is-generated-and-collected>`
   - :doc:`Accessing Data: Internal and External Sources <019-accessing-data-internal-and-external-sources>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/open-data-and-openness-in-data-ethics/ <https://insightful-data-lab.com/2023/09/04/open-data-and-openness-in-data-ethics/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: bias_ethics
