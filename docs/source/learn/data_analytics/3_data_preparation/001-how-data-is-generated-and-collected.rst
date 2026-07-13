:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-001:
.. _data-analytics-prep-001:
.. _da-foundations-prep-001:
.. _da-decisions-prep-001:
.. _da-prep-prep-001:
.. _da-cleaning-prep-001:
.. _da-analyze-prep-001:
.. _da-viz-prep-001:
.. _da-python-prep-001:
.. _da-jobsearch-prep-001:

========================================================================
How Data Is Generated and Collected
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`🧬 Data Types & Structure` :bdg-info:`Lesson 001`

:doc:`Next <002-choosing-the-right-data-to-collect>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Where data comes from
-----------------------

Section 2 treated data as something you *have*; this section starts one step
earlier, with where it *comes from*. Understanding how data is generated and
collected is the foundation of the Prepare phase, because the origin of a
dataset determines what it can honestly be used for — a point the context and
bias lessons already foreshadowed.

How data is generated
-----------------------

Data comes into existence in a few characteristic ways:

- **Observational** — recording what happens without intervening: transactions
  as they occur, clicks as users browse, sensor readings over time. Most
  business data is observational, and it shows what *did* happen, not
  necessarily what *causes* what.
- **Experimental** — deliberately varying something and measuring the result:
  the A/B test that shows two homepage designs to comparable groups.
  Experiments are what let analysis speak about *causes* rather than only
  associations.
- **Self-reported** — people telling you directly: surveys, forms,
  registrations. Rich and often the only route to *why*, but filtered through
  memory, honesty, and who chose to respond.
- **Derived** — computed from other data: a "customer lifetime value" field
  built from transaction history. Convenient, but only as sound as its inputs
  and its formula.

Sources: first-, second-, and third-party
-------------------------------------------

Independently of *how* it is generated, data is classified by *whose* it is:

- **First-party** — collected by your own organisation directly from its own
  activity and customers. Usually the most trustworthy and relevant, because
  you control and understand its collection.
- **Second-party** — another organisation's first-party data, obtained
  directly from them through a partnership. Its quality depends on their
  collection practices, which you must ask about.
- **Third-party** — aggregated and sold by an entity that did not collect it
  from the original source. Broad and convenient, but its provenance and
  quality are the hardest to verify — treat with corresponding caution.

The reliability gradient generally runs first → second → third-party, and it
maps directly onto how much you can know about the collection context.

Why origin governs use
------------------------

Every downstream question about a dataset traces to its origin. *Can this show
causation?* — only if it was experimental. *Does it represent all customers?* —
only if collection reached them all. *Can I trust the definitions?* — most where
you controlled collection, least where a third party did. Knowing generation
and source is how you answer these before, not after, building an analysis on
the data.

The caveat
------------

Origin is often *undocumented* — data arrives without a clear record of how or
by whom it was collected, and reconstructing that is real detective work. When
origin cannot be established, that uncertainty is itself a finding to state, not
a detail to gloss: an analysis built on data of unknown provenance inherits
unknown risk. The next lesson turns from where data comes from to *which* data
a question actually needs.

.. hint::

   - :doc:`Understanding the Data Ecosystem <../1_foundations/007-understanding-the-data-ecosystem>`
   - :doc:`Choosing the Right Data to Collect <002-choosing-the-right-data-to-collect>`
   - :doc:`Accessing Data: Internal and External Sources <019-accessing-data-internal-and-external-sources>`
   - :doc:`Quantitative and Qualitative Data in Decision-Making <../2_data_driven_decisions/006-quantitative-and-qualitative-data-in-decision-making>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/how-data-is-generated-and-collected/ <https://insightful-data-lab.com/2023/09/04/how-data-is-generated-and-collected/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: types
