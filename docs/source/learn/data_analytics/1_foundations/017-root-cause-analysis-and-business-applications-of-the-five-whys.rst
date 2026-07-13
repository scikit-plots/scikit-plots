:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-foundations-017:
.. _data-analytics-foundations-017:
.. _da-foundations-foundations-017:
.. _da-decisions-foundations-017:
.. _da-prep-foundations-017:
.. _da-cleaning-foundations-017:
.. _da-analyze-foundations-017:
.. _da-viz-foundations-017:
.. _da-python-foundations-017:
.. _da-jobsearch-foundations-017:

========================================================================
Root Cause Analysis and Business Applications of the Five Whys
========================================================================

:bdg-primary:`🌱 Foundations` :bdg-secondary:`🧠 Analytical Skills & Thinking` :bdg-info:`Lesson 017`

◀ :doc:`Previous <016-analytical-thinking-and-questions-for-problem-solving>` · :doc:`Next <018-data-driven-decision-making-and-the-role-of-analytical-skills>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Fix causes, not symptoms
--------------------------

**Root cause analysis** is the discipline of finding the fundamental reason a
problem occurs, so the fix prevents recurrence instead of mopping up. Its most
famous tool, the **five whys**, was developed by **Sakichi Toyoda**, founder of
Toyota Industries, in the 1930s, and became a pillar of the Toyota Production
System — whose architect, Taiichi Ohno, called it *"the basis of Toyota's
scientific approach: by repeating why five times, the nature of the problem as
well as its solution becomes clear."* From Toyota it spread into Kaizen, lean
manufacturing, and Six Sigma, and from there into general business practice.

How it works
--------------

State the problem concretely, ask *why* it happened, then ask *why* of each
answer in turn — about five layers, though the real stopping rule is reaching a
cause you can act on. The classic factory illustration:

  There is a puddle on the floor. **Why?** The overhead pipe is leaking.
  **Why?** Water pressure is too high. **Why?** A control valve is faulty.
  **Why?** Valves are not being tested. **Why?** Valve testing is not on the
  maintenance schedule.

Mopping the puddle fixes nothing; even replacing the valve only fixes *this*
valve. Adding valve testing to the maintenance schedule — the root — prevents
the whole class of puddles. Notice the answers move from *things* (puddle,
pipe) toward **process** (testing, scheduling): root causes usually live in
processes, which is why fixing them sticks.

Business applications
-----------------------

The tool transfers directly to analyst work because data supplies the *evidence
for each why*. Churn rose — why? Cancellations spiked in month two of
subscriptions (the data shows it). Why? Users who never completed onboarding
cancel at triple the rate (the data shows it). Why? The onboarding email
sequence broke for one signup path (logs show it). Each link is a checkable
claim, and the chain ends at a fixable process. The five whys also structures
the *Ask* phase: run it on the stakeholder's stated problem before touching
data, and you often discover the question behind the question.

Honest limits
---------------

The technique has documented criticisms, and good practitioners know them. It
can be **arbitrarily shallow** — five is a habit, not a law — and it tends to
surface a **single causal chain** when real problems often have several
contributing causes. Its results depend on the asker's knowledge: you cannot
"why" your way past what nobody in the room understands. The remedies are
practical: verify each link with data rather than plausibility, branch when an
answer has two credible causes (why-trees rather than why-chains), and treat
the output as a **hypothesis to test**, not a verdict. Used that way, it is the
cheapest good idea in problem-solving; used as a ritual, it decorates guesses.

.. hint::

   - :doc:`Analytical Thinking and Questions for Problem Solving <016-analytical-thinking-and-questions-for-problem-solving>`
   - :doc:`Detectives and Data Analysts <004-detectives-and-data-analysts>`
   - :doc:`Data-Driven Decision-Making <003-data-driven-decision-making>`
   - :doc:`Applying Data Analytics Problem Types in Real Business Scenarios <../2_data_driven_decisions/003-applying-data-analytics-problem-types-in-real-business-scenarios>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/07/30/root-cause-analysis-and-business-applications-of-the-five-whys/ <https://insightful-data-lab.com/2023/07/30/root-cause-analysis-and-business-applications-of-the-five-whys/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: foundations, topic: thinking
