"""
CONTENT + MINDMAP for learn/bayesian_data_analysis lessons.

Keys must be EXACT titles from bda_inventory.tsv (see BAYESIAN_DATA_ANALYSIS.md;
landmines: \ufb03 \ufb00 ligatures, \u03b8, en/em dashes, the 'opinons' source typo,
and the disambiguated '... (continued)' duplicate).
CONTENT[title] = raw RST body; MINDMAP[title] = lateral 'See also' titles.
Populated batch by batch in curriculum order.
"""

CONTENT: dict[str, str] = {}
MINDMAP: dict[str, list[str]] = {}


# ----------------------------------------------------------------------
# Part I / Stage 1 — The Bayesian Idea
# ----------------------------------------------------------------------

CONTENT["The three steps of Bayesian data analysis"] = r"""
A process, not a formula
--------------------------

Bayesian data analysis is often reduced to a single equation, but in practice it is a **three-step
process** — and the equation is only the middle step. Naming the steps makes the workflow explicit and
shows where the real work lies: not in applying Bayes' rule (mechanical), but in **building** the model
and **checking** it.

The three steps
-----------------

1. **Set up a full probability model** — a joint distribution over everything unknown and everything
   observed. This means choosing a **likelihood** :math:`p(y \mid \theta)` for how the data arise, and
   a **prior** :math:`p(\theta)` for the parameters, consistent with what you know about the problem.
2. **Condition on the observed data** — compute and interpret the **posterior**
   :math:`p(\theta \mid y)`, the conditional distribution of the unknowns given what was actually seen.
3. **Evaluate the fit** — does the model describe the data? Are its implications reasonable? Are the
   conclusions sensitive to the assumptions? If not, refine the model and return to step 1.

The middle step
-----------------

Only the second step is fixed by mathematics. It is Bayes' rule:

.. math::

   p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)}
   \;\;\propto\;\; \underbrace{p(y \mid \theta)}_{\text{likelihood}} \;
   \underbrace{p(\theta)}_{\text{prior}} .

The denominator :math:`p(y)` is a normalising constant, so for inference about :math:`\theta` the
proportionality is what matters. In code the whole pipeline is short:

.. code-block:: python

   import pymc as pm

   with pm.Model() as model:            # step 1: the full probability model
       theta = pm.Beta("theta", 1, 1)   #   prior
       pm.Binomial("y", n=10, p=theta, observed=8)   # likelihood
       idata = pm.sample(2000, tune=1000)            # step 2: condition on data
   # step 3: check the fit (posterior predictive, diagnostics) — later lessons

Uncertainty, quantified directly
----------------------------------

What distinguishes the approach is the **direct quantification of uncertainty**: the answer is a whole
distribution, not a point estimate with an asterisk. And step 3 is not a formality — a model is a
*simplification*, and a Bayesian analysis is only as trustworthy as the checks in its final step. The
three steps are also a **loop**, iterated as each check teaches you what the model missed.
"""

CONTENT["General Notation for Statistical Inference"] = r"""
The three symbols
-------------------

Bayesian writing is remarkably compact once its notation is fixed. Three symbols carry most of the
weight:

* :math:`\theta` — the **unobservable** quantities: parameters, or anything else you want to learn.
* :math:`y` — the **observed** data.
* :math:`\tilde{y}` — **potentially observable** data: future or replicated observations you have not
  yet seen. (Predictors, when present, are written :math:`x`.)

The joint model
-----------------

A full probability model is a **joint distribution** over the unknowns and the observables, which
always factors into prior times likelihood:

.. math::

   p(\theta, y) = p(\theta)\, p(y \mid \theta).

Everything else follows by conditioning and marginalising. Inference about parameters is
:math:`p(\theta \mid y)`; prediction of new data is the **posterior predictive**
:math:`p(\tilde{y} \mid y)`, obtained by averaging the likelihood over the posterior.

Exchangeability
-----------------

Why is it ever legitimate to model observations as **identically distributed**? The Bayesian answer is
**exchangeability**: if the labels on the observations carry no information — if any reordering of
:math:`y_1, \dots, y_n` is equally plausible — then their joint distribution can be written as
independent draws given some parameter, mixed over a distribution for that parameter. Exchangeability,
not an assumption of "random sampling", is what licenses the usual iid likelihood, and it becomes the
foundation of hierarchical models later in this course.

Read the conditioning bar
---------------------------

One habit repays itself constantly: read every vertical bar as "**given**", and check what is on its
right. :math:`p(y \mid \theta)` and :math:`p(\theta \mid y)` are built from the same joint model but
answer opposite questions. Bayesian and frequentist methods differ mainly in **what they condition on**
— the Bayesian conditions on the data actually observed and treats :math:`\theta` as random; the
frequentist conditions on :math:`\theta` and treats the data as random.
"""

CONTENT["Bayesian Inference"] = r"""
Conclusions as probabilities
------------------------------

**Bayesian inference** is the process of drawing conclusions about unknown quantities as **probability
statements conditional on the observed data**. Not "the estimate is 0.58 ± 0.05", but "given these
data, there is a 93% probability that :math:`\theta` exceeds 0.5". Every conclusion is read off the
posterior distribution, and every conclusion carries its uncertainty with it.

Bayes' rule, again
--------------------

The machinery is one line. Starting from the joint model :math:`p(\theta, y) = p(\theta) p(y \mid
\theta)` and conditioning on :math:`y`:

.. math::

   p(\theta \mid y) = \frac{p(\theta)\, p(y \mid \theta)}{p(y)},
   \qquad p(y) = \int p(\theta)\, p(y \mid \theta) \, d\theta .

The denominator — the **marginal likelihood** or *evidence* — does not depend on :math:`\theta`, so for
inference it merely normalises. This is why the unnormalised form
:math:`p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta)` is the working equation, and why samplers
need only the numerator.

Reading a posterior
---------------------

Once you have the posterior (analytically or as samples), every question is answered by summarising it:

.. code-block:: python

   import numpy as np
   post = ...                                   # draws from p(theta | y)
   post.mean(), np.median(post)                 # point summaries
   np.percentile(post, [2.5, 97.5])             # 95% credible interval
   (post > 0.5).mean()                          # P(theta > 0.5 | y), directly

That last line is the Bayesian signature: a probability of a hypothesis, computed by counting draws.

What differs, and why
-----------------------

Bayesian and frequentist conclusions often agree in simple problems with plenty of data. They diverge
where **conditioning** matters: small samples, many parameters, hierarchical structure, or genuine
prior information. The cost is that you must state a prior; the benefit is that the answer is a
distribution you may interpret directly, and that uncertainty propagates automatically into any
derived quantity.
"""

CONTENT["Discrete Bayesian Examples \u2013 Genetics and Spell Checking (with \u03b8)"] = r"""
Bayes' rule on a discrete unknown
-----------------------------------

The clearest way to see Bayes' rule work is when :math:`\theta` takes only a **few discrete values**.
No integrals, no sampling — just arithmetic on probabilities. Two classic examples, both in Gelman's
first chapter, show the whole logic of updating.

Genetics: the carrier problem
-------------------------------

Hemophilia is X-linked recessive. A woman whose brother is affected has a carrier status
:math:`\theta \in \{0, 1\}` with **prior** :math:`\Pr(\theta = 1) = \tfrac{1}{2}`. Suppose she then
has two **unaffected** sons, :math:`y = (0, 0)`. A carrier passes the gene with probability
:math:`\tfrac{1}{2}` per son, so the likelihoods are
:math:`p(y \mid \theta = 1) = (\tfrac{1}{2})^2 = \tfrac14` and :math:`p(y \mid \theta = 0) = 1`.
Bayes' rule updates her carrier probability:

.. math::

   \Pr(\theta = 1 \mid y)
   = \frac{\tfrac14 \cdot \tfrac12}{\tfrac14 \cdot \tfrac12 + 1 \cdot \tfrac12}
   = \frac{1}{5} = 0.20 .

Two healthy sons drop the probability from 0.50 to 0.20 — **evidence, not proof**. Each further
unaffected son multiplies the odds again; the update is **sequential**, and today's posterior is
tomorrow's prior.

Spell checking: which word was meant?
---------------------------------------

The same rule powers a spell checker. Seeing the typed string ``radom``, the candidate corrections
:math:`\theta \in \{\text{random}, \text{radon}, \text{radom}\}` are scored by
:math:`p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta)`: the **prior** is how common each word is
in the language, and the **likelihood** is how likely that typo is given the intended word.

.. code-block:: python

   prior = {"random": 7.6e-5, "radon": 6.1e-6, "radom": 1.2e-7}   # word frequencies
   like  = {"random": 0.00193, "radon": 0.000143, "radom": 0.975} # typo model
   post  = {w: prior[w] * like[w] for w in prior}
   Z = sum(post.values())
   {w: p / Z for w, p in post.items()}    # normalised posterior over intended words

A rare word needs a *much* better typo-likelihood to win. Both examples make the same point: the prior
supplies context, the likelihood supplies evidence, and the posterior is their **compromise** — the
theme of the next stage.
"""


MINDMAP.update({
    "The three steps of Bayesian data analysis": [
        "General Notation for Statistical Inference", "Bayesian Inference",
        "The Place of Model Checking in Applied Bayesian Statistics",
        "Bayesian Inference in Applied Statistics",
    ],
    "General Notation for Statistical Inference": [
        "The three steps of Bayesian data analysis", "Bayesian Inference",
        "Some Useful Results from Probability Theory",
        "Exchangeability and hierarchical models",
    ],
    "Bayesian Inference": [
        "The three steps of Bayesian data analysis", "General Notation for Statistical Inference",
        "Discrete Bayesian Examples \u2013 Genetics and Spell Checking (with \u03b8)",
        "Probability as a Measure of Uncertainty",
    ],
    "Discrete Bayesian Examples \u2013 Genetics and Spell Checking (with \u03b8)": [
        "Bayesian Inference", "Probability as a Measure of Uncertainty",
        "Posterior as a Compromise Between Data and Prior Information",
        "Estimating a Probability from Binomial Data",
    ],
})


# ----------------------------------------------------------------------
# Part I / Stage 1 — The Bayesian Idea (cont.)
# ----------------------------------------------------------------------

CONTENT["Probability as a Measure of Uncertainty"] = r"""
Uncertainty, not just frequency
---------------------------------

Bayesian analysis uses **probability to quantify uncertainty** — about anything, not merely about
repeatable experiments. "The probability this coin lands heads is 0.5" and "the probability this
patient carries the gene is 0.2" are, in this framework, the same kind of statement: a **numerical
measure of how uncertain you are**, given what you know.

Where the numbers come from
-----------------------------

Three standard justifications are offered for assigning a probability:

* **Symmetry** — if :math:`n` outcomes are indistinguishable in every relevant respect, each gets
  :math:`1/n` (a fair die, a shuffled deck).
* **Frequency** — the long-run proportion in a sequence of similar trials (the rate of a disease).
* **Subjective assessment** — a considered degree of belief, when neither symmetry nor a reference
  sequence is available.

Look closely and all three lean on **judgement**. Symmetry requires deciding *which* respects are
relevant; frequency requires choosing *which* trials count as "similar". The choice of a **reference
set** is unavoidably a modelling decision, so even "objective" probabilities are conditional on
context.

Conditional on what you know
------------------------------

This is why Bayesian probabilities are always written with a **conditioning bar**: they are
:math:`p(\theta \mid \text{information})`. The probability that the patient carries the gene changes —
legitimately, not fickly — when a son is born unaffected. Probability is not a property of the
parameter; it is a property of your **state of knowledge** about it.

Subjective, but disciplined
-----------------------------

The obvious objection is that subjective probabilities are arbitrary. Two disciplines answer it.
**Coherence**: degrees of belief must obey the probability axioms, or you can be led into a set of bets
you are certain to lose (the Dutch-book argument). And **calibration**: of all the events to which you
assign probability 0.7, about 70% should actually occur — an empirically checkable standard, and the
subject of the next two lessons. Bayesian analysis does not demand that priors match anyone's inner
convictions; it demands that assumptions be **stated clearly** and their implications **checked**.
"""

CONTENT["Example \u2014 Probabilities from Football Point Spreads"] = r"""
Assignment, not inference
---------------------------

This example illustrates **probability assignment** — how to arrive at a number — rather than Bayesian
inference itself. Its subject is the American-football **point spread**: the bookmakers' published
prediction of the margin by which the favourite will win. Given a spread, what is the probability the
favourite actually covers it, or simply wins?

Three routes to a number
--------------------------

The same question is approached three ways, matching the three justifications of the previous lesson:

* **Subjective** — an informed fan states a probability directly.
* **Empirical** — count outcomes in a database of games. Across **672** professional games, one can
  simply tabulate how often favourites at a given spread won.
* **Parametric** — build a probability **model** for the outcome and read the probability off it.

The parametric model
----------------------

The empirical route runs out of data at any particular spread, so the model earns its keep. Plotting
:math:`d = (\text{actual outcome}) - (\text{point spread})` against the spread shows the differences
are roughly **centred at zero** with a spread of about **14 points**, and largely **independent of the
spread itself**. That suggests

.. math::

   d \sim \mathrm{N}(0,\; 14^2),

so the favourite (spread :math:`s`) wins when the actual margin exceeds 0, i.e. when :math:`d > -s`:

.. code-block:: python

   from scipy.stats import norm
   s = 3.5                                  # point spread
   p_win = 1 - norm.cdf(-s, loc=0, scale=14)   # P(favourite wins)  ≈ 0.60
   p_cover = 1 - norm.cdf(0, loc=0, scale=14)  # P(covers spread)   = 0.50

The lessons
-------------

Two. First, the **model smooths and extrapolates**: it gives a probability at spreads where few games
were ever played, which raw counts cannot. Second, the model is **checked against data** — the
zero-centred, constant-variance normal is adopted *because* the scatterplot supports it, not because it
is convenient. Probability assignment, done honestly, already involves the third of the three steps.
"""

CONTENT["Example \u2014 Calibration for Record Linkage"] = r"""
Are two records the same person?
----------------------------------

**Record linkage** is the problem of deciding which records in two files — a census and a survey, two
hospital databases — refer to the **same individual**, when there is no shared unique key and the
fields are noisy: misspelled names, transposed digits, missing values. Each candidate pair gets a
**score** from comparing its fields, and the question is what that score *means*.

From score to probability
---------------------------

A score is only useful if it can be turned into :math:`\Pr(\text{match} \mid \text{score})`. Bayes'
rule supplies the conversion: with a **prior** probability that a random pair matches, and the
**likelihood** of the observed score under matching and non-matching pairs,

.. math::

   \Pr(\text{match} \mid \text{score})
   = \frac{p(\text{score} \mid \text{match}) \; \pi}
          {p(\text{score} \mid \text{match}) \; \pi
         + p(\text{score} \mid \text{non-match}) \, (1 - \pi)} .

The prior :math:`\pi` matters enormously: comparing two files of size :math:`n` produces :math:`n^2`
pairs but at most :math:`n` true matches, so **most pairs are non-matches** and :math:`\pi` is tiny. A
score that looks convincing can still leave the pair more likely a coincidence.

Calibration
-------------

The point of the example is **calibration**: among all pairs assigned probability 0.9, roughly 90%
should truly be matches. Calibration is **checkable**, by holding out pairs whose truth is known:

.. code-block:: python

   import numpy as np
   # bin predicted match probabilities; compare bin mean to observed match rate
   bins = np.linspace(0, 1, 11)
   idx = np.digitize(p_match, bins) - 1
   for b in range(10):
       m = idx == b
       if m.sum():
           print(f"{bins[b]:.1f}-{bins[b+1]:.1f}: predicted {p_match[m].mean():.2f}"
                 f"  observed {is_match[m].mean():.2f}  (n={m.sum()})")

A model whose stated probabilities survive this test can be **trusted downstream**; one that does not
will quietly corrupt every analysis built on the linked file. The mismatch between fitted and observed
probabilities is a first taste of the **posterior predictive check** in Part II — and the decision of
*which* pairs to declare matched is a decision problem, with its own costs for false links and missed
ones.
"""

CONTENT["Some Useful Results from Probability Theory"] = r"""
The toolkit
-------------

A handful of identities do nearly all the work in Bayesian derivations. They are elementary, but worth
stating once, precisely, because every later chapter leans on them.

Conditioning and marginalising
--------------------------------

Two operations turn joint distributions into the ones you want. **Marginalising** integrates a variable
away; **conditioning** fixes it. Together they give the law of total probability:

.. math::

   p(y) = \int p(y, \theta) \, d\theta = \int p(y \mid \theta) \, p(\theta) \, d\theta .

This is exactly the evidence in Bayes' rule, and the same averaging produces the **posterior
predictive** distribution :math:`p(\tilde{y} \mid y) = \int p(\tilde{y} \mid \theta) p(\theta \mid y)
\, d\theta` — a mixture of predictions, weighted by posterior plausibility.

Iterated expectation and variance
-----------------------------------

Summaries of a marginal can be built from summaries of conditionals, without doing the integral:

.. math::

   \mathrm{E}(u) = \mathrm{E}\bigl(\mathrm{E}(u \mid v)\bigr), \qquad
   \mathrm{var}(u) = \mathrm{E}\bigl(\mathrm{var}(u \mid v)\bigr)
                   + \mathrm{var}\bigl(\mathrm{E}(u \mid v)\bigr).

The variance decomposition is the more revealing: total uncertainty splits into the **average
conditional uncertainty** plus the **uncertainty in the conditional mean**. Averaging over a parameter
you have not learned always *adds* variance — which is precisely why point estimates understate
uncertainty, and why hierarchical models must propagate it.

Transformation of variables
-----------------------------

Reparameterising (to :math:`\log \sigma`, to :math:`\mathrm{logit}\,\theta`) is routine, and densities
do not simply carry over: they pick up a **Jacobian**,

.. math::

   p_{\phi}(\phi) = p_{\theta}(\theta) \left| \frac{d\theta}{d\phi} \right| .

Forgetting the Jacobian is a classic bug — a "noninformative" flat prior on :math:`\theta` is **not**
flat on :math:`\log \theta`, a subtlety the noninformative-prior lesson confronts directly. Simulation
sidesteps the algebra entirely: transform the **draws** and summarise them.

.. code-block:: python

   import numpy as np
   theta = ...                       # posterior draws
   phi = np.log(theta)               # any function of the draws is itself posterior draws
   phi.mean(), np.percentile(phi, [2.5, 97.5])

That last property — that a function of posterior draws gives the posterior of that function, with no
Jacobian and no delta method — is one of the quiet advantages of the simulation-based workflow this
course adopts.
"""


MINDMAP.update({
    "Probability as a Measure of Uncertainty": [
        "Bayesian Inference", "Example \u2014 Probabilities from Football Point Spreads",
        "Example \u2014 Calibration for Record Linkage", "Noninformative Prior Distributions",
    ],
    "Example \u2014 Probabilities from Football Point Spreads": [
        "Probability as a Measure of Uncertainty", "Example \u2014 Calibration for Record Linkage",
        "Some Useful Results from Probability Theory", "Normal Distribution with Known Variance",
    ],
    "Example \u2014 Calibration for Record Linkage": [
        "Probability as a Measure of Uncertainty", "Example \u2014 Probabilities from Football Point Spreads",
        "Posterior predictive checking", "Bayesian decision theory in di\ufb00erent contexts",
    ],
    "Some Useful Results from Probability Theory": [
        "General Notation for Statistical Inference", "Bayesian Inference",
        "Averaging Over Nuisance Parameters", "Noninformative Prior Distributions",
    ],
})


# ----------------------------------------------------------------------
# Part I / Stage 1 — The Bayesian Idea (cont.)  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Computation and Software"] = r"""
Why computation matters
-------------------------

Bayes' rule is one line, but the posterior it defines is usually **intractable**: the evidence
:math:`p(y) = \int p(y \mid \theta) p(\theta) d\theta` has no closed form once the model has more than
a couple of parameters. For most of the twentieth century that was a fatal obstacle. What made
Bayesian analysis practical is **computation** — and specifically, the decision to stop trying to
*evaluate* posteriors and instead **sample** from them.

Samples are enough
--------------------

The pivotal insight is that a large collection of draws :math:`\theta^{(1)}, \dots, \theta^{(S)}` from
:math:`p(\theta \mid y)` answers every question you would have asked of the density. Means, intervals,
tail probabilities, and the posterior of **any function** of the parameters are all obtained by
summarising draws:

.. math::

   \mathrm{E}[h(\theta) \mid y] \;\approx\; \frac{1}{S} \sum_{s=1}^{S} h\bigl(\theta^{(s)}\bigr),

with Monte Carlo error shrinking like :math:`1/\sqrt{S}`. Simulation also propagates uncertainty for
free — no delta method, no Jacobians (as the previous lesson noted).

The modern stack
------------------

In Python the workflow is standardised. **``scipy.stats``** covers the conjugate and closed-form cases
directly; **PyMC** (or **Stan**) expresses a model declaratively and samples it with **HMC/NUTS**; and
**ArviZ** handles the diagnostics and plots that step 3 demands:

.. code-block:: python

   import pymc as pm, arviz as az

   with pm.Model() as m:
       theta = pm.Beta("theta", 1, 1)
       pm.Binomial("y", n=10, p=theta, observed=8)
       idata = pm.sample(2000, tune=1000, chains=4)

   az.summary(idata)      # mean, sd, 94% HDI, r_hat, ess
   az.plot_trace(idata)   # visual convergence check

Never trust unchecked draws
-----------------------------

Sampling is approximate, so the output must be **audited** before it is believed: :math:`\hat{R}`
close to 1.0 (chains agree), an adequate **effective sample size**, and **no divergences**. A sampler
that has not converged produces confident nonsense. This is why the computation stages of this course
(Part III) devote as much attention to **diagnosing** samplers as to running them — and why "the folk
theorem" holds that computational trouble usually signals a problem with the **model**, not just the
algorithm.
"""

CONTENT["Bayesian Inference in Applied Statistics"] = r"""
Where the approach pays
-------------------------

Bayesian inference is not merely a philosophical stance; it earns its place in applied work by solving
problems that are awkward otherwise. The recurring theme is that a **full probability model** handles
complications — small samples, nuisance parameters, structure, missing data — by the *same* mechanism
it uses for everything else: write the joint distribution, then condition.

What it makes easy
--------------------

Several everyday difficulties become routine:

* **Small samples and rare events** — a weakly informative prior stabilises estimates that would
  otherwise be wild (zero events out of twenty need not imply a rate of zero).
* **Nuisance parameters** — integrate them out honestly rather than fixing them at estimates; the
  resulting uncertainty is correctly inflated.
* **Grouped data** — hierarchical models share strength across groups, an idea with no natural
  frequentist counterpart (Stage 5).
* **Derived quantities** — the posterior of :math:`h(\theta)` comes free from the draws, however
  nonlinear :math:`h` is.
* **Missing data and censoring** — unobserved values are simply more unknowns, given their own
  distribution.
* **Sequential updating** — today's posterior is tomorrow's prior, exactly as in the genetics example.

Decisions, not just estimates
-------------------------------

Because the output is a full distribution, it plugs directly into **decision making**: choose the
action maximising expected utility, averaged over the posterior. Applied Bayesian work therefore runs
from inference straight through to consequences — pricing, screening, remediation — a thread this
course picks up in Stage 7.

The honest caveats
--------------------

Three costs are real, and worth stating plainly. You must **specify a prior**, and defend it — with
sparse data, conclusions can be sensitive to it, so sensitivity analysis is part of the job.
Computation can be **expensive**, and may fail silently without diagnostics. And a Bayesian model, like
any model, can be **wrong**: conditioning on a misspecified likelihood yields a confident, coherent,
misleading posterior. This is precisely why the three-step process ends with **model checking**, and
why Part II of this course is devoted to it.
"""


# ----------------------------------------------------------------------
# Part I / Stage 2 — Single-Parameter Models & Priors
# ----------------------------------------------------------------------

CONTENT["Estimating a Probability from Binomial Data"] = r"""
The workhorse model
---------------------

The simplest interesting Bayesian problem: estimate an unknown probability :math:`\theta` — a
conversion rate, a survival rate, the chance of a female birth — from :math:`y` successes in :math:`n`
independent trials. The likelihood is **binomial**,

.. math::

   p(y \mid \theta) = \binom{n}{y} \theta^{y} (1 - \theta)^{n - y}
   \;\;\propto\;\; \theta^{y} (1 - \theta)^{n - y},

and the whole of single-parameter Bayesian inference can be seen in miniature here.

A Beta prior
--------------

For a parameter confined to :math:`[0, 1]`, the natural prior is a **Beta** distribution,
:math:`\theta \sim \mathrm{Beta}(\alpha, \beta)`, whose density is proportional to
:math:`\theta^{\alpha - 1} (1 - \theta)^{\beta - 1}`. Note the shape: it is *the same functional form*
as the likelihood. That is not a coincidence but the definition of **conjugacy**, and it makes the
update exact.

The update is addition
------------------------

Multiply prior by likelihood and read off the kernel:

.. math::

   p(\theta \mid y) \;\propto\; \theta^{y}(1-\theta)^{n-y} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}
   = \theta^{\alpha + y - 1} (1 - \theta)^{\beta + n - y - 1},

so

.. math::

   \theta \mid y \;\sim\; \mathrm{Beta}(\alpha + y,\; \beta + n - y).

Bayesian updating here is nothing more than **counting**: add successes to :math:`\alpha`, failures to
:math:`\beta`. This licenses reading :math:`\alpha` and :math:`\beta` as **prior successes and
failures**, with :math:`\alpha + \beta` a **prior sample size**.

In code
---------

With :math:`\mathrm{Beta}(1,1)` (uniform) and 8 successes in 10 trials:

.. code-block:: python

   from scipy import stats
   post = stats.beta(1 + 8, 1 + 10 - 8)          # Beta(9, 3)
   post.mean()                                    # 0.75
   post.interval(0.95)                            # 95% credible interval
   1 - post.cdf(0.5)                              # P(theta > 0.5 | y) ≈ 0.981

The MLE is :math:`8/10 = 0.80`; the posterior mean is :math:`0.75`, pulled toward the prior mean of
:math:`0.5`. That pull — its size, and its fate as :math:`n` grows — is the subject of the next lesson.
"""

CONTENT["Posterior as a Compromise Between Data and Prior Information"] = r"""
Between two answers
---------------------

The posterior always lies **between** the prior and the data. The Beta–Binomial makes this exact rather
than metaphorical: the posterior mean is a **weighted average** of the prior mean and the sample
proportion, and the weights are interpretable.

The weighted average
----------------------

With :math:`\theta \mid y \sim \mathrm{Beta}(\alpha + y,\ \beta + n - y)`, the posterior mean is

.. math::

   \mathrm{E}[\theta \mid y] = \frac{\alpha + y}{\alpha + \beta + n}
   = \underbrace{\frac{\alpha + \beta}{\alpha + \beta + n}}_{\text{prior weight}}
     \cdot \frac{\alpha}{\alpha + \beta}
   \;+\;
     \underbrace{\frac{n}{\alpha + \beta + n}}_{\text{data weight}}
     \cdot \frac{y}{n} .

The prior mean is :math:`\alpha / (\alpha+\beta)`; the data's answer is the sample proportion
:math:`y/n`. Each is weighted by its **effective sample size** — :math:`\alpha + \beta` for the prior,
:math:`n` for the data. A :math:`\mathrm{Beta}(2, 8)` prior carries the weight of **ten** prior
observations; against :math:`n = 10` it contributes about half the answer, against :math:`n = 200`,
about five per cent.

The data wins, eventually
---------------------------

The data weight :math:`n / (n + \alpha + \beta)` climbs to **1** as :math:`n \to \infty`. So the
posterior mean converges to the sample proportion, and the posterior itself concentrates — a preview of
the large-sample theory in Stage 4. Two analysts with different (reasonable) priors are **driven to
agreement** by enough data. The prior matters most exactly where it should: when data are **sparse**.

.. code-block:: python

   from scipy import stats
   for n in (10, 100, 1000):
       y = int(0.8 * n)                       # same proportion, growing n
       a, b = 2, 8                            # prior mean 0.2, prior "sample size" 10
       print(n, round(stats.beta(a + y, b + n - y).mean(), 3))
   # 10 -> 0.500   100 -> 0.745   1000 -> 0.792   (approaching 0.8)

Shrinkage, and its price
--------------------------

Pulling the estimate toward the prior mean is **shrinkage**, and it is a feature: it regularises noisy
small-sample estimates and prevents the absurdity of a 0% or 100% rate from three trials. The price is
that a **badly chosen** informative prior biases the answer, most damagingly when :math:`n` is small
and the pull is strongest. Hence the discipline: state the prior, justify it, and **check the
sensitivity** of conclusions to reasonable alternatives.
"""


MINDMAP.update({
    "Computation and Software": [
        "The three steps of Bayesian data analysis", "Bayesian Inference in Applied Statistics",
        "Inference and assessing convergence", "Stan: developing a computing environment",
    ],
    "Bayesian Inference in Applied Statistics": [
        "The three steps of Bayesian data analysis", "Computation and Software",
        "Exchangeability and hierarchical models",
        "Bayesian decision theory in di\ufb00erent contexts",
    ],
    "Estimating a Probability from Binomial Data": [
        "Posterior as a Compromise Between Data and Prior Information",
        "Summarizing Posterior Inference", "Informative Prior Distributions",
        "Discrete Bayesian Examples \u2013 Genetics and Spell Checking (with \u03b8)",
    ],
    "Posterior as a Compromise Between Data and Prior Information": [
        "Estimating a Probability from Binomial Data", "Informative Prior Distributions",
        "Noninformative Prior Distributions", "Large-Sample Theory",
    ],
})


# ----------------------------------------------------------------------
# Part I / Stage 2 — Single-Parameter Models & Priors (cont.)
# ----------------------------------------------------------------------

CONTENT["Summarizing Posterior Inference"] = r"""
The posterior is the answer
-----------------------------

Formally, the whole answer of a Bayesian analysis is the posterior **distribution**. But a distribution
is hard to report and harder to act on, so we summarise it — carefully, because every summary discards
information, and some discard the wrong information.

Point summaries
-----------------

Three point estimates are standard, and they differ once the posterior is **skewed**:

* the **posterior mean** :math:`\mathrm{E}[\theta \mid y]` — the Bayes estimator under squared-error
  loss, and the usual default;
* the **posterior median** — robust, and optimal under absolute-error loss;
* the **posterior mode** (the **MAP** estimate) — the density's peak, and the point where Bayesian
  estimation coincides with penalised maximum likelihood.

For a symmetric unimodal posterior all three agree. When they disagree, the disagreement itself is
information: the posterior is **asymmetric**, and no single number represents it.

Intervals: two kinds
----------------------

A **credible interval** is any region holding a stated posterior probability — say 95%. Two
constructions dominate:

* the **equal-tailed interval (ETI)**, cutting 2.5% from each tail (the 2.5th and 97.5th percentiles);
* the **highest-density interval (HDI/HPD)**, the **shortest** interval containing 95% of the mass —
  equivalently, every point inside has higher density than every point outside.

They coincide for symmetric posteriors. For **skewed** ones they do not, and the ETI can include
parameter values *less* plausible than some it excludes; the HDI always contains the **mode**.

.. code-block:: python

   import numpy as np, arviz as az
   draws = ...                                  # posterior draws
   draws.mean(), np.median(draws)               # point summaries
   np.percentile(draws, [2.5, 97.5])            # equal-tailed 95% CI
   az.hdi(draws, hdi_prob=0.95)                 # highest-density 95% CI
   (draws > 0.5).mean()                         # P(theta > 0.5 | y)

Say what you mean
-------------------

The interpretive prize is real: a 95% credible interval supports the statement *"there is a 95%
probability :math:`\theta` lies in this range"* — precisely what a frequentist **confidence** interval
does **not** license. But do not let the summary replace the distribution: plot the posterior, and
report the probability of the hypotheses you actually care about, which the draws give directly.
"""

CONTENT["Informative Prior Distributions"] = r"""
Using what you know
---------------------

An **informative prior** deliberately encodes substantive knowledge about a parameter before the
current data arrive — from previous studies, physical constraints, or expert judgement. Its purpose is
not to bias the answer toward a preferred conclusion but to stop the analysis from pretending to know
nothing when, in fact, a great deal is known.

Priors as pseudo-data
-----------------------

Conjugate priors make "how much" precise: their hyperparameters read as **prior observations**. In the
Beta–Binomial, :math:`\mathrm{Beta}(\alpha, \beta)` acts like :math:`\alpha` prior successes and
:math:`\beta` prior failures, an effective prior sample size of :math:`\alpha + \beta`. In the
Poisson–Gamma, :math:`\mathrm{Gamma}(\alpha, \beta)` acts like :math:`\alpha` **pseudo-events**
observed in :math:`\beta` **pseudo-exposure**. This is the honest way to calibrate an informative
prior: ask *how many observations is my belief worth?*

.. code-block:: python

   from scipy import stats
   # "I've seen roughly 20 conversions in 100 visits before" -> Beta(20, 80)
   prior = stats.beta(20, 80)
   prior.mean(), prior.std()            # 0.20, ~0.04   (prior sample size 100)
   # 5 conversions in 40 new visits:
   post = stats.beta(20 + 5, 80 + 35)   # exact conjugate update
   post.mean()                          # 0.179 — pulled from 0.125 toward 0.20

Eliciting a prior
-------------------

Elicitation usually works backwards from summaries an expert can state: a plausible **central value**
and a range they would be surprised to see violated. Match those to a distribution's mean and spread,
then **check the implications** — draw from the prior, simulate data (a **prior predictive check**),
and ask whether the simulated datasets look like ones the world could produce. A prior that generates
absurd data is wrong, however reasonable it looked as a density.

The discipline
----------------

Informative priors carry real risk: with sparse data, they can dominate, and a confidently wrong prior
yields a confidently wrong posterior. Three habits keep them honest. **State** the prior and its
justification explicitly. **Report** its effective sample size, so readers see how much it contributes.
And run a **sensitivity analysis** — refit under reasonable alternatives, and say so when conclusions
move. Between the extremes of an informative prior and a deliberately vague one lies the
**weakly informative** prior of the coming lessons.
"""

CONTENT["Normal Distribution with Known Variance"] = r"""
The second workhorse
----------------------

After the Beta–Binomial, the **normal model with known variance** is the most useful closed-form case,
and the template for nearly every hierarchical and regression result later in the course. Take
:math:`n` observations :math:`y_i \sim \mathrm{N}(\theta, \sigma^2)` with :math:`\sigma^2` **known**,
and a conjugate prior :math:`\theta \sim \mathrm{N}(\mu_0, \tau_0^2)`.

Precision adds
----------------

The algebra is cleanest in terms of **precision** — the reciprocal of variance, :math:`1/\sigma^2` —
because precisions simply **add**. The posterior is normal,

.. math::

   \theta \mid y \;\sim\; \mathrm{N}(\mu_n, \tau_n^2), \qquad
   \frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2},

and its mean is the **precision-weighted average** of the prior mean and the sample mean:

.. math::

   \mu_n = \frac{\dfrac{1}{\tau_0^2}\,\mu_0 + \dfrac{n}{\sigma^2}\,\bar{y}}
                {\dfrac{1}{\tau_0^2} + \dfrac{n}{\sigma^2}} .

This is the compromise property again, now with **precision** as the weight. Data contribute precision
:math:`n / \sigma^2`; the prior contributes :math:`1/\tau_0^2`, equivalent to
:math:`\sigma^2 / \tau_0^2` observations. Each new observation **increases** posterior precision, so
the posterior variance can only shrink.

In code
---------

.. code-block:: python

   import numpy as np
   from scipy import stats

   mu0, tau0, sigma = 0.0, 2.0, 1.0        # prior N(0, 2^2); known sd = 1
   y = np.array([1.2, 0.8, 1.5, 0.9])
   n, ybar = len(y), y.mean()

   prec = 1 / tau0**2 + n / sigma**2
   mun = (mu0 / tau0**2 + n * ybar / sigma**2) / prec
   post = stats.norm(mun, np.sqrt(1 / prec))
   post.mean(), post.interval(0.95)         # ≈ 1.04, (0.61, 1.47)

Where it leads
----------------

Two consequences echo through the rest of the course. As :math:`n \to \infty` the prior's precision
becomes negligible and :math:`\mu_n \to \bar{y}` — data win, as always. And a **flat** prior
(:math:`\tau_0 \to \infty`) gives exactly :math:`\theta \mid y \sim \mathrm{N}(\bar{y}, \sigma^2/n)`,
the familiar sampling-theory result reappearing as a Bayesian posterior. That coincidence, and its
limits, is the subject of Stage 4.
"""

CONTENT["Other Standard Single-Parameter Models"] = r"""
The same pattern, different data
----------------------------------

Binomial and normal are two instances of one recipe. Whenever the likelihood belongs to the
**exponential family**, a conjugate prior exists, the posterior stays in the prior's family, and the
update amounts to **adding sufficient statistics** to the hyperparameters. Three more cases cover most
of applied practice.

Poisson–Gamma (counts)
------------------------

For counts, :math:`y_i \sim \mathrm{Poisson}(\theta)` with a :math:`\mathrm{Gamma}(\alpha, \beta)`
prior on the rate:

.. math::

   \theta \mid y \;\sim\; \mathrm{Gamma}\Bigl(\alpha + \textstyle\sum_i y_i,\; \beta + n\Bigr),
   \qquad
   \mathrm{E}[\theta \mid y] = \frac{\alpha + \sum_i y_i}{\beta + n}.

The hyperparameters read as **pseudo-events** :math:`\alpha` observed in **pseudo-exposure**
:math:`\beta`, and the posterior mean is again a weighted average of the prior mean
:math:`\alpha/\beta` and the sample rate :math:`\bar{y}`. With exposures :math:`x_i` (person-years,
say), the update uses :math:`\beta + \sum_i x_i`.

Exponential and normal-variance
---------------------------------

For waiting times, :math:`y_i \sim \mathrm{Exponential}(\theta)` also takes a **Gamma** conjugate
prior, updating to :math:`\mathrm{Gamma}(\alpha + n,\ \beta + \sum_i y_i)`. And for a normal with
**known mean and unknown variance**, the conjugate prior on :math:`\sigma^2` is the
**inverse-gamma** (equivalently, scaled inverse-:math:`\chi^2`) — the piece needed to complete the
normal model in the next stage.

.. code-block:: python

   from scipy import stats
   y = [3, 5, 2, 4]                       # counts over n = 4 periods
   a, b = 2, 1                            # Gamma prior: 2 pseudo-events / 1 pseudo-period
   post = stats.gamma(a + sum(y), scale=1 / (b + len(y)))
   post.mean(), post.interval(0.95)       # posterior rate and 95% CI

Convenient, not sacred
------------------------

Conjugacy is prized for being **exact, instant and interpretable** — ideal for teaching, for sequential
updating, and as a building block inside larger samplers (Gibbs, in Part III). But conjugate families
exist for **algebraic convenience**, not because they encode anyone's real prior beliefs, and most
useful models have no conjugate form at all. Modern practice keeps the intuition — priors as pseudo-data,
posteriors as compromises — and reaches for **MCMC** whenever the model demands it.
"""


MINDMAP.update({
    "Summarizing Posterior Inference": [
        "Estimating a Probability from Binomial Data", "Bayesian Inference",
        "Posterior as a Compromise Between Data and Prior Information",
        "Normal Approximations to the Posterior Distribution",
    ],
    "Informative Prior Distributions": [
        "Posterior as a Compromise Between Data and Prior Information",
        "Noninformative Prior Distributions", "Weakly Informative Prior Distributions",
        "Informative Prior Distribution for Cancer Rates",
    ],
    "Normal Distribution with Known Variance": [
        "Estimating a Probability from Binomial Data",
        "Normal Data with a Conjugate Prior Distribution",
        "Other Standard Single-Parameter Models", "Normal model with exchangeable parameters",
    ],
    "Other Standard Single-Parameter Models": [
        "Normal Distribution with Known Variance", "Estimating a Probability from Binomial Data",
        "Informative Prior Distributions", "Standard generalized linear model likelihoods",
    ],
})


# ----------------------------------------------------------------------
# Part I / Stage 2 — Single-Parameter Models & Priors (cont.)  [completes]
# ----------------------------------------------------------------------

CONTENT["Informative Prior Distribution for Cancer Rates"] = r"""
The map that lies
-------------------

Colour a map of US counties by **kidney-cancer rate** and something strange appears: the highest-rate
counties are mostly **small, rural** ones. Colour it by the **lowest** rates and — the same counties
appear again. No environmental story explains both. The pattern is a **statistical artifact**, and it
is the classic argument for informative priors.

Small denominators, wild rates
--------------------------------

Kidney cancer is rare. A county of 60,000 people might record **4** cases (about 6.6 per 100,000); a
county of 17,000 might record **7** (about 41 per 100,000). One case fewer in the small county would
drop its rate by roughly a sixth. The raw rate :math:`y_j / n_j` is an **unbiased but hopelessly noisy**
estimate when :math:`n_j` is tiny, so the extremes of the ranking are populated not by the most
dangerous places, but by the **smallest** ones.

The Poisson–Gamma remedy
--------------------------

Model the counts as :math:`y_j \sim \mathrm{Poisson}(\theta_j n_j)` with a
:math:`\mathrm{Gamma}(\alpha, \beta)` prior on the county rate, and the conjugate posterior mean is a
weighted average of the county's own rate and the prior (national) rate:

.. math::

   \mathrm{E}[\theta_j \mid y_j] = \frac{\alpha + y_j}{\beta + n_j}
   = \frac{\beta}{\beta + n_j} \cdot \underbrace{\frac{\alpha}{\beta}}_{\text{prior rate}}
   \;+\; \frac{n_j}{\beta + n_j} \cdot \underbrace{\frac{y_j}{n_j}}_{\text{raw rate}} .

Small counties (:math:`n_j \ll \beta`) are pulled hard toward the national rate; large counties keep
their own. The prior does not distort — it **stabilises**.

.. code-block:: python

   from scipy import stats
   alpha, beta = 20, 1_000_000        # prior: ~20 cases per million person-years
   for y, n in [(7, 17_000), (4, 60_000), (250, 1_200_000)]:
       raw = 1e5 * y / n
       shrunk = 1e5 * (alpha + y) / (beta + n)
       print(f"raw {raw:6.1f}   shrunk {shrunk:6.1f}   (n={n:,})")
   # small counties move a lot; the large county barely moves

Shrinkage, honestly
---------------------

The estimates are **shrunk** toward a common centre by an amount governed by how much data each county
supplies. This buys enormous stability at the price of a small bias toward the mean — an excellent
trade when the alternative is ranking noise. And note what this analysis is quietly reaching for: the
prior rate :math:`\alpha/\beta` should really be **estimated from the counties themselves**. That is a
**hierarchical model**, and it arrives in Stage 5.
"""

CONTENT["Noninformative Prior Distributions"] = r"""
Letting the data speak
------------------------

Sometimes you want the prior to contribute as **little** as possible — to report what *these* data say,
with minimal external input. A **noninformative** (or *reference*, or *vague*) prior aims at that. The
ambition is honourable and the execution is subtler than it looks, because "knowing nothing" turns out
not to be a well-defined state.

Flat is not neutral
---------------------

The obvious candidate is a **uniform** prior, :math:`p(\theta) \propto 1`. But flatness is **not
invariant** under reparameterisation: if :math:`\theta` is uniform, then :math:`\log \theta` is not —
the Jacobian from the probability-theory lesson intervenes. A prior that claims ignorance about a rate
therefore asserts something quite specific about the **log**-rate. "Noninformative" always means
noninformative *on some scale*, and that scale is a choice.

Jeffreys' prior
-----------------

Harold **Jeffreys** solved the invariance problem by tying the prior to the model itself. The
**Jeffreys prior** is proportional to the square root of the **Fisher information**:

.. math::

   p_J(\theta) \;\propto\; \sqrt{\det I(\theta)}, \qquad
   I(\theta) = -\,\mathrm{E}\!\left[\frac{\partial^2}{\partial\theta^2} \log p(y \mid \theta)\right] .

Because the Fisher information transforms with exactly the Jacobian that densities need, this prior is
**invariant under smooth reparameterisation** — the same beliefs, whatever coordinates you use. It
recovers the intuitive answers: **flat** :math:`p(\theta) \propto 1` for a location parameter, and
:math:`p(\theta) \propto 1/\theta` for a scale parameter. For the binomial it gives
:math:`\mathrm{Beta}(\tfrac12, \tfrac12)`.

Improper priors
-----------------

Many noninformative priors are **improper**: they do not integrate to a finite value (a flat prior on
the whole real line; :math:`1/\sigma` on :math:`(0, \infty)`). This is tolerable **only if the
resulting posterior is proper**, which must be checked, not assumed — for hierarchical variance
parameters especially, a natural-looking improper prior can yield a posterior that does not exist,
while the sampler reports numbers regardless.

.. code-block:: python

   from scipy import stats
   y, n = 8, 10
   flat     = stats.beta(1 + y, 1 + n - y)          # Beta(1,1): uniform
   jeffreys = stats.beta(0.5 + y, 0.5 + n - y)      # Beta(1/2,1/2)
   flat.mean(), jeffreys.mean()                     # 0.750, 0.773

Modern practice has largely moved on: rather than chase an unattainable neutrality, use a
**weakly informative** prior that rules out the absurd while letting the data dominate.
"""

CONTENT["Weakly Informative Prior Distributions"] = r"""
The sensible middle
---------------------

Between an informative prior that asserts a specific belief and a noninformative one that tries to
assert nothing lies the **weakly informative** prior — the modern default. It deliberately contains
**less** information than is actually available, but enough to **regularise**: to keep the inference
inside the range of physically or scientifically plausible values, and to stabilise estimation when the
data are weak.

What it does
--------------

A weakly informative prior is chosen by asking what values are **conceivable**, not what values are
*likely*. On a standardised logistic-regression coefficient, a :math:`\mathrm{N}(0, 2.5^2)` prior says:
an odds ratio of 2 or 5 is unremarkable, an odds ratio of :math:`10^6` is not. That single, mild
statement:

* **prevents separation** — the infinite coefficients of a perfectly separated logistic fit become
  finite (a problem met again in Part IV);
* **tames weak identification** — parameters the data barely constrain get finite posteriors instead
  of wandering chains;
* **shrinks noise** — as in the cancer-rate example, small-:math:`n` estimates stop being extreme;
* leaves conclusions **essentially unchanged** where the data are strong.

Common defaults
-----------------

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # coefficients (predictors standardised): mild, symmetric, finite-tailed
       beta = pm.Normal("beta", mu=0, sigma=2.5, shape=k)
       # scale parameters: positive, heavy-tailed near zero, no hard upper bound
       sigma = pm.HalfNormal("sigma", sigma=1)        # or pm.HalfCauchy("sigma", 1)

For **variance** parameters in hierarchies, a **half-normal** or **half-Cauchy** on the standard
deviation is standard practice — it allows the group-level scale to be near zero (complete pooling)
without the pathologies of the once-popular inverse-gamma-with-tiny-parameters.

Scale matters
---------------

A weakly informative prior is a statement about **units**. :math:`\mathrm{N}(0, 2.5^2)` is mild for a
coefficient on a standardised predictor and wildly informative for one measured in dollars, so
**standardise predictors** (or scale the prior to the data). And the standard defence applies: run the
**prior predictive check** — simulate data from the prior alone and confirm the simulated datasets are
merely varied, not absurd. A prior generating impossible data is too weak, not too strong.
"""


# ----------------------------------------------------------------------
# Part I / Stage 3 — Multiparameter Models
# ----------------------------------------------------------------------

CONTENT["Averaging Over Nuisance Parameters"] = r"""
The parameters you don't want
-------------------------------

Real models have more parameters than questions. Estimating a mean :math:`\mu` usually drags along an
unknown variance :math:`\sigma^2`; a regression coefficient of interest comes with a dozen others. The
unwanted ones are **nuisance parameters** — necessary for the model to be honest, irrelevant to the
conclusion.

Integrate, don't fix
----------------------

The Bayesian treatment is uniform and unremarkable: obtain the **joint** posterior, then **marginalise**
the nuisance away. To learn about :math:`\theta_1` in the presence of nuisance :math:`\theta_2`,

.. math::

   p(\theta_1 \mid y) = \int p(\theta_1, \theta_2 \mid y) \; d\theta_2
   = \int \underbrace{p(\theta_1 \mid \theta_2, y)}_{\text{conditional}}
          \; \underbrace{p(\theta_2 \mid y)}_{\text{weight}} \; d\theta_2 .

That second form is the useful one: the marginal posterior of :math:`\theta_1` is a **mixture** of its
conditional posteriors, weighted by how plausible each value of the nuisance is. Uncertainty about
:math:`\sigma^2` is not discarded — it is **averaged in**.

Why plugging in is wrong
--------------------------

The tempting shortcut is to fix the nuisance at an estimate, :math:`p(\theta_1 \mid \hat{\theta}_2, y)`.
This **understates uncertainty**, and the variance decomposition says exactly by how much:

.. math::

   \mathrm{var}(\theta_1 \mid y) = \mathrm{E}\bigl[\mathrm{var}(\theta_1 \mid \theta_2, y)\bigr]
                                 + \mathrm{var}\bigl(\mathrm{E}[\theta_1 \mid \theta_2, y]\bigr) .

Plugging in keeps only the first term and throws away the second — the variation induced by not
knowing :math:`\theta_2`. This is why a normal mean with **unknown** variance has a heavier-tailed
:math:`t` posterior rather than a normal one: the extra width is the price of honesty about
:math:`\sigma^2`.

Marginalising with draws
--------------------------

In simulation the operation is invisible: draw from the joint posterior, then **ignore the columns you
do not need**.

.. code-block:: python

   # draws[:, 0] = mu, draws[:, 1] = sigma  (joint posterior draws)
   mu = draws[:, 0]                 # already the marginal posterior of mu
   mu.mean(), np.percentile(mu, [2.5, 97.5])

Dropping a column *is* integration over that parameter. It is one of the quiet reasons the
simulation-based workflow scales to models where the integrals could never be done in closed form —
starting with the normal model of the next lesson.
"""


MINDMAP.update({
    "Informative Prior Distribution for Cancer Rates": [
        "Informative Prior Distributions", "Other Standard Single-Parameter Models",
        "Exchangeability and hierarchical models",
        "Posterior as a Compromise Between Data and Prior Information",
    ],
    "Noninformative Prior Distributions": [
        "Weakly Informative Prior Distributions", "Informative Prior Distributions",
        "Some Useful Results from Probability Theory",
        "Normal Data with a Noninformative Prior Distribution",
    ],
    "Weakly Informative Prior Distributions": [
        "Noninformative Prior Distributions", "Informative Prior Distributions",
        "Weakly Informative Priors for Variance Parameters",
        "Weakly informative priors for logistic regression",
    ],
    "Averaging Over Nuisance Parameters": [
        "Some Useful Results from Probability Theory",
        "Normal Data with a Noninformative Prior Distribution",
        "Normal Data with a Conjugate Prior Distribution",
        "Conditional and marginal posterior approximations",
    ],
})


# ----------------------------------------------------------------------
# Part I / Stage 3 — Multiparameter Models (cont.)
# ----------------------------------------------------------------------

CONTENT["Normal Data with a Noninformative Prior Distribution"] = r"""
Two unknowns at last
----------------------

The normal model with **both** :math:`\mu` and :math:`\sigma^2` unknown is the first genuinely
multiparameter problem, and it delivers a classical result as a by-product. Take
:math:`y_i \sim \mathrm{N}(\mu, \sigma^2)` with the standard noninformative prior

.. math::

   p(\mu, \sigma^2) \;\propto\; \frac{1}{\sigma^2},

which is flat in :math:`\mu` and in :math:`\log \sigma` — the location/scale answer from the Jeffreys
lesson.

Factor the joint posterior
----------------------------

The joint posterior factors conveniently as :math:`p(\mu, \sigma^2 \mid y) = p(\mu \mid \sigma^2, y)
\, p(\sigma^2 \mid y)`. The conditional is the known-variance result from Stage 2, and the marginal for
the variance is a scaled inverse-:math:`\chi^2`:

.. math::

   \mu \mid \sigma^2, y \sim \mathrm{N}\!\left(\bar{y},\, \frac{\sigma^2}{n}\right),
   \qquad
   \sigma^2 \mid y \sim \text{Inv-}\chi^2\!\left(n - 1,\, s^2\right),

with :math:`s^2 = \frac{1}{n-1}\sum_i (y_i - \bar{y})^2`. Draw :math:`\sigma^2` first, then
:math:`\mu` given it, and you have exact joint draws — no MCMC needed.

The t emerges
---------------

Now **average over the nuisance** :math:`\sigma^2`, exactly as the last lesson prescribed. The mixture
of normals, weighted by the inverse-:math:`\chi^2`, is a **Student :math:`t`**:

.. math::

   \mu \mid y \;\sim\; t_{\,n-1}\!\left(\bar{y},\; \frac{s^2}{n}\right).

The heavy tails are not an assumption — they are the **price of not knowing** :math:`\sigma^2`, the
second term of the variance decomposition made visible. And note the coincidence: the resulting 95%
interval is numerically the classical :math:`t` confidence interval, though its interpretation as a
**probability statement about** :math:`\mu` is available only here.

.. code-block:: python

   import numpy as np
   from scipy import stats
   y = np.array([5.1, 4.8, 5.6, 5.0, 4.7]); n = len(y)
   ybar, s2 = y.mean(), y.var(ddof=1)
   # exact joint draws: sigma^2 first, then mu | sigma^2
   sigma2 = (n - 1) * s2 / stats.chi2(n - 1).rvs(10_000)
   mu = stats.norm(ybar, np.sqrt(sigma2 / n)).rvs()
   np.percentile(mu, [2.5, 97.5])          # matches the t_{n-1} interval

One caveat: this prior is **improper**, and with :math:`n = 1` the posterior for :math:`\mu` fails to
integrate — a reminder to check propriety rather than trust the algebra.
"""

CONTENT["Normal Data with a Conjugate Prior Distribution"] = r"""
Prior information on both
---------------------------

To bring real prior knowledge to the normal model, the conjugate choice is the
**normal--inverse-**:math:`\chi^2` family, specified hierarchically — the prior for the mean is scaled
by the very variance it accompanies:

.. math::

   \mu \mid \sigma^2 \sim \mathrm{N}\!\left(\mu_0,\, \frac{\sigma^2}{\kappa_0}\right),
   \qquad
   \sigma^2 \sim \text{Inv-}\chi^2(\nu_0,\, \sigma_0^2).

Four hyperparameters carry four meanings: :math:`\mu_0` a prior mean, :math:`\kappa_0` how many
observations that mean is worth; :math:`\sigma_0^2` a prior variance, :math:`\nu_0` how many
observations *it* is worth.

The update adds counts
------------------------

Conjugacy delivers a posterior in the same family, with hyperparameters that update by **addition**:

.. math::

   \kappa_n = \kappa_0 + n, \qquad \nu_n = \nu_0 + n, \qquad
   \mu_n = \frac{\kappa_0\, \mu_0 + n\, \bar{y}}{\kappa_0 + n},

so :math:`\mu_n` is the familiar weighted average, with :math:`\kappa_0` acting as an **equivalent
sample size** for the prior mean. The posterior scale :math:`\sigma_n^2` combines three ingredients:
the prior sum of squares, the sample sum of squares, and a term
:math:`\frac{\kappa_0 n}{\kappa_0 + n}(\bar{y} - \mu_0)^2` that **inflates the variance when the data
and the prior disagree** — a built-in conflict detector.

Marginals, and the limit
--------------------------

Marginalising :math:`\sigma^2` again yields a :math:`t`, now centred between prior and data:

.. math::

   \mu \mid y \;\sim\; t_{\,\nu_n}\!\left(\mu_n,\; \frac{\sigma_n^2}{\kappa_n}\right).

Setting :math:`\kappa_0 \to 0` and :math:`\nu_0 \to -1` recovers the noninformative
:math:`p(\mu, \sigma^2) \propto 1/\sigma^2` of the previous lesson: the flat prior is the **limiting
case** of the conjugate one, carrying zero pseudo-observations.

.. code-block:: python

   mu0, k0, nu0, s0sq = 5.0, 2, 2, 0.25       # prior: mean worth 2 obs, var worth 2 obs
   n, ybar, s2 = len(y), y.mean(), y.var(ddof=1)
   kn, nun = k0 + n, nu0 + n
   mun = (k0 * mu0 + n * ybar) / kn
   snsq = (nu0 * s0sq + (n - 1) * s2 + k0 * n / kn * (ybar - mu0) ** 2) / nun

Convenient, but coupled
-------------------------

The tidy algebra has a cost worth naming: this prior **couples** :math:`\mu` and :math:`\sigma^2` — a
belief about the mean is expressed in units of the unknown variance. If that dependence is not
something you actually believe, the conjugate family is buying convenience at the price of realism, and
a non-conjugate prior fitted by MCMC is the more honest route.
"""

CONTENT["Multinomial Model for Categorical Data"] = r"""
Beyond two categories
-----------------------

Generalise the binomial from two outcomes to :math:`k`. Each observation falls in exactly one of
:math:`k` categories with probabilities :math:`\theta = (\theta_1, \dots, \theta_k)`,
:math:`\sum_j \theta_j = 1`, and the counts :math:`y = (y_1, \dots, y_k)` follow a **multinomial**
likelihood:

.. math::

   p(y \mid \theta) \;\propto\; \prod_{j=1}^{k} \theta_j^{\,y_j} .

Poll responses, survey categories, and the components of the mixture models in Part V all live here.

The Dirichlet prior
---------------------

The conjugate prior on the simplex is the **Dirichlet**, the Beta's multivariate sibling:
:math:`\theta \sim \mathrm{Dirichlet}(\alpha_1, \dots, \alpha_k)`, with density proportional to
:math:`\prod_j \theta_j^{\,\alpha_j - 1}`. Same functional form as the likelihood — so the update is
again pure counting:

.. math::

   \theta \mid y \;\sim\; \mathrm{Dirichlet}(\alpha_1 + y_1,\; \dots,\; \alpha_k + y_k).

Each :math:`\alpha_j` is a **prior count** in category :math:`j`, and :math:`\sum_j \alpha_j` is the
prior sample size. :math:`\mathrm{Dirichlet}(1, \dots, 1)` is uniform on the simplex; Jeffreys' choice
is all :math:`\alpha_j = \tfrac12`.

Contrasts come free
---------------------

The real payoff is that questions about **differences** are answered directly from draws — no delta
method, no covariance algebra:

.. code-block:: python

   import numpy as np
   y = np.array([420, 380, 200])            # candidate A, B, undecided
   alpha = np.ones(3)                       # uniform prior
   draws = np.random.dirichlet(alpha + y, size=20_000)
   lead = draws[:, 0] - draws[:, 1]         # posterior of the A - B margin
   lead.mean(), (lead > 0).mean()           # P(A leads B | data) ≈ 0.91

Two structural facts
----------------------

Marginally, each :math:`\theta_j` is :math:`\mathrm{Beta}(\alpha_j,\ \alpha_0 - \alpha_j)` — collapse
the other categories and the binomial reappears. And the components are **negatively correlated** by
construction: they must sum to one, so probability given to one category is taken from another. That
constraint is exactly what makes the simplex the natural home for mixture weights in Stage 16.
"""

CONTENT["Multivariate Normal Model with Known Variance"] = r"""
Vectors instead of scalars
----------------------------

Extend the normal model to :math:`d` dimensions: observations :math:`y_i \sim \mathrm{N}(\theta,
\Sigma)` are now **vectors**, :math:`\theta` is a mean vector, and :math:`\Sigma` is a known
:math:`d \times d` covariance matrix. With a conjugate prior :math:`\theta \sim \mathrm{N}(\mu_0,
\Lambda_0)`, every scalar formula from Stage 2 survives — with matrices in place of numbers.

Precision matrices add
------------------------

Working with **precision matrices** (:math:`\Sigma^{-1}`, :math:`\Lambda_0^{-1}`) makes the result a
direct translation of the scalar case. The posterior is normal, precisions add, and the mean is a
precision-weighted average:

.. math::

   \theta \mid y \sim \mathrm{N}(\mu_n, \Lambda_n), \qquad
   \Lambda_n^{-1} = \Lambda_0^{-1} + n\,\Sigma^{-1},

.. math::

   \mu_n = \Lambda_n \left( \Lambda_0^{-1} \mu_0 + n\, \Sigma^{-1} \bar{y} \right).

Set :math:`d = 1` and these collapse to the scalar formulas exactly. A flat prior
(:math:`\Lambda_0^{-1} \to 0`) gives :math:`\theta \mid y \sim \mathrm{N}(\bar{y}, \Sigma / n)`.

.. code-block:: python

   import numpy as np
   Sigma = np.array([[1.0, 0.5], [0.5, 2.0]])          # known covariance
   L0    = np.array([[10.0, 0.0], [0.0, 10.0]])        # vague prior covariance
   mu0   = np.zeros(2)
   Y = np.random.multivariate_normal([1.0, -1.0], Sigma, size=30)
   n, ybar = len(Y), Y.mean(axis=0)

   Ln = np.linalg.inv(np.linalg.inv(L0) + n * np.linalg.inv(Sigma))
   mun = Ln @ (np.linalg.inv(L0) @ mu0 + n * np.linalg.inv(Sigma) @ ybar)

Correlation carries information
---------------------------------

The genuinely multivariate feature is that :math:`\Sigma` **couples** the components: observing one
coordinate informs the others whenever they are correlated. The posterior for :math:`\theta` inherits a
full covariance, so any **contrast** :math:`c^{\top}\theta` (a difference of means, a linear
combination) has posterior :math:`\mathrm{N}(c^{\top}\mu_n,\ c^{\top}\Lambda_n c)` — computed from
draws with a single dot product. Assuming :math:`\Sigma` known is of course a fiction, which the next
lesson removes.
"""


MINDMAP.update({
    "Normal Data with a Noninformative Prior Distribution": [
        "Averaging Over Nuisance Parameters", "Normal Data with a Conjugate Prior Distribution",
        "Noninformative Prior Distributions", "Normal Distribution with Known Variance",
    ],
    "Normal Data with a Conjugate Prior Distribution": [
        "Normal Data with a Noninformative Prior Distribution",
        "Informative Prior Distributions", "Normal Distribution with Known Variance",
        "Multivariate Normal with Unknown Mean and Variance",
    ],
    "Multinomial Model for Categorical Data": [
        "Estimating a Probability from Binomial Data",
        "Setting up and interpreting mixture models",
        "Dirichlet process prior distributions",
        "Models for multivariate and multinomial responses",
    ],
    "Multivariate Normal Model with Known Variance": [
        "Normal Distribution with Known Variance",
        "Multivariate Normal with Unknown Mean and Variance",
        "Averaging Over Nuisance Parameters", "Gaussian process regression",
    ],
})


# ----------------------------------------------------------------------
# Part I / Stage 3 — Multiparameter Models (cont.)  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Multivariate Normal with Unknown Mean and Variance"] = r"""
The full multivariate model
-----------------------------

Drop the fiction that :math:`\Sigma` is known. With :math:`y_i \sim \mathrm{N}(\theta, \Sigma)` and
both unknown, the conjugate prior is the matrix version of the normal--inverse-:math:`\chi^2` from
earlier in this stage: a **normal--inverse-Wishart**,

.. math::

   \Sigma \sim \text{Inv-Wishart}(\nu_0, \Lambda_0^{-1}), \qquad
   \theta \mid \Sigma \sim \mathrm{N}\!\left(\mu_0,\, \frac{\Sigma}{\kappa_0}\right).

Again the mean's prior is scaled by the very covariance being estimated, and again :math:`\kappa_0`
and :math:`\nu_0` are **equivalent sample sizes** — for the mean and for the covariance respectively.

The update
------------

Conjugacy holds, and the hyperparameters update by addition exactly as in the scalar case:

.. math::

   \kappa_n = \kappa_0 + n, \qquad \nu_n = \nu_0 + n, \qquad
   \mu_n = \frac{\kappa_0 \mu_0 + n \bar{y}}{\kappa_0 + n},

.. math::

   \Lambda_n = \Lambda_0 + S + \frac{\kappa_0 n}{\kappa_0 + n}
               (\bar{y} - \mu_0)(\bar{y} - \mu_0)^{\top},

where :math:`S = \sum_i (y_i - \bar{y})(y_i - \bar{y})^{\top}`. The last term is the multivariate
**conflict detector**: prior–data disagreement about the mean inflates the posterior covariance.
Marginalising :math:`\Sigma` gives a **multivariate** :math:`t` for :math:`\theta`, the vector analogue
of the scalar result.

Why the inverse-Wishart disappoints
-------------------------------------

The inverse-Wishart is conjugate, guarantees positive-definiteness, and makes Gibbs sampling trivial —
which is why it dominated Bayesian software for decades. It is also, in Gelman's own later assessment,
a **poor default**. A *single* degrees-of-freedom parameter :math:`\nu_0` controls the certainty of
**every** variance at once; the implied marginal for each variance has **little density near zero**,
biasing small variances upward; and it forces an **a priori dependence between correlations and
variances** that nobody believes.

The modern alternative
------------------------

Modern practice decomposes :math:`\Sigma` into **scales** and a **correlation matrix**, giving each its
own prior — an **LKJ** prior on the correlations, half-normal or half-Cauchy on the standard
deviations:

.. code-block:: python

   import pymc as pm
   with pm.Model():
       sd = pm.HalfNormal("sd", sigma=1, shape=d)             # scales, separately
       chol, corr, _ = pm.LKJCholeskyCov("Sigma", n=d, eta=2,  # correlations, separately
                                          sd_dist=pm.HalfNormal.dist(1))
       pm.MvNormal("y", mu=theta, chol=chol, observed=Y)

Conjugacy bought tractability when computation was scarce. With HMC available, the honest prior wins —
a theme that recurs whenever Part I's closed forms meet Part III's samplers.
"""

CONTENT["Example: Bayesian analysis of a bioassay experiment (logistic, nonconjugate)"] = r"""
Where the closed forms end
----------------------------

Every posterior so far has had a formula. This one does not, and that is the point. In a **bioassay**,
groups of animals receive increasing **doses** :math:`x_i` of a compound and the number of **deaths**
:math:`y_i` out of :math:`n_i` is recorded. A typical experiment is tiny — four dose groups, five
animals each.

The logistic dose–response model
----------------------------------

Deaths are binomial with a dose-dependent probability, modelled on the **log-odds** scale:

.. math::

   y_i \sim \mathrm{Binomial}(n_i,\, \theta_i), \qquad
   \mathrm{logit}(\theta_i) = \log\frac{\theta_i}{1 - \theta_i} = \alpha + \beta x_i .

Here :math:`\beta` is the **dose effect** — the scientific question is usually whether
:math:`\beta > 0`. A Beta prior cannot help: the parameters :math:`(\alpha, \beta)` enter through a
nonlinear link, so **no conjugate prior exists** and the posterior

.. math::

   p(\alpha, \beta \mid y) \;\propto\; p(\alpha, \beta)
   \prod_{i} \bigl[\mathrm{logit}^{-1}(\alpha + \beta x_i)\bigr]^{y_i}
             \bigl[1 - \mathrm{logit}^{-1}(\alpha + \beta x_i)\bigr]^{n_i - y_i}

has no closed form. This is the normal situation in applied work; Stage 2's tidy algebra was the
exception.

Compute it on a grid
----------------------

With only **two** parameters, you can simply **evaluate** the unnormalised posterior over a grid,
normalise by summing, and sample from the discrete approximation. It is brute force, and it is exact
enough to be a benchmark for the samplers of Part III.

.. code-block:: python

   import numpy as np
   from scipy.special import expit

   x = np.array([-0.86, -0.30, -0.05, 0.73])       # log dose
   n = np.array([5, 5, 5, 5]); y = np.array([0, 1, 3, 5])

   a, b = np.meshgrid(np.linspace(-5, 10, 400), np.linspace(-10, 40, 400))
   th = expit(a[..., None] + b[..., None] * x)     # broadcast over dose groups
   logpost = (y * np.log(th) + (n - y) * np.log1p(-th)).sum(-1)   # flat prior
   post = np.exp(logpost - logpost.max()); post /= post.sum()
   (post.sum(axis=0) @ np.linspace(-5, 10, 400))   # E[alpha | y], etc.

Reading the answer
--------------------

The posterior for :math:`(\alpha, \beta)` is **skewed and correlated** — no normal approximation would
capture its banana shape well — and :math:`\Pr(\beta > 0 \mid y)` is essentially 1: the dose kills.
Because any function of draws is itself a posterior draw, the **LD50** (the dose at which
:math:`\theta = 0.5`, namely :math:`-\alpha / \beta`) comes free, with a wide and asymmetric interval
that a plug-in estimate would badly misrepresent. Grids work in two dimensions; beyond three or four
they die of dimensionality — which is precisely why Part III exists.
"""

CONTENT["Summary of Elementary Modeling and Computation"] = r"""
What Part I established
-------------------------

Everything so far rests on a single move: write a **full probability model**
:math:`p(\theta, y) = p(\theta) p(y \mid \theta)`, condition on the data, and summarise. The stages
built this out — one parameter, then several — while the arithmetic stayed the same. Three ideas do
most of the work.

The recurring three
---------------------

* **The posterior is a compromise.** Beta–Binomial, Normal–Normal, Poisson–Gamma, the multivariate
  normal: in each, the posterior mean is a **weighted average** of prior and data, with weights given
  by **effective sample size** or **precision**. Data eventually win; the prior matters most when they
  are scarce.
* **Nuisance parameters are integrated, not fixed.** Marginalising honestly is what turns a normal into
  a :math:`t`. Plug-in estimates discard the second term of the variance decomposition and understate
  uncertainty.
* **Priors are choices, and they are checkable.** Informative (pseudo-data), noninformative (invariant,
  possibly improper), weakly informative (regularising) — each states something, and prior predictive
  simulation reveals what.

Two computational lessons
---------------------------

**Conjugacy is a convenience.** It gives exact, instant, interpretable updates, and it will return in
Part III as the engine inside Gibbs samplers. But it exists for algebraic reasons, not scientific ones,
and the bioassay showed how quickly a realistic model escapes it.

**Draws are enough.** Every quantity of interest — a mean, an interval, :math:`\Pr(\beta > 0 \mid y)`,
the LD50, a contrast — is a summary of posterior draws. Functions of draws are draws from the function's
posterior; **dropping a column is marginalisation**. This is why the simulation-based workflow scales
where the algebra cannot.

.. code-block:: python

   # the entire Part I workflow, in five lines
   import pymc as pm, arviz as az
   with pm.Model():
       theta = pm.Beta("theta", 1, 1)                 # prior
       pm.Binomial("y", n=10, p=theta, observed=8)    # likelihood
       idata = pm.sample()                            # condition on data
   az.summary(idata)                                  # summarise; then check the fit

What comes next
-----------------

Part I answered *how to compute a posterior* when the model is given. Three questions remain, and they
organise the rest of the course: **is the model any good?** (checking and comparison, Part II);
**what if the posterior has no formula?** (approximation and MCMC, Part III); and **how do we build
models with structure** — groups, predictors, nonlinearity? (hierarchies, regression, nonparametrics,
Parts IV–V). The next stage begins the answer by asking what happens to a posterior as data
accumulate.
"""


# ----------------------------------------------------------------------
# Part I / Stage 4 — Asymptotics & Frequentist Ties
# ----------------------------------------------------------------------

CONTENT["Normal Approximations to the Posterior Distribution"] = r"""
A quadratic on the log scale
------------------------------

Posteriors are often awkward; **normal** distributions are not. The **normal (Laplace)
approximation** exploits a happy fact: near its peak, a smooth log-posterior looks like a downward
parabola. Expand :math:`\log p(\theta \mid y)` in a Taylor series about the **posterior mode**
:math:`\hat{\theta}`, where the gradient vanishes:

.. math::

   \log p(\theta \mid y) \;\approx\; \log p(\hat{\theta} \mid y)
   - \tfrac{1}{2} (\theta - \hat{\theta})^{\top} I(\hat{\theta}) (\theta - \hat{\theta}) .

Exponentiating a quadratic gives a normal. So

.. math::

   p(\theta \mid y) \;\approx\; \mathrm{N}\!\left(\hat{\theta},\; I(\hat{\theta})^{-1}\right),
   \qquad
   I(\hat{\theta}) = -\left. \frac{\partial^2 \log p(\theta \mid y)}
                                  {\partial \theta \, \partial \theta^{\top}}
                     \right|_{\theta = \hat{\theta}} ,

where :math:`I(\hat{\theta})` is the **observed information** — the curvature at the mode. Sharp peak,
high information, small variance.

Computing it
--------------

Optimise, then differentiate twice:

.. code-block:: python

   import numpy as np
   from scipy.optimize import minimize

   neg_log_post = lambda th: -log_posterior(th, y)      # any log posterior
   fit = minimize(neg_log_post, x0, method="BFGS")
   mode = fit.x
   cov = fit.hess_inv                    # inverse observed information ≈ posterior covariance
   sd = np.sqrt(np.diag(cov))            # approximate posterior standard deviations

Two lines give a point estimate and an uncertainty — for pennies, compared to MCMC.

When it works, and when it lies
---------------------------------

The approximation improves as :math:`n` grows (next lesson), and it is excellent for smooth,
unimodal, roughly symmetric posteriors. It **fails**, sometimes badly, when the posterior is
**skewed** (the bioassay's banana), **multimodal** (mixtures, Stage 16), constrained near a
**boundary** (a variance close to zero), or **heavy-tailed** — and it always describes only the
neighbourhood of *one* mode. A common repair is to approximate on a **transformed** scale where the
posterior is more nearly normal (:math:`\log \sigma` rather than :math:`\sigma`,
:math:`\mathrm{logit}\,\theta` rather than :math:`\theta`), then transform the draws back.

Still useful
--------------

Even where it is not the final answer, the normal approximation earns its keep: as a fast **first
look**, as a source of **starting values and proposal scales** for MCMC, and as the analytical device
behind **variational** and **modal** methods in Part III. Its accuracy is exactly the subject of
large-sample theory — and its failure modes are the subject of the counterexamples two lessons on.
"""


MINDMAP.update({
    "Multivariate Normal with Unknown Mean and Variance": [
        "Multivariate Normal Model with Known Variance",
        "Normal Data with a Conjugate Prior Distribution",
        "Weakly Informative Priors for Variance Parameters",
        "Hierarchical models for batches of variance components",
    ],
    "Example: Bayesian analysis of a bioassay experiment (logistic, nonconjugate)": [
        "Standard generalized linear model likelihoods",
        "Numerical integration", "Normal Approximations to the Posterior Distribution",
        "Summary of Elementary Modeling and Computation",
    ],
    "Summary of Elementary Modeling and Computation": [
        "The three steps of Bayesian data analysis",
        "Example: Bayesian analysis of a bioassay experiment (logistic, nonconjugate)",
        "Posterior as a Compromise Between Data and Prior Information",
        "The Place of Model Checking in Applied Bayesian Statistics",
    ],
    "Normal Approximations to the Posterior Distribution": [
        "Large-Sample Theory", "Finding posterior modes",
        "Counterexamples to large-sample (asymptotic) Bayesian theorems",
        "Summarizing Posterior Inference",
    ],
})


# ----------------------------------------------------------------------
# Part I / Stage 4 — Asymptotics & Frequentist Ties (cont.)  [completes]
# ----------------------------------------------------------------------

CONTENT["Large-Sample Theory"] = r"""
What happens as data pile up
------------------------------

The normal approximation of the last lesson is not merely convenient — under regularity conditions it
becomes **exact in the limit**. Large-sample theory describes the two things a posterior does as
:math:`n \to \infty`: it **concentrates**, and its shape becomes **normal**.

Consistency: the posterior concentrates
-----------------------------------------

If the model is correctly specified and the prior gives positive probability to a neighbourhood of the
true value :math:`\theta_0`, the posterior piles up on :math:`\theta_0`: for any :math:`\epsilon > 0`,
:math:`\Pr(|\theta - \theta_0| > \epsilon \mid y) \to 0`. Two consequences follow, and both were
foreshadowed in Stage 2. The **prior washes out** — its fixed weight :math:`\alpha + \beta` is swamped
by growing :math:`n` — so analysts with different reasonable priors are driven into agreement. And a
prior that assigns **zero** probability to the truth can never recover: *Cromwell's rule*, the reason
weakly informative beats dogmatically informative.

Bernstein–von Mises: the shape becomes normal
-----------------------------------------------

More sharply, the **Bernstein–von Mises theorem** says the posterior converges (in total variation) to
a normal centred at the maximum-likelihood estimator, with covariance the inverse Fisher information:

.. math::

   p(\theta \mid y) \;\longrightarrow\;
   \mathrm{N}\!\left(\hat{\theta}_{\mathrm{MLE}},\; \frac{1}{n}\, I(\theta_0)^{-1}\right).

The posterior standard deviation therefore shrinks like :math:`1/\sqrt{n}` — halving it costs **four
times** the data. The prior enters the limit **not at all**; only the likelihood survives.

Why it matters
----------------

Bernstein–von Mises is the bridge between the two schools: asymptotically, a Bayesian **credible**
interval *is* a frequentist **confidence** interval, with correct long-run coverage. It licenses the
normal approximation as a fast substitute for MCMC in large, regular problems, and it explains why
Bayesian and classical answers so often agree when data are plentiful.

.. code-block:: python

   import numpy as np
   from scipy import stats
   theta_true, a, b = 0.3, 2, 8               # informative prior, mean 0.2
   for n in (10, 100, 10_000):
       y = stats.binom(n, theta_true).rvs(random_state=0)
       post = stats.beta(a + y, b + n - y)
       print(n, round(post.mean(), 3), round(post.std(), 4))
   # posterior mean -> 0.3 ; posterior sd shrinks ~ 1/sqrt(n)

The fine print
----------------

Every word above rests on **regularity conditions**: the true value lies in the **interior** of the
parameter space, the parameter is **identified**, the dimension is **fixed** (not growing with
:math:`n`), and the model is **correctly specified**. Under misspecification the posterior still
becomes normal around the best-fitting parameter, but its variance is **no longer** the inverse Fisher
information — so credible intervals lose their coverage guarantee. The next lesson collects the cases
where these conditions break.
"""

CONTENT["Counterexamples to large-sample (asymptotic) Bayesian theorems"] = r"""
When the guarantees fail
--------------------------

Large-sample theory is a promise with fine print, and the fine print matters. Each regularity condition
can fail in practice, and when it does, the reassuring picture — concentration, normality, correct
coverage — can fail with it.

At the boundary
-----------------

If the true value sits on the **edge** of the parameter space, the posterior cannot be normal there: it
has nowhere to put mass on one side. A hierarchical variance :math:`\tau^2` whose true value is
**zero** (groups genuinely identical) produces a posterior heaped against the boundary — asymmetric,
non-normal, and badly summarised by a mode and curvature. The same happens for a probability at 0 or 1.
The normal approximation on a **transformed** scale (:math:`\log \tau`) helps, but the boundary case is
inherently awkward.

Not identified, or growing
----------------------------

**Unidentified** parameters — those the likelihood cannot distinguish — never concentrate; the
posterior in that direction remains the prior, however much data arrive. **Label switching** in
mixtures (Stage 16) is a benign instance; a multimodal likelihood is a harsher one. And when the
number of parameters **grows with** :math:`n` — one per observation, as in the classic
Neyman–Scott problem — consistency for the parameters of interest can fail outright, since new data
bring new unknowns.

Infinite dimensions
---------------------

Most striking are the **nonparametric** counterexamples of Diaconis and Freedman: with an
infinite-dimensional parameter, seemingly innocuous priors yield posteriors that **converge to the
wrong answer**. In some of their examples, the posterior mean and density converge on a false value
while the posterior **mode** remains consistent — a warning that in infinite dimensions, intuition
built on finite parameter counts is not merely imprecise but wrong. The nonparametric models of Part V
must therefore be chosen with care, not adopted casually.

Misspecification, the everyday case
-------------------------------------

The likeliest failure is the mundane one: **the model is wrong**. The posterior then concentrates on
the parameter minimising Kullback–Leibler divergence from the truth — the "best available lie" — and
becomes normal around it, but with a **sandwich** variance rather than the inverse Fisher information.
Credible intervals shrink like :math:`1/\sqrt{n}` around a value that is not the truth, growing more
confident and no less wrong.

.. code-block:: python

   # symptom, not proof: posterior concentrating away from any sensible value
   # while posterior predictive checks fail -> suspect misspecification, not sample size

The moral is not that asymptotics are useless but that they are **conditional**. They justify
approximations in regular, correctly specified, fixed-dimension problems — and they justify **model
checking** everywhere else.
"""

CONTENT["Frequency Evaluations of Bayesian Inferences"] = r"""
A different question
----------------------

Bayesian inference conditions on the data you have. **Frequency evaluation** asks a different, entirely
legitimate question: if this procedure were applied repeatedly, across many datasets, how would it
behave? The two views are not rivals — the frequency properties of a Bayesian method are a way of
**checking** it, and Gelman treats them as diagnostics rather than definitions.

Coverage
----------

The headline criterion is **calibration** of intervals: do 95% credible intervals contain the true
parameter 95% of the time? Bernstein–von Mises promises this **asymptotically**. In small samples the
answer depends on the prior — and often, informatively, in the Bayesian method's favour: a
weakly informative prior that shrinks noisy estimates can achieve **better** coverage and much smaller
average error than an unbiased estimator, particularly in the many-parameter, small-:math:`n` settings
(the cancer-rate map) where classical methods flounder.

Simulate to check
-------------------

The evaluation is mechanical: draw parameters from the prior, simulate data from the model, fit, and
count. This **simulation-based calibration** checks the prior, the likelihood **and** the sampler
together — if a 50% interval covers 70% of the time, something is wrong somewhere.

.. code-block:: python

   import numpy as np
   from scipy import stats
   rng = np.random.default_rng(0)
   n, a, b, cover = 20, 2, 8, 0
   for _ in range(2000):
       theta = stats.beta(a, b).rvs(random_state=rng)      # draw from the prior
       y = stats.binom(n, theta).rvs(random_state=rng)     # simulate data
       lo, hi = stats.beta(a + y, b + n - y).interval(0.95)
       cover += lo <= theta <= hi
   cover / 2000        # ≈ 0.95 when prior, model and computation all agree

Bias, variance, and honesty
-----------------------------

Bayesian estimators are typically **biased** — that is what shrinkage means — and typically achieve
lower **mean squared error** for it, trading a little bias for a lot of variance. Frequency evaluation
makes the bargain explicit rather than hiding it. Two honest limits: the calibration above averages
over the **prior**, so it certifies the procedure only if that prior is the one you believe; and no
amount of frequency checking rescues a **misspecified likelihood**. Coverage is a necessary condition
for trust, not a sufficient one.
"""

CONTENT["Bayesian interpretations of other statistical methods"] = r"""
Priors in disguise
--------------------

Many familiar non-Bayesian procedures turn out to be Bayesian estimates under some prior. Recognising
this is not point-scoring: it clarifies **what a method assumes**, and it converts tuning parameters
chosen by cross-validation into statements about prior beliefs — which can then be examined, criticised
and improved.

Penalised likelihood is MAP estimation
----------------------------------------

Maximising a **penalised** log-likelihood is exactly finding a **posterior mode**:

.. math::

   \underbrace{\arg\max_{\theta} \; \bigl[\log p(y \mid \theta) + \log p(\theta)\bigr]}_{\text{MAP}}
   \;\;=\;\;
   \underbrace{\arg\min_{\theta} \; \bigl[-\log p(y \mid \theta)
              + \lambda\, \mathrm{pen}(\theta)\bigr]}_{\text{penalised likelihood}} ,

with the penalty as the negative log prior. Two headline cases:

* **Ridge regression** :math:`= ` MAP under a **Gaussian** prior :math:`\beta_j \sim \mathrm{N}(0,
  \tau^2)`, with :math:`\lambda = \sigma^2 / \tau^2`. The :math:`\ell_2` penalty *is*
  :math:`-\log p(\beta)` up to a constant.
* **Lasso** :math:`= ` MAP under a **Laplace** (double-exponential) prior, whose peak at zero produces
  exact zeros at the mode.

Maximum likelihood itself is MAP under a flat prior — which is why, per Bernstein–von Mises, the two
agree asymptotically.

Where the analogy stops
-------------------------

The correspondence is between **point estimates**, not distributions, and that is the catch. The lasso's
sparse solution is a property of the **mode**; the full posterior under a Laplace prior puts **zero**
probability on any coefficient being exactly zero. Reporting the mode alone hides this. The Bayesian
version supplies uncertainty — and reveals that the sparsity was an artifact of the summary. (Modern
Bayesian sparsity uses **horseshoe** or spike-and-slab priors instead.)

.. code-block:: python

   from sklearn.linear_model import Ridge
   import numpy as np
   # Ridge with alpha = sigma^2 / tau^2 reproduces the Gaussian-prior posterior MEAN
   # (for linear-Gaussian models the mode and mean coincide)
   Ridge(alpha=1.0).fit(X, y).coef_

Others in the family
----------------------

The list extends: **shrinkage/empirical-Bayes** estimators (James–Stein) are hierarchical posteriors
with hyperparameters fitted from data; **smoothing splines** are Gaussian-process posteriors with a
roughness prior (Stage 15); **regularised logistic regression** is the weakly-informative prior that
cures separation (Part IV). The Bayesian reading gives each a language for **why** it works: not "the
penalty stabilises the fit", but "here is what the analysis assumes about the world" — a claim that can
be stated, checked, and defended.
"""


MINDMAP.update({
    "Large-Sample Theory": [
        "Normal Approximations to the Posterior Distribution",
        "Counterexamples to large-sample (asymptotic) Bayesian theorems",
        "Posterior as a Compromise Between Data and Prior Information",
        "Frequency Evaluations of Bayesian Inferences",
    ],
    "Counterexamples to large-sample (asymptotic) Bayesian theorems": [
        "Large-Sample Theory", "Normal Approximations to the Posterior Distribution",
        "Dirichlet process prior distributions", "Label switching and posterior computation",
    ],
    "Frequency Evaluations of Bayesian Inferences": [
        "Large-Sample Theory", "Bayesian interpretations of other statistical methods",
        "Informative Prior Distribution for Cancer Rates", "Posterior predictive checking",
    ],
    "Bayesian interpretations of other statistical methods": [
        "Frequency Evaluations of Bayesian Inferences", "Large-Sample Theory",
        "Regularization and dimension reduction", "Weakly Informative Prior Distributions",
    ],
})


# ----------------------------------------------------------------------
# Part I / Stage 5 — Hierarchical Models
# ----------------------------------------------------------------------

CONTENT["Constructing a Parameterized Prior Distribution"] = r"""
Where does the prior come from?
---------------------------------

Every analysis so far fixed the prior's parameters by hand: :math:`\mathrm{Beta}(2, 8)`,
:math:`\mathrm{N}(0, 2.5^2)`. But in the cancer-rate example the honest answer was that the "national
rate" :math:`\alpha/\beta` should have been **learned from the counties themselves**. That single move
— treating the prior's parameters as **unknowns with their own distribution** — is the whole of
hierarchical modelling.

Hyperparameters and hyperpriors
---------------------------------

Give the prior parameters a name and a distribution. The parameters :math:`\phi` of the prior
:math:`p(\theta \mid \phi)` are **hyperparameters**; the distribution :math:`p(\phi)` placed over them
is a **hyperprior**. The joint model gains a level:

.. math::

   p(\phi, \theta, y) = \underbrace{p(\phi)}_{\text{hyperprior}}
                        \; \underbrace{p(\theta \mid \phi)}_{\text{population}}
                        \; \underbrace{p(y \mid \theta)}_{\text{likelihood}} .

Nothing new is required to fit it: condition on :math:`y`, and marginalise whatever you do not want.
The population distribution :math:`p(\theta \mid \phi)` is now a **prior estimated from data** — not
circular, because it is estimated from *all* the groups jointly while each group's :math:`\theta_j` is
informed by its own data.

Why this is not cheating
--------------------------

The obvious worry: are we using the data twice? No. The model is a single joint distribution written
down **before** seeing :math:`y`, and Bayes' rule is applied once. What looks like "learning the prior"
is simply the marginal posterior :math:`p(\phi \mid y)` — inference about a parameter that happens to
sit one level up. The distinction from **empirical Bayes**, which plugs in a point estimate
:math:`\hat{\phi}` and thereby understates uncertainty, is exactly the nuisance-parameter lesson: a
full Bayesian analysis **integrates** over :math:`\phi`.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # hyperprior: the population's location and spread are unknowns
       mu  = pm.Normal("mu", 0, 5)
       tau = pm.HalfNormal("tau", 5)
       # population distribution: each group's parameter drawn from it
       theta = pm.Normal("theta", mu, tau, shape=J)
       pm.Normal("y", theta[group], sigma_obs, observed=y)

What it buys
--------------

Three things, all visible in the next lessons. Estimates for data-poor groups are **stabilised** by
borrowing strength from the others. The **amount** of borrowing is not chosen by the analyst but
**inferred** from how similar the groups turn out to be. And the uncertainty in that similarity
propagates into every group's interval. But the construction needs a justification — why should
:math:`\theta_j` share a distribution at all? That is **exchangeability**, and it is the next lesson.
"""

CONTENT["Exchangeability and hierarchical models"] = r"""
The licence to pool
---------------------

Hierarchical models let groups **borrow strength** from one another. What entitles them to? The answer
is **exchangeability**: the assumption that the group parameters are, before seeing the data,
**interchangeable** — that nothing in the labelling distinguishes them.

The definition
----------------

Parameters :math:`\theta_1, \dots, \theta_J` are exchangeable if their joint distribution is invariant
under any permutation :math:`\pi` of the indices:

.. math::

   p(\theta_1, \dots, \theta_J) = p(\theta_{\pi(1)}, \dots, \theta_{\pi(J)}) .

It is a statement of **ignorance, not of equality**. The schools may differ enormously; you simply have
no prior reason, before the data, to expect school 3 to differ from school 7 in any particular
direction. Independent and identically distributed is a special case; exchangeability is weaker and
more useful.

De Finetti's bridge
---------------------

De Finetti's theorem makes the connection precise: an exchangeable sequence can (in the infinite case)
be represented as **iid draws given some parameter**, mixed over a distribution for that parameter,

.. math::

   p(\theta_1, \dots, \theta_J)
   = \int \left[ \prod_{j=1}^{J} p(\theta_j \mid \phi) \right] p(\phi) \; d\phi .

Read the right-hand side: that **is** the hierarchical model of the last lesson. Exchangeability does
not merely permit hierarchy — it **implies** it. The population distribution :math:`p(\theta \mid \phi)`
is not an extra assumption bolted on; it is what exchangeability means.

Three pooling regimes
-----------------------

The hyperparameter governing spread (call it :math:`\tau`) interpolates between two extremes, and the
data choose where to sit:

* :math:`\tau \to 0` — **complete pooling**: all :math:`\theta_j` collapse to a common value; group
  identity is ignored.
* :math:`\tau \to \infty` — **no pooling**: each group is estimated alone, noisily.
* :math:`0 < \tau < \infty` — **partial pooling**: each estimate is shrunk toward the population mean
  by an amount reflecting its own precision and the between-group spread.

When exchangeability fails
----------------------------

If you **do** know something distinguishing — schools differ by funding, counties by population — then
the parameters are not exchangeable as they stand. The repair is not to abandon the hierarchy but to
make the known structure explicit: model :math:`\theta_j` as depending on covariates :math:`x_j`, and
assume exchangeability of the **residuals**. This is **conditional exchangeability**, and it is the
road from hierarchical models to hierarchical *regression* in Part IV.
"""

CONTENT["Bayesian analysis of conjugate hierarchical models"] = r"""
The rat tumours
-----------------

The canonical worked example: :math:`J = 71` historical laboratory experiments, each reporting
:math:`y_j` rats developing tumours out of :math:`n_j`. Some experiments are tiny. Estimating each rate
separately gives wild answers (an experiment with 0 of 5 suggests a rate of exactly zero); pooling them
all into one rate denies that experiments differ. Exchangeability says: model the rates as drawn from a
**common population distribution**.

The three-level model
-----------------------

Binomial likelihood, Beta population, hyperprior on the Beta's parameters:

.. math::

   y_j \mid \theta_j \sim \mathrm{Binomial}(n_j, \theta_j), \qquad
   \theta_j \mid \alpha, \beta \sim \mathrm{Beta}(\alpha, \beta), \qquad
   (\alpha, \beta) \sim p(\alpha, \beta) .

Because the Beta is conjugate to the binomial, the **conditional** posterior of each group is immediate:

.. math::

   \theta_j \mid \alpha, \beta, y \;\sim\; \mathrm{Beta}(\alpha + y_j,\; \beta + n_j - y_j),

which makes :math:`\mathrm{E}[\theta_j \mid \alpha, \beta, y]` the familiar weighted average of the
group's own rate :math:`y_j/n_j` and the population mean :math:`\alpha/(\alpha+\beta)` — with the
population now acting as a **prior estimated from all 71 experiments**. Small experiments are pulled
hard toward the population; large ones barely move.

Two levels, two computations
------------------------------

The hyperparameters are handled by marginalising the group parameters analytically (conjugacy again),
leaving a two-dimensional marginal posterior :math:`p(\alpha, \beta \mid y)` that can be evaluated on a
**grid** — exactly the bioassay trick. Then draw :math:`(\alpha, \beta)`, and draw each
:math:`\theta_j` from its conditional Beta. Modern practice simply hands the whole thing to a sampler:

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # hyperprior on population mean and "prior sample size"
       mu  = pm.Beta("mu", 1, 1)                    # population mean rate
       kap = pm.HalfNormal("kappa", 50)             # concentration = alpha + beta
       theta = pm.Beta("theta", mu * kap, (1 - mu) * kap, shape=J)
       pm.Binomial("y", n=n, p=theta, observed=y)
       idata = pm.sample()

Choosing the hyperprior
-------------------------

One trap deserves naming. A flat prior on :math:`(\alpha, \beta)` is **improper** and yields an
**improper posterior** — the concentration :math:`\alpha + \beta` runs off to infinity. Gelman
reparameterises to the population mean :math:`\alpha/(\alpha+\beta)` and a transformed concentration,
placing a proper prior there. The lesson from Stage 2 returns with teeth: for hierarchical variance and
concentration parameters, **check propriety**, and prefer weakly informative hyperpriors.
"""

CONTENT["Normal model with exchangeable parameters"] = r"""
The hierarchical normal
-------------------------

Replace binomial groups with normal ones and the hierarchy becomes fully transparent — every quantity
has a closed form, and the mechanics of shrinkage can be read off directly. Each of :math:`J` groups
supplies an estimate :math:`\bar{y}_j` of its own mean :math:`\theta_j`, with **known** standard error
:math:`\sigma_j`:

.. math::

   \bar{y}_j \mid \theta_j \sim \mathrm{N}(\theta_j, \sigma_j^2), \qquad
   \theta_j \mid \mu, \tau \sim \mathrm{N}(\mu, \tau^2), \qquad
   (\mu, \tau) \sim p(\mu, \tau).

Here :math:`\mu` is the **population mean** and :math:`\tau` the **between-group standard deviation**:
how different the groups really are.

Shrinkage, in closed form
---------------------------

Conditional on :math:`(\mu, \tau)`, each group's posterior is the Stage 2 normal update — precisions
add, and the mean is precision-weighted:

.. math::

   \theta_j \mid \mu, \tau, y \;\sim\; \mathrm{N}(\hat{\theta}_j,\; V_j), \qquad
   \hat{\theta}_j = \frac{\frac{1}{\sigma_j^2}\bar{y}_j + \frac{1}{\tau^2}\mu}
                         {\frac{1}{\sigma_j^2} + \frac{1}{\tau^2}}, \qquad
   \frac{1}{V_j} = \frac{1}{\sigma_j^2} + \frac{1}{\tau^2}.

Define the **shrinkage factor** :math:`B_j = \sigma_j^2 / (\sigma_j^2 + \tau^2)`; then
:math:`\hat{\theta}_j = (1 - B_j)\, \bar{y}_j + B_j\, \mu`. A noisy group (large :math:`\sigma_j`) has
:math:`B_j` near 1 and is pulled almost entirely to the population mean; a precise group keeps its own
estimate. And crucially, :math:`\tau` is **not chosen** — it is inferred, so the data decide how much
pooling is warranted.

The whole model in code
-------------------------

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu  = pm.Normal("mu", 0, 10)
       tau = pm.HalfNormal("tau", 10)               # weakly informative; never inverse-gamma(eps,eps)
       theta = pm.Normal("theta", mu, tau, shape=J)
       pm.Normal("y", theta, sigma=sigma_j, observed=ybar)   # sigma_j known
       idata = pm.sample(target_accept=0.95)

Two warnings
--------------

When :math:`\tau` is near **zero**, the posterior sits at a boundary — the counterexample from Stage 4 —
and the geometry becomes a funnel that samplers negotiate badly; the standard repair is the
**non-centred** parameterisation, :math:`\theta_j = \mu + \tau \eta_j` with
:math:`\eta_j \sim \mathrm{N}(0,1)`. And the prior on :math:`\tau` **matters**, especially with few
groups: with :math:`J = 8`, a careless inverse-gamma can dominate. Both issues are met head-on in the
eight-schools example that follows.
"""


MINDMAP.update({
    "Constructing a Parameterized Prior Distribution": [
        "Exchangeability and hierarchical models",
        "Informative Prior Distribution for Cancer Rates",
        "Bayesian analysis of conjugate hierarchical models",
        "Averaging Over Nuisance Parameters",
    ],
    "Exchangeability and hierarchical models": [
        "Constructing a Parameterized Prior Distribution",
        "General Notation for Statistical Inference",
        "Normal model with exchangeable parameters",
        "Regression coe\ufb03cients exchangeable in batches",
    ],
    "Bayesian analysis of conjugate hierarchical models": [
        "Estimating a Probability from Binomial Data",
        "Constructing a Parameterized Prior Distribution",
        "Normal model with exchangeable parameters",
        "Informative Prior Distribution for Cancer Rates",
    ],
    "Normal model with exchangeable parameters": [
        "Normal Distribution with Known Variance",
        "Exchangeability and hierarchical models",
        "Example: parallel experiments in eight schools",
        "Weakly Informative Priors for Variance Parameters",
    ],
})


# ----------------------------------------------------------------------
# Part I / Stage 5 — Hierarchical Models (cont.)  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Example: parallel experiments in eight schools"] = r"""
Eight coaching programs
-------------------------

The most-analysed dataset in Bayesian statistics. The Educational Testing Service ran **separate
randomised experiments** in eight high schools to measure whether short-term coaching raised SAT-Verbal
scores. Each school reported an estimated effect :math:`y_j` and its standard error :math:`\sigma_j`,
both taken as given:

.. list-table::
   :header-rows: 1

   * - School
     - A
     - B
     - C
     - D
     - E
     - F
     - G
     - H
   * - Effect :math:`y_j`
     - 28
     - 8
     - -3
     - 7
     - -1
     - 1
     - 8
     - 12
   * - Std. error :math:`\sigma_j`
     - 15
     - 10
     - 16
     - 11
     - 9
     - 11
     - 10
     - 18

School A's effect of 28 points looks dramatic. Its standard error is 15. Should we believe it?

Three analyses
----------------

**No pooling** takes each :math:`y_j` at face value: school A's effect is 28. But the standard errors
are large, and the eight estimates are perfectly consistent with the schools being identical. **Complete
pooling** averages everything into one effect (about 8 points) and denies the schools differ at all.
Neither is credible. **Partial pooling** — the hierarchical normal model of the last lesson — lets the
data decide:

.. math::

   y_j \mid \theta_j \sim \mathrm{N}(\theta_j, \sigma_j^2), \qquad
   \theta_j \mid \mu, \tau \sim \mathrm{N}(\mu, \tau^2), \qquad
   \mu \sim \mathrm{N}(0, 5^2), \quad \tau \sim \text{half-Cauchy}(0, 5).

What the data say about :math:`\tau`
--------------------------------------

The pivotal quantity is :math:`\tau`, the **between-school** standard deviation. Its posterior is
concentrated near **zero** — the eight observed spreads are about what you would see if every school
had the *same* true effect and only sampling noise separated them. Consequently the shrinkage factors
:math:`B_j = \sigma_j^2/(\sigma_j^2 + \tau^2)` are large, and school A's estimate is pulled from 28 down
to roughly **10**, close to the pooled mean, with an interval that comfortably includes zero. The
posterior probability that school A's programme is the best is far below what its raw ranking suggests.

.. code-block:: python

   import numpy as np, pymc as pm
   y = np.array([28, 8, -3, 7, -1, 1, 8, 12])
   sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

   with pm.Model():
       mu  = pm.Normal("mu", 0, 5)
       tau = pm.HalfCauchy("tau", 5)
       eta = pm.Normal("eta", 0, 1, shape=8)        # non-centred: avoids the funnel
       theta = pm.Deterministic("theta", mu + tau * eta)
       pm.Normal("y", theta, sigma=sigma, observed=y)
       idata = pm.sample(target_accept=0.9)
   # (idata.posterior["theta"].sel(theta_dim_0=0) > 28).mean()  -> tiny

The funnel
------------

Fit the **centred** version (:math:`\theta_j \sim \mathrm{N}(\mu, \tau)` directly) and the sampler
reports **divergences**. The reason is geometric: as :math:`\tau \to 0` the :math:`\theta_j` are squeezed
into an ever-narrowing neck, and no single step size explores both the neck and the wide mouth. The
**non-centred** parameterisation above — sampling standard-normal :math:`\eta_j` and rescaling —
decouples them and fixes it. Eight schools is a small dataset that teaches three large lessons:
shrinkage tames dramatic claims, :math:`\tau` is the parameter that matters, and **model geometry**
determines whether your sampler tells the truth.
"""

CONTENT["Hierarchical modeling applied to a meta-analysis"] = r"""
The same model, a different name
----------------------------------

**Meta-analysis** combines results from separate studies of the same question — a drug trial run in
twelve hospitals, a treatment tested in a dozen papers. Each study reports an effect estimate and a
standard error. That is precisely the eight-schools structure, and the Bayesian hierarchical model is
the **random-effects meta-analysis** familiar to medicine, arrived at from first principles rather than
by convention.

The model
-----------

Study :math:`j` estimates :math:`y_j` of its own true effect :math:`\theta_j`, with within-study error
:math:`\sigma_j` treated as known; the true effects are exchangeable draws around a population effect:

.. math::

   y_j \mid \theta_j \sim \mathrm{N}(\theta_j, \sigma_j^2), \qquad
   \theta_j \mid \mu, \tau \sim \mathrm{N}(\mu, \tau^2).

Here :math:`\mu` is the **overall effect** and :math:`\tau` the **between-study heterogeneity**. Classical
meta-analysis calls :math:`\tau = 0` the *fixed-effect* model and :math:`\tau > 0` the *random-effects*
model, then chooses between them by a test. The Bayesian analysis does not choose: it **estimates**
:math:`\tau` and averages over its uncertainty.

Why that matters
------------------

Two payoffs follow. First, the pooled effect's interval **widens honestly** when studies disagree,
because uncertainty in :math:`\tau` propagates into :math:`\mu` — a plug-in :math:`\hat{\tau}` (the
DerSimonian–Laird estimator, say) understates it, and badly so when :math:`J` is small. Second, each
study gets a **shrunken** estimate :math:`\theta_j`, so a small outlying trial is not read literally.
And a genuinely new quantity comes free: the **predictive** effect in a *future* study,
:math:`\tilde{\theta} \sim \mathrm{N}(\mu, \tau^2)`, which is what a clinician deciding for the next
patient actually needs — always wider than the interval for :math:`\mu`.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu  = pm.Normal("mu", 0, 1)                  # scale: log odds ratio
       tau = pm.HalfNormal("tau", 0.5)              # heterogeneity, weakly informative
       eta = pm.Normal("eta", 0, 1, shape=J)
       theta = pm.Deterministic("theta", mu + tau * eta)
       pm.Normal("y", theta, sigma=se, observed=y)  # se = within-study standard errors
       # effect in a new study:
       theta_new = pm.Normal("theta_new", mu, tau)
       idata = pm.sample(target_accept=0.9)

The assumption worth interrogating
------------------------------------

Everything rests on **exchangeability of the studies**. If trials differ systematically — dose,
population, decade, industry funding — they are not exchangeable as they stand, and pooling them
averages apples with oranges. The repair is the one from the exchangeability lesson: model the known
differences as **covariates** (a hierarchical *meta-regression*) and assume exchangeability of what
remains. Publication bias is a harder problem still: it corrupts the likelihood itself, and no prior
fixes a biased sample of studies.
"""

CONTENT["Weakly Informative Priors for Variance Parameters"] = r"""
The parameter that decides everything
---------------------------------------

In a hierarchical model, :math:`\tau` — the group-level standard deviation — governs how much the
groups pool. It is also the parameter the data constrain **least**, especially when the number of
groups :math:`J` is small (eight schools: :math:`J = 8`). So the prior on :math:`\tau` is not a
formality; it can decide the answer.

Why the old default failed
----------------------------

For years the reflex was an **inverse-gamma**(:math:`\epsilon, \epsilon`) prior on :math:`\tau^2`, with
:math:`\epsilon = 0.001`, chosen because it is conjugate and "nearly noninformative". Gelman's 2006
analysis showed it is neither. As :math:`\epsilon \to 0` the prior approaches an improper limit whose
posterior may not exist; and for any small :math:`\epsilon` the prior has **almost no mass near zero**
while placing weight far out in the tail. When the data genuinely suggest :math:`\tau \approx 0` — the
eight-schools case — this prior **fights them**, inflating :math:`\tau` and under-pooling. Worse, the
answer is sensitive to :math:`\epsilon`, a number chosen for its irrelevance.

What to use instead
---------------------

Put the prior on the **standard deviation** :math:`\tau`, not the variance, and choose a density that is
**positive at zero** and has a finite scale:

* **half-normal** — :math:`\tau \sim \mathrm{N}^{+}(0, s)`, light tails, a sensible default when a rough
  scale is known;
* **half-Cauchy** — :math:`\tau \sim \mathrm{C}^{+}(0, s)`, heavy-tailed, allowing large
  :math:`\tau` if the data insist while still regularising;
* **uniform on** :math:`(0, A)` — acceptable when :math:`J` is large and :math:`A` is generous.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       tau = pm.HalfNormal("tau", sigma=5)     # or pm.HalfCauchy("tau", beta=5)
       # NOT: pm.InverseGamma("tau2", alpha=0.001, beta=0.001)
       ...

Scale, and honesty
--------------------

The scale :math:`s` is a real statement: it should be large relative to the plausible group-level
variation and small relative to absurdity. On SAT points (sd ≈ 100) a half-Cauchy(0, 5) says
between-school effects of a few points are ordinary and of fifty points are not. **Standardise, or
scale the prior to the data.**

Two closing notes. Allowing :math:`\tau` to be near zero is a **feature** — the model can discover that
complete pooling is right — but it creates the funnel geometry that demands the non-centred
parameterisation. And with very small :math:`J` (say :math:`J \le 5`), no prior is truly weak: report
the sensitivity, because the prior is doing visible work.
"""


# ----------------------------------------------------------------------
# Part II / Stage 6 — Model Checking & Comparison
# ----------------------------------------------------------------------

CONTENT["The Place of Model Checking in Applied Bayesian Statistics"] = r"""
The third step, at last
-------------------------

Part I was about steps one and two: build a full probability model, condition on data. Everything
downstream — every posterior mean, every credible interval, every decision — is **conditional on the
model being adequate**. Step three asks whether it is. This is not an afterthought appended to a
completed analysis; it is what makes the analysis trustworthy, and it is where Part II begins.

Bayes' rule cannot check itself
---------------------------------

Conditioning is a closed operation: given a model, the posterior follows deterministically. If the
likelihood is wrong, Bayes' rule reports a **coherent, confident, wrong** posterior, and as Stage 4
showed, more data make it more confident without making it less wrong. Nothing **inside** the model
signals the problem. The check must therefore compare the model's implications against something
outside it — the **data**, and what you know about the world.

Three questions
-----------------

Model checking asks, in ascending order of ambition:

* **Do the inferences make sense?** Are parameter values physically possible, effects of plausible
  magnitude, signs in the right direction?
* **Does the model fit the data?** Would data generated by the fitted model resemble the data actually
  observed — in the features you care about? (This is the **posterior predictive check**, next lesson.)
* **How sensitive are the conclusions?** Would a different reasonable prior, likelihood or subset of
  the data change what you would do?

A model is a hypothesis, not a truth
--------------------------------------

Every model is false; the question is whether it is **false in a way that matters** for the purpose at
hand. So checking is purposive: a model may be perfectly adequate for estimating a mean and hopeless for
predicting extremes. Failures are not embarrassments but **discoveries** — each rejected check points at
a missing feature, and the response is to **expand** the model, refit, and check again. The three steps
are a loop.

.. code-block:: python

   import arviz as az
   # the checking toolkit of the coming lessons, in one place
   az.plot_ppc(idata)          # replicated vs observed data
   az.loo(idata)               # leave-one-out predictive accuracy
   az.plot_loo_pit(idata)      # calibration of predictive distributions

Not hypothesis testing
------------------------

A final distinction. Checking is not a significance test that accepts or rejects the model; a
:math:`p`-value that looks extreme tells you *where* the model misses, not whether to "reject" it at
5%. The aim is not a verdict but **understanding** — and, ultimately, a better model.
"""


MINDMAP.update({
    "Example: parallel experiments in eight schools": [
        "Normal model with exchangeable parameters",
        "Hierarchical modeling applied to a meta-analysis",
        "Weakly Informative Priors for Variance Parameters",
        "Model checking for the educational testing example",
    ],
    "Hierarchical modeling applied to a meta-analysis": [
        "Example: parallel experiments in eight schools",
        "Exchangeability and hierarchical models",
        "Normal model with exchangeable parameters",
        "Robust inference for the eight schools",
    ],
    "Weakly Informative Priors for Variance Parameters": [
        "Weakly Informative Prior Distributions",
        "Example: parallel experiments in eight schools",
        "Multivariate Normal with Unknown Mean and Variance",
        "Hierarchical models for batches of variance components",
    ],
    "The Place of Model Checking in Applied Bayesian Statistics": [
        "The three steps of Bayesian data analysis",
        "Do the Inferences from the Model Make Sense?",
        "Posterior predictive checking", "Continuous model expansion",
    ],
})


# ----------------------------------------------------------------------
# Part II / Stage 6 — Model Checking & Comparison (cont.)
# ----------------------------------------------------------------------

CONTENT["Do the Inferences from the Model Make Sense?"] = r"""
The cheapest check
--------------------

Before any simulation, before any :math:`p`-value, look at the answer and ask whether it is **possible**.
This costs nothing and catches a startling share of real errors — coding mistakes, unit confusions,
mis-specified priors, sign flips. External knowledge that never entered the model is exactly what makes
the check informative.

What to look at
-----------------

Read the posterior against what you already know:

* **Magnitude.** Is a coaching effect of 28 SAT points plausible, given that the test is designed to
  resist short-term preparation? Is an odds ratio of 400 credible, or a symptom of separation?
* **Sign and direction.** Does the coefficient point the way theory demands? A negative dose effect in
  a toxicity assay is a red flag, not a discovery.
* **Physical constraints.** Probabilities in :math:`[0,1]`, variances positive, rates non-negative,
  populations integral.
* **Derived quantities.** Push draws through the functions you care about — an LD50, a predicted count,
  a break-even price — and see whether the implied statements are absurd.
* **Extremes.** Look at the widest intervals and the most-shrunken groups. Do they behave as the
  structure of the data predicts?

External validation
---------------------

The strongest version compares model output with information **deliberately excluded** from the fit:
held-out data, a later replication, an independent measurement, a published estimate. Agreement is
weak evidence for the model; sharp disagreement is strong evidence against it, and localises the fault.

.. code-block:: python

   import numpy as np, arviz as az
   post = idata.posterior
   # 1. do parameters respect their constraints?
   assert float(post["tau"].min()) >= 0
   # 2. do derived quantities land in plausible ranges?
   ld50 = -post["alpha"] / post["beta"]
   np.percentile(ld50, [2.5, 50, 97.5])       # compare with known chemistry
   # 3. does the model's implied prediction match a value you did not fit to?
   az.summary(idata, var_names=["mu"])        # against the published meta-analytic estimate

Being wrong usefully
----------------------

An implausible inference has three possible causes, in decreasing order of frequency: a **bug**, a
**bad prior**, or a **wrong likelihood**. Rule them out in that order — check the data pipeline and
units first, then simulate from the prior, then question the model. And keep the caveat honest: a
result that merely *surprises* you may be the finding. The check is against what is **impossible** or
**incoherent**, not against what is merely unexpected.
"""

CONTENT["Posterior predictive checking"] = r"""
Let the model generate data
-----------------------------

The central technique of Part II is disarmingly simple: **if the model fits, data simulated from it
should look like the data you actually saw.** Simulate many replicated datasets from the fitted model,
compare them to the real one, and any **systematic** difference is a failure of the model.

The posterior predictive distribution
---------------------------------------

Replicated data :math:`y^{\text{rep}}` are drawn from the **posterior predictive distribution** —
the likelihood averaged over posterior uncertainty in the parameters:

.. math::

   p(y^{\text{rep}} \mid y) = \int p(y^{\text{rep}} \mid \theta) \; p(\theta \mid y) \; d\theta .

In simulation this is one line per draw: for each posterior draw :math:`\theta^{(s)}`, generate a full
dataset :math:`y^{\text{rep}(s)} \sim p(y \mid \theta^{(s)})`. Note what is averaged in — parameter
uncertainty **and** sampling variability, exactly the two sources present in the real data.

Test quantities
-----------------

You cannot compare whole datasets by eye at scale, so choose a **test quantity** :math:`T(y, \theta)`
summarising the feature you care about: the maximum, the number of zeros, a lag-1 autocorrelation, the
variance-to-mean ratio. Unlike a classical test statistic, :math:`T` may depend on the **parameters**.
The **posterior predictive** :math:`p`-value is the probability that the replicated data are more
extreme than what you observed:

.. math::

   p_B = \Pr\bigl(T(y^{\text{rep}}, \theta) \ge T(y, \theta) \;\bigm|\; y\bigr)
       \;\approx\; \frac{1}{S} \sum_{s=1}^{S}
         \mathbf{1}\bigl\{T(y^{\text{rep}(s)}, \theta^{(s)}) \ge T(y, \theta^{(s)})\bigr\} .

Values near 0 or 1 mean the observed data are **atypical** of the model in that respect.

.. code-block:: python

   import numpy as np, pymc as pm
   with model:
       ppc = pm.sample_posterior_predictive(idata)
   yrep = ppc.posterior_predictive["y"].values.reshape(-1, len(y))

   T = lambda d: d.max()                                # any feature you care about
   p_B = (np.array([T(r) for r in yrep]) >= T(y)).mean()
   p_B                                                  # ≈ 0.5: unremarkable; ≈ 0 or 1: misfit

Read it as a diagnostic, not a verdict
----------------------------------------

Two honest caveats. The check **uses the data twice** — to fit and to test — so :math:`p_B` is
conservative, tending to concentrate near 0.5; it is not calibrated like a frequentist :math:`p`-value
and is not "the probability the model is true". And a check can only find what its test quantity looks
for: choosing :math:`T` = the sample mean will almost always pass, because the model was fitted to match
the mean. **Choose test quantities the model does not automatically reproduce**, and ones that matter for
your purpose. The point is never to accept or reject, but to learn **where** the model misses.
"""

CONTENT["Graphical posterior predictive checks"] = r"""
Look before you compute
-------------------------

A :math:`p`-value collapses a comparison to one number and answers only the question its test quantity
encodes. A **plot** shows the shape of the misfit — and often reveals a failure nobody thought to test
for. Graphical checks are therefore the first move, not the last.

The three displays
--------------------

* **Density overlay.** Draw the density of each of ~50 replicated datasets in light grey, the observed
  data in bold. Systematic offsets appear as the real curve sitting outside the grey band: skew the
  model cannot make, a peak at zero it cannot produce, tails too thin.
* **Test-statistic histogram.** Histogram :math:`T(y^{\text{rep}})` over replications and mark
  :math:`T(y)` with a line. This is the picture of which the Bayesian :math:`p`-value is the caption.
* **Predictive intervals versus observations.** For structured data (time series, groups), plot each
  observation against its 50%/90% predictive interval. Roughly the right proportion should fall inside,
  and the misses should look **random** — a run of consecutive misses signals unmodelled structure.

.. code-block:: python

   import arviz as az
   az.plot_ppc(idata, num_pp_samples=50)                 # density overlay
   az.plot_bpv(idata, kind="t_stat", t_stat="std")       # test statistic vs replications
   az.plot_loo_pit(idata, y="y")                         # calibration: should be uniform

Calibration plots
-------------------

A sharper display asks where each observation falls **within** its own predictive distribution — its
PIT value. If the model is calibrated, those values are **uniform**; a U-shape means predictive
distributions are too narrow (overconfident), a hump in the middle means too wide. LOO-PIT does this
using **leave-one-out** predictions, avoiding the double-use of data that makes ordinary predictive
:math:`p`-values conservative.

Reading a failure
-------------------

The value of the visual grammar is that the *shape* of the discrepancy names the missing feature. Too
many zeros in the data and none in the replications: the likelihood needs a zero-inflation component.
Replicated tails too thin: swap normal errors for :math:`t`. Replications too smooth across groups: the
hierarchy is over-pooling. Each verdict points to a specific **model expansion** — which is the closing
theme of this stage. Choose the plot that would embarrass the model if the model deserved it.
"""

CONTENT["Model checking for the educational testing example"] = r"""
Checking eight schools
------------------------

The hierarchical model of Stage 5 gave a satisfying answer: school A's dramatic 28-point effect shrinks
toward the pooled mean, and :math:`\tau` sits near zero. Satisfying is not the same as **correct**. Step
three demands that we ask whether data generated by this fitted model would look like the eight
estimates actually reported.

Generating replicated schools
-------------------------------

For each posterior draw :math:`(\mu^{(s)}, \tau^{(s)}, \theta^{(s)})`, simulate a complete replicated
experiment — eight new estimates, each with the **same known standard errors**:

.. math::

   y^{\text{rep}(s)}_j \sim \mathrm{N}\bigl(\theta_j^{(s)},\; \sigma_j^2\bigr),
   \qquad j = 1, \dots, 8 .

Now choose test quantities the model does not automatically match. The **largest** of the eight effects
is the natural one, since the largest observed value (28, from school A) is what motivated the analysis
in the first place. The **smallest** and the **range** (largest minus smallest) probe the spread.

.. code-block:: python

   import numpy as np
   yrep = ...                       # shape (S, 8) replicated datasets
   for name, T in [("max", np.max), ("min", np.min),
                   ("range", lambda d: d.max() - d.min())]:
       p_B = (np.array([T(r) for r in yrep]) >= T(y)).mean()
       print(f"{name:6s}  observed {T(y):6.1f}   p_B = {p_B:.2f}")

What the checks say
---------------------

The observed maximum, minimum and range all fall **comfortably inside** their replicated distributions:
these :math:`p`-values are unremarkable, and there is no evidence that the hierarchical model misfits
the data in these respects. Specifically, a largest-of-eight value of 28 is **entirely ordinary** for
eight noisy estimates drawn around a common mean with these standard errors. That is the whole point:
the "striking" result that provoked the study is exactly what the model predicts should happen by
chance.

What a check cannot do
------------------------

Passing does not make the model true. With :math:`J = 8` groups the checks have little power: many
models would survive them, including ones with quite different :math:`\tau` priors. And the check tests
the **likelihood and hierarchy**, not the assumption that the eight schools are exchangeable or that
:math:`\sigma_j` are truly known. Model checking bounds what the data can refute; it does not certify
the assumptions the data cannot see. The honest next step is **sensitivity analysis** — refit under a
:math:`t` population distribution, or a different prior on :math:`\tau`, and see whether any conclusion
you would act on moves.
"""


MINDMAP.update({
    "Do the Inferences from the Model Make Sense?": [
        "The Place of Model Checking in Applied Bayesian Statistics",
        "Posterior predictive checking", "Weakly Informative Prior Distributions",
        "Debugging Bayesian computing",
    ],
    "Posterior predictive checking": [
        "The Place of Model Checking in Applied Bayesian Statistics",
        "Graphical posterior predictive checks",
        "Model checking for the educational testing example",
        "Measures of predictive accuracy",
    ],
    "Graphical posterior predictive checks": [
        "Posterior predictive checking",
        "Model checking for the educational testing example",
        "Continuous model expansion", "Measures of predictive accuracy",
    ],
    "Model checking for the educational testing example": [
        "Example: parallel experiments in eight schools",
        "Posterior predictive checking", "Graphical posterior predictive checks",
        "Robust inference for the eight schools",
    ],
})


# ----------------------------------------------------------------------
# Part II / Stage 6 — Model Checking & Comparison (cont.)
# ----------------------------------------------------------------------

CONTENT["Measures of predictive accuracy"] = r"""
Scoring a model by what it predicts
-------------------------------------

Checking asks whether a model fits. **Comparison** asks which of several models to prefer, and the
Bayesian answer is: the one that would **predict new data best**. That requires a score. The default
is the **log predictive density** — reward a model for putting high probability where the data actually
land, and it is *proper*, meaning it cannot be gamed by reporting a distribution you do not believe.

Expected log pointwise predictive density
-------------------------------------------

For future observations, the target quantity is the **elpd**, the expected log predictive density
summed over data points:

.. math::

   \mathrm{elpd} = \sum_{i=1}^{n} \int p_t(\tilde{y}_i) \,
                   \log p(\tilde{y}_i \mid y) \; d\tilde{y}_i ,

where :math:`p_t` is the (unknown) true data-generating process. Since :math:`p_t` is unavailable, elpd
must be **estimated**. The naive estimate uses the observed data themselves:

.. math::

   \mathrm{lppd} = \sum_{i=1}^{n} \log \left( \frac{1}{S} \sum_{s=1}^{S}
                   p(y_i \mid \theta^{(s)}) \right) .

This is the **log pointwise predictive density**, and it is **optimistic**: the same data fitted the
model, so the score is inflated by exactly the amount the model overfit.

Correcting the optimism
-------------------------

Information criteria subtract an estimate of that overfitting. **WAIC** uses the posterior variance of
the pointwise log-likelihood as an **effective number of parameters** :math:`p_{\text{WAIC}}`:

.. math::

   \widehat{\mathrm{elpd}}_{\text{WAIC}} = \mathrm{lppd} - p_{\text{WAIC}},
   \qquad
   p_{\text{WAIC}} = \sum_{i=1}^{n} \mathrm{var}_{s}\bigl(\log p(y_i \mid \theta^{(s)})\bigr).

Unlike AIC and DIC — which count parameters, or use a point estimate — WAIC is **fully Bayesian**: it
averages over the posterior and works when the effective number of parameters is not obvious, as in
hierarchical models where partial pooling makes :math:`J` groups behave like far fewer.

.. code-block:: python

   import arviz as az
   az.waic(idata)        # elpd_waic, p_waic, se
   az.loo(idata)         # elpd_loo — usually preferred (next lesson)

Pointwise is the point
------------------------

Note that every quantity here is a **sum over observations**. That decomposition is what makes standard
errors available (the sd of the :math:`n` components times :math:`\sqrt{n}`), what lets you see **which
observations** a model predicts badly, and what makes leave-one-out cross-validation — the more direct
estimate of the same elpd — computable from a single fit.
"""

CONTENT["Model comparison based on predictive performance"] = r"""
Cross-validation, done cheaply
--------------------------------

WAIC corrects the optimism of in-sample scoring with an analytic penalty. **Leave-one-out
cross-validation** estimates the same elpd more directly: predict each observation from a posterior
fitted without it,

.. math::

   \mathrm{elpd}_{\text{loo}} = \sum_{i=1}^{n} \log p(y_i \mid y_{-i}) .

Done literally this means refitting :math:`n` times. The practical alternative reuses the **single**
posterior you already have, reweighting its draws by :math:`1/p(y_i \mid \theta^{(s)})` — importance
sampling — to approximate the leave-one-out posterior.

Pareto-smoothed importance sampling
-------------------------------------

Raw importance weights are unstable: a single influential observation gives one draw enormous weight.
**PSIS-LOO** (Vehtari, Gelman and Gabry) stabilises them by fitting a generalised Pareto distribution to
the largest weights and replacing them with smoothed order statistics. The fitted **shape parameter**
:math:`\hat{k}` is a free diagnostic, per observation:

* :math:`\hat{k} < 0.5` — the estimate is reliable;
* :math:`0.5 \le \hat{k} < 0.7` — acceptable, but the observation is influential;
* :math:`\hat{k} \ge 0.7` — the approximation may be **unreliable**; refit exactly for those points, or
  use :math:`k`-fold.

A high :math:`\hat{k}` is itself informative: it names an observation the model finds surprising.

.. code-block:: python

   import arviz as az
   loo_a, loo_b = az.loo(idata_a, pointwise=True), az.loo(idata_b, pointwise=True)
   az.compare({"simple": idata_a, "hierarchical": idata_b})   # elpd_diff, se_diff, weights
   (loo_a.pareto_k > 0.7).sum()                               # observations to worry about

Comparing honestly
--------------------

Report the **difference** in elpd together with the standard error of that difference — computed from
the pointwise components, which is far more precise than differencing two noisy totals. A rough rule:
if `elpd_diff` is not several times `se_diff`, the data do not distinguish the models. And the
comparison is **relative**: LOO ranks the candidates you supply and says nothing about whether the best
of them is any good. That question belongs to posterior predictive checking.

Two limits worth naming. LOO presupposes that predicting a **new observation** is the relevant task; for
time series or grouped data, leave-**future**-out or leave-one-**group**-out is the honest analogue. And
selecting a model by LOO from a large set reintroduces overfitting — to the selection criterion itself.
Where models are many and similar, **averaging** them (stacking) usually beats picking one.
"""

CONTENT["Model comparison using Bayes factors"] = r"""
Betting on whole models
-------------------------

If the candidate models are treated as hypotheses with prior probabilities, Bayes' rule applies to
**them** as well. The **Bayes factor** is the ratio of the two **marginal likelihoods** — the evidence
term that inference could ignore, now doing all the work:

.. math::

   \mathrm{BF}_{12} = \frac{p(y \mid M_1)}{p(y \mid M_2)},
   \qquad
   p(y \mid M_k) = \int p(y \mid \theta_k, M_k) \; p(\theta_k \mid M_k) \; d\theta_k .

It multiplies the prior odds into posterior odds. Because the marginal likelihood **averages** the
likelihood over the prior, it automatically penalises models that spread their prior mass over regions
the data reject — an intrinsic Occam's razor, with no explicit parameter count.

Three difficulties
--------------------

That elegance carries a heavy bill.

* **Improper priors make it undefined.** An improper prior has an arbitrary normalising constant, which
  does not cancel; the Bayes factor takes an arbitrary value.
* **Vague proper priors favour the simpler model** — the **Jeffreys–Lindley (Bartlett) paradox**.
  Widening a prior dilutes the alternative's marginal likelihood, so a diffuse prior on the alternative
  drives the Bayes factor toward the null **no matter what the data say**, and the effect does not
  vanish as :math:`n` grows.
* **It is hard to compute.** The marginal likelihood is exactly the normalising constant that MCMC is
  built to avoid, and naive estimators of it are notoriously bad.

The sensitivity is the real objection
---------------------------------------

Notice what makes this different from ordinary prior sensitivity. Changing :math:`\mathrm{Beta}(1,1)` to
:math:`\mathrm{Beta}(30,30)` may barely move the **posterior** for :math:`\theta` — but it can change the
Bayes factor substantially, because the evidence integrates the likelihood against the prior itself. A
quantity that is insensitive where inference is sensitive, and sensitive where inference is not, is a
poor guide.

.. code-block:: python

   # Bayes factors demand priors you would defend as *predictions*, not as regularisers.
   # In practice, prefer:
   import arviz as az
   az.compare({"m1": idata1, "m2": idata2})   # predictive comparison, no marginal likelihood

When to use it
----------------

Bayes factors are appropriate when the model space is genuinely **discrete and exhaustive** — one of
these hypotheses is true — and the priors are honest, informative statements you would stake a
prediction on. That describes some scientific hypothesis tests and few applied models. Gelman's
recommended route is to **bypass the choice**: check models predictively, compare them by elpd, and,
where they disagree, **expand** rather than select — the subject of the next lesson.
"""

CONTENT["Continuous model expansion"] = r"""
Don't choose — embed
----------------------

Model comparison presumes a **discrete** menu: :math:`M_1` or :math:`M_2`. But rival models are usually
special cases of a richer one, and the more informative move is to **embed** them in a continuous
family, fit it, and let the posterior for the extra parameter say how much of each the data want. The
choice becomes an **estimate**, and its uncertainty is reported rather than hidden.

The pattern
-------------

Introduce a parameter whose values recover the candidates as limits:

* Normal errors versus heavy tails: fit a :math:`t` with **unknown degrees of freedom** :math:`\nu`.
  :math:`\nu \to \infty` is the normal; a posterior for :math:`\nu` concentrated near 4 says the data
  want heavy tails, and by how much.
* Complete versus no pooling: the hierarchical model with unknown :math:`\tau` already **contains**
  both, at :math:`\tau = 0` and :math:`\tau \to \infty`.
* Fixed versus varying slopes: allow varying slopes with a variance parameter that can shrink to zero.
* Linear versus nonlinear: add a spline or GP term whose amplitude can vanish.

Each replaces a hypothesis test with a **continuous parameter and a posterior**.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # embeds normal (nu -> inf) and heavy-tailed alternatives in one model
       nu = pm.Gamma("nu", alpha=2, beta=0.1)      # weakly informative, mass over 2..50
       sigma = pm.HalfNormal("sigma", 1)
       pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y)
   # posterior for nu answers "how non-normal?" instead of "normal: yes/no?"

Why this is better
--------------------

Three reasons. It **propagates uncertainty**: selecting a model and then analysing as if it were true
ignores the uncertainty in the selection, and understates the final intervals. It avoids the
**discontinuity** of a decision rule that flips on an arbitrary threshold. And it turns a failed
posterior predictive check into a **research direction**: the check told you *which* feature was missed;
the expansion adds a parameter for exactly that feature and asks the data how big it is.

The discipline
----------------

Expansion is not licence to add everything. Each new parameter needs a **weakly informative prior**
(otherwise a barely-identified addition wanders), and the expanded model must itself be **checked** —
the loop from the first lesson of this stage. Nor is expansion free: it can make the geometry harder
(the funnel), so diagnostics matter more. But the direction of travel is right, and it is the workflow
that the remainder of this course follows: fit, check, find the misfit, **expand**, refit.
"""


MINDMAP.update({
    "Measures of predictive accuracy": [
        "Posterior predictive checking", "Model comparison based on predictive performance",
        "Model comparison using Bayes factors", "Bayesian interpretations of other statistical methods",
    ],
    "Model comparison based on predictive performance": [
        "Measures of predictive accuracy", "Model comparison using Bayes factors",
        "Continuous model expansion", "Graphical posterior predictive checks",
    ],
    "Model comparison using Bayes factors": [
        "Measures of predictive accuracy", "Model comparison based on predictive performance",
        "Noninformative Prior Distributions", "Continuous model expansion",
    ],
    "Continuous model expansion": [
        "Model comparison based on predictive performance",
        "Implicit assumptions and model expansion: an example",
        "The Place of Model Checking in Applied Bayesian Statistics",
        "Robust regression using t-distributed errors",
    ],
})


# ----------------------------------------------------------------------
# Part II / Stage 6 — Model Checking & Comparison  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Implicit assumptions and model expansion: an example"] = r"""
The assumptions you forgot you made
-------------------------------------

Every model states some assumptions **explicitly** — the likelihood, the prior — and smuggles in others
**implicitly**, in the very act of writing it down. Implicit assumptions are dangerous precisely because
they are invisible: nothing in the output flags them, and a posterior predictive check will not test a
feature the model was never asked about. Expansion is how you drag them into the open.

Inventory the silences
------------------------

Take the eight-schools model of Stage 5, an analysis that passed its checks:

.. math::

   y_j \mid \theta_j \sim \mathrm{N}(\theta_j, \sigma_j^2), \qquad
   \theta_j \mid \mu, \tau \sim \mathrm{N}(\mu, \tau^2) .

Four assumptions were never stated. That the school effects are **normally distributed** around
:math:`\mu` — no outlying school with a genuinely different programme. That the standard errors
:math:`\sigma_j` are **known exactly**, rather than estimated from modest samples. That the schools are
**exchangeable** — nothing about size, funding or curriculum predicts the effect. And that each school's
estimate is **unbiased** for its own effect, so no selective reporting occurred.

Expand to test
----------------

Each silent assumption becomes a parameter, and the posterior for that parameter tells you whether it
mattered. Replace the normal population with a :math:`t`:

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu  = pm.Normal("mu", 0, 5)
       tau = pm.HalfCauchy("tau", 5)
       nu  = pm.Gamma("nu", 2, 0.1)            # was: implicitly infinity (normal)
       eta = pm.StudentT("eta", nu=nu, mu=0, sigma=1, shape=8)
       theta = pm.Deterministic("theta", mu + tau * eta)
       pm.Normal("y", theta, sigma=sigma_j, observed=y)

If the posterior for :math:`\nu` piles up at large values, the normal assumption was harmless and you
have **earned** the right to it. If :math:`\nu` concentrates near 3, one school is an outlier, and the
original analysis over-shrank it. Similarly, letting :math:`\theta_j` depend on a school-level covariate
tests exchangeability; giving :math:`\sigma_j` its own uncertainty tests the known-variance assumption.

The general move
------------------

The pattern is: **name the assumption, embed it in a family, fit, inspect the posterior of the embedding
parameter.** This converts an untestable premise into an estimated quantity. Where the data are
informative the expansion settles the question; where they are not, the posterior for the new parameter
simply reproduces its prior — which is itself the answer: *the data cannot tell*, and the conclusion
rests on judgement rather than evidence.

Two honest limits
-------------------

Not every implicit assumption can be expanded away. The claim that :math:`\sigma_j` describes the right
sampling process, or that no results were suppressed before publication, lies **outside** the
likelihood — no parameter inside the model can detect it. And each expansion costs identifiability:
with :math:`J = 8`, :math:`\nu` will be weakly determined, so the check reports "no evidence against"
rather than "assumption verified". Model expansion widens what the data can speak to; it does not make
them omniscient.
"""


# ----------------------------------------------------------------------
# Part II / Stage 7 — Data Collection & Decisions
# ----------------------------------------------------------------------

CONTENT["Bayesian inference requires a model for data collection"] = r"""
The data did not arrive by magic
----------------------------------

Every analysis so far quietly assumed that :math:`y` is simply "the data". But some process **chose**
which units you observe: a survey sampled households, an experiment assigned treatments, a clinic
recorded only patients who returned. That process is part of the probability model, and ignoring it can
invalidate everything downstream.

Model the inclusion indicator
-------------------------------

Write :math:`y = (y_{\text{obs}}, y_{\text{mis}})` for the complete data — everything that *could* have
been observed — and let :math:`I` be the **inclusion indicator**, :math:`I_i = 1` when unit :math:`i` is
observed. The honest joint model covers both:

.. math::

   p(y, I \mid \theta, \phi) = p(y \mid \theta) \; p(I \mid y, \phi),

where :math:`\phi` parameterises the **data-collection mechanism**. What you actually condition on is
:math:`(y_{\text{obs}}, I)`, so the correct posterior integrates the missing values out:

.. math::

   p(\theta, \phi \mid y_{\text{obs}}, I) \;\propto\;
   p(\theta, \phi) \int p(y_{\text{obs}}, y_{\text{mis}} \mid \theta)
   \; p(I \mid y_{\text{obs}}, y_{\text{mis}}, \phi) \; d y_{\text{mis}} .

Analysing :math:`p(\theta \mid y_{\text{obs}})` alone — as if the observed data were the whole story —
is a **modelling assumption**, not a neutral default.

When it bites
---------------

The mechanism matters exactly when :math:`I` depends on values you did **not** observe. Truncate a
sample at a threshold and the sample mean is biased for the population mean, however large :math:`n`
grows. Let sicker patients drop out and the surviving cohort looks healthier. Publish only significant
results and the literature's effect sizes are inflated. In each case the likelihood you wrote is not the
likelihood of the data you have.

.. code-block:: python

   import numpy as np
   # a truncated sample: only y > 0 recorded. Naive mean is biased upward.
   rng = np.random.default_rng(0)
   y_full = rng.normal(0.0, 1.0, 100_000)
   y_obs = y_full[y_full > 0]
   y_full.mean(), y_obs.mean()          # ≈ 0.00  vs  ≈ 0.80 — the mechanism is not ignorable

The good news
---------------

Under identifiable conditions — the subject of the next lesson — the mechanism **factors out** and can
be ignored entirely, which is why the first seven stages were not wrong, merely conditional. Knowing
those conditions is what separates an analysis that may safely omit the design from one that must model
it. Randomisation, it will turn out, is precisely a device for **making the mechanism ignorable by
construction**.
"""

CONTENT["Data-collection models and ignorability"] = r"""
When can the design be ignored?
---------------------------------

The previous lesson wrote the joint model :math:`p(y \mid \theta) \, p(I \mid y, \phi)`. Ignorability
is the precise statement of when the second factor can be **dropped** — when the posterior obtained from
modelling :math:`y` alone equals the posterior obtained from modelling :math:`y` **and** :math:`I`.

Two conditions
----------------

Ignorability requires **both**:

* **Missing at random (MAR).** The inclusion mechanism does not depend on the values you failed to
  observe, given the ones you did:

  .. math::

     p(I \mid y_{\text{obs}}, y_{\text{mis}}, \phi) = p(I \mid y_{\text{obs}}, \phi).

* **Parameter distinctness.** The parameters :math:`\theta` of the data model and :math:`\phi` of the
  mechanism are **a priori independent**, :math:`p(\theta, \phi) = p(\theta) p(\phi)`.

Grant both, and the mechanism contributes a factor free of :math:`\theta` that cancels in the
normalisation. The likelihood you wanted to use is the one you may use.

The hierarchy of mechanisms
-----------------------------

* **MCAR** (missing completely at random) — :math:`I` is independent of the data altogether; a simple
  random sample. Strong, and rarely true.
* **MAR** — :math:`I` may depend on **observed** quantities: sampling more heavily in some strata, or
  non-response predicted by recorded covariates. Ignorable, *provided those quantities are in the
  model*.
* **MNAR** (missing not at random) — :math:`I` depends on the unobserved values themselves: people with
  high incomes decline to state their income. **Not ignorable**; the mechanism must be modelled, and the
  answer will depend on assumptions the data cannot check.

The practical corollary
-------------------------

MAR is a statement about a **model**, not about the world: a mechanism that is non-ignorable given no
covariates can become ignorable once the variables driving selection are **included as predictors**.
This is the single most useful fact in the chapter — it is why survey analyses must include the design
variables (strata, cluster, weights' determinants), and why omitting them silently converts an
ignorable design into a biased analysis.

.. code-block:: python

   import pymc as pm
   # Non-response depends on age and region (both recorded) -> MAR *if* they are in the model.
   with pm.Model():
       beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])   # X includes age, region, strata
       pm.Bernoulli("y", logit_p=X @ beta, observed=y_obs)
       # omit those columns and the same design becomes non-ignorable

Randomisation, then, is not a ritual: assigning treatment by a coin flip makes :math:`I` depend on
nothing but a known probability, guaranteeing ignorability **by design** rather than by assumption.
"""

CONTENT["Sample surveys"] = r"""
Inference for a finite population
-----------------------------------

Survey inference has an unusual target. You do not want a parameter of an infinite superpopulation; you
want a **finite-population quantity** — the mean income of *these* :math:`N` households, of whom you
sampled :math:`n`. The Bayesian formulation is disarmingly direct: the unsampled values
:math:`y_{\text{mis}}` are simply **unknowns**, and the population mean is a function of them.

.. math::

   \bar{Y} = \frac{1}{N}\left( \sum_{i \in \text{sample}} y_i
             + \sum_{i \notin \text{sample}} y_i \right).

Draw :math:`y_{\text{mis}}` from its posterior predictive distribution, compute :math:`\bar{Y}` for each
draw, and you have the posterior for the population mean — uncertainty about the unsampled units
included, and the :math:`n/N` finite-population correction appearing automatically rather than being
bolted on.

Designs, and what they demand
-------------------------------

Real surveys are not simple random samples. **Stratified** designs sample strata at different rates;
**cluster** designs sample groups then units within them; **unequal probability** designs oversample
rare populations. Under each, :math:`I` depends on design variables — and by the ignorability lesson the
design is ignorable **exactly when those variables are in the model**. So:

* stratification → include stratum as a predictor (naturally, **hierarchically**);
* clustering → a group-level random effect, which is a hierarchical model;
* unequal probabilities → model whatever determined them.

Design-based practice handles this with **weights**; the model-based route puts the same information in
as **predictors**, and gains partial pooling for small strata for free.

Multilevel regression and poststratification
----------------------------------------------

The modern synthesis, **MRP**: fit a hierarchical regression of the outcome on demographic and
geographic cells, then **poststratify** — reweight the predicted cell means by the known population cell
counts from a census.

.. code-block:: python

   import numpy as np, pymc as pm
   with pm.Model():                              # 1. multilevel regression over cells
       a_state = pm.Normal("a_state", 0, pm.HalfNormal("s_state", 1), shape=n_states)
       a_age   = pm.Normal("a_age",   0, pm.HalfNormal("s_age", 1),   shape=n_ages)
       pm.Bernoulli("y", logit_p=a_state[state] + a_age[age], observed=y)
   # 2. poststratify: weight predicted cell means by census population counts
   theta_pop = (cell_pred * N_cells).sum() / N_cells.sum()

MRP produces stable estimates for small subgroups (a state with twelve respondents borrows from the
others) and corrects unrepresentative samples — the reason it can extract state-level estimates from
national polls, and even from famously non-representative online panels.

The limits
------------

MRP corrects for what it **models**. If non-response depends on something unmeasured — political
enthusiasm, say, not captured by age, race, education and region — the design remains **MNAR** and no
amount of poststratification fixes it. And poststratification needs **population cell counts**, which
constrains which variables you may adjust for. The honest summary: surveys are a missing-data problem,
and their difficulty is exactly the difficulty of knowing why people did not answer.
"""


MINDMAP.update({
    "Implicit assumptions and model expansion: an example": [
        "Continuous model expansion", "Example: parallel experiments in eight schools",
        "Model checking for the educational testing example",
        "Robust inference for the eight schools",
    ],
    "Bayesian inference requires a model for data collection": [
        "Data-collection models and ignorability", "Censoring and truncation",
        "Notation", "Sample surveys",
    ],
    "Data-collection models and ignorability": [
        "Bayesian inference requires a model for data collection",
        "Sensitivity and the role of randomization", "Observational studies",
        "Multiple imputation",
    ],
    "Sample surveys": [
        "Data-collection models and ignorability", "Designed experiments",
        "State-level opinons from national polls",
        "Exchangeability and hierarchical models",
    ],
})


# ----------------------------------------------------------------------
# Part II / Stage 7 — Data Collection & Decisions (cont.)
# ----------------------------------------------------------------------

CONTENT["Designed experiments"] = r"""
Choosing the mechanism yourself
---------------------------------

In a survey the inclusion mechanism is something you must **discover**. In an experiment you **choose**
it. That is the whole advantage: the assignment mechanism :math:`p(W \mid y, \phi)` — which units get
treatment, :math:`W_i \in \{0, 1\}` — is **known, controlled, and probabilistic** by construction, so
ignorability is guaranteed rather than assumed.

Causal inference as missing data
----------------------------------

Each unit has two **potential outcomes**: :math:`y_i(1)` under treatment and :math:`y_i(0)` under
control. You observe exactly one. The causal effect :math:`y_i(1) - y_i(0)` is therefore unobservable
for every individual — one of its two terms is always **missing**. Causal inference *is* a missing-data
problem, and the assignment mechanism *is* the missingness mechanism:

.. math::

   \Pr\bigl(W \mid y(0), y(1), X, \phi\bigr) .

Complete randomisation makes this depend on **nothing** (the parallel of MCAR). Stratified or blocked
randomisation makes it depend only on **covariates** :math:`X` (the parallel of MAR, ignorable given
:math:`X`). Either way the mechanism factors out — *provided* the design variables are in the model.

The Bayesian analysis
-----------------------

Once ignorability holds, fitting is unremarkable: model the outcome, include the design, and read the
causal effect off the posterior. Blocking becomes a hierarchical term; the estimand is a function of
draws.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       a_block = pm.Normal("a_block", 0, pm.HalfNormal("s", 1), shape=n_blocks)  # blocking
       tau  = pm.Normal("tau", 0, 1)                    # treatment effect
       mu   = a_block[block] + tau * W
       pm.Normal("y", mu, pm.HalfNormal("sigma", 1), observed=y)
       idata = pm.sample()
   # P(effect > 0 | data) is a count over draws; the design justified the model

Design earns you the likelihood
---------------------------------

Two consequences worth stating. **Blocking on a covariate** must be matched by **including it in the
model**: randomising within blocks and then ignoring blocks discards the design's benefit and can
mis-state uncertainty. And randomisation buys **ignorability, not precision** — a badly designed
randomised experiment gives an unbiased answer with an interval too wide to act on. The next lesson
asks exactly what randomisation does and does not protect against.
"""

CONTENT["Sensitivity and the role of randomization"] = r"""
What randomisation actually does
----------------------------------

Randomisation is often described as "balancing the covariates". That is a consequence, not the
mechanism. What randomisation does, precisely, is make the assignment mechanism **known and independent
of the potential outcomes**, so that :math:`p(W \mid y(0), y(1), \phi) = p(W \mid \phi)`. Ignorability
then follows **by construction**, and the analyst need not enumerate the confounders — including the
ones nobody thought of.

Not a guarantee of balance
----------------------------

In any single randomised experiment, covariates may be **imbalanced** by chance: the treatment group
happens to be older. Randomisation does not prevent this; it makes the *probability* of each imbalance
known. The Bayesian response is neither to re-randomise nor to ignore it, but to **condition on the
observed covariates** — include age in the model. Adjustment is legitimate because it is *conditioning*,
not because it repairs a broken randomisation.

Sensitivity analysis, where ignorability is assumed
-----------------------------------------------------

Where the design does **not** guarantee ignorability, the conclusions rest on an assumption the data
cannot verify. The honest reply is **sensitivity analysis**: posit an unmeasured confounder of a given
strength, refit, and report how strong it would have to be to overturn the finding.

.. code-block:: python

   import numpy as np, pymc as pm
   # How strong must an unmeasured confounder U be to erase the effect?
   for gamma in [0.0, 0.25, 0.5, 1.0]:                # U's effect on the outcome
       with pm.Model():
           U   = pm.Normal("U", 0, 1, shape=n)        # unobserved, correlated with W
           tau = pm.Normal("tau", 0, 1)
           mu  = alpha + tau * W + gamma * U
           pm.Normal("y", mu, sigma, observed=y)
           # report the posterior for tau at each assumed gamma

If the effect survives confounders far stronger than any measured covariate, the conclusion is robust.
If a modest one erases it, say so.

The limits
------------

Randomisation protects against **confounding**, and against nothing else. It does not fix
**non-compliance** (assignment is not receipt), **attrition** (drop-out may depend on outcomes,
reintroducing MNAR), **interference** (one unit's treatment affecting another's outcome, violating the
stable-unit assumption), or **generalisation** to a population the units were not sampled from. Each is
a separate modelling problem. Randomisation is the cheapest way to buy ignorability of *assignment* — a
genuine and rare gift — but the phrase "randomised, therefore unbiased" quietly assumes that nothing
else went wrong.
"""

CONTENT["Observational studies"] = r"""
When you cannot randomise
---------------------------

Most questions cannot be settled by an experiment: you cannot randomise people to smoke, to be poor, or
to live near a highway. In an **observational study** the treatment was chosen — by the units, by
doctors, by circumstance — and the assignment mechanism is **unknown**. Everything then depends on
whether ignorability can be recovered by conditioning.

Ignorability conditional on covariates
----------------------------------------

The working assumption is that treatment is as good as random **given the observed covariates**:

.. math::

   \Pr\bigl(W \mid y(0), y(1), X\bigr) = \Pr(W \mid X)
   \qquad \text{(``no unmeasured confounding'')}.

This is the MAR condition transposed to causal inference. Its opposite — assignment depending on the
unobserved potential outcomes — is the MNAR case, and it is **not testable**: two datasets identical in
:math:`(y_{\text{obs}}, W, X)` are consistent with both. No amount of data settles it; only knowledge of
*why* treatment was assigned can.

Two routes to conditioning
----------------------------

Given the assumption, the estimand is recovered by conditioning on :math:`X` — either by **modelling the
outcome** :math:`p(y \mid W, X)` and averaging its predictions over the covariate distribution, or by
modelling the **propensity** :math:`e(x_i) = \Pr(W_i = 1 \mid x_i)` and comparing units of like
propensity.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])
       tau  = pm.Normal("tau", 0, 1)
       pm.Normal("y", X @ beta + tau * W, pm.HalfNormal("s", 1), observed=y)
       idata = pm.sample()
   # ATE = posterior mean of predictions with W=1 minus W=0, averaged over the sample

Bayesian practice tends to favour the outcome model — it uses the posterior directly, propagates
uncertainty, and permits hierarchical structure — while treating the propensity as a **diagnostic**:
where the treated and untreated propensity distributions barely overlap, the comparison is an
extrapolation, and no model can rescue it.

Which covariates?
-------------------

Include the variables that **affect assignment**, and the pre-treatment variables that affect the
outcome. But conditioning is not universally benign: adjusting for a variable on the causal pathway
*between* treatment and outcome removes part of the effect you want, and adjusting for a **collider**
can create bias where none existed. Covariate choice requires **causal reasoning about the mechanism**,
not a search over predictors — and the final report should say, plainly, how strong an unmeasured
confounder would need to be to change the conclusion.
"""

CONTENT["Censoring and truncation"] = r"""
Two ways data go missing
--------------------------

Both words describe incomplete observation, and confusing them yields a wrong likelihood. The
distinction is what you know about the units you did not fully see.

* **Censoring**: the unit **is** in your dataset, but its value is known only to lie in a range. A
  patient still alive at the end of a trial has survival time :math:`> c`. A scale reading "over 100 kg".
* **Truncation**: the unit is **absent entirely**. Only patients who survived long enough to enrol
  appear; light sources fainter than the telescope's limit are never recorded. You do not know how many
  you missed.

The likelihoods differ
------------------------

For **censoring**, the censored observation contributes the probability of the event it represents —
the survival function, not the density:

.. math::

   p(y \mid \theta) = \prod_{i \, \text{obs}} f(y_i \mid \theta)
                      \prod_{i \, \text{cens}} \bigl[1 - F(c_i \mid \theta)\bigr] .

For **truncation**, every observed value must be **renormalised** by the probability of being observed
at all, because the sample space itself is restricted:

.. math::

   p(y \mid \theta) = \prod_{i} \frac{f(y_i \mid \theta)}{\Pr(\text{observed} \mid \theta)}
                    = \prod_{i} \frac{f(y_i \mid \theta)}{1 - F(c \mid \theta)} .

Ignore the denominator and you fit a model to a biased sample — the truncated-normal demonstration from
the data-collection lesson, where the sample mean converged confidently to the wrong number.

In code
---------

Both are one line in a modern PPL, which is precisely why the distinction must be made **before**
coding:

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu, sigma = pm.Normal("mu", 0, 10), pm.HalfNormal("sigma", 5)
       # censoring: value known only to exceed c (unit IS in the data)
       pm.Censored("y_cens", pm.Normal.dist(mu, sigma), lower=None, upper=c, observed=y)
       # truncation: values beyond c never enter the sample at all
       pm.Truncated("y_trunc", pm.Normal.dist(mu, sigma), lower=c, observed=y)

Censoring is ignorable, truncation less so
--------------------------------------------

Where the censoring **time** is fixed in advance, or depends only on observed quantities, the mechanism
is ignorable in the technical sense — the censored contributions must appear in the likelihood, but no
separate model for :math:`\phi` is needed. When censoring depends on the **unobserved** value itself —
patients withdrawing *because* they are deteriorating — the mechanism is **MNAR**, and the analysis
must model why they left. That is not a computational difficulty but an **identification** one: the
data are silent, and the answer will move with the assumption. Report the sensitivity.
"""


MINDMAP.update({
    "Designed experiments": [
        "Data-collection models and ignorability", "Sensitivity and the role of randomization",
        "Observational studies", "Regression for causal inference: incumbency and voting",
    ],
    "Sensitivity and the role of randomization": [
        "Designed experiments", "Observational studies",
        "Data-collection models and ignorability", "Aspects of robustness",
    ],
    "Observational studies": [
        "Sensitivity and the role of randomization", "Designed experiments",
        "Regression for causal inference: incumbency and voting",
        "Assembling the matrix of explanatory variables",
    ],
    "Censoring and truncation": [
        "Bayesian inference requires a model for data collection",
        "Data-collection models and ignorability", "Missing values with counted data",
        "Multiple imputation",
    ],
})


# ----------------------------------------------------------------------
# Part II / Stage 7 — Data Collection & Decisions (decision analysis)
# ----------------------------------------------------------------------

CONTENT["Bayesian decision theory in di\ufb00erent contexts"] = r"""
From inference to action
--------------------------

A posterior is not a decision. To act you need one more ingredient: a **utility function** stating what
outcomes are worth. Bayesian decision theory then supplies the rule — among the available actions,
choose the one whose utility is greatest **on average over the posterior**.

The rule
----------

Let :math:`a` be an action, :math:`\theta` the unknown state, and :math:`U(a, \theta)` the utility of
taking :math:`a` when the truth is :math:`\theta`. The optimal action maximises **expected utility**:

.. math::

   a^{*} = \arg\max_{a} \; \mathrm{E}_{\theta \mid y}\bigl[U(a, \theta)\bigr]
         = \arg\max_{a} \int U(a, \theta) \; p(\theta \mid y) \; d\theta .

With posterior draws this is a one-line computation — the expectation becomes an average, and the
optimisation a search over the (usually few) candidate actions.

.. code-block:: python

   import numpy as np
   theta = idata.posterior["theta"].values.ravel()      # posterior draws
   actions = np.linspace(0, 1, 101)                     # candidate actions
   EU = [np.mean(utility(a, theta)) for a in actions]   # expected utility per action
   best = actions[int(np.argmax(EU))]

Note what the posterior does here: it **weights** each possible truth by its plausibility. The full
distribution matters, not just its mean — because utilities are usually **nonlinear**, and
:math:`\mathrm{E}[U(a, \theta)] \ne U(a, \mathrm{E}[\theta])`.

Why estimation is not decision
--------------------------------

Familiar Bayesian summaries are decisions in disguise, each optimal under a particular loss (negative
utility). Squared-error loss gives the **posterior mean**; absolute-error loss gives the **median**;
0–1 loss gives the **mode**. So the choice of point estimate is not a matter of taste but of the cost
of being wrong — and when those costs are **asymmetric** (a flood barrier too low versus too high) the
optimal action can lie far out in a tail, nowhere near any conventional estimate.

Contexts differ
-----------------

Three settings recur through this stage, and their utilities differ in kind. A **one-shot** choice: pick
:math:`a` once, as with a survey incentive. A **sequential** problem: act, observe, act again, where an
early action buys information for later ones. And a **hierarchical** problem, where decisions are made
for many units at once and shrinkage governs each. The final lesson adds the sharpest distinction of
all — **whose** utility, the individual's or the institution's, since the two can rationally disagree.
"""

CONTENT["Using regression predictions: survey incentives"] = r"""
Should you pay respondents?
-----------------------------

A survey organisation must choose whether to offer respondents an incentive, of what size, in what
form, and when. The stakes are concrete: incentives cost money but raise **response rates**, and a
higher response rate reduces both the number of calls needed and the nonresponse bias. This is a
cost–benefit problem, and Gelman, Stevens and Chan turned it into a worked Bayesian decision analysis.

Meta-analysis feeding a decision
----------------------------------

There is no single experiment that answers the question, so the analysis proceeds in two stages. First,
a **hierarchical meta-analysis** of many surveys' incentive experiments estimates the effect of incentive
value, timing and mode on response rate. The design variables matter:

* **prepaid** incentives (sent with the request) versus **postpaid** (paid on completion);
* the **value** in dollars, whose effect need not be linear;
* the survey's **burden** and mode.

The regression's output is a posterior for the **expected increase in response rate** as a function of
the incentive — with uncertainty, and with partial pooling across the studies, since the surveys are
exchangeable but not identical.

From response rate to utility
-------------------------------

The second stage converts that posterior into money. The utility of an incentive scheme is the
**net cost per respondent**, combining the incentive paid to everyone contacted, the interviewer time
saved by fewer callbacks, and the value placed on a marginal completed interview:

.. code-block:: python

   import numpy as np
   # posterior draws of the incentive's effect on response rate, from the meta-analysis
   d_rate = idata.posterior["beta_incentive"].values.ravel()

   def net_cost_per_respondent(incentive, d_rate, prepaid, base_rate=0.30,
                               call_cost=1.25, calls_per_contact=8):
       rate = base_rate + d_rate * incentive
       # prepaid: paid to everyone contacted; postpaid: paid only to respondents
       paid = incentive if prepaid else incentive / rate
       interviewing = call_cost * calls_per_contact / rate
       return paid + interviewing

   for inc in [0, 5, 10, 20]:
       c = net_cost_per_respondent(inc, d_rate, prepaid=True)
       print(inc, c.mean(), np.percentile(c, [2.5, 97.5]))   # posterior cost, with uncertainty

The lessons
-------------

Three, and they generalise. **The decision needs the whole posterior**, because cost is a nonlinear
function of the response rate and averaging the rate first would give the wrong answer. **Prepaid and
postpaid differ structurally**, not just in magnitude: a prepaid incentive is paid to *everyone
contacted*, a postpaid one only to *respondents*, so their cost curves diverge as the response rate
falls. And the analysis is honest about what it optimises — small incentives typically repay themselves,
but the recommendation depends on the dollar value assigned to a completed interview, which is a
**judgement**, stated openly rather than buried.
"""

CONTENT["Multistage decision making: medical screening"] = r"""
Decisions that buy information
--------------------------------

Screening is not one decision but a **sequence**. Test, and if the test is positive, decide whether to
run a second, more invasive test; then decide whether to treat. Each stage costs something and yields
**information** that improves the next choice. The value of a test lies not in the test but in the
**better decisions it enables**.

Working backwards
-------------------

The solution method is **backward induction**. At the final stage, the treatment decision is a plain
expected-utility maximisation given whatever has been learned. Step back: the value of *being* at that
stage is the expected utility of acting optimally there. So the earlier decision — whether to test —
compares the cost of testing against the improvement in the expected utility it produces:

.. math::

   \mathrm{EVSI} = \mathrm{E}_{y_{\text{new}}}\Bigl[\max_{a} \,
                   \mathrm{E}_{\theta \mid y, y_{\text{new}}} \, U(a, \theta) \Bigr]
                 - \max_{a} \, \mathrm{E}_{\theta \mid y} \, U(a, \theta) .

That quantity — the **expected value of sample information** — is the most one should ever pay for the
test. It is necessarily non-negative: information cannot hurt a decision-maker who is free to ignore it.

The screening structure
-------------------------

Screening makes the interplay of prevalence and accuracy vivid. Bayes' rule converts a positive test
into a posterior probability of disease, and when **prevalence is low**, even an accurate test leaves
that probability modest — the record-linkage lesson again, with lives at stake.

.. code-block:: python

   import numpy as np
   prev, sens, spec = 0.004, 0.95, 0.95            # rare disease, good test
   ppv = prev * sens / (prev * sens + (1 - prev) * (1 - spec))
   ppv                                              # ≈ 0.071 — most positives are false

   # stage 2: treat only if expected utility of treating exceeds not treating
   U_treat    = ppv * U_treat_sick + (1 - ppv) * U_treat_well     # side effects if well
   U_no_treat = ppv * U_none_sick  + (1 - ppv) * U_none_well
   act = "treat" if U_treat > U_no_treat else "wait"

Where it gets hard
--------------------

Three honest complications. Utilities must be **commensurable** — a life-year, a false-positive's
anxiety and a dollar placed on one scale, an ethical act, not a technical one. The state space grows
**combinatorially** with stages, so realistic problems are solved by simulation rather than by exact
backward induction. And the analysis assumes the model for test performance is right; a
**miscalibrated** sensitivity propagates through every stage. Screening policy is where Bayesian
decision theory is at its most useful and its most contested.
"""

CONTENT["Hierarchical decision analysis for home radon"] = r"""
A decision for every county
-----------------------------

Radon is a radioactive soil gas that accumulates indoors and causes lung cancer. A homeowner may
**measure** their house (cheap, noisy) and, depending on the reading, **remediate** (expensive,
effective). The recommendation should differ by county — some have far higher radon — but many counties
have only a handful of measurements. This is the cancer-rate problem, now attached to a decision.

Hierarchical prediction first
-------------------------------

Model log radon concentration hierarchically: houses within counties, counties drawn from a population,
with a county-level predictor (soil uranium) explaining part of the variation. Partial pooling gives
every county a stable predictive distribution, and a data-poor county borrows from the rest:

.. math::

   \log y_{ij} \sim \mathrm{N}(\alpha_j + \beta \, x_{ij}, \; \sigma_y^2), \qquad
   \alpha_j \sim \mathrm{N}(\gamma_0 + \gamma_1 u_j, \; \sigma_\alpha^2),

with :math:`x_{ij}` the basement indicator and :math:`u_j` county uranium.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       g0, g1 = pm.Normal("g0", 0, 5), pm.Normal("g1", 0, 5)
       s_a = pm.HalfNormal("s_a", 1)
       a = pm.Normal("a", g0 + g1 * u, s_a, shape=n_counties)     # county intercepts
       b = pm.Normal("b", 0, 1)
       pm.Normal("y", a[county] + b * basement, pm.HalfNormal("s_y", 1), observed=log_radon)

Then the decision
-------------------

Remediation costs a known amount; radon exposure costs expected life-years, monetised. For each house
the actions are *do nothing*, *measure then decide*, or *remediate immediately*. Expected utility is
computed **over the posterior predictive distribution of that house's radon level** — which is exactly
what the hierarchical model supplies, uncertainty included.

The result is a **decision rule that varies by county**: in high-radon counties, remediate without
measuring; in low-radon counties, do nothing; in the intermediate band — where the predictive
distribution straddles the action threshold — **measure first**, because there the measurement changes
what you would do. That middle region is precisely where the expected value of information is positive.

Why the hierarchy earns its keep
----------------------------------

Use each county's raw mean and the small counties give absurd recommendations, exactly as raw cancer
rates gave absurd rankings. Pool completely and every county gets the same advice, wasting the real
variation. Partial pooling produces recommendations that are **stable where data are thin and
responsive where data are plentiful** — and because the decision uses the **predictive** distribution for
an individual house, it correctly accounts for within-county spread, not merely uncertainty about the
county mean. The caveat is the usual one: the answer depends on the monetary value placed on a life-year,
and that number should be stated, not smuggled.
"""


MINDMAP.update({
    "Bayesian decision theory in di\ufb00erent contexts": [
        "Using regression predictions: survey incentives",
        "Multistage decision making: medical screening",
        "Summarizing Posterior Inference", "Personal vs. institutional decision analysis",
    ],
    "Using regression predictions: survey incentives": [
        "Bayesian decision theory in di\ufb00erent contexts",
        "Hierarchical modeling applied to a meta-analysis",
        "Sample surveys", "Conditional modeling",
    ],
    "Multistage decision making: medical screening": [
        "Bayesian decision theory in di\ufb00erent contexts",
        "Example \u2014 Calibration for Record Linkage",
        "Hierarchical decision analysis for home radon",
        "Personal vs. institutional decision analysis",
    ],
    "Hierarchical decision analysis for home radon": [
        "Bayesian decision theory in di\ufb00erent contexts",
        "Exchangeability and hierarchical models",
        "Informative Prior Distribution for Cancer Rates",
        "Varying intercepts and slopes",
    ],
})


# ----------------------------------------------------------------------
# Part II / Stage 7 — Data Collection & Decisions  [completes Part II]
# ----------------------------------------------------------------------

CONTENT["Personal vs. institutional decision analysis"] = r"""
Whose utility?
----------------

Expected-utility maximisation presumes a utility function. But **whose**? A homeowner deciding whether
to remediate radon and a public-health agency deciding what to recommend nationally face the same
posterior and reach **different optimal actions** — not because one is irrational, but because their
utilities and their information genuinely differ.

Three ways they diverge
-------------------------

* **The value of a risk reduction varies between people.** The dollar amount an individual will pay to
  remove a given increment of lung-cancer risk is not universal; it varies with wealth, age, risk
  aversion and belief. An institution must adopt a single figure; the individual has their own.
* **Individuals condition on more.** A homeowner knows their basement, their habits, whether they smoke
  (radon risk is far higher for smokers). Their posterior for their **own** exposure is sharper than the
  agency's posterior for a house drawn from their county.
* **The decision spaces differ.** The individual chooses an action; the institution chooses a **rule**,
  applied to millions of houses. A rule optimal in aggregate is not optimal for most of the individuals
  it binds.

Aggregating individual decisions
----------------------------------

The subtle point is that a policy is not evaluated by its expected utility for a typical house, but by
the **aggregate consequences of the individual decisions it induces**. Set a recommendation threshold
and some households will remediate who need not, and others will not measure who should. The
distribution of outcomes across the population is the object of interest, and it must be simulated:

.. code-block:: python

   import numpy as np
   remediation_cost = 2000.0                               # dollars per house
   # posterior predictive radon for each of many houses, from the hierarchical model
   radon = ppc["radon"]                                    # (S draws, n houses)
   for threshold in [2.0, 4.0, 8.0]:                       # pCi/L action level
       remediate = radon.mean(axis=0) > threshold          # rule applied per house
       cost = remediate.sum() * remediation_cost
       exposure = radon[:, ~remediate].mean()              # residual risk borne by the rest
       print(threshold, cost, exposure)                    # the trade-off a policy makes

Where the two must disagree
-----------------------------

An institution's recommendation is rationally **more conservative** in some directions (it bears
reputational and legal cost for under-warning) and **less** in others (it cannot price an individual's
risk aversion). It also faces genuinely collective considerations — equity across households, the cost
of a programme, the credibility of future advice — which never appear in a personal utility.

The Bayesian contribution is not to dissolve this conflict but to **make it explicit**. The posterior is
shared; the utilities are not. When a personal and an institutional analysis disagree, the disagreement
should be traceable to a **named** difference in utility or information — and that is a far healthier
disagreement than one hidden inside a threshold nobody can justify.
"""


# ----------------------------------------------------------------------
# Part III / Stage 8 — Simulation Basics
# ----------------------------------------------------------------------

CONTENT["Numerical integration"] = r"""
Every Bayesian answer is an integral
--------------------------------------

Posterior expectations, marginal distributions, predictive densities, the evidence: all are integrals
of the form :math:`\int h(\theta) \, p(\theta \mid y) \, d\theta`. Part I evaded them with conjugacy;
the bioassay showed how quickly that fails. Part III is about computing them when no formula exists.

Deterministic quadrature
--------------------------

The classical approach evaluates the integrand on a **grid** and sums with weights:

.. math::

   \int h(\theta) \, p(\theta \mid y) \, d\theta \;\approx\;
   \sum_{k=1}^{K} w_k \, h(\theta_k) \, p(\theta_k \mid y) .

Simple rules (trapezoid, Simpson) use equally spaced points; **Gaussian quadrature** places them
cleverly and achieves high accuracy with few evaluations. In **one or two dimensions** this is superb —
it is exactly what the bioassay grid did, and what the conjugate hierarchical model did for
:math:`(\alpha, \beta)`.

.. code-block:: python

   import numpy as np
   from scipy import integrate
   grid = np.linspace(-5, 10, 400)
   dens = np.exp(log_posterior(grid) - log_posterior(grid).max())
   dens /= integrate.trapezoid(dens, grid)              # normalise
   integrate.trapezoid(grid * dens, grid)               # E[theta | y]

The wall
----------

Quadrature dies of **dimensionality**. A grid of :math:`K` points per dimension costs :math:`K^d`
evaluations: 100 points in 10 dimensions is :math:`10^{20}` — impossible. Worse, in high dimensions
almost all of that grid lies where the posterior has **no mass**. The posterior of a modern model
concentrates in a thin, curved shell whose position you do not know in advance; enumerating the space is
hopeless.

Monte Carlo turns the problem around
--------------------------------------

Instead of choosing points and weighting by density, **draw points from the density** and weight
equally:

.. math::

   \int h(\theta) \, p(\theta \mid y) \, d\theta \;\approx\;
   \frac{1}{S} \sum_{s=1}^{S} h\bigl(\theta^{(s)}\bigr), \qquad
   \theta^{(s)} \sim p(\theta \mid y).

The error shrinks like :math:`1/\sqrt{S}` — **regardless of dimension**. That independence from
:math:`d` is the single fact on which all of modern Bayesian computation rests. The catch, of course, is
the premise: how do you draw from a distribution you can only evaluate up to a constant? The remaining
lessons of Part III are answers to that question.
"""

CONTENT["Distributional approximations"] = r"""
Replace the posterior with something tractable
------------------------------------------------

Before sampling, consider **approximating**. If a simple distribution stands in for the posterior, every
integral becomes analytic or trivially simulated. The approximation is never exact, but it is fast, and
it is often the right first move — and, as it turns out, a component of the sophisticated methods later.

The normal approximation, revisited
-------------------------------------

The workhorse is the **Laplace** approximation from Stage 4: a normal centred at the posterior mode with
covariance the inverse observed information. Two refinements matter in practice.

**Work on a transformed scale.** A posterior for :math:`\sigma > 0` is skewed and bounded; the posterior
for :math:`\log \sigma` is far closer to normal. Approximate there, then transform the draws back — the
Jacobian handles itself, because functions of draws are draws.

**Use a** :math:`t` **instead of a normal** when the approximation will serve as a proposal or
importance-sampling density: heavier tails guarantee the approximation **covers** the target, which
matters enormously for the stability of the weights.

.. code-block:: python

   import numpy as np
   from scipy import optimize, stats

   fit = optimize.minimize(lambda t: -log_post(t), x0, method="BFGS")
   mode, cov = fit.x, fit.hess_inv
   approx = stats.multivariate_normal(mode, cov)         # Laplace
   # heavier-tailed variant for use as a proposal:
   draws = mode + stats.multivariate_t(shape=cov, df=4).rvs(1000)

Mixtures, and the modern descendants
--------------------------------------

A single normal cannot represent a **multimodal** posterior. Fitting a **mixture** of normals — one
component per mode found by repeated optimisation from dispersed starting points — extends the reach at
modest cost. Push the idea further and you arrive at the methods of Stage 10: **variational inference**,
which fits the best member of a chosen family by minimising a divergence, and **expectation propagation**,
which matches moments locally.

Honest uses
-------------

An approximation is trustworthy when the posterior is smooth, unimodal and not near a boundary — which
large-sample theory says becomes true as :math:`n` grows. It is untrustworthy exactly where Bayesian
methods are most valuable: small samples, hierarchical variance parameters, weakly identified models.
So use approximations for **starting values**, for **proposal distributions**, for a **quick first
look**, and for problems too large to sample — but check them against MCMC where you can, and report
which you used.
"""

CONTENT["Direct simulation and rejection sampling"] = r"""
When you can draw directly
----------------------------

Some posteriors can be sampled **exactly**, with no iteration. Conjugate models are the obvious case —
`stats.beta(a + y, b + n - y).rvs()` is an exact posterior draw. So are models that **factor**: the
normal model with unknown variance draws :math:`\sigma^2` from its marginal, then :math:`\mu` from its
conditional, giving exact joint draws. Direct simulation is the gold standard: independent draws, no
convergence to diagnose.

Rejection sampling
--------------------

When direct draws are unavailable but the **unnormalised** density :math:`q(\theta) \propto p(\theta
\mid y)` can be evaluated, rejection sampling manufactures exact draws from a proposal
:math:`g(\theta)` you *can* sample. The requirement is an **envelope**: a constant :math:`M` with
:math:`q(\theta) \le M \, g(\theta)` everywhere.

The algorithm is three lines. Draw :math:`\theta^{*} \sim g`; draw :math:`u \sim \mathrm{Uniform}(0,1)`;
**accept** :math:`\theta^{*}` if

.. math::

   u \;\le\; \frac{q(\theta^{*})}{M \, g(\theta^{*})} .

Accepted draws are **exact** samples from the target — no approximation, no burn-in. The acceptance
probability is :math:`1/M` (for normalised :math:`q`), so :math:`M` measures the waste.

.. code-block:: python

   import numpy as np
   from scipy import stats

   g = stats.t(df=4, loc=mode, scale=scale)             # heavy-tailed proposal, must cover target
   M = np.exp((log_q(grid) - g.logpdf(grid)).max())     # envelope constant, found numerically
   out = []
   while len(out) < 1000:
       th = g.rvs()
       if np.random.rand() <= np.exp(log_q(th) - np.log(M) - g.logpdf(th)):
           out.append(th)
   accept_rate = 1000 / n_proposed                       # ≈ 1/M

Two failure modes
-------------------

If the envelope condition is **violated** anywhere — a proposal with lighter tails than the target — the
draws are silently **wrong**, not merely inefficient. Hence the standing advice to use a heavy-tailed
proposal (a :math:`t`, not a normal), and to find :math:`M` by maximising the log-ratio rather than
guessing.

And even a valid envelope becomes useless in high dimensions: the acceptance rate falls **exponentially**
with :math:`d`, because a proposal that is a decent match in each coordinate is a poor match in all of
them at once. Rejection sampling survives as a component — for univariate draws, for truncated
distributions (reject anything outside the support) — but not as a general engine. That role belongs to
**MCMC**, which abandons independent draws in exchange for scaling.
"""


MINDMAP.update({
    "Personal vs. institutional decision analysis": [
        "Bayesian decision theory in di\ufb00erent contexts",
        "Hierarchical decision analysis for home radon",
        "Multistage decision making: medical screening",
        "Probability as a Measure of Uncertainty",
    ],
    "Numerical integration": [
        "Distributional approximations", "Direct simulation and rejection sampling",
        "Example: Bayesian analysis of a bioassay experiment (logistic, nonconjugate)",
        "How many simulation draws are needed?",
    ],
    "Distributional approximations": [
        "Normal Approximations to the Posterior Distribution", "Numerical integration",
        "Variational inference", "Importance sampling",
    ],
    "Direct simulation and rejection sampling": [
        "Numerical integration", "Importance sampling",
        "Distributional approximations", "Gibbs sampler",
    ],
})


# ----------------------------------------------------------------------
# Part III / Stage 8 — Simulation Basics (cont.)  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Importance sampling"] = r"""
Draw from the wrong distribution, then correct
------------------------------------------------

Rejection sampling throws draws away. **Importance sampling** keeps them all and **reweights** instead.
Sample from a convenient proposal :math:`g`, and correct for having used the wrong distribution by
weighting each draw by how much the target wants it relative to the proposal.

The identity
--------------

For any :math:`g` whose support covers the target,

.. math::

   \mathrm{E}_{p}[h(\theta)] = \int h(\theta) \, \frac{p(\theta \mid y)}{g(\theta)} \, g(\theta) \,
   d\theta \;\approx\; \frac{\sum_{s=1}^{S} w_s \, h(\theta^{(s)})}{\sum_{s=1}^{S} w_s},
   \qquad w_s = \frac{q(\theta^{(s)})}{g(\theta^{(s)})},

with :math:`\theta^{(s)} \sim g` and :math:`q \propto p(\theta \mid y)` the unnormalised posterior.
Dividing by :math:`\sum_s w_s` gives the **self-normalised** estimator — which is why the normalising
constant of the posterior is never needed.

The weights are the whole story
---------------------------------

Everything hinges on the **variance of the weights**. If a few draws carry nearly all the weight, the
estimate is effectively based on a handful of points. The standard summary is the **effective sample
size**:

.. math::

   S_{\text{eff}} = \frac{\bigl(\sum_{s} w_s\bigr)^2}{\sum_{s} w_s^2} .

Equal weights give :math:`S_{\text{eff}} = S`; one dominant weight gives :math:`S_{\text{eff}} \approx 1`.
Worse, when :math:`g` has **lighter tails** than the target, the weights can have **infinite variance**
and the estimator has no central limit theorem at all — it will look stable, then jump.

Pareto smoothing
------------------

The modern repair, from Stage 6's LOO lesson, is **PSIS**: fit a generalised Pareto distribution to the
largest weights and replace them with the expected order statistics of that fit. This stabilises the
estimate and — the real prize — the fitted shape :math:`\hat{k}` **diagnoses** the problem. Weights have
finite variance when :math:`\hat{k} < 1/2`; values above :math:`0.7` warn that the proposal is a poor
match.

.. code-block:: python

   import numpy as np
   logw = log_q(draws) - g.logpdf(draws)                 # always work in logs
   logw -= logw.max()                                    # stabilise before exponentiating
   w = np.exp(logw)
   ess = w.sum() ** 2 / (w ** 2).sum()                   # effective sample size
   est = (w * h(draws)).sum() / w.sum()                  # self-normalised estimate

Where it is used
------------------

Importance sampling rarely fits a posterior from scratch — like rejection, it degrades exponentially
with dimension. Its value is as a **correction**: reweighting a variational or Laplace approximation
toward the true posterior; computing **leave-one-out** predictions from a single MCMC fit (PSIS-LOO);
diagnosing prior sensitivity by reweighting to a perturbed prior; and inside particle filters. Use it to
adjust a nearly-right answer, not to find one.
"""

CONTENT["How many simulation draws are needed?"] = r"""
Monte Carlo error is not posterior uncertainty
------------------------------------------------

Two uncertainties coexist in a simulation-based analysis. The **posterior** standard deviation
:math:`\sigma_{\theta}` expresses what the data leave unknown — it cannot be reduced by computing
harder. The **Monte Carlo standard error** expresses how imprecisely your finite sample of draws
estimates a posterior summary — and it shrinks as you run longer:

.. math::

   \mathrm{MCSE}(\bar{\theta}) = \frac{\sigma_{\theta}}{\sqrt{S_{\text{eff}}}} .

Note :math:`S_{\text{eff}}`, not :math:`S`: MCMC draws are **autocorrelated**, so a thousand draws may
carry the information of a hundred.

Fewer than you think, for the mean
------------------------------------

The classic argument is bracing. Suppose you estimate the posterior mean with :math:`S = 100`
independent draws. The MCSE is :math:`\sigma_{\theta}/10`, so the *total* uncertainty about
:math:`\theta` — posterior plus simulation — inflates from :math:`\sigma_{\theta}` to
:math:`\sigma_{\theta}\sqrt{1 + 1/100} \approx 1.005 \, \sigma_{\theta}`. **A 0.5% increase.** For
reporting a posterior mean and interval, a hundred effective draws is already enough, and the fourth
decimal place of a posterior mean was never meaningful anyway.

More than you think, for tails
--------------------------------

The picture changes for quantities that depend on **rare** draws. A 2.5% quantile is estimated from the
draws in that tail; a probability like :math:`\Pr(\theta > c \mid y) = 0.001` requires enough draws to
see the event repeatedly. As a working rule, modern practice targets :math:`S_{\text{eff}} \gtrsim 400`
per quantity of interest — enough for stable tail quantiles and for the convergence diagnostics of Stage
9 to be trustworthy themselves.

.. code-block:: python

   import arviz as az
   az.summary(idata)     # columns: mcse_mean, mcse_sd, ess_bulk, ess_tail, r_hat
   # ess_bulk governs the mean/sd; ess_tail governs the 5%/95% quantiles.
   # Report a number only to the precision its MCSE supports.

The discipline
----------------

Three habits follow. **Report MCSE**, or at least check it: a posterior mean of 2.43 with an MCSE of
0.05 should be written as 2.4. **Check ess_tail separately** from ess_bulk, because a chain that mixes
well in the middle can crawl in the tails. And remember what more draws cannot buy: they shrink Monte
Carlo error toward zero and leave posterior uncertainty exactly where it was. If the interval is too
wide to act on, the remedy is more **data** or a better **model**, never a longer chain.
"""

CONTENT["Computing environments"] = r"""
The tools, and what they hide
-------------------------------

Bayesian computation is now something you **declare** rather than implement. You write the model —
priors, likelihood — and a **probabilistic programming language** derives the log posterior, differentiates
it, and runs an adaptive sampler. Understanding what the layers do is what lets you diagnose them when
they fail.

The Python stack
------------------

* **``scipy.stats``** — closed-form distributions, conjugate updates, direct simulation. Reach here
  first; the Beta–Binomial needs no sampler.
* **PyMC** — model declaration in Python, gradients via PyTensor, sampling with **NUTS**. Idiomatic and
  interactive.
* **Stan** (via ``cmdstanpy``) — a dedicated modelling language, the reference implementation of NUTS,
  and the fastest path for large hierarchical models.
* **NumPyro / BlackJAX** — JAX-based, for GPU acceleration and very large models.
* **ArviZ** — the common currency: an ``InferenceData`` object holding draws, diagnostics and predictive
  samples, whatever produced them.

.. code-block:: python

   import pymc as pm, arviz as az
   with pm.Model() as m:
       theta = pm.Beta("theta", 1, 1)
       pm.Binomial("y", n=10, p=theta, observed=8)
       idata = pm.sample(2000, tune=1000, chains=4, random_seed=0)
   az.summary(idata)          # r_hat, ess_bulk, ess_tail, mcse — read these first

What automatic differentiation buys
-------------------------------------

The gradient of the log posterior is what makes **Hamiltonian Monte Carlo** possible, and computing it
by hand for a hierarchical model is error-prone drudgery. Autodiff makes it exact and free, which is why
the modern default sampler is gradient-based. The cost is a **constraint**: parameters must be
continuous, so discrete unknowns are marginalised out rather than sampled — a change in how models are
written, not merely in how they are fitted.

Reproducibility
-----------------

Simulation results are not deterministic unless you make them so. Set the **seed**, record the library
**versions**, save the ``InferenceData`` rather than re-running, and run **multiple chains** from
dispersed starting points — the last of which is not a courtesy but a prerequisite for the convergence
diagnostics of the next stage. A result you cannot reproduce is a result you cannot debug.
"""

CONTENT["Debugging Bayesian computing"] = r"""
Silence is the danger
-----------------------

A broken sampler rarely raises an exception. It returns numbers — plausible, well-formatted, confidently
summarised numbers — that are simply wrong. Bayesian debugging therefore cannot wait for a crash; it
must **manufacture** situations in which the right answer is known.

Fake-data simulation
----------------------

The single most valuable technique. **Choose** parameter values, **simulate** data from your own model,
then fit that data and check whether the posterior recovers the parameters you chose:

.. code-block:: python

   import numpy as np, pymc as pm, arviz as az
   rng = np.random.default_rng(0)
   true_mu, true_sigma = 1.5, 0.8                       # 1. pick the truth
   y_fake = rng.normal(true_mu, true_sigma, size=200)   # 2. simulate from the model

   with pm.Model():                                     # 3. fit the same model
       mu = pm.Normal("mu", 0, 5)
       sigma = pm.HalfNormal("sigma", 5)
       pm.Normal("y", mu, sigma, observed=y_fake)
       idata = pm.sample(random_seed=0)
   az.summary(idata)      # 4. do the intervals contain 1.5 and 0.8?

If the posterior misses the truth, the bug is in the model, the code, or the sampler — and you know
before touching real data. Repeat the loop many times and count coverage, and you have the
**simulation-based calibration** of Stage 4: 50% intervals should contain the truth 50% of the time.

Read the diagnostics, always
------------------------------

Then check what the sampler is telling you. :math:`\hat{R}` near 1.00 (chains agree), adequate
``ess_bulk`` and ``ess_tail``, and — with HMC — **zero divergences**. A divergence is not a warning to
be suppressed: it says the sampler encountered geometry it could not follow, and the region it failed to
explore is systematically **excluded** from your posterior. The eight-schools funnel is the archetype,
and the fix was a reparameterisation, not a longer chain.

The folk theorem
------------------

Gelman's rule of thumb: *when your computation fails, it is usually a problem with your model*. Slow
mixing, divergences and wandering chains most often signal a posterior that is weakly identified,
badly scaled, or improper — not merely an unlucky algorithm.

So the debugging ladder runs: **start simple** and add one component at a time; **fit to fake data**
before real; **check the priors** by simulating from them alone; **rescale** so parameters are of order
one; **reparameterise** (non-centred) when a hierarchy funnels; and only then blame the sampler. Each
rung tells you something about the model, which is the point.
"""


MINDMAP.update({
    "Importance sampling": [
        "Direct simulation and rejection sampling", "Distributional approximations",
        "Model comparison based on predictive performance",
        "How many simulation draws are needed?",
    ],
    "How many simulation draws are needed?": [
        "Importance sampling", "Numerical integration",
        "E\ufb00ective number of simulation draws", "Inference and assessing convergence",
    ],
    "Computing environments": [
        "Computation and Software", "Debugging Bayesian computing",
        "Stan: developing a computing environment", "Hamiltonian Monte Carlo",
    ],
    "Debugging Bayesian computing": [
        "Computing environments", "Do the Inferences from the Model Make Sense?",
        "Frequency Evaluations of Bayesian Inferences", "Inference and assessing convergence",
    ],
})


# ----------------------------------------------------------------------
# Part III / Stage 9 — MCMC: Gibbs, Metropolis & HMC
# ----------------------------------------------------------------------

CONTENT["Gibbs sampler"] = r"""
Sampling one coordinate at a time
-----------------------------------

Rejection and importance sampling collapse in high dimensions because they try to hit the posterior in
**all** coordinates at once. **Markov chain Monte Carlo** gives up independent draws and instead builds
a chain whose stationary distribution *is* the posterior. The **Gibbs sampler** is the simplest such
chain: update each parameter in turn, drawing it from its **full conditional** distribution given the
current values of all the others.

The algorithm
---------------

For :math:`\theta = (\theta_1, \dots, \theta_d)`, one sweep at iteration :math:`t` is

.. math::

   \theta_1^{(t)} \sim p\bigl(\theta_1 \mid \theta_2^{(t-1)}, \dots, \theta_d^{(t-1)}, y\bigr), \quad
   \theta_2^{(t)} \sim p\bigl(\theta_2 \mid \theta_1^{(t)}, \theta_3^{(t-1)}, \dots, y\bigr), \;\dots

Each draw conditions on the **most recent** value of every other coordinate. Note there is no accept/reject
step — Gibbs is the special case of Metropolis–Hastings whose proposal *is* the full conditional, for
which the acceptance probability is identically **one**. A :math:`d`-dimensional problem becomes
:math:`d` one-dimensional ones.

Where the conditionals come from
----------------------------------

Conjugacy, which is why Part I's algebra returns here as **infrastructure**. In the normal hierarchical
model each conditional is a standard distribution:

.. code-block:: python

   import numpy as np
   from scipy import stats
   # y_j ~ N(theta_j, sigma_j^2);  theta_j ~ N(mu, tau^2)
   for t in range(n_iter):
       V = 1 / (1 / sigma**2 + 1 / tau**2)                    # theta_j | mu, tau, y
       m = V * (ybar / sigma**2 + mu / tau**2)
       theta = stats.norm(m, np.sqrt(V)).rvs()
       mu = stats.norm(theta.mean(), tau / np.sqrt(J)).rvs()  # mu | theta, tau
       ss = ((theta - mu) ** 2).sum()                         # tau^2 | theta, mu
       tau = np.sqrt(ss / stats.chi2(J - 1).rvs())

Strengths and the failure mode
--------------------------------

Gibbs is **tuning-free** and every draw is accepted, which made it the engine of BUGS and JAGS. But it
moves only **along the coordinate axes**, so when parameters are strongly **correlated** in the
posterior, each step is tiny relative to the diagonal ridge the chain must traverse. Successive draws
become nearly identical, mixing crawls, and the effective sample size collapses.

Two remedies define the rest of this stage: **reparameterise** so the coordinates are less correlated,
or **block** highly dependent parameters and update them jointly. And where a conditional has no
closed form, a Metropolis step can be substituted for that coordinate — the hybrid that the next two
lessons build.
"""

CONTENT["Metropolis and Metropolis-Hastings algorithms"] = r"""
Propose, then decide
----------------------

Gibbs needs full conditionals you can sample. **Metropolis** needs only the ability to **evaluate** the
unnormalised posterior :math:`q(\theta) \propto p(\theta \mid y)`. Propose a move; accept it with a
probability chosen so that the chain's stationary distribution is exactly the posterior.

The Metropolis algorithm
--------------------------

From the current :math:`\theta^{(t-1)}`, draw a proposal :math:`\theta^{*}` from a **symmetric** jumping
distribution, :math:`J(\theta^{*} \mid \theta^{(t-1)}) = J(\theta^{(t-1)} \mid \theta^{*})` — a random
walk, :math:`\mathrm{N}(\theta^{(t-1)}, c^2 \Sigma)`, is standard. Accept with probability

.. math::

   r = \min\left(1, \; \frac{q(\theta^{*})}{q(\theta^{(t-1)})}\right).

Moves **uphill** are always accepted; moves **downhill** are accepted with probability equal to the
density ratio. That occasional downhill step is what makes the chain explore rather than optimise. The
normalising constant cancels — the reason MCMC never needs the evidence.

Metropolis–Hastings
---------------------

Drop the symmetry requirement and correct for the asymmetry with a **proposal ratio**:

.. math::

   r = \min\left(1, \; \frac{q(\theta^{*}) \; J(\theta^{(t-1)} \mid \theta^{*})}
                            {q(\theta^{(t-1)}) \; J(\theta^{*} \mid \theta^{(t-1)})}\right).

This generality is what allows asymmetric proposals near boundaries, independence samplers, and — with
the proposal set to the full conditional — recovers **Gibbs**, whose ratio is identically 1. Both satisfy
**detailed balance**, which together with ergodicity guarantees the posterior is the stationary
distribution.

.. code-block:: python

   import numpy as np
   theta, out = init, []
   for t in range(n_iter):
       prop = theta + step * np.random.standard_normal(d)      # symmetric random walk
       if np.log(np.random.rand()) < log_q(prop) - log_q(theta):   # always work in logs
           theta = prop                                        # accept
       out.append(theta.copy())                                # note: reject means REPEAT the draw

Tuning is everything
----------------------

The step size :math:`c` decides the outcome. **Too small**: nearly every proposal is accepted, but the
chain inches along and successive draws are highly correlated. **Too large**: proposals land in the
tails, almost all are rejected, and the chain **stands still** — repeating the same value, which is a
legitimate draw but adds no information. The classic guidance targets an acceptance rate near
:math:`0.44` in one dimension and about :math:`0.23` in high dimensions.

Even tuned, a random walk explores by **diffusion**: to travel a distance :math:`L` it needs roughly
:math:`(L/c)^2` steps. That quadratic cost is what Hamiltonian Monte Carlo was invented to escape.
"""

CONTENT["Using Gibbs and Metropolis as building blocks"] = r"""
Mix and match
---------------

Gibbs and Metropolis are not rivals; they are **components**. A realistic model has some parameters with
tidy conjugate conditionals and others without. The natural sampler updates each block by whatever
method suits it, and the composition still targets the correct posterior.

Metropolis-within-Gibbs
-------------------------

Sweep through the parameter blocks as in Gibbs. For a block whose full conditional is a standard
distribution, **draw from it** (acceptance 1). For a block whose conditional is only known up to a
constant, take a **Metropolis step** targeting that conditional. Each update leaves the posterior
invariant, so the composed chain does too.

.. code-block:: python

   import numpy as np
   from scipy import stats
   for t in range(n_iter):
       # block 1: conjugate -> exact Gibbs draw
       sigma2 = stats.invgamma(a + n / 2, scale=b + 0.5 * ((y - X @ beta) ** 2).sum()).rvs()
       # block 2: no closed form -> Metropolis step on the conditional
       prop = beta + step * np.random.standard_normal(len(beta))
       if np.log(np.random.rand()) < log_cond(prop, sigma2) - log_cond(beta, sigma2):
           beta = prop

Blocking and reparameterising
-------------------------------

The two levers that fix slow mixing, both aimed at **posterior correlation**:

* **Blocking** — update strongly dependent parameters **jointly** rather than one at a time. Gibbs
  moves along axes; a correlated pair traversed jointly moves along the ridge. Regression coefficients
  under a conjugate prior can be drawn as a whole vector.
* **Reparameterising** — change coordinates so the posterior is less correlated. Centre predictors;
  replace :math:`\theta_j \sim \mathrm{N}(\mu, \tau^2)` with :math:`\theta_j = \mu + \tau \eta_j`,
  :math:`\eta_j \sim \mathrm{N}(0,1)`. The **non-centred** parameterisation from eight schools is exactly
  this move, and it works for Gibbs and HMC alike.

Auxiliary variables
---------------------

A third trick: **add** parameters to make the conditionals tractable. Data augmentation introduces latent
variables (a :math:`t` distribution written as a scale-mixture of normals; a probit model given latent
normals) whose presence turns an awkward joint into a chain of conjugate conditionals. You sample in a
larger space and **discard the extra columns**, which — per Part I — is marginalisation.

The upshot
------------

These compositions dominated Bayesian computation for two decades, and they still matter: they explain
what BUGS and JAGS do, they remain the right tool for discrete parameters (which gradient methods cannot
touch), and their diagnostics are the subject of the next lesson. Their limitation is universal — all of
them explore by **random walk**, and none escape its quadratic cost.
"""

CONTENT["Inference and assessing convergence"] = r"""
The two questions
-------------------

MCMC draws are neither independent nor, at first, from the posterior. So two questions must be answered
before any summary is trusted. **Has the chain converged** — is it now sampling the target rather than
still travelling toward it? And **how much information** do its correlated draws actually carry?

Warm-up, and many chains
--------------------------

Discard an initial **warm-up** phase (the sampler is also adapting its step size there). Then run
**several chains from dispersed starting points**: a single chain stuck in one mode looks perfectly
healthy from the inside. Convergence is diagnosed by **disagreement between chains**, which requires
more than one.

R-hat
-------

The classical diagnostic compares **between-chain** to **within-chain** variance: if the chains have
forgotten where they started and are exploring the same distribution, the pooled variance is no larger
than the average within-chain variance, and

.. math::

   \hat{R} = \sqrt{\frac{\widehat{\mathrm{var}}^{+}(\theta \mid y)}{W}} \;\longrightarrow\; 1 .

Vehtari, Gelman, Simpson, Carpenter and Bürkner showed the original :math:`\hat{R}` has **serious
flaws** — it fails to detect trouble when a chain has a **heavy tail** or when the **variance differs
across chains**. Their replacement **rank-normalises** the draws (robust to heavy tails), **folds** them
about the median to catch differing scales, and **splits** each chain in half to catch drift within a
chain. Modern practice: require :math:`\hat{R} < 1.01`, not the old 1.1.

Effective sample size
-----------------------

Autocorrelation means :math:`S` draws carry the information of :math:`S_{\text{eff}} < S` independent
ones. Report **bulk-ESS** (governing the mean and sd) and **tail-ESS** (governing the 5% and 95%
quantiles) separately, and require **both above about 400** — a chain can mix well in the middle and
crawl in the tails.

.. code-block:: python

   import arviz as az
   az.summary(idata)                 # r_hat, ess_bulk, ess_tail, mcse_mean, mcse_sd
   az.plot_rank(idata)               # rank plots: uniform bars = chains agree
   idata.sample_stats["diverging"].sum()    # HMC: must be zero

Prefer **rank plots** to trace plots: with many chains the "fat hairy caterpillar" is unreadable, whereas
rank histograms should be uniform across chains if all are sampling the same distribution.

What convergence is not
-------------------------

These diagnostics are **necessary, not sufficient**. They can only detect the failures the chains
happened to reveal: a mode never visited by any chain leaves no trace in :math:`\hat{R}`. And a converged
sampler faithfully reproduces the posterior of a model that may be **wrong** — computation and model
adequacy are separate questions, checked by separate tools. Convergence buys you the right posterior for
the model you actually wrote.
"""


MINDMAP.update({
    "Gibbs sampler": [
        "Metropolis and Metropolis-Hastings algorithms",
        "Using Gibbs and Metropolis as building blocks",
        "Example: hierarchical normal model", "Direct simulation and rejection sampling",
    ],
    "Metropolis and Metropolis-Hastings algorithms": [
        "Gibbs sampler", "Using Gibbs and Metropolis as building blocks",
        "E\ufb03cient Metropolis jumping rules", "Hamiltonian Monte Carlo",
    ],
    "Using Gibbs and Metropolis as building blocks": [
        "Gibbs sampler", "Metropolis and Metropolis-Hastings algorithms",
        "E\ufb03cient Gibbs samplers", "Further extensions to Gibbs and Metropolis",
    ],
    "Inference and assessing convergence": [
        "E\ufb00ective number of simulation draws", "Debugging Bayesian computing",
        "How many simulation draws are needed?", "Hamiltonian Monte Carlo",
    ],
})


# ----------------------------------------------------------------------
# Part III / Stage 9 — MCMC (cont.)
# ----------------------------------------------------------------------

CONTENT["E\ufb00ective number of simulation draws"] = r"""
Correlated draws carry less information
-----------------------------------------

An MCMC chain's draws are **dependent** by construction: each is a perturbation of the last. A thousand
such draws therefore say less about the posterior than a thousand independent ones. The **effective
sample size** measures how much less.

The definition
----------------

If the chain's autocorrelation at lag :math:`t` is :math:`\rho_t`, then

.. math::

   S_{\text{eff}} = \frac{S}{1 + 2 \sum_{t=1}^{\infty} \rho_t} .

Independent draws (:math:`\rho_t = 0`) give :math:`S_{\text{eff}} = S`. Positively autocorrelated draws
— the usual case for random-walk Metropolis and Gibbs — give :math:`S_{\text{eff}} \ll S`. The estimate
truncates the sum once the estimated :math:`\rho_t` becomes noise, and in practice pools information
across chains.

Better than independent
-------------------------

A pleasing surprise: the sum can be **negative**. When successive draws are **anticorrelated**, the
effective sample size **exceeds** the number of iterations, :math:`S_{\text{eff}} > S`. Hamiltonian
Monte Carlo, and the NUTS sampler in particular, routinely achieves this for parameters whose posterior
is close to Gaussian with little dependence on the others — the chain deliberately overshoots, so
consecutive draws sit on opposite sides of the mode. For such problems MCMC beats independent sampling.

Bulk and tail
---------------

One number is not enough. **bulk-ESS** governs the precision of the mean and standard deviation;
**tail-ESS** governs the extreme quantiles, and a chain that mixes briskly through the centre can crawl
in the tails. Report both, and require both above roughly **400** before trusting any summary — including
:math:`\hat{R}` itself, whose reliability depends on adequate ESS.

.. code-block:: python

   import arviz as az
   az.summary(idata)                    # ess_bulk, ess_tail, mcse_mean, mcse_sd, r_hat
   az.ess(idata, method="tail")         # tail-ESS explicitly
   az.plot_autocorr(idata, var_names=["tau"])   # where does rho_t decay to zero?

Don't thin
------------

A persistent habit deserves retiring. **Thinning** — keeping every tenth draw — does not improve
inference. Discarding draws throws away information; the effective sample size of the thinned chain is
at best what the full chain already carried, and if the draws are anticorrelated, thinning actively
**destroys** the advantage. The only legitimate reason to thin is **memory**. If ESS is too low, the
answer is a better sampler, a reparameterisation, or a longer run — never a smaller one.
"""

CONTENT["Example: hierarchical normal model"] = r"""
Gibbs, worked through
-----------------------

The hierarchical normal model of Stage 5 has closed-form conditionals at every level, which makes it the
canonical Gibbs example — and a useful place to watch a sampler misbehave. Take :math:`J` groups,
:math:`n_j` observations each:

.. math::

   y_{ij} \sim \mathrm{N}(\theta_j, \sigma^2), \qquad
   \theta_j \sim \mathrm{N}(\mu, \tau^2), \qquad
   p(\mu, \log \sigma, \log \tau) \propto 1 .

The unknowns are the group means :math:`\theta`, the population mean :math:`\mu`, and the two variances.

The four conditionals
-----------------------

Each is a Stage 2 or Stage 3 result, reused:

.. math::

   \theta_j \mid \cdot \sim \mathrm{N}\!\left(
     \frac{\frac{n_j}{\sigma^2}\bar{y}_{\cdot j} + \frac{1}{\tau^2}\mu}
          {\frac{n_j}{\sigma^2} + \frac{1}{\tau^2}}, \;
     \left(\frac{n_j}{\sigma^2} + \frac{1}{\tau^2}\right)^{-1}\right),
   \qquad
   \mu \mid \cdot \sim \mathrm{N}\!\left(\bar{\theta}, \; \frac{\tau^2}{J}\right),

with :math:`\sigma^2` and :math:`\tau^2` drawn from scaled inverse-:math:`\chi^2` distributions built
from the within- and between-group sums of squares. Cycle through them and the chain converges to the
joint posterior — no tuning, no rejections.

.. code-block:: python

   import numpy as np
   from scipy import stats
   for t in range(n_iter):
       V = 1.0 / (n_j / sigma**2 + 1.0 / tau**2)                  # theta | mu, sigma, tau, y
       m = V * (n_j * ybar_j / sigma**2 + mu / tau**2)
       theta = stats.norm(m, np.sqrt(V)).rvs()
       mu = stats.norm(theta.mean(), tau / np.sqrt(J)).rvs()      # mu | theta, tau
       ss_w = ((y - theta[group]) ** 2).sum()                     # sigma^2 | theta, y
       sigma = np.sqrt(ss_w / stats.chi2(n).rvs())
       ss_b = ((theta - mu) ** 2).sum()                           # tau^2 | theta, mu
       tau = np.sqrt(ss_b / stats.chi2(J - 1).rvs())

Watch it stick
----------------

Now look at what the chain does when :math:`\tau` wanders near **zero**. The conditional for
:math:`\theta_j` collapses onto :math:`\mu`; the conditional for :math:`\tau^2` is then built from
:math:`\theta_j` that are all nearly equal, so it stays small. The two conditionals **trap each other**,
and the sampler crawls through the neck of the funnel from eight schools. Gibbs does not diverge or warn
— it simply mixes so slowly that the low-:math:`\tau` region is under-visited, and :math:`\hat{R}` may
look fine on a short run.

The lesson
------------

Closed-form conditionals guarantee **correctness in the limit**, not **efficiency in practice**. The
diagnosis is posterior **correlation between levels**, and the two cures are the subject of the next
lesson: reparameterise so the levels decouple, or update them jointly.
"""

CONTENT["E\ufb03cient Gibbs samplers"] = r"""
Fixing what slows Gibbs down
------------------------------

Gibbs moves parallel to the coordinate axes. Its efficiency is therefore governed entirely by the
**posterior correlation** among the parameters it updates separately: a narrow diagonal ridge is
traversed in tiny axis-aligned steps. Three standard cures attack that correlation directly.

Reparameterise
----------------

Change coordinates so the posterior is closer to spherical.

* **Centring predictors.** In :math:`y = \alpha + \beta x`, an uncentred :math:`x` makes
  :math:`\alpha` and :math:`\beta` strongly correlated; subtracting :math:`\bar{x}` makes them nearly
  independent, and Gibbs mixes immediately.
* **Non-centred hierarchies.** Replace :math:`\theta_j \sim \mathrm{N}(\mu, \tau^2)` with
  :math:`\theta_j = \mu + \tau \eta_j`, :math:`\eta_j \sim \mathrm{N}(0, 1)`. The funnel of the previous
  lesson vanishes because :math:`\eta` no longer depends on :math:`\tau`.
* **Rescaling.** Put parameters on comparable scales so no single conditional dominates.

Which parameterisation wins depends on the data: **centred** works when groups are informative (large
:math:`n_j`, large :math:`\tau`); **non-centred** when they are not. Hierarchical models with weak data
per group want the non-centred form.

Block
-------

Update correlated parameters **jointly**, drawing from their joint conditional. In a conjugate linear
model the entire coefficient vector :math:`\beta` is drawn in one multivariate normal step rather than
coordinate by coordinate, and correlation among coefficients stops mattering at all.

.. code-block:: python

   import numpy as np
   from scipy import stats
   # blocked draw of the whole coefficient vector, given sigma^2
   V = np.linalg.inv(X.T @ X / sigma**2 + np.linalg.inv(Sigma0))
   m = V @ (X.T @ y / sigma**2 + np.linalg.inv(Sigma0) @ mu0)
   beta = stats.multivariate_normal(m, V).rvs()      # one step, any correlation

Augment
---------

Add **auxiliary variables** that make the conditionals conjugate. A :math:`t` likelihood is a
scale-mixture of normals: introduce per-observation scale parameters and every conditional becomes
standard. Probit regression given latent normal utilities is conjugate throughout. You sample in a
bigger space, then **drop the extra columns** — marginalisation, again.

The residual limit
--------------------

Even a well-tuned Gibbs sampler explores by **random walk**, and its cost grows quadratically with the
distance it must travel. Reparameterising and blocking buy constants, sometimes large ones. Escaping the
random walk itself requires **gradients**, which is Hamiltonian Monte Carlo.
"""

CONTENT["E\ufb03cient Metropolis jumping rules"] = r"""
The proposal is the algorithm
-------------------------------

Metropolis is correct for *any* symmetric proposal, and useless for most of them. The **jumping rule**
determines how far the chain travels per unit of computation, and tuning it is not decoration but the
whole game.

The trade-off, quantified
---------------------------

Too small a step: proposals almost always accepted, but each move is negligible and the draws are
strongly autocorrelated. Too large: proposals land in the tails, are rejected, and the chain **repeats
its current value** — draws that are valid but carry no new information. The optimum sits between, and
for a random-walk proposal :math:`\mathrm{N}(\theta^{(t-1)}, c^2 \Sigma)` targeting a roughly normal
posterior with covariance :math:`\Sigma`, the asymptotically optimal scale is

.. math::

   c \;\approx\; \frac{2.4}{\sqrt{d}},

giving an acceptance rate near :math:`0.44` in one dimension and settling to about :math:`0.23` as
:math:`d` grows. Efficiency falls like :math:`1/d`: each added dimension makes the random walk worse.

Adapt, then stop
------------------

Practical samplers tune :math:`c` during **warm-up**, raising it when acceptance is too high and lowering
it when too low, and estimate :math:`\Sigma` from the draws so far so that proposals follow the
posterior's shape rather than a sphere.

.. code-block:: python

   import numpy as np
   accepted, step = 0, 0.1
   for t in range(n_warmup):
       ...                                            # one Metropolis update
       if (t + 1) % 50 == 0:                          # adapt in batches
           rate = accepted / 50
           step *= np.exp(0.5 * (rate - 0.234))       # nudge toward the target rate
           accepted = 0
   # then FREEZE step and Sigma; sampling draws must come from a time-homogeneous chain

The final clause matters: adaptation that continues during sampling destroys the Markov property, and the
chain no longer targets the posterior. Adapt in warm-up, freeze, then collect.

Better proposals, and their ceiling
-------------------------------------

Beyond scaling, the proposal can be improved by **preconditioning** with an estimate of :math:`\Sigma`
(equivalently, sampling in a whitened space), by **blocking** correlated parameters, or by adaptive
schemes that learn the covariance online. Each buys a constant factor.

None of them changes the fundamental cost. A random walk needs :math:`\mathcal{O}((L/c)^2)` steps to
cross a distance :math:`L`, and :math:`c` must shrink like :math:`1/\sqrt{d}`. **Gradients** break this
scaling by proposing *directed* moves, which is what the next lesson introduces.
"""


MINDMAP.update({
    "E\ufb00ective number of simulation draws": [
        "Inference and assessing convergence", "How many simulation draws are needed?",
        "E\ufb03cient Metropolis jumping rules", "Hamiltonian Monte Carlo",
    ],
    "Example: hierarchical normal model": [
        "Gibbs sampler", "E\ufb03cient Gibbs samplers",
        "Normal model with exchangeable parameters",
        "Example: hierarchical normal model (continued)",
    ],
    "E\ufb03cient Gibbs samplers": [
        "Gibbs sampler", "Example: hierarchical normal model",
        "Using Gibbs and Metropolis as building blocks",
        "Computation: batching and transformation",
    ],
    "E\ufb03cient Metropolis jumping rules": [
        "Metropolis and Metropolis-Hastings algorithms",
        "E\ufb00ective number of simulation draws",
        "Hamiltonian Monte Carlo", "Further extensions to Gibbs and Metropolis",
    ],
})


# ----------------------------------------------------------------------
# Part III / Stage 9 — MCMC (cont.)  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Further extensions to Gibbs and Metropolis"] = r"""
Beyond the two basics
-----------------------

Gibbs needs conditionals; Metropolis needs a tuned proposal. A family of extensions removes one
requirement or the other, usually by the same device: **enlarge the space** with auxiliary variables, so
that a hard move in :math:`\theta` becomes an easy move in :math:`(\theta, u)`.

Slice sampling
----------------

Introduce a height :math:`u` beneath the unnormalised density and sample **uniformly from the area under
the curve**. Given :math:`\theta`, draw :math:`u \sim \mathrm{Uniform}(0, q(\theta))`; given :math:`u`,
draw :math:`\theta` uniformly from the **slice** :math:`\{\theta : q(\theta) \ge u\}`. Marginally,
:math:`\theta` has the target distribution.

.. math::

   p(\theta, u) \propto \mathbf{1}\{0 < u < q(\theta)\}
   \quad \Longrightarrow \quad p(\theta) \propto q(\theta).

The appeal is that it is **self-tuning**: no step size, no acceptance rate. The slice is found by
stepping out from the current point and shrinking on rejection. Slice sampling is the workhorse for
one-dimensional conditionals inside a larger Gibbs sweep, and it handles awkward univariate shapes that
would defeat a fixed proposal.

Reversible jump
-----------------

When the **number of parameters is itself unknown** — how many mixture components, which predictors
belong in the regression — the posterior lives on a union of spaces of different dimension. **Reversible
jump MCMC** moves between them, proposing births and deaths of parameters and correcting the acceptance
ratio with a Jacobian for the dimension change. It is powerful, notoriously fiddly to tune, and largely
superseded in applied work by two alternatives: **continuous model expansion** (Stage 6) and the
**nonparametric** priors of Stage 16, which let complexity grow without discrete jumps.

Other auxiliary-variable tricks
---------------------------------

* **Data augmentation** — latent variables that restore conjugacy: a :math:`t` as a scale-mixture of
  normals, probit regression given latent utilities.
* **Simulated tempering / parallel tempering** — run chains at several "temperatures", flattening the
  posterior so a chain can cross **low-probability valleys** between modes, and swap states between
  them.
* **Adaptive MCMC** — learn the proposal covariance online, with the adaptation vanishing over time so
  that ergodicity is preserved.

.. code-block:: python

   # PyMC assigns samplers per variable: NUTS for continuous, specialised steppers otherwise
   import pymc as pm
   with model:
       idata = pm.sample(step=[pm.NUTS([mu, tau]), pm.BinaryGibbsMetropolis([z])])

Where they matter now
-----------------------

Gradient-based sampling handles continuous parameters better than any of these. Their enduring role is
the **complement**: discrete unknowns (which HMC cannot touch, since it needs derivatives), multimodal
posteriors (where tempering still helps), and univariate conditionals inside composite samplers. Modern
practice **marginalises** discrete parameters analytically where possible, and reaches for these tools
where it is not.
"""

CONTENT["Hamiltonian Monte Carlo"] = r"""
Stop wandering; use the gradient
----------------------------------

Random-walk Metropolis explores by diffusion, needing :math:`\mathcal{O}((L/c)^2)` steps to travel a
distance :math:`L`. **Hamiltonian Monte Carlo** replaces the aimless jump with a **directed trajectory**,
computed from the **gradient of the log posterior**. It is the reason modern Bayesian models with
thousands of parameters are fittable at all.

Physics on the posterior
--------------------------

Treat :math:`\theta` as the position of a particle on a landscape whose **potential energy** is the
negative log posterior. Add an auxiliary **momentum** :math:`\rho`, drawn afresh each iteration from
:math:`\mathrm{N}(0, M)`. The total energy is the **Hamiltonian**

.. math::

   H(\theta, \rho) = \underbrace{-\log p(\theta \mid y)}_{\text{potential}}
                   + \underbrace{\tfrac{1}{2} \rho^{\top} M^{-1} \rho}_{\text{kinetic}},

and the joint density :math:`p(\theta, \rho) \propto e^{-H(\theta, \rho)}` has the posterior as its
:math:`\theta`-marginal — so simulating the particle's motion and **discarding the momentum** samples the
posterior. Hamiltonian flow conserves :math:`H`, is time-reversible, and preserves phase-space volume:
exactly the properties a valid MCMC proposal needs.

Leapfrog, and the correction
------------------------------

The dynamics cannot be solved exactly, so they are integrated with the **leapfrog** scheme — a half-step
of momentum along the gradient, a full step of position, another half-step of momentum. Leapfrog is
time-reversible and volume-preserving **exactly**, but conserves energy only **approximately**. The
residual error is removed by a Metropolis accept/reject on :math:`e^{-H}`, whose acceptance is near one
when the step size is small.

.. code-block:: python

   def leapfrog(theta, rho, eps, L, grad_logp):
       rho = rho + 0.5 * eps * grad_logp(theta)          # half step
       for _ in range(L):
           theta = theta + eps * rho                     # full position step
           rho = rho + eps * grad_logp(theta)            # full momentum step
       rho = rho - 0.5 * eps * grad_logp(theta)          # correct the last half step
       return theta, -rho                                # negate for reversibility

Divergences: not a warning, a wound
-------------------------------------

The leapfrog integrator is only **conditionally stable**. Where the posterior's curvature is sharp
relative to the step size, the simulated trajectory departs from the true one and the energy error
explodes — a **divergent transition**. Stan and PyMC flag these, and the flag matters: positions after a
divergence are **never selected** as draws, so the sampler cannot explore that region. HMC silently
degenerates toward a random walk, and estimates are **biased** by the systematic omission.

Non-negotiable: divergences must be **fixed**, not suppressed. Raise ``target_accept`` (smaller steps),
rescale parameters, or — usually the real answer — **reparameterise** the geometry that caused them. The
next lesson is that story.
"""

CONTENT["Hamiltonian Monte Carlo for a hierarchical model"] = r"""
HMC meets the funnel
----------------------

Hierarchical models are where HMC's power and its fragility both appear. Return to eight schools in its
**centred** form, :math:`\theta_j \sim \mathrm{N}(\mu, \tau)`, and watch the sampler struggle exactly
where Gibbs crawled.

The geometry
--------------

Plot :math:`\log \tau` against any :math:`\theta_j`. The joint posterior is a **funnel**: when
:math:`\tau` is large, the :math:`\theta_j` are spread widely; when :math:`\tau` is small, they are
squeezed into a narrow neck. The posterior's **curvature** therefore changes by orders of magnitude
along :math:`\tau`, and a leapfrog step size that is stable in the wide mouth is catastrophically too
large in the neck.

The symptom is **divergences**, and they cluster at small :math:`\tau` — precisely the region that
decides whether the schools should be pooled. The sampler reports a value for :math:`\tau`; it is biased
away from zero, because the neck was never explored.

The cure is a change of coordinates
-------------------------------------

Write the same model so that the group parameters no longer depend on :math:`\tau`:

.. math::

   \theta_j = \mu + \tau \, \eta_j, \qquad \eta_j \sim \mathrm{N}(0, 1).

This is the **non-centred** parameterisation. The sampler now moves in :math:`(\mu, \tau, \eta)`, where
:math:`\eta` has a fixed standard-normal geometry independent of :math:`\tau`. The funnel is gone; one
step size fits everywhere.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu  = pm.Normal("mu", 0, 5)
       tau = pm.HalfCauchy("tau", 5)
       eta = pm.Normal("eta", 0, 1, shape=8)             # <- flat geometry
       theta = pm.Deterministic("theta", mu + tau * eta) # <- recovered by transformation
       pm.Normal("y", theta, sigma=sigma_j, observed=y)
       idata = pm.sample(target_accept=0.9)
   idata.sample_stats["diverging"].sum()                 # must be 0

Which parameterisation, when
------------------------------

Not always non-centred. When each group has **plenty of data** — large :math:`n_j`, or a large
:math:`\tau` — the likelihood pins each :math:`\theta_j` down and the **centred** form is better
conditioned; forcing the non-centred version then *creates* the correlation it was meant to remove. The
rule of thumb: **weak data per group → non-centred; strong data per group → centred**, and with mixed
groups, some models parameterise them differently.

The general lesson is the one that closes this stage. HMC's efficiency is determined by the posterior's
**geometry**, and geometry is something the modeller controls through parameterisation. A divergence is
not a complaint about the algorithm; it is information about the model.
"""

CONTENT["Stan: developing a computing environment"] = r"""
A language for models
-----------------------

Stan is a probabilistic programming language: you declare data, parameters and a model, and Stan compiles
that declaration into C++ that computes the log posterior **and its gradient** by automatic
differentiation, then samples it with an adaptive HMC variant. The point is separation of concerns — the
statistician states the model; the machine derives the calculus and tunes the sampler.

The blocks
------------

A Stan program is organised so that its structure mirrors the probability model, which is itself a
discipline: what is data, what is unknown, what is derived.

.. code-block:: stan

   data {
     int<lower=0> J;                 // number of schools
     vector[J] y;                    // estimated effects
     vector<lower=0>[J] sigma;       // known standard errors
   }
   parameters {
     real mu;
     real<lower=0> tau;
     vector[J] eta;                  // non-centred
   }
   transformed parameters {
     vector[J] theta = mu + tau * eta;
   }
   model {
     mu ~ normal(0, 5);
     tau ~ cauchy(0, 5);
     eta ~ std_normal();
     y ~ normal(theta, sigma);       // likelihood
   }

Constraints (``<lower=0>``) are not assertions but **transformations**: Stan samples an unconstrained
:math:`\log \tau` internally and applies the Jacobian, which is why HMC never proposes a negative
variance.

NUTS
------

Plain HMC has two free parameters: the step size and the **number of leapfrog steps**. Too few and the
sampler random-walks; too many and the trajectory curls back on itself, wasting computation. The
**No-U-Turn Sampler** (Hoffman and Gelman) removes the second: it doubles the trajectory forward and
backward until the path begins to double back on itself, then samples a point from it. Doubling in both
directions is what preserves reversibility. The step size is tuned during warm-up to hit a target
acceptance rate.

.. code-block:: python

   from cmdstanpy import CmdStanModel
   import arviz as az
   fit = CmdStanModel(stan_file="eight_schools.stan").sample(data=data, adapt_delta=0.9)
   az.summary(az.from_cmdstanpy(fit))         # same diagnostics, whatever the backend

Why it changed practice
-------------------------

Three things. **Gradients for free**, so HMC became usable by people who do not want to differentiate
hierarchical likelihoods by hand. **Diagnostics by default** — R-hat, ESS, divergences and energy
reported without asking, which made unchecked sampling embarrassing rather than normal. And a
**declarative model** that can be read, reviewed and reparameterised as a statement about the world
rather than as an algorithm. PyMC, NumPyro and BlackJAX offer the same bargain in Python. The remaining
craft is the modelling — which is where Parts IV and V go.
"""


MINDMAP.update({
    "Further extensions to Gibbs and Metropolis": [
        "Using Gibbs and Metropolis as building blocks", "Gibbs sampler",
        "Unspecified number of mixture components", "Continuous model expansion",
    ],
    "Hamiltonian Monte Carlo": [
        "Metropolis and Metropolis-Hastings algorithms",
        "Hamiltonian Monte Carlo for a hierarchical model",
        "Stan: developing a computing environment", "Inference and assessing convergence",
    ],
    "Hamiltonian Monte Carlo for a hierarchical model": [
        "Hamiltonian Monte Carlo", "Example: parallel experiments in eight schools",
        "E\ufb03cient Gibbs samplers", "Weakly Informative Priors for Variance Parameters",
    ],
    "Stan: developing a computing environment": [
        "Hamiltonian Monte Carlo", "Computing environments",
        "Computation and Software", "Debugging Bayesian computing",
    ],
})


# ----------------------------------------------------------------------
# Part III / Stage 10 — Modal & Variational Approximation
# ----------------------------------------------------------------------

CONTENT["Finding posterior modes"] = r"""
Optimisation instead of sampling
----------------------------------

Sampling explores the posterior; **optimisation** finds its peak. The **posterior mode** — the MAP
estimate — is the cheapest possible summary, obtained by maximising :math:`\log p(\theta \mid y)` with
the same machinery used for maximum likelihood, since a prior is just an additive penalty on the log
scale.

How
-----

Because the objective is a sum of log-likelihood and log-prior, any gradient-based optimiser applies.
Newton-type methods use the curvature and give the **observed information** for free, which is exactly
what the Laplace approximation of Stage 4 needs:

.. code-block:: python

   import numpy as np
   from scipy import optimize

   neg_log_post = lambda t: -(log_lik(t, y) + log_prior(t))
   fit = optimize.minimize(neg_log_post, x0, method="BFGS")
   mode = fit.x
   cov = fit.hess_inv            # inverse observed information ~ posterior covariance
   sd = np.sqrt(np.diag(cov))

Practical advice from the sampling lessons carries over: optimise on an **unconstrained** scale
(:math:`\log \sigma`, :math:`\mathrm{logit}\, \theta`), start from several dispersed points to detect
multiple modes, and rescale so parameters are of order one.

Why the mode can mislead
--------------------------

Three failures deserve naming. The mode is **not invariant** to reparameterisation: the mode of
:math:`\sigma` is not the transform of the mode of :math:`\log \sigma`, because densities pick up a
Jacobian — whereas posterior *means* of transformed draws behave sensibly. The mode may sit at a
**boundary** (a variance of exactly zero), which the next lesson addresses. And in **high dimensions**
the mode is not where the probability is: almost all posterior mass lies in a thin shell away from the
peak, so a point estimate at the mode is unrepresentative of the draws you would obtain by sampling.

Where it belongs
------------------

The mode remains useful in three roles, all honest. As a **fast approximate answer** for large, regular
problems, where large-sample theory says the posterior is nearly normal around it. As **starting values
and a metric** for MCMC — the inverse Hessian is a good initial mass matrix. And as the point estimate
that connects Bayesian analysis to **penalised likelihood** (Stage 4): ridge, lasso and multilevel
point-estimation software are all computing posterior modes. What the mode never provides is
**uncertainty**; the curvature around it is an approximation whose adequacy must itself be checked.
"""

CONTENT["Boundary-avoiding priors for modal summaries"] = r"""
Zero variance, implausibly
----------------------------

Fit a hierarchical model by maximum likelihood with only a few groups, and a familiar pathology appears:
the estimated group-level variance is exactly **zero**. Not small — zero. The likelihood's maximum lies
on the boundary of the parameter space, and the software reports complete pooling as a certainty.
Anyone who believes the groups differ at all knows the estimate is wrong.

Why it happens
----------------

With :math:`J` small, the between-group variation observed in the data can easily be **no larger than**
what sampling noise alone would produce. The likelihood for :math:`\tau` is then monotonically
decreasing, so its maximum is at :math:`\tau = 0`. Nothing is broken; the *mode* is simply a bad
summary of a posterior that has plenty of mass at positive :math:`\tau`. Full Bayes avoids the problem
by **averaging** rather than maximising — but modal estimation is popular for speed, so it needs a fix
of its own.

The fix: penalise the boundary
--------------------------------

Chung, Rabe-Hesketh, Dorie, Gelman and Liu proposed a **boundary-avoiding prior**: put a
:math:`\mathrm{Gamma}(\alpha, \lambda)` prior on the group-level **standard deviation** with shape
:math:`\alpha > 1`, so that the density vanishes at zero and the posterior mode cannot land there.

.. math::

   p(\tau) \propto \tau^{\alpha - 1} e^{-\lambda \tau}, \qquad \alpha = 2, \; \lambda \to 0 .

With :math:`\alpha = 2` the prior is proportional to :math:`\tau` near the origin — enough to push the
mode off the boundary, not enough to fight real data. The default :math:`\mathrm{Gamma}(2, \lambda \to
0)` has three properties worth stating precisely: the penalised estimate is **approximately one standard
error from zero** when the ML estimate is zero; it closely approximates the **posterior median** under a
noninformative prior; and as the number of groups grows it **coincides with maximum likelihood**. The
prior does work exactly where the data are silent, and withdraws where they speak.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # boundary-avoiding: density -> 0 as tau -> 0, so the MODE stays positive
       tau = pm.Gamma("tau", alpha=2, beta=1e-4)
       # (for full-Bayes sampling, HalfNormal/HalfCauchy remain the default;
       #  boundary-avoidance is specifically about modal summaries)
       ...

Covariance matrices too
-------------------------

The same authors extended the idea to group-level **covariance** matrices, where maximum likelihood
routinely returns a **singular** estimate: use a **Wishart** prior — deliberately *not* the
inverse-Wishart of Stage 3 — with degrees of freedom equal to the number of varying coefficients plus
two, which keeps the modal estimate positive definite.

The moral generalises. A prior chosen for **modal** estimation is not the prior you would choose for
**sampling**: it must shape the density's *peak*, not merely its mass. Naming which summary you intend
is part of specifying the model.
"""

CONTENT["Normal and related mixture approximations"] = r"""
One normal is rarely enough
-----------------------------

The Laplace approximation places a single normal at a single mode. Real posteriors are skewed,
heavy-tailed, or **multimodal** — a mixture likelihood has one mode per labelling, a weakly identified
model has ridges. The natural repair is to approximate with a **mixture** of normals, one component per
mode.

Building the mixture
----------------------

Run the optimiser from many **dispersed starting points**, collect the distinct modes
:math:`\hat{\theta}_k` and their curvatures :math:`\Sigma_k`, and weight each component by the posterior
mass it accounts for — the Laplace estimate of the local integral:

.. math::

   p(\theta \mid y) \;\approx\; \sum_{k} \omega_k \,
   \mathrm{N}\bigl(\theta \mid \hat{\theta}_k, \Sigma_k\bigr),
   \qquad
   \omega_k \;\propto\; p(\hat{\theta}_k \mid y) \; |\Sigma_k|^{1/2} .

The determinant factor is doing real work: a **broad, shallow** mode can hold more probability than a
**narrow, tall** one, so mixture weights must account for width, not just height.

Use heavier tails
-------------------

If the approximation will serve as a **proposal** or an **importance-sampling** density, replace each
normal by a multivariate :math:`t` with a few degrees of freedom. Light tails are the failure mode that
matters: a proposal thinner than the target produces weights with infinite variance, and the estimate
degrades without warning (Stage 8).

.. code-block:: python

   import numpy as np
   from scipy import optimize, stats

   modes = [optimize.minimize(neg_log_post, x0, method="BFGS") for x0 in dispersed_starts]
   modes = dedupe(modes)                                    # distinct optima only
   logw = [-m.fun + 0.5 * np.log(np.linalg.det(m.hess_inv)) for m in modes]
   w = np.exp(logw - max(logw)); w /= w.sum()               # mixture weights
   comps = [stats.multivariate_t(m.x, m.hess_inv, df=4) for m in modes]   # heavy tails

Honest limits
---------------

Two. Modes found are modes **searched for**: an optimiser started nowhere near a component will never
report it, and in high dimensions dispersed starts cover the space poorly. And the approximation's
quality has no internal diagnostic — importance weights (with their :math:`\hat{k}`) give one from
outside.

Still, the construction is the conceptual bridge to what follows. Fitting the **best member of a chosen
family** to a posterior, by optimising a divergence rather than by curvature at a mode, is precisely
**variational inference** — and it is the workhorse for models too large to sample.
"""

CONTENT["Finding marginal posterior modes using EM"] = r"""
Modes of what, exactly?
-------------------------

In a hierarchical model the joint mode of :math:`(\theta, \phi)` — group parameters and hyperparameters
together — is often useless: it is the point where all :math:`\theta_j` equal :math:`\mu` and
:math:`\tau = 0`, the degenerate solution of the boundary lesson. What you usually want is the
**marginal** mode of the hyperparameters,

.. math::

   \hat{\phi} = \arg\max_{\phi} \; p(\phi \mid y)
              = \arg\max_{\phi} \int p(\theta, \phi \mid y) \, d\theta ,

with the nuisance :math:`\theta` **integrated out** rather than maximised over. The integral usually has
no closed form, and **EM** computes the maximiser without doing it directly.

The algorithm
---------------

Treat :math:`\theta` as **missing data**. Alternate:

* **E-step.** Given the current :math:`\phi^{(t)}`, form the expected complete-data log posterior,
  averaging over the conditional distribution of the missing parameters:

  .. math::

     Q\bigl(\phi \mid \phi^{(t)}\bigr) =
     \mathrm{E}_{\theta \mid \phi^{(t)}, y}\bigl[\log p(\theta, \phi \mid y)\bigr].

* **M-step.** Maximise :math:`Q` over :math:`\phi` to get :math:`\phi^{(t+1)}`.

Each iteration **cannot decrease** the marginal posterior density — the guarantee that makes EM stable
without a step size. It converges to a **local** mode, monotonically, from wherever it starts.

.. code-block:: python

   import numpy as np
   # hierarchical normal: E-step gives E[theta_j], var[theta_j]; M-step updates mu, tau
   for _ in range(n_iter):
       V = 1 / (1 / sigma**2 + 1 / tau**2)              # E-step: conditional moments
       Etheta = V * (ybar / sigma**2 + mu / tau**2)
       mu = Etheta.mean()                                # M-step: closed-form maximisers
       tau = np.sqrt(np.mean((Etheta - mu) ** 2 + V))    # note: +V, the E-step variance

That ``+ V`` is the whole point: EM adds back the **uncertainty** in :math:`\theta`, which a naive
"plug in the estimates and maximise" would discard, and which is why the resulting :math:`\tau` is not
biased toward zero.

Uses, and the ceiling
-----------------------

EM shines where the complete-data problem is easy: mixture models (the latent component labels are the
missing data), models with censoring, factor models, and hierarchical normal models. Variants extend it
— **ECM** for hard M-steps, **SEM** for standard errors, **MCEM** when the E-step needs simulation.

Its ceiling is inherent. EM returns a **point**, not a distribution: the curvature at the mode gives an
approximate covariance, but the uncertainty in :math:`\phi` is not propagated into inferences about
:math:`\theta`, which is the empirical-Bayes understatement flagged in Stage 5. EM is the right tool when
:math:`n` is large, the posterior is regular, and speed matters — and a **starting point** for full Bayes
otherwise.
"""


MINDMAP.update({
    "Finding posterior modes": [
        "Normal Approximations to the Posterior Distribution",
        "Boundary-avoiding priors for modal summaries",
        "Finding marginal posterior modes using EM",
        "Bayesian interpretations of other statistical methods",
    ],
    "Boundary-avoiding priors for modal summaries": [
        "Finding posterior modes", "Weakly Informative Priors for Variance Parameters",
        "Counterexamples to large-sample (asymptotic) Bayesian theorems",
        "Multivariate Normal with Unknown Mean and Variance",
    ],
    "Normal and related mixture approximations": [
        "Distributional approximations", "Finding posterior modes",
        "Variational inference", "Importance sampling",
    ],
    "Finding marginal posterior modes using EM": [
        "Finding posterior modes", "Averaging Over Nuisance Parameters",
        "Conditional and marginal posterior approximations",
        "Setting up and interpreting mixture models",
    ],
})


# ----------------------------------------------------------------------
# Part III / Stage 10 — Modal & Variational Approximation (cont.)
# ----------------------------------------------------------------------

CONTENT["Conditional and marginal posterior approximations"] = r"""
Approximate in stages
-----------------------

A hierarchical posterior :math:`p(\theta, \phi \mid y)` is hard as a whole and easy in pieces. Factor it,

.. math::

   p(\theta, \phi \mid y) = \underbrace{p(\phi \mid y)}_{\text{marginal, hard}}
                            \; \underbrace{p(\theta \mid \phi, y)}_{\text{conditional, often easy}} ,

and treat the two factors differently. The **conditional** is frequently a standard distribution — the
Gibbs conditionals of Stage 9 are exactly these. The **marginal** for the hyperparameters is
low-dimensional, so it can be approximated well, or evaluated on a grid.

The recipe
------------

Approximate :math:`p(\phi \mid y)` — by a normal at its mode, or on a grid — then draw:

1. draw :math:`\phi^{(s)}` from the approximate marginal;
2. draw :math:`\theta^{(s)} \sim p(\theta \mid \phi^{(s)}, y)`, **exactly**, from the conditional.

The result is approximate joint draws, and crucially they **propagate uncertainty in** :math:`\phi`, in
contrast to empirical Bayes, which fixes :math:`\hat{\phi}` and understates every interval. This is
precisely what the conjugate hierarchical model of Stage 5 did with its two-dimensional grid.

.. code-block:: python

   import numpy as np
   from scipy import stats
   # 1. approximate the low-dimensional marginal p(phi | y) on a grid
   logm = np.array([log_marginal(p, y) for p in grid])       # theta integrated out
   w = np.exp(logm - logm.max()); w /= w.sum()
   phi = np.random.choice(grid, size=4000, p=w)              # draws from the marginal
   # 2. exact conditional draws given each phi
   theta = stats.norm(cond_mean(phi, y), cond_sd(phi, y)).rvs()

The marginal is where the work is
-----------------------------------

Computing :math:`p(\phi \mid y) = \int p(\theta, \phi \mid y) \, d\theta` requires integrating the group
parameters out. Conjugacy does it in closed form; otherwise a **Laplace approximation of the inner
integral**, evaluated at each :math:`\phi`, is the standard device. That nested-Laplace idea, applied to
latent Gaussian models, is the engine of **INLA** — accurate, and far faster than MCMC for the model
class it covers.

Why it still matters
----------------------

Two reasons, both practical. The factorisation tells you **where the difficulty lives**: almost always in
the hyperparameters, whose posterior is the funnel-shaped, weakly identified part. And it explains the
family relationship among methods — EM maximises the marginal, empirical Bayes plugs in its maximiser,
this approach **integrates** over it, and full MCMC samples the joint. They differ only in how honestly
they treat :math:`p(\phi \mid y)`.
"""

CONTENT["Example: hierarchical normal model (continued)"] = r"""
The same model, approximated
------------------------------

Return to the hierarchical normal model that Stage 9 sampled with Gibbs, and apply the machinery of this
stage instead. It is a fair test: the answers are known, so every approximation can be scored.

.. math::

   y_{ij} \sim \mathrm{N}(\theta_j, \sigma^2), \qquad
   \theta_j \sim \mathrm{N}(\mu, \tau^2), \qquad \phi = (\mu, \sigma, \tau).

Why the joint mode fails
--------------------------

Maximise :math:`p(\theta, \phi \mid y)` jointly and the optimiser walks straight to the degenerate point:
every :math:`\theta_j` equal to :math:`\mu`, and :math:`\tau = 0`. The joint density there is
**unbounded** — the likelihood of the group parameters concentrates without limit as the population
variance shrinks. The joint mode is not merely a poor summary; it does not exist as a finite maximum.

Marginal mode, via EM
-----------------------

Integrate the :math:`\theta_j` out and maximise the marginal :math:`p(\phi \mid y)` instead. EM does this
without doing the integral: the E-step supplies conditional means **and variances** of the
:math:`\theta_j`, and the M-step's update for :math:`\tau^2` includes those variances, so it cannot
collapse to zero. The degeneracy disappears the moment the nuisance parameters are averaged rather than
maximised.

.. code-block:: python

   import numpy as np
   for _ in range(200):
       V = 1 / (n_j / sigma**2 + 1 / tau**2)                  # E-step
       Etheta = V * (n_j * ybar_j / sigma**2 + mu / tau**2)
       mu = Etheta.mean()                                     # M-step
       tau = np.sqrt(np.mean((Etheta - mu) ** 2 + V))         # + V prevents collapse
   # then: Laplace around (mu, log sigma, log tau) for approximate uncertainty

Then the conditional
----------------------

Having a marginal for :math:`\phi` — from EM plus curvature, or from a grid — the previous lesson's
recipe completes the picture: draw :math:`\phi`, then draw each :math:`\theta_j` **exactly** from its
normal conditional. The group-level answers inherit the hyperparameter uncertainty, which is what
distinguishes this from empirical Bayes.

What the comparison teaches
-----------------------------

Against long-run MCMC, the modal approximation is accurate for :math:`\mu` and for the :math:`\theta_j`
in data-rich groups, and least accurate for :math:`\tau` — the parameter whose posterior is skewed,
bounded below, and often heaped near zero. That is the general pattern: **approximations fail on the
variance parameters of hierarchies**, which are precisely the parameters that decide how much pooling
occurs. Use modal methods for speed and starting values; check the conclusions that hinge on
:math:`\tau` against a sampler.
"""

CONTENT["Variational inference"] = r"""
Turn inference into optimisation
----------------------------------

MCMC samples the posterior; **variational inference** *approximates* it, by choosing the member of a
tractable family :math:`\mathcal{Q}` that is closest to it. Inference becomes optimisation, which scales
to models and datasets where sampling is hopeless.

The ELBO
----------

Closeness is measured by Kullback–Leibler divergence, and the decomposition is exact:

.. math::

   \log p(y) = \underbrace{\mathrm{E}_{q}\bigl[\log p(\theta, y)\bigr] - \mathrm{E}_{q}\bigl[\log q(\theta)\bigr]}_{\text{ELBO}(q)}
             \;+\; \mathrm{KL}\bigl(q(\theta) \,\|\, p(\theta \mid y)\bigr) .

Since :math:`\log p(y)` is fixed and the KL term is non-negative, **maximising the ELBO minimises the
divergence** — and the intractable evidence never has to be computed. The ELBO's two pieces read
naturally: fit the joint density well, but keep the approximation's entropy high.

Families, and the mean-field default
--------------------------------------

The commonest choice factorises the posterior across parameters — the **mean-field** family,
:math:`q(\theta) = \prod_k q_k(\theta_k)` — usually with Gaussian factors on an unconstrained scale.
**ADVI** automates the whole thing: transform constrained parameters, fit a Gaussian, optimise the ELBO
by stochastic gradients using automatic differentiation.

.. code-block:: python

   import pymc as pm
   with model:
       approx = pm.fit(n=30_000, method="advi")     # optimise the ELBO
       idata = approx.sample(2000)                  # draws from q, not from the posterior

The direction of the KL
-------------------------

Everything that goes wrong follows from **which** KL is minimised. :math:`\mathrm{KL}(q \| p)` is
**mode-seeking**: it heavily penalises putting :math:`q` mass where :math:`p` has none, and barely
penalises the reverse. So :math:`q` prefers to cover one mode and **underestimate variance** and tails.
A mean-field family compounds this: forcing independence across correlated parameters shrinks the
approximation's spread further.

The consequences are systematic, not random. VI point estimates are often excellent; VI **uncertainties
are usually too narrow**; and multimodal posteriors are typically represented by one mode. In a
hierarchical model, VI will underestimate :math:`\tau`'s uncertainty exactly where it matters.

Check it, always
------------------

Unlike MCMC, VI has no :math:`\hat{R}` — the optimiser converging says nothing about whether
:math:`\mathcal{Q}` **contains** anything close to the posterior. Diagnostics exist and should be used:
**PSIS** importance-reweighting of the VI draws (whose :math:`\hat{k}` flags a bad approximation), and
**simulation-based calibration**. Use VI for exploration, for very large models, and as an initialiser —
and never report a VI interval without checking it.
"""

CONTENT["Expectation propagation"] = r"""
Approximate one factor at a time
----------------------------------

The posterior is a **product** of factors — a prior and one likelihood term per observation (or per
group):

.. math::

   p(\theta \mid y) \;\propto\; p(\theta) \prod_{i=1}^{n} p(y_i \mid \theta).

**Expectation propagation** replaces each awkward factor :math:`p(y_i \mid \theta)` by a tractable
**site approximation** :math:`\tilde{t}_i(\theta)` — typically an unnormalised Gaussian — so that the
product is Gaussian and every quantity is available in closed form.

The iteration
---------------

Sites are refined one at a time. To update site :math:`i`:

1. form the **cavity** distribution by removing that site from the current approximation,
   :math:`q_{-i} \propto q / \tilde{t}_i`;
2. form the **tilted** distribution :math:`q_{-i}(\theta) \, p(y_i \mid \theta)` — the cavity times the
   *true* factor;
3. **match moments**: choose a Gaussian with the same mean and variance as the tilted distribution;
4. divide the cavity back out to recover the new :math:`\tilde{t}_i`.

Each moment-matching step is the local minimiser of :math:`\mathrm{KL}(p \| q)` — the **opposite**
direction from variational inference. That direction is **mass-covering**: it penalises :math:`q` for
missing regions where :math:`p` has mass, so EP tends to produce approximations that are **wider**, with
better-calibrated variances than mean-field VI.

.. code-block:: python

   # sketch: Gaussian sites, natural parameters, one sweep
   for i in range(n):
       cav_prec = q_prec - site_prec[i]                    # 1. cavity
       cav_mean = (q_prec * q_mean - site_prec[i] * site_mean[i]) / cav_prec
       m, v = tilted_moments(cav_mean, cav_prec, y[i])     # 2-3. moment match
       new_prec = 1 / v - cav_prec                          # 4. new site
       site_prec[i], site_mean[i] = new_prec, (m / v - cav_mean * cav_prec) / new_prec
       q_prec, q_mean = 1 / v, m                            # refresh global approximation

Strengths and cautions
------------------------

EP is often strikingly accurate for **latent Gaussian** models — Gaussian-process classification, probit
regression — and it is naturally parallel across sites. It also yields an estimate of the **marginal
likelihood** as a by-product, which VI's ELBO only bounds.

But EP carries no guarantees. It **need not converge** — there is no objective function being decreased,
so cycles are possible and damping is often required. Cavity variances can go **negative**, breaking the
Gaussian assumption. And, like all approximations in this stage, it is silent about its own quality. The
place of EP is alongside VI and Laplace: fast, sometimes excellent, always to be **checked against a
sampler** on a case you can afford to sample.
"""


MINDMAP.update({
    "Conditional and marginal posterior approximations": [
        "Averaging Over Nuisance Parameters", "Finding marginal posterior modes using EM",
        "Example: hierarchical normal model (continued)", "Other approximations",
    ],
    "Example: hierarchical normal model (continued)": [
        "Example: hierarchical normal model", "Finding marginal posterior modes using EM",
        "Conditional and marginal posterior approximations",
        "Boundary-avoiding priors for modal summaries",
    ],
    "Variational inference": [
        "Distributional approximations", "Expectation propagation",
        "Normal and related mixture approximations", "Importance sampling",
    ],
    "Expectation propagation": [
        "Variational inference", "Other approximations",
        "Normal and related mixture approximations", "Gaussian process regression",
    ],
})


# ----------------------------------------------------------------------
# Part III / Stage 10 — Modal & Variational Approximation  [completes Part III]
# ----------------------------------------------------------------------

CONTENT["Other approximations"] = r"""
The wider family
------------------

Laplace, EM, variational inference and expectation propagation are the landmarks; several other methods
occupy the space between them, each trading accuracy for speed in a different currency.

INLA
------

**Integrated nested Laplace approximation** (Rue, Martino and Chopin) is the standout for a specific,
large class: **latent Gaussian models**, in which observations are conditionally independent given a
latent Gaussian field, whose precision matrix depends on a few hyperparameters :math:`\theta`. That
class covers most hierarchical regressions, spatial and spatio-temporal models, and smoothers.

INLA implements the conditional/marginal factorisation of two lessons ago, twice over. It approximates
the hyperparameter marginal by evaluating the joint against a Gaussian approximation of the latent field
at its mode,

.. math::

   \tilde{p}(\theta \mid y) \;\propto\;
   \left. \frac{p(x, \theta, y)}{\tilde{p}_G(x \mid \theta, y)} \right|_{x = x^{*}(\theta)},

then recovers each latent marginal by **numerical integration** over a small grid of :math:`\theta`
values, :math:`p(x_i \mid y) \approx \sum_k p(x_i \mid y, \theta_k) \, \tilde{p}(\theta_k \mid y) \,
\Delta_k`. Being deterministic, it has **no mixing to diagnose** and no chains to run — minutes where
MCMC takes hours. Its price is the model class: leave latent-Gaussian territory and INLA does not apply.

Others in brief
-----------------

* **Laplace / nested Laplace** — the building block of INLA; excellent when the integrand is unimodal
  and smooth.
* **Pseudo-marginal and particle methods** — replace an intractable likelihood with an **unbiased
  estimate** inside the Metropolis ratio; remarkably, the chain still targets the exact posterior.
* **Approximate Bayesian computation (ABC)** — when the likelihood cannot even be evaluated but data can
  be **simulated**: accept parameter draws whose simulated summaries land near the observed ones.
* **Stochastic-gradient MCMC** — subsample the data per step; scalable, biased, and increasingly
  understood.

Choosing
----------

The decision rests on two questions. **Is your model in a class with a specialised method?** (Latent
Gaussian → INLA; simulator-only → ABC.) And **what will the answer be used for?** Point estimates
tolerate crude approximations; tail probabilities and hierarchical variance parameters do not.

.. code-block:: python

   import arviz as az
   # whatever the approximation, check it from outside:
   #   PSIS-reweight the approximate draws toward the true posterior
   logw = log_posterior(draws) - approx.logpdf(draws)
   #   k_hat < 0.7 -> the approximation is usable; larger -> do not trust it
   az.psislw(logw)

That last line is the discipline of this whole stage. Every approximation here is silent about its own
error; **importance-reweighting supplies the missing diagnostic**, and where the model is small enough,
so does a run of the sampler you were trying to avoid.
"""

CONTENT["Unknown normalizing factors"] = r"""
When the likelihood has a constant you cannot compute
-------------------------------------------------------

MCMC is celebrated for not needing the posterior's normalising constant, since it cancels in the
Metropolis ratio. But some **likelihoods** are themselves unnormalised:

.. math::

   p(y \mid \theta) = \frac{q(y \mid \theta)}{Z(\theta)}, \qquad
   Z(\theta) = \int q(y \mid \theta) \, dy ,

with :math:`Z(\theta)` intractable — a sum over :math:`2^{n}` configurations for an Ising model or an
undirected graphical model, an integral with no closed form for a spatial point process. Now the
constant **depends on** :math:`\theta` and therefore does **not** cancel:

.. math::

   r = \frac{q(y \mid \theta^{*}) \, p(\theta^{*})}{q(y \mid \theta^{(t-1)}) \, p(\theta^{(t-1)})}
       \cdot \underbrace{\frac{Z(\theta^{(t-1)})}{Z(\theta^{*})}}_{\text{unknown}} .

Such posteriors are called **doubly intractable**: the evidence :math:`p(y)` is intractable, as always,
*and* so is :math:`Z(\theta)` at every step.

The exchange algorithm
------------------------

The elegant solution introduces an **auxiliary dataset**. Propose :math:`\theta^{*}`, then simulate
fresh data :math:`y' \sim p(\cdot \mid \theta^{*})` from the model itself. In the acceptance ratio for
the augmented target, the auxiliary term contributes :math:`Z(\theta^{*}) / Z(\theta^{(t-1)})`, which
**cancels** the offending factor exactly:

.. math::

   r_{\text{ex}} = \frac{q(y \mid \theta^{*}) \, p(\theta^{*}) \, q(y' \mid \theta^{(t-1)})}
                        {q(y \mid \theta^{(t-1)}) \, p(\theta^{(t-1)}) \, q(y' \mid \theta^{*})} .

Every term is computable. Building on Møller and colleagues' auxiliary-variable scheme, Murray,
Ghahramani and MacKay's **exchange algorithm** is asymptotically exact — its only demand is the ability
to draw **exact** samples from the likelihood, which perfect-sampling algorithms provide for several
model classes.

Estimating the constant instead
---------------------------------

Where exact sampling is impossible, estimate :math:`Z(\theta)` — or, better, **ratios** of it.
Gelman and Meng's unifying account runs from **importance sampling** through **bridge sampling** to
**path sampling**, the last computing :math:`\log Z(\theta_1) - \log Z(\theta_0)` as an integral of an
expected score along a path connecting the two parameter values. A separate route is **pseudo-marginal**
MCMC: substitute an **unbiased estimate** of the likelihood into the Metropolis ratio, and — the
surprising theorem — the chain still targets the **exact** posterior, at the cost of extra variance.

.. code-block:: python

   import numpy as np
   # exchange step: cancel Z(theta) using auxiliary data drawn from the model
   th_prop = propose(th)
   y_aux = simulate_exactly(th_prop)                       # exact draw from p(. | th_prop)
   log_r = (log_q(y, th_prop) - log_q(y, th)
            + log_q(y_aux, th) - log_q(y_aux, th_prop)     # <- the Z's cancel here
            + log_prior(th_prop) - log_prior(th))
   th = th_prop if np.log(np.random.rand()) < log_r else th

Why it closes Part III
------------------------

This is where the computational story reaches its edge. Parts I and II assumed a likelihood you could
evaluate; Part III has been about integrating one. Models whose likelihood you cannot even *evaluate*
demand a different bargain — auxiliary variables, unbiased estimators, or simulation-based inference —
and each buys correctness with either exact sampling or extra Monte Carlo noise. With the machinery
established, Part IV returns to modelling: regression, and the structure that makes models useful.
"""


# ----------------------------------------------------------------------
# Part IV / Stage 11 — Regression Foundations
# ----------------------------------------------------------------------

CONTENT["Conditional modeling"] = r"""
Model y given x, not (x, y)
-----------------------------

Almost every applied model is a **regression**: a distribution for an outcome :math:`y` **conditional
on** predictors :math:`x`. The move is so routine that its justification goes unexamined. Why is it
legitimate to model :math:`p(y \mid x, \theta)` and simply **ignore** the distribution of :math:`x`?

The factorisation
-------------------

Write the joint model for everything observed, with parameters :math:`\theta` for the conditional and
:math:`\psi` for the predictors:

.. math::

   p(x, y \mid \theta, \psi) = \underbrace{p(y \mid x, \theta)}_{\text{what you care about}}
                               \; \underbrace{p(x \mid \psi)}_{\text{nuisance}} .

If :math:`\theta` and :math:`\psi` are **distinct** and **a priori independent**, then the second factor
contributes nothing to :math:`p(\theta \mid x, y)`: it cancels in the normalisation. Conditional
modelling is therefore not an approximation but an exact consequence — under exactly the conditions of
the **ignorability** lesson in Stage 7, transposed from data collection to predictors.

When it fails
---------------

The conditions are not automatic, and each failure is a known pathology:

* **Measurement error in** :math:`x` — the observed predictor is not the one the outcome responds to, so
  the parameters are linked and :math:`p(x \mid \psi)` re-enters the likelihood.
* **Missing predictors** — imputing them requires a model for :math:`x`.
* **Selection on** :math:`x` **or** :math:`y` — if inclusion depends on the outcome, the conditional
  likelihood is wrong (Stage 7 again).
* **Shared parameters** — a design in which :math:`\psi` informs :math:`\theta` breaks distinctness.

What conditioning buys
------------------------

You are relieved of modelling the predictors' joint distribution, which is usually complicated,
high-dimensional, and irrelevant to the question. You may then choose :math:`p(y \mid x, \theta)` to
suit the outcome — normal, binomial, Poisson (Stage 13) — and the same posterior machinery applies.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])    # predictors standardised
       sigma = pm.HalfNormal("sigma", 1)
       pm.Normal("y", X @ beta, sigma, observed=y)           # x is conditioned on, not modelled
       idata = pm.sample()

One caution to carry into Part IV: conditioning on :math:`x` makes the model **agnostic about causation**.
:math:`p(y \mid x)` is a statement about association; whether the coefficient is a causal effect depends
on the data-collection lessons, not on the regression.
"""

CONTENT["Bayesian analysis of classical regression"] = r"""
The normal linear model
-------------------------

The workhorse of applied statistics, read Bayesianly. With :math:`n` observations, :math:`k` predictors
in a matrix :math:`X`, coefficients :math:`\beta` and noise variance :math:`\sigma^2`:

.. math::

   y \mid \beta, \sigma^2, X \sim \mathrm{N}(X\beta, \; \sigma^2 I).

Everything from Stage 3's normal model carries over, with :math:`X\beta` in place of a scalar mean.

Noninformative prior
----------------------

Take the standard :math:`p(\beta, \sigma^2) \propto 1/\sigma^2`. The posterior factors exactly as before
— conditional for the coefficients, marginal for the variance:

.. math::

   \beta \mid \sigma^2, y \sim \mathrm{N}\bigl(\hat{\beta}, \; \sigma^2 (X^{\top} X)^{-1}\bigr),
   \qquad
   \sigma^2 \mid y \sim \text{Inv-}\chi^2\bigl(n - k, \; s^2\bigr),

where :math:`\hat{\beta} = (X^{\top}X)^{-1} X^{\top} y` is the **least-squares estimate** and
:math:`s^2 = \frac{1}{n-k}(y - X\hat{\beta})^{\top}(y - X\hat{\beta})`. Marginalising :math:`\sigma^2`
gives a **multivariate** :math:`t` for :math:`\beta` — the heavy tails again the price of not knowing the
noise scale.

The classical results, reinterpreted
--------------------------------------

The posterior mean of :math:`\beta` **is** the least-squares estimate; the posterior covariance **is**
:math:`s^2 (X^{\top}X)^{-1}`; the marginal for each coefficient **is** a :math:`t` on :math:`n - k`
degrees of freedom. Every number in a regression printout is recovered — but reinterpreted: the interval
is now a probability statement about :math:`\beta`, and the :math:`t` arises by **integration** rather
than by a sampling-distribution argument.

Sampling without MCMC
-----------------------

Because the factorisation is exact, joint draws are direct — no chain, no diagnostics:

.. code-block:: python

   import numpy as np
   from scipy import stats
   n, k = X.shape
   XtX_inv = np.linalg.inv(X.T @ X)
   beta_hat = XtX_inv @ X.T @ y
   s2 = ((y - X @ beta_hat) ** 2).sum() / (n - k)

   sigma2 = (n - k) * s2 / stats.chi2(n - k).rvs(4000)                   # marginal draw
   beta = np.array([stats.multivariate_normal(beta_hat, s * XtX_inv).rvs()
                    for s in sigma2])                                     # conditional draw
   np.percentile(beta[:, 1], [2.5, 97.5])       # posterior interval for the 2nd coefficient

What the closed form hides
----------------------------

Three assumptions are doing quiet work: **normal errors** (relaxed in Stage 14), **constant variance and
independence** (relaxed later in this stage), and an :math:`X^{\top}X` that is **invertible** and
well-conditioned. When :math:`k` approaches :math:`n`, or predictors are collinear, the noninformative
posterior is diffuse or improper — and the remedy is a **prior**, which is exactly regularisation.
"""


MINDMAP.update({
    "Other approximations": [
        "Variational inference", "Expectation propagation",
        "Conditional and marginal posterior approximations", "Unknown normalizing factors",
    ],
    "Unknown normalizing factors": [
        "Other approximations", "Metropolis and Metropolis-Hastings algorithms",
        "Model comparison using Bayes factors", "Importance sampling",
    ],
    "Conditional modeling": [
        "Bayesian analysis of classical regression", "Goals of regression analysis",
        "Data-collection models and ignorability", "Standard generalized linear model likelihoods",
    ],
    "Bayesian analysis of classical regression": [
        "Conditional modeling", "Normal Data with a Noninformative Prior Distribution",
        "Regularization and dimension reduction", "Goals of regression analysis",
    ],
})


# ----------------------------------------------------------------------
# Part IV / Stage 11 — Regression Foundations (cont.)
# ----------------------------------------------------------------------

CONTENT["Regression for causal inference: incumbency and voting"] = r"""
Does holding office help you win?
-----------------------------------

The **incumbency advantage** is the extra vote share a candidate receives merely by being the sitting
representative. It cannot be randomised — nobody assigns incumbency by coin flip — so the question is
observational, and regression must carry the causal weight. Stage 7's warnings apply in full.

Why the naive measures fail
-----------------------------

Two classical estimators look reasonable and are **biased**. **Sophomore surge** compares a legislator's
vote share in their second election (as incumbent) with their first (as challenger). **Retirement
slump** compares a party's share before and after its incumbent retires. Each conflates the incumbency
effect with the fact that **incumbents are not a random sample**: they are the candidates who won, in
districts that favoured them. Gelman and King showed the bias directly — and showed that the information
in these comparisons, placed in a **regression framework**, yields an unbiased estimate.

The model
-----------

Model district :math:`i`'s vote share :math:`v_{it}` in election year :math:`t`, controlling for the
district's **previous** vote share (a proxy for partisan strength) and party, with an indicator
:math:`I_{it}` for whether an incumbent is running:

.. math::

   v_{it} = \beta_0 + \beta_1 \, v_{i,t-1} + \beta_2 \, P_{it} + \underbrace{\gamma}_{\text{incumbency}} I_{it}
            + \epsilon_{it} .

The coefficient :math:`\gamma` is the estimand. Conditioning on the lagged vote is what removes the
selection: districts where incumbents run are compared with **similar** districts holding open seats.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       b = pm.Normal("b", 0, 0.5, shape=3)
       gamma = pm.Normal("gamma", 0, 0.1)               # incumbency effect, on vote-share scale
       mu = b[0] + b[1] * v_lag + b[2] * party + gamma * incumbent
       pm.Normal("v", mu, pm.HalfNormal("sigma", 0.1), observed=v)
       idata = pm.sample()
   # posterior for gamma, by decade, shows the advantage growing over the century

What makes it causal, and what does not
-----------------------------------------

The regression **is** the causal claim only under the no-unmeasured-confounding assumption of Stage 7:
that, given the lagged vote and party, whether an incumbent runs is as good as random. That is an
assumption about **why** incumbents retire — health, scandal, redistricting, or anticipated defeat — and
the last of these breaks it. If incumbents strategically retire when they expect to lose, retirement is
informative about :math:`\epsilon_{it}`, and :math:`\gamma` is biased upward.

So the honest report has three parts: the estimate, the assumption it rests on, and a **sensitivity
analysis** for how strong a confounder would need to be to erase it. Regression supplies the arithmetic;
the design supplies the license.
"""

CONTENT["Goals of regression analysis"] = r"""
Three questions, one equation
-------------------------------

The same fitted regression can serve three purposes, and confusing them is the most common error in
applied statistics. Ask which one you are pursuing **before** choosing predictors, priors, or a way to
evaluate the fit.

Prediction
------------

You want :math:`p(\tilde{y} \mid \tilde{x}, y)` to be accurate for new cases. Nothing else matters:
predictors may be proxies, collinear, or scientifically meaningless. **Include** anything that predicts,
regularise heavily, and evaluate by **out-of-sample predictive accuracy** (elpd/LOO, Stage 6).
Interpreting an individual coefficient is beside the point — and with collinear predictors, impossible.

Description and explanation
-----------------------------

You want to characterise **associations** in the population: how does income vary with education, given
age? Here coefficients are the output, so their **interpretation must be defensible**: they are
comparisons between units that differ in one predictor and agree on the others. No causal claim is
implied. Adjust for what makes the comparison meaningful, and beware that "controlling for" a variable
changes the meaning of every other coefficient.

Causal inference
------------------

You want the effect of **intervening** on a predictor. This is a different quantity — it concerns
potential outcomes, not conditional expectations — and it demands the machinery of Stage 7: a design, or
an assumption that assignment is ignorable given the covariates. Only some coefficients in a regression
are causal, and typically only the one you designed for.

.. code-block:: python

   import arviz as az
   az.loo(idata)                  # prediction: which model forecasts best?
   az.summary(idata, var_names=["beta"])   # description: which comparisons, with what uncertainty?
   # causation: the number above is causal only if the DESIGN says so

Choices that depend on the goal
---------------------------------

The purpose determines everything downstream:

* **Which predictors.** Prediction: everything useful. Description: what makes comparisons sensible.
  Causation: confounders in, **mediators and colliders out** — adjusting for a variable on the causal
  path removes part of the effect, and adjusting for a collider **creates** bias.
* **Which prior.** Prediction tolerates strong shrinkage; causal estimands usually want a weakly
  informative prior on the coefficient of interest so the data speak.
* **Which check.** Prediction: LOO. Description: posterior predictive checks. Causation: sensitivity to
  unmeasured confounding.

A model excellent for one goal can be worthless for another. The equation does not know which question
you are asking; **you must**.
"""

CONTENT["Assembling the matrix of explanatory variables"] = r"""
The model is in the design matrix
-----------------------------------

A "linear" model is linear **in the coefficients**, not in the predictors. Almost all of the modelling
happens when you build :math:`X` — coding categories, transforming scales, forming interactions. This is
where subject knowledge enters, and where most fitted models go wrong.

Categorical predictors
------------------------

A factor with :math:`K` levels becomes :math:`K - 1` indicator columns plus the intercept (else
:math:`X^{\top} X` is singular). But the classical choice of a **baseline** level is arbitrary and makes
coefficients hard to interpret. When :math:`K` is large — counties, schools, products — the better move
is to treat the levels as **exchangeable** and give them a hierarchical prior, which is exactly the
varying-intercept model of Stage 12. Indicator coding is the :math:`\tau \to \infty` special case.

Transformations
-----------------

Choose the scale on which the relationship is **plausibly linear and the errors plausibly symmetric**:

* **Logarithms** for positive, right-skewed quantities (income, concentration, counts). A log–log
  coefficient is an **elasticity**; a log–linear one is a proportional change per unit.
* **Standardising** predictors, :math:`(x - \bar{x}) / \mathrm{sd}(x)` — this is what makes a
  :math:`\mathrm{N}(0, 2.5^2)` prior weakly informative rather than arbitrary, and it decorrelates the
  intercept from the slopes.
* **Splines** where the functional form is unknown (Stage 15).

Interactions
--------------

An interaction says the effect of one predictor **depends on** another. Include one when theory or plots
suggest it, and **always with its main effects** — a model with :math:`x_1 x_2` but not :math:`x_1` makes
a claim about the origin that is almost never intended. **Centre** the components first, or the main
effects become conditional on :math:`x = 0`, a value that may not exist.

.. code-block:: python

   import numpy as np, pymc as pm
   x1c = (x1 - x1.mean()) / x1.std()                  # centre and scale
   x2c = (x2 - x2.mean()) / x2.std()
   X = np.column_stack([np.ones_like(x1c), x1c, x2c, x1c * x2c])   # interaction + main effects
   with pm.Model():
       beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])   # prior now means something
       pm.Normal("y", X @ beta, pm.HalfNormal("s", 1), observed=y)

Two disciplines
-----------------

Choose the design matrix from **knowledge**, not from a search over :math:`p`-values: stepwise selection
invalidates every inference computed afterwards. And **check the implications**: simulate from the prior
predictive to see what data your coding and priors imply, before you look at the outcome. A design
matrix that yields absurd prior predictions is a modelling error, not a detail.
"""

CONTENT["Regularization and dimension reduction"] = r"""
When there are too many predictors
------------------------------------

With :math:`k` comparable to or larger than :math:`n`, least squares breaks: :math:`X^{\top}X` is
singular or nearly so, coefficients are wild, and the noninformative posterior is diffuse or improper.
The Bayesian answer is not to delete predictors but to **shrink** them — with a prior that expresses what
you believe about their sizes.

The ridge and lasso, and their limitation
-------------------------------------------

From Stage 4: a :math:`\mathrm{N}(0, \tau^2)` prior on each coefficient gives **ridge** at the mode; a
Laplace prior gives the **lasso**. Both apply the **same** amount of shrinkage to every coefficient,
governed by one global scale. That is exactly wrong when the truth is **sparse**: a single global scale
must either shrink the null coefficients enough (and thereby crush the real signals) or preserve the
signals (and leave noise unshrunk).

Global–local shrinkage: the horseshoe
---------------------------------------

The fix is a scale **per coefficient**, drawn from a heavy-tailed distribution. The **horseshoe** prior
of Carvalho, Polson and Scott:

.. math::

   \beta_j \mid \lambda_j, \tau \sim \mathrm{N}(0, \; \tau^2 \lambda_j^2), \qquad
   \lambda_j \sim \mathrm{C}^{+}(0, 1), \qquad \tau \sim \mathrm{C}^{+}(0, 1).

The **global** scale :math:`\tau` pulls everything toward zero; the **local** scales :math:`\lambda_j`,
with their Cauchy tails, let a genuinely large coefficient **escape** the shrinkage entirely. The result
is aggressive shrinkage of noise and near-zero shrinkage of signal — behaviour the lasso cannot achieve,
and achieved **continuously**, without the discrete mixture of spike-and-slab.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       tau = pm.HalfCauchy("tau", 1)                       # global
       lam = pm.HalfCauchy("lam", 1, shape=k)              # local, heavy-tailed
       z = pm.Normal("z", 0, 1, shape=k)
       beta = pm.Deterministic("beta", z * lam * tau)      # non-centred: the funnel again
       pm.Normal("y", X @ beta, pm.HalfNormal("s", 1), observed=y)

The horseshoe's own pathology
-------------------------------

Those Cauchy tails create an **extreme funnel** in :math:`(\tau, \lambda, \beta)` — the geometry of Stage
9, worse — and NUTS reports divergences even on simple problems. Piironen and Vehtari's **regularized
horseshoe** repairs it: a slab caps how large the escaped coefficients may be, and the prior on
:math:`\tau` is set from a **guess at the number of nonzero coefficients**, turning a vague default into
a statement of sparsity you can defend.

Alternatives and the caveat
-----------------------------

Principal components and other **dimension reduction** replace predictors with fewer combinations,
buying stability with interpretability. Whatever the route, one caution: **regularisation is a prior**,
and a prior distorts causal estimands. Shrink the nuisance coefficients; leave the effect you came to
measure weakly informative, so the data — not the penalty — determine it.
"""


MINDMAP.update({
    "Regression for causal inference: incumbency and voting": [
        "Observational studies", "Goals of regression analysis",
        "Sensitivity and the role of randomization",
        "Bayesian analysis of classical regression",
    ],
    "Goals of regression analysis": [
        "Conditional modeling", "Regression for causal inference: incumbency and voting",
        "Assembling the matrix of explanatory variables",
        "Model comparison based on predictive performance",
    ],
    "Assembling the matrix of explanatory variables": [
        "Goals of regression analysis", "Regularization and dimension reduction",
        "Splines and weighted sums of basis functions", "Varying intercepts and slopes",
    ],
    "Regularization and dimension reduction": [
        "Bayesian interpretations of other statistical methods",
        "Assembling the matrix of explanatory variables",
        "Weakly Informative Prior Distributions", "Basis selection and shrinkage of coe\ufb03cients",
    ],
})


# ----------------------------------------------------------------------
# Part IV / Stage 11 — Regression Foundations  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Unequal variances and correlations"] = r"""
When the errors are not i.i.d.
--------------------------------

Classical regression assumes :math:`y \sim \mathrm{N}(X\beta, \sigma^2 I)` — errors with **equal
variance** and **no correlation**. Real errors routinely violate both: measurements of differing
precision, variance that grows with the mean, observations clustered in time or space. The general
normal linear model replaces :math:`\sigma^2 I` with a full covariance :math:`\Sigma`:

.. math::

   y \mid \beta, \Sigma \sim \mathrm{N}(X\beta, \; \Sigma).

Known :math:`\Sigma`: reduce to the standard case
---------------------------------------------------

If :math:`\Sigma` is known, a single transformation recovers everything. Factor :math:`\Sigma = L
L^{\top}` (Cholesky) and multiply through by :math:`L^{-1}`:

.. math::

   \underbrace{L^{-1} y}_{y^{*}} = \underbrace{L^{-1} X}_{X^{*}} \beta + \epsilon^{*},
   \qquad \epsilon^{*} \sim \mathrm{N}(0, I).

The transformed model is ordinary regression, so its posterior mean is the **generalized least squares**
estimate :math:`\hat{\beta} = (X^{\top}\Sigma^{-1}X)^{-1} X^{\top}\Sigma^{-1}y`. The **diagonal** case —
uncorrelated but unequal variances, :math:`\Sigma = \mathrm{diag}(\sigma_i^2)` — is **weighted least
squares**, with each observation weighted by its precision :math:`1/\sigma_i^2`. Whitening and weighting
are the same operation, one general and one diagonal.

.. code-block:: python

   import numpy as np
   L = np.linalg.cholesky(Sigma)                    # Sigma = L L^T
   Xs = np.linalg.solve(L, X)                        # L^{-1} X  (whitened design)
   ys = np.linalg.solve(L, y)                        # L^{-1} y
   beta_gls = np.linalg.lstsq(Xs, ys, rcond=None)[0] # OLS on whitened data == GLS

Unknown :math:`\Sigma`: model it
----------------------------------

Usually :math:`\Sigma` is unknown, and here the Bayesian treatment separates cleanly from the classical
one: rather than plug in an estimate, give :math:`\Sigma` a **structure with its own parameters** and
infer everything jointly.

* **Variances a function of a predictor** — :math:`\log \sigma_i = \gamma_0 + \gamma_1 z_i`, letting the
  data learn how the noise scales.
* **Correlation from structure** — an AR(1) covariance for time series, a distance-decaying kernel for
  spatial data (Stage 15), a compound-symmetry block for grouped data.
* **Hierarchical variances** — a prior across the :math:`\sigma_i` when there are many.

The posterior then propagates uncertainty in :math:`\Sigma` into the coefficients — which plug-in GLS
ignores, understating the intervals exactly as empirical Bayes did.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])
       g = pm.Normal("g", 0, 1, shape=2)
       sigma = pm.Deterministic("sigma", pm.math.exp(g[0] + g[1] * z))   # variance model
       pm.Normal("y", X @ beta, sigma, observed=y)                       # heteroscedastic

Why it matters
----------------

Ignoring unequal variance does not bias :math:`\hat{\beta}`, but it makes the standard errors **wrong**
and the estimator **inefficient** — the tidy printout is confidently miscalibrated. Correlated errors are
worse: treating clustered or serially dependent observations as independent **overstates** the effective
sample size, producing intervals far too narrow. Modelling :math:`\Sigma` is what makes the uncertainty
honest, and it is the doorway to the mixed models and Gaussian processes ahead.
"""

CONTENT["Including numerical prior information"] = r"""
Quantitative prior knowledge
------------------------------

Regression rarely starts from ignorance. A previous study estimated a similar coefficient; theory bounds
an elasticity; physics fixes a sign or a scale. Bayesian regression **incorporates** such knowledge
directly, through informative priors on :math:`\beta` and :math:`\sigma` — turning "we already know
roughly this" from a footnote into part of the model.

Where the numbers come from
-----------------------------

Three honest sources, in decreasing order of comfort:

* **A previous analysis.** If an earlier study reported :math:`\hat{\beta}` with standard error
  :math:`s`, a prior :math:`\beta \sim \mathrm{N}(\hat{\beta}, s^2)` carries that result forward — the
  Bayesian version of accumulating evidence across studies, and the honest alternative to ignoring prior
  work or double-counting it.
* **Substantive constraints.** A rate must be positive; a probability lies in :math:`[0, 1]`; a dose
  response is monotone. Encode these as bounded or ordered priors so the posterior cannot violate what
  is known a priori.
* **Elicited ranges.** An expert's plausible interval becomes a prior with roughly that spread —
  useful, but the softest source, and the one to report most carefully.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # coefficient centred on a previous study's estimate, with its uncertainty
       beta_prev = pm.Normal("beta_prev", mu=0.32, sigma=0.08)   # from prior work
       # a coefficient known to be non-negative on theoretical grounds
       beta_pos = pm.HalfNormal("beta_pos", sigma=1.0)
       mu = alpha + beta_prev * x1 + beta_pos * x2
       pm.Normal("y", mu, pm.HalfNormal("s", 1), observed=y)

Calibrating strength
----------------------

An informative prior is a claim about how much you know, so its **width** must be defensible. Too tight
and it overrides data it should merely inform; too wide and it contributes nothing. Standardising the
predictors first makes the width interpretable — a :math:`\mathrm{N}(0, 1)` prior on a standardised
coefficient says the effect of a one-SD change is probably within a couple of SDs of the outcome. The
prior predictive check is the test: simulate data from the prior alone and see whether it spans the
plausible and excludes the absurd.

The discipline, and the next step
-----------------------------------

Report the prior and its **source**, always — a prior encoding a citable study is science; a prior tuned
until the answer looks right is not. When the prior is genuinely informative, the posterior blends it
with the data in proportion to their precisions, exactly as combining two datasets would. That blending
has an exact and illuminating algebraic form — a normal prior behaves like a set of extra data points —
which the next lesson makes precise.
"""


# ----------------------------------------------------------------------
# Part IV / Stage 12 — Hierarchical Linear Models
# ----------------------------------------------------------------------

CONTENT["Regression coe\ufb03cients exchangeable in batches"] = r"""
Structure among the coefficients
----------------------------------

Ordinary regression treats coefficients as unrelated unknowns, each with its own flat prior. Often they
are not unrelated: a factor with many levels produces a **batch** of coefficients that are
**exchangeable** — the fifty state effects, the coefficients for twenty indicator categories, a set of
interactions. Treating a batch as exchangeable means giving its members a **common prior** whose
parameters are estimated, which is the hierarchical idea of Stage 5 applied inside a regression.

The model
-----------

Partition :math:`\beta` into batches. Within batch :math:`b`, the coefficients share a distribution:

.. math::

   \beta_j \sim \mathrm{N}(\mu_b, \tau_b^2) \quad \text{for } j \in \text{batch } b,
   \qquad \tau_b \sim \text{half-normal},

with :math:`\tau_b` — the batch's spread — **inferred from the data**. This is exactly a varying-intercept
model written in regression notation: each batch is a grouping factor, and :math:`\tau_b` controls how
much its coefficients are **pooled** toward the batch mean.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # fixed effects: their own weak priors
       gamma = pm.Normal("gamma", 0, 5, shape=n_fixed)
       # a batch of exchangeable coefficients: shared, inferred mean and scale
       mu_b = pm.Normal("mu_b", 0, 5)
       tau = pm.HalfNormal("tau", 1)
       z = pm.Normal("z", 0, 1, shape=n_batch)              # non-centred
       beta = pm.Deterministic("beta", mu_b + tau * z)      # pooled toward mu_b
       mu = Xf @ gamma + Xb @ beta
       pm.Normal("y", mu, pm.HalfNormal("s", 1), observed=y)

What the pooling buys
-----------------------

The batch scale :math:`\tau_b` is learned, so the amount of shrinkage is **adaptive**, exactly as in
Stage 5. A batch whose coefficients genuinely vary gets a large :math:`\tau_b` and little pooling; a
batch indistinguishable from noise gets a small :math:`\tau_b` and is shrunk hard toward its mean. The
data decide, per batch. This is far better than the two fixed alternatives: **no pooling** (ordinary
indicators, :math:`\tau_b = \infty`) overfits when levels are many and data per level are thin, while
**complete pooling** (:math:`\tau_b = 0`) ignores real differences.

Where it appears
------------------

The batched view organises much of applied modelling: the levels of every categorical predictor, the
coefficients of a spline basis (Stage 15), the many interactions in a deep model, varying slopes across
groups. Treating each such set as an exchangeable batch with its own variance is the unifying move of
this stage — and the varying-intercept, varying-slope, and ANOVA lessons that follow are all special
cases of it.
"""

CONTENT["Example: forecasting U.S. presidential elections"] = r"""
A puzzle of two timescales
----------------------------

Gelman and King posed a question that looks paradoxical: U.S. presidential elections are **predictable**
months in advance, to within a couple of points, from a handful of fundamentals — yet the campaign
**polls swing wildly**. How can the outcome be forecastable while opinion is so volatile? The resolution
and the forecasting model together make this the capstone of regression foundations.

The forecasting model
-----------------------

Predict each state's vote share from national and state-level predictors — economic growth, presidential
approval, incumbency, past partisan lean — with the states treated as an **exchangeable batch** around
the national swing:

.. math::

   y_{s} = X_{s}\beta + \alpha_{s} + \epsilon_{s}, \qquad
   \alpha_{s} \sim \mathrm{N}(0, \tau^2),

where :math:`X_s\beta` carries the fundamentals common to all states and :math:`\alpha_s` is a
state-level deviation, **pooled** toward zero by an inferred :math:`\tau`. States with little history
borrow strength from the national pattern — the hierarchical move of the previous lesson, in a
consequential setting.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       beta = pm.Normal("beta", 0, 1, shape=k)              # national fundamentals
       tau = pm.HalfNormal("tau", 0.05)                     # state-deviation scale
       z = pm.Normal("z", 0, 1, shape=n_states)
       alpha = pm.Deterministic("alpha", tau * z)           # pooled state effects
       mu = X @ beta + alpha[state]
       pm.Normal("y", mu, pm.HalfNormal("s", 0.05), observed=vote_share)
       idata = pm.sample()
   # electoral-college distribution: simulate all states jointly from the posterior

The resolution of the puzzle
------------------------------

The key insight is about what the fundamentals do. Variables like incumbency and economic conditions are
**fixed months ahead**, so in the model they are *held constant* — and, as Gelman and King put it, a
predictor held constant is **effectively controlled** and cannot contribute to forecast variance. The
outcome is predictable because it is driven by these stable quantities. The polls swing because
respondents, mid-campaign, report **unenlightened** preferences that have not yet converged on the vote
their fundamentals imply; by election day they arrive at their **enlightened** preferences, which the
fundamentals predicted all along.

Why it is the capstone
------------------------

Every thread of the stage runs through it. It is a **regression** on chosen predictors (conditional
modelling); its predictors are assembled with judgement (past vote as a control); its states are an
**exchangeable batch** with a learned scale (hierarchical coefficients); and its output is not a point
but a **posterior distribution over the electoral college**, obtained by simulating all states jointly —
propagating correlated uncertainty in a way no single number could. Prediction, description and a careful
causal story, in one model. Part V now takes regression beyond linearity and beyond a fixed set of
coefficients.
"""


MINDMAP.update({
    "Unequal variances and correlations": [
        "Bayesian analysis of classical regression", "Including numerical prior information",
        "Models for multivariate and multinomial responses", "Gaussian process regression",
    ],
    "Including numerical prior information": [
        "Bayesian analysis of classical regression", "Regularization and dimension reduction",
        "Unequal variances and correlations", "Normal Data with a Conjugate Prior Distribution",
    ],
    "Regression coe\ufb03cients exchangeable in batches": [
        "Exchangeability and hierarchical models", "Example: forecasting U.S. presidential elections",
        "Varying intercepts and slopes", "Analysis of variance and the batching of coe\ufb03cients",
    ],
    "Example: forecasting U.S. presidential elections": [
        "Regression coe\ufb03cients exchangeable in batches",
        "Regression for causal inference: incumbency and voting",
        "Varying intercepts and slopes", "Sample surveys",
    ],
})


# ----------------------------------------------------------------------
# Part IV / Stage 12 — Hierarchical Linear Models (cont.)
# ----------------------------------------------------------------------

CONTENT["Varying intercepts and slopes"] = r"""
Let the relationship itself vary
----------------------------------

A varying-intercept model lets each group have its own baseline but forces a **common slope** — every
group responds to the predictor identically. Often that is wrong: the effect of income on voting differs
by state, the effect of a treatment differs by clinic. The **varying-intercept, varying-slope** model
lets both the level *and* the response vary by group, pooling each toward a common value.

The model
-----------

Write the coefficient **vector** for group :math:`j` — intercept and slope together — as a draw from a
common multivariate distribution:

.. math::

   y_i = \alpha_{j[i]} + \beta_{j[i]} \, x_i + \epsilon_i, \qquad
   \begin{pmatrix} \alpha_j \\ \beta_j \end{pmatrix}
   \sim \mathrm{N}\!\left(
     \begin{pmatrix} \mu_\alpha \\ \mu_\beta \end{pmatrix}, \;
     \Sigma \right).

The notation :math:`j[i]` — group :math:`j` containing observation :math:`i` — is the multilevel
convention: the data :math:`y_i` exist at the individual level, and the grouping enters as an index, not
as a reordering. Each group's intercept and slope are **partially pooled** toward the population means,
by amounts the data determine through :math:`\Sigma`.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu = pm.Normal("mu", 0, 5, shape=2)                  # population (intercept, slope)
       sd = pm.HalfNormal("sd", 1, shape=2)                 # scales of each
       L = pm.LKJCholeskyCov("L", n=2, eta=2,               # correlation + scales
                             sd_dist=pm.HalfNormal.dist(1))
       z = pm.Normal("z", 0, 1, shape=(n_groups, 2))
       ab = pm.Deterministic("ab", mu + z @ L.T)            # non-centred group coeffs
       mu_i = ab[group, 0] + ab[group, 1] * x
       pm.Normal("y", mu_i, pm.HalfNormal("s", 1), observed=y)

Fixed effects are a special case
----------------------------------

The multilevel frame **subsumes** the classical distinction. A "fixed effect" is the limit of a random
effect as its group-level variance goes to **infinity** — no pooling, each group estimated on its own
data. A single pooled coefficient is the **zero-variance** limit. The varying-slope model sits between,
and — crucially — **estimates where**, rather than forcing you to choose. There is no separate machinery
for fixed versus random; there is one model with a variance the data inform.

Why pool the slopes
---------------------

The same argument as always, now for effects rather than means. A group with little data gets an
**unstable** slope on its own; pooling toward the population slope stabilises it, trading a little bias
for a large variance reduction. Groups with ample data are barely moved. And the population-level
:math:`\mu_\beta` and :math:`\Sigma` are themselves of interest: they describe **how much** the effect
varies across groups — often the substantive question. The one subtlety, addressed next, is that
intercepts and slopes typically **covary**, and modelling that correlation is what :math:`\Sigma` is for.
"""



CONTENT["Analysis of variance and the batching of coe\ufb03cients"] = r"""
ANOVA, rebuilt as a hierarchy
-------------------------------

Classical analysis of variance partitions variation among **sources** — main effects, interactions,
residual — and tests each with an :math:`F` statistic. The Bayesian view keeps the partition but changes
its meaning: each source of variation is a **batch of exchangeable coefficients** with its own variance
component, and ANOVA becomes the estimation of those variances.

The reframing
---------------

A two-way layout with factors :math:`A` and :math:`B` and their interaction becomes

.. math::

   y_i = \mu + a_{j[i]} + b_{k[i]} + (ab)_{jk[i]} + \epsilon_i,

with each batch drawn from its own distribution,

.. math::

   a_j \sim \mathrm{N}(0, \sigma_A^2), \quad
   b_k \sim \mathrm{N}(0, \sigma_B^2), \quad
   (ab)_{jk} \sim \mathrm{N}(0, \sigma_{AB}^2), \quad
   \epsilon_i \sim \mathrm{N}(0, \sigma_y^2).

The classical **mean squares** are essentially estimates of these variance components. But where ANOVA
asks "is :math:`\sigma_A^2` exactly zero?" — a point null — the hierarchical model **estimates each
:math:`\sigma`**, which is almost always the more useful quantity. How large the batch of effects is
matters more than whether it is precisely null.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu = pm.Normal("mu", 0, 5)
       sA = pm.HalfNormal("sA", 1); sB = pm.HalfNormal("sB", 1); sAB = pm.HalfNormal("sAB", 1)
       a  = pm.Normal("a", 0, sA, shape=nA)                 # batch A
       b  = pm.Normal("b", 0, sB, shape=nB)                 # batch B
       ab = pm.Normal("ab", 0, sAB, shape=(nA, nB))         # interaction batch
       mu_i = mu + a[jA] + b[jB] + ab[jA, jB]
       pm.Normal("y", mu_i, pm.HalfNormal("sy", 1), observed=y)
       # posteriors for sA, sB, sAB ARE the analysis of variance

Finite and superpopulation variance
--------------------------------------

Gelman's framing sharpens what a variance component *means*. The **superpopulation** standard deviation
describes the distribution from which the batch of effects is drawn — relevant to a *new*, unobserved
level. The **finite-population** standard deviation is the actual spread of the effects you *have* — the
:math:`nA` levels in this study. They differ when the batch is small, and the distinction answers a real
question: are you generalising to new levels, or summarising these?

Why the batched view wins
---------------------------

Three advantages over the :math:`F`-test tradition. It handles **unbalanced** and **complex** designs
uniformly, where classical ANOVA formulas fragment. It gives **partial pooling** of the effects within
each batch, stabilising estimates that classical ANOVA leaves noisy. And it reports **magnitudes with
uncertainty** rather than a reject/retain verdict on a null nobody believes exactly. ANOVA was never
really about testing; it was about **decomposing variation**, and the hierarchical model does that
directly — the same batching that ran through this entire stage, applied to designed factors.
"""


MINDMAP.update({
    "Varying intercepts and slopes": [
        "Regression coe\ufb03cients exchangeable in batches", "Modeling the group-level covariance",
        "Prediction and hierarchical models", "Exchangeability and hierarchical models",
    ],
    "Analysis of variance and the batching of coe\ufb03cients": [
        "Regression coe\ufb03cients exchangeable in batches", "Varying intercepts and slopes",
        "Hierarchical models for batches of variance components", "Computation: batching and transformation",
    ],
})


# ----------------------------------------------------------------------
# Part IV / Stage 12 — Hierarchical Linear Models (corrective additions: 101, 103)
# ----------------------------------------------------------------------

CONTENT["Interpreting a normal prior distribution as extra data"] = r"""
A prior is imaginary data
---------------------------

The previous lesson added quantitative prior knowledge through a prior on :math:`\beta`. There is an
exact algebraic identity underneath it, and it is one of the most clarifying facts in Bayesian
regression: **a normal prior on the coefficients is equivalent to a set of extra data points** appended
to the regression.

The augmented-data identity
-----------------------------

Take the linear model with a normal prior :math:`\beta \sim \mathrm{N}(\beta_0, \Sigma_0)`. Its
posterior mode is **identical** to the least-squares fit of an augmented dataset — the real observations,
plus one pseudo-observation per prior constraint:

.. math::

   \bar{X} = \begin{bmatrix} X \\ \Sigma_0^{-1/2} \end{bmatrix}, \qquad
   \bar{y} = \begin{bmatrix} y \\ \Sigma_0^{-1/2}\beta_0 \end{bmatrix},
   \qquad
   \bar{X}^{\top}\bar{X} = \underbrace{X^{\top}X}_{\text{data}} + \underbrace{\Sigma_0^{-1}}_{\text{prior}} .

The precision decomposes into a data term and a prior term that simply **add**. Each augmented row is one
imaginary observation stating "at this design point the response was :math:`\beta_0`", carrying a
precision set by the prior. The special case :math:`\beta_0 = 0`, :math:`\Sigma_0 = (\sigma^2/\lambda) I`
recovers **ridge regression** exactly — appending :math:`\sqrt{\lambda}\, e_j` rows with response zero.

.. code-block:: python

   import numpy as np
   # prior beta ~ N(beta0, diag(tau^2)) as pseudo-observations, then plain least squares
   P = np.diag(1.0 / tau)                            # Sigma0^{-1/2}
   X_aug = np.vstack([X, P])
   y_aug = np.concatenate([y, P @ beta0])
   beta_post_mode = np.linalg.lstsq(X_aug, y_aug, rcond=None)[0]   # == posterior mode

Why the picture helps
-----------------------

It makes the **strength** of a prior concrete. A tight prior is *many* pseudo-observations; a vague one
is *few*; their precisions add to the data's exactly as a second dataset would. It demystifies
regularisation — ridge, and its relatives, are priors and nothing more. And it clarifies the balance of
evidence: where the real data are informative about a coefficient, they swamp the handful of pseudo-rows;
where they are silent (collinearity, few observations), the prior carries the estimate, which is
precisely when you want it to.

The counting interpretation
-----------------------------

The identity even assigns the prior an **effective sample size**. A prior with precision
:math:`\Sigma_0^{-1}` contributes as much information as that many real observations at the corresponding
design points, so "how strong is this prior?" has a literal answer in units of data. That is the honest
way to feel the weight of a prior — and the honest warning against a prior so tight it silently adds
hundreds of observations you never collected.
"""

CONTENT["Computation: batching and transformation"] = r"""
Making batched models sample
------------------------------

A model with several batches of exchangeable coefficients — varying intercepts, varying slopes, their
covariance — is easy to *write* and can be hard to *fit*. The geometry of hierarchical posteriors, first
met at eight schools, returns in force here, and two transformations make these models tractable.

The funnel, at scale
----------------------

Each batch carries the same pathology: when its scale :math:`\tau_b` is small, the coefficients in that
batch are squeezed into a narrow neck whose curvature the sampler cannot follow, and NUTS reports
divergences (Stage 9). With multiple batches the funnels compound. The **non-centred** parameterisation
is the standard cure — write each coefficient as a standard-normal draw scaled and shifted by the batch
parameters, so the raw parameters have a geometry independent of :math:`\tau_b`:

.. math::

   \beta_j = \mu_b + \tau_b \, \eta_j, \qquad \eta_j \sim \mathrm{N}(0, 1).

Apply it per batch, and — for varying intercepts and slopes together — to the **whole coefficient vector
at once** through the Cholesky factor of the group covariance.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu = pm.Normal("mu", 0, 5, shape=2)
       chol, _, _ = pm.LKJCholeskyCov("cov", n=2, eta=2,
                                      sd_dist=pm.HalfNormal.dist(1.0), compute_corr=True)
       z = pm.Normal("z", 0, 1, shape=(n_groups, 2))        # flat geometry
       ab = pm.Deterministic("ab", mu + z @ chol.T)         # correlated, non-centred
       mu_i = ab[grp, 0] + ab[grp, 1] * x
       pm.Normal("y", mu_i, pm.HalfNormal("s", 1), observed=y)

Transformation and scaling
----------------------------

The second lever is the same one that helped Gibbs. **Centre and scale predictors** so intercepts and
slopes decorrelate and share a common metric; a :math:`\mathrm{N}(0, 1)` prior then means the same thing
everywhere, and the sampler sees a roughly spherical posterior. Where a batch is data-rich, the
**centred** parameterisation is actually better conditioned — so the choice of centred versus non-centred
is per batch, decided by how much information each group carries, exactly as in the eight-schools lesson.

Practical workflow
--------------------

Three habits keep batched models honest. **Standardise inputs** before fitting. **Default to
non-centred** for batches with weak data per group, centred for strong. And **read the diagnostics**
per batch — divergences, :math:`\hat{R}`, and bulk/tail ESS for each :math:`\tau_b` — because a single
badly-parameterised batch can stall an otherwise healthy model. Batching organises the model; these
transformations are what let the sampler explore it.
"""


MINDMAP.update({
    "Including numerical prior information": [
        "Bayesian analysis of classical regression",
        "Interpreting a normal prior distribution as extra data",
        "Regularization and dimension reduction", "Weakly Informative Prior Distributions",
    ],
    "Interpreting a normal prior distribution as extra data": [
        "Including numerical prior information", "Regularization and dimension reduction",
        "Bayesian analysis of classical regression", "Normal Data with a Conjugate Prior Distribution",
    ],
    "Computation: batching and transformation": [
        "Varying intercepts and slopes", "Hamiltonian Monte Carlo for a hierarchical model",
        "E\ufb03cient Gibbs samplers", "Analysis of variance and the batching of coe\ufb03cients",
    ],
    # repair the two dangling neighbours left by removing the invented titles
    "Varying intercepts and slopes": [
        "Regression coe\ufb03cients exchangeable in batches", "Computation: batching and transformation",
        "Analysis of variance and the batching of coe\ufb03cients", "Exchangeability and hierarchical models",
    ],
})


# ----------------------------------------------------------------------
# Part IV / Stage 12 — Hierarchical Linear Models  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Hierarchical models for batches of variance components"] = r"""
When the variances themselves have structure
-----------------------------------------------

The ANOVA lesson gave every source of variation its own standard deviation: :math:`\sigma_A`,
:math:`\sigma_B`, :math:`\sigma_{AB}`, and so on. A complex design can have **many** such variance
components — crossed and nested factors, multiway interactions — and they are not unrelated. When there
are enough of them, the natural move is the one this whole part has been making: treat the **variance
components as a batch** and model them hierarchically too.

A hierarchy on the standard deviations
----------------------------------------

Give the collection of batch scales a common prior with its own parameters:

.. math::

   y_i = \mu + \sum_{m} \beta^{(m)}_{j_m[i]} + \epsilon_i, \qquad
   \beta^{(m)}_j \sim \mathrm{N}(0, \sigma_m^2), \qquad
   \sigma_m \sim \text{half-}\mathrm{N}(0, \tau^2),

where each :math:`\sigma_m` is the spread of variance-component batch :math:`m`, and the
:math:`\sigma_m` are themselves drawn from a shared distribution with hyperscale :math:`\tau`. The
variance components are now **partially pooled** toward each other — a component estimated from few levels
borrows from the others, exactly as the group means did one level down.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       tau = pm.HalfNormal("tau", 1)                        # hyperscale over components
       sigma = pm.HalfNormal("sigma", tau, shape=n_components)   # batch of scales, pooled
       # each factor's effects drawn from its own (pooled) scale, non-centred
       effects = [pm.Normal(f"b{m}", 0, 1, shape=n_levels[m]) * sigma[m]
                  for m in range(n_components)]
       mu_i = mu + sum(e[idx[m]] for m, e in enumerate(effects))
       pm.Normal("y", mu_i, pm.HalfNormal("sy", 1), observed=y)

Why pool variances
--------------------

The same logic that pools means, one storey higher. A variance component estimated from **few levels** is
badly determined on its own — a two-level factor gives almost no information about its own spread — and
pooling toward the other components stabilises it. This matters most in the designs where classical ANOVA
is least reliable: many factors, few levels each, unbalanced cells. The estimate of any one
:math:`\sigma_m` improves by borrowing from the ensemble.

The finite-population reminder
--------------------------------

The distinction from the ANOVA lesson persists at this level. For a factor with a handful of levels, the
**finite-population** standard deviation — the spread of the effects actually present — is often the
meaningful summary, and it is estimated more precisely than the **superpopulation** scale, which
describes hypothetical new levels. Report the one that answers the question. This closes the hierarchical
regression stage: the batching idea, applied first to coefficients and then to their variances, turns a
tangle of factors into one coherent model whose every scale is estimated with appropriate pooling. Part
IV now leaves the normal likelihood behind.
"""


# ----------------------------------------------------------------------
# Part IV / Stage 13 — Generalized Linear Models
# ----------------------------------------------------------------------

CONTENT["Standard generalized linear model likelihoods"] = r"""
Beyond the normal outcome
---------------------------

Linear regression assumes a **continuous, unbounded, normally-distributed** response. Most real outcomes
are none of these: a yes/no, a count, a category, a waiting time. The **generalized linear model** keeps
the linear predictor :math:`X\beta` but connects it to the outcome through two changes — a **link
function** and a non-normal **likelihood** — so the same machinery covers a whole zoo of data types.

The three ingredients
-----------------------

Every GLM is specified by:

1. a **linear predictor** :math:`\eta = X\beta`, the familiar weighted sum;
2. a **link function** :math:`g` mapping the mean of the outcome to that unbounded scale,
   :math:`g(\mu) = \eta`, so predictions respect the outcome's range;
3. a **distribution** for the outcome given its mean, from the exponential family.

.. math::

   g\bigl(\mathrm{E}[y_i]\bigr) = X_i \beta, \qquad
   y_i \sim \text{ExponentialFamily}(\mu_i).

The link is what keeps a probability in :math:`[0, 1]` and a rate positive: the linear predictor roams
the whole real line, and :math:`g^{-1}` folds it back into the legal range.

The common members
--------------------

Four cover most applied work:

* **Logistic** — binary :math:`y`. Link: **logit**, :math:`\log\frac{\mu}{1-\mu}`. Likelihood:
  Bernoulli/binomial. Coefficients are **log odds ratios**.
* **Poisson** — counts. Link: **log**, :math:`\log \mu`. Likelihood: Poisson. Coefficients are **log
  rate ratios**; an offset handles exposure.
* **Normal** — the identity link, the special case where the whole apparatus collapses back to linear
  regression.
* **Multinomial / ordinal** — categorical outcomes, via softmax or cumulative-logit links (next stage).

.. code-block:: python

   import pymc as pm
   # logistic regression
   with pm.Model():
       beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])
       p = pm.Deterministic("p", pm.math.sigmoid(X @ beta))     # inverse logit
       pm.Bernoulli("y", p=p, observed=y)

   # Poisson regression with an exposure offset
   with pm.Model():
       beta = pm.Normal("beta", 0, 1, shape=X.shape[1])
       rate = pm.math.exp(X @ beta + log_exposure)              # log link + offset
       pm.Poisson("y", mu=rate, observed=counts)

Why one framework
-------------------

Unifying these under one structure is not mere tidiness. The **same** priors, the **same** hierarchical
extensions (varying coefficients, batching), the **same** diagnostics (PPC, LOO), and the **same**
sampler apply across every outcome type — only the link and likelihood change. Learn the pattern once and
binary, count and categorical data are variations on a theme, not separate subjects. The interpretation
of coefficients changes with the link, though, and that — along with the priors that keep GLMs
well-behaved — is what the rest of this stage is about.
"""

CONTENT["Working with generalized linear models"] = r"""
Fitting is easy; interpreting is the work
--------------------------------------------

A GLM is a one-line change from linear regression to fit. The effort moves to **interpretation** — the
link function makes coefficients nonlinear on the outcome scale — and to the **checks** that catch a
misspecified likelihood.

Interpreting on the right scale
---------------------------------

A coefficient lives on the **link** scale, not the outcome scale, and must be translated.

* **Logistic.** :math:`\beta_j` is a change in **log odds** per unit of :math:`x_j`;
  :math:`e^{\beta_j}` is an **odds ratio**. On the probability scale the effect is *nonlinear* — the same
  :math:`\beta_j` moves the probability a lot near :math:`0.5` and little near the extremes. The
  **divide-by-4 rule** gives a quick upper bound: :math:`\beta_j / 4` is the maximum change in
  probability per unit.
* **Poisson.** :math:`e^{\beta_j}` is a **rate ratio** — a multiplicative effect on the count.

Because effects are nonlinear, a single number rarely captures them; **average predictive comparisons**
and predicted-probability plots communicate far better than a coefficient table.

.. code-block:: python

   import numpy as np
   beta = idata.posterior["beta"].values.reshape(-1, k)
   odds_ratio = np.exp(beta[:, j])                         # logistic: multiplicative on odds
   np.percentile(odds_ratio, [2.5, 97.5])

   # honest effect: predicted-probability difference at representative x, averaged over posterior
   from scipy.special import expit
   p_hi = expit(X_hi @ beta.T); p_lo = expit(X_lo @ beta.T)
   (p_hi - p_lo).mean()                                    # average change in probability

Checking the fit
------------------

The posterior predictive check adapts to the outcome type. For **counts**, compare the observed and
predicted frequency of each value, and especially the number of **zeros** — excess zeros are the classic
sign of a wrong likelihood. For **binary** data, check **calibration**: among cases with predicted
probability near :math:`p`, is the observed rate near :math:`p`?

.. code-block:: python

   import arviz as az
   az.plot_ppc(idata)                     # observed vs predicted outcome distribution
   # counts: does the model reproduce the spike at zero?  binary: is it calibrated?

The recurring failure
-----------------------

Most GLM trouble is the **variance assumption**. The Poisson forces variance to equal the mean; real
counts are usually **overdispersed**, with variance far larger, so Poisson intervals come out much too
narrow. The binomial makes an analogous assumption. Detecting this — variance exceeding what the
likelihood permits — and fixing it with a richer likelihood is the subject two lessons on. First,
though, the priors that make even the basic logistic model behave.
"""

CONTENT["Weakly informative priors for logistic regression"] = r"""
When flat priors fail
-----------------------

Logistic regression has a specific, common pathology: **separation**. If some combination of predictors
perfectly (or nearly perfectly) predicts the outcome, the maximum-likelihood coefficient runs off to
**infinity** — the likelihood keeps rising as the coefficient grows, with no maximum. Improper flat
priors inherit the problem: the posterior is improper, and standard software returns absurd estimates
with enormous standard errors. Separation is not rare; it appears even with large samples and few
predictors.

The weakly informative fix
----------------------------

Gelman, Jakulin, Pittau and Su proposed a default that solves this cleanly. Two steps: **scale** the
predictors, then put a **heavy-tailed prior** on the coefficients.

.. math::

   \beta_j \sim \mathrm{Cauchy}(0, s_j), \qquad
   s_0 = 10 \;\; (\text{intercept}), \quad s_j = 2.5 \;\; (\text{scaled predictors}).

The recipe centres every nonbinary predictor at mean 0 and scales it to standard deviation 0.5, so that
a unit change is a two-SD shift and the scale 2.5 means the same thing across coefficients. The prior is
**weakly informative**: it barely constrains coefficients in the plausible range but rules out the
absurd — a logistic coefficient of 40, implying an effect no finite dataset could support.

.. code-block:: python

   import pymc as pm
   # scale nonbinary predictors to mean 0, sd 0.5 (per Gelman et al. 2008)
   Xs = (X - X.mean(0)) / (2 * X.std(0))
   with pm.Model():
       b0 = pm.Cauchy("b0", 0, 10)                          # intercept
       b = pm.StudentT("b", nu=1, mu=0, sigma=2.5, shape=Xs.shape[1])   # Cauchy = t(1)
       p = pm.math.sigmoid(b0 + Xs @ b)
       pm.Bernoulli("y", p=p, observed=y)

Why Cauchy
------------

The choice of a heavy tail is deliberate, and it balances two demands. The tails must be **heavy enough**
not to over-shrink a coefficient that really is large — a genuine strong effect should survive. But the
prior must be **proper and informative enough** to give a finite answer under separation, which a flat
prior cannot. The Cauchy — Student-:math:`t` with one degree of freedom — sits at exactly that balance:
its tails are heavier than any normal, so large true coefficients are respected, yet its mass near zero
provides the gentle shrinkage that tames separation and **always yields an answer**.

The general lesson
--------------------

This is the template for weakly informative priors everywhere. Not the false neutrality of a flat prior,
which secretly asserts that a coefficient of 100 is as plausible as one of 1; and not a tight informative
prior that overrides the data. A weakly informative prior encodes what any analyst knows before seeing
the data — that effects on a sensible scale are not astronomically large — and that mild knowledge is
often exactly enough to turn an ill-posed problem into a well-behaved one.
"""


MINDMAP.update({
    "Hierarchical models for batches of variance components": [
        "Analysis of variance and the batching of coe\ufb03cients",
        "Regression coe\ufb03cients exchangeable in batches",
        "Weakly Informative Priors for Variance Parameters", "Varying intercepts and slopes",
    ],
    "Standard generalized linear model likelihoods": [
        "Conditional modeling", "Working with generalized linear models",
        "Weakly informative priors for logistic regression", "Models for multivariate and multinomial responses",
    ],
    "Working with generalized linear models": [
        "Standard generalized linear model likelihoods", "Weakly informative priors for logistic regression",
        "Overdispersed Poisson regression for police stops", "Do the Inferences from the Model Make Sense?",
    ],
    "Weakly informative priors for logistic regression": [
        "Working with generalized linear models", "Weakly Informative Prior Distributions",
        "Standard generalized linear model likelihoods", "Regularization and dimension reduction",
    ],
})


# ----------------------------------------------------------------------
# Part IV / Stage 13 — Generalized Linear Models (cont.)  [completes Part IV]
# ----------------------------------------------------------------------

CONTENT["Overdispersed Poisson regression for police stops"] = r"""
Counts with too much variance
-------------------------------

The Poisson model forces a rigid tie: variance equals mean. Real count data almost never obey it —
stop counts, disease cases, accident tallies vary far more than Poisson allows, because unmodelled
heterogeneity inflates the spread. This lesson works through the canonical applied example and the fix
for **overdispersion**.

The stop-and-frisk study
--------------------------

Gelman, Fagan and Kiss analysed roughly 125,000 pedestrian stops by the New York police over fifteen
months, asking whether minority pedestrians were stopped more often than a race-neutral policy would
predict. The count of stops of ethnic group :math:`e` in precinct :math:`p` is modelled as Poisson,
with an **offset** for the group's baseline rate of prior arrests — the appropriate denominator, since
stops should be compared against involvement in crime, not raw population:

.. math::

   y_{ep} \sim \mathrm{Poisson}\bigl(n_{ep} \, e^{\mu + \alpha_e + \beta_p + \epsilon_{ep}}\bigr),
   \qquad \epsilon_{ep} \sim \mathrm{N}(0, \sigma^2),

where :math:`\alpha_e` is the ethnic-group effect, :math:`\beta_p` a precinct effect, and
:math:`n_{ep}` the arrest-based offset. The per-observation :math:`\epsilon_{ep}` is the crucial term.

Overdispersion, modelled honestly
-----------------------------------

That :math:`\epsilon_{ep}` — an individual-level normal error inside the log rate — is what turns a
Poisson into an **overdispersed** Poisson. It lets each cell's rate deviate from what the fixed effects
predict, absorbing the excess variance that a plain Poisson would deny. Without it, the model forces
variance to equal mean, the residuals blow past that ceiling, and every standard error comes out far
too small — a confident, wrong answer about a charged question.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu = pm.Normal("mu", 0, 5)
       a = pm.Normal("a", 0, 1, shape=n_eth)                # ethnic-group effects
       s_p = pm.HalfNormal("s_p", 1)
       b = pm.Normal("b", 0, s_p, shape=n_prec)             # precinct effects (pooled)
       s_e = pm.HalfNormal("s_e", 1)
       eps = pm.Normal("eps", 0, s_e, shape=n_obs)          # <- overdispersion term
       rate = pm.math.exp(mu + a[eth] + b[prec] + eps) * n_offset
       pm.Poisson("y", rate, observed=stops)

The finding, and the method's point
-------------------------------------

Controlling for precinct and for race-specific arrest rates, the analysis found that Black and Hispanic
pedestrians were stopped **more** than whites — the disparity survived the controls that might have
explained it away. Two methodological lessons carry beyond the application. The **offset** encodes the
right comparison: rates per unit of a relevant baseline, not raw counts. And the **overdispersion term**
is what makes the uncertainty honest — an ordinary Poisson would have reported spuriously precise
estimates on a question where precision claims have real consequences. Diagnosing variance in excess of
the mean, and modelling it, is not optional refinement; it is the difference between a defensible finding
and an artefact.
"""

CONTENT["State-level opinons from national polls"] = r"""
Small areas from big surveys
------------------------------

A national poll of a couple of thousand respondents estimates *national* opinion well but says little
about any single **state** — some states contain a handful of respondents. Yet state-level opinion is
exactly what redistricting, forecasting and representation require. **Multilevel regression and
poststratification** (MRP) extracts reliable small-area estimates from national data, and it is one of
the most consequential applications of the hierarchical models built in this part.

The two steps
---------------

MRP is a **model** followed by a **weighting**.

**Regression.** Fit a multilevel logistic model for the individual response, with demographic predictors
(age, race, sex, education) as varying effects and a **state effect** that is partially pooled — often
with a state-level predictor such as past vote:

.. math::

   \Pr(y_i = 1) = \mathrm{logit}^{-1}\!\bigl(\alpha_{\text{state}[i]}
                  + \beta_{\text{age}[i]} + \beta_{\text{race}[i]} + \cdots\bigr),
   \qquad \alpha_s \sim \mathrm{N}(\gamma_0 + \gamma_1 v_s, \sigma^2).

Partial pooling stabilises the estimate for every demographic-by-state cell, even cells with almost no
respondents — the shrinkage of this entire part, doing the heavy lifting.

**Poststratification.** Reweight the model's cell predictions by the **known population frequency** of
each cell from the census, so the state estimate reflects that state's actual demographic composition:

.. math::

   \theta_s = \frac{\sum_{c \in s} N_c \, \hat{p}_c}{\sum_{c \in s} N_c} .

.. code-block:: python

   import numpy as np
   # 1. multilevel model gives p_hat for every (demographic x state) cell
   p_cell = posterior_cell_predictions(idata)              # shape: (draws, n_cells)
   # 2. reweight by census population counts N_cell within each state
   theta_state = (N_cell * p_cell).sum(axis=1) / N_cell.sum()   # per state, with uncertainty

Why it works
--------------

The pieces cover each other's weaknesses. The **model** borrows strength across states so that sparse
cells get sensible estimates instead of noise; **poststratification** corrects the sample's demographic
imbalances against known population totals, removing the bias that makes raw subgroup means unreliable.
Together they turn a survey never designed for state estimates into a state-level instrument — and the
same machinery **adjusts for non-representative samples** generally, which is why MRP has become central
to modern survey inference and to forecasting from imperfect polls. It is the hierarchical logistic model
of this stage, put to work on the small-area problem.
"""

CONTENT["Models for multivariate and multinomial responses"] = r"""
When the outcome is not a single number
-----------------------------------------

So far each observation had one response. Two common cases break that: a **multinomial** outcome, where
the response is one of several unordered (or ordered) categories, and a **multivariate** outcome, where
several correlated responses are measured together. Both extend the GLM, and both hinge on modelling the
**structure among the outcomes**.

Multinomial and categorical
-----------------------------

For an **unordered** categorical outcome — choice of transport, party, product — the multinomial
(softmax) logit gives each category :math:`k` its own linear predictor and normalises:

.. math::

   \Pr(y_i = k) = \frac{e^{X_i \beta_k}}{\sum_{k'} e^{X_i \beta_{k'}}},

with one category fixed as baseline for identifiability. For an **ordered** outcome — a Likert scale, a
severity grade — the **ordered logit** is better: a single linear predictor with a set of increasing
**cutpoints** slicing a latent continuum, which respects the ordering and spends far fewer parameters.

.. code-block:: python

   import pymc as pm
   # ordered categorical: shared slope, learned cutpoints
   with pm.Model():
       beta = pm.Normal("beta", 0, 1, shape=k)
       cut = pm.Normal("cut", 0, 5, shape=n_cat - 1,
                       transform=pm.distributions.transforms.ordered)   # increasing
       pm.OrderedLogistic("y", eta=X @ beta, cutpoints=cut, observed=y)

Multivariate responses
------------------------

When several responses are measured per unit — height and weight, a battery of test scores — they are
**correlated**, and modelling them jointly is more informative than one regression each. Give the vector
of responses a multivariate normal likelihood with a covariance to estimate, using the **LKJ**
decomposition from the hierarchy stage:

.. math::

   y_i \sim \mathrm{N}(X_i B, \; \Sigma), \qquad
   \Sigma = \mathrm{diag}(\sigma)\,\Omega\,\mathrm{diag}(\sigma), \quad \Omega \sim \mathrm{LKJ}(\eta).

The estimated correlations are often the point — how the outcomes move together, given the predictors.

Why model the joint structure
-------------------------------

Two payoffs. Modelling outcomes jointly **borrows information across them**: a missing component is
predicted from the others through the learned correlation, and estimates are more efficient than
separate fits. And the **dependence itself is substantive** — the correlation between test scores given
background, the covariance of symptoms — a quantity separate univariate models simply cannot report.
Categorical and multivariate responses complete the GLM's reach: with them, the framework covers binary,
count, ordered, unordered and vector-valued outcomes under one set of tools.
"""

CONTENT["Loglinear models for multivariate discrete data"] = r"""
Modelling a table of counts
-----------------------------

Cross-classify a sample by several categorical variables — sex by education by vote — and the data become
a **contingency table** of counts. **Loglinear models** describe such tables by modelling the log
expected count in each cell as a linear function of the variables and their interactions, turning
questions about **association** into questions about **coefficients**.

The model
-----------

For a two-way table with factors :math:`A` and :math:`B`, the log expected cell count is

.. math::

   \log \mu_{ij} = \lambda + \lambda^A_i + \lambda^B_j + \lambda^{AB}_{ij},

with main effects for each margin and an **interaction** term. The interaction is the object of interest:
:math:`\lambda^{AB}_{ij} = 0` for all :math:`i, j` is exactly the statement that :math:`A` and :math:`B`
are **independent**. Testing independence becomes testing whether an interaction batch is negligible —
and, Bayesianly, *estimating* how large it is.

.. code-block:: python

   import pymc as pm
   # loglinear model for a two-way table of counts
   with pm.Model():
       lam = pm.Normal("lam", 0, 5)
       aA = pm.Normal("aA", 0, 2, shape=I)                  # margin A effects
       aB = pm.Normal("aB", 0, 2, shape=J)                  # margin B effects
       s_ab = pm.HalfNormal("s_ab", 1)
       aAB = pm.Normal("aAB", 0, s_ab, shape=(I, J))        # interaction (association), pooled
       log_mu = lam + aA[:, None] + aB[None, :] + aAB
       pm.Poisson("y", pm.math.exp(log_mu).flatten(), observed=counts.flatten())

The connections
-----------------

Loglinear models tie together several threads. Their equivalence to **Poisson regression** with
categorical predictors is exact — a table of counts *is* count data with factor predictors — so
everything from the overdispersion lesson applies, including the caution that cell counts may vary more
than Poisson allows. For binary outcomes they connect directly to **logistic** regression: the loglinear
interaction involving the response reproduces the logistic coefficient. And treating the interaction
terms as an **exchangeable batch** with a shared scale — the batching idea one final time — gives a
hierarchical loglinear model that **pools** sparse cells, taming the zeros and small counts that make
large contingency tables notoriously unstable.

Where it sits
---------------

For **high-dimensional** discrete data — many categorical variables at once — loglinear models with
hierarchical interaction terms are a principled tool, letting the data determine which associations are
real while pooling away the noise. They close the generalized linear model stage by handling the last
data type on the list: whole tables of categorical counts. Part IV has taken the linear model from
continuous responses through binary, count, ordered, multivariate and tabular data; Part V now abandons
the linear predictor itself, for functions and infinite-dimensional models.
"""


MINDMAP.update({
    "Overdispersed Poisson regression for police stops": [
        "Standard generalized linear model likelihoods", "Working with generalized linear models",
        "Overdispersed versions of standard models", "Hierarchical models for batches of variance components",
    ],
    "State-level opinons from national polls": [
        "Working with generalized linear models", "Weakly informative priors for logistic regression",
        "Varying intercepts and slopes", "Sample surveys",
    ],
    "Models for multivariate and multinomial responses": [
        "Standard generalized linear model likelihoods", "Loglinear models for multivariate discrete data",
        "Multivariate Normal with Unknown Mean and Variance", "Working with generalized linear models",
    ],
    "Loglinear models for multivariate discrete data": [
        "Models for multivariate and multinomial responses", "Overdispersed Poisson regression for police stops",
        "Standard generalized linear model likelihoods", "Working with generalized linear models",
    ],
})


# ----------------------------------------------------------------------
# Part V / Stage 14 — Robust Inference & Missing Data
# ----------------------------------------------------------------------

CONTENT["Aspects of robustness"] = r"""
When the model is wrong in a particular way
---------------------------------------------

Every model is an approximation, but some approximations fail gracefully and others catastrophically.
**Robustness** is the study of how sensitive a conclusion is to assumptions that are not quite true —
and, more usefully, how to build models whose conclusions survive the failures that matter most. The
recurring culprit is the **normal likelihood** and its intolerance of outliers.

Why the normal is fragile
---------------------------

The normal distribution has **light tails**: its density falls off like :math:`e^{-x^2/2}`, so a value
five standard deviations out is astronomically improbable. When a real observation lands there anyway —
a recording error, a genuine anomaly — the normal model cannot treat it as a rare event. It must instead
**move the mean** to accommodate the point, because under the normal likelihood the far-out value is so
costly that shifting every estimate is cheaper than tolerating it. A single outlier can drag the
posterior a long way, and the sample mean, its estimator, is the classic non-robust statistic.

The Bayesian view of robustness
---------------------------------

Robustness in this framework is not a patch applied afterward; it is a **modelling choice about the
likelihood and prior**. Three questions organise it:

* **Likelihood robustness** — does one aberrant observation dominate the fit? The cure is a
  heavy-tailed likelihood, the next two lessons.
* **Prior robustness** — does the conclusion hinge on a specific prior? The check is to vary the prior
  and watch the posterior, the sensitivity analysis of Stage 6.
* **Structural robustness** — does it depend on the model's form? Answered by model expansion and
  predictive comparison.

The general principle
-----------------------

Heavy tails buy robustness. A likelihood or prior with **heavier tails than the normal** can assign a
far-out point reasonable probability *as an outlier*, so the point no longer forces the bulk of the model
to move — the estimate is determined by the mass of the data, not held hostage by its extremes. This is
the same insight that motivated the Cauchy prior for separation: heavy tails let a model **tolerate** the
unusual instead of contorting to explain it. The lessons that follow make it concrete for the
likelihood, where outliers do their damage.

One caution
-------------

Robustness is not free and not total. A heavy-tailed likelihood down-weights outliers automatically, but
it is only **partially** robust — an outlier in a *predictor* can still distort a regression, and
down-weighting a genuine signal as if it were noise loses information. Robustness is about surviving the
assumptions most likely to be wrong, not immunity to all of them; naming which failure you are guarding
against is part of the modelling.
"""

CONTENT["Overdispersed versions of standard models"] = r"""
Adding a tail to every likelihood
-----------------------------------

The standard likelihoods — normal, Poisson, binomial — each impose a **rigid variance**. Robustness is
bought by relaxing that rigidity: replace each with an **overdispersed** version carrying an extra
parameter that lets the tails or the variance grow. The pattern is uniform across families.

The normal to the t
---------------------

For continuous data, replace the normal with the **Student-**:math:`t`, whose degrees of freedom
:math:`\nu` tune tail weight:

.. math::

   y_i \sim t_\nu(\mu, \sigma), \qquad \nu > 0.

Small :math:`\nu` gives heavy tails that tolerate outliers; large :math:`\nu` recovers the normal (the
:math:`t` converges to it as :math:`\nu \to \infty`). Estimating :math:`\nu` lets the data decide **how
heavy-tailed** they are — a self-diagnosing robustness.

Poisson and binomial, overdispersed
-------------------------------------

The count families gain a variance parameter the same way. The **negative binomial** is a Poisson whose
rate is itself Gamma-distributed, giving variance that **exceeds** the mean by a controllable amount —
the closed-form counterpart of the log-normal error added by hand in the police-stops lesson. The
**beta-binomial** does the same for the binomial, letting the success probability vary across trials.

.. code-block:: python

   import pymc as pm
   # robust continuous: Student-t with estimated tail weight
   with pm.Model():
       nu = pm.Gamma("nu", 2, 0.1)                          # degrees of freedom, >~ 1
       pm.StudentT("y", nu=nu, mu=X @ beta, sigma=sigma, observed=y)

   # overdispersed counts: negative binomial
   with pm.Model():
       mu = pm.math.exp(X @ beta)
       alpha = pm.Exponential("alpha", 1)                   # dispersion; Poisson as alpha -> inf
       pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=counts)

Nested, so testable
---------------------

The elegance is that the standard model is a **special case** of its overdispersed version — Poisson is
negative-binomial with infinite dispersion, normal is :math:`t` with infinite :math:`\nu`. So fitting
the richer model and inspecting the extra parameter's posterior is a **direct test** of the restrictive
assumption: if :math:`\nu` is estimated small, the data are heavy-tailed and the normal was wrong; if the
negative-binomial dispersion is large, Poisson was adequate. You never have to choose in advance —
embed the standard model in its generalisation and let the posterior report which you needed.

The unifying idea
-------------------

Each overdispersed model adds **one parameter** for the variance the standard model fixed by fiat, and
partial pooling of that parameter connects straight back to the hierarchical stage: the police-stops
model *built* overdispersion from a normal error term, while the negative binomial *packages* the same
idea in closed form. Robustness, here, is just refusing to let a convenient distribution dictate how much
your data are allowed to vary.
"""

CONTENT["Posterior inference and computation"] = r"""
Fitting robust models
-----------------------

Heavy-tailed and overdispersed likelihoods are more expressive, and modestly harder to compute.
Two techniques — one classical, one modern — make them routine, and the classical one reveals *why* the
Student-:math:`t` works.

The t as a scale mixture of normals
-------------------------------------

The key representation: a Student-:math:`t` is a **normal whose variance is itself random**, drawn from
an inverse-gamma (equivalently, a normal whose precision is Gamma). Introduce a per-observation scale
:math:`\lambda_i`:

.. math::

   y_i \mid \lambda_i \sim \mathrm{N}(\mu, \sigma^2 / \lambda_i), \qquad
   \lambda_i \sim \mathrm{Gamma}(\nu/2, \nu/2)
   \quad \Longrightarrow \quad y_i \sim t_\nu(\mu, \sigma).

This is not just a computational trick; it is the **mechanism of robustness made visible**. Each
:math:`\lambda_i` is a weight, and an outlier is simply an observation whose posterior :math:`\lambda_i`
is **small** — its variance inflates, so it pulls on :math:`\mu` weakly. The model down-weights
aberrant points *automatically*, and the weights are readable: a small :math:`\lambda_i` flags exactly
which observations the model treated as outliers.

.. code-block:: python

   import pymc as pm
   # explicit scale-mixture form: the lambda_i ARE the outlier weights
   with pm.Model():
       nu = pm.Gamma("nu", 2, 0.1)
       lam = pm.Gamma("lam", nu / 2, nu / 2, shape=n)       # per-obs weights
       sigma = pm.HalfNormal("sigma", 1)
       pm.Normal("y", mu=X @ beta, sigma=sigma / pm.math.sqrt(lam), observed=y)
       # posterior mean of lam[i] << 1  <=>  observation i is an outlier

With augmentation, every conditional is standard, so **Gibbs** samples the model directly — the classical
route, and the reason robust models were tractable before HMC. In modern practice you simply write the
:math:`t` and let **NUTS** handle it; the mixture form is then a diagnostic and an explanation rather
than a necessity.

Computational cautions
------------------------

Two recur. The degrees of freedom :math:`\nu` is often **weakly identified** — data rarely pin down tail
weight precisely — so it needs a sensible prior (a Gamma keeping :math:`\nu` away from zero) and should
be reported with its full, often wide, posterior. And heavy-tailed models can induce **funnel** geometry
between :math:`\nu`, the scales, and the coefficients, so the non-centred parameterisation and the
divergence diagnostics of Stage 9 apply here too.

The payoff
------------

Robust computation costs little beyond a standard fit and returns two things: estimates that **do not
lurch** when one point is extreme, and an explicit account — through the weights :math:`\lambda_i` or the
posterior of :math:`\nu` — of *how much* robustness the data actually demanded. The next lesson puts the
whole apparatus to work on a familiar dataset.
"""

CONTENT["Robust inference for the eight schools"] = r"""
Robustness meets the canonical example
----------------------------------------

The eight-schools model has run through this book as the archetype of hierarchical inference. Here it
returns as a test of **robustness**: what happens to the pooling when the population distribution of
school effects is given **heavy tails** instead of the usual normal?

The standard model, recalled
------------------------------

The original assumes the true school effects are drawn from a normal population:

.. math::

   y_j \sim \mathrm{N}(\theta_j, \sigma_j^2), \qquad \theta_j \sim \mathrm{N}(\mu, \tau^2).

The normal population is what drives the shrinkage — every school pulled toward :math:`\mu` by an amount
set by :math:`\tau`. But it also encodes an assumption: that no school's true effect is a genuine
**outlier**. A normal population makes a school far from :math:`\mu` very improbable, so the model shrinks
it hard, whether or not that is warranted.

The robust version
--------------------

Replace the normal population with a :math:`t`:

.. math::

   \theta_j \sim t_\nu(\mu, \tau).

Now the population itself tolerates outliers. A school whose observed effect is genuinely extreme is
**shrunk less** — the heavy-tailed population grants that a true effect can lie far from the centre, so
the model does not force it back as aggressively. Ordinary schools are still pooled as before; only the
apparent outlier is treated differently.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu  = pm.Normal("mu", 0, 5)
       tau = pm.HalfCauchy("tau", 5)
       nu  = pm.Gamma("nu", 2, 0.1)                         # population tail weight
       eta = pm.StudentT("eta", nu=nu, mu=0, sigma=1, shape=8)   # non-centred, heavy-tailed
       theta = pm.Deterministic("theta", mu + tau * eta)
       pm.Normal("y", theta, sigma=sigma_j, observed=y)

What the comparison shows
---------------------------

Two lessons, both general. First, on the eight-schools data the robust and normal fits are **similar** —
none of the eight effects is a dramatic outlier, so heavy tails change little, which is itself
reassuring: the robust model does not distort an already-well-behaved analysis. Robustness that only
acts when needed is the goal. Second, the value of the exercise is the **sensitivity check** it
constitutes — fitting both and comparing is exactly the prior-robustness discipline of Stage 6, applied
to the population distribution. If the conclusions had diverged, that divergence would itself be the
finding: a signal that one school was driving the result and that the normal assumption was doing more
work than the data justified.

The closing point
-------------------

Robustness completes the hierarchical story. The normal population is a *choice*, and the
:math:`t`-population is the tool for asking how much that choice matters. Fitting the robust version
costs almost nothing and answers a question every hierarchical analysis should ask: are my pooled
estimates a property of the data, or an artefact of assuming no outliers? Stage 14 turns next from
outliers to the other great departure from clean data — **missingness**.
"""


MINDMAP.update({
    "Aspects of robustness": [
        "Overdispersed versions of standard models", "Continuous model expansion",
        "Weakly informative priors for logistic regression", "Robust inference for the eight schools",
    ],
    "Overdispersed versions of standard models": [
        "Aspects of robustness", "Posterior inference and computation",
        "Overdispersed Poisson regression for police stops", "Robust regression using t-distributed errors",
    ],
    "Posterior inference and computation": [
        "Overdispersed versions of standard models", "Robust regression using t-distributed errors",
        "Gibbs sampler", "Robust inference for the eight schools",
    ],
    "Robust inference for the eight schools": [
        "Example: parallel experiments in eight schools", "Aspects of robustness",
        "Posterior inference and computation", "Continuous model expansion",
    ],
})


# ----------------------------------------------------------------------
# Part V / Stage 14 — Robust Inference & Missing Data (cont.)
# ----------------------------------------------------------------------

CONTENT["Robust regression using t-distributed errors"] = r"""
Regression that ignores outliers
----------------------------------

Ordinary regression puts a **normal** on the residuals, so a single aberrant point drags the whole line
toward it — least squares is exquisitely sensitive to outliers in :math:`y`. **Robust regression**
replaces the normal error with a **Student-**:math:`t`, and the fix is a one-line change with an
outsized effect on reliability.

The model
-----------

Keep the linear predictor; change only the error distribution:

.. math::

   y_i \sim t_\nu\bigl(X_i \beta, \; \sigma\bigr).

The degrees of freedom :math:`\nu` set the tail weight, and can be **estimated** so the data report their
own outlier-proneness. Everything else — priors on :math:`\beta`, hierarchical extensions, the sampler —
is unchanged from the normal model; only the likelihood's tails have thickened.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])
       sigma = pm.HalfNormal("sigma", 1)
       nu = pm.Gamma("nu", 2, 0.1)                          # estimate tail weight; Gamma keeps nu > 0
       pm.StudentT("y", nu=nu, mu=X @ beta, sigma=sigma, observed=y)
       idata = pm.sample()

Why it works, and what to watch
---------------------------------

The mechanism is the scale-mixture of the previous lesson: a :math:`t` residual is a normal whose
variance can inflate for a single point, so an outlier is granted a **large private variance** and pulls
on the coefficients only weakly. The line is fitted to the **bulk** of the data. Because :math:`\nu` is
estimated, the model interpolates: heavy tails when outliers are present, near-normal when they are not.

Two cautions carry the honest limits. Robustness is against outliers in :math:`y`, **not in the
predictors** — a high-leverage point at an extreme :math:`x` can still distort the fit, because it sits
where the line has little data to argue with it. And down-weighting is not deletion: if an "outlier" is a
real signal (a regime change, a rare but valid case), the :math:`t` will quietly discount evidence you
might have wanted. Robust regression is the right default for messy continuous outcomes — reach for it
before least squares on real data — but it guards against one failure mode, not all of them.
"""

CONTENT["Notation"] = r"""
A language for missingness
----------------------------

Missing data need notation before they need methods, because the **right analysis depends entirely on
why** the data are missing. This lesson sets up the framework — due to Rubin — that makes "why" a precise,
model-able quantity rather than a vague worry.

The pieces
------------

Split the complete data into what you see and what you do not, and add an indicator for which is which:

.. math::

   y = (y_{\text{obs}}, y_{\text{mis}}), \qquad
   R_{ij} = \begin{cases} 1 & y_{ij} \text{ observed} \\ 0 & y_{ij} \text{ missing.} \end{cases}

The **missingness indicator** :math:`R` is itself data — a matrix you fully observe — and the object that
makes the theory work is its distribution :math:`p(R \mid y, \phi)`, the **missingness mechanism**. The
question is how :math:`R` depends on the values :math:`y`, including the ones you cannot see.

Three mechanisms
------------------

Rubin's taxonomy, in decreasing order of convenience:

* **MCAR — missing completely at random.** :math:`p(R \mid y) = p(R)`: missingness is independent of all
  data, observed and missing alike. Dropped records are then a random subsample, so complete-case
  analysis is unbiased (if wasteful). Rarely true.
* **MAR — missing at random.** :math:`p(R \mid y) = p(R \mid y_{\text{obs}})`: missingness depends only
  on **observed** values. Income missing more often for the young is MAR *if age is recorded*. This is
  the workhorse assumption.
* **MNAR — missing not at random.** Missingness depends on the **unobserved** values themselves — income
  missing because it is high. Here the mechanism cannot be ignored and must be modelled explicitly.

Ignorability
--------------

The payoff is a precise condition. When data are **MAR** and the missingness parameters :math:`\phi` are
distinct from the model parameters :math:`\theta`, the mechanism is **ignorable**: the term
:math:`p(R \mid y_{\text{obs}}, \phi)` factors out of the likelihood for :math:`\theta`, so you may model
the data and simply **ignore** :math:`R`. This is the *same* ignorability condition met in Stage 7 for
data collection — MAR plus parameter distinctness — now applied to missing values. Under it, Bayesian
inference proceeds by treating :math:`y_{\text{mis}}` as **unknown parameters** and integrating them out,
which is exactly what imputation does.

The catch
-----------

MAR versus MNAR **cannot be tested from the data** — distinguishing them needs the very values that are
missing. The assumption rests on subject knowledge (why *would* these be missing?) and, where it is
doubtful, on a **sensitivity analysis** across plausible MNAR mechanisms. The notation's value is exactly
this: it turns an untestable worry into an explicit assumption you can state, defend, and vary.
"""

CONTENT["Multiple imputation"] = r"""
Fill in the blanks, honestly
------------------------------

The obvious fix for a missing value — plug in a single best guess (the mean, a regression prediction) —
is quietly wrong: it treats an **estimate as if it were known**, so every downstream standard error comes
out too small. **Multiple imputation** repairs this by filling the gaps not once but **many** times,
carrying the uncertainty about the missing values all the way through to the final answer.

The three steps
-----------------

1. **Impute.** Draw :math:`m` complete datasets, each with the missing entries replaced by values sampled
   from their **posterior predictive distribution** given the observed data — genuine draws, not point
   estimates, so the imputations differ from one another.
2. **Analyse.** Run the intended analysis on each completed dataset separately, giving :math:`m`
   estimates :math:`\hat{Q}_k` and their variances :math:`U_k`.
3. **Pool.** Combine them with **Rubin's rules**.

.. math::

   \bar{Q} = \frac{1}{m}\sum_k \hat{Q}_k, \qquad
   \bar{U} = \frac{1}{m}\sum_k U_k, \qquad
   B = \frac{1}{m-1}\sum_k (\hat{Q}_k - \bar{Q})^2,

.. math::

   T = \bar{U} + \Bigl(1 + \tfrac{1}{m}\Bigr) B .

The total variance :math:`T` is the heart of it. :math:`\bar{U}` is the ordinary within-imputation
uncertainty; :math:`B`, the **between-imputation** variance, measures how much the answer wobbles as the
imputations change — and that term is **exactly the extra uncertainty due to missingness** that single
imputation throws away.

.. code-block:: python

   import numpy as np
   Qk = np.array([analyze(dataset) for dataset in imputed_datasets])   # m estimates
   Uk = np.array([variance(dataset) for dataset in imputed_datasets])  # m variances
   Q_bar = Qk.mean()
   U_bar = Uk.mean()
   B = Qk.var(ddof=1)                                                  # between-imputation
   T = U_bar + (1 + 1/len(Qk)) * B                                     # total variance
   se = np.sqrt(T)

The Bayesian reading
----------------------

Multiple imputation *is* a Bayesian computation in disguise: the missing values are **unknown
parameters**, and the :math:`m` completed datasets are **draws from their posterior**. In a fully
Bayesian model you get this for free — sample :math:`y_{\text{mis}}` jointly with :math:`\theta` in one
MCMC run, and the posterior already integrates over the missing values with no separate pooling step.
Rubin's rules are what you use when the analysis model and the imputation model are **separate programs**;
a joint Bayesian model makes them one.

Practicalities
----------------

A few settled points. Classic advice said :math:`m = 5`; modern practice prefers **twenty or more**,
since it is cheap and stabilises :math:`B`. The **imputation model must be at least as rich as the
analysis model** — if you will study an interaction, the imputation must include it, or the imputations
will erase it. And it all rests on **MAR**: multiple imputation handles ignorable missingness, and MNAR
still demands an explicit model. Within those limits it is the standard, principled way to keep
incomplete data from silently understating what you do not know.
"""

CONTENT["Missing data in the multivariate normal and t models"] = r"""
Imputation from a joint model
-------------------------------

The cleanest setting for missing-data methods is a set of continuous variables modelled **jointly** as
multivariate normal. There the imputation is not a heuristic but a **direct consequence** of the model:
the conditional distribution of the missing entries given the observed ones is available in closed form,
so imputing is exact.

The multivariate normal case
------------------------------

Model the complete rows as :math:`y_i \sim \mathrm{N}(\mu, \Sigma)`. For a row with some entries missing,
partition into observed and missing parts; the conditional of the missing given the observed is again
normal, with the familiar regression form:

.. math::

   y_{\text{mis}} \mid y_{\text{obs}} \sim \mathrm{N}\bigl(
     \mu_m + \Sigma_{mo}\Sigma_{oo}^{-1}(y_{\text{obs}} - \mu_o), \;
     \Sigma_{mm} - \Sigma_{mo}\Sigma_{oo}^{-1}\Sigma_{om}\bigr).

The conditional mean **regresses** the missing entries on the observed ones through the covariance, and
the conditional variance is what a proper imputation must sample from — plugging in only the mean would
be single imputation, discarding exactly the spread this formula provides.

Computation by data augmentation
----------------------------------

Since :math:`\mu` and :math:`\Sigma` are themselves unknown, alternate — the Gibbs pattern of Stage 9,
here with missingness as the latent layer:

.. math::

   \text{(I-step) } y_{\text{mis}} \sim p(y_{\text{mis}} \mid y_{\text{obs}}, \mu, \Sigma), \qquad
   \text{(P-step) } \mu, \Sigma \sim p(\mu, \Sigma \mid y_{\text{obs}}, y_{\text{mis}}).

Impute the missing values given current parameters; update the parameters given the completed data;
repeat. Each cycle produces both a posterior draw of the parameters and a **set of imputations** —
multiple imputation and parameter estimation from one algorithm.

.. code-block:: python

   import pymc as pm
   import numpy as np
   # PyMC imputes automatically when observed data are a masked array
   y_masked = np.ma.masked_invalid(Y)                       # NaNs flagged as missing
   with pm.Model():
       mu = pm.Normal("mu", 0, 5, shape=Y.shape[1])
       chol, _, _ = pm.LKJCholeskyCov("chol", n=Y.shape[1], eta=2,
                                      sd_dist=pm.HalfNormal.dist(1.0), compute_corr=True)
       pm.MvNormal("y", mu=mu, chol=chol, observed=y_masked)   # missing entries become parameters
       idata = pm.sample()                                     # posterior includes the imputations

The t extension, and the limits
---------------------------------

Swap the multivariate normal for a **multivariate** :math:`t` and the imputation becomes **robust**: the
model resists letting an outlier in the observed part of a row distort the imputed values in the rest —
the heavy-tailed lesson applied to filling gaps. Two limits remain. Everything here assumes **MAR**; the
joint model integrates over missing values correctly only when the mechanism is ignorable. And the
approach wants roughly **continuous, jointly-modellable** variables — genuinely mixed data (categorical
with continuous) are handled by chained-equations imputation or by more elaborate joint models, but the
principle is unchanged: **model the variables together, and the conditional distribution of the missing
given the observed is the imputation.**
"""


MINDMAP.update({
    "Robust regression using t-distributed errors": [
        "Posterior inference and computation", "Overdispersed versions of standard models",
        "Bayesian analysis of classical regression", "Aspects of robustness",
    ],
    "Notation": [
        "Data-collection models and ignorability", "Multiple imputation",
        "Missing data in the multivariate normal and t models", "Sample surveys",
    ],
    "Multiple imputation": [
        "Notation", "Missing data in the multivariate normal and t models",
        "Example: multiple imputation for a series of polls", "Missing values with counted data",
    ],
    "Missing data in the multivariate normal and t models": [
        "Multiple imputation", "Notation",
        "Multivariate Normal with Unknown Mean and Variance", "Robust regression using t-distributed errors",
    ],
})


# ----------------------------------------------------------------------
# Part V / Stage 14 — Robust Inference & Missing Data  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Example: multiple imputation for a series of polls"] = r"""
Missingness across many surveys
---------------------------------

Pre-election polling assembles many surveys, and no two ask exactly the same questions of exactly the
same people. Some respondents skip items; some questions appear in only a subset of polls; demographic
variables are recorded unevenly. Analysing the pooled series requires **filling these gaps coherently**,
and it is a natural showcase for multiple imputation at scale.

The structure
---------------

The data form a large, ragged table: respondents in rows, survey items in columns, with a scattered
pattern of missingness. Crucially the gaps are plausibly **MAR** — whether an item is missing depends on
*which survey* a respondent was in and on their recorded demographics, not on the hidden answer itself.
That is exactly the condition under which imputation is valid, and it holds here by design: the
missingness is a feature of survey construction, observed in the data.

The imputation model
----------------------

Fit a joint model rich enough to capture the relationships the analysis will use — demographic
predictors, survey indicators, and the correlations among responses — then draw completed datasets from
its posterior predictive distribution.

.. code-block:: python

   import numpy as np, pymc as pm
   Y = np.ma.masked_invalid(responses)                      # ragged missingness flagged
   with pm.Model():
       mu = pm.Normal("mu", 0, 5, shape=Y.shape[1])
       chol, _, _ = pm.LKJCholeskyCov("chol", n=Y.shape[1], eta=2,
                                      sd_dist=pm.HalfNormal.dist(1.0), compute_corr=True)
       pm.MvNormal("y", mu=mu, chol=chol, observed=Y)       # missing entries imputed
       idata = pm.sample()
   # each posterior draw carries a completed dataset; analyse across draws, not one

Why it matters here
---------------------

Three lessons the polling context makes vivid. **Completing the data unlocks the pooled analysis** — with
gaps filled coherently, every survey contributes to every question, rather than fragmenting into
question-by-question subsamples. The imputation **propagates uncertainty**: where a respondent's answer
was never observed, the spread across imputations widens the final interval honestly, so the pooled
estimate is not falsely precise. And it **respects survey structure** — conditioning on the survey
indicator means a poll that asked a question informs imputations for polls that did not, without
pretending they are identical. The same machinery underlies modern poll aggregation: incomplete,
heterogeneous surveys combined into one coherent picture, with missingness modelled rather than deleted.
"""

CONTENT["Missing values with counted data"] = r"""
Imputation beyond the normal
------------------------------

The multivariate-normal imputation of the previous stage assumed continuous variables. **Count** data —
disease cases, survey tallies, event frequencies — need imputation on their own terms: a fractional or
negative imputed count is nonsense, so the imputation model must respect the discreteness.

Model the counts directly
---------------------------

The Bayesian principle is unchanged: treat missing counts as **unknown parameters** and give them a
count likelihood, so imputations are proper draws from a Poisson or binomial rather than rounded normals.

.. math::

   y_i^{\text{mis}} \sim \mathrm{Poisson}(\lambda_i), \qquad
   \log \lambda_i = X_i \beta,

with the same linear-predictor machinery as the GLM stage. A missing count is drawn from its posterior
predictive Poisson, guaranteeing a non-negative integer, and — if the counts are overdispersed — a
**negative binomial** imputation carries the extra variance, exactly as robustness demanded for observed
counts.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       beta = pm.Normal("beta", 0, 1, shape=k)
       rate = pm.math.exp(X @ beta)
       # observed counts constrain beta; missing entries are drawn as integer parameters
       pm.Poisson("y", mu=rate, observed=y_counts_with_missing)   # masked array
       idata = pm.sample()

The offset subtlety
---------------------

Counts usually come with an **exposure** — population at risk, area, time — and a missing count often
sits beside a *known* exposure. The imputation must condition on it: impute the **rate** from the model,
then scale by the observed exposure to draw the count, so a small-population cell gets a correspondingly
small imputed count. Ignoring the offset would impute as if every cell had the same exposure, distorting
exactly the comparison the counts were meant to support.

Where it fits
---------------

The lesson generalises the point that **imputation inherits the likelihood**: use a normal model and you
impute normals; use a count model and you impute counts. Getting the imputation distribution right — its
support, its variance, its offset — is what keeps completed data coherent with the process that generated
them. It closes the mechanics of missing data; the next example puts the whole apparatus, ignorability
and imputation together, on a real survey.
"""

CONTENT["Example: an opinion poll in Slovenia"] = r"""
A referendum with missing answers
-----------------------------------

Before Slovenia's 1990 independence plebiscite, a survey asked citizens how they intended to vote — and,
as always, many respondents gave **no answer** to one or more questions. The stakes made the missing data
consequential: the result hinged on how "don't know" and non-response were treated, and the analysis is a
clean demonstration of ignorability and imputation deciding a real number.

The problem
-------------

Respondents answered (or did not) several related items — intention to vote, intended choice, attitude
toward independence. Simply **dropping** non-responders assumes their views match responders', which is
precisely the MCAR assumption that is rarely safe: people who decline to answer a charged political
question are plausibly different from those who answer. The question is whether the missingness is
**ignorable** given what *was* observed.

The model-based analysis
--------------------------

Treat the non-responses as **missing data under MAR** — missingness depending on observed covariates and
other answered items, not on the hidden intention itself — and build a joint model over the survey items.
Missing responses become parameters, imputed from their posterior predictive distribution given each
respondent's observed answers:

.. math::

   \Pr(\text{vote} = k \mid \text{observed items}) \; \text{modelled jointly, then}\;
   \theta = \sum_{\text{cells}} N_c \, \hat{p}_c \Big/ \sum_{\text{cells}} N_c ,

so each respondent's likely position is inferred from the pattern of their *other* answers, and the
overall estimate integrates over what they did not say.

.. code-block:: python

   import numpy as np, pymc as pm
   items = np.ma.masked_invalid(survey_items)               # non-responses flagged
   with pm.Model():
       # joint model over correlated categorical items; missing entries imputed
       ...                                                   # observed answers inform the gaps
       idata = pm.sample()
   # estimate turnout/support by integrating over imputed non-responses, with honest intervals

The lessons
-------------

Three, and they close the stage. **The missingness assumption changes the answer** — treating
non-responders as a random subsample (MCAR) versus modelling them from their observed answers (MAR) gives
materially different estimates, so the assumption must be stated, not buried in a default. **Ignorability
lets the observed answers speak for the missing ones**: a respondent's other items carry information
about the answer they withheld, and a joint model extracts it. And where MAR itself is in doubt — perhaps
non-response *is* informative about the vote — the honest response is a **sensitivity analysis** across
plausible mechanisms, reporting how much the conclusion moves. Real surveys are never complete; this is
what taking their gaps seriously looks like.
"""


# ----------------------------------------------------------------------
# Part V / Stage 15 — Nonlinear, Nonparametric & Basis Models
# ----------------------------------------------------------------------

CONTENT["Example: serial dilution assay"] = r"""
Nonlinear calibration, done properly
--------------------------------------

A serial dilution assay measures the concentration of a compound — an allergen, an antibody, a
pollutant — by reading an instrument at several **dilutions** of a sample and comparing against a
calibration curve. Gelman, Chew and Shnaidman used it to open the nonlinear-models stage because it
exposes, in one applied problem, why linear-normal thinking fails and what replaces it.

Why the standard approach wastes data
---------------------------------------

The relation between concentration and instrument reading is **nonlinear and heteroscedastic**: it
saturates at high concentrations and its measurement noise grows with the signal. The common practice
copes badly — it **discards** every reading that falls above or below the instrument's detection limits,
throwing away a large fraction of the data, and weights the survivors equally despite their unequal
precision. Estimates are noisy, and a sample whose readings all lie below the limit yields **no estimate
at all**.

The Bayesian model
--------------------

Model the full nonlinear calibration curve and the unknown concentrations **jointly**, with a variance
that grows with the mean. A standard form maps concentration :math:`x` to expected reading through a
four-parameter curve:

.. math::

   \mathrm{E}[y \mid x] = \beta_1 + \frac{\beta_2}{1 + (x/\beta_3)^{-\beta_4}}, \qquad
   \mathrm{sd}(y \mid x) \propto \bigl(\mathrm{E}[y \mid x]\bigr)^{\alpha},

the second equation encoding heteroscedasticity — noise scaling with signal — so measurements are
weighted by their actual precision rather than equally.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       b = pm.Normal("b", 0, 5, shape=4)                    # calibration-curve parameters
       alpha = pm.HalfNormal("alpha", 1)                    # variance-scaling exponent
       conc = pm.HalfNormal("conc", 10, shape=n_unknown)    # unknown concentrations (parameters)
       mu = b[0] + b[1] / (1 + (x / b[2]) ** (-b[3]))       # nonlinear curve
       sigma = pm.Deterministic("sigma", (mu ** alpha))     # signal-dependent noise
       pm.Normal("y", mu, sigma, observed=readings)         # ALL readings, none discarded
       idata = pm.sample()

Why it wins
-------------

Using **all** the data — including the below-detection readings, which are informative even when
imprecise — the joint model achieves **much lower standard errors** than the discard-and-average
standard, and returns an estimate even when every measurement is off-scale, where the classical method
simply fails. Three general lessons open the stage. Real relationships are often **nonlinear**, and
forcing linearity discards structure. Measurement noise is often **not constant**, and modelling its
dependence on the signal is what makes the weighting correct. And a **joint** model — curve and
concentrations together, with uncertainty flowing between them — beats a two-step fit that treats the
calibration as fixed. Part V builds on all three, moving from this parametric nonlinearity toward
splines, Gaussian processes and models with no fixed functional form at all.
"""


MINDMAP.update({
    "Example: multiple imputation for a series of polls": [
        "Multiple imputation", "Missing data in the multivariate normal and t models",
        "Missing values with counted data", "State-level opinons from national polls",
    ],
    "Missing values with counted data": [
        "Multiple imputation", "Standard generalized linear model likelihoods",
        "Overdispersed Poisson regression for police stops", "Example: an opinion poll in Slovenia",
    ],
    "Example: an opinion poll in Slovenia": [
        "Multiple imputation", "Notation",
        "Missing values with counted data", "Sample surveys",
    ],
    "Example: serial dilution assay": [
        "Splines and weighted sums of basis functions", "Example: population toxicokinetics",
        "Bayesian analysis of classical regression", "Unequal variances and correlations",
    ],
})


# ----------------------------------------------------------------------
# Part V / Stage 15 — Nonlinear, Nonparametric & Basis Models (cont.)
# ----------------------------------------------------------------------

CONTENT["Example: population toxicokinetics"] = r"""
A mechanistic model, fit hierarchically
-----------------------------------------

How does an inhaled solvent move through the body — absorbed, distributed among tissues, metabolised,
exhaled? Gelman, Bois and Jiang answered this for tetrachloroethylene with a model that fuses three
threads of this book: a **mechanistic** differential-equation model of physiology, a **hierarchical**
treatment of person-to-person variation, and **informative priors** grounded in real biology.

The physiological model
-------------------------

The body is divided into **compartments** — well-perfused tissues, poorly-perfused tissues, fat, and the
liver where metabolism occurs — and the flow of the compound between them is governed by a system of
**ordinary differential equations**. The parameters are physical quantities: blood flows, tissue
volumes, partition coefficients, metabolic rates. The concentration over time is the ODE solution, and
the likelihood compares it to measured blood or breath concentrations.

.. math::

   \frac{dC_{\text{tissue}}}{dt} = \frac{Q_{\text{tissue}}}{V_{\text{tissue}}}
     \Bigl(C_{\text{art}} - \frac{C_{\text{tissue}}}{\lambda_{\text{tissue}}}\Bigr) - (\text{metabolism}),

one such equation per compartment, coupled through the arterial concentration.

Hierarchy and physiological priors
------------------------------------

Each subject has their own physiological parameters, drawn from a **population** distribution — partial
pooling, so a subject with sparse data borrows from the group. And because every parameter is a genuine
physiological variable, **informative priors** are not a convenience but a duty: a tissue volume has a
known plausible range, a blood flow a measured typical value. The priors carry decades of physiology.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # population means of log physiological parameters, with literature-based priors
       mu = pm.Normal("mu", mu=prior_means, sigma=prior_sds, shape=n_params)
       tau = pm.HalfNormal("tau", 1, shape=n_params)
       theta = pm.Normal("theta", mu, tau, shape=(n_subj, n_params))   # per-subject, pooled
       C_pred = solve_pbpk_odes(theta, dosing)                         # ODE solution
       pm.Lognormal("y", mu=pm.math.log(C_pred), sigma=sigma, observed=concentration)

Why it is a landmark
----------------------

The analysis shows Bayesian inference **automatically propagating** the uncertainty in a large,
mechanistic, hierarchical model — the many parameters, the ODE nonlinearity, the person-to-person spread,
all carried into the predictions without approximation. Posterior predictive simulation checks the fit
and its sensitivity to the priors, exactly the discipline of Stage 6. The lesson for Part V: nonlinearity
here is **scientific**, dictated by physiology, not a flexible curve — and Bayesian methods let a genuine
mechanistic model be fit with honest uncertainty. The lessons that follow relax that structure, replacing
mechanism with flexible **basis functions** when the true form is unknown.
"""

CONTENT["Splines and weighted sums of basis functions"] = r"""
Flexibility from fixed pieces
-------------------------------

When a relationship is clearly nonlinear but no mechanism dictates its form, model it as a **weighted sum
of basis functions** — simple fixed building blocks whose linear combination can approximate any smooth
curve. The model stays **linear in its coefficients**, so all the regression machinery still applies;
only the design matrix changes.

The construction
------------------

Replace the predictor :math:`x` with a set of :math:`K` **basis functions** :math:`B_k(x)` and regress on
those:

.. math::

   f(x) = \sum_{k=1}^{K} \beta_k \, B_k(x), \qquad
   y_i = \sum_k \beta_k B_k(x_i) + \epsilon_i .

The bases are chosen, not learned. **B-splines** — smooth, local, bump-shaped functions each supported on
a short interval — are the standard choice: local support means each coefficient controls the curve only
near its knot, giving stable, interpretable flexibility. **Splines** stitch low-degree polynomials
together at **knots** with smoothness enforced at the joins.

.. code-block:: python

   import numpy as np, pymc as pm
   from patsy import dmatrix
   B = np.asarray(dmatrix("bs(x, df=12, degree=3)", {"x": x}))   # B-spline basis matrix
   with pm.Model():
       beta = pm.Normal("beta", 0, 1, shape=B.shape[1])          # basis coefficients
       pm.Normal("y", B @ beta, pm.HalfNormal("s", 1), observed=y)

The bias-variance knob
------------------------

The number of basis functions sets flexibility. **Too few** knots and the curve is too stiff to follow
the data — bias. **Too many** and it chases noise — variance, a wiggly overfit. The classical response is
to tune :math:`K` by cross-validation; the Bayesian response, in the next lesson, is more elegant — use
**many** basis functions and let a **prior shrink** their coefficients, so the data decide the effective
smoothness rather than a discrete knot count.

Why the basis view unifies
----------------------------

Splines are just **regression with a cleverly chosen design matrix**, which is why everything transfers:
priors on the coefficients, hierarchical structure, the same sampler. A basis expansion also connects
directly to the batching stage — the spline coefficients are a **batch** of related parameters, ripe for
a shared prior — and forward to Gaussian processes, which are the infinite-basis limit of this same idea.
Flexible curves, built from fixed pieces and fit by ordinary means.
"""

CONTENT["Basis selection and shrinkage of coe\ufb03cients"] = r"""
How smooth should the curve be?
---------------------------------

A basis expansion poses one question: how many functions, or equivalently how wiggly a fit? Choosing the
knot count by hand or by cross-validation is crude. The Bayesian answer inverts the problem — use
**many** basis functions and let a **prior on the coefficients** determine the effective smoothness,
so the data, not a discrete choice, set the flexibility.

The penalized-spline idea
---------------------------

Lay down a rich basis — more knots than you could need — and place a prior that **penalises roughness**.
The standard device is a prior on the **differences** between adjacent coefficients: if neighbours are
tied together, the curve cannot wiggle sharply.

.. math::

   \beta_k - \beta_{k-1} \sim \mathrm{N}(0, \tau^2), \quad\text{equivalently}\quad
   \beta_k \sim \mathrm{N}(\beta_{k-1}, \tau^2),

a **random-walk** prior on the coefficients. The smoothing scale :math:`\tau` is the one knob, and —
this is the elegant part — it is **estimated from the data** as an ordinary variance parameter. Small
:math:`\tau` forces a smooth curve; large :math:`\tau` permits wiggliness; its posterior finds the
balance.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       tau = pm.HalfNormal("tau", 1)                        # smoothing scale, ESTIMATED
       z = pm.Normal("z", 0, 1, shape=B.shape[1])
       beta = pm.Deterministic("beta", pm.math.cumsum(z * tau))   # random-walk (P-spline) prior
       pm.Normal("y", B @ beta, pm.HalfNormal("s", 1), observed=y)

Smoothness as a variance component
------------------------------------

The reframing is the whole point: **the amount of smoothing is a variance parameter**, so choosing it
becomes *estimating* it, with all the hierarchical machinery of this book. Adaptive smoothing —
different roughness in different regions — follows by letting :math:`\tau` vary. And for **selecting**
which basis functions matter, a **sparsity** prior (the horseshoe of the regularisation lesson) drives
irrelevant coefficients to zero, choosing the basis automatically.

Why it matters
----------------

Two payoffs. **No discrete model selection** — instead of fitting many knot counts and comparing, fit
once with a rich basis and let the prior shrink to the right complexity, uncertainty in the smoothness
included. And a **unifying view**: penalised splines are hierarchical models, their smoothing parameter a
variance component, so the batching and regularisation stages apply directly. This is the bridge to
Gaussian processes, where the basis becomes infinite and the smoothing prior becomes a covariance
function.
"""

CONTENT["Non-normal models and regression surfaces"] = r"""
Flexibility in two directions at once
---------------------------------------

Basis expansions handle a nonlinear function of **one** predictor. Two generalisations complete the
flexible-regression picture: non-normal **outcomes** with a flexible predictor, and flexible functions of
**several** predictors — a regression **surface** rather than a curve.

Flexible GLMs
--------------

The basis idea composes with the generalized linear model. Keep the link and the non-normal likelihood,
but let the linear predictor be a **spline** instead of a straight line — a logistic regression whose
log-odds is a smooth function of a continuous covariate, a Poisson whose log-rate bends:

.. math::

   g\bigl(\mathrm{E}[y_i]\bigr) = \sum_k \beta_k B_k(x_i),

with the smoothing prior of the previous lesson on the :math:`\beta_k`. Binary and count outcomes gain
flexible dose-response shapes without abandoning the GLM machinery.

Regression surfaces
---------------------

For a function of several predictors, one-dimensional bases combine. **Tensor-product** splines build a
multivariate basis from the products of univariate ones; **additive models** keep things tractable by
summing separate smooth functions, :math:`f(x_1, x_2) = f_1(x_1) + f_2(x_2)`, sacrificing interactions
for interpretability and far fewer parameters.

.. math::

   \mathrm{E}[y] = f_1(x_1) + f_2(x_2) + \cdots, \qquad f_j \text{ each a smooth spline.}

.. code-block:: python

   import numpy as np, pymc as pm
   # additive model: a smooth term per predictor, each with its own smoothing scale
   with pm.Model():
       contributions = []
       for j, Bj in enumerate(basis_matrices):             # one B-spline basis per predictor
           tau_j = pm.HalfNormal(f"tau_{j}", 1)
           z = pm.Normal(f"z_{j}", 0, 1, shape=Bj.shape[1])
           contributions.append(Bj @ (pm.math.cumsum(z) * tau_j))
       eta = sum(contributions)
       pm.Bernoulli("y", logit_p=eta, observed=y)           # flexible additive logistic

The curse, and the bridge
---------------------------

Flexibility in many dimensions meets the **curse of dimensionality**: a full tensor-product basis grows
exponentially with the number of predictors, so a surface over ten inputs is hopeless by basis expansion
alone. Additive models dodge it by forbidding interactions; the truly general tool handles arbitrary
smooth surfaces **without** an explicit growing basis — the **Gaussian process**, which specifies
smoothness through a covariance function over inputs and is the subject of the next lessons. Basis methods
take flexible regression a long way; Gaussian processes take the same idea to its infinite-dimensional
conclusion.
"""


MINDMAP.update({
    "Example: population toxicokinetics": [
        "Example: serial dilution assay", "Bayesian analysis of conjugate hierarchical models",
        "Hierarchical decision analysis for home radon", "Splines and weighted sums of basis functions",
    ],
    "Splines and weighted sums of basis functions": [
        "Basis selection and shrinkage of coe\ufb03cients", "Non-normal models and regression surfaces",
        "Assembling the matrix of explanatory variables", "Gaussian process regression",
    ],
    "Basis selection and shrinkage of coe\ufb03cients": [
        "Splines and weighted sums of basis functions", "Regularization and dimension reduction",
        "Regression coe\ufb03cients exchangeable in batches", "Non-normal models and regression surfaces",
    ],
    "Non-normal models and regression surfaces": [
        "Splines and weighted sums of basis functions", "Standard generalized linear model likelihoods",
        "Gaussian process regression", "Basis selection and shrinkage of coe\ufb03cients",
    ],
})


# ----------------------------------------------------------------------
# Part V / Stage 15 — Nonlinear, Nonparametric & Basis Models  [completes the stage]
# ----------------------------------------------------------------------

CONTENT["Gaussian process regression"] = r"""
A prior over functions
------------------------

Basis expansions build a flexible curve from a **finite** set of functions. A **Gaussian process** takes
the idea to its limit: a prior directly over **functions themselves**, with no fixed basis. Instead of
parameterising a curve and putting priors on coefficients, a GP places a distribution on the space of
smooth functions and conditions it on the data.

The definition
----------------

A Gaussian process is a distribution over functions such that the values at **any finite set of inputs**
are jointly multivariate normal. It is specified completely by two objects: a **mean function**
:math:`m(x)` (often taken as zero) and a **covariance function** — the **kernel** — :math:`k(x, x')`:

.. math::

   f \sim \mathcal{GP}\bigl(m(x), \, k(x, x')\bigr), \qquad
   \bigl(f(x_1), \dots, f(x_n)\bigr) \sim \mathrm{N}(m, K), \;\; K_{ij} = k(x_i, x_j).

The kernel is the whole model. It sets how strongly the function values at two inputs covary, and
therefore what kinds of functions are probable — smooth, wiggly, periodic, or trending.

The kernel encodes assumptions
--------------------------------

The most common choice, the **squared-exponential** kernel, encodes smooth, infinitely differentiable
functions:

.. math::

   k(x, x') = \sigma^2 \exp\!\left(-\frac{\lVert x - x' \rVert^2}{2 \ell^2}\right),

with two interpretable hyperparameters: the **length-scale** :math:`\ell` — how far apart inputs must be
before their function values decorrelate, i.e. how wiggly the curve — and the **signal variance**
:math:`\sigma^2` — the amplitude. Other kernels encode other beliefs: **Matérn** for rougher functions,
**periodic** for cycles, **linear** for trends. And kernels **compose** — sums and products of kernels
build structured functions from simple parts, the key to the next lesson.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       ell = pm.Gamma("ell", 2, 1)                          # length-scale
       eta = pm.HalfNormal("eta", 1)                        # amplitude
       cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)          # squared-exponential kernel
       gp = pm.gp.Marginal(cov_func=cov)
       gp.marginal_likelihood("y", X=x[:, None], y=y, sigma=pm.HalfNormal("s", 1))

Prediction and the cost
-------------------------

Conditioning the joint normal on observed points gives a **closed-form** posterior over the function at
new inputs — a predictive mean and, crucially, a predictive **variance** that widens away from the data,
so a GP says "I don't know" where it has seen nothing. The catch is computational: prediction requires
**inverting the** :math:`n \times n` **covariance matrix**, an :math:`O(n^3)` operation that becomes
prohibitive beyond a few thousand points. Sparse approximations, inducing points, and basis-function
approximations (the previous lesson, in reverse) are the standard escapes. What you buy for that cost is
nonparametric flexibility with calibrated uncertainty — a curve that adapts to the data and honestly
reports where it cannot.
"""

CONTENT["Example: birthdays and birthdates"] = r"""
Decomposing a time series with kernels
----------------------------------------

The number of babies born in the United States each day, over two decades, is a deceptively rich time
series — and the showcase, on the cover of *Bayesian Data Analysis*, for what additive Gaussian-process
kernels can do. The daily count carries several overlapping patterns at different timescales, and the GP
recovers them by **adding kernels**, one per component.

The components
----------------

Births vary on at least four rhythms, each captured by a kernel matched to its shape:

* a **long-term trend** over the years — a slow squared-exponential (large length-scale);
* a **seasonal** pattern within the year (more births in late summer) — a **periodic** kernel with a
  one-year period;
* a **day-of-week** effect (far fewer births on weekends, when scheduled deliveries pause) — a periodic
  kernel with a one-week period;
* **special days** — sharp dips on holidays, a spike or dip on dates like Valentine's Day and a notable
  avoidance of the 13th — short-scale deviations.

The additive model
--------------------

Because a sum of independent GPs is a GP whose kernel is the **sum** of the components' kernels, the whole
decomposition is a single Gaussian process:

.. math::

   f(t) = f_{\text{trend}}(t) + f_{\text{year}}(t) + f_{\text{week}}(t) + f_{\text{special}}(t),
   \qquad
   k = k_{\text{trend}} + k_{\text{year}} + k_{\text{week}} + k_{\text{special}} .

Fitting the sum and then examining each term's posterior **separates** the rhythms — the model discovers
the trend, the seasonal curve and the weekly dip individually, each with uncertainty.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # additive kernel: long trend + yearly + weekly periodic components
       k_trend = pm.HalfNormal("a1", 1)**2 * pm.gp.cov.ExpQuad(1, ls=pm.Gamma("l1", 4, 0.1))
       k_year  = pm.HalfNormal("a2", 1)**2 * pm.gp.cov.Periodic(1, period=365.25,
                                                                ls=pm.Gamma("l2", 2, 1))
       k_week  = pm.HalfNormal("a3", 1)**2 * pm.gp.cov.Periodic(1, period=7,
                                                                ls=pm.Gamma("l3", 2, 1))
       gp = pm.gp.Marginal(cov_func=k_trend + k_year + k_week)     # sum of GPs is a GP
       gp.marginal_likelihood("y", X=t[:, None], y=births, sigma=pm.HalfNormal("s", 1))

What it demonstrates
----------------------

Two lessons close out the GP method. **Kernel composition is a modelling language** — adding and
multiplying kernels expresses structural beliefs (a trend *plus* a cycle *times* a slow modulation)
directly in the covariance, and the posterior teases the pieces apart. And a Gaussian process turns
**time-series decomposition** into inference: trend, seasonality and calendar effects that classical
methods extract by separate procedures fall out of one coherent model, with propagated uncertainty on
every component. The birthday series is the vivid proof that flexible, structured, interpretable models
need not be built from a fixed basis.
"""

CONTENT["Latent Gaussian process models"] = r"""
Gaussian processes for non-Gaussian data
------------------------------------------

Gaussian-process regression as introduced assumes a **Gaussian** likelihood — continuous, normally-noised
observations. Most GP applications are not like that: the outcome is binary, a count, a category. A
**latent** Gaussian process handles them by putting the GP **inside** the model, on an unobserved
function that is then passed through a link and a non-Gaussian likelihood — exactly the GLM move, with a
GP where the linear predictor used to be.

The construction
------------------

Let a latent function :math:`f` have a GP prior, and let the observations depend on it through a link:

.. math::

   f \sim \mathcal{GP}(0, k), \qquad
   g\bigl(\mathrm{E}[y_i]\bigr) = f(x_i), \qquad
   y_i \sim \text{ExponentialFamily}.

For binary data this is **GP classification** — a logistic link on a smoothly varying latent function,
giving a probability surface that bends with the inputs. For counts it is a **log-Gaussian Cox process** —
a Poisson whose log-rate is a GP, the natural model for events whose intensity varies smoothly over space
or time.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       ell = pm.Gamma("ell", 2, 1); eta = pm.HalfNormal("eta", 1)
       cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)
       gp = pm.gp.Latent(cov_func=cov)                      # explicit latent function
       f = gp.prior("f", X=X)                               # the latent GP values
       pm.Bernoulli("y", p=pm.math.sigmoid(f), observed=y)  # non-Gaussian likelihood

The computational cost
------------------------

The price is real. With a Gaussian likelihood the latent function integrates out analytically; with a
non-Gaussian one it **cannot**, so :math:`f` must be sampled or approximated. This is precisely the
**latent Gaussian model** class that motivated **INLA** back in the computation stage — nested Laplace
approximation was built for exactly this structure, and is far faster than MCMC here. Sampling the latent
values with HMC also invites the funnel geometry of hierarchical models, so non-centred parameterisations
and careful diagnostics apply.

Why it matters
----------------

Latent GPs are the general-purpose tool for **flexible, non-Gaussian regression**: spatial disease
mapping (a Cox process over geography), smoothly-varying classification boundaries, time-varying rates.
They unify the threads of Part V — the GP supplies the flexible function, the link and likelihood handle
the data type, and the computation connects back to the approximation methods built earlier. Wherever a
smooth latent surface drives non-normal observations, a latent Gaussian process is the model.
"""

CONTENT["Functional data analysis"] = r"""
When each observation is a curve
----------------------------------

Sometimes a single data point is an entire **function**: a growth curve for one child, a daily
temperature profile, a spectrum, a gait cycle. **Functional data analysis** treats these curves as the
unit of analysis, and Bayesian methods bring to it what they bring everywhere — a generative model with
propagated uncertainty over the functions themselves.

Representing functions
------------------------

The observed curves are noisy, irregularly sampled realisations of underlying smooth functions, so the
first step is a **smooth representation**. Two routes, both from this stage: expand each curve in a
**basis** and model its coefficients, or model each curve as a draw from a **Gaussian process**. Either
turns an infinite-dimensional object into something estimable, with the smoothing priors of the earlier
lessons controlling how rough each curve may be.

.. math::

   y_{ij} = f_i(t_{ij}) + \epsilon_{ij}, \qquad
   f_i \sim \mathcal{GP}(\mu, k) \;\;\text{or}\;\; f_i(t) = \sum_k \beta_{ik} B_k(t),

with a **hierarchical** structure across curves: the :math:`f_i` share a common mean function and
covariance, so each individual curve borrows strength from the population — the batching idea, applied to
whole functions.

Analysing the functions
-------------------------

Once represented, the functions become objects to explore. **Functional principal components** find the
dominant modes of variation — the few characteristic ways the curves differ from their mean, a
dimension reduction on function space. Curves can be **registered** (aligned in time to separate
variation in *shape* from variation in *timing*), regressed on covariates (**function-on-scalar**), or
used as predictors (**scalar-on-function**).

.. code-block:: python

   import pymc as pm
   # hierarchical functional model: each curve a GP deviation around a shared mean
   with pm.Model():
       mean_coef = pm.Normal("mean_coef", 0, 1, shape=K)         # population mean curve
       tau = pm.HalfNormal("tau", 1)
       dev = pm.Normal("dev", 0, tau, shape=(n_curves, K))       # per-curve deviations, pooled
       f = (mean_coef + dev) @ B.T                               # individual smooth curves
       pm.Normal("y", f[curve_id, t_idx], pm.HalfNormal("s", 1), observed=y)

Where it sits
---------------

Functional data analysis is the natural **culmination** of the flexible-modelling stage: it combines
basis expansions and Gaussian processes (to represent curves), hierarchical models (to pool across
curves), and dimension reduction (to summarise them) into one framework for data whose fundamental unit
is a function. It closes Part V's parametric-nonlinear thread — from mechanistic ODE models through
splines and Gaussian processes to whole-function data — and hands off to the final stage, where the
flexibility comes not from smooth functions but from **mixtures** and infinite-dimensional
**nonparametric** priors.
"""


MINDMAP.update({
    "Gaussian process regression": [
        "Non-normal models and regression surfaces", "Example: birthdays and birthdates",
        "Latent Gaussian process models", "Splines and weighted sums of basis functions",
    ],
    "Example: birthdays and birthdates": [
        "Gaussian process regression", "Latent Gaussian process models",
        "Unequal variances and correlations", "Functional data analysis",
    ],
    "Latent Gaussian process models": [
        "Gaussian process regression", "Other approximations",
        "Standard generalized linear model likelihoods", "Functional data analysis",
    ],
    "Functional data analysis": [
        "Gaussian process regression", "Basis selection and shrinkage of coe\ufb03cients",
        "Hierarchical models for batches of variance components", "Latent Gaussian process models",
    ],
})


# ----------------------------------------------------------------------
# Part V / Stage 16 — Mixtures & Nonparametric Models
# ----------------------------------------------------------------------

CONTENT["Density estimation and regression"] = r"""
Flexibility from components, not curves
-----------------------------------------

Part V has pursued flexibility through smooth functions — splines, Gaussian processes. The final stage
takes a different route: build flexible distributions and regressions by **combining simple components**.
This lesson frames the two problems the stage solves, and the shift in thinking that unites them.

Two nonparametric problems
----------------------------

**Density estimation** asks for the whole distribution of a variable, :math:`p(y)`, when no parametric
family fits — a distribution with several bumps, heavy skew, or unknown shape. **Density regression**
goes further: it lets the *entire* conditional distribution :math:`p(y \mid x)` change with predictors,
not merely its mean. A standard regression models :math:`\mathrm{E}[y \mid x]` and assumes the rest of
the shape is fixed; density regression frees the variance, the skew, even the number of modes to vary
with :math:`x`.

The mixture idea
------------------

Both are solved by **mixtures**. Any sufficiently complex distribution can be approximated by a **weighted
sum of simple components** — usually normals:

.. math::

   p(y) = \sum_{k=1}^{K} \pi_k \, \mathrm{N}(y \mid \mu_k, \sigma_k^2), \qquad
   \sum_k \pi_k = 1 .

With enough components a mixture of normals can approximate **any** continuous density to arbitrary
accuracy — the theoretical guarantee that makes mixtures a general-purpose nonparametric tool. The
components need not correspond to real subpopulations; they are basis elements for building a shape.

.. code-block:: python

   import numpy as np, pymc as pm
   with pm.Model():
       w = pm.Dirichlet("w", a=np.ones(K))                  # mixture weights, sum to 1
       mu = pm.Normal("mu", 0, 5, shape=K)
       sigma = pm.HalfNormal("sigma", 1, shape=K)
       pm.NormalMixture("y", w=w, mu=mu, sigma=sigma, observed=y)   # flexible density

Two readings, and the plan
----------------------------

Mixtures carry a productive ambiguity. **Model-based clustering** reads each component as a real
**subpopulation** — the mixture *discovers groups*, and the membership of each point is the inference of
interest. **Density estimation** reads the components as mere **building blocks** for an arbitrary shape,
with no claim that they are real. The same model, two purposes; which one you intend governs how you
interpret the fit. The stage builds from finite mixtures (this lesson and the next few) to the case of an
**unknown** number of components, and finally to **infinite** mixtures — the Dirichlet process — where the
component count is itself learned. Flexibility, assembled from simple parts.
"""

CONTENT["Setting up and interpreting mixture models"] = r"""
The latent-class formulation
------------------------------

A mixture becomes easy to fit once written with an explicit **membership indicator**. Introduce, for each
observation, an unobserved label :math:`z_i` saying which component generated it:

.. math::

   z_i \sim \mathrm{Categorical}(\pi), \qquad
   y_i \mid z_i = k \sim \mathrm{N}(\mu_k, \sigma_k^2).

Marginalising :math:`z_i` recovers the weighted-sum density of the previous lesson, but *keeping*
:math:`z_i` as a latent variable turns an awkward sum inside the likelihood into a two-level model with
simple conditionals — the same data-augmentation trick used for the :math:`t` distribution and for probit
regression.

Fitting: EM and Gibbs
-----------------------

The augmented form yields two classic algorithms, both alternating between the labels and the parameters.
**EM** (from the modal-approximation stage) computes, in its E-step, the posterior **responsibilities**
:math:`\Pr(z_i = k \mid y_i)` — soft assignments — then in the M-step updates :math:`(\pi, \mu, \sigma)`
given them. **Gibbs** does the Bayesian version: sample each :math:`z_i` from its conditional, then sample
the parameters given the assignments, and repeat.

.. code-block:: python

   import numpy as np
   # EM for a Gaussian mixture: responsibilities, then weighted updates
   for _ in range(n_iter):
       # E-step: responsibility of component k for point i
       r = pi * norm_pdf(y[:, None], mu, sigma)             # (n, K)
       r /= r.sum(axis=1, keepdims=True)
       # M-step: update weights, means, sds from soft assignments
       Nk = r.sum(axis=0)
       pi = Nk / len(y)
       mu = (r * y[:, None]).sum(0) / Nk
       sigma = np.sqrt((r * (y[:, None] - mu)**2).sum(0) / Nk)

Interpreting the fit
----------------------

The responsibilities are the useful output when the goal is **clustering**: each :math:`\Pr(z_i = k)` is
a *soft* membership, honestly expressing that a point between two components belongs partly to each —
better than a hard assignment that hides the ambiguity. When the goal is **density estimation**, the
labels are a computational device and only the resulting smooth density matters.

Cautions
----------

Two recur through the stage. Mixture likelihoods are **multimodal** — genuinely, beyond label switching —
so optimisation and sampling can stick in local solutions; multiple starts and careful diagnostics are
needed. And a component's variance can **collapse** toward zero as it latches onto a single point, sending
the likelihood to infinity — the same boundary pathology met in hierarchical models, cured the same way,
with a prior that keeps variances off zero. The next lesson puts the setup to work; the one after
confronts label switching head-on.
"""

CONTENT["Example: reaction times and schizophrenia"] = r"""
A mixture with a scientific meaning
-------------------------------------

Not every mixture is a mere density approximation — sometimes the components are **real and
interpretable**. A study of reaction times, comparing schizophrenic patients with a control group,
provides the archetype: the mixture structure encodes a genuine hypothesis about how the mind produces
the data.

The finding, and the model
----------------------------

Schizophrenic patients show more variable reaction times than controls, but the excess is not a uniform
slowing. The data suggest that on **most** trials a patient responds like a control, while on a
**minority** of trials an attentional lapse produces a much slower response. That is a **mixture within
each patient**: two components, a "normal" reaction-time distribution and a "delayed" one, with a
patient-specific probability of lapsing.

.. math::

   y_{ij} \sim \lambda_i \, \mathrm{N}(\mu_i, \sigma^2)
              + (1 - \lambda_i) \, \mathrm{N}(\mu_i + \tau, \sigma^2),

where :math:`\lambda_i` is patient :math:`i`'s probability of a normal (non-lapse) response and
:math:`\tau > 0` the extra delay when attention lapses. The parameters carry meaning: :math:`\tau` is the
size of a lapse, :math:`\lambda_i` how often patient :math:`i` lapses.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # hierarchical: each patient's lapse probability, pooled across patients
       lam = pm.Beta("lam", 2, 2, shape=n_patients)         # P(normal response)
       mu = pm.Normal("mu", 0, 5, shape=n_patients)
       tau = pm.HalfNormal("tau", 5)                         # lapse delay (ordered => identified)
       w = pm.math.stack([lam[pid], 1 - lam[pid]], axis=1)
       comp = [pm.Normal.dist(mu[pid], sigma), pm.Normal.dist(mu[pid] + tau, sigma)]
       pm.Mixture("y", w=w, comp_dists=comp, observed=rt)

Why this example matters
--------------------------

Three points. The components are **substantive**, not basis elements — "normal response" and
"attentional lapse" are the hypothesis, and :math:`\lambda_i`, :math:`\tau` are the quantities of
scientific interest. The **hierarchy** ties it to the rest of the book: lapse probabilities are pooled
across patients, so a patient with few trials borrows strength. And the ordering :math:`\tau > 0`
**identifies** the components — the delayed component is, by construction, the slower one — which
sidesteps the label-switching problem the next lesson tackles in general. A mixture, here, is a **model of
a mechanism**: two cognitive states, their sizes and frequencies estimated from response times.
"""

CONTENT["Label switching and posterior computation"] = r"""
The symmetry problem
----------------------

Mixture models carry a computational pathology absent from everything before them. If the components
share a prior and come from the same family, then **permuting the component labels leaves the model
unchanged** — calling component 1 "component 2" and vice versa gives an identical likelihood and prior.
The posterior is therefore invariant under all :math:`K!` relabellings, and this is **label switching**.

Why it breaks inference
-------------------------

The posterior has :math:`K!` identical modes, one per permutation. A sampler that mixes well **visits all
of them**, so a chain's draws for "the mean of component 1" jump between what are really different
components — averaging them gives the overall mean of all components, a meaningless number. Any inference
that refers to a component *by its label* is corrupted: the marginal posterior of :math:`\mu_1` is
identical to that of :math:`\mu_2`, both equal to the mixture of all components' means.

.. code-block:: python

   import numpy as np
   # symptom: per-component means are identical across the chain because labels swap
   mu = idata.posterior["mu"].values.reshape(-1, K)
   mu.mean(axis=0)          # all K entries nearly equal  <-  label switching

Solutions
-----------

Three standard remedies, in rough order of robustness:

* **Ordering constraint.** Impose an identifying restriction — :math:`\mu_1 < \mu_2 < \cdots` or ordered
  weights — that picks one labelling. Simple and often enough, as the reaction-time delay
  (:math:`\tau > 0`) showed; but a poor choice of ordering variable can distort a genuinely multimodal
  posterior.
* **Post-hoc relabelling.** Let the sampler run unconstrained, then **permute each draw** to a common
  labelling by solving an assignment problem — matching components across draws by their parameters.
  More flexible than a hard constraint and the standard modern approach.
* **Label-invariant summaries.** Report only quantities that do **not** depend on labels — the density
  :math:`p(y)`, the number of components, the clustering of observations (which points group together) —
  all unchanged by permutation.

The deeper lesson
-------------------

Which fix to use follows from the **goal**. For **density estimation**, label switching is a *non-problem*:
the estimated density is label-invariant, so an unconstrained chain gives exactly the right answer with
nothing to fix. It bites only for **component-specific** interpretation — "what is the mean of the second
group?" — which presupposes the components are real and distinguishable. So the presence of a
label-switching problem is a signal to ask whether component-level inference is even meaningful for the
question, or whether a label-invariant summary answers it directly.
"""


MINDMAP.update({
    "Density estimation and regression": [
        "Functional data analysis", "Setting up and interpreting mixture models",
        "Bayesian histograms", "Dirichlet process prior distributions",
    ],
    "Setting up and interpreting mixture models": [
        "Density estimation and regression", "Example: reaction times and schizophrenia",
        "Label switching and posterior computation", "Finding marginal posterior modes using EM",
    ],
    "Example: reaction times and schizophrenia": [
        "Setting up and interpreting mixture models", "Label switching and posterior computation",
        "Exchangeability and hierarchical models", "Mixture models for classification and regression",
    ],
    "Label switching and posterior computation": [
        "Setting up and interpreting mixture models", "Unspecified number of mixture components",
        "Example: reaction times and schizophrenia", "Density estimation and regression",
    ],
})


# ----------------------------------------------------------------------
# Part V / Stage 16 — Mixtures & Nonparametric Models (cont.)
# ----------------------------------------------------------------------

CONTENT["Unspecified number of mixture components"] = r"""
How many components?
----------------------

A finite mixture demands a number :math:`K` of components fixed in advance — but that number is usually
**unknown**, and often it is the very question (how many subtypes, how many clusters?). Choosing
:math:`K` badly wastes the model: too few and it cannot fit; too many and empty or duplicated components
appear. This lesson surveys the ways of letting the data speak to :math:`K`, and motivates the infinite
mixture that follows.

Selection versus inference
----------------------------

The crude approach **selects** :math:`K`: fit several values and compare by cross-validated predictive
accuracy or an information criterion (Stage 6). It works, but treats a genuine unknown as a tuning knob
and ignores uncertainty in :math:`K` itself. The Bayesian instinct is to make :math:`K` a **parameter**
and infer it — putting a prior on the number of components and letting the posterior weigh the evidence.

Trans-dimensional inference
-----------------------------

Making :math:`K` random means the parameter space changes dimension with :math:`K` — more components,
more means and weights. **Reversible-jump MCMC** (from the extensions-to-Gibbs lesson) navigates this by
proposing **births and deaths** of components, with a Jacobian correcting for the dimension change. It is
general and powerful, and notoriously **fiddly**: good birth-death moves are hard to design, especially
in high dimensions, and mixing can be poor.

.. code-block:: python

   # mixture of finite mixtures: put a prior on K, infer it with the rest
   #   K ~ prior on component count (e.g. Poisson-like)
   #   given K:  weights ~ Dirichlet_K,  components ~ base distribution
   # reversible-jump or allocation samplers explore across K
   # -- in practice, the overfitted / infinite-mixture route below is often simpler

The overfitted-mixture shortcut
---------------------------------

A pragmatic alternative sidesteps trans-dimensional moves: fit a mixture with **more components than
needed** and a prior on the weights that **empties** the surplus ones. With a sparse Dirichlet prior
(small concentration), unneeded components receive near-zero weight, and the **number of occupied**
components estimates the effective :math:`K` — all at fixed dimension, so ordinary MCMC suffices. This
"sparse finite mixture" is a bridge to the cleaner idea: rather than overfit and prune, start with
**infinitely many** components and let the data occupy as many as they need.

Toward the infinite mixture
-----------------------------

Each route here strains against the same awkwardness — :math:`K` as a discrete unknown resists tidy
inference. The elegant resolution, in the Dirichlet-process lessons ahead, is to **stop bounding**
:math:`K` at all: a prior over mixtures with a countably infinite number of components, where any finite
dataset uses a finite but **unbounded, data-determined** number. The question "how many components?"
dissolves into "how many does the data reveal?"
"""

CONTENT["Mixture models for classification and regression"] = r"""
Mixtures for supervised learning
----------------------------------

Mixtures are usually met in **unsupervised** settings — clustering, density estimation. The same
structure powers **supervised** learning too: classification and regression gain flexibility when the
model is a mixture, either of the classes themselves or of local expert models. The latent-component idea
carries directly across.

Mixture discriminant analysis
-------------------------------

Ordinary discriminant analysis models each class as a single Gaussian — too rigid when a class is itself
**heterogeneous** (handwritten "4"s come in two styles; a disease has subtypes). Model each class as its
**own mixture** of Gaussians, and the decision boundary bends to the real, multimodal structure:

.. math::

   p(x \mid y = c) = \sum_{k=1}^{K_c} \pi_{ck} \, \mathrm{N}(x \mid \mu_{ck}, \Sigma_{ck}), \qquad
   \Pr(y = c \mid x) \propto p(x \mid y = c) \, \Pr(y = c),

with classification by the posterior class probability. Each class density is as flexible as it needs to
be, and the Bayesian fit carries uncertainty into the predicted class probabilities.

Mixture of experts
--------------------

For regression, a **mixture of experts** lets *different regressions hold in different regions* of the
predictor space. Several "expert" models each fit part of the space, and a **gating** function — itself a
function of the inputs — decides which expert governs where:

.. math::

   p(y \mid x) = \sum_{k=1}^{K} g_k(x) \, p_k(y \mid x), \qquad \sum_k g_k(x) = 1,

the gate :math:`g_k(x)` a softmax over the inputs. The result is a flexible regression that can switch
regime with :math:`x` — piecewise-linear where the experts are linear, but with soft, learned boundaries.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # gate: input-dependent component probabilities (softmax)
       Wg = pm.Normal("Wg", 0, 1, shape=(k_experts, X.shape[1]))
       g = pm.math.softmax(X @ Wg.T, axis=1)                # which expert, per input
       beta = pm.Normal("beta", 0, 1, shape=(k_experts, X.shape[1]))   # each expert's slope
       mu = (g * (X @ beta.T)).sum(axis=1)                  # gated combination
       pm.Normal("y", mu, pm.HalfNormal("s", 1), observed=y)

Why mixtures help here
------------------------

Two reasons close the supervised case. Mixtures grant **local flexibility** — different structure in
different regions or classes — without a global nonlinear form, and the components can carry
interpretation (a subtype, a regime). And because the whole apparatus is Bayesian, the **uncertainty in
which component applies** flows into predictions: a point near a class boundary or an expert's edge gets
an honestly wider predictive distribution. Mixtures thus extend supervised learning the same way they
extended density estimation — flexibility assembled from simple, interpretable local pieces.
"""

CONTENT["Bayesian histograms"] = r"""
The simplest nonparametric density
-------------------------------------

Before the elaborate machinery, the humblest density estimator deserves a Bayesian treatment: the
**histogram**. Partition the range into bins and estimate the probability in each. A Bayesian histogram
puts a **prior on the bin probabilities** and reports a posterior — turning a familiar picture into a
model with honest uncertainty, and illustrating nonparametric ideas at their most transparent.

The model
-----------

With :math:`B` bins and counts :math:`n_b` of the :math:`N` observations falling in each, the natural
model is multinomial with a **Dirichlet prior** on the bin probabilities:

.. math::

   (n_1, \dots, n_B) \sim \mathrm{Multinomial}(N, p), \qquad
   p \sim \mathrm{Dirichlet}(\alpha_1, \dots, \alpha_B).

Conjugacy makes the posterior immediate — :math:`p \mid n \sim \mathrm{Dirichlet}(\alpha_b + n_b)` — so
each bin's height comes with a full posterior, and sparsely-populated bins are **smoothed** toward the
prior instead of estimated as zero. The Dirichlet concentration controls that smoothing.

.. code-block:: python

   import numpy as np
   from scipy import stats
   counts, edges = np.histogram(y, bins=B)
   alpha = np.ones(B) * 1.0                                  # symmetric Dirichlet prior
   post = stats.dirichlet(alpha + counts)                    # conjugate posterior on bin probs
   heights = post.mean() / np.diff(edges)                    # density estimate
   draws = post.rvs(500) / np.diff(edges)                    # posterior uncertainty bands

Smoothing and the bin-width problem
-------------------------------------

The histogram's perennial weakness is the **bin width** — too wide oversmooths, too narrow is noisy.
Bayesian versions address it by borrowing this stage's tools: a **prior favouring smoothness** across
adjacent bins (neighbouring probabilities tied, as in the P-spline lesson) regularises the jagged
edges; the bin count or width can itself be **inferred** rather than fixed; and finer partitions can be
combined hierarchically. **Pólya trees** carry the idea to its logical end — a *nested*, recursively
refined binning with a prior at every level, giving a genuine nonparametric prior on densities built
from binary splits.

Where it fits
---------------

The Bayesian histogram is the **conceptual bridge** into nonparametrics. It shows the essential moves —
a prior over a flexible, high-dimensional object (the bin probabilities), a posterior that **smooths where
data are thin**, and uncertainty on the whole estimated density — in the simplest possible container. The
Dirichlet prior here is the finite ancestor of the **Dirichlet process** of the next lessons, which
replaces a fixed grid of bins with an adaptive, unbounded partition. From counting in boxes to infinite
mixtures is one continuous idea.
"""

CONTENT["Dirichlet process prior distributions"] = r"""
A prior over distributions
----------------------------

The finite mixtures of this stage all fixed, or struggled to infer, the number of components. The
**Dirichlet process** removes the bound entirely: it is a prior over **probability distributions
themselves**, supporting a *countably infinite* number of components, of which any finite dataset uses a
finite but data-determined number. It is the foundation of Bayesian nonparametric modelling.

The definition
----------------

A Dirichlet process :math:`\mathrm{DP}(\alpha, H)` is specified by two things: a **base distribution**
:math:`H`, the "mean" around which the random distribution is centred, and a **concentration parameter**
:math:`\alpha > 0`, governing how much a draw deviates from :math:`H`. A draw :math:`G \sim
\mathrm{DP}(\alpha, H)` is itself a (discrete) probability distribution — the DP is a distribution *over
distributions*.

The stick-breaking construction
---------------------------------

The DP becomes concrete through **stick-breaking**, which builds :math:`G` explicitly. Start with a unit
stick; repeatedly break off a Beta-distributed fraction; the pieces are the mixture weights, each
attached to a location drawn from :math:`H`:

.. math::

   \beta_k \sim \mathrm{Beta}(1, \alpha), \quad
   \pi_k = \beta_k \prod_{j=1}^{k-1}(1 - \beta_j), \quad
   \theta_k \sim H, \qquad
   G = \sum_{k=1}^{\infty} \pi_k \, \delta_{\theta_k}.

The weights :math:`\pi_k` **decay** (each break takes a fraction of what remains), so although there are
infinitely many components, a handful carry most of the mass — the reason a finite dataset engages only
finitely many.

.. code-block:: python

   import numpy as np
   def stick_breaking(alpha, K_trunc):                      # truncated DP weights
       betas = np.random.beta(1, alpha, size=K_trunc)
       remaining = np.concatenate([[1.0], np.cumprod(1 - betas)[:-1]])
       return betas * remaining                             # pi_k, summing to ~1

   # small alpha -> few dominant components; large alpha -> many, closer to H

The concentration parameter
-----------------------------

:math:`\alpha` tunes complexity. **Small** :math:`\alpha` concentrates the mass on a **few** components,
so the data are explained by few clusters; **large** :math:`\alpha` spreads it over **many**, approaching
the base distribution :math:`H`. Because :math:`\alpha` controls the expected number of occupied
components, it can be given its own prior and **inferred** — the model learns not just the components but
how many to use.

Why it matters
----------------

The DP dissolves the model-selection problem this stage opened with. There is **no** :math:`K` to choose
or to jump between: the number of components is unbounded a priori and **determined by the data** a
posteriori, growing gracefully as more data arrive. What remains is to attach a likelihood to each
component — placing a smooth kernel at each atom :math:`\theta_k` — which is the **Dirichlet process
mixture** of the next lesson, the workhorse of nonparametric density estimation and clustering.
"""


MINDMAP.update({
    "Unspecified number of mixture components": [
        "Setting up and interpreting mixture models", "Dirichlet process prior distributions",
        "Model comparison based on predictive performance", "Label switching and posterior computation",
    ],
    "Mixture models for classification and regression": [
        "Setting up and interpreting mixture models", "Standard generalized linear model likelihoods",
        "Example: reaction times and schizophrenia", "Density regression",
    ],
    "Bayesian histograms": [
        "Density estimation and regression", "Dirichlet process prior distributions",
        "Setting up and interpreting mixture models", "Continuous model expansion",
    ],
    "Dirichlet process prior distributions": [
        "Unspecified number of mixture components", "Dirichlet process mixtures",
        "Bayesian histograms", "Bayesian analysis of conjugate hierarchical models",
    ],
})


# ----------------------------------------------------------------------
# Part V / Stage 16 — Mixtures & Nonparametric Models  [completes the course]
# ----------------------------------------------------------------------

CONTENT["Dirichlet process mixtures"] = r"""
Putting a likelihood on the process
-------------------------------------

A Dirichlet process draw is a **discrete** distribution — a countable set of atoms — so it cannot model
continuous data directly. The **Dirichlet process mixture** (DPM) fixes this by using the DP not for the
data but for the **parameters**: each atom :math:`\theta_k` is the parameter of a smooth kernel, and the
data are drawn from the resulting infinite mixture. This is *the* workhorse of Bayesian nonparametrics.

The model
-----------

Draw a random distribution from the DP; draw each observation's parameter from it; draw the observation
from a kernel at that parameter:

.. math::

   G \sim \mathrm{DP}(\alpha, H), \qquad
   \theta_i \sim G, \qquad
   y_i \sim f(y \mid \theta_i).

Because :math:`G` is discrete, the :math:`\theta_i` **repeat** — several observations share the same
drawn value — and points sharing a :math:`\theta_i` form a **cluster**. With a normal kernel this is an
infinite mixture of Gaussians whose number of occupied components is learned, dissolving the
:math:`K`-selection problem entirely.

The Chinese restaurant process
--------------------------------

Integrating :math:`G` out gives the **Chinese restaurant process**, the DPM's computational heart and its
clearest intuition. Customers (observations) enter a restaurant and choose tables (clusters): a new
customer joins an existing table with probability proportional to how many already sit there, and starts
a **new** table with probability proportional to :math:`\alpha`:

.. math::

   \Pr(\text{join table } k) \propto n_k, \qquad
   \Pr(\text{new table}) \propto \alpha .

This "rich get richer" rule produces a few large clusters and a tail of small ones, and — crucially —
lets the number of clusters **grow with the data**. It is the basis of the collapsed Gibbs samplers
(Neal's algorithms) that fit DPMs by reseating one customer at a time.

.. code-block:: python

   import pymc as pm
   K = 20                                                   # truncation level (generous)
   with pm.Model():
       alpha = pm.Gamma("alpha", 1, 1)                      # concentration, inferred
       beta = pm.Beta("beta", 1, alpha, shape=K)            # stick-breaking fractions
       w = pm.Deterministic("w", beta * pm.math.concatenate(
           [[1.0], pm.math.cumprod(1 - beta)[:-1]]))        # mixture weights
       mu = pm.Normal("mu", 0, 5, shape=K)                  # atom locations ~ H
       pm.NormalMixture("y", w=w, mu=mu, sigma=pm.HalfNormal("s", 1, shape=K), observed=y)

Why it matters
----------------

The DPM delivers what the stage promised: **density estimation and clustering with the number of
components learned, not chosen**. It adapts complexity to the data — more data can reveal more clusters —
and gives a full posterior over partitions, honestly expressing uncertainty about *how many* groups there
are and *which* points belong together. In practice a **truncated** stick-breaking (a generous finite
:math:`K` with most weights near zero) makes it fit with standard HMC. From here the nonparametric idea
extends outward: to functionals beyond the density, to shared clustering across groups, and to
covariate-dependent distributions — the final three lessons.
"""

CONTENT["Beyond density estimation"] = r"""
Nonparametrics for other functionals
--------------------------------------

Dirichlet process mixtures were introduced for density estimation, but their reach is far wider. Once you
have a flexible posterior over a **whole distribution**, any **functional** of that distribution inherits
a posterior — so nonparametric Bayes answers questions that are not about the density's shape at all.

Functionals come free
------------------------

A DPM yields posterior draws of the entire distribution :math:`G` (or the implied density). Push each
draw through any functional and you get its posterior automatically — no new model required:

.. math::

   T(G) : \quad \text{mean, variance, quantiles, } \Pr(Y > c), \; \text{entropy, mode count, } \dots

Each posterior draw of :math:`G` gives one draw of :math:`T(G)`, so a **median**, a **tail probability**,
or the **number of modes** arrives with full uncertainty — including uncertainty about the distributional
*shape*, which a parametric model would have suppressed by assuming it away. Estimating a 99th percentile
from a skewed, multimodal distribution is exactly where this pays off.

.. code-block:: python

   import numpy as np
   # each posterior draw is a full mixture -> evaluate any functional per draw
   def functional_posterior(weights_draws, mu_draws, sigma_draws, T):
       return np.array([T(w, m, s)                          # one value per posterior draw
                        for w, m, s in zip(weights_draws, mu_draws, sigma_draws)])
   # e.g. T = tail probability P(Y > c), or a quantile, or the number of modes

Model-based clustering as inference
-------------------------------------

The DPM's **partition** is itself a rich object. Because the model puts a posterior over *how the data
divide into groups*, clustering stops being a point estimate from an algorithm and becomes **inference**:
you get the posterior probability that two points share a cluster, the distribution of the number of
clusters, and a principled way to report clustering **uncertainty** — none of which :math:`k`-means or a
dendrogram provides.

Other nonparametric objects
-----------------------------

The DP is one of a **family**. The **Indian buffet process** gives a nonparametric prior for
**latent-feature** models — objects possessing an unbounded set of overlapping features rather than
belonging to one cluster. **Pólya trees** and **Gaussian processes** are nonparametric priors on
densities and functions. **Survival** and **hazard** functions get nonparametric treatments too. The
unifying theme: put a prior on an **infinite-dimensional** object — a distribution, a function, a feature
matrix — and let the data determine its complexity, with every downstream quantity carrying honest
posterior uncertainty.
"""

CONTENT["Hierarchical dependence"] = r"""
Sharing nonparametric structure across groups
-----------------------------------------------

Data often arrive in **groups** — documents in a corpus, patients across hospitals, measurements per
site — and each group may need its own flexible distribution. Fitting a **separate** DP per group shares
nothing; fitting **one** DP to the pooled data ignores group identity. The **hierarchical Dirichlet
process** (Teh, Jordan, Beal and Blei) resolves the tension, letting groups have distinct distributions
that **share components**.

The construction
------------------

Make the DPs themselves draws from a **higher-level** DP. A global :math:`G_0` is drawn from a top DP;
each group's :math:`G_j` is then drawn from a DP with :math:`G_0` as its base distribution:

.. math::

   G_0 \sim \mathrm{DP}(\gamma, H), \qquad
   G_j \mid G_0 \sim \mathrm{DP}(\alpha, G_0) \;\; \text{for each group } j .

Because :math:`G_0` is itself discrete, every group-level :math:`G_j` draws its atoms **from the same
shared set** — so a cluster discovered in one group is *available* to all, while each group has its own
weights over that common inventory. Groups differ in **how much** they use each component, not in the
components available.

.. code-block:: python

   # Chinese restaurant franchise: the HDP's metaphor
   #   each group is a restaurant; tables within a restaurant share DISHES from a global menu
   #   a popular dish (component) recurs across restaurants -> shared clusters
   #   new dish chosen w.p. ~ gamma (global);  new table w.p. ~ alpha (local)
   # collapsed Gibbs (Teh et al. 2006) reseats customers and re-serves dishes

Why sharing matters
---------------------

The HDP is the nonparametric version of the **partial pooling** that runs through this entire book —
groups borrow strength by sharing a common set of components, and the number of shared components is
**learned**, not fixed. Its most famous use is **topic modelling**: documents are groups, words are
observations, and the shared components are **topics** discovered across the corpus, with the number of
topics inferred rather than set. The same structure serves grouped density estimation, multi-population
clustering, and infinite hidden Markov models — anywhere related groups each need a flexible distribution
but ought to share what they have in common.
"""

CONTENT["Density regression"] = r"""
The whole distribution as a function of x
-------------------------------------------

The final synthesis. Ordinary regression models how the **mean** of :math:`y` changes with predictors;
even flexible regressions (splines, GPs) usually still assume the *shape* of the conditional distribution
is fixed. **Density regression** removes that last assumption, letting the **entire** conditional
distribution :math:`p(y \mid x)` — its variance, skewness, modality, everything — vary with :math:`x`.

Why the mean is not enough
----------------------------

Sometimes the interesting effect is not on the average. A treatment might not shift the mean response but
**split** it into responders and non-responders — a change from one mode to two. Income dispersion may
widen with education while the mean barely moves. Any model that reports only :math:`\mathrm{E}[y \mid x]`
is blind to these, and they are often the substantive finding.

Dependent Dirichlet processes
-------------------------------

The tool is a DP mixture whose ingredients **depend on** :math:`x` — a **dependent Dirichlet process**
(MacEachern). The mixture that generates :math:`y` has weights and/or component parameters that are
functions of the predictors, so the whole conditional density is free to change smoothly across the
covariate space:

.. math::

   p(y \mid x) = \sum_{k=1}^{\infty} \pi_k(x) \, f\bigl(y \mid \theta_k(x)\bigr),

with :math:`\pi_k(x)` and :math:`\theta_k(x)` varying with :math:`x` — for instance weights from a
covariate-dependent stick-breaking, or component means that are themselves regressions or Gaussian
processes in :math:`x`.

.. code-block:: python

   import pymc as pm
   # covariate-dependent mixture: component means are regressions in x
   K = 15
   with pm.Model():
       beta = pm.Beta("beta", 1, pm.Gamma("alpha", 1, 1), shape=K)
       w = pm.Deterministic("w", beta * pm.math.concatenate(
           [[1.0], pm.math.cumprod(1 - beta)[:-1]]))
       coef = pm.Normal("coef", 0, 1, shape=(K, X.shape[1]))    # each component a regression
       mus = X @ coef.T                                          # component means depend on x
       # observation drawn from the x-dependent mixture (weights could depend on x too)
       pm.NormalMixture("y", w=w, mu=mus, sigma=pm.HalfNormal("s", 1, shape=K), observed=y)

The synthesis, and the close
------------------------------

Density regression **unites the whole book**. It is a regression (Part IV), made flexible by a mixture
(this stage), rendered nonparametric by a Dirichlet process (the last lessons), often with Gaussian-process
or spline components (Part V) and hierarchical sharing across groups (throughout) — and fitted, checked
and compared by the computational and model-checking machinery of Parts II and III. It is the most
general regression in this course: predictors may reshape the response distribution in **any** smooth way,
with full posterior uncertainty over the entire family of conditional densities. From Bayes' rule for a
single unknown to an infinite, covariate-indexed family of distributions — the arc of Bayesian data
analysis, from its one idea to its fullest expression.
"""


MINDMAP.update({
    "Dirichlet process mixtures": [
        "Dirichlet process prior distributions", "Beyond density estimation",
        "Setting up and interpreting mixture models", "Density regression",
    ],
    "Beyond density estimation": [
        "Dirichlet process mixtures", "Hierarchical dependence",
        "Bayesian histograms", "Mixture models for classification and regression",
    ],
    "Hierarchical dependence": [
        "Dirichlet process mixtures", "Exchangeability and hierarchical models",
        "Beyond density estimation", "State-level opinons from national polls",
    ],
    "Density regression": [
        "Dirichlet process mixtures", "Gaussian process regression",
        "Hierarchical dependence", "Mixture models for classification and regression",
    ],
})
