:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _bayesian-data-analysis-index:

.. _bayes-discovery:
.. _bayes-foundations:
.. _bayes-computation:
.. _bayes-advanced:
.. _bayes-skplt-map:
.. _bayes-sources:
.. _bayes-theorem:
.. _bayes-pieces:
.. _bayes-credible:
.. _bayes-conjugacy:
.. _bayes-mcmc:
.. _bayes-ppc:
.. _bayes-hierarchical:
.. _bayes-mixtures:
.. _bayes-dp:

:raw-html:`<div style="text-align:center"><strong>` 🎲 Bayesian Data Analysis
|br| Reasoning about uncertainty with priors, likelihoods and posteriors
|br| |full_version| - |today|
:raw-html:`</strong></div>`

======================================================================
Bayesian Data Analysis
======================================================================

Bayesian analysis treats unknown quantities as **probability distributions** and updates them
with data: instead of a single best estimate you get a full **posterior**. This course follows
the arc of Gelman et al., *Bayesian Data Analysis* — from first principles through computation
and regression up to the **nonparametric** models (mixtures, Dirichlet processes) that close it
— as an ordered sequence of 144 lessons across 16 stages in 5 parts.

Read it at any depth:

* **newcomers** — the intuition of prior → likelihood → posterior;
* **practitioners** — how to compute, check and compare posteriors;
* **researchers** — hierarchical, GP and nonparametric (infinite-mixture) models.

.. note::

   **Type in the filter box** for instant lookup by title or keyword, expand a stage to browse,
   or open the A–Z index at the bottom. Code snippets use real ``scipy.stats`` / ``PyMC`` /
   ``ArviZ`` / ``scikit-learn`` calls. This course pairs with the
   :ref:`Terminology reference <terminology-index>` (probability and distributions).

.. raw:: html

   <div style="text-align:center;margin:0.4rem 0 0.4rem">
   <input id="term-filter" type="search" autocomplete="off" spellcheck="false"
          placeholder="&#128269;&nbsp; Type to filter 144 lessons &mdash; by title or keyword&hellip;"
          style="width:100%;max-width:100%;padding:0.55rem 1rem;font-size:1rem;
                 border:1px solid var(--pst-color-border,#ccc);border-radius:0.55rem;box-sizing:border-box;
                 background:transparent;color:inherit"/>
   <div id="term-filter-count" style="opacity:0.65;font-size:0.85rem;
        min-height:1.2em;margin-top:0.35rem"></div>
   </div>
   <script>
   document.addEventListener('DOMContentLoaded',function(){
     var inp=document.getElementById('term-filter');if(!inp){return;}
     var dds=[].slice.call(document.querySelectorAll('details.sd-dropdown'));
     var az=document.querySelector('details.term-az');
     var items=[];
     dds.forEach(function(d){[].slice.call(d.querySelectorAll('li')).forEach(
       function(li){items.push({li:li,d:d,t:li.textContent.toLowerCase()});});});
     var cnt=document.getElementById('term-filter-count');
     inp.addEventListener('input',function(){
       var q=inp.value.trim().toLowerCase();var n=0;
       dds.forEach(function(d){d.tHits=0;});
       items.forEach(function(it){
         var hit=!q||it.t.indexOf(q)!==-1;
         it.li.style.display=hit?'':'none';
         if(hit){it.d.tHits+=1;if(az&&it.d===az){n+=1;}}});
       dds.forEach(function(d){
         if(q){d.style.display=d.tHits?'':'none';d.open=d.tHits>0;}
         else{d.style.display='';d.open=false;}});
       if(cnt){cnt.textContent=(q&&az)?(n+' of 144 match'+(n===1?'':'s')):'';}
     });
   });
   </script>

Fundamentals
------------------------------------------------------------------------

.. _bda-stage-bayes_idea:

.. dropdown:: 🎲 Stage 1 — The Bayesian Idea  ·  10 lessons
   :animate: fade-in-slide-down

   *The three steps, notation, probability as uncertainty, and first worked examples.*

   * :doc:`001 · The three steps of Bayesian data analysis <001-the-three-steps-of-bayesian-data-analysis>` — Set up a full probability model, condition on data, then evaluate the fit.
   * :doc:`002 · General Notation for Statistical Inference <002-general-notation-for-statistical-inference>` — The notation for parameters, data and distributions used throughout.
   * :doc:`003 · Bayesian Inference <003-bayesian-inference>` — Drawing conclusions as posterior probability statements about parameters.
   * :doc:`004 · Discrete Bayesian Examples – Genetics and Spell Checking (with θ) <004-discrete-bayesian-examples-genetics-and-spell-checking-with>` — Bayes' rule on discrete unknowns, from carrier genetics to spell-checking.
   * :doc:`005 · Probability as a Measure of Uncertainty <005-probability-as-a-measure-of-uncertainty>` — Interpreting probability as a quantified degree of belief.
   * :doc:`006 · Example — Probabilities from Football Point Spreads <006-example-probabilities-from-football-point-spreads>` — Reading calibrated probabilities off betting point spreads.
   * :doc:`007 · Example — Calibration for Record Linkage <007-example-calibration-for-record-linkage>` — Assigning match probabilities when merging records across files.
   * :doc:`008 · Some Useful Results from Probability Theory <008-some-useful-results-from-probability-theory>` — Key identities — conditioning, expectation, transformation — the analysis relies on.
   * :doc:`009 · Computation and Software <009-computation-and-software>` — The computing tools and workflow for fitting Bayesian models.
   * :doc:`010 · Bayesian Inference in Applied Statistics <010-bayesian-inference-in-applied-statistics>` — How the Bayesian approach plays out across real applications.

.. _bda-stage-single_param:

.. dropdown:: 📍 Stage 2 — Single-Parameter Models & Priors  ·  9 lessons
   :animate: fade-in-slide-down

   *Binomial and normal models; informative, noninformative and weakly-informative priors.*

   * :doc:`011 · Estimating a Probability from Binomial Data <011-estimating-a-probability-from-binomial-data>` — The Beta–Binomial model: inferring a success probability.
   * :doc:`012 · Posterior as a Compromise Between Data and Prior Information <012-posterior-as-a-compromise-between-data-and-prior-information>` — How the posterior blends prior belief with observed evidence.
   * :doc:`013 · Summarizing Posterior Inference <013-summarizing-posterior-inference>` — Point estimates, intervals and quantiles from a posterior.
   * :doc:`014 · Informative Prior Distributions <014-informative-prior-distributions>` — Priors that encode real prior knowledge about a parameter.
   * :doc:`015 · Normal Distribution with Known Variance <015-normal-distribution-with-known-variance>` — Conjugate inference for a normal mean when the variance is fixed.
   * :doc:`016 · Other Standard Single-Parameter Models <016-other-standard-single-parameter-models>` — Poisson, exponential and related conjugate one-parameter models.
   * :doc:`017 · Informative Prior Distribution for Cancer Rates <017-informative-prior-distribution-for-cancer-rates>` — A worked informative-prior analysis of county cancer rates.
   * :doc:`018 · Noninformative Prior Distributions <018-noninformative-prior-distributions>` — Priors meant to let the data dominate the inference.
   * :doc:`019 · Weakly Informative Prior Distributions <019-weakly-informative-prior-distributions>` — Priors that gently regularize without imposing strong belief.

.. _bda-stage-multiparam:

.. dropdown:: 🧮 Stage 3 — Multiparameter Models  ·  8 lessons
   :animate: fade-in-slide-down

   *Nuisance parameters, the multinomial and multivariate normal, and the bioassay example.*

   * :doc:`020 · Averaging Over Nuisance Parameters <020-averaging-over-nuisance-parameters>` — Marginalizing out parameters you do not care about.
   * :doc:`021 · Normal Data with a Noninformative Prior Distribution <021-normal-data-with-a-noninformative-prior-distribution>` — Joint inference for a normal mean and variance under a flat prior.
   * :doc:`022 · Normal Data with a Conjugate Prior Distribution <022-normal-data-with-a-conjugate-prior-distribution>` — The normal–inverse-gamma conjugate analysis for mean and variance.
   * :doc:`023 · Multinomial Model for Categorical Data <023-multinomial-model-for-categorical-data>` — Dirichlet–multinomial inference for category probabilities.
   * :doc:`024 · Multivariate Normal Model with Known Variance <024-multivariate-normal-model-with-known-variance>` — Inferring a mean vector under a known covariance.
   * :doc:`025 · Multivariate Normal with Unknown Mean and Variance <025-multivariate-normal-with-unknown-mean-and-variance>` — Joint inference for mean and covariance via the inverse-Wishart.
   * :doc:`026 · Example: Bayesian analysis of a bioassay experiment (logistic, nonconjugate) <026-example-bayesian-analysis-of-a-bioassay-experiment-logistic-nonconjugate>` — A nonconjugate logistic dose–response fit on a grid and by sampling.
   * :doc:`027 · Summary of Elementary Modeling and Computation <027-summary-of-elementary-modeling-and-computation>` — Consolidating the single- and multi-parameter modeling toolkit.

.. _bda-stage-asymptotics:

.. dropdown:: 📏 Stage 4 — Asymptotics & Frequentist Ties  ·  5 lessons
   :animate: fade-in-slide-down

   *Normal approximation, large-sample theory, and how Bayesian and frequentist inference relate.*

   * :doc:`028 · Normal Approximations to the Posterior Distribution <028-normal-approximations-to-the-posterior-distribution>` — Approximating a posterior by a normal around its mode.
   * :doc:`029 · Large-Sample Theory <029-large-sample-theory>` — How posteriors concentrate and become normal as data grow.
   * :doc:`030 · Counterexamples to large-sample (asymptotic) Bayesian theorems <030-counterexamples-to-large-sample-asymptotic-bayesian-theorems>` — When the normal-approximation guarantees fail.
   * :doc:`031 · Frequency Evaluations of Bayesian Inferences <031-frequency-evaluations-of-bayesian-inferences>` — Judging Bayesian procedures by their long-run frequency behavior.
   * :doc:`032 · Bayesian interpretations of other statistical methods <032-bayesian-interpretations-of-other-statistical-methods>` — Seeing penalized and classical methods as Bayesian in disguise.

.. _bda-stage-hierarchical:

.. dropdown:: 🏛️ Stage 5 — Hierarchical Models  ·  7 lessons
   :animate: fade-in-slide-down

   *Exchangeability, the eight-schools model, and meta-analysis through partial pooling.*

   * :doc:`033 · Constructing a Parameterized Prior Distribution <033-constructing-a-parameterized-prior-distribution>` — Building a prior whose parameters are themselves estimated.
   * :doc:`034 · Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>` — Exchangeability as the justification for hierarchical structure.
   * :doc:`035 · Bayesian analysis of conjugate hierarchical models <035-bayesian-analysis-of-conjugate-hierarchical-models>` — Fitting hierarchical models that admit conjugate updates.
   * :doc:`036 · Normal model with exchangeable parameters <036-normal-model-with-exchangeable-parameters>` — The normal hierarchical model with a shared population distribution.
   * :doc:`037 · Example: parallel experiments in eight schools <037-example-parallel-experiments-in-eight-schools>` — The canonical partial-pooling example across eight studies.
   * :doc:`038 · Hierarchical modeling applied to a meta-analysis <038-hierarchical-modeling-applied-to-a-meta-analysis>` — Pooling effect estimates across studies hierarchically.
   * :doc:`039 · Weakly Informative Priors for Variance Parameters <039-weakly-informative-priors-for-variance-parameters>` — Sensible priors for group-level variance in hierarchies.

Checking & Deciding
------------------------------------------------------------------------

.. _bda-stage-checking:

.. dropdown:: 🔍 Stage 6 — Model Checking & Comparison  ·  10 lessons
   :animate: fade-in-slide-down

   *Posterior predictive checks, predictive accuracy, Bayes factors, and model expansion.*

   * :doc:`040 · The Place of Model Checking in Applied Bayesian Statistics <040-the-place-of-model-checking-in-applied-bayesian-statistics>` — Why checking fit is integral, not optional, to modeling.
   * :doc:`041 · Do the Inferences from the Model Make Sense? <041-do-the-inferences-from-the-model-make-sense>` — Sanity-checking posterior conclusions against knowledge.
   * :doc:`042 · Posterior predictive checking <042-posterior-predictive-checking>` — Comparing data replicated from the model to the real data.
   * :doc:`043 · Graphical posterior predictive checks <043-graphical-posterior-predictive-checks>` — Visual comparisons of replicated and observed data.
   * :doc:`044 · Model checking for the educational testing example <044-model-checking-for-the-educational-testing-example>` — Applying predictive checks to the eight-schools fit.
   * :doc:`045 · Measures of predictive accuracy <045-measures-of-predictive-accuracy>` — Log predictive density, WAIC and cross-validation scores.
   * :doc:`046 · Model comparison based on predictive performance <046-model-comparison-based-on-predictive-performance>` — Choosing models by out-of-sample predictive ability.
   * :doc:`047 · Model comparison using Bayes factors <047-model-comparison-using-bayes-factors>` — Comparing models via marginal-likelihood ratios.
   * :doc:`048 · Continuous model expansion <048-continuous-model-expansion>` — Embedding a model in a richer family to test assumptions.
   * :doc:`049 · Implicit assumptions and model expansion: an example <049-implicit-assumptions-and-model-expansion-an-example>` — Surfacing hidden assumptions by expanding a model.

.. _bda-stage-collection_decision:

.. dropdown:: 🗳️ Stage 7 — Data Collection & Decisions  ·  12 lessons
   :animate: fade-in-slide-down

   *Ignorability, surveys and experiments, censoring, and Bayesian decision analysis.*

   * :doc:`050 · Bayesian inference requires a model for data collection <050-bayesian-inference-requires-a-model-for-data-collection>` — Why the sampling/collection mechanism must be modeled.
   * :doc:`051 · Data-collection models and ignorability <051-data-collection-models-and-ignorability>` — When the data-collection process can be safely ignored.
   * :doc:`052 · Sample surveys <052-sample-surveys>` — Bayesian inference for survey sampling designs.
   * :doc:`053 · Designed experiments <053-designed-experiments>` — Modeling randomized experiments the Bayesian way.
   * :doc:`054 · Sensitivity and the role of randomization <054-sensitivity-and-the-role-of-randomization>` — How randomization protects inference from hidden bias.
   * :doc:`055 · Observational studies <055-observational-studies>` — Inference when treatment is not randomized.
   * :doc:`056 · Censoring and truncation <056-censoring-and-truncation>` — Modeling data that are cut off or partially observed.
   * :doc:`057 · Bayesian decision theory in diﬀerent contexts <057-bayesian-decision-theory-in-different-contexts>` — Choosing actions to maximize expected utility.
   * :doc:`058 · Using regression predictions: survey incentives <058-using-regression-predictions-survey-incentives>` — A decision analysis of incentive choices via regression.
   * :doc:`059 · Multistage decision making: medical screening <059-multistage-decision-making-medical-screening>` — Sequential decisions under uncertainty in screening.
   * :doc:`060 · Hierarchical decision analysis for home radon <060-hierarchical-decision-analysis-for-home-radon>` — A hierarchical decision model for radon remediation.
   * :doc:`061 · Personal vs. institutional decision analysis <061-personal-vs-institutional-decision-analysis>` — Whose utilities and information a decision should use.

Computation
------------------------------------------------------------------------

.. _bda-stage-simulation:

.. dropdown:: 🧰 Stage 8 — Simulation Basics  ·  7 lessons
   :animate: fade-in-slide-down

   *Numerical integration, rejection and importance sampling, and how many draws are needed.*

   * :doc:`062 · Numerical integration <062-numerical-integration>` — Quadrature and simulation for intractable integrals.
   * :doc:`063 · Distributional approximations <063-distributional-approximations>` — Approximating posteriors with tractable distributions.
   * :doc:`064 · Direct simulation and rejection sampling <064-direct-simulation-and-rejection-sampling>` — Drawing exact samples by proposal and acceptance.
   * :doc:`065 · Importance sampling <065-importance-sampling>` — Reweighting draws from a proposal to target the posterior.
   * :doc:`066 · How many simulation draws are needed? <066-how-many-simulation-draws-are-needed>` — Sizing Monte Carlo samples for a target accuracy.
   * :doc:`067 · Computing environments <067-computing-environments>` — The software environments for Bayesian computation.
   * :doc:`068 · Debugging Bayesian computing <068-debugging-bayesian-computing>` — Diagnosing and fixing errors in model fitting.

.. _bda-stage-mcmc:

.. dropdown:: ⛓️ Stage 9 — MCMC: Gibbs, Metropolis & HMC  ·  12 lessons
   :animate: fade-in-slide-down

   *Samplers, convergence diagnostics, Hamiltonian Monte Carlo, and Stan.*

   * :doc:`069 · Gibbs sampler <069-gibbs-sampler>` — Sampling each parameter in turn from its full conditional.
   * :doc:`070 · Metropolis and Metropolis-Hastings algorithms <070-metropolis-and-metropolis-hastings-algorithms>` — Accept–reject proposals that target the posterior.
   * :doc:`071 · Using Gibbs and Metropolis as building blocks <071-using-gibbs-and-metropolis-as-building-blocks>` — Composing samplers for complex models.
   * :doc:`072 · Inference and assessing convergence <072-inference-and-assessing-convergence>` — R-hat, effective sample size and mixing diagnostics.
   * :doc:`073 · Eﬀective number of simulation draws <073-effective-number-of-simulation-draws>` — Effective sample size after autocorrelation.
   * :doc:`074 · Example: hierarchical normal model <074-example-hierarchical-normal-model>` — A worked Gibbs sampler for the normal hierarchy.
   * :doc:`075 · Eﬃcient Gibbs samplers <075-efficient-gibbs-samplers>` — Reparameterizations that speed up Gibbs sampling.
   * :doc:`076 · Eﬃcient Metropolis jumping rules <076-efficient-metropolis-jumping-rules>` — Tuning proposal scale for efficient Metropolis moves.
   * :doc:`077 · Further extensions to Gibbs and Metropolis <077-further-extensions-to-gibbs-and-metropolis>` — Slice, reversible-jump and auxiliary-variable extensions.
   * :doc:`078 · Hamiltonian Monte Carlo <078-hamiltonian-monte-carlo>` — Gradient-guided proposals for efficient exploration.
   * :doc:`079 · Hamiltonian Monte Carlo for a hierarchical model <079-hamiltonian-monte-carlo-for-a-hierarchical-model>` — Applying HMC to a hierarchical example.
   * :doc:`080 · Stan: developing a computing environment <080-stan-developing-a-computing-environment>` — The Stan language and NUTS sampler for Bayesian models.

.. _bda-stage-approximation:

.. dropdown:: 🎛️ Stage 10 — Modal & Variational Approximation  ·  10 lessons
   :animate: fade-in-slide-down

   *Posterior modes, EM, variational inference, and expectation propagation.*

   * :doc:`081 · Finding posterior modes <081-finding-posterior-modes>` — Optimization to locate the posterior's peak.
   * :doc:`082 · Boundary-avoiding priors for modal summaries <082-boundary-avoiding-priors-for-modal-summaries>` — Priors that keep modal estimates off the boundary.
   * :doc:`083 · Normal and related mixture approximations <083-normal-and-related-mixture-approximations>` — Approximating posteriors with normal mixtures.
   * :doc:`084 · Finding marginal posterior modes using EM <084-finding-marginal-posterior-modes-using-em>` — The EM algorithm for marginal mode-finding.
   * :doc:`085 · Conditional and marginal posterior approximations <085-conditional-and-marginal-posterior-approximations>` — Approximating conditionals and marginals in stages.
   * :doc:`086 · Example: hierarchical normal model (continued) <086-example-hierarchical-normal-model-continued>` — The hierarchical normal model via modal approximation.
   * :doc:`087 · Variational inference <087-variational-inference>` — Fitting the closest tractable distribution to the posterior.
   * :doc:`088 · Expectation propagation <088-expectation-propagation>` — Iteratively matching moments for posterior approximation.
   * :doc:`089 · Other approximations <089-other-approximations>` — Laplace, INLA and further approximate-inference methods.
   * :doc:`090 · Unknown normalizing factors <090-unknown-normalizing-factors>` — Inference when the likelihood's constant is intractable.

Regression
------------------------------------------------------------------------

.. _bda-stage-regression:

.. dropdown:: 📈 Stage 11 — Regression Foundations  ·  8 lessons
   :animate: fade-in-slide-down

   *Bayesian classical regression, causal inference, and regularization.*

   * :doc:`091 · Conditional modeling <091-conditional-modeling>` — Modeling an outcome conditional on predictors.
   * :doc:`092 · Bayesian analysis of classical regression <092-bayesian-analysis-of-classical-regression>` — Posterior inference for the normal linear model.
   * :doc:`093 · Regression for causal inference: incumbency and voting <093-regression-for-causal-inference-incumbency-and-voting>` — Using regression to estimate an incumbency effect.
   * :doc:`094 · Goals of regression analysis <094-goals-of-regression-analysis>` — Prediction, explanation and causal aims of regression.
   * :doc:`095 · Assembling the matrix of explanatory variables <095-assembling-the-matrix-of-explanatory-variables>` — Coding, transforming and interacting predictors.
   * :doc:`096 · Regularization and dimension reduction <096-regularization-and-dimension-reduction>` — Shrinkage priors and reducing predictor dimension.
   * :doc:`097 · Unequal variances and correlations <097-unequal-variances-and-correlations>` — Modeling heteroscedastic and correlated errors.
   * :doc:`098 · Including numerical prior information <098-including-numerical-prior-information>` — Adding quantitative prior knowledge to regression.

.. _bda-stage-hier_regression:

.. dropdown:: 🏗️ Stage 12 — Hierarchical Regression  ·  7 lessons
   :animate: fade-in-slide-down

   *Batched coefficients, varying intercepts and slopes, and election forecasting.*

   * :doc:`099 · Regression coeﬃcients exchangeable in batches <099-regression-coefficients-exchangeable-in-batches>` — Grouping coefficients into exchangeable batches.
   * :doc:`100 · Example: forecasting U.S. presidential elections <100-example-forecasting-u-s-presidential-elections>` — A hierarchical election-forecasting regression.
   * :doc:`101 · Interpreting a normal prior distribution as extra data <101-interpreting-a-normal-prior-distribution-as-extra-data>` — Reading a coefficient prior as pseudo-observations.
   * :doc:`102 · Varying intercepts and slopes <102-varying-intercepts-and-slopes>` — Letting regression coefficients vary across groups.
   * :doc:`103 · Computation: batching and transformation <103-computation-batching-and-transformation>` — Efficient computation for hierarchical regressions.
   * :doc:`104 · Analysis of variance and the batching of coeﬃcients <104-analysis-of-variance-and-the-batching-of-coefficients>` — ANOVA as hierarchical batching of effects.
   * :doc:`105 · Hierarchical models for batches of variance components <105-hierarchical-models-for-batches-of-variance-components>` — Priors over multiple variance components.

.. _bda-stage-glm:

.. dropdown:: 🔗 Stage 13 — Generalized Linear Models  ·  7 lessons
   :animate: fade-in-slide-down

   *GLM likelihoods, priors for logistic regression, and overdispersed Poisson models.*

   * :doc:`106 · Standard generalized linear model likelihoods <106-standard-generalized-linear-model-likelihoods>` — Binomial, Poisson and other GLM data models.
   * :doc:`107 · Working with generalized linear models <107-working-with-generalized-linear-models>` — Fitting, checking and interpreting Bayesian GLMs.
   * :doc:`108 · Weakly informative priors for logistic regression <108-weakly-informative-priors-for-logistic-regression>` — Regularizing logistic coefficients to avoid separation.
   * :doc:`109 · Overdispersed Poisson regression for police stops <109-overdispersed-poisson-regression-for-police-stops>` — A hierarchical Poisson model of stop-and-frisk data.
   * :doc:`110 · State-level opinons from national polls <110-state-level-opinons-from-national-polls>` — Multilevel regression and poststratification of opinion.
   * :doc:`111 · Models for multivariate and multinomial responses <111-models-for-multivariate-and-multinomial-responses>` — GLMs for vector and categorical outcomes.
   * :doc:`112 · Loglinear models for multivariate discrete data <112-loglinear-models-for-multivariate-discrete-data>` — Loglinear structure for contingency tables.

.. _bda-stage-robust_missing:

.. dropdown:: 🛡️ Stage 14 — Robustness & Missing Data  ·  11 lessons
   :animate: fade-in-slide-down

   *Heavy-tailed errors, overdispersion, and multiple imputation.*

   * :doc:`113 · Aspects of robustness <113-aspects-of-robustness>` — Making inference resilient to model misspecification.
   * :doc:`114 · Overdispersed versions of standard models <114-overdispersed-versions-of-standard-models>` — Heavy-tailed and overdispersed extensions.
   * :doc:`115 · Posterior inference and computation <115-posterior-inference-and-computation>` — Computing posteriors for robust models.
   * :doc:`116 · Robust inference for the eight schools <116-robust-inference-for-the-eight-schools>` — A t-based robust reanalysis of eight schools.
   * :doc:`117 · Robust regression using t-distributed errors <117-robust-regression-using-t-distributed-errors>` — Down-weighting outliers with t errors.
   * :doc:`118 · Notation <118-notation>` — Notation for the missing-data framework.
   * :doc:`119 · Multiple imputation <119-multiple-imputation>` — Filling in missing values with several posterior draws.
   * :doc:`120 · Missing data in the multivariate normal and t models <120-missing-data-in-the-multivariate-normal-and-t-models>` — Imputation under multivariate normal and t models.
   * :doc:`121 · Example: multiple imputation for a series of polls <121-example-multiple-imputation-for-a-series-of-polls>` — Imputing missing responses across repeated polls.
   * :doc:`122 · Missing values with counted data <122-missing-values-with-counted-data>` — Handling missingness in count data.
   * :doc:`123 · Example: an opinion poll in Slovenia <123-example-an-opinion-poll-in-slovenia>` — A worked missing-data analysis of a plebiscite poll.

Nonlinear & Nonparametric
------------------------------------------------------------------------

.. _bda-stage-gp_basis:

.. dropdown:: 🌊 Stage 15 — Basis Functions & Gaussian Processes  ·  9 lessons
   :animate: fade-in-slide-down

   *Splines, Gaussian-process regression, and functional data analysis.*

   * :doc:`124 · Example: serial dilution assay <124-example-serial-dilution-assay>` — A nonlinear Bayesian calibration for dilution assays.
   * :doc:`125 · Example: population toxicokinetics <125-example-population-toxicokinetics>` — A hierarchical nonlinear pharmacokinetic model.
   * :doc:`126 · Splines and weighted sums of basis functions <126-splines-and-weighted-sums-of-basis-functions>` — Flexible curves built from weighted basis functions.
   * :doc:`127 · Basis selection and shrinkage of coeﬃcients <127-basis-selection-and-shrinkage-of-coefficients>` — Shrinking basis coefficients to control smoothness.
   * :doc:`128 · Non-normal models and regression surfaces <128-non-normal-models-and-regression-surfaces>` — Nonlinear and non-normal regression surfaces.
   * :doc:`129 · Gaussian process regression <129-gaussian-process-regression>` — Priors over functions for flexible regression.
   * :doc:`130 · Example: birthdays and birthdates <130-example-birthdays-and-birthdates>` — A Gaussian-process decomposition of a birth time series.
   * :doc:`131 · Latent Gaussian process models <131-latent-gaussian-process-models>` — GPs as latent components inside larger models.
   * :doc:`132 · Functional data analysis <132-functional-data-analysis>` — Modeling whole curves as data objects.

.. _bda-stage-nonparametric:

.. dropdown:: ♾️ Stage 16 — Mixtures & Nonparametric Bayes  ·  12 lessons
   :animate: fade-in-slide-down

   *Mixtures, label switching, Bayesian histograms, and Dirichlet processes.*

   * :doc:`133 · Density estimation and regression <133-density-estimation-and-regression>` — Estimating whole distributions, not just parameters.
   * :doc:`134 · Setting up and interpreting mixture models <134-setting-up-and-interpreting-mixture-models>` — Modeling populations as blends of components.
   * :doc:`135 · Example: reaction times and schizophrenia <135-example-reaction-times-and-schizophrenia>` — A mixture model separating response-time subgroups.
   * :doc:`136 · Label switching and posterior computation <136-label-switching-and-posterior-computation>` — Handling component permutation in mixture posteriors.
   * :doc:`137 · Unspecified number of mixture components <137-unspecified-number-of-mixture-components>` — Letting the number of components be unknown.
   * :doc:`138 · Mixture models for classification and regression <138-mixture-models-for-classification-and-regression>` — Mixtures as flexible supervised models.
   * :doc:`139 · Bayesian histograms <139-bayesian-histograms>` — Nonparametric density via random histograms.
   * :doc:`140 · Dirichlet process prior distributions <140-dirichlet-process-prior-distributions>` — Priors over distributions with unbounded support.
   * :doc:`141 · Dirichlet process mixtures <141-dirichlet-process-mixtures>` — Infinite mixtures whose clusters grow with the data.
   * :doc:`142 · Beyond density estimation <142-beyond-density-estimation>` — Nonparametric priors for broader modeling tasks.
   * :doc:`143 · Hierarchical dependence <143-hierarchical-dependence>` — Dependent nonparametric priors across groups.
   * :doc:`144 · Density regression <144-density-regression>` — Letting an entire density vary with predictors.

🔤 Every lesson, A–Z
---------------------

.. dropdown:: Open the full alphabetical index
   :class-container: term-az

   .. hlist::
      :columns: 2

      * :doc:`Analysis of variance and the batching of coeﬃcients <104-analysis-of-variance-and-the-batching-of-coefficients>`
      * :doc:`Aspects of robustness <113-aspects-of-robustness>`
      * :doc:`Assembling the matrix of explanatory variables <095-assembling-the-matrix-of-explanatory-variables>`
      * :doc:`Averaging Over Nuisance Parameters <020-averaging-over-nuisance-parameters>`
      * :doc:`Basis selection and shrinkage of coeﬃcients <127-basis-selection-and-shrinkage-of-coefficients>`
      * :doc:`Bayesian analysis of classical regression <092-bayesian-analysis-of-classical-regression>`
      * :doc:`Bayesian analysis of conjugate hierarchical models <035-bayesian-analysis-of-conjugate-hierarchical-models>`
      * :doc:`Bayesian decision theory in diﬀerent contexts <057-bayesian-decision-theory-in-different-contexts>`
      * :doc:`Bayesian histograms <139-bayesian-histograms>`
      * :doc:`Bayesian Inference <003-bayesian-inference>`
      * :doc:`Bayesian Inference in Applied Statistics <010-bayesian-inference-in-applied-statistics>`
      * :doc:`Bayesian inference requires a model for data collection <050-bayesian-inference-requires-a-model-for-data-collection>`
      * :doc:`Bayesian interpretations of other statistical methods <032-bayesian-interpretations-of-other-statistical-methods>`
      * :doc:`Beyond density estimation <142-beyond-density-estimation>`
      * :doc:`Boundary-avoiding priors for modal summaries <082-boundary-avoiding-priors-for-modal-summaries>`
      * :doc:`Censoring and truncation <056-censoring-and-truncation>`
      * :doc:`Computation and Software <009-computation-and-software>`
      * :doc:`Computation: batching and transformation <103-computation-batching-and-transformation>`
      * :doc:`Computing environments <067-computing-environments>`
      * :doc:`Conditional and marginal posterior approximations <085-conditional-and-marginal-posterior-approximations>`
      * :doc:`Conditional modeling <091-conditional-modeling>`
      * :doc:`Constructing a Parameterized Prior Distribution <033-constructing-a-parameterized-prior-distribution>`
      * :doc:`Continuous model expansion <048-continuous-model-expansion>`
      * :doc:`Counterexamples to large-sample (asymptotic) Bayesian theorems <030-counterexamples-to-large-sample-asymptotic-bayesian-theorems>`
      * :doc:`Data-collection models and ignorability <051-data-collection-models-and-ignorability>`
      * :doc:`Debugging Bayesian computing <068-debugging-bayesian-computing>`
      * :doc:`Density estimation and regression <133-density-estimation-and-regression>`
      * :doc:`Density regression <144-density-regression>`
      * :doc:`Designed experiments <053-designed-experiments>`
      * :doc:`Direct simulation and rejection sampling <064-direct-simulation-and-rejection-sampling>`
      * :doc:`Dirichlet process mixtures <141-dirichlet-process-mixtures>`
      * :doc:`Dirichlet process prior distributions <140-dirichlet-process-prior-distributions>`
      * :doc:`Discrete Bayesian Examples – Genetics and Spell Checking (with θ) <004-discrete-bayesian-examples-genetics-and-spell-checking-with>`
      * :doc:`Distributional approximations <063-distributional-approximations>`
      * :doc:`Do the Inferences from the Model Make Sense? <041-do-the-inferences-from-the-model-make-sense>`
      * :doc:`Eﬀective number of simulation draws <073-effective-number-of-simulation-draws>`
      * :doc:`Eﬃcient Gibbs samplers <075-efficient-gibbs-samplers>`
      * :doc:`Eﬃcient Metropolis jumping rules <076-efficient-metropolis-jumping-rules>`
      * :doc:`Estimating a Probability from Binomial Data <011-estimating-a-probability-from-binomial-data>`
      * :doc:`Example — Calibration for Record Linkage <007-example-calibration-for-record-linkage>`
      * :doc:`Example — Probabilities from Football Point Spreads <006-example-probabilities-from-football-point-spreads>`
      * :doc:`Example: an opinion poll in Slovenia <123-example-an-opinion-poll-in-slovenia>`
      * :doc:`Example: Bayesian analysis of a bioassay experiment (logistic, nonconjugate) <026-example-bayesian-analysis-of-a-bioassay-experiment-logistic-nonconjugate>`
      * :doc:`Example: birthdays and birthdates <130-example-birthdays-and-birthdates>`
      * :doc:`Example: forecasting U.S. presidential elections <100-example-forecasting-u-s-presidential-elections>`
      * :doc:`Example: hierarchical normal model <074-example-hierarchical-normal-model>`
      * :doc:`Example: hierarchical normal model (continued) <086-example-hierarchical-normal-model-continued>`
      * :doc:`Example: multiple imputation for a series of polls <121-example-multiple-imputation-for-a-series-of-polls>`
      * :doc:`Example: parallel experiments in eight schools <037-example-parallel-experiments-in-eight-schools>`
      * :doc:`Example: population toxicokinetics <125-example-population-toxicokinetics>`
      * :doc:`Example: reaction times and schizophrenia <135-example-reaction-times-and-schizophrenia>`
      * :doc:`Example: serial dilution assay <124-example-serial-dilution-assay>`
      * :doc:`Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>`
      * :doc:`Expectation propagation <088-expectation-propagation>`
      * :doc:`Finding marginal posterior modes using EM <084-finding-marginal-posterior-modes-using-em>`
      * :doc:`Finding posterior modes <081-finding-posterior-modes>`
      * :doc:`Frequency Evaluations of Bayesian Inferences <031-frequency-evaluations-of-bayesian-inferences>`
      * :doc:`Functional data analysis <132-functional-data-analysis>`
      * :doc:`Further extensions to Gibbs and Metropolis <077-further-extensions-to-gibbs-and-metropolis>`
      * :doc:`Gaussian process regression <129-gaussian-process-regression>`
      * :doc:`General Notation for Statistical Inference <002-general-notation-for-statistical-inference>`
      * :doc:`Gibbs sampler <069-gibbs-sampler>`
      * :doc:`Goals of regression analysis <094-goals-of-regression-analysis>`
      * :doc:`Graphical posterior predictive checks <043-graphical-posterior-predictive-checks>`
      * :doc:`Hamiltonian Monte Carlo <078-hamiltonian-monte-carlo>`
      * :doc:`Hamiltonian Monte Carlo for a hierarchical model <079-hamiltonian-monte-carlo-for-a-hierarchical-model>`
      * :doc:`Hierarchical decision analysis for home radon <060-hierarchical-decision-analysis-for-home-radon>`
      * :doc:`Hierarchical dependence <143-hierarchical-dependence>`
      * :doc:`Hierarchical modeling applied to a meta-analysis <038-hierarchical-modeling-applied-to-a-meta-analysis>`
      * :doc:`Hierarchical models for batches of variance components <105-hierarchical-models-for-batches-of-variance-components>`
      * :doc:`How many simulation draws are needed? <066-how-many-simulation-draws-are-needed>`
      * :doc:`Implicit assumptions and model expansion: an example <049-implicit-assumptions-and-model-expansion-an-example>`
      * :doc:`Importance sampling <065-importance-sampling>`
      * :doc:`Including numerical prior information <098-including-numerical-prior-information>`
      * :doc:`Inference and assessing convergence <072-inference-and-assessing-convergence>`
      * :doc:`Informative Prior Distribution for Cancer Rates <017-informative-prior-distribution-for-cancer-rates>`
      * :doc:`Informative Prior Distributions <014-informative-prior-distributions>`
      * :doc:`Interpreting a normal prior distribution as extra data <101-interpreting-a-normal-prior-distribution-as-extra-data>`
      * :doc:`Label switching and posterior computation <136-label-switching-and-posterior-computation>`
      * :doc:`Large-Sample Theory <029-large-sample-theory>`
      * :doc:`Latent Gaussian process models <131-latent-gaussian-process-models>`
      * :doc:`Loglinear models for multivariate discrete data <112-loglinear-models-for-multivariate-discrete-data>`
      * :doc:`Measures of predictive accuracy <045-measures-of-predictive-accuracy>`
      * :doc:`Metropolis and Metropolis-Hastings algorithms <070-metropolis-and-metropolis-hastings-algorithms>`
      * :doc:`Missing data in the multivariate normal and t models <120-missing-data-in-the-multivariate-normal-and-t-models>`
      * :doc:`Missing values with counted data <122-missing-values-with-counted-data>`
      * :doc:`Mixture models for classification and regression <138-mixture-models-for-classification-and-regression>`
      * :doc:`Model checking for the educational testing example <044-model-checking-for-the-educational-testing-example>`
      * :doc:`Model comparison based on predictive performance <046-model-comparison-based-on-predictive-performance>`
      * :doc:`Model comparison using Bayes factors <047-model-comparison-using-bayes-factors>`
      * :doc:`Models for multivariate and multinomial responses <111-models-for-multivariate-and-multinomial-responses>`
      * :doc:`Multinomial Model for Categorical Data <023-multinomial-model-for-categorical-data>`
      * :doc:`Multiple imputation <119-multiple-imputation>`
      * :doc:`Multistage decision making: medical screening <059-multistage-decision-making-medical-screening>`
      * :doc:`Multivariate Normal Model with Known Variance <024-multivariate-normal-model-with-known-variance>`
      * :doc:`Multivariate Normal with Unknown Mean and Variance <025-multivariate-normal-with-unknown-mean-and-variance>`
      * :doc:`Non-normal models and regression surfaces <128-non-normal-models-and-regression-surfaces>`
      * :doc:`Noninformative Prior Distributions <018-noninformative-prior-distributions>`
      * :doc:`Normal and related mixture approximations <083-normal-and-related-mixture-approximations>`
      * :doc:`Normal Approximations to the Posterior Distribution <028-normal-approximations-to-the-posterior-distribution>`
      * :doc:`Normal Data with a Conjugate Prior Distribution <022-normal-data-with-a-conjugate-prior-distribution>`
      * :doc:`Normal Data with a Noninformative Prior Distribution <021-normal-data-with-a-noninformative-prior-distribution>`
      * :doc:`Normal Distribution with Known Variance <015-normal-distribution-with-known-variance>`
      * :doc:`Normal model with exchangeable parameters <036-normal-model-with-exchangeable-parameters>`
      * :doc:`Notation <118-notation>`
      * :doc:`Numerical integration <062-numerical-integration>`
      * :doc:`Observational studies <055-observational-studies>`
      * :doc:`Other approximations <089-other-approximations>`
      * :doc:`Other Standard Single-Parameter Models <016-other-standard-single-parameter-models>`
      * :doc:`Overdispersed Poisson regression for police stops <109-overdispersed-poisson-regression-for-police-stops>`
      * :doc:`Overdispersed versions of standard models <114-overdispersed-versions-of-standard-models>`
      * :doc:`Personal vs. institutional decision analysis <061-personal-vs-institutional-decision-analysis>`
      * :doc:`Posterior as a Compromise Between Data and Prior Information <012-posterior-as-a-compromise-between-data-and-prior-information>`
      * :doc:`Posterior inference and computation <115-posterior-inference-and-computation>`
      * :doc:`Posterior predictive checking <042-posterior-predictive-checking>`
      * :doc:`Probability as a Measure of Uncertainty <005-probability-as-a-measure-of-uncertainty>`
      * :doc:`Regression coeﬃcients exchangeable in batches <099-regression-coefficients-exchangeable-in-batches>`
      * :doc:`Regression for causal inference: incumbency and voting <093-regression-for-causal-inference-incumbency-and-voting>`
      * :doc:`Regularization and dimension reduction <096-regularization-and-dimension-reduction>`
      * :doc:`Robust inference for the eight schools <116-robust-inference-for-the-eight-schools>`
      * :doc:`Robust regression using t-distributed errors <117-robust-regression-using-t-distributed-errors>`
      * :doc:`Sample surveys <052-sample-surveys>`
      * :doc:`Sensitivity and the role of randomization <054-sensitivity-and-the-role-of-randomization>`
      * :doc:`Setting up and interpreting mixture models <134-setting-up-and-interpreting-mixture-models>`
      * :doc:`Some Useful Results from Probability Theory <008-some-useful-results-from-probability-theory>`
      * :doc:`Splines and weighted sums of basis functions <126-splines-and-weighted-sums-of-basis-functions>`
      * :doc:`Stan: developing a computing environment <080-stan-developing-a-computing-environment>`
      * :doc:`Standard generalized linear model likelihoods <106-standard-generalized-linear-model-likelihoods>`
      * :doc:`State-level opinons from national polls <110-state-level-opinons-from-national-polls>`
      * :doc:`Summarizing Posterior Inference <013-summarizing-posterior-inference>`
      * :doc:`Summary of Elementary Modeling and Computation <027-summary-of-elementary-modeling-and-computation>`
      * :doc:`The Place of Model Checking in Applied Bayesian Statistics <040-the-place-of-model-checking-in-applied-bayesian-statistics>`
      * :doc:`The three steps of Bayesian data analysis <001-the-three-steps-of-bayesian-data-analysis>`
      * :doc:`Unequal variances and correlations <097-unequal-variances-and-correlations>`
      * :doc:`Unknown normalizing factors <090-unknown-normalizing-factors>`
      * :doc:`Unspecified number of mixture components <137-unspecified-number-of-mixture-components>`
      * :doc:`Using Gibbs and Metropolis as building blocks <071-using-gibbs-and-metropolis-as-building-blocks>`
      * :doc:`Using regression predictions: survey incentives <058-using-regression-predictions-survey-incentives>`
      * :doc:`Variational inference <087-variational-inference>`
      * :doc:`Varying intercepts and slopes <102-varying-intercepts-and-slopes>`
      * :doc:`Weakly Informative Prior Distributions <019-weakly-informative-prior-distributions>`
      * :doc:`Weakly informative priors for logistic regression <108-weakly-informative-priors-for-logistic-regression>`
      * :doc:`Weakly Informative Priors for Variance Parameters <039-weakly-informative-priors-for-variance-parameters>`
      * :doc:`Working with generalized linear models <107-working-with-generalized-linear-models>`

🗺️ scikit-plots & the Bayesian stack
----------------------------------------

.. dropdown:: Where scikit-plots and the PPL stack fit

   scikit-plots' role here is diagnostic and model-selection visual support; the heavy
   lifting is done by the probabilistic-programming stack.

   * **Gaussian Mixture Models (AIC / BIC)** — choose the number of components:
     https://scikit-plots.github.io/dev/auto_examples/stats/plot_gaussian_mixture_models.html
   * **Residuals distribution** — distributional / Q–Q model checks:
     https://scikit-plots.github.io/dev/auto_examples/stats/plot_residuals_distribution_script.html
   * **PyMC** — probabilistic programming: https://www.pymc.io/
   * **ArviZ** — diagnostics and plots for Bayesian inference: https://python.arviz.org/

.. dropdown:: Sources & standard reference

   Framing re-expressed in our own words; API calls verified against official docs.

   * Source context (144 posts): https://insightful-data-lab.com/category/bayesian-data-analysis/
   * SciPy stats: https://docs.scipy.org/doc/scipy/reference/stats.html
   * scikit-learn mixtures: https://scikit-learn.org/stable/modules/mixture.html
   * Gelman, Carlin, Stern, Dunson, Vehtari & Rubin, *Bayesian Data Analysis* (3rd ed.):
     http://www.stat.columbia.edu/~gelman/book/
   * Terminology reference: :ref:`terminology-index`

.. toctree::
   :hidden:
   :maxdepth: 1

   001-the-three-steps-of-bayesian-data-analysis
   002-general-notation-for-statistical-inference
   003-bayesian-inference
   004-discrete-bayesian-examples-genetics-and-spell-checking-with
   005-probability-as-a-measure-of-uncertainty
   006-example-probabilities-from-football-point-spreads
   007-example-calibration-for-record-linkage
   008-some-useful-results-from-probability-theory
   009-computation-and-software
   010-bayesian-inference-in-applied-statistics
   011-estimating-a-probability-from-binomial-data
   012-posterior-as-a-compromise-between-data-and-prior-information
   013-summarizing-posterior-inference
   014-informative-prior-distributions
   015-normal-distribution-with-known-variance
   016-other-standard-single-parameter-models
   017-informative-prior-distribution-for-cancer-rates
   018-noninformative-prior-distributions
   019-weakly-informative-prior-distributions
   020-averaging-over-nuisance-parameters
   021-normal-data-with-a-noninformative-prior-distribution
   022-normal-data-with-a-conjugate-prior-distribution
   023-multinomial-model-for-categorical-data
   024-multivariate-normal-model-with-known-variance
   025-multivariate-normal-with-unknown-mean-and-variance
   026-example-bayesian-analysis-of-a-bioassay-experiment-logistic-nonconjugate
   027-summary-of-elementary-modeling-and-computation
   028-normal-approximations-to-the-posterior-distribution
   029-large-sample-theory
   030-counterexamples-to-large-sample-asymptotic-bayesian-theorems
   031-frequency-evaluations-of-bayesian-inferences
   032-bayesian-interpretations-of-other-statistical-methods
   033-constructing-a-parameterized-prior-distribution
   034-exchangeability-and-hierarchical-models
   035-bayesian-analysis-of-conjugate-hierarchical-models
   036-normal-model-with-exchangeable-parameters
   037-example-parallel-experiments-in-eight-schools
   038-hierarchical-modeling-applied-to-a-meta-analysis
   039-weakly-informative-priors-for-variance-parameters
   040-the-place-of-model-checking-in-applied-bayesian-statistics
   041-do-the-inferences-from-the-model-make-sense
   042-posterior-predictive-checking
   043-graphical-posterior-predictive-checks
   044-model-checking-for-the-educational-testing-example
   045-measures-of-predictive-accuracy
   046-model-comparison-based-on-predictive-performance
   047-model-comparison-using-bayes-factors
   048-continuous-model-expansion
   049-implicit-assumptions-and-model-expansion-an-example
   050-bayesian-inference-requires-a-model-for-data-collection
   051-data-collection-models-and-ignorability
   052-sample-surveys
   053-designed-experiments
   054-sensitivity-and-the-role-of-randomization
   055-observational-studies
   056-censoring-and-truncation
   057-bayesian-decision-theory-in-different-contexts
   058-using-regression-predictions-survey-incentives
   059-multistage-decision-making-medical-screening
   060-hierarchical-decision-analysis-for-home-radon
   061-personal-vs-institutional-decision-analysis
   062-numerical-integration
   063-distributional-approximations
   064-direct-simulation-and-rejection-sampling
   065-importance-sampling
   066-how-many-simulation-draws-are-needed
   067-computing-environments
   068-debugging-bayesian-computing
   069-gibbs-sampler
   070-metropolis-and-metropolis-hastings-algorithms
   071-using-gibbs-and-metropolis-as-building-blocks
   072-inference-and-assessing-convergence
   073-effective-number-of-simulation-draws
   074-example-hierarchical-normal-model
   075-efficient-gibbs-samplers
   076-efficient-metropolis-jumping-rules
   077-further-extensions-to-gibbs-and-metropolis
   078-hamiltonian-monte-carlo
   079-hamiltonian-monte-carlo-for-a-hierarchical-model
   080-stan-developing-a-computing-environment
   081-finding-posterior-modes
   082-boundary-avoiding-priors-for-modal-summaries
   083-normal-and-related-mixture-approximations
   084-finding-marginal-posterior-modes-using-em
   085-conditional-and-marginal-posterior-approximations
   086-example-hierarchical-normal-model-continued
   087-variational-inference
   088-expectation-propagation
   089-other-approximations
   090-unknown-normalizing-factors
   091-conditional-modeling
   092-bayesian-analysis-of-classical-regression
   093-regression-for-causal-inference-incumbency-and-voting
   094-goals-of-regression-analysis
   095-assembling-the-matrix-of-explanatory-variables
   096-regularization-and-dimension-reduction
   097-unequal-variances-and-correlations
   098-including-numerical-prior-information
   099-regression-coefficients-exchangeable-in-batches
   100-example-forecasting-u-s-presidential-elections
   101-interpreting-a-normal-prior-distribution-as-extra-data
   102-varying-intercepts-and-slopes
   103-computation-batching-and-transformation
   104-analysis-of-variance-and-the-batching-of-coefficients
   105-hierarchical-models-for-batches-of-variance-components
   106-standard-generalized-linear-model-likelihoods
   107-working-with-generalized-linear-models
   108-weakly-informative-priors-for-logistic-regression
   109-overdispersed-poisson-regression-for-police-stops
   110-state-level-opinons-from-national-polls
   111-models-for-multivariate-and-multinomial-responses
   112-loglinear-models-for-multivariate-discrete-data
   113-aspects-of-robustness
   114-overdispersed-versions-of-standard-models
   115-posterior-inference-and-computation
   116-robust-inference-for-the-eight-schools
   117-robust-regression-using-t-distributed-errors
   118-notation
   119-multiple-imputation
   120-missing-data-in-the-multivariate-normal-and-t-models
   121-example-multiple-imputation-for-a-series-of-polls
   122-missing-values-with-counted-data
   123-example-an-opinion-poll-in-slovenia
   124-example-serial-dilution-assay
   125-example-population-toxicokinetics
   126-splines-and-weighted-sums-of-basis-functions
   127-basis-selection-and-shrinkage-of-coefficients
   128-non-normal-models-and-regression-surfaces
   129-gaussian-process-regression
   130-example-birthdays-and-birthdates
   131-latent-gaussian-process-models
   132-functional-data-analysis
   133-density-estimation-and-regression
   134-setting-up-and-interpreting-mixture-models
   135-example-reaction-times-and-schizophrenia
   136-label-switching-and-posterior-computation
   137-unspecified-number-of-mixture-components
   138-mixture-models-for-classification-and-regression
   139-bayesian-histograms
   140-dirichlet-process-prior-distributions
   141-dirichlet-process-mixtures
   142-beyond-density-estimation
   143-hierarchical-dependence
   144-density-regression

.. tags:: purpose: reference, topic: data analysis, domain: bayesian
