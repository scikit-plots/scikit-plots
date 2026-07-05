# ======================================================================
# ts_content.py  —  content store for the learn/time_series course
# ----------------------------------------------------------------------
# CONTENT[title] : full RST body of a lesson page (raw string; LaTeX
#                  backslashes stay literal, Unicode is written directly).
# MINDMAP[title] : lateral "See also" links (exact inventory titles).
# Keys MUST match ts_inventory.tsv titles exactly (fail-fast enforces).
# Prev/next navigation is generated automatically from curriculum order;
# MINDMAP is only for *lateral* jumps. Grows one batch at a time.
# ======================================================================

CONTENT: dict[str, str] = {}
MINDMAP: dict[str, list[str]] = {}


# ----------------------------------------------------------------------
# Stage 1 — Orientation
# ----------------------------------------------------------------------

CONTENT["What Are Time Series, and How Are They Used?"] = r"""
What it is
------------

A **time series** is a sequence of observations recorded **in time order**, usually at regular
intervals — daily sales, hourly temperature, quarterly GDP. Written :math:`\{x_t\}` for
:math:`t = 1, \dots, T`, its defining feature is that the **index is time** and the ordering is
part of the data: each point is related to the ones before it.

The moving parts
------------------

Classical analysis decomposes a series into a few recurring components:

* **trend** — the long-run drift up or down;
* **seasonality** — a **fixed-period** repeating pattern (weekly, monthly, yearly);
* **cyclic** behaviour — wandering swings of **no fixed length**;
* **residual / irregular** — the noise left once the rest is removed.

``statsmodels``' ``seasonal_decompose`` splits trend, seasonal and residual parts as an additive
or multiplicative sum. A useful subtlety: a series with **cycles but no fixed-length
seasonality** can still be stationary.

Why order matters
------------------

Because neighbouring points are **dependent**, time series break the **i.i.d.** assumption most
machine learning rests on. You cannot shuffle rows or use ordinary k-fold cross-validation — that
leaks future information into the past. Order is not a nuisance here; it is the **signal** that
makes forecasting possible at all.

Where it's used
----------------

Two complementary goals recur across every domain:

* **analysis** — understand the structure (trend, seasonality, autocorrelation);
* **forecasting** — predict future values, ideally with uncertainty intervals.

Typical applications include demand, price and capacity forecasting; monitoring and anomaly
detection; economics and finance; weather and climate; and any sensor or telemetry stream.
"""

CONTENT["Getting Started with R"] = r"""
The toolkit
------------

The source course is taught in **R**, the classic Box–Jenkins environment. This reference
reframes the same ideas in **Python**, using the standard scientific stack: **NumPy** for arrays,
**pandas** for time-indexed data, **statsmodels** for the models (AR / ARMA / ARIMA / SARIMAX,
the ACF / PACF, and the ADF / KPSS stationarity tests), and **Matplotlib** for plots. The
mathematics is identical in either language; only the syntax differs.

Loading a series
------------------

In Python a time series is a ``pandas.Series`` (or a DataFrame column) carrying a
**DatetimeIndex**, so pandas knows the spacing and can resample, align and difference for you:

.. code-block:: python

   import pandas as pd

   s = pd.read_csv("sales.csv", parse_dates=["date"], index_col="date")["value"]
   s = s.asfreq("MS")          # pin an explicit monthly-start frequency
   s.plot(title="Monthly sales")

Setting an explicit frequency (``asfreq``) matters: many models need to know the season length.

From R to Python
------------------

The classic R verbs map cleanly onto the Python stack:

* ``ts()`` → a ``pandas.Series`` with a ``DatetimeIndex``;
* ``acf()`` / ``pacf()`` → ``statsmodels.graphics.tsaplots.plot_acf`` / ``plot_pacf``;
* ``arima()`` / ``Arima()`` → ``statsmodels.tsa.arima.model.ARIMA``;
* ``forecast()`` → ``results.get_forecast(steps=...)``;
* ``auto.arima()`` → ``pmdarima.auto_arima``.

Reach for whichever environment you like; this course uses the Python calls throughout.
"""


# ----------------------------------------------------------------------
# Stage 2 — Stationarity
# ----------------------------------------------------------------------

CONTENT["A Gentle Introduction to Stationarity"] = r"""
What it is
------------

A series is **stationary** when its statistical behaviour **does not depend on time**: loosely,
the mean, the variance, and the way points co-vary with their lagged selves all stay **constant**
as the observation window slides. A stationary series has **no predictable long-run pattern** — no
trend and no fixed seasonal swing.

The intuition
--------------

Stationarity is what makes a series **learnable from a single realisation**. If the rules do not
change over time, the past is a fair guide to the future, and one long stretch of data is enough
to estimate them. Trend and seasonality break this because the distribution of the value shifts
with **when** you look. (The subtlety worth remembering: a series with cycles of **no fixed
length** can still be stationary — you cannot say in advance where the peaks will fall.)

Achieving stationarity
------------------------

The classical fix is **differencing** — model the change rather than the level:

.. math::

   \nabla x_t = x_t - x_{t-1},

which removes a linear trend; a **seasonal** difference :math:`x_t - x_{t-s}` removes a
period-:math:`s` pattern. (Detrending by regression is an alternative.) This "integrate" step is
the **I** in ARIMA. Over-differencing injects artefacts, so use the **smallest** order that works.

How to check
--------------

Plot first: a stationary series looks **flat and stable** around a constant level, with no drift.
Then formalise with tests — the **Augmented Dickey–Fuller (ADF)** test (null hypothesis:
*non-stationary*, so a small p-value argues for stationarity) and the **KPSS** test (null:
*stationary*), commonly read together. In ``statsmodels`` these are ``adfuller(x)`` and ``kpss(x)``.
"""


MINDMAP.update({
    "What Are Time Series, and How Are They Used?": [
        "A Gentle Introduction to Stationarity", "Getting Started with R",
        "ARIMA Models: How Nonstationary Models Are Built from Stationary Ones",
        "Exponential Smoothing Models",
    ],
    "Getting Started with R": [
        "What Are Time Series, and How Are They Used?",
        "A Gentle Introduction to Stationarity", "Sample ACF and Sample PACF",
    ],
    "A Gentle Introduction to Stationarity": [
        "Weak and Strong Stationarity", "What Are Time Series, and How Are They Used?",
        "ARIMA Models: How Nonstationary Models Are Built from Stationary Ones",
        "Linear Processes",
    ],
})


# ----------------------------------------------------------------------
# Stage 2 — Stationarity (cont.)
# ----------------------------------------------------------------------

CONTENT["Weak and Strong Stationarity"] = r"""
Two definitions
----------------

"Stationarity" comes in two strengths. **Strong** (strict) stationarity constrains the **entire
distribution**; **weak** (second-order, or covariance) stationarity constrains only the **first
two moments**. Most classical time-series theory — including ARMA — needs only the weaker one.

Strong stationarity
--------------------

A series is **strictly stationary** if shifting time leaves its **whole joint distribution**
unchanged: for any lag :math:`h` and any set of times, the block :math:`(x_{t_1}, \dots, x_{t_k})`
has the same joint distribution as :math:`(x_{t_1+h}, \dots, x_{t_k+h})`,

.. math::

   (x_{t_1}, \dots, x_{t_k}) \;\stackrel{d}{=}\; (x_{t_1+h}, \dots, x_{t_k+h}).

Nothing about the probabilistic structure depends on **when** you look — a strong condition that
is hard to verify from a single finite sample.

Weak stationarity
-----------------

A series is **weakly stationary** if its **mean is constant**, its **variance is finite and
constant**, and its **autocovariance depends only on the lag** — not on absolute time:

.. math::

   \mathbb{E}[x_t] = \mu, \qquad
   \gamma(s, t) = \operatorname{Cov}(x_s, x_t) = \gamma(|s - t|).

This is all that ARMA modelling requires, and — unlike strict stationarity — it is checkable from
data through the sample mean and the sample autocorrelation.

How they relate
----------------

The two coincide under mild conditions. **Strong stationarity plus finite second moments implies
weak** stationarity. The converse fails in general, **except for Gaussian** processes: a Gaussian
process is fully described by its mean and covariance, so **weak + Gaussian implies strong**.
**White noise** — zero mean, constant variance, zero autocorrelation — is the canonical weakly
stationary building block.
"""


# ----------------------------------------------------------------------
# Stage 3 — Linear & ARMA Processes
# ----------------------------------------------------------------------

CONTENT["Linear Processes"] = r"""
What it is
------------

A **linear process** writes a series as a **weighted sum of white-noise shocks** — a linear filter
applied to an underlying noise sequence. It is the general form from which AR, MA and ARMA models
all descend.

The white-noise filter
------------------------

With white noise :math:`\{w_t\}` (mean zero, variance :math:`\sigma^2`), a linear process is

.. math::

   x_t = \mu + \sum_{j=-\infty}^{\infty} \psi_j\, w_{t-j},
   \qquad \sum_{j=-\infty}^{\infty} |\psi_j| < \infty.

The **absolute summability** of the weights :math:`\psi_j` guarantees the sum **converges** and the
result is **stationary**; it is a little stronger than square-summability and is what lets the
usual limit theorems apply.

Causal and one-sided
----------------------

A linear process is **causal** when it uses only **present and past** shocks — the future never
appears:

.. math::

   x_t = \mu + \sum_{j=0}^{\infty} \psi_j\, w_{t-j}.

This one-sided form is the **MA(∞)** representation. It is exactly the sense in which a causal ARMA
model can be "unrolled" into an infinite moving average of past noise.

Why it's central
----------------

**Wold's decomposition** says that *every* weakly stationary process has a linear-process (MA(∞))
component — so linear processes are not one model among many but the **canonical template** for
stationary series. Causality and invertibility (next lesson) are precisely the conditions under
which an ARMA model collapses into, or inverts back out of, this form.
"""

CONTENT["Understanding ARMA Processes"] = r"""
The model
----------

An **ARMA(p, q)** process blends two mechanisms: an **autoregressive (AR)** part, where the value
depends on its own **p** past values, and a **moving-average (MA)** part, where it depends on the
last **q** white-noise **shocks**:

.. math::

   x_t = \phi_1 x_{t-1} + \dots + \phi_p x_{t-p}
         + w_t + \theta_1 w_{t-1} + \dots + \theta_q w_{t-q}.

Backshift form
----------------

Using the **backshift operator** :math:`B` (with :math:`B^j x_t = x_{t-j}`), the model compresses
to

.. math::

   \phi(B)\, x_t = \theta(B)\, w_t,

where :math:`\phi(B) = 1 - \phi_1 B - \dots - \phi_p B^p` is the **AR polynomial** and
:math:`\theta(B) = 1 + \theta_1 B + \dots + \theta_q B^q` the **MA polynomial**. This algebra makes
the next two properties easy to state.

Causality and invertibility
----------------------------

Two root conditions govern behaviour. The process is **causal** — expressible as a one-sided MA(∞)
of past shocks — when **all roots of** :math:`\phi(z)` lie **outside** the unit circle
(:math:`|z| > 1`). It is **invertible** — expressible as an AR(∞) in past values — when **all roots
of** :math:`\theta(z)` lie **outside** the unit circle. Causality is what makes an ARMA a proper
linear process; invertibility makes its parameters **identifiable** from the data.

Watch for redundancy
----------------------

If the AR and MA polynomials share a **common factor**, the model is **over-parameterised**: the
factor cancels and a simpler model fits identically. The extreme case :math:`\phi(B) = \theta(B)`
reduces the whole model to :math:`x_t = w_t`, plain white noise. Always **cancel common roots**
before trusting a fit; ``statsmodels`` exposes the fitted ``.arroots`` and ``.maroots`` for exactly
this check.
"""


MINDMAP.update({
    "Weak and Strong Stationarity": [
        "A Gentle Introduction to Stationarity", "Linear Processes",
        "Understanding ARMA Processes", "Sample ACF and Sample PACF",
    ],
    "Linear Processes": [
        "Understanding ARMA Processes", "Weak and Strong Stationarity",
        "Best Linear Predictor of a Stationary Process",
        "Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q)",
    ],
    "Understanding ARMA Processes": [
        "Linear Processes",
        "Computing ACFs of Causal AR(2) Processes Using Difference Equations",
        "Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q)",
        "Maximum Likelihood Estimation for ARMA Models (Gaussian MLE)",
        "ARIMA Models: How Nonstationary Models Are Built from Stationary Ones",
    ],
})


# ----------------------------------------------------------------------
# Stage 3 — Linear & ARMA Processes (cont.)
# ----------------------------------------------------------------------

CONTENT["Computing ACFs of Causal AR(2) Processes Using Difference Equations"] = r"""
The recursion
--------------

For a causal **AR(2)**, :math:`x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + w_t`, the autocorrelations
obey the **same recursion as the process itself** — a homogeneous linear **difference equation**:

.. math::

   \rho(h) = \phi_1\,\rho(h-1) + \phi_2\,\rho(h-2), \qquad h \ge 1.

With :math:`\rho(0) = 1`, the **Yule–Walker** start gives :math:`\rho(1) = \phi_1 / (1 - \phi_2)`,
and every later lag follows by iterating.

Solving it
------------

Rather than iterate forever, solve the difference equation **in closed form** through its
**characteristic equation** — equivalently, the **roots** :math:`z_1, z_2` of the AR polynomial
:math:`1 - \phi_1 z - \phi_2 z^2`. The ACF is then a combination of the terms :math:`z_i^{-h}`,
whose magnitudes are controlled by how far the roots sit **outside** the unit circle (causality
guarantees they do).

Two regimes
------------

The **discriminant** :math:`\phi_1^2 + 4\phi_2` decides the shape. When it is **positive**, the
roots are **real** and the ACF is a sum of two **damped exponentials** (decaying monotonically, with
uniform or alternating sign). When it is **negative**, the roots are **complex conjugates** and the
ACF is a **damped sinusoid** — a decaying oscillation with system frequency

.. math::

   f_0 = \frac{1}{2\pi}\cos^{-1}\!\left( \frac{\phi_1}{2\sqrt{-\phi_2}} \right).

What it tells you
------------------

Either way the ACF **tails off** toward zero but never truly **cuts off** — the signature of an
autoregressive process. (Its partner, the **PACF**, does cut off, after lag 2.) Reading whether the
decay is exponential or oscillatory is a first clue to the underlying dynamics.
"""

CONTENT["Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q)"] = r"""
The general rule
------------------

The AR(2) result generalises. For an **AR(p)** — and for an **ARMA(p, q)** once you are far enough
from the start — the autocorrelations satisfy a **homogeneous difference equation driven by the AR
polynomial**:

.. math::

   \gamma(h) - \phi_1\,\gamma(h-1) - \dots - \phi_p\,\gamma(h-p) = 0, \qquad h > q.

Its general solution is a **linear combination of the terms** :math:`z_i^{-h}` from the **p roots**
of :math:`\phi(z)` — damped exponentials for real roots, damped sinusoids for complex ones.

Where the MA part enters
--------------------------

The moving-average order **q** does not change this decay law; it only sets the **initial
conditions**. For lags :math:`h \le q` the MA terms contribute directly, so the first few
autocorrelations look "irregular"; from :math:`h > q` onward the pure AR recursion takes over and
the smooth decay begins. In short, the **AR roots fix the pattern**, the **MA part fixes the first
few values**.

The shape of the ACF
----------------------

Because the solution is built from roots **outside** the unit circle, every term **decays**, so an
AR or ARMA autocorrelation **tails off** geometrically (possibly oscillating) but **never reaches
exactly zero**. A pure **MA(q)**, by contrast, has an ACF that **cuts off** sharply after lag
:math:`q` — a clean visual distinction.

ACF vs PACF
------------

This is the heart of **Box–Jenkins identification**: the **ACF cutting off** points to an **MA**
order, while the **PACF cutting off** points to an **AR** order. AR and ARMA both leave the ACF
tailing off, so the PACF (next lesson) is what pins down the autoregressive order.
"""


# ----------------------------------------------------------------------
# Stage 4 — Prediction & the Sample ACF / PACF
# ----------------------------------------------------------------------

CONTENT["Best Linear Predictor of a Stationary Process"] = r"""
The prediction problem
------------------------

Given a stationary series observed up to now, the **best linear predictor** of the next value is
the linear combination of past observations
:math:`\hat{x}_{n+1} = \sum_{k=1}^{n} a_k\, x_{n+1-k}` that **minimises the mean-squared error**.
"Best" here means **within linear rules**; for a Gaussian process it is the best predictor of
**any** kind.

The projection principle
--------------------------

The solution is **geometric**: treat random variables as vectors with inner product
:math:`\langle X, Y\rangle = \mathbb{E}[XY]`, and the best predictor is the **orthogonal
projection** of the target onto the span of the predictors. Optimality is characterised by the
**orthogonality principle** — the prediction **error is uncorrelated with every predictor**:

.. math::

   \mathbb{E}\big[(x_{n+1} - \hat{x}_{n+1})\, x_{n+1-k}\big] = 0, \qquad k = 1, \dots, n.

Writing these conditions out gives the **prediction equations**, a linear system
:math:`\Gamma_n \mathbf{a} = \gamma_n` in the autocovariances — the same **Toeplitz** system as
Yule–Walker.

Solving efficiently
--------------------

Solving that system afresh at each order is wasteful. The **Durbin–Levinson recursion** builds the
order-:math:`h` predictor from the order-:math:`(h-1)` one **without inverting a matrix**, updating
the coefficients and the error variance in place — cheap and numerically stable.

From predictor to PACF
------------------------

The recursion delivers a bonus. The **last coefficient** :math:`\phi_{hh}` of the best
order-:math:`h` predictor **is the partial autocorrelation** at lag :math:`h` — the correlation
between :math:`x_t` and :math:`x_{t-h}` **after removing** the linear influence of the intervening
values. For an **AR(p)** these coefficients vanish for :math:`h > p`, so the **PACF cuts off at lag
p** — the property that makes it the tool for reading autoregressive order.
"""


MINDMAP.update({
    "Computing ACFs of Causal AR(2) Processes Using Difference Equations": [
        "Understanding ARMA Processes",
        "Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q)",
        "Sample ACF and Sample PACF", "Weak and Strong Stationarity",
    ],
    "Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q)": [
        "Computing ACFs of Causal AR(2) Processes Using Difference Equations",
        "Understanding ARMA Processes", "Sample ACF and Sample PACF",
        "Order Selection for Time Series Models",
    ],
    "Best Linear Predictor of a Stationary Process": [
        "Sample ACF and Sample PACF",
        "Preliminary Estimation for AR Models and the Yule–Walker Equations",
        "Understanding ARMA Processes", "Beyond One-Step Ahead Predictions",
    ],
})


# ----------------------------------------------------------------------
# Stage 4 — Prediction & the Sample ACF / PACF (cont.)
# ----------------------------------------------------------------------

CONTENT["Sample ACF and Sample PACF"] = r"""
From process to sample
------------------------

In practice you never see the true ACF — you **estimate** it from one finite record. The **sample
autocovariance** and **sample ACF** are

.. math::

   \hat{\gamma}(h) = \frac{1}{n}\sum_{t=1}^{n-|h|} (x_t - \bar{x})(x_{t+|h|} - \bar{x}),
   \qquad \hat{\rho}(h) = \frac{\hat{\gamma}(h)}{\hat{\gamma}(0)}.

Both grow noisier at **large lags**, where few pairs contribute — a common rule keeps
:math:`n \ge 50` and lags :math:`h \le n/4`.

Significance bands
--------------------

How large is "large"? If the series were **white noise**, then for big :math:`n` each
:math:`\hat{\rho}(h)` is approximately :math:`\mathcal{N}(0, 1/n)`, so correlations beyond

.. math::

   \pm \frac{1.96}{\sqrt{n}}

are unlikely by chance — the dashed bands on every correlogram. For a genuinely correlated series
the bands widen, following **Bartlett's formula**
:math:`\operatorname{Var}(\hat{\rho}_k) \approx \frac{1}{n}\big(1 + 2\sum_{j<k}\rho_j^2\big)`.

The sample PACF
----------------

The **sample PACF** applies the **Durbin–Levinson recursion** to the *sample* autocovariances,
producing :math:`\hat{\phi}_{hh}` at each lag. It shares the same white-noise bands
:math:`\pm 1.96/\sqrt{n}`, so significant spikes stand out in the same way.

Reading them together
----------------------

Identification is a two-plot habit. An **ACF that cuts off** after lag :math:`q` alongside a
**PACF that tails off** suggests **MA(q)**; a **PACF that cuts off** after lag :math:`p` with a
**tailing ACF** suggests **AR(p)**; **both tailing off** suggests **ARMA**. In ``statsmodels`` these
plots are ``plot_acf`` and ``plot_pacf``.
"""


# ----------------------------------------------------------------------
# Stage 5 — Estimation
# ----------------------------------------------------------------------

CONTENT["Preliminary Estimation for AR Models and the Yule–Walker Equations"] = r"""
The idea
----------

The **Yule–Walker** method is the simplest way to fit an **AR(p)** model: it is **method of
moments** — match the model's theoretical autocovariances to the ones you measure, and solve for
the coefficients. No optimisation, no starting values.

The equations
--------------

Multiplying the AR recursion by lagged values and taking expectations links the coefficients to the
autocovariances through a **Toeplitz** linear system:

.. math::

   \Gamma_p\, \boldsymbol{\phi} = \boldsymbol{\gamma}_p, \qquad
   \Gamma_p = \big[\gamma(i-j)\big]_{i,j=1}^{p}, \quad
   \boldsymbol{\gamma}_p = \big(\gamma(1), \dots, \gamma(p)\big)^{\!\top}.

Because :math:`\Gamma_p` is a full-rank symmetric Toeplitz matrix, the solution
:math:`\boldsymbol{\phi} = \Gamma_p^{-1}\boldsymbol{\gamma}_p` always exists.

Plugging in the data
----------------------

Replace each :math:`\gamma(\cdot)` with its **sample** counterpart :math:`\hat{\gamma}(\cdot)` to
get :math:`\hat{\boldsymbol{\phi}} = \hat{\Gamma}_p^{-1}\hat{\boldsymbol{\gamma}}_p`, and read off
the noise variance as
:math:`\hat{\sigma}_w^2 = \hat{\gamma}(0) - \hat{\boldsymbol{\phi}}^{\!\top}\hat{\boldsymbol{\gamma}}_p`.
The **Durbin–Levinson recursion** solves the system without an explicit inverse; in ``statsmodels``
this is ``yule_walker``.

Why it's a good start
----------------------

For a genuine **AR(p)**, Yule–Walker estimators are **consistent**, **asymptotically normal**, and
**as efficient as maximum likelihood** — and they always return a **causal (stationary)** model.
That reliability makes them the standard **preliminary estimate** and a natural **starting point**
for the likelihood methods below. (For **ARMA** the same idea works but is **no longer efficient**,
so it is used only to seed the optimiser.)
"""

CONTENT["Maximum Likelihood Estimation for ARMA Models (Gaussian MLE)"] = r"""
The likelihood
----------------

If the ARMA process is driven by **Gaussian** white noise, then the whole sample
:math:`(x_1, \dots, x_n)` is **multivariate normal** with a covariance matrix determined by the
parameters. **Maximum likelihood** picks the :math:`(\boldsymbol{\phi}, \boldsymbol{\theta},
\sigma_w^2)` that make the observed data **most probable** under that model.

The innovations trick
----------------------

Working with the raw :math:`n \times n` covariance is expensive. The standard route rewrites the
likelihood through the **one-step prediction errors** ("innovations") and their variances —
computed cheaply by the **innovations algorithm** or a **Kalman filter**. This turns the likelihood
into a product over time and avoids inverting a large matrix.

A harder optimisation
----------------------

Unlike Yule–Walker, there is **no closed form** — the likelihood is **nonlinear**, especially in the
**moving-average** parameters, so it must be **maximised numerically**. That optimisation can be
finicky, which is exactly why good **starting values** (from Yule–Walker or the Hannan–Rissanen
method) matter. One further choice: the **conditional** likelihood fixes the initial values, while
the **exact** likelihood models them too.

Why it's the default
----------------------

Maximum likelihood is **consistent**, **asymptotically normal**, and **efficient** — the
lowest-variance estimates available — and, remarkably, its **asymptotic distribution is the same
even when the data are not Gaussian**. That combination is why ``statsmodels``' ``ARIMA`` fits by
**exact MLE** (via a state-space / Kalman filter) by default.
"""


MINDMAP.update({
    "Sample ACF and Sample PACF": [
        "Best Linear Predictor of a Stationary Process",
        "Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q)",
        "Preliminary Estimation for AR Models and the Yule–Walker Equations",
        "Order Selection for Time Series Models",
    ],
    "Preliminary Estimation for AR Models and the Yule–Walker Equations": [
        "Best Linear Predictor of a Stationary Process", "Sample ACF and Sample PACF",
        "Maximum Likelihood Estimation for ARMA Models (Gaussian MLE)",
        "Understanding ARMA Processes",
    ],
    "Maximum Likelihood Estimation for ARMA Models (Gaussian MLE)": [
        "Preliminary Estimation for AR Models and the Yule–Walker Equations",
        "Understanding ARMA Processes", "Order Selection for Time Series Models",
        "Diagnostics After Fitting a Time Series Model",
    ],
})


# ----------------------------------------------------------------------
# Stage 6 — Building & Forecasting Models
# ----------------------------------------------------------------------

CONTENT["Diagnostics After Fitting a Time Series Model"] = r"""
The goal
----------

A fitted model is only as good as its **residuals**. If the model has captured all the structure,
what remains should be **white noise** — patternless, uncorrelated, constant-variance. Diagnostics
**standardise** the residuals (divide by their standard error) and then test that claim.

What to look at
----------------

Three views, read together:

* **residual plot** over time — should scatter randomly around zero, with **no trend and no
  changing spread** (heteroscedasticity);
* **residual ACF / correlogram** — essentially **no spikes** outside the
  :math:`\pm 1.96/\sqrt{n}` bands (spikes at **seasonal** lags hint at missed seasonality);
* **normality** — a **Q–Q plot** and histogram, since valid prediction intervals lean on roughly
  normal errors.

``statsmodels`` bundles these panels into ``results.plot_diagnostics()``.

The Ljung-Box test
--------------------

Eyeballing the ACF is subjective; the **Ljung–Box** test makes it formal. It is a **portmanteau**
test of the residual autocorrelations **jointly** up to lag :math:`h`:

.. math::

   Q = n(n+2)\sum_{k=1}^{h} \frac{\hat{r}_k^2}{n-k},

compared against a :math:`\chi^2` distribution. The null is **no autocorrelation** (white-noise
residuals), so a **small p-value means the model is inadequate**. The degrees of freedom are reduced
by the number of fitted AR / MA parameters. (``acorr_ljungbox`` in ``statsmodels``.)

When it fails
--------------

Failing diagnostics are a **diagnosis, not a dead end**. Autocorrelation left in the residuals says
the model is **missing structure** — add an **AR or MA** term, or, if the leftover spikes fall at
**seasonal** lags, move to a **seasonal** model. Re-fit and re-check until the residuals look like
noise.
"""

CONTENT["Order Selection for Time Series Models"] = r"""
The trade-off
--------------

Every extra parameter **improves the in-sample fit** but risks **overfitting** — chasing noise that
will not repeat. Order selection is the search for a model that is **complex enough to fit, simple
enough to generalise**: the **parsimony** principle at the heart of Box–Jenkins.

Information criteria
----------------------

The standard tools score fit **against** complexity. Both the **Akaike** and **Bayesian**
information criteria reward the likelihood and **penalise** the parameter count :math:`k`:

.. math::

   \mathrm{AIC} = 2k - 2\ln \hat{L}, \qquad \mathrm{BIC} = k\ln n - 2\ln \hat{L}.

**Lower is better**, and you compare candidates fitted to the **same** data. (The small-sample
correction **AICc** is safer when :math:`n` is not large.)

AIC versus BIC
----------------

The two differ only in the penalty. **BIC**'s per-parameter cost :math:`\ln n` is harsher than
**AIC**'s :math:`2` (once :math:`n > 7`), so **BIC favours simpler models** and is preferred when
**parsimony** matters; **AIC** tends to pick slightly richer models and suits **predictive
accuracy**. When they disagree, the choice is yours to justify.

Putting it together
--------------------

In practice: pick **d** by differencing until stationary (ADF / KPSS), read tentative **p, q** off
the ACF / PACF, then **grid-search** nearby orders and keep the lowest-criterion model **that also
passes diagnostics** — a lower AIC means nothing if the residuals are still autocorrelated. Tools
like ``pmdarima.auto_arima`` automate the search; ``statsmodels`` exposes ``.aic`` and ``.bic``.
"""

CONTENT["ARIMA Models: How Nonstationary Models Are Built from Stationary Ones"] = r"""
The core idea
--------------

ARMA needs a **stationary** series, but real data trends and drifts. **ARIMA** bridges the gap with
one move: **difference the series until it is stationary**, fit an ordinary **ARMA** to the
differenced version, and the model inherits ARMA's whole toolkit. The **"I"** stands for
**Integrated** — the series must be *un*-differenced (integrated) to recover the original.

The model
----------

An **ARIMA(p, d, q)** applies the :math:`d`-th difference :math:`(1-B)^d` before the ARMA machinery:

.. math::

   \phi(B)\,(1 - B)^d\, x_t = \theta(B)\, w_t.

Here :math:`(1-B)^d` is the **differencing operator**, :math:`\phi(B)` the AR polynomial and
:math:`\theta(B)` the MA polynomial. A series needing :math:`d` differences to become stationary is
called **integrated of order** :math:`d`, or :math:`I(d)`.

Choosing d
------------

One difference (:math:`d = 1`) removes a **linear** trend; two (:math:`d = 2`) removes a
**quadratic** one; seasonal patterns need a **seasonal** difference (next lesson). Pick the
**smallest** :math:`d` that makes the ADF / KPSS tests read stationary — **over-differencing**
inflates the variance and injects artificial correlation, so more is not better.

Forecasting back
------------------

Fitting happens on the **differenced** scale, but forecasts are wanted on the **original** one. The
model **"integrates"** — cumulatively sums — its differenced-scale predictions back up to the level
of the raw series, carrying the forecast **uncertainty** with it. In ``statsmodels`` this is all
handled by ``ARIMA(y, order=(p, d, q))``.
"""


MINDMAP.update({
    "Diagnostics After Fitting a Time Series Model": [
        "Order Selection for Time Series Models",
        "Maximum Likelihood Estimation for ARMA Models (Gaussian MLE)",
        "Sample ACF and Sample PACF",
        "ARIMA Models: How Nonstationary Models Are Built from Stationary Ones",
    ],
    "Order Selection for Time Series Models": [
        "Diagnostics After Fitting a Time Series Model", "Sample ACF and Sample PACF",
        "ARIMA Models: How Nonstationary Models Are Built from Stationary Ones",
        "Understanding ARMA Processes",
    ],
    "ARIMA Models: How Nonstationary Models Are Built from Stationary Ones": [
        "A Gentle Introduction to Stationarity", "Understanding ARMA Processes",
        "SARIMA Models: Seasonal ARIMA", "Order Selection for Time Series Models",
    ],
})


# ----------------------------------------------------------------------
# Stage 6 — Building & Forecasting Models (cont.)  [completes the course]
# ----------------------------------------------------------------------

CONTENT["SARIMA Models: Seasonal ARIMA"] = r"""
The seasonal problem
----------------------

Plain ARIMA handles **trend** but not **seasonality** — a pattern that repeats every :math:`s` steps
(12 for monthly data, 4 for quarterly). **SARIMA** extends it by adding a **second, seasonal** ARIMA
operating at the seasonal lag, so one model captures **both** short-term and season-to-season
structure.

The notation
--------------

A SARIMA is written **SARIMA(p, d, q)(P, D, Q)** with seasonal period **s**. The first triple is the
ordinary non-seasonal part; the second is its **seasonal mirror** — :math:`P` seasonal AR terms,
:math:`D` seasonal differences, :math:`Q` seasonal MA terms — each acting at multiples of :math:`s`.
In operator form the two multiply:

.. math::

   \Phi_P(B^s)\,\phi_p(B)\,(1-B)^d\,(1-B^s)^D\, x_t
     = \Theta_Q(B^s)\,\theta_q(B)\, w_t.

The seasonal difference
------------------------

The workhorse is the **seasonal difference** :math:`(1 - B^s)`, which subtracts the value from **one
full season ago**:

.. math::

   \nabla_s x_t = x_t - x_{t-s}.

Just as ordinary differencing removes a trend, this removes a **repeating seasonal** pattern. Data
with **both** trend and seasonality may need **both** a regular difference (:math:`d`) and a seasonal
one (:math:`D`). Spikes in the ACF / PACF at lags :math:`s, 2s, \dots` point to the seasonal orders.

Fitting and pitfalls
----------------------

In ``statsmodels`` this is ``SARIMAX(order=(p,d,q), seasonal_order=(P,D,Q,s))``. Two cautions:
seasonal terms **cost parameters**, so imposing them on a **non-seasonal** series adds noise and can
**degrade** forecasts; and for **multiple** overlapping seasonalities (say daily *and* weekly),
specialised tools like **TBATS** or **Prophet** fit better than a single seasonal period.
"""

CONTENT["Beyond One-Step Ahead Predictions"] = r"""
Forecasting further
--------------------

A one-step forecast is just the start. Usually you want the value :math:`h` steps ahead — the
**h-step forecast** — which is the **conditional expectation**
:math:`\hat{x}_{n+h} = \mathbb{E}[x_{n+h} \mid x_1, \dots, x_n]`: the mean of the future given
everything observed so far.

Building it up
----------------

The practical route is **recursive** ("plug-in"): apply the one-step predictor, then **feed its
output back in** as if it were observed, and step forward again. Each future value the model needs
but has not seen is replaced by its **own forecast**, so errors from early steps **propagate** into
later ones.

Uncertainty grows
------------------

The crucial consequence: **forecast uncertainty widens with the horizon**. Every step adds a fresh
shock whose variance **accumulates**, so the **prediction intervals fan out** the further you look —
a forecast for next month is far tighter than one for next year. Reporting the **interval**, not just
the point, is essential. (``get_forecast(steps=h)`` in ``statsmodels`` returns both.)

Two limiting cases
--------------------

How the intervals grow depends on the model. For a **stationary ARMA**, forecasts **revert to the
mean** and the interval width **levels off** at the unconditional variance — the process forgets its
current state. For an **ARIMA** with differencing, there is no mean to revert to, so the intervals
**keep widening without bound**, random-walk style. Either way, small **parameter errors compound**
over long horizons.
"""

CONTENT["Exponential Smoothing Models"] = r"""
The idea
----------

**Exponential smoothing** forecasts with a **weighted average of past observations**, where the
weights **decay exponentially** into the past — recent data counts most, older data fades but never
fully vanishes. It is a different lineage from ARIMA, built around **components** (level, trend,
season) rather than autocorrelations.

Simple smoothing
------------------

The simplest form, **Simple Exponential Smoothing (SES)**, tracks a single **level** and suits
series with **no trend or seasonality**:

.. math::

   \hat{x}_{t+1} = \alpha\, x_t + (1 - \alpha)\, \hat{x}_t, \qquad 0 < \alpha < 1.

The smoothing parameter :math:`\alpha` sets the memory: near 1 reacts fast to recent values, near 0
stays smooth and sluggish. Its forecasts are **flat**.

Adding trend and season
------------------------

Two extensions handle richer data. **Holt's linear** method adds a **trend** component (a second
parameter :math:`\beta`), giving sloped forecasts. **Holt–Winters** adds a **seasonal** component too
(a third parameter :math:`\gamma`), either **additive** or **multiplicative** — the standard choice
for series with **both** trend and seasonality. Together these form the **ETS** (Error–Trend–Seasonal)
family.

Smoothing or ARIMA?
--------------------

The two approaches are **complementary**, not rivals: ARIMA describes a series through its
**autocorrelations**, exponential smoothing through its **trend and seasonal structure**. Which wins
is **empirical** — smoothing often shines on strongly trended, seasonal data, ARIMA on stationary,
mean-reverting data. In ``statsmodels`` these live in ``SimpleExpSmoothing``, ``Holt`` and
``ExponentialSmoothing`` (Holt–Winters).
"""


MINDMAP.update({
    "SARIMA Models: Seasonal ARIMA": [
        "ARIMA Models: How Nonstationary Models Are Built from Stationary Ones",
        "A Gentle Introduction to Stationarity", "Exponential Smoothing Models",
        "Beyond One-Step Ahead Predictions",
    ],
    "Beyond One-Step Ahead Predictions": [
        "Best Linear Predictor of a Stationary Process",
        "ARIMA Models: How Nonstationary Models Are Built from Stationary Ones",
        "Exponential Smoothing Models", "Diagnostics After Fitting a Time Series Model",
    ],
    "Exponential Smoothing Models": [
        "ARIMA Models: How Nonstationary Models Are Built from Stationary Ones",
        "SARIMA Models: Seasonal ARIMA", "Beyond One-Step Ahead Predictions",
        "What Are Time Series, and How Are They Used?",
    ],
})
