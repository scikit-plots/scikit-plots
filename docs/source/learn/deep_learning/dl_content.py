"""
CONTENT + MINDMAP for learn/deep_learning lessons.

Keys must be EXACT titles from dl_inventory.tsv (see DEEP_LEARNING.md).
CONTENT[title] = raw RST body; MINDMAP[title] = lateral 'See also' titles.
Populated batch by batch in curriculum order.
"""

CONTENT: dict[str, str] = {}
MINDMAP: dict[str, list[str]] = {}


# ----------------------------------------------------------------------
# Stage 1 — Introduction to Deep Learning
# ----------------------------------------------------------------------

CONTENT["What is a Neural Network?"] = r"""
A single neuron
----------------

The atom of a neural network is the **neuron**. It takes inputs, forms a **weighted sum** plus a
**bias**, and passes the result through a **nonlinear activation**:

.. math::

   z = w_1 x_1 + \dots + w_n x_n + b = \mathbf{w}^{\!\top}\mathbf{x} + b, \qquad a = g(z).

Andrew Ng's opening example fits **house size → price** with one neuron whose activation is a
**ReLU** (*rectified linear unit*), :math:`g(z) = \max(0, z)` — flat at zero for negative inputs,
rising linearly after. That single unit is already a tiny predictor.

Stacking into a network
------------------------

Real problems have many inputs. Stack neurons side by side into a **layer**, feed one layer's
outputs into the next, and you have a **network** — Ng's Lego-brick analogy. With features like
**size, number of bedrooms, zip code, wealth** feeding a **hidden layer** that feeds a **price**
output, the middle units can come to represent intermediate ideas such as **family size**,
**walkability** or **school quality**. Every input connects to every hidden unit
(**fully connected**, or **dense**).

Learning the features
----------------------

The crucial point: **you never specify those intermediate concepts**. You supply only input–output
pairs :math:`(x, y)` — sizes and prices — and **gradient descent** discovers whatever hidden-unit
features best predict the target. Given enough units, a network can approximate very complex
mappings; the rest of this course builds that learning machinery from a single neuron up.

Not really a brain
--------------------

The "a neural network is like the brain" line is a **loose analogy** — handy for a first mental
picture, but easily **oversold**. An artificial neuron is a small piece of arithmetic (a weighted
sum and a nonlinearity), not a biological cell. It is more honest to picture a network as a
**flexible, trainable function approximator**.
"""

CONTENT["Supervised Learning and Neural Networks"] = r"""
Learning a mapping
--------------------

Nearly all the commercial value of neural networks comes from **supervised learning**: you are given
labelled examples — inputs :math:`x` paired with correct outputs :math:`y` — and you learn the
**function that maps** :math:`x \to y`. House features → price, an ad and a user → click-or-not, an
image → the object it contains: each is the same problem, *learn* :math:`x \to y` *from examples*.

Matching the architecture
--------------------------

The **input type** dictates the network. A **standard (fully connected)** network suits **tabular**
features (home prices, ad click-through). **Convolutional** networks (**CNNs**) suit **images** —
computer vision, photo tagging. **Recurrent** networks (**RNNs**) suit **sequences** — audio → text,
language translation. Rich inputs (autonomous driving from image + radar) often use
**hybrid / custom** designs. Same learning principle, different wiring.

Structured vs unstructured
----------------------------

Data splits into two kinds. **Structured** data is table-like: rows and columns where **each feature
has a clear meaning** (a house's size, a user's age). **Unstructured** data is raw **audio, images,
text**, where the features are pixels or waveform samples or words and **no single feature means much
on its own**. Classical algorithms handle structured data well but **struggle on unstructured** data.

Why it matters
----------------

This is exactly where neural networks changed the game: they made computers **far better at
unstructured data** — speech, vision, language — unlocking applications that were out of reach a
decade ago. Much of the economic value, meanwhile, still comes from **structured** data inside
companies, where accurate predictions on large databases translate directly into money.
"""

CONTENT["Why Deep Learning is Taking Off"] = r"""
An old idea, newly working
----------------------------

The mathematics of neural networks is **decades old**, so why the recent explosion? Not one
breakthrough but a **convergence** — the raw ingredients finally reached the scale where deep
networks **decisively outperform** the alternatives.

Scale drives performance
--------------------------

Ng summarises it with a single picture: plot **performance** against the **amount of labelled
data**. Traditional methods (logistic regression, SVMs) improve for a while, then **plateau**.
Neural networks keep climbing, and **bigger networks climb higher** — small < medium < large. The
pattern has two regimes: with **little** data, careful feature engineering and skill can matter more
than model size, so the ordering blurs; with **lots** of data, a **large network** wins clearly.

The three drivers
------------------

Three forces made that scale reachable. **Data** — a digitised world (phones, sensors, the web)
produces the huge labelled datasets networks feed on. **Computation** — GPUs, faster hardware and
distributed training make large models trainable in reasonable time. **Algorithms** — better design
speeds learning; the switch from the **sigmoid** to the **ReLU** activation is the classic example,
easing the **vanishing-gradient** problem so gradient descent converges much faster.

The virtuous cycle
--------------------

These drivers reinforce each other. Faster hardware and better algorithms shorten the
**idea → code → experiment** loop, so researchers iterate more quickly; better models attract more
**users**, who generate more **data**, which trains still-better models. That feedback loop is much
of why progress has felt so **fast** — and why the fundamentals in this course sit beneath so many
modern systems.
"""


MINDMAP.update({
    "What is a Neural Network?": [
        "Supervised Learning and Neural Networks",
        "Logistic Regression (Binary Classification Model)",
        "Why Deep Learning is Taking Off", "Computation Graph",
    ],
    "Supervised Learning and Neural Networks": [
        "What is a Neural Network?",
        "Binary Classification and Logistic Regression (Neural Network Basics)",
        "Why Deep Learning is Taking Off",
        "Logistic Regression (Binary Classification Model)",
    ],
    "Why Deep Learning is Taking Off": [
        "What is a Neural Network?", "Supervised Learning and Neural Networks",
        "Geoffrey Hinton Interview", "Vectorizing Logistic Regression",
    ],
})


# ----------------------------------------------------------------------
# Stage 1 — Introduction to Deep Learning (cont.)
# ----------------------------------------------------------------------

CONTENT["Geoffrey Hinton Interview"] = r"""
Heroes of Deep Learning
------------------------

This lesson is a short detour from the mathematics: a look at the ideas from Andrew Ng's
**"Heroes of Deep Learning"** conversation with **Geoffrey Hinton**, one of the researchers most
responsible for the field existing at all. It is history and perspective rather than a technique —
but the history explains *why* the tools in this course look the way they do.

Backpropagation and representations
-------------------------------------

Hinton is best known for the 1986 paper with **Rumelhart and Williams**, *"Learning representations
by back-propagating errors"*, which **popularised backpropagation** for training multi-layer
networks. He is careful that they were **not the first** to the idea — versions were proposed years
earlier — and that the paper's real contribution was showing backprop could learn useful
**internal (distributed) representations**: hidden units that come to stand for meaningful features,
exactly the "learned features" idea from Lesson 1.

Through the winter
--------------------

Backpropagation is just **gradient descent plus the chain rule**, and in the 1990s it hit a wall: in
deep networks the gradients **shrank** as they propagated back through the layers (the
**vanishing-gradient** problem), and interest drifted to other methods. Through that
"neural-network winter" Hinton kept the flame alive with **Boltzmann machines** (with Sejnowski) and
later **restricted Boltzmann machines**. His 2006 work on **deep belief networks** — pre-training a
deep net one layer at a time, then fine-tuning with backprop — is widely credited with sparking the
modern **deep-learning** revival.

Newer directions, and advice
------------------------------

Hinton never stopped pushing past the standard recipe — proposing **dropout**, and **capsule
networks** aimed at capturing part–whole structure in images — and argued that **unsupervised
learning** would ultimately matter more than the supervised setting this course begins with. His
advice to newcomers is quietly encouraging: **trust your intuitions**, read enough but not so much
that you only reproduce others' thinking, **replicate results** to learn them deeply, and keep going
on the problems that feel right.
"""


# ----------------------------------------------------------------------
# Stage 2 — Logistic Regression as a Neuron
# ----------------------------------------------------------------------

CONTENT["Binary Classification and Logistic Regression (Neural Network Basics)"] = r"""
The task
----------

**Binary classification** asks a yes/no question: given an input, output a label
:math:`y \in \{0, 1\}`. The running example is a **cat classifier** — an image goes in, and the model
should output **1** for "cat" and **0** for "not cat". Logistic regression, the subject of this
stage, is the simplest model for this — and, read the right way, a **single neuron**.

One example, as a vector
--------------------------

A model needs numbers, so an image is **unrolled into a feature vector**. A 64×64 colour image has
three channels (red, green, blue), giving :math:`n_x = 64 \times 64 \times 3 = 12{,}288` values
stacked into one column :math:`x \in \mathbb{R}^{n_x}`. A single labelled example is the pair
:math:`(x, y)` with :math:`x \in \mathbb{R}^{n_x}` and :math:`y \in \{0, 1\}`.

Stacking the whole set
------------------------

With :math:`m` training examples :math:`(x^{(1)}, y^{(1)}), \dots, (x^{(m)}, y^{(m)})`, Ng stacks
them into matrices. Each example becomes a **column**, so the data form

.. math::

   X \in \mathbb{R}^{n_x \times m}, \qquad Y \in \mathbb{R}^{1 \times m},

with :math:`X` holding one example per column and :math:`Y` the matching row of labels.

Why columns
-------------

Putting examples in **columns** rather than rows is a deliberate convention: it makes the
**vectorised** forward and backward passes later in this stage line up as clean matrix products, with
no transposes to track. A small choice now, much tidier code from Lesson 15 onward. In ``numpy`` the
shapes are ``X.shape == (n_x, m)`` and ``Y.shape == (1, m)``.
"""

CONTENT["Logistic Regression (Binary Classification Model)"] = r"""
From score to probability
---------------------------

Given an input :math:`x`, logistic regression predicts :math:`\hat{y} = P(y = 1 \mid x)` — the
**probability** the label is 1. A plain linear score :math:`\mathbf{w}^{\!\top}\mathbf{x} + b` can be
any real number, from large negative to large positive, so it **cannot** serve as a probability
directly. It has to be squashed into :math:`[0, 1]`.

The sigmoid
-------------

The squashing function is the **sigmoid** (logistic) function:

.. math::

   \hat{y} = \sigma(\mathbf{w}^{\!\top}\mathbf{x} + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}},

with parameters :math:`\mathbf{w} \in \mathbb{R}^{n_x}` (a weight per feature) and a bias
:math:`b \in \mathbb{R}`.

Reading it
------------

The sigmoid is an **S-curve** between 0 and 1: as :math:`z \to +\infty` it approaches **1**, as
:math:`z \to -\infty` it approaches **0**, and at :math:`z = 0` it is exactly **0.5**. So a large
positive score means "confidently 1", a large negative score "confidently 0", and a score near zero
an undecided, halfway probability.

One neuron
------------

Put together, logistic regression is precisely a **single neuron**: a linear combination of the
inputs followed by a nonlinear activation — here the sigmoid. That is the exact template from
Lesson 1, and stacking many such units is all a neural network is. The next lesson gives this neuron
a **loss**, so it can learn :math:`\mathbf{w}` and :math:`b` from data.
"""


MINDMAP.update({
    "Geoffrey Hinton Interview": [
        "What is a Neural Network?", "Why Deep Learning is Taking Off",
        "Logistic Regression Gradient Descent", "Computation Graph",
    ],
    "Binary Classification and Logistic Regression (Neural Network Basics)": [
        "Logistic Regression (Binary Classification Model)",
        "Logistic Regression \u2013 Loss Function and Cost Function",
        "What is a Neural Network?", "Vectorizing Logistic Regression",
    ],
    "Logistic Regression (Binary Classification Model)": [
        "Binary Classification and Logistic Regression (Neural Network Basics)",
        "Logistic Regression \u2013 Loss Function and Cost Function",
        "What is a Neural Network?", "Gradient Descent in Logistic Regression",
    ],
})


# ----------------------------------------------------------------------
# Stage 2 — Logistic Regression as a Neuron (cont.)
# ----------------------------------------------------------------------

CONTENT["Logistic Regression \u2013 Loss Function and Cost Function"] = r"""
Measuring one prediction
--------------------------

To learn, the neuron needs a **score of how wrong it is**. A **loss function**
:math:`\mathcal{L}(\hat{y}, y)` measures the error on a **single** example — comparing the predicted
probability :math:`\hat{y}` against the true label :math:`y`. Training then adjusts :math:`\mathbf{w}`
and :math:`b` to make that loss small.

Why not squared error
-----------------------

The obvious choice, **squared error** :math:`\tfrac{1}{2}(\hat{y} - y)^2`, works for linear
regression but is a poor fit here. Because :math:`\hat{y}` passes through the **sigmoid**, squared
error makes the cost **non-convex** — a bumpy surface with many local minima where gradient descent
can get **stuck**. We want a loss whose surface is a single smooth bowl.

The cross-entropy loss
------------------------

The right choice is the **cross-entropy** (log) loss:

.. math::

   \mathcal{L}(\hat{y}, y) = -\big(y \log \hat{y} + (1 - y)\log(1 - \hat{y})\big).

Read it by cases. If :math:`y = 1` it reduces to :math:`-\log \hat{y}`, which is small only when
:math:`\hat{y}` is **close to 1**; if :math:`y = 0` it reduces to :math:`-\log(1 - \hat{y})`, small
only when :math:`\hat{y}` is **close to 0**. Confident wrong answers are punished hard — and for
logistic regression this loss is **convex**. (It is exactly the negative log-likelihood of a
Bernoulli label.)

From loss to cost
-------------------

Loss scores **one** example; the **cost function** averages it over **all** :math:`m`:

.. math::

   J(\mathbf{w}, b) = \frac{1}{m}\sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})
     = -\frac{1}{m}\sum_{i=1}^{m}\Big[y^{(i)}\log \hat{y}^{(i)}
       + (1 - y^{(i)})\log(1 - \hat{y}^{(i)})\Big].

**Loss is per-example; cost is the whole-set average.** Training means finding the
:math:`\mathbf{w}, b` that **minimise** :math:`J` — the job of the next lesson.
"""


# ----------------------------------------------------------------------
# Stage 3 — Derivatives & the Computation Graph
# ----------------------------------------------------------------------

CONTENT["Gradient Descent in Logistic Regression"] = r"""
The optimization problem
--------------------------

With a cost :math:`J(\mathbf{w}, b)` in hand, learning becomes a search: **find the**
:math:`\mathbf{w}, b` **that make** :math:`J` **smallest**. The cross-entropy cost for logistic
regression is **convex** — a single bowl-shaped surface with **one global minimum** and no misleading
local dips — which is what makes the search reliable.

Rolling downhill
------------------

**Gradient descent** finds that minimum by repeatedly stepping **downhill**. At the current point the
**gradient** — the partial derivatives :math:`\partial J / \partial \mathbf{w}` and
:math:`\partial J / \partial b` — points in the direction of steepest **increase**; moving the
**opposite** way decreases the cost. Start anywhere (for a convex cost, even all-zeros works) and
repeat.

The update rule
-----------------

Each iteration nudges the parameters against the gradient:

.. math::

   \mathbf{w} := \mathbf{w} - \alpha\,\frac{\partial J}{\partial \mathbf{w}}, \qquad
   b := b - \alpha\,\frac{\partial J}{\partial b}.

In code the derivatives are conventionally named ``dw`` and ``db``, so the step reads
``w -= alpha * dw`` and ``b -= alpha * db``. Repeat until the cost stops decreasing.

The learning rate
-------------------

The step size :math:`\alpha` is the **learning rate**. Too **small** and training crawls, needing
many iterations; too **large** and the steps overshoot the minimum and may **diverge**. Choosing
:math:`\alpha` well — and computing those derivatives efficiently — is what the rest of this stage is
about, starting with the calculus itself.
"""

CONTENT["Derivatives"] = r"""
Derivative means slope
------------------------

Gradient descent runs on **derivatives**, so this lesson pins down what a derivative actually is: a
**slope**. The derivative of :math:`f` at a point measures **how much** :math:`f` **changes when its
input changes by a tiny amount** — the steepness of the function right there.

A worked nudge
----------------

Take the straight line :math:`f(a) = 3a`. At :math:`a = 2`, :math:`f = 6`. Nudge the input a hair, to
:math:`a = 2.001`; then :math:`f = 6.003`. The output moved by :math:`0.003` when the input moved by
:math:`0.001` — **three times as much**. That ratio *is* the derivative: :math:`\frac{df}{da} = 3`.

Slope of a straight line
--------------------------

Geometrically the derivative is **rise over run** — the height of a tiny triangle divided by its
width. For a **straight line** the slope is the **same everywhere**: :math:`f(a) = 3a` has derivative
:math:`3` at every point, whether :math:`a` is 2, 5 or :math:`-10`. (Curved functions have a slope
that *changes* from point to point — the subject of the next lesson.)

Why it matters here
---------------------

Every gradient in this course is built from derivatives like this one. Knowing that the derivative is
just "**how fast the output moves when you wiggle the input**" is enough to follow the **chain rule**
and **backpropagation** ahead — the machinery that computes :math:`\partial J / \partial \mathbf{w}`
and :math:`\partial J / \partial b` for a whole network.
"""


MINDMAP.update({
    "Logistic Regression \u2013 Loss Function and Cost Function": [
        "Logistic Regression (Binary Classification Model)",
        "Gradient Descent in Logistic Regression", "Logistic Regression Gradient Descent",
        "Binary Classification and Logistic Regression (Neural Network Basics)",
    ],
    "Gradient Descent in Logistic Regression": [
        "Logistic Regression \u2013 Loss Function and Cost Function", "Derivatives",
        "Logistic Regression Gradient Descent", "Gradient Descent on m Training Examples",
    ],
    "Derivatives": [
        "More Derivative Examples", "Computation Graph",
        "Gradient Descent in Logistic Regression", "Derivatives with a Computation Graph",
    ],
})


# ----------------------------------------------------------------------
# Stage 3 — Derivatives & the Computation Graph (cont.)  [completes stage]
# ----------------------------------------------------------------------

CONTENT["More Derivative Examples"] = r"""
When the slope changes
------------------------

The line :math:`f(a) = 3a` had the **same slope everywhere**. Most functions do not — their
**derivative changes from point to point**. This lesson makes that concrete, because a neuron's
sigmoid is exactly such a curve.

A curved example
------------------

Take :math:`f(a) = a^2`. At :math:`a = 2`, :math:`f = 4`; nudge to :math:`a = 2.001` and
:math:`f \approx 4.004` — a slope of about **4**. But at :math:`a = 5`, :math:`f = 25`; nudge to
:math:`5.001` and :math:`f \approx 25.010` — a slope of about **10**. The slope is **twice the
input**, which is exactly the rule

.. math::

   f(a) = a^2 \;\Rightarrow\; \frac{df}{da} = 2a.

The derivative is now a **function of** :math:`a`, not a constant.

A few more
------------

The same pattern holds across the standard functions — :math:`f(a) = a^3` has derivative
:math:`3a^2`, and :math:`f(a) = \ln a` has derivative :math:`1/a`. You need not re-derive these from
nudges each time; they are tabulated in any calculus reference. What matters is reading them the same
way: *how fast does the output move as I wiggle the input, right here?*

The takeaway
--------------

For a **curve**, "the derivative" always means the slope **at a particular point**. That single idea
— a slope that varies — is all the calculus the rest of the course needs. Next we organise a
multi-step computation so these per-point slopes can be combined **mechanically**, through a
**computation graph**.
"""

CONTENT["Computation Graph"] = r"""
Breaking it into steps
------------------------

A **computation graph** writes a calculation as a chain of **elementary steps**, each a node feeding
the next. It looks like extra bookkeeping, but it is the structure that makes **backpropagation** —
computing every derivative in one sweep — both possible and efficient.

A small example
-----------------

Ng's example computes :math:`J = 3(a + bc)`. Broken into steps it becomes three nodes:

.. math::

   u = bc, \qquad v = a + u, \qquad J = 3v.

Each intermediate value (:math:`u`, then :math:`v`, then :math:`J`) depends only on ones already
computed — a strict left-to-right flow from inputs :math:`a, b, c` to output :math:`J`.

The forward pass
------------------

Computing the graph **left to right** is the **forward pass**: plug in the inputs and fill in each
node. With :math:`a = 5, b = 3, c = 2`:

.. math::

   u = bc = 6, \qquad v = a + u = 11, \qquad J = 3v = 33.

This is exactly the forward propagation that produces a neuron's prediction — inputs in, output out.

Why bother
------------

The payoff comes next. Because the graph records **how each value was built from the previous ones**,
we can walk it **backward** and apply the chain rule step by step, computing
:math:`\partial J / \partial a`, :math:`\partial J / \partial b`, :math:`\partial J / \partial c`
without ever untangling the whole nested expression at once.
"""

CONTENT["Derivatives with a Computation Graph"] = r"""
Walking backward
------------------

With the forward pass done, the **backward pass** computes derivatives by moving **right to left**
through the same graph, applying the **chain rule** at each node. This right-to-left sweep is exactly
what **backpropagation** does.

One step at a time
--------------------

Start at the output. Since :math:`J = 3v`, nudging :math:`v` moves :math:`J` three times as much, so
:math:`\partial J / \partial v = 3`. Step back through :math:`v = a + u`:

.. math::

   \frac{\partial J}{\partial a} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial a}
     = 3 \cdot 1 = 3, \qquad
   \frac{\partial J}{\partial u} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial u}
     = 3 \cdot 1 = 3.

Then back through :math:`u = bc`:

.. math::

   \frac{\partial J}{\partial b} = \frac{\partial J}{\partial u}\frac{\partial u}{\partial b}
     = 3c = 6, \qquad
   \frac{\partial J}{\partial c} = \frac{\partial J}{\partial u}\frac{\partial u}{\partial c}
     = 3b = 9,

using :math:`a = 5, b = 3, c = 2`.

The dvar convention
---------------------

A coding shorthand runs through the whole course: the variable named **``dvar``** always means "the
derivative of the **final output** :math:`J` with respect to :math:`\text{var}`". So ``dv`` is
:math:`\partial J / \partial v`, ``da`` is :math:`\partial J / \partial a`, ``du`` is
:math:`\partial J / \partial u`, and so on — every ``d``-something is a gradient of the same target,
which keeps the code readable.

Reuse is the point
--------------------

Notice that :math:`\partial J / \partial v` was computed **once** and then **reused** for both
:math:`\partial J / \partial a` and :math:`\partial J / \partial u`, and :math:`\partial J / \partial u`
was reused for :math:`b` and :math:`c`. That reuse is why the backward pass is **efficient**: each
node's derivative follows from the one just downstream, so the whole gradient costs about as much as
the forward pass. Applying this to the logistic-regression neuron is the next lesson.
"""


MINDMAP.update({
    "More Derivative Examples": [
        "Derivatives", "Computation Graph", "Derivatives with a Computation Graph",
        "Logistic Regression Gradient Descent",
    ],
    "Computation Graph": [
        "Derivatives with a Computation Graph", "Derivatives",
        "Logistic Regression Gradient Descent", "Gradient Descent in Logistic Regression",
    ],
    "Derivatives with a Computation Graph": [
        "Computation Graph", "More Derivative Examples",
        "Logistic Regression Gradient Descent", "Gradient Descent on m Training Examples",
    ],
})


# ----------------------------------------------------------------------
# Stage 4 — Backprop & Vectorization
# ----------------------------------------------------------------------

CONTENT["Logistic Regression Gradient Descent"] = r"""
The neuron's graph
--------------------

The computation-graph machinery now applies to the neuron itself. For a two-feature example the
forward graph is

.. math::

   z = w_1 x_1 + w_2 x_2 + b, \qquad a = \sigma(z) = \hat{y}, \qquad
   \mathcal{L}(a, y) = -\big(y \log a + (1 - y)\log(1 - a)\big).

Gradient descent needs :math:`\partial \mathcal{L}/\partial w_1`, :math:`\partial \mathcal{L}/\partial w_2`
and :math:`\partial \mathcal{L}/\partial b` — found by walking this graph backward.

Backprop, step by step
------------------------

Start at the loss and step back. The derivative of the loss with respect to the activation is

.. math::

   \mathrm{d}a = \frac{\partial \mathcal{L}}{\partial a} = -\frac{y}{a} + \frac{1 - y}{1 - a}.

The sigmoid contributes :math:`\partial a / \partial z = a(1 - a)`, so by the chain rule
:math:`\mathrm{d}z = \mathrm{d}a \cdot a(1 - a)`.

The clean result
------------------

Those two pieces multiply out to something remarkably simple:

.. math::

   \mathrm{d}z = \frac{\partial \mathcal{L}}{\partial z} = a - y.

The gradient of the loss with respect to the pre-activation is just **prediction minus truth**. From
there the parameter gradients fall out immediately:

.. math::

   \mathrm{d}w_1 = x_1\,\mathrm{d}z, \qquad \mathrm{d}w_2 = x_2\,\mathrm{d}z, \qquad
   \mathrm{d}b = \mathrm{d}z.

Then the update
-----------------

With the gradients in hand, one gradient-descent step nudges each parameter downhill:

.. math::

   w_1 := w_1 - \alpha\,\mathrm{d}w_1, \qquad w_2 := w_2 - \alpha\,\mathrm{d}w_2, \qquad
   b := b - \alpha\,\mathrm{d}b.

That is a full learning step — for a **single** example. Real training averages over many, which is
the next lesson.
"""

CONTENT["Gradient Descent on m Training Examples"] = r"""
Averaging the gradient
------------------------

A single example gives a noisy gradient; training uses the **whole set**. The cost is the average
loss, so — because differentiation is linear — **its gradient is the average of the per-example
gradients**:

.. math::

   J(\mathbf{w}, b) = \frac{1}{m}\sum_{i=1}^{m} \mathcal{L}(a^{(i)}, y^{(i)}), \qquad
   \frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m} x_j^{(i)}\,\mathrm{d}z^{(i)}, \qquad
   \frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m} \mathrm{d}z^{(i)}.

The explicit algorithm
------------------------

Written out naively, one gradient-descent step **accumulates** over the examples, then divides by
:math:`m`:

.. code-block:: python

   J = 0; dw1 = 0; dw2 = 0; db = 0
   for i in range(m):                     # loop over the m examples
       z = w1 * x1[i] + w2 * x2[i] + b
       a = sigmoid(z)
       J += -(y[i] * log(a) + (1 - y[i]) * log(1 - a))
       dz = a - y[i]
       dw1 += x1[i] * dz                  # one line per feature ...
       dw2 += x2[i] * dz
       db += dz
   J /= m; dw1 /= m; dw2 /= m; db /= m
   w1 -= alpha * dw1; w2 -= alpha * dw2; b -= alpha * db

Two loops, and slow
---------------------

Look closely and there are **two** nested loops: the visible one over the :math:`m` examples, and a
hidden one over the :math:`n` **features** — with :math:`n` weights you would need
:math:`\mathrm{d}w_1, \mathrm{d}w_2, \dots, \mathrm{d}w_n`. Explicit Python loops like these are
**badly suited to parallel hardware** and crawl on large datasets. Removing them — **vectorization** —
is the subject of the next lesson.
"""

CONTENT["Vectorization in Logistic Regression"] = r"""
What vectorization is
-----------------------

**Vectorization** is, in Andrew Ng's words, "the art of getting rid of explicit for-loops in your
code". Instead of looping element by element, you express the computation as **whole-array
operations** — matrix and vector products — and let an optimised numerical library do the looping in
fast compiled code.

The forward pass, vectorized
------------------------------

Recall the per-example forward step :math:`z^{(i)} = \mathbf{w}^{\!\top} x^{(i)} + b`. Stacking all
:math:`m` examples as **columns** of :math:`X` (the convention from Lesson 5) collapses the whole
loop into one line:

.. math::

   Z = \mathbf{w}^{\!\top} X + b, \qquad A = \sigma(Z),

where :math:`Z` and :math:`A` are :math:`1 \times m`. In ``numpy`` this is literally
``Z = np.dot(w.T, X) + b`` followed by ``A = sigmoid(Z)`` — the scalar ``b`` is **broadcast** across
all columns automatically.

Why it's faster
-----------------

``np.dot`` hands the arithmetic to routines that use the CPU's (or GPU's) **parallel** vector
instructions, processing many numbers at once. Ng's demonstration times the same dot product both
ways, and the vectorised version runs **hundreds of times faster** than the Python loop. On large
datasets that is the difference between minutes and hours.

The habit
-----------

The rule to carry forward is Ng's programming guideline: **"whenever possible, avoid explicit
for-loops."** The remaining lessons apply it relentlessly — vectorising the **gradients** too, until
an entire step of logistic regression over all :math:`m` examples runs without a single Python loop
over the data.
"""


MINDMAP.update({
    "Logistic Regression Gradient Descent": [
        "Derivatives with a Computation Graph",
        "Logistic Regression \u2013 Loss Function and Cost Function",
        "Gradient Descent on m Training Examples", "Computation Graph",
    ],
    "Gradient Descent on m Training Examples": [
        "Logistic Regression Gradient Descent", "Vectorization in Logistic Regression",
        "Gradient Descent in Logistic Regression", "Vectorizing Logistic Regression",
    ],
    "Vectorization in Logistic Regression": [
        "Gradient Descent on m Training Examples", "More Vectorization Examples",
        "Vectorizing Logistic Regression",
        "Binary Classification and Logistic Regression (Neural Network Basics)",
    ],
})


# ----------------------------------------------------------------------
# Stage 4 — Backprop & Vectorization (cont.)  [completes the course]
# ----------------------------------------------------------------------

CONTENT["More Vectorization Examples"] = r"""
Element-wise functions
------------------------

Loops are not only for sums. Any operation applied **to every element** of an array has a vectorised
form. Instead of writing a loop to exponentiate each entry of a vector :math:`v`, call ``np.exp(v)``;
the same holds for ``np.log``, ``np.abs``, ``np.maximum(0, v)`` (a ReLU), powers like ``v ** 2``, and
reciprocals ``1 / v``. Each runs the loop internally in compiled code, over the whole array at once.

Broadcasting
--------------

Vectorised code often combines arrays of **different shapes**, and ``numpy`` reconciles them by
**broadcasting**: the smaller array is **stretched** to match the larger. Add a scalar to a vector and
the scalar is applied to every element; add a :math:`1 \times n` row to an :math:`m \times n` matrix
and the row is **copied down** all :math:`m` rows. It is the mechanism that let the bias ``b`` add
cleanly across every column in the last lesson.

A worked example
------------------

Ng's example computes each food's macronutrient split as a percentage of its calories. With a matrix
:math:`A` of nutrient values, the totals and percentages take **two lines, no loop**:

.. code-block:: python

   cal = A.sum(axis=0)                 # column sums -> total calories per food
   percentage = 100 * A / cal.reshape(1, 4)

The division **broadcasts** the :math:`1 \times 4` totals across every row of :math:`A`. (Here
``axis=0`` sums down columns; ``axis=1`` would sum across rows.)

The guideline
---------------

All of this serves one rule — Ng's programming guideline, **"whenever possible, avoid explicit
for-loops."** Reach first for a whole-array operation or a broadcast; fall back to a Python loop only
when no vectorised form exists. With these tools the entire logistic-regression step vectorises, which
the final lesson assembles.
"""

CONTENT["Vectorizing Logistic Regression"] = r"""
The whole step, no loops
--------------------------

Everything in this stage now combines into a **single** gradient-descent iteration with **no Python
loop over the data** — not over the :math:`m` examples, not over the :math:`n` features. Both the
forward pass and the gradient computation become a handful of matrix operations on
:math:`X \in \mathbb{R}^{n_x \times m}` and :math:`Y \in \mathbb{R}^{1 \times m}`.

Forward, then backward
------------------------

The forward pass predicts all examples at once; the backward pass forms all gradients at once:

.. code-block:: python

   Z = np.dot(w.T, X) + b           # (1, m)   all pre-activations
   A = sigmoid(Z)                   # (1, m)   all predictions
   dZ = A - Y                       # (1, m)   the clean "prediction - truth"
   dw = (1 / m) * np.dot(X, dZ.T)   # (n_x, 1) averaged weight gradient
   db = (1 / m) * np.sum(dZ)        # scalar   averaged bias gradient
   w = w - alpha * dw
   b = b - alpha * db

The vectorised :math:`\mathrm{d}Z = A - Y` carries the per-example result :math:`\mathrm{d}z = a - y`
across the whole set, and ``np.dot(X, dZ.T)`` sums :math:`x^{(i)}\,\mathrm{d}z^{(i)}` over all examples
in one product.

One loop remains
------------------

This is **one** step of gradient descent. To actually **train**, you repeat it — and that outer loop
over **iterations** (epochs) is the **one loop you cannot vectorise away**, because each step depends
on the parameters the previous step produced. Everything *inside* the step, though, is loop-free.

The gateway to deep networks
------------------------------

That compact block is the whole of logistic regression — a single sigmoid neuron, trained by
vectorised gradient descent. A deep network is the **same pattern repeated**: stack more layers, run
the forward pass and backpropagation through each, and reuse exactly these ideas — the sigmoid (or
ReLU) activation, the cross-entropy cost, the computation graph, and vectorisation. With this stage
complete, you have built every piece a neural network is made of.
"""


MINDMAP.update({
    "More Vectorization Examples": [
        "Vectorization in Logistic Regression", "Vectorizing Logistic Regression",
        "Gradient Descent on m Training Examples", "Logistic Regression Gradient Descent",
    ],
    "Vectorizing Logistic Regression": [
        "Vectorization in Logistic Regression", "More Vectorization Examples",
        "Gradient Descent on m Training Examples", "What is a Neural Network?",
    ],
})
