"""
Content corpus for the Data Analytics hub.

Filled incrementally by content batches. Three dicts, all keyed by the EXACT
lesson title as it appears in ``da_inventory.tsv``:

- ``CONTENT[title]`` : the full RST body of the lesson (raw string).
- ``MINDMAP[title]`` : list of 4 cross-link titles (each an exact inventory
  title, possibly in another section).
- ``GLOSS[title]``   : a one-line description shown in the browser index lines.
                       Optional; defaults to "" when absent.

A title with no CONTENT entry renders as a stub page ("Lesson in progress").
The generator fail-fast-validates that every CONTENT/MINDMAP/GLOSS key and every
MINDMAP neighbour is an exact inventory title.
"""

CONTENT: dict[str, str] = {}
MINDMAP: dict[str, list[str]] = {}
GLOSS: dict[str, str] = {}


# ======================================================================
# Section 1 — Foundations / Stage: why  (lessons 001-004)
# ======================================================================

GLOSS.update({
    "Why Data Analytics Matters Today":
        "why organisations that decide with data outperform those that decide by instinct",
    "How Data Analytics Improves the Workplace":
        "where analytics pays off day to day: operations, decisions, and shared facts",
    "Data-Driven Decision-Making":
        "the loop from question to data to decision, and the evidence it works",
    "Detectives and Data Analysts":
        "the investigator's mindset: questions, evidence, and conclusions that hold up",
})

CONTENT["Why Data Analytics Matters Today"] = r"""
Data is everywhere; answers are not
-------------------------------------

Every organisation now produces data as a by-product of simply operating — sales
records, website clicks, sensor readings, support tickets, survey responses. What
is scarce is not data but the ability to turn it into **answers**: decisions that
are better because someone looked at the evidence. Data analytics is that ability,
and this course teaches it end to end, from asking the right question to
presenting the result.

What data analytics is
------------------------

**Data analytics** is the collection, transformation, and organisation of data in
order to draw conclusions, make predictions, and drive informed decision-making.
The definition has three working parts, and each maps to a later section of this
course:

- **Collection and preparation** — finding the right data and making it usable
  (Sections 3 and 4).
- **Analysis** — organising, aggregating, and computing on it to surface the
  answer (Sections 5 and 7).
- **Communication** — turning the answer into a decision through visuals and
  presentation (Section 6).

An analyst is the person who carries a question through that whole pipeline.

The evidence it matters
-------------------------

The claim that data beats gut feel is itself an empirical question, and it has
been studied. A well-known line of research by Brynjolfsson and colleagues
compared large firms that adopted **data-driven decision-making** with otherwise
similar firms and found the adopters showed roughly **5–6% higher output and
productivity** than their other investments would predict, with matching gains in
asset utilisation, return on equity, and market value. Follow-up work in US
manufacturing found the productivity benefit robust and, by the study's causal
tests, not merely a correlation: firms did not adopt data practices *because*
they were already better — the practices themselves paid.

The demand for analysts
-------------------------

That payoff is why the job exists. Organisations across every industry — retail,
healthcare, finance, logistics, entertainment, government — collect far more data
than they can interpret, and they hire people who can close that gap. The skills
this course builds (spreadsheets, SQL, visualization, Python, and above all the
analytical habit of mind) are the working toolkit of that role.

An honest caveat
------------------

Data is an input to judgement, not a replacement for it. Numbers can be wrong,
biased, or beside the point, and later lessons deal squarely with dirty data,
sampling bias, and misleading charts. "Data-driven" done well means *evidence
disciplines the decision* — not that a spreadsheet makes it for you.
"""

CONTENT["How Data Analytics Improves the Workplace"] = r"""
From reports to better everyday work
--------------------------------------

Analytics is often pictured as quarterly reports for executives. Its real effect
in a workplace is more ordinary and more constant: it changes **how everyday
decisions get made**, at every level, by replacing "I think" with "the data
shows" often enough that the whole organisation steers better.

Where the improvement shows up
--------------------------------

Four recurring areas, each a pattern you will see across industries:

- **Smarter operations.** Tracking the right numbers — production targets,
  costs, quality rates, delivery times — reveals where a process leaks time or
  money. What gets measured can be fixed; what is invisible cannot.
- **Better decisions under uncertainty.** Should we stock more of product A or
  B? Which marketing channel earns its budget? Data turns these from debates
  into comparisons.
- **A shared source of truth.** When teams argue from the same dashboard rather
  than competing anecdotes, disagreements become questions ("why did region 3
  dip in May?") instead of stalemates.
- **Earlier warnings.** Trends surface in data before they are obvious on the
  ground — rising churn, a slipping quality metric, a seasonal shift — giving
  time to respond.

A concrete miniature
----------------------

The pattern in its smallest form — a team deciding which support issues to fix
first, from a ticket log rather than from whoever complains loudest:

.. code-block:: sql

   SELECT issue_category,
          COUNT(*)              AS tickets,
          AVG(hours_to_resolve) AS avg_hours
   FROM   support_tickets
   WHERE  opened_date >= '2024-01-01'
   GROUP  BY issue_category
   ORDER  BY tickets DESC;

One query, and the debate about "what our customers struggle with" has a factual
answer to start from. Most workplace analytics is exactly this shape, scaled up.

Why the culture matters as much as the tools
----------------------------------------------

Research on data-driven firms keeps finding the same precondition: the gains
arrive when leadership is genuinely willing to **put data ahead of instinct and
politics** — to let evidence overrule the highest-paid opinion in the room.
Buying dashboards without that willingness produces decoration, not improvement.
The complement runs the other way too: data practices pay most where the
supporting IT and the habit of reviewing key indicators already exist.

The caveat
------------

Metrics can be gamed, and a workplace that measures everything can drown in
numbers that matter little. Part of the analyst's job — developed throughout
this course — is choosing the *few* measures that genuinely track the goal, and
being honest when the data cannot answer the question being asked.
"""

CONTENT["Data-Driven Decision-Making"] = r"""
The loop, not the buzzword
----------------------------

**Data-driven decision-making** (DDDM) is the practice of using facts derived
from data — rather than intuition, habit, or hierarchy alone — to guide business
decisions. Stripped of buzz, it is a repeatable **loop**:

1. **Ask** — state the decision as a question the data could answer.
2. **Gather** — find or collect the relevant data.
3. **Analyse** — organise and compute until the evidence is visible.
4. **Decide and act** — make the call, informed by what you found.
5. **Measure** — check the outcome, which becomes data for the next loop.

Every section of this course serves one or more steps of that loop; the six-phase
process in the next stage is its formal version.

Intuition versus evidence
---------------------------

The contrast case is decision by **gut feel** — experience-based instinct.
Instinct is fast and sometimes right, but it is also where every cognitive bias
lives: the vivid recent anecdote outweighs the quiet trend; the option someone is
invested in gets the benefit of the doubt. Data-driven does not mean discarding
experience — domain knowledge is what makes an analysis sensible — it means
**making instinct show its evidence**. The strongest decisions use both: experience
to frame the question and sanity-check the answer, data to settle what is
actually happening.

Does it work? The evidence
----------------------------

Yes, measurably. The research introduced in the first lesson quantifies it:
large firms practising DDDM showed about **5–6% higher productivity** than
comparable firms, after accounting for their technology and other investments,
with the gains confirmed across profitability and market-value measures and
supported by causal tests rather than mere correlation. Later work found the
advantage was strongest for **early adopters**, and that as basic data use became
universal, the frontier moved on to predictive analytics — a reminder that the
edge comes from using data *better* than the alternative, not from merely having
it.

A small worked contrast
-------------------------

An online shop must choose which of two homepage designs to keep.

- *Gut version:* the team debates which looks better; the loudest voice wins.
- *Data version:* both designs run for two weeks as an A/B test; design B
  converts 3.1% of visitors against A's 2.6%; B ships.

The data version is not smarter people — it is a **process** that lets a
measurable difference, rather than persuasion, decide.

The caveat
------------

DDDM is only as good as the data and the question. If the data is biased, the
decision inherits the bias; if the question is wrong, a precise answer to it is
still useless. And some genuinely important factors — morale, trust, long-term
brand — are hard to measure, which makes them easy to ignore in a numbers-only
culture. The best practitioners treat data as the strongest voice at the table,
not the only one.
"""

CONTENT["Detectives and Data Analysts"] = r"""
Two jobs, one method
----------------------

A detective and a data analyst do strikingly similar work. Both start with a
question (*who did it? why are sales falling?*), gather evidence, test
explanations against that evidence, discard the ones that fail, and present a
conclusion that has to **hold up under scrutiny** — to a court in one case, to
stakeholders in the other. The analogy is worth taking seriously because it
captures the *mindset* this course trains, before any tool.

What transfers
----------------

- **Questions first.** A detective does not collect fingerprints at random; the
  investigation is shaped by what needs answering. Likewise, analysis begins
  with a sharp question — the subject of a whole later stage — because data
  gathered without one is just clutter.
- **Evidence over assumption.** Detectives distrust the obvious suspect;
  analysts distrust the obvious explanation. Both ask: *what does the evidence
  actually show?* — and let it overrule a comfortable story.
- **Multiple hypotheses.** Good investigators hold several explanations at once
  and look for the evidence that separates them. "Sales fell because of price"
  and "sales fell because a competitor launched" predict different patterns;
  the data can say which.
- **The chain of custody.** A conclusion is only as credible as the trail behind
  it. Documenting where data came from and what was done to it — a major theme
  of the cleaning section — is the analyst's chain of custody.
- **Presenting the case.** Neither job ends at the private "aha". The finding
  must be assembled into a case that a non-specialist can follow and believe,
  which is what the visualization and presentation section teaches.

Where the analogy bends
-------------------------

One difference matters. A detective usually seeks a single past fact — who did
it. An analyst often characterises an **ongoing pattern** (what drives churn,
which segment is growing) where there is no one culprit and the answer is a
distribution, a trend, or a trade-off. Analytical conclusions are therefore
usually **probabilistic** — "strong evidence that", not "proof beyond doubt" —
and stating that uncertainty honestly is part of the job, not a weakness in it.

The takeaway
--------------

Tools change; the investigator's discipline does not. Ask a precise question,
gather evidence deliberately, test explanations rather than defend them, keep
the trail, and present a case that survives cross-examination. Every technique
in the coming sections — spreadsheets, SQL, cleaning, charts, Python — is in
service of that discipline.
"""


MINDMAP.update({
    "Why Data Analytics Matters Today": [
        "How Data Analytics Improves the Workplace", "Data-Driven Decision-Making",
        "The Six Phases of the Data Analysis Process", "Overview of Core Tools Used by Data Analysts",
    ],
    "How Data Analytics Improves the Workplace": [
        "Why Data Analytics Matters Today", "Data-Driven Decision-Making",
        "Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making",
        "The Relationship Between Data and Decision-Making",
    ],
    "Data-Driven Decision-Making": [
        "Why Data Analytics Matters Today", "Detectives and Data Analysts",
        "Data-Driven Decision-Making and the Role of Analytical Skills",
        "Quantitative and Qualitative Data in Decision-Making",
    ],
    "Detectives and Data Analysts": [
        "Data-Driven Decision-Making", "Analytical Skills and Their Core Components",
        "Analytical Thinking and Questions for Problem Solving",
        "Why Asking the Right Questions Matters in Data Analytics",
    ],
})


# ======================================================================
# Section 1 — Foundations / Stage: process  (lessons 005-008)
# ======================================================================

GLOSS.update({
    "The Six Phases of the Data Analysis Process":
        "ask, prepare, process, analyze, share, act — the map for every project in this course",
    "The Origins of Data Analysis and the Many Ways to Structure It":
        "from early statistics to EDA and CRISP-DM: one process, many framings",
    "Understanding the Data Ecosystem":
        "the interlocking pieces — sources, storage, tools, people — data moves through",
    "Understanding the Data Analysis Process and the Data Life Cycle":
        "two different journeys: what the analyst does vs. what happens to the data",
})

CONTENT["The Six Phases of the Data Analysis Process"] = r"""
A map for every project
-------------------------

Every analysis in this course — and most you will do professionally — follows the
same six-phase path: **Ask, Prepare, Process, Analyze, Share, Act**. The phases
turn a vague request ("figure out why sales dipped") into a sequence of concrete,
manageable steps, which is the essence of **structured thinking**.

The six phases
----------------

1. **Ask.** Define the problem and the question. Who are the stakeholders, what
   decision hinges on the answer, and what would a useful answer look like? You
   cannot solve a problem you have not stated.
2. **Prepare.** Decide what data can answer the question, then find or collect
   it — identifying sources, checking credibility, and organising it for use.
3. **Process.** Clean the data: remove duplicates and errors, handle missing
   values, fix inconsistencies, and document every change. Clean data is the
   foundation everything after stands on.
4. **Analyze.** Organise, aggregate, and compute until the pattern that answers
   the question is visible — the sorting, formatting, pivoting, and querying of
   the analysis section.
5. **Share.** Communicate the finding to the people who asked, with visuals and
   narrative matched to the audience.
6. **Act.** Put the insight to work: recommendations, decisions, changes — and
   the feedback from acting becomes input for the next Ask.

A worked thread
-----------------

A bike-share company wants more annual members. *Ask:* how do casual riders and
members use the service differently? *Prepare:* twelve months of trip records.
*Process:* remove test rides and corrupted rows, standardise timestamps.
*Analyze:* compare ride length and day-of-week patterns by rider type. *Share:*
a short deck showing casual riders concentrate on long weekend rides. *Act:*
marketing targets weekend riders with a membership offer. One question, six
phases, a decision at the end — this exact case structure recurs in published
walkthroughs of the framework.

Not a straight line
---------------------

The phases are a map, not a railway. Real projects loop: analysis exposes a data
problem that sends you back to Process; sharing raises a follow-up that restarts
Ask. The map's value is knowing **where you are** and what the current phase owes
the next one — not forbidding movement between them.

This course, in phase order
-----------------------------

The course sections mirror the phases: asking and deciding (Section 2), preparing
data (Section 3), processing and cleaning (Section 4), analysing (Sections 5 and
7), and sharing (Section 6). Keep the six-phase map in mind and every technique
that follows has an obvious home.
"""

CONTENT["The Origins of Data Analysis and the Many Ways to Structure It"] = r"""
An old craft with new names
-----------------------------

Analysing data to answer questions long predates the job title. States counted
people and harvests for millennia; the statistics of the 18th and 19th centuries
formalised inference from samples; the 20th century added machine computation.
Two more recent turns shaped the modern craft. In 1977 John Tukey's
*Exploratory Data Analysis* argued that analysts should **look at data first** —
plot it, summarise it, let it suggest hypotheses — rather than jumping straight
to confirming a preconceived model. And from the 1990s, industry codified the
workflow itself: **CRISP-DM** (the Cross-Industry Standard Process for Data
Mining, published in 1999) became the most widely used formal process model for
analytics projects.

Many framings, one process
----------------------------

Different communities carve the same journey differently:

- **Ask–Prepare–Process–Analyze–Share–Act** — the six-phase framing this course
  uses, oriented around the analyst's tasks.
- **CRISP-DM** — Business Understanding, Data Understanding, Data Preparation,
  Modeling, Evaluation, Deployment; explicitly **cyclical**, with movement back
  and forth between phases expected rather than exceptional.
- Compressed or expanded variants — some teams merge exploration and modelling
  into five steps; others split deployment into more.

Lay them side by side and the correspondence is plain: *Ask* is business
understanding; *Prepare* and *Process* are data understanding and preparation;
*Analyze* spans exploration and modelling; *Share* is evaluation and
communication; *Act* is deployment. Learning one framing well makes you fluent
in all of them — the vocabulary changes, the discipline does not.

Why structure at all
----------------------

The recurring failure of unstructured analysis is starting in the middle:
grabbing data and computing before the question is clear, then discovering the
data cannot answer what was actually needed. Every framework above exists to
prevent that — each front-loads *understanding the problem* and treats cleaning
as a first-class phase, because those are the steps that unstructured work
skips. Structure is also what makes work **repeatable and reviewable**: a
colleague can pick up a CRISP-DM project and know where things stand.

The caveat
------------

Frameworks describe; they do not think. Following the phases mechanically, with
a weak question or credulous data, produces well-organised nonsense. Treat the
structure as scaffolding for judgement — the judgement itself is what the rest
of this course builds.
"""

CONTENT["Understanding the Data Ecosystem"] = r"""
Data does not live alone
--------------------------

A **data ecosystem** is the full set of interacting elements that produce,
move, store, and consume an organisation's data: the devices and processes that
generate it, the databases and cloud platforms that hold it, the tools that
transform and analyse it, and — easy to forget — the **people** whose decisions
it feeds. The ecology metaphor is apt: the parts depend on each other, and a
weakness anywhere (an unreliable source, a stale warehouse, an unmaintained
dashboard) degrades everything downstream.

The parts, in flow order
--------------------------

- **Sources.** Where data is born: transactions, web and app events, sensors,
  surveys, third-party feeds, public datasets.
- **Storage.** Where it lives: operational databases, data warehouses, cloud
  storage — organised so it can be found and queried.
- **Processing and movement.** The pipelines that collect, clean, and reshape
  data between storage and use.
- **Analysis tools.** Spreadsheets, SQL engines, BI platforms, and languages
  like Python and R — the layer this course concentrates on.
- **Consumers.** Dashboards, reports, applications, and ultimately the
  stakeholders acting on what the data says.

An analyst's query is one hop in a longer journey; knowing the whole route tells
you where an odd number might have gone wrong.

Ecosystems differ by industry
-------------------------------

A hospital's ecosystem centres on patient records under strict privacy rules; a
retailer's on transactions and inventory; a farm's on sensors, weather feeds,
and yield data. The components rhyme, but the sources, constraints, and
consumers differ — which is why the same analyst skills transfer across
industries while the domain knowledge must be relearned.

Neighbouring terms, kept straight
-----------------------------------

Three commonly confused labels sit inside the ecosystem. **Data analysis** is
the discipline this course teaches: drawing conclusions from data to inform
decisions. **Data science** overlaps but leans toward building predictive
models and new methods. **Data engineering** builds and maintains the pipelines
and storage the other two rely on. Titles blur in practice — small teams wear
all the hats — but knowing the distinction helps you read job descriptions and
know who to ask when the pipeline, not the analysis, is broken.

The caveat
------------

Ecosystems accrete: real organisations run overlapping tools, half-migrated
warehouses, and undocumented spreadsheets that turn out to be load-bearing. Part
of an analyst's practical skill is mapping the ecosystem *as it actually is* —
where the trustworthy source of a number lives — rather than as the architecture
diagram claims.
"""

CONTENT["Understanding the Data Analysis Process and the Data Life Cycle"] = r"""
Two journeys, often confused
------------------------------

Two sequences run through every data project, and they are not the same. The
**data analysis process** (Ask, Prepare, Process, Analyze, Share, Act) describes
what the **analyst** does to answer a question. The **data life cycle** describes
what happens to the **data itself**, from the moment someone decides to collect
it to the day it is deleted. One follows the worker; the other follows the
material. Keeping them straight prevents a common muddle — and interviewers like
asking about exactly this distinction.

The data life cycle
---------------------

A widely taught version has six stages:

1. **Plan.** Before any collection: decide what data is needed, how it will be
   managed, who is responsible for it, and under what rules.
2. **Capture.** Bring the data into existence or into the organisation —
   collecting from sources, sensors, forms, or external providers.
3. **Manage.** Store, secure, organise, and maintain it so it stays usable:
   where it lives, how it is backed up, who may access it.
4. **Analyze.** Use it — the stage where the entire analysis *process* happens.
5. **Archive.** Move data no longer in active use into long-term storage, still
   retrievable if needed.
6. **Destroy.** Delete it — securely and deliberately — when retention rules or
   privacy obligations say its time is up.

The exact stages and names vary by company and industry; regulated sectors add
compliance checkpoints. The shape, though — from planned birth to deliberate
death — is universal.

How the two interlock
-----------------------

The whole six-phase analysis process lives **inside** one stage of the life
cycle: *Analyze*. Conversely, the analyst constantly depends on the other
stages. Good **planning** upstream determines whether the data you need even
exists; good **management** determines whether you can find and trust it;
**archive** and **destroy** determine whether last year's comparison data is
still there — or legally must not be. When a lesson later in this course says
"check where the data came from," it is sending you back up the life cycle.

Why analysts should care about the whole cycle
------------------------------------------------

Because the biggest analysis problems are usually born outside the analysis. A
question that cannot be answered often traces to a Plan stage that never
anticipated it; dirty data traces to Capture; a missing year traces to Destroy.
Analysts who understand the life cycle diagnose these quickly — and, when
consulted early, help design collection so the *next* question is answerable.

The caveat
------------

Life-cycle diagrams look tidier than reality: data gets copied, forked into
spreadsheets, and half-archived, so the same record can sit at several stages at
once. Treat the cycle as the intended governance path, and expect to do some
detective work about where a given dataset really is on it.
"""


MINDMAP.update({
    "The Six Phases of the Data Analysis Process": [
        "Why Data Analytics Matters Today",
        "Understanding the Data Analysis Process and the Data Life Cycle",
        "The Stages of the Data Analysis Process and Their Roles",
        "Practical Application of the Data Analysis Process",
    ],
    "The Origins of Data Analysis and the Many Ways to Structure It": [
        "The Six Phases of the Data Analysis Process",
        "Understanding the Data Ecosystem",
        "Data-Driven Decision-Making",
        "The Stages of the Data Analysis Process and Their Roles",
    ],
    "Understanding the Data Ecosystem": [
        "The Origins of Data Analysis and the Many Ways to Structure It",
        "Understanding the Data Life Cycle",
        "Overview of Core Tools Used by Data Analysts",
        "How Data Is Generated and Collected",
    ],
    "Understanding the Data Analysis Process and the Data Life Cycle": [
        "The Six Phases of the Data Analysis Process",
        "Understanding the Data Life Cycle",
        "A Review of the Six Stages of the Data Life Cycle",
        "Why Asking the Right Questions Matters in Data Analytics",
    ],
})


# ======================================================================
# Section 1 — Foundations / Stage: process (cont.)  (lessons 009-012)
# ======================================================================

GLOSS.update({
    "Understanding the Data Life Cycle":
        "plan, capture, manage, analyze, archive, destroy — the data's own biography",
    "A Review of the Six Stages of the Data Life Cycle":
        "the six stages consolidated, with what can go wrong at each",
    "The Stages of the Data Analysis Process and Their Roles":
        "what each analysis phase contributes, and what it hands to the next",
    "Practical Application of the Data Analysis Process":
        "the six phases run end-to-end on a real-shaped business case",
})

CONTENT["Understanding the Data Life Cycle"] = r"""
The data's own biography
--------------------------

The previous lesson separated the analyst's process from the **data life
cycle** — the journey the data itself travels. This lesson walks that journey
stage by stage, because an analyst who knows where data comes from and where it
is going works faster and trusts the right things.

The six stages, in depth
--------------------------

1. **Plan.** The stage most people never see, and the one that decides
   everything after. Before collection, someone chooses *what* data is needed,
   *how* it will be managed, *who* is responsible, and under what privacy and
   retention rules. A well-planned dataset arrives with definitions and owners;
   a badly planned one arrives as a mystery.
2. **Capture.** The data comes into existence or into the organisation:
   recorded by transactions and sensors, typed into forms, imported from
   external providers or public sources. Capture choices — what fields, what
   granularity, what validation at entry — set the ceiling on later quality.
3. **Manage.** The custodial stage: storing the data, securing it, organising
   it so it can be found, backing it up, and controlling who may access it.
   Most of the ecosystem lesson's storage layer lives here.
4. **Analyze.** The data is put to work answering questions — the stage where
   this whole course happens.
5. **Archive.** Data no longer in active use moves to long-term storage: out of
   the way, cheaper to keep, but retrievable when an audit or a historical
   comparison needs it.
6. **Destroy.** The deliberate end. When retention schedules or privacy
   obligations require it, data is securely deleted — a governed act, not
   housekeeping neglect.

Variation is normal
---------------------

Companies and industries carve the cycle differently — a hospital inserts
compliance reviews, a bank extends retention for regulators, a startup may
barely formalise it at all. The stage *names* matter less than the underlying
questions each stage answers: is this data planned, captured, kept, used,
parked, or gone?

Why the analyst should walk the cycle
---------------------------------------

Each stage upstream of *Analyze* is a place your data could have been shaped or
damaged: a Plan that never defined "active customer", a Capture form that made
the field optional, a Manage migration that truncated text. When a number looks
wrong, the life cycle is your checklist of where to look. And downstream, Archive
and Destroy explain the gaps: the missing 2019 data may not be lost — it may
have been destroyed on schedule, which is an answer, not a dead end.
"""

CONTENT["A Review of the Six Stages of the Data Life Cycle"] = r"""
Consolidating the cycle
-------------------------

This lesson consolidates the life cycle into a compact reference: each stage,
its core question, its typical owner, and its characteristic failure — the thing
that goes wrong there and surfaces later as an analyst's headache.

The stages at a glance
------------------------

- **Plan** — *What data, managed how, by whom, under what rules?* Owned by data
  governance and the teams commissioning collection. Characteristic failure:
  undefined terms, so two departments capture "customer" differently and their
  numbers never reconcile.
- **Capture** — *How does the data enter?* Owned by the systems and people at
  the point of entry. Characteristic failure: no validation at entry — free-text
  dates, optional required fields — producing the dirty data Section 4 cleans.
- **Manage** — *Where does it live, and who can reach it?* Owned by data
  engineering and IT. Characteristic failure: silos and stale copies, where the
  warehouse and the team spreadsheet quietly disagree.
- **Analyze** — *What does it tell us?* Owned by analysts. Characteristic
  failure: analysing without checking the upstream stages — precise answers
  from compromised inputs.
- **Archive** — *What do we keep, and can we still find it?* Characteristic
  failure: archives nobody can query, so history is technically kept but
  practically lost.
- **Destroy** — *What must go, and did it?* Characteristic failure at both
  extremes: deleting too eagerly (losing the baseline for next year's
  comparison) or never deleting (hoarding personal data past its lawful
  purpose).

Two threads through every stage
---------------------------------

**Security and privacy** are not a stage; they are obligations at *every* stage
— planned rules, protected capture, controlled access, careful analysis,
encrypted archives, verified destruction. Likewise **documentation**: each stage
should leave a record the next stage can rely on, which is exactly the chain of
custody the detective lesson demanded.

Using this review
-------------------

Two practical habits fall out. When you receive a dataset, *walk it backward* —
who manages it, how was it captured, what did the plan define? — before trusting
it. And when an analysis will recur, *walk it forward* — will the data still
exist, unarchived and legal to use, when the next cycle runs? Ten minutes of
life-cycle thinking routinely saves days of confused analysis.
"""

CONTENT["The Stages of the Data Analysis Process and Their Roles"] = r"""
What each phase is *for*
--------------------------

Naming the six phases is easy; using them well means knowing each phase's
**role** — the specific contribution it makes and the deliverable it owes the
next phase. This lesson treats the process as a relay: each stage exists to hand
something concrete onward.

The relay, hand-off by hand-off
---------------------------------

- **Ask** delivers a *defined problem*: the question, the stakeholders, the
  success criteria. Its role is to prevent the most expensive failure —
  precisely answering the wrong question. Everything later inherits its
  clarity or its vagueness.
- **Prepare** delivers *relevant, credible data*: identified sources, assessed
  quality, organised access. Its role is scoping — deciding what evidence can
  bear on the question at all.
- **Process** delivers *trustworthy data*: cleaned, validated, documented. Its
  role is integrity; it converts "data we have" into "data we can stand
  behind", and its documentation is what lets others verify the work.
- **Analyze** delivers *findings*: the patterns, comparisons, and numbers that
  actually answer the Ask. Its role is discovery — but only within the frame
  the earlier phases built.
- **Share** delivers *understanding*: the finding, communicated so the audience
  genuinely grasps it. Its role is translation; an unshared insight has the
  same business value as no insight.
- **Act** delivers *change*: decisions taken, experiments launched, processes
  adjusted — and the measured outcome, which seeds the next Ask.

Reading failures through the roles
------------------------------------

The roles turn vague project trouble into a diagnosis. "The analysis was right
but nothing happened" is a **Share/Act** failure, not an Analyze one. "We
answered, but it wasn't what they needed" is an **Ask** failure. "The numbers
kept changing under us" is a **Process** failure. Locating the broken hand-off
tells you what to fix — and it is rarely more computation.

Effort is front-loaded
------------------------

Beginners budget most of their time for Analyze; practitioners learn the
opposite. Asking well and preparing/processing thoroughly typically consume the
majority of a real project, precisely because their deliverables — a sharp
question and trustworthy data — determine whether the glamorous phases mean
anything. The next lesson runs the whole relay on a concrete case.
"""

CONTENT["Practical Application of the Data Analysis Process"] = r"""
The process, end to end
-------------------------

Frameworks earn their keep only in use. This lesson runs the six phases on one
realistic case from start to finish — the same shape as published walkthroughs
of the framework, including a well-known bike-share analysis that follows
exactly these steps.

The case
----------

A city bike-share company earns more from **annual members** than from
**casual riders** (single-ride and day passes). Marketing believes converting
casual riders to members is the cheapest growth available, and asks the
analytics team to help.

Running the phases
--------------------

**Ask.** The business question is sharpened to something data can answer: *how
do members and casual riders use the service differently, and what do those
differences suggest for converting casuals?* Stakeholders: marketing (will act),
finance (cares about revenue), the analytics lead (reviews the work). Success:
findings concrete enough to shape a campaign.

**Prepare.** Twelve months of trip records are identified as the evidence —
rider type, start and end times, stations. Credibility check: the data is the
company's own system of record, current, and complete; a known limitation is
that privacy rules prevent linking trips to individual riders, so the analysis
must work at the trip level.

**Process.** Cleaning finds what cleaning always finds: test rides from staff,
a handful of negative durations from clock errors, inconsistent station names
after a renaming. Rules are applied — drop rides under one minute, standardise
names — and every rule is documented so the counts are reproducible.

**Analyze.** Aggregation by rider type reveals the story: members ride briefly
and steadily on weekdays (commutes); casual riders take **longer rides,
concentrated on weekends** and afternoons (leisure). A simple pivot of average
duration and ride count by day-of-week and rider type makes the contrast vivid.

**Share.** A short deck leads with the one chart that carries the finding —
weekday-versus-weekend usage by rider type — and states the implication in
plain language: casual riders are leisure users, so membership pitches framed
around commuting will miss them.

**Act.** Marketing pilots a weekend-oriented membership offer at the busiest
leisure stations, with sign-ups tracked. The measured result — whatever it turns
out to be — becomes the data behind the *next* Ask.

What the walkthrough teaches
------------------------------

Three things worth carrying to your own projects. Most of the elapsed effort
sat in **Ask through Process**, exactly as the previous lesson predicted. The
Analyze step was, computationally, a modest aggregation — the value came from
asking a sharp question of clean data, not from sophisticated math. And the
process did not end at the insight: it ended at an **action with a measurement
attached**, which is what makes the loop a loop.
"""


MINDMAP.update({
    "Understanding the Data Life Cycle": [
        "Understanding the Data Analysis Process and the Data Life Cycle",
        "A Review of the Six Stages of the Data Life Cycle",
        "Understanding the Data Ecosystem",
        "Data Privacy in Data Ethics",
    ],
    "A Review of the Six Stages of the Data Life Cycle": [
        "Understanding the Data Life Cycle",
        "The Stages of the Data Analysis Process and Their Roles",
        "The Importance of Clean Data",
        "Data Ethics in Data Analysis",
    ],
    "The Stages of the Data Analysis Process and Their Roles": [
        "The Six Phases of the Data Analysis Process",
        "Practical Application of the Data Analysis Process",
        "A Review of the Six Stages of the Data Life Cycle",
        "Analytical Skills and Their Core Components",
    ],
    "Practical Application of the Data Analysis Process": [
        "The Stages of the Data Analysis Process and Their Roles",
        "The Six Phases of the Data Analysis Process",
        "Using Data Analysis to Choose the Right Advertising Strategy",
        "Understanding Data Analysis",
    ],
})


# ======================================================================
# Section 1 — Foundations / Stage: thinking  (lessons 013-016)
# ======================================================================

GLOSS.update({
    "Analytical Skills and Their Core Components":
        "the five skills — curiosity, context, technical mindset, data design, data strategy",
    "Applying Analytical Skills in a Business Context":
        "the five skills at work on a real business problem, phase by phase",
    "Analytical Thinking and Its Core Components":
        "the five aspects — visualization, strategy, problem-orientation, correlation, big picture + detail",
    "Analytical Thinking and Questions for Problem Solving":
        "turning the aspects into questions: root causes, gaps, and the unconsidered",
})

CONTENT["Analytical Skills and Their Core Components"] = r"""
Skills you already have
-------------------------

**Analytical skills** are the qualities and characteristics associated with
solving problems using facts. The encouraging news, before any tool is taught:
these are not exotic gifts. The standard framing focuses on **five essential
skills**, and everyday life exercises all of them — the job is to apply them
deliberately to data.

The five skills
-----------------

- **Curiosity.** Wanting to learn; seeking out new challenges and experiences,
  which is how knowledge accumulates. In analysis, curiosity is what makes you
  poke at an odd number instead of shrugging past it — the single habit behind
  most real discoveries.
- **Understanding context.** Context is the condition in which something exists
  or happens — a structure or an environment. Count "1, 2, 4, 5, 3" aloud and
  the misplaced three jars only because the *sequence* supplies context. In
  data, context is what makes a value meaningful: a labelled header row, the
  time period a table covers, the units a column is in.
- **A technical mindset.** The ability to break things into smaller steps and
  work through them in an orderly, logical way. Balancing a budget, following a
  recipe, debugging why the Wi-Fi is down — all technical mindset. It is the
  skill the six-phase process formalises.
- **Data design.** How you organise information. In the analyst's world this
  usually means literal structures — how a spreadsheet or database is laid out —
  but the instinct is the same one that organises a contact list so the right
  entry is findable.
- **Data strategy.** The management of the **people, processes, and tools**
  used in data analysis: making sure the right people know the plan, the
  process fits the problem, and the tools fit the data. Strategy is what keeps
  the other four pointed at the goal.

Why exactly these five
------------------------

Map them onto the analysis process and each earns its place: curiosity powers
*Ask*; context governs *Prepare* (is this data appropriate for this question?);
data design shapes *Process* and *Analyze*; a technical mindset carries every
phase's step-by-step work; and data strategy holds the whole project together
through *Share* and *Act*. Weakness in one shows up as a recognisable project
pathology — incurious analysts miss anomalies, context-blind ones compare
mismatched units, strategy-free projects sprawl.

The caveat
------------

Self-assessed skill lists invite box-ticking. The five are habits to *practise*,
not traits to claim: the next lesson takes one business problem and shows each
skill doing actual work — which is also the honest way to demonstrate them in
an interview.
"""

CONTENT["Applying Analytical Skills in a Business Context"] = r"""
From list to practice
-----------------------

A skills list means little until it changes what you *do* on a live problem.
This lesson takes one ordinary business situation and shows each of the five
analytical skills earning its keep — the pattern to imitate whenever a vague
request lands on your desk.

The situation
---------------

You analyse data for a mid-sized coffee-shop chain. The operations lead says:
*"Afternoon sales are weak at some stores. Figure out what's going on."* No
dataset attached, no definition of "weak", no deadline. This is exactly how real
work arrives.

Each skill, doing work
------------------------

- **Curiosity** resists the first easy story ("afternoons are just slow").
  Which stores? How weak, versus what baseline? Since when? Curious questions
  turn a complaint into an investigable phenomenon.
- **Understanding context** asks what surrounds the numbers before comparing
  them: store locations (office district versus residential — their natural
  afternoon traffic differs), seasonality, a recent menu change, roadworks
  outside two branches. Context determines which comparisons are fair.
- **A technical mindset** decomposes the vague ask into ordered steps: define
  "afternoon" (2–5 pm) and "weak" (below the store's own trailing average),
  pull sales by store and hour, compare against each store's baseline, then
  rank the gaps. A muddle becomes a checklist.
- **Data design** organises the working data so the analysis is possible: one
  row per store-day-hour, columns for sales, transactions, and store
  attributes — a tidy layout that makes the pivot in the Analyze step a
  one-liner instead of a wrestling match.
- **Data strategy** manages the surrounding people, process, and tools:
  confirming with the operations lead what decision hangs on the answer
  (staffing? promotions?), agreeing what "done" looks like, and choosing tools
  the stakeholders can actually open.

What the skills produced
--------------------------

Notice that four of the five did their work **before any analysis ran**. The
eventual finding — say, that the weakness concentrates in office-district
stores after a competitor's loyalty-app launch — is only reachable because
curiosity widened the question, context flagged the store types, the technical
mindset ordered the steps, design made the comparison computable, and strategy
kept the output aimed at a decision.

The caveat
------------

Business context also includes what data *cannot* say: three weeks of sales
cannot prove the competitor caused the dip, only that timing and geography are
consistent with it. Applying analytical skills in business means stating that
boundary plainly — stakeholders trust analysts who are precise about
uncertainty far longer than ones who overclaim.
"""

CONTENT["Analytical Thinking and Its Core Components"] = r"""
From skills to a way of thinking
----------------------------------

The five skills describe *capacities*; **analytical thinking** describes the
way of working that deploys them: identifying and defining a problem, then
solving it using data in an organised, step-by-step manner. The standard
framing again names **five key aspects** — and they double as a checklist for
whether an analysis is actually thought through.

The five aspects
------------------

- **Visualization.** The graphical representation of information — graphs,
  charts, maps. Its role in *thinking* (not just presenting) is that visuals
  let you and others grasp structure faster than words: explaining the Grand
  Canyon verbally is hard; showing a picture is instant. Analysts plot early,
  not only at the end.
- **Strategy.** With endless data available, strategic thinking keeps the work
  focused: what exactly do we want to achieve, and how will this data get us
  there? Strategy also improves the *quality* of what gets collected, because
  data gathered with a goal is data worth keeping.
- **Problem-orientation.** Keeping the problem front and centre through the
  whole effort — every query, chart, and detour judged by whether it moves the
  actual question forward. It is the antidote to interesting-but-irrelevant
  rabbit holes.
- **Correlation.** Noticing relationships: two things rising together, one
  metric leading another, patterns across stores or seasons. Correlations are
  where hypotheses come from — with the permanent caution that **correlation is
  not causation**; ice-cream sales and sunburns rise together because of
  summer, not each other.
- **Big-picture and detail-oriented thinking.** The jigsaw-puzzle pair: seeing
  the image on the box *and* fitting individual pieces. Big-picture thinking
  keeps the analysis relevant to the organisation's goals; detail thinking
  makes the plan executable — the specifics that turn an idea into steps.

The aspects work as a system
------------------------------

They interlock rather than stack: strategy and problem-orientation choose
*what* to look at; visualization and correlation are *how* patterns get
noticed; the big-picture/detail pair keeps zooming calibrated so neither the
goal nor the specifics get lost. An analysis weak in one aspect usually shows
it — a beautiful dashboard with no problem behind it, or a rigorous answer to a
question nobody strategically needed.

The caveat
------------

Analytical thinking is *slower* than intuition on purpose — its value is
exactly the discipline of defining before solving. The skill to build is
knowing when the stakes justify the full apparatus and when a quick, honest
look suffices; the next lesson sharpens the apparatus into concrete questions.
"""

CONTENT["Analytical Thinking and Questions for Problem Solving"] = r"""
Thinking as asking
--------------------

Analytical thinking becomes practical the moment it turns into **questions** —
specific, answerable ones aimed at a problem. Experienced analysts carry a
small battery of them and fire it at every new situation. Three question
families do most of the work.

Root causes: asking why, five times
-------------------------------------

The first family digs for the **root cause** — the real reason a problem
happens, as opposed to its symptoms. The simplest tool is the **five whys**:
state the problem, ask *why* it happened, then ask *why* of each answer,
roughly five layers deep, until the answer stops changing.

  Afternoon sales dropped. *Why?* Fewer customers after 2 pm. *Why?* Regulars
  from nearby offices stopped coming. *Why?* A competitor opened with an app
  discount. *Why does that pull our regulars?* We have no comparable loyalty
  offer. — The fix now targets loyalty, not, say, the menu.

Treating a symptom feels productive and changes nothing; the whys are cheap
insurance against solving the wrong layer. (The next lesson gives this tool a
fuller treatment.)

Gaps: where are we, versus where we want to be
------------------------------------------------

The second family is **gap analysis**: examining how a process works *now*,
specifying where it should be, and studying the distance between. Ship in five
days but promise three? The gap is the object of analysis — where exactly do
the two days go? Gap questions convert ambitions ("get faster") into measurable
targets, and they pair naturally with data: current state and desired state are
both numbers.

The unconsidered: what have we not thought about
--------------------------------------------------

The third family guards against blind spots: *What have we not considered? Who
is not represented in this data? What would make this conclusion wrong?* These
questions have no formula — they are curiosity and context, weaponised — but
asking them routinely catches the omitted store, the unlogged failure case,
the seasonal effect the date range happened to exclude.

A worked battery
------------------

Faced with "customer complaints are up," the battery runs: *Why* (five times —
up because response times rose, because tickets are misrouted, because the new
category list confuses agents); *what is the gap* (current 18-hour response
versus the 8-hour target); *what is unconsidered* (complaints arriving via
social media are not in the ticket data at all). Three families, one problem,
and the investigation now has direction, a measure, and a known blind spot.

The caveat
------------

Questions structure an investigation; they do not replace evidence. A five-whys
chain is a **hypothesis** about causation until data confirms each link —
plausible chains that verify beautifully in the room and fail in the data are
common. Ask the questions, then make the data answer them.
"""


MINDMAP.update({
    "Analytical Skills and Their Core Components": [
        "Detectives and Data Analysts",
        "Applying Analytical Skills in a Business Context",
        "Analytical Thinking and Its Core Components",
        "Data-Driven Decision-Making and the Role of Analytical Skills",
    ],
    "Applying Analytical Skills in a Business Context": [
        "Analytical Skills and Their Core Components",
        "Analytical Thinking and Questions for Problem Solving",
        "The Six Phases of the Data Analysis Process",
        "Understanding Common Problem Types in Data Analytics",
    ],
    "Analytical Thinking and Its Core Components": [
        "Analytical Skills and Their Core Components",
        "Analytical Thinking and Questions for Problem Solving",
        "Mathematical Thinking",
        "Data-Driven Decision-Making",
    ],
    "Analytical Thinking and Questions for Problem Solving": [
        "Analytical Thinking and Its Core Components",
        "Root Cause Analysis and Business Applications of the Five Whys",
        "Why Asking the Right Questions Matters in Data Analytics",
        "Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making",
    ],
})


# ======================================================================
# Section 1 — Foundations / thinking (cont.) + tools opener  (017-020)
# ======================================================================

GLOSS.update({
    "Root Cause Analysis and Business Applications of the Five Whys":
        "Toyota's why-times-five: digging past symptoms to causes worth fixing",
    "Data-Driven Decision-Making and the Role of Analytical Skills":
        "how the five skills power each step of the decision loop",
    "Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making":
        "the evidence in action: research findings and worked cases where data changed the outcome",
    "Overview of Core Tools Used by Data Analysts":
        "the working toolkit — spreadsheets, SQL, visualization tools, and code — and when each fits",
})

CONTENT["Root Cause Analysis and Business Applications of the Five Whys"] = r"""
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
"""

CONTENT["Data-Driven Decision-Making and the Role of Analytical Skills"] = r"""
The loop meets the skills
---------------------------

Two earlier threads join here. Data-driven decision-making is a **loop** — ask,
gather, analyse, decide, measure. The five analytical skills are **capacities**
— curiosity, understanding context, a technical mindset, data design, data
strategy. This lesson makes the connection explicit: the loop is *powered* by
the skills, and each turn of it draws on specific ones. The standard framing
says it directly: analysts use the five skills *to make data-driven decisions*.

Skill by skill, around the loop
---------------------------------

- **Curiosity** opens the loop and keeps it honest. It generates the questions
  worth asking, and later refuses to let an odd result pass unexamined — the
  difference between a decision informed by data and one merely decorated
  with it.
- **Understanding context** guards the gather-and-analyse steps. Is this data
  appropriate for this decision? Collected when, under what conditions,
  measuring what exactly? Context is what stops a technically correct
  computation from being a practically wrong answer.
- **A technical mindset** carries the middle of the loop: decomposing the
  decision into checkable sub-questions, working through cleaning and
  analysis in an orderly sequence, and making the path reproducible.
- **Data design** determines whether the analysis is even feasible: data
  organised for the question (tidy rows, meaningful labels, joinable keys)
  makes the comparison a query; disorganised data makes it a salvage
  operation.
- **Data strategy** wraps the whole loop in management — the right *people*
  informed, a *process* matched to the decision's stakes, and *tools* the
  organisation can sustain. Strategy is also what ensures the final step,
  measurement, actually happens, so the loop closes instead of trailing off.

Why decisions fail without them
---------------------------------

Run the loop with a skill missing and the failure is predictable. No curiosity:
the first plausible answer ships. No context: last year's data answers this
year's question. No technical mindset: irreproducible spreadsheet archaeology.
No design: a week lost reshaping data. No strategy: a fine analysis lands on
the wrong desk, in the wrong format, after the decision was made. Organisations
that decide well with data are, concretely, organisations whose analysts
exercise these five habits at each step — the firm-level productivity evidence
from earlier lessons is these micro-behaviours, aggregated.

The takeaway
--------------

"Be data-driven" is advice about an outcome. The five skills are advice about
**behaviour** — what to actually do at each step so the outcome follows. When a
decision process feels off, locate the loop step that is struggling and ask
which skill it is starving for; that diagnosis is usually the fix.
"""

CONTENT["Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making"] = r"""
From claims to cases
----------------------

The foundations stage has argued that data-driven decisions outperform
instinct. This closing lesson of the thinking stage collects the evidence into
cases — one from research at the scale of whole firms, and two at the scale of
a single team's project — because patterns you have seen in cases are patterns
you can reproduce.

Case: the firm-level evidence
-------------------------------

The strongest general evidence remains the research line introduced earlier.
Across large firms, adopters of data-driven decision-making showed roughly
**5–6% higher output and productivity** than their other investments would
predict, with matching gains in asset utilisation, return on equity, and
market value — and the study design supported causation rather than mere
correlation. Follow-up work in US manufacturing confirmed the productivity
benefit as robust and causal, found the advantage strongest for **early
adopters**, and observed the frontier shifting toward predictive analytics as
basic data use became universal. The practical reading: the payoff is real,
and it goes to those who use data *better than their competitors do*, not to
those who merely possess it.

Case: the bike-share conversion
---------------------------------

The process stage's worked example is itself a canonical case shape. A
bike-share company wants casual riders converted to members; trip data reveals
casual riders concentrate in **long weekend leisure rides** while members ride
short weekday commutes; marketing re-targets its membership pitch at weekend
leisure users at the stations where they actually are. The impact mechanism is
worth naming: the data did not make the decision — it **changed which decision
was available**, replacing a commuter-framed campaign that would have missed
its audience with one aimed where the audience demonstrably is.

Case: the experiment as a habit
---------------------------------

The third case is a practice rather than a single event: **A/B testing** as an
institutional habit. A team that routinely tests variants — two homepage
designs, two email subject lines, two price presentations — converts opinion
disputes into measurements, accumulates many small verified wins, and, just as
valuably, *kills* confident ideas that measure badly before they scale. The
compounding of small, verified improvements is how data-driven cultures pull
ahead without any single dramatic insight.

What the cases share
----------------------

Three constants. In every case the decisive move happened **before analysis**
— a sharp question, an honest baseline, a designed comparison. In every case
the output was an **action with a measurement attached**, closing the loop. And
in every case the alternative was not "no decision" but a decision made anyway,
on weaker grounds — which is the honest comparison for the value of this work.

The caveat
------------

Published cases oversample successes. For every celebrated analytics win there
are quiet projects where the data was inadequate or the finding unwelcome, and
the discipline's real value includes the studies that *prevented* bad
launches — impact that rarely gets a write-up. Read cases for their method, not
as promises of guaranteed results.
"""

CONTENT["Overview of Core Tools Used by Data Analysts"] = r"""
The working toolkit
---------------------

Everything so far has been mindset and process; the rest of the course is
largely **tools**. Four families cover the vast majority of working analysts'
time, and this course teaches all four. The right question is never "which is
best?" but "which fits this task, this data size, and this audience?"

The four families
-------------------

- **Spreadsheets** (Excel, Google Sheets). The universal entry point: data
  visible in a grid, formulas for calculation, built-in sorting, filtering,
  pivot tables, and charts. Strengths: immediacy, transparency, and the fact
  that every stakeholder can open one. Fit: small-to-medium datasets, quick
  analyses, and anything a business partner must inspect themselves.
- **SQL** (Structured Query Language). The language for asking questions of
  **databases**, where organisational data actually lives. A few clauses —
  ``SELECT``, ``FROM``, ``WHERE``, ``GROUP BY`` — retrieve and aggregate
  millions of rows in seconds. Fit: data too large or too shared for a
  spreadsheet; the single most consistently demanded analyst skill.
- **Visualization tools** (Tableau and its peers). Purpose-built for turning
  results into interactive charts and dashboards. Fit: exploration by eye and
  communication to stakeholders — the Share phase, industrialised.
- **Programming languages** (Python — this course's Section 7 — and R). Code
  handles what the others cannot: automation of repeated work, cleaning logic
  too complex for formulas, statistical analysis, and reproducible pipelines
  where the script *is* the documentation.

One task, four lenses
-----------------------

The same monthly-sales-by-region question: in a spreadsheet, a pivot table; in
SQL, ``SELECT region, SUM(sales) ... GROUP BY region``; in Tableau, a map
coloured by the same aggregate; in Python, three lines of pandas that can run
automatically every month. Identical logic — grouping and summing — in four
costumes. Learn the *logic* once and each new tool is mostly new syntax, which
is why this course keeps re-solving familiar problems as the tools advance.

Choosing, in practice
-----------------------

Three questions settle most choices. **How big is the data?** Spreadsheets
strain past tens of thousands of rows; SQL and Python do not. **Who consumes
the result?** A stakeholder who lives in Excel should receive Excel; a team
that monitors continuously deserves a dashboard. **Will it repeat?** One-off
work favours the fastest tool to hand; anything monthly favours a scripted,
rerunnable pipeline. Real projects chain the families — SQL to extract, Python
to clean, a spreadsheet or dashboard to deliver — and fluency across the chain
is precisely what the coming sections build.

The caveat
------------

Tools date; the toolkit's *shape* does not. Vendors and versions will change
after this course, but "a grid for inspection, a query language for scale, a
canvas for communication, code for automation" has been the stable anatomy for
decades. Invest accordingly: deepest in the logic, comfortably in today's
tools, and calmly toward tomorrow's.
"""


MINDMAP.update({
    "Root Cause Analysis and Business Applications of the Five Whys": [
        "Analytical Thinking and Questions for Problem Solving",
        "Detectives and Data Analysts",
        "Data-Driven Decision-Making",
        "Applying Data Analytics Problem Types in Real Business Scenarios",
    ],
    "Data-Driven Decision-Making and the Role of Analytical Skills": [
        "Analytical Skills and Their Core Components",
        "Data-Driven Decision-Making",
        "Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making",
        "Applying Analytical Skills in a Business Context",
    ],
    "Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making": [
        "How Data Analytics Improves the Workplace",
        "Data-Driven Decision-Making and the Role of Analytical Skills",
        "Practical Application of the Data Analysis Process",
        "Using Data Analysis to Choose the Right Advertising Strategy",
    ],
    "Overview of Core Tools Used by Data Analysts": [
        "The Role of Spreadsheets in Data Analysis and Basic Concepts",
        "The Concept and Basic Use of SQL (Query Language)",
        "The Role and Importance of Data Visualization",
        "Introduction to Python and Programming Fundamentals",
    ],
})


# ======================================================================
# Section 1 — Foundations / Stage: tools (cont.)  (lessons 021-024)
# ======================================================================

GLOSS.update({
    "The Role of Spreadsheets in Data Analysis and Basic Concepts":
        "the analyst's first instrument: cells, formulas, and functions in a visible grid",
    "The Concept and Basic Use of SQL (Query Language)":
        "asking databases questions: SELECT, FROM, WHERE and why they scale",
    "The Role and Importance of Data Visualization":
        "why pictures beat tables: seeing structure that summary numbers hide",
    "Industries Where Data Analysts Work and How Data Is Used":
        "the same craft in many rooms: retail to healthcare to public service",
})

CONTENT["The Role of Spreadsheets in Data Analysis and Basic Concepts"] = r"""
The visible workbench
-----------------------

A **spreadsheet** is data you can *see*: a grid where every value sits in a
labelled cell, every change is immediate, and every intermediate step is
inspectable. That visibility is why it is the first tool this course teaches
and the tool most analyses still begin in — it makes the abstract operations of
analysis (sort, filter, compute, summarise) concrete and watchable.

The anatomy
-------------

- **Cells** — the atoms, each addressed by column letter and row number
  (``B7``). A cell holds a value *or* a formula that computes one.
- **Rows and columns** — by convention, a **row is one record** (one sale, one
  customer, one day) and a **column is one attribute** (date, region, amount).
  Keeping that convention is half of good data design.
- **Headers** — the first row, naming each column. A header is context made
  explicit: ``ORDER_DATE`` tells you what the values below mean.
- **Formulas** — expressions beginning with ``=`` that compute from other
  cells: ``=C2*D2`` multiplies price by quantity. Change an input and every
  dependent formula updates — the spreadsheet's quiet superpower.
- **Functions** — named, prebuilt operations used inside formulas:
  ``=SUM(E2:E100)``, ``=AVERAGE(...)``, ``=COUNT(...)``, ``=MAX(...)``. They
  are the vocabulary the analysis sections expand enormously.

What analysts actually do with them
-------------------------------------

Across the six phases, the spreadsheet serves at least four roles: **inspect**
(eyeball raw data for obvious problems), **organise** (sort and filter into
meaningful order), **calculate** (derive new columns and summary figures), and
**communicate** (a labelled table or quick chart a stakeholder can open with no
special tools). A first pass on almost any small dataset — scan it, sort it,
total it — is a spreadsheet task done in minutes.

A worked miniature
--------------------

A sheet of orders with ``PRICE`` in column C and ``QUANTITY`` in column D:
add a header ``REVENUE`` in E1, put ``=C2*D2`` in E2, fill it down, and
``=SUM(E2:E101)`` gives total revenue for a hundred orders. Three formulas,
and raw records have become a business number — the entire shape of analysis,
in miniature.

The honest limits
-------------------

Spreadsheets strain as data grows past tens of thousands of rows, and their
flexibility is a double edge: any cell can be quietly overtyped, so errors hide
in plain sight and there is no built-in record of what changed. Later lessons
treat spreadsheet *errors* and *verification* as first-class topics for exactly
this reason. The scaling limit is what SQL, next, exists to remove.
"""

CONTENT["The Concept and Basic Use of SQL (Query Language)"] = r"""
Asking questions of databases
-------------------------------

Organisational data mostly does not live in spreadsheets; it lives in
**databases** — structured collections of tables holding millions of records.
**SQL** (Structured Query Language) is how you talk to them: a language for
*describing the data you want* and letting the database fetch it. Where a
spreadsheet shows you everything and lets you point, SQL lets you **ask** — and
the asking scales to sizes no grid could display.

The core idea: declare, don't fetch-and-loop
----------------------------------------------

An SQL **query** states *what* you want, not *how* to find it:

.. code-block:: sql

   SELECT customer_id, order_date, amount
   FROM   orders
   WHERE  amount > 100;

Read it as a sentence: *select* these columns, *from* this table, *where* this
condition holds. The three clauses are the irreducible core:

- ``SELECT`` — which columns (attributes) to return; ``SELECT *`` means all.
- ``FROM`` — which table the data lives in.
- ``WHERE`` — which rows qualify, via conditions (``=``, ``>``, ``<``,
  combined with ``AND`` / ``OR``).

The database engine — not you — figures out the efficient way to satisfy the
request across millions of rows, which is precisely why the language stays this
simple at this scale.

The spreadsheet translation
-----------------------------

Every clause has a spreadsheet counterpart you already know: ``SELECT`` is
choosing columns to look at, ``FROM`` is opening the right sheet, ``WHERE`` is
a filter. The difference is not conceptual but operational: the query is
**text** — repeatable, shareable, reviewable — and it runs against the live,
shared source of record rather than a private copy. When the analysis section
later adds grouping and joining, those too will be filters and pivots you have
already met, in query form.

Why every analyst learns it
-----------------------------

Three durable reasons. **Location:** the data you need is already in a
database; SQL goes to it rather than exporting fragile copies. **Scale:** a
``WHERE`` over ten million rows returns in seconds. **Ubiquity:** SQL has been
the standard interface to relational data for decades, is asked for in more
analyst job postings than any other technical skill, and transfers almost
unchanged across database products.

The caveat
------------

A query answers exactly what it says, which is not always what you meant —
``WHERE amount > 100`` silently excludes the row where amount is missing, and
no error will tell you so. SQL's precision is a feature that demands the same
precision of your question; the deeper query techniques and their pitfalls get
full treatment in the cleaning and analysis sections.
"""

CONTENT["The Role and Importance of Data Visualization"] = r"""
Seeing what summaries hide
----------------------------

**Data visualization** is the graphical representation of data — charts,
graphs, maps. Its role is not decoration: the human visual system processes
patterns, trends, and outliers far faster from a picture than from a table of
numbers, so a good chart is frequently the difference between an insight
noticed and an insight missed.

The classic demonstration
---------------------------

The statistician Francis Anscombe made the point permanently in 1973 with four
small datasets now known as **Anscombe's quartet**. All four share nearly
identical summary statistics — same means, same variances, essentially the
same correlation and fitted line — yet *plotted*, they are utterly different: a
clean linear trend, a smooth curve, a tight line with one gross outlier, and a
vertical stack with a single leveraging point. The lesson is exact: **summary
numbers can agree while the realities they summarise disagree**, and only the
picture reveals it. "Plot your data before you trust your statistics" has been
standard advice ever since.

Two jobs: exploring and explaining
------------------------------------

Visualization serves the analyst twice, at opposite ends of the process.

- **Exploratory** charts are for *you*, early: quick, rough plots to see
  distributions, spot outliers, and let the data suggest hypotheses — the
  look-first habit of exploratory data analysis.
- **Explanatory** charts are for *others*, late: deliberate, polished visuals
  built to carry one finding clearly to an audience — the heart of the Share
  phase and of Section 6.

The same chart types serve both, but the standards differ: exploration
optimises for speed and coverage, explanation for clarity and honesty.

The basic repertoire
----------------------

Four forms cover most needs, matched to the question. **Bar charts** compare
categories (sales by region). **Line charts** show change over time (revenue by
month). **Scatter plots** expose relationships between two measures (price
versus demand — where Anscombe's quartet lives). **Histograms** show a single
variable's distribution (order sizes). Choosing among them — and beyond them —
is a craft Section 6 develops fully; the founding rule is simply that the
**question picks the chart**.

The caveat
------------

The same power that makes charts persuasive makes them dangerous: a truncated
axis, a cherry-picked window, or a misleading scale can manufacture an
impression the data does not support. Visualization is an argument, and the
analyst's obligation is that the argument be honest — a responsibility treated
in depth when this course reaches chart design and data ethics.
"""

CONTENT["Industries Where Data Analysts Work and How Data Is Used"] = r"""
One craft, many rooms
-----------------------

The skills in this course are deliberately industry-agnostic: the six phases,
the five skills, and the four tool families work the same everywhere. What
changes by industry is the **data, the questions, and the constraints** — and
knowing the landscape helps you both choose where to work and translate your
experience across sectors.

A tour of the landscape
-------------------------

- **Retail and e-commerce.** Transactions, inventory, and web behaviour feed
  pricing, stock planning, and marketing decisions — which products to
  promote, where demand is shifting, which campaigns pay.
- **Finance and banking.** Risk assessment, fraud detection, and portfolio
  reporting; heavy regulation makes accuracy and auditability paramount, and
  analysts spend real effort on data lineage.
- **Healthcare.** Patient outcomes, treatment effectiveness, resource
  scheduling; strict privacy rules (who may see what) shape every dataset an
  analyst touches.
- **Marketing and media.** Campaign performance, audience segmentation, and
  content engagement — a field practically built on A/B testing.
- **Logistics and manufacturing.** Delivery times, route efficiency, quality
  rates, downtime — operational data where small percentage improvements
  compound into large savings.
- **Technology.** Product usage analytics: which features are used, where
  users struggle, what drives retention.
- **Government and public service.** Census, transport, health, and budget
  data informing policy and services — often with an obligation to publish
  openly.
- **Entertainment and sport.** Audience behaviour and performance statistics
  driving content, scheduling, and team decisions.

What actually differs
-----------------------

Across the tour, three variables do the differentiating. The **unit of
analysis** (a transaction, a patient, a shipment, a user session). The
**constraints** (privacy in healthcare, regulation in finance, openness in
government). And the **cadence** (real-time fraud detection versus annual
policy analysis). The *methods* — clean, aggregate, compare, visualise,
recommend — are the constant, which is why analysts genuinely can move between
industries: the toolkit transfers, and the domain knowledge is learnable.

Choosing, and being chosen
----------------------------

Two practical implications for your own path. When picking a sector, weigh the
*data you would touch daily* and the *decisions it feeds* — the texture of the
work varies more than the job title suggests. And in applications, translate:
a retail analyst's "basket analysis" is a healthcare analyst's "treatment
co-occurrence" — the later job-search section returns to exactly this skill of
mapping your experience into a new industry's vocabulary.

The caveat
------------

Industry lists date quickly — sectors rise, merge, and rename — but the
underlying pattern has held for decades: wherever an organisation records what
it does, someone is needed to turn the record into decisions. Bet on the
pattern, not the particular boom.
"""


MINDMAP.update({
    "The Role of Spreadsheets in Data Analysis and Basic Concepts": [
        "Overview of Core Tools Used by Data Analysts",
        "The Concept and Basic Use of SQL (Query Language)",
        "Building and Organizing a Spreadsheet",
        "Spreadsheet Calculations with Formulas",
    ],
    "The Concept and Basic Use of SQL (Query Language)": [
        "Overview of Core Tools Used by Data Analysts",
        "The Role of Spreadsheets in Data Analysis and Basic Concepts",
        "Introduction to SQL",
        "Querying Data with SQL",
    ],
    "The Role and Importance of Data Visualization": [
        "Overview of Core Tools Used by Data Analysts",
        "Data Visualization",
        "Choosing the Right Visualization: Audience-Centered Design and Chart Selection",
        "The Role of Spreadsheets in Data Analysis and Basic Concepts",
    ],
    "Industries Where Data Analysts Work and How Data Is Used": [
        "How Data Analytics Improves the Workplace",
        "The Role of Business Tasks in Data Analysis",
        "Key Factors to Consider When Choosing a Data Analytics Role",
        "Fairness in Data Analysis",
    ],
})


# ======================================================================
# Section 1 — Foundations / Stage: tools (closer)  (lessons 025-027)
# ======================================================================

GLOSS.update({
    "The Role of Business Tasks in Data Analysis":
        "the question behind the work: how business tasks anchor every analysis",
    "Fairness in Data Analysis":
        "analysis that does not create or reinforce bias — and a famous failure to learn from",
    "Key Factors to Consider When Choosing a Data Analytics Role":
        "industry, company size, specialisation, growth: weighing where to start",
})

CONTENT["The Role of Business Tasks in Data Analysis"] = r"""
The question behind the work
------------------------------

A **business task** is the question or problem that a data analysis answers for
a business. It is the anchor of the entire six-phase process: the *Ask* phase
exists to define it, every later phase is judged against it, and an analysis
without one is activity without purpose. This short lesson makes the concept
precise, because "we analysed the data" means nothing until you can say *which
task the analysis served*.

From situation to task
------------------------

Business tasks rarely arrive well-formed. They arrive as situations — "sales
are down", "customers are complaining", "we're launching in a new city" — and
the analyst's first job is converting the situation into a task: specific,
answerable with data, and attached to a decision.

- Situation: *afternoon sales feel weak.* Task: *identify which stores'
  2–5 pm sales fall below their own trailing average, and what distinguishes
  them* — feeding a staffing and promotion decision.
- Situation: *we want more members.* Task: *determine how casual riders and
  members use the service differently* — feeding a marketing decision.
- Situation: *the warehouse seems slow.* Task: *measure where time is spent
  between order and shipment* — feeding a process decision.

The pattern: a task names the **comparison or measurement** to perform and the
**decision** waiting on it. If you cannot state both, the Ask phase is not
finished.

Why the task governs everything
---------------------------------

Each phase consults the task. *Prepare* asks: what data bears on **this**
question? *Process* asks: clean to what standard **this** decision needs?
*Analyze* asks: does this computation move **this** question forward?
*Share* asks: what does the decision-maker need to understand? Scope creep,
rabbit holes, and beautiful-but-irrelevant charts are all, at bottom, moments
when work detached from the task. The cheapest project-management tool an
analyst owns is rereading the task statement.

The caveat
------------

Tasks can be wrong. Sometimes honest work reveals that the stated task
misdiagnoses the situation — sales are not "down", they moved channels — and
the analyst's duty is to surface that, not to answer the broken question
faithfully. A business task is the anchor of the analysis, not a gag order on
what the data actually shows.
"""

CONTENT["Fairness in Data Analysis"] = r"""
What fairness means here
--------------------------

**Fairness** in data analysis means ensuring that the analysis does not create
or reinforce bias — that its conclusions and the decisions built on them do not
systematically disadvantage groups of people. It belongs in the foundations of
this course, not an appendix, because unfair analysis is usually not malicious:
it is ordinary work done on unexamined data, and every analyst is one
unexamined dataset away from producing it.

A famous failure, instructive in every detail
-----------------------------------------------

The canonical case: as reported by Reuters in 2018, Amazon built an
experimental tool to score job applicants' resumes, training it on roughly ten
years of resumes the company had received. Because the historical applicant
pool was heavily male, the tool **taught itself that male candidates were
preferable** — penalising resumes containing the word "women's" (as in
"women's chess club captain") and downgrading graduates of all-women's
colleges. Engineers tried to correct it, could not guarantee neutrality, and
the tool was scrapped.

Every step generalises. The data faithfully recorded a biased history; the
analysis faithfully learned the bias; the output would have faithfully
projected it into future decisions. Nothing was "wrong" technically — which is
precisely the warning. As one civil-liberties commentary put it, such systems
do not remove human bias; they **launder it through software**: ask a model to
find candidates who resemble past successes, and reproducing the demographics
of the past workforce is virtually guaranteed.

Where unfairness enters ordinary analysis
-------------------------------------------

You do not need machine learning to repeat the pattern. The entry points are
mundane:

- **Unrepresentative data** — a customer survey answered mostly by one
  demographic, treated as "what customers think".
- **Historical bias in the target** — measuring "success" by past outcomes
  that themselves reflect unequal treatment.
- **Proxy variables** — a neutral-looking field (postal code, college name)
  that stands in for a protected characteristic.
- **Aggregation that hides harm** — an average that improves while a subgroup
  worsens.

The prep section's bias lessons dissect these mechanisms; here the point is
that each is *detectable* if someone looks.

The analyst's fairness habits
-------------------------------

Four habits, cheap and repeatable. **Ask who is in the data** — and who is
missing — before trusting it. **Disaggregate**: check results across relevant
groups, not just overall. **Interrogate proxies**: for any influential
variable, ask what it might be standing in for. **Trace the decision**: follow
the analysis to the people it will affect, and ask who bears the cost of its
errors. None requires special tools; all require deciding that fairness is
part of the job — which, in this course, it is.

The caveat
------------

Fairness is not a checkbox with a formula; reasonable definitions can even
conflict, and judgement is unavoidable. But the failure mode to fear is not
subtle philosophy — it is nobody having looked at all. The habits above are the
looking.
"""

CONTENT["Key Factors to Consider When Choosing a Data Analytics Role"] = r"""
The same title, very different jobs
-------------------------------------

"Data analyst" names a family of jobs, not one job. Two analysts with identical
titles can spend their days utterly differently — one building dashboards for
marketing, another writing SQL pipelines for a hospital, a third doing a bit of
everything at a startup. Closing the foundations section, this lesson gives you
the factors that actually differentiate roles, so you can read postings — and
eventually offers — with clear eyes.

The factors that matter
-------------------------

- **Industry.** The industries lesson showed how the data, questions, and
  constraints differ by sector. Choose partly by the *subject matter* you want
  to think about daily — patient outcomes, product usage, logistics — because
  domain interest sustains the curiosity the work demands.
- **Company size.** A large company usually means specialisation (you own one
  slice, with mentors and established tooling); a small one means breadth (you
  are the data function, touching everything with less guidance). Neither is
  better; they develop different strengths at different speeds.
- **Team placement.** Embedded in a business team, you sit close to decisions
  and go deep on one domain; in a central analytics team, you see many
  problems and learn from other analysts. Ask where the role reports and who
  reviews the work.
- **Specialisation versus generalism.** Some roles lean toward a craft —
  visualization-heavy, SQL/pipeline-heavy, experimentation-heavy. Early on,
  breadth builds the foundation; later, a deliberate specialisation is often
  what commands seniority.
- **Growth and mentorship.** Who would you learn from? Is there a path from
  this role to the next one? A modest role with strong mentorship frequently
  outruns a shinier one without it.
- **Tools and stack.** The posting's tool list tells you what you will
  practise daily. Widely used tools (SQL, the spreadsheet family, Python)
  transfer; exotic internal ones may not.

Reading a posting with the factors
------------------------------------

The factor list turns a vague posting into concrete questions for the
interview: *What does a typical week look like? Who uses my analysis, and for
what decisions? Who reviews my work? What does growth from this seat look
like?* Answers to those four reveal more than any title.

Closing the foundations
-------------------------

This ends the foundations section: you have the case for the field, the
process, the thinking, the toolkit, the ethics, and now the map of the roles
themselves. Everything from here is depth — starting, in the next section,
with the craft that shapes every project before a single row of data is
touched: turning business situations into the right questions.
"""


MINDMAP.update({
    "The Role of Business Tasks in Data Analysis": [
        "Why Asking the Right Questions Matters in Data Analytics",
        "Data-Driven Decision-Making",
        "Defining the Problem Domain",
        "Industries Where Data Analysts Work and How Data Is Used",
    ],
    "Fairness in Data Analysis": [
        "Understanding Bias in Data Analysis",
        "Common Types of Data Bias",
        "Data Ethics in Data Analysis",
        "Context and Bias in Data Analysis",
    ],
    "Key Factors to Consider When Choosing a Data Analytics Role": [
        "Industries Where Data Analysts Work and How Data Is Used",
        "Analytical Skills and Their Core Components",
        "Exploring Data Analyst Job Opportunities",
        "Transferable Skills",
    ],
})


# ======================================================================
# Section 2 — Data-Driven Decisions / Stage: framing  (ddd 001-004)
# ======================================================================

GLOSS.update({
    "Using Data Analysis to Choose the Right Advertising Strategy":
        "a worked prediction problem: past campaign data steering the next ad spend",
    "Understanding Common Problem Types in Data Analytics":
        "the six shapes of analyst problems, from predictions to patterns",
    "Applying Data Analytics Problem Types in Real Business Scenarios":
        "recognising the six types in the wild — and why naming the type speeds the work",
    "Why Asking the Right Questions Matters in Data Analytics":
        "SMART questions, and the leading/closed/assuming questions to avoid",
})

CONTENT["Using Data Analysis to Choose the Right Advertising Strategy"] = r"""
A decision every business faces
---------------------------------

Where should the next advertising money go? Radio, social media, search ads,
billboards, a sponsorship? Every organisation that markets faces this choice,
and it is the canonical example of a **prediction problem**: use data about
what past campaigns produced to make an informed decision about what a future
one will produce. This section opens with it because it shows the entire
decision-making arc of the coming lessons in one familiar case.

What the data can say
-----------------------

Suppose the company has records of past campaigns: the **medium**, the
**location or audience targeted**, the **spend**, and the **number of new
customers acquired** in each campaign window. Analysis can then compare like
with like:

- cost per new customer, by medium — which channel acquires cheapest;
- performance by location or audience — where each medium works best;
- trend over time — whether a channel's performance is rising or fading.

A simple table of *cost per acquisition by channel and region* is often the
whole deliverable: it converts "which advertising do we believe in?" into
"which row of this table is smallest, and do we trust why?"

What the data cannot say
--------------------------

Past performance cannot **guarantee** future results — the honest phrasing is
that analysis helps *predict the best placement to reach the target audience*,
not that it certifies an outcome. Markets shift, competitors react, and a
channel that worked at small spend may saturate at large spend. Two
disciplines keep the prediction honest: state the assumption (next quarter
resembles last), and **instrument the decision** — run the chosen strategy in
a way that measures its result, so the next round of this same decision has
fresher evidence. That is the decision loop from the foundations, operating.

Why this case opens the section
---------------------------------

Notice what the analysis required *before* any computation: someone had to
frame the vague worry ("is our advertising working?") as a specific,
answerable, decision-attached question ("which channel acquires customers
cheapest, for whom, and is that stable?"). The rest of this stage teaches that
framing skill in general — the problem types that recur, and the questions
that unlock them.
"""

CONTENT["Understanding Common Problem Types in Data Analytics"] = r"""
Six shapes of problems
------------------------

Analyst problems feel endlessly varied, but the standard framing observes that
they overwhelmingly take **six recurring shapes**. Recognising the shape early
matters: each type suggests its own data, methods, and outputs, so naming the
type is the first act of structuring the work.

The six types
---------------

- **Making predictions.** Using data to make informed decisions about how
  things may be in the future — the advertising-channel choice of the previous
  lesson. Past data cannot guarantee outcomes, only inform them.
- **Categorizing things.** Assigning items to groups based on common
  features: classifying customer-service calls by keywords or scores, tagging
  products into price tiers, grading suppliers.
- **Spotting something unusual.** Identifying data outside the norm: a smart
  watch flagging an abnormal heart rate, a sudden dip in daily sign-ups, one
  region's numbers that stopped moving (often a broken pipeline, not a calm
  market).
- **Identifying themes.** Taking categorisation a step further by grouping
  categories into **broader concepts**: hundreds of differently-worded survey
  comments becoming the themes "pricing", "reliability", "support tone".
- **Discovering connections.** Finding how different entities' problems
  relate, and combining data to solve them: a logistics firm analysing wait
  times at shipping hubs to change schedules and lift on-time deliveries.
- **Finding patterns.** Using historical data to understand what tends to
  happen and is therefore likely to happen again: maintenance records showing
  most machine failures occur when regular servicing slips past a certain
  window.

The pair people confuse
-------------------------

**Categorizing** versus **identifying themes**: categorising groups *the same
kind* of thing together (all calls scored 1–3); themes group *similar but
different* things under a broader concept (many distinct complaints, one
theme of "delivery frustration"). Categories are boxes; themes are the ideas
the boxes suggest.

Using the typology
--------------------

When a request lands, ask: *which type is this?* "Why did churn spike in
March?" is spotting something unusual, then finding patterns. "Which customers
should get the premium pitch?" is categorizing, in service of a prediction.
Many real projects **chain types** — unusual → pattern → prediction is the
classic investigation arc — and recognising the chain tells you the order of
work. The next lesson practises exactly this recognition on realistic
scenarios.
"""

CONTENT["Applying Data Analytics Problem Types in Real Business Scenarios"] = r"""
From definitions to recognition
---------------------------------

Knowing the six problem types is recall; the working skill is **recognition**
— hearing a messy business situation and naming its type fast, because the
name brings a method with it. This lesson drills that recognition on scenarios
of the kind analysts actually receive.

Scenario practice
-------------------

- *A hospital wants to anticipate patient volumes as the local population
  ages.* — **Making predictions**: historical admissions plus demographics
  inform staffing for the future.
- *An online retailer wants its review flood made sense of.* — **Categorizing
  things** first (scores, product areas), then **identifying themes** (the
  broader concepts — sizing, shipping speed — that many different reviews
  share).
- *A payments team wants fraud caught as it happens.* — **Spotting something
  unusual**: define normal transaction behaviour, flag departures.
- *A delivery company keeps missing promised dates.* — **Discovering
  connections**: link warehouse wait times, carrier handoffs, and route data
  to find where delay is created, then change schedules.
- *A manufacturer wants less machine downtime.* — **Finding patterns**:
  failure logs against maintenance records reveal the servicing delay beyond
  which failures cluster.
- *A subscription service wants to know which trial users will convert.* —
  **Making predictions**, built on **categorizing** users by behaviour.

Notice how naming the type immediately suggests the *data to request* and the
*shape of the answer* — a forecast, a grouping, an alert, a theme list, a
linkage, a rule.

Chains in the wild
--------------------

Real investigations chain types. "Sales dipped in the northwest" begins as
spotting something unusual; explaining it is finding patterns (does it recur
seasonally?) or discovering connections (did a distributor change?); acting on
it becomes a prediction (if we restore X, sales recover). Writing down the
chain — *unusual → connection → prediction* — is a one-line project plan, and
it tells stakeholders what kind of answer each stage can deliver.

The caveat
------------

Typologies are lenses, not laws. Some problems genuinely straddle types, and
forcing a fit wastes time. The test of a good type assignment is practical:
did it tell you what data to get and what output to produce? If yes, it
served; if you are debating taxonomy instead of gathering evidence, put the
lens down and look at the problem again — starting with the questions the
next lesson teaches you to ask.
"""

CONTENT["Why Asking the Right Questions Matters in Data Analytics"] = r"""
The question is the steering wheel
------------------------------------

Every failure mode met so far — answering the wrong problem, gathering
irrelevant data, precise answers nobody can act on — traces back to the
question asked at the start. Questions are the steering of the entire process,
and the standard framing gives them a memorable quality bar: effective
questions are **SMART**.

SMART questions
-----------------

- **Specific** — simple, significant, and focused on a single topic or a few
  closely related ideas. Not "how are sales?" but "how did repeat-customer
  revenue change after the March price update?"
- **Measurable** — answerable with something you can quantify and assess. "Do
  customers like us?" becomes "what fraction of survey respondents rate us 8+,
  and how has it moved?"
- **Action-oriented** — framed to encourage change: "what features would make
  the timesheet page easier to complete by Friday noon?" invites answers you
  can act on, where "why don't people do timesheets?" invites blame.
- **Relevant** — mattering to the problem at hand; a fascinating question that
  no pending decision needs is a detour.
- **Time-bound** — specifying the period under study, because "recently" is
  not a date range and every comparison needs one.

Run a draft question through the five letters and its weaknesses announce
themselves — usually a missing measure or a missing time window.

The questions to avoid
------------------------

Three anti-patterns corrupt answers before any data arrives:

- **Leading questions** presuppose their answer ("how great was the new
  design?") and harvest agreement, not information.
- **Closed-ended questions** ("did you like it?") collect yes/no where the
  value lives in elaboration; they have their place in structured surveys but
  starve discovery.
- **Assumption-laden questions** build on unproven premises ("why did the
  discount drive the sales spike?" — did it? was there a spike?), smuggling a
  conclusion into the framing.

The fairness thread from the foundations applies here too: questions should
not create or reinforce bias — who a question is asked *of*, and who it
silently excludes, shapes the answer as much as its wording.

A worked sharpening
---------------------

Stakeholder: *"Can you look into whether marketing is working?"* SMART pass:
**Specific** — which campaigns, which products? **Measurable** — define
"working": cost per acquisition? revenue lift? **Action-oriented** — what
decision waits: reallocate budget? **Relevant** — is the summer campaign
review the actual occasion? **Time-bound** — compare which quarters? Ten
minutes of this produces: *"For Q1–Q2, what was cost per new customer by
channel, and which channels should receive Q3 budget?"* — a question the rest
of the process can actually serve.

The caveat
------------

SMART is a quality filter, not a source of curiosity. The sharpest analysts
still begin with open, exploratory wondering — then *refine* the promising
wonder into SMART form before committing the project to it. Filter too early
and you only ever ask what is easy to measure.
"""


MINDMAP.update({
    "Using Data Analysis to Choose the Right Advertising Strategy": [
        "Understanding Common Problem Types in Data Analytics",
        "Data-Driven Decision-Making",
        "Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making",
        "The Relationship Between Data and Decision-Making",
    ],
    "Understanding Common Problem Types in Data Analytics": [
        "Using Data Analysis to Choose the Right Advertising Strategy",
        "Applying Data Analytics Problem Types in Real Business Scenarios",
        "The Role of Business Tasks in Data Analysis",
        "Analytical Thinking and Questions for Problem Solving",
    ],
    "Applying Data Analytics Problem Types in Real Business Scenarios": [
        "Understanding Common Problem Types in Data Analytics",
        "Why Asking the Right Questions Matters in Data Analytics",
        "Root Cause Analysis and Business Applications of the Five Whys",
        "Practical Application of the Data Analysis Process",
    ],
    "Why Asking the Right Questions Matters in Data Analytics": [
        "Applying Data Analytics Problem Types in Real Business Scenarios",
        "The Role of Business Tasks in Data Analysis",
        "The Relationship Between Data and Decision-Making",
        "Data Creates Value Only When It Is Communicated",
    ],
})


# ======================================================================
# Section 2 — DDD / framing (close) + metrics (open)  (ddd 005-008)
# ======================================================================

GLOSS.update({
    "The Relationship Between Data and Decision-Making":
        "data-driven vs data-inspired: how evidence and judgement actually combine",
    "Quantitative and Qualitative Data in Decision-Making":
        "what/how many/how often meets why: two data kinds, one decision",
    "Data Creates Value Only When It Is Communicated":
        "an insight nobody hears changes nothing: communication as the last mile",
    "The Difference Between Data and Metrics, and the Role of Metrics":
        "from raw facts to quantified goals: what makes a number a metric",
})

CONTENT["The Relationship Between Data and Decision-Making"] = r"""
Two ways evidence enters a decision
-------------------------------------

The foundations established *that* data improves decisions; this stage's
working question is *how* — what does the connection actually look like in
practice? Two modes are worth distinguishing, because both are legitimate and
confusing them causes friction.

**Data-driven** decision-making uses facts derived from data as the primary
guide for a specific choice: the A/B test picks the homepage, the
cost-per-acquisition table allocates the ad budget. The data speaks directly
to the decision at hand.

**Data-inspired** decision-making is looser and earlier: exploring different
sources of data to find commonalities, patterns, and ideas — evidence shaping
*what to consider* rather than settling *what to choose*. Browsing usage data
until a neglected feature's quiet popularity suggests a product direction is
data-inspired; the later decision to invest in it, tested and measured,
becomes data-driven.

Where each belongs
--------------------

Data-driven suits decisions that are **repeatable, measurable, and
comparable**: pricing, targeting, operational tuning — anywhere alternatives
can be stated and outcomes counted. Data-inspired suits the **fuzzy front
end**: strategy, product discovery, hypothesis generation — where the options
themselves are not yet known and the job is noticing. A healthy organisation
runs both, in sequence: inspiration proposes, data-driven testing disposes.

The joints where it fails
---------------------------

The relationship breaks in recognisable ways. **Decoration**: the decision was
made first and data was gathered to justify it — the form of data-driven
without the substance. **Abdication**: "the data decided" used to dodge
accountability for a judgement call the data could not actually settle.
**Paralysis**: refusing to decide until data is complete, when data is never
complete. The remedy for all three is the same honesty about *what role the
evidence played*: informed the options, settled the choice, or merely
accompanied it.

Judgement stays in the loop
-----------------------------

Even the most data-driven decision contains judgement — in the question asked,
the metric chosen, the threshold set, the costs weighed. The mature statement
is never "the data decided" but "given the evidence, weighing these
considerations, we decided." The next lessons sharpen the ingredients of that
sentence: the kinds of data (quantitative and qualitative), and the fact that
none of it counts until it is communicated.
"""

CONTENT["Quantitative and Qualitative Data in Decision-Making"] = r"""
Two kinds of evidence
-----------------------

Decisions draw on two fundamentally different kinds of data, and the standard
definitions are worth memorising exactly:

- **Quantitative data** is a specific and objective measure of numerical
  facts. It answers **what, how many, and how often**: revenue was 1.2
  million; 340 customers churned; the page loads in 1.8 seconds.
- **Qualitative data** is a subjective or explanatory measure of qualities and
  characteristics. It answers **why**: the interview where a churned customer
  explains the cancellation; the free-text reviews; the support transcript's
  tone.

Neither is the "better" data. They answer different questions, and most real
decisions need both.

Why each needs the other
--------------------------

Quantitative data **detects and sizes**; qualitative data **explains and
humanises**. The numbers show *that* sign-ups dropped 20% after the redesign —
they cannot say the new form confused people; five usability interviews can.
Conversely, a vivid customer complaint (qualitative) may be one voice or a
epidemic — only counting (quantitative) says which. The failure modes are
symmetric: quantitative-only decisions optimise metrics while missing the
human reason behind them; qualitative-only decisions scale one articulate
anecdote into policy.

A worked pairing
------------------

A streaming service sees cancellations spike (quantitative: the *what*). Exit
surveys and interviews surface a theme: users feel the catalogue "got worse"
after a licensing change (qualitative: the *why*). Counting again shows
cancellations concentrate among heavy viewers of the removed content
(quantitative: the *who and how much*). The decision — restore, replace, or
message — now rests on sized, explained evidence. Notice the alternation:
number, story, number. That rhythm is the practical technique.

Collecting them
-----------------

Quantitative sources: transactions, logs, structured survey scales, sensors —
the material of most of this course. Qualitative sources: interviews,
open-ended survey questions, reviews, support tickets. The theme-identification
problem type from the previous lessons is exactly how qualitative material
becomes analysable: categorise the many different sayings, then name the
broader themes.

The caveat
------------

The boundary is a spectrum, not a wall: a 1–10 satisfaction score quantifies a
quality, and counting theme frequencies quantifies the qualitative. What
matters is not policing the boundary but matching evidence to question — and
resisting the pull to treat what is easily counted as all that counts.
"""

CONTENT["Data Creates Value Only When It Is Communicated"] = r"""
The last mile is the value
----------------------------

An analysis that is correct, rigorous, and unread has the same business value
as no analysis. This is the uncomfortable truth that closes the framing stage:
data creates value **only when it is communicated** — the insight must reach
the person who can act, in a form they can act on, at a time when action is
possible. Everything upstream is cost; the communicated insight is the
product.

Why good analysis dies in transit
-----------------------------------

The recurring failure modes are rarely analytical:

- **Wrong audience**: the finding went to whoever asked, not to whoever
  decides.
- **Wrong form**: a forty-tab workbook where a one-chart summary was needed —
  or a glossy chart where the engineer needed the table.
- **Wrong time**: delivered after the budget was set, the launch shipped, the
  quarter closed.
- **No so-what**: numbers presented without the implication stated, leaving
  the audience to do the analyst's final step — most won't.

Each is a communication decision the analyst controls, which is why the Share
phase is a phase and not an afterthought.

Communicating so it lands
---------------------------

Three disciplines carry most of the weight. **Lead with the answer**: state
the finding and its implication first, then support it — stakeholders are not
reading a mystery novel. **Match the medium to the audience**: executives get
the one-pager, operators get the dashboard, the analyst peer gets the notebook;
same finding, three costumes. **Anticipate the decision**: frame the
communication around what the audience must decide, because "interesting" is
not a call to action and "therefore we should choose between A and B" is.

The two-way street
--------------------

Communication is also how analysis *improves*: presenting a finding surfaces
the question you did not consider, the context you lacked, the constraint that
reshapes the recommendation. Analysts who treat sharing as a defence of
finished work learn less than those who treat it as the next round of
evidence-gathering — this time about the decision itself.

The caveat
------------

Communication amplifies whatever it carries, including error. A wrong number
confidently and clearly communicated does more damage than the same number
buried in a workbook. The obligation therefore runs both ways: make the true
thing land, and make the uncertainty land with it — which is why the later
presentation lessons treat honest framing as a design requirement, not a
disclaimer.
"""

CONTENT["The Difference Between Data and Metrics, and the Role of Metrics"] = r"""
From facts to yardsticks
--------------------------

**Data** is the raw material: the collection of facts an organisation records
— every transaction, click, and timestamp. A **metric** is something more
deliberate: a **single, quantifiable type of data used when setting and
evaluating goals**. Revenue rows are data; *monthly recurring revenue* is a
metric. Ride records are data; *average ride length by rider type* is a
metric. The difference is purpose: a metric is data that has been given a job.

What turning data into a metric involves
------------------------------------------

Three decisions convert raw facts into a yardstick, and each is a judgement:

1. **A definition** — precisely which records count. Is an "active user"
   anyone who logged in this month? Performed an action? The metric's meaning
   lives in this choice.
2. **A calculation** — the formula applied. **Customer retention rate**: of
   the customers present at the period's start, what fraction remain at its
   end. **Return on investment (ROI)**: the profit an investment produced
   relative to its cost. Same data, different formulas, different stories.
3. **A comparison basis** — against what: last quarter, a target, a
   competitor, the metric's own history. A number without a comparison is a
   fact; with one, it is a signal.

Why metrics matter
--------------------

Metrics are how goals become **checkable**. "Improve customer loyalty" is a
wish; "raise 90-day retention from 78% to 85% by Q4" is a metric-defined goal
the whole organisation can steer by, measure progress against, and honestly
declare met or missed. Metrics are also the shared vocabulary between analysts
and stakeholders: when both sides agree what *retention* means and how it is
computed, a whole class of talking-past-each-other disappears.

Choosing them well
--------------------

Good metrics share three properties: they **track the actual goal** (not a
convenient proxy for it), they are **hard to game** (a metric people can
inflate without improving anything will be), and they are **few** — a handful
watched seriously beats a wall of numbers watched by no one. The foundations'
warning recurs with force here: what gets measured gets managed, *including
when the measure is wrong*.

The caveat
------------

Every metric compresses, and compression discards. Retention rate says
nothing about *which* customers stayed; average ride length hides the
distribution's shape. Treat metrics as instruments on a dashboard —
indispensable for steering, and always an invitation to look underneath when
one moves strangely. The next lesson is about exactly that dashboard.
"""


MINDMAP.update({
    "The Relationship Between Data and Decision-Making": [
        "Data-Driven Decision-Making",
        "Quantitative and Qualitative Data in Decision-Making",
        "Why Asking the Right Questions Matters in Data Analytics",
        "Using Data Analysis to Choose the Right Advertising Strategy",
    ],
    "Quantitative and Qualitative Data in Decision-Making": [
        "The Relationship Between Data and Decision-Making",
        "Data Creates Value Only When It Is Communicated",
        "Understanding Common Problem Types in Data Analytics",
        "The Difference Between Data and Metrics, and the Role of Metrics",
    ],
    "Data Creates Value Only When It Is Communicated": [
        "Quantitative and Qualitative Data in Decision-Making",
        "Sharing Data to Drive Impact",
        "Clear Communication with Stakeholders and Teams",
        "Data Storytelling: Giving Numbers a Clear and Convincing Voice",
    ],
    "The Difference Between Data and Metrics, and the Role of Metrics": [
        "Data Creates Value Only When It Is Communicated",
        "Dashboards",
        "Mathematical Thinking",
        "The Relationship Between Data and Decision-Making",
    ],
})


# ======================================================================
# Section 2 — DDD / metrics (close) + spreadsheets (open)  (ddd 009-012)
# ======================================================================

GLOSS.update({
    "Dashboards":
        "live metrics in one place: what dashboards are for, and when a report beats one",
    "Mathematical Thinking":
        "step-by-step decomposition, orders of magnitude, and choosing data sized to the decision",
    "Spreadsheets in Data Analysis":
        "where the spreadsheet sits in the decision workflow, phase by phase",
    "Building and Organizing a Spreadsheet":
        "one row per record, clean headers, raw kept raw: layouts that survive analysis",
})

CONTENT["Dashboards"] = r"""
Metrics, made ambient
-----------------------

The previous lesson turned data into metrics; a **dashboard** is where metrics
live. The standard definition draws the key contrast: a dashboard **monitors
live, incoming data** from multiple datasets, organised in one central
location — while a **report** is a **static collection of data** delivered
periodically. A report is a photograph; a dashboard is a window.

What a dashboard is for
-------------------------

Dashboards serve the *ongoing* relationship with a metric: is retention
holding, are today's orders on pace, did the error rate move after the deploy?
Their value is threefold. **Currency** — the numbers are now, not last month.
**Centralisation** — the handful of metrics that matter, together, instead of
scattered across systems. **Shared truth** — everyone steering by the same
instruments, the workplace benefit from the foundations made literal.

Dashboard or report?
----------------------

The choice follows the decision's tempo. **Continuous decisions** (operations,
monitoring, campaigns in flight) want a dashboard: the question recurs, so the
answer should stand ready. **Periodic or one-off decisions** (the quarterly
review, the pricing study) want a report: a curated, stable snapshot with
narrative, where numbers do not shift under the reader mid-discussion. Teams
misfire in both directions — dashboards nobody opens standing in for analysis
that was never done, and hand-built weekly reports that a dashboard would
automate.

Designing one that earns its screen
-------------------------------------

Four habits separate working dashboards from decoration. **Few metrics,
chosen** — the handful from the metrics lesson, not everything measurable.
**Comparison built in** — each number against its target or history, since a
lone value is not a signal. **Hierarchy** — the decision-critical figure large
and first; supporting detail below. **A named audience** — one dashboard per
decision-making group beats one dashboard for everyone. (The visualization
section later covers the craft of the charts themselves; a dedicated lesson
there returns to dashboards as products.)

The caveat
------------

A dashboard shows *what* is happening, never *why* — it is a smoke detector,
not an investigation. Its glanceability also breeds false confidence: a
metric can be green while its definition has quietly rotted. Treat every
surprising dashboard movement as the start of an analysis, and audit the
definitions behind the tiles on a schedule, because screens age faster than
they look.
"""

CONTENT["Mathematical Thinking"] = r"""
Thinking in structure and size
--------------------------------

**Mathematical thinking** here does not mean advanced mathematics. It means
looking at a problem and **breaking it down step by step**, so relationships
and patterns become visible — the same decomposition habit as the technical
mindset, applied to quantities. Two everyday forms of it do most of the work
in analytics: decomposing calculations, and reasoning about size.

Decomposition: the arithmetic of questions
--------------------------------------------

Most business quantities are compositions of simpler ones, and writing the
composition down is often the analysis. *Revenue = customers × orders per
customer × average order value.* Revenue fell 12%? The decomposition converts
one mystery into three checks — which factor moved? *Cost per acquisition =
spend ÷ new customers*; *retention = remaining ÷ starting*. Each formula is a
small model of the business, and disagreements about "why the number moved"
become inspections of named terms instead of debates.

Orders of magnitude: the sanity check
---------------------------------------

The second form is estimating **roughly how big** things should be before
computing exactly. If the dashboard says yesterday's revenue was 4.2 million
and quick mental math says ~40 thousand customers × ~$10 average order ≈
$400k, something is off by 10× — a units error, a duplicated join, a decimal
slip. Rough estimation catches these instantly, which is why experienced
analysts approximate *first* and compute *second*. The habit also scopes
work: an effect worth at most $5k a year does not merit a month of analysis.

Matching data size to decision size
-------------------------------------

Mathematical thinking includes choosing the *scale* of data a decision needs.
**Small data** — a small number of specific metrics over a short period — is
effective for day-to-day decisions: this week's staffing, this campaign's
pacing. **Big data** — larger, broader, less immediately specific — serves
more substantial decisions: market strategy, long-term trends. Reaching for
big-data machinery to answer a small-data question wastes time; deciding a
strategic question from one week's numbers mistakes noise for signal.

The caveat
------------

Decompositions are models, and models simplify: *customers × frequency ×
value* hides that the three factors interact (a discount raises frequency and
lowers value at once). Use the decomposition to locate where to look, then
look at the real data — the formula points, the data answers. Next, the tool
in which most of this thinking first happens: the spreadsheet.
"""

CONTENT["Spreadsheets in Data Analysis"] = r"""
The spreadsheet's seat at the decision table
----------------------------------------------

The foundations introduced spreadsheets as a tool; this stage places them in
the **decision workflow**. For data-driven decisions of everyday scale — the
small-data tier of the previous lesson — the spreadsheet is frequently the
entire pipeline: where the data lands, where the metric is computed, and what
the stakeholder opens. Knowing *where in the process* it serves keeps its use
deliberate rather than habitual.

Phase by phase
----------------

Mapped onto the six phases, the spreadsheet contributes to each:

- **Ask** — a scratchpad for the decomposition: sketching *revenue =
  customers × frequency × value* in cells makes the question concrete.
- **Prepare** — the landing zone for extracts and collected data, where a
  first eyeball happens.
- **Process** — first-pass cleaning: spotting blanks, fixing obvious typos,
  standardising labels (the cleaning section deepens this greatly).
- **Analyze** — sorting, filtering, formulas, and pivot tables that turn rows
  into the comparison the decision needs.
- **Share** — a labelled summary table or chart any stakeholder can open, no
  special tools required.
- **Act** — often the humble tracker where the decision's follow-through is
  recorded, closing the loop.

Its comparative advantage
---------------------------

Against the rest of the toolkit, the spreadsheet's edge is **transparency plus
ubiquity**: every intermediate value is visible and clickable, and every
collaborator already has the tool. That makes it unbeatable for work that must
be *inspected* by non-analysts — a budget model a manager will poke at, a
shared tracker a team maintains. The corresponding costs are the familiar
ones: scale limits, silent overtyping, and no audit trail — which is why the
same workflow graduates to SQL and Python as data grows and repetition sets
in.

The caveat
------------

The spreadsheet's danger is precisely its convenience: analyses that deserved
a durable, reviewable pipeline live for years as ``final_v3_REAL.xlsx``. A
useful rule: the *first* pass at almost anything belongs in a spreadsheet; the
*tenth* pass belongs in something rerunnable. The next lessons make the first
pass excellent — starting with the layout decisions that determine whether a
sheet helps or fights you.
"""

CONTENT["Building and Organizing a Spreadsheet"] = r"""
Layout is destiny
-------------------

Whether a spreadsheet analysis takes ten minutes or a wrestling match is
usually decided before any formula is written — by the **layout**. A
well-organised sheet makes sorting, filtering, pivoting, and charting
one-step operations; a poorly organised one fights every move. The
organising principles are few and learnable in one sitting.

The rules of a workable sheet
-------------------------------

- **One row, one record.** Each row is one observation — one order, one
  customer, one day. Never let a single record sprawl across several rows, or
  two records share one.
- **One column, one attribute.** Each column holds exactly one kind of value,
  consistently typed: dates in the date column, numbers unmixed with text
  (no ``"12 approx"``).
- **Headers in row one, once.** Short, descriptive, unique names —
  ``order_date``, ``region``, ``amount`` — with no merged cells and no
  repeated header blocks mid-sheet. Headers are the context every later tool
  reads.
- **No layout-as-data.** Colour, bolding, and merged cells communicate to
  eyes but are invisible to formulas, sorts, and pivots. If a distinction
  matters, give it a **column** (``status``), not a highlight.
- **Raw stays raw.** Keep the untouched data on its own tab; do cleaning and
  analysis on copies or in adjacent tabs. When (not if) a step goes wrong,
  the original is still there.
- **A notes tab.** Where the data came from, when, and what was changed — the
  chain of custody, in its cheapest form.

Why these rules are the rules
-------------------------------

They all serve one principle: **the sheet should be readable by software, not
only by people**. Sorting breaks on merged cells; pivots break on repeated
headers; filters break on mixed types. The "one row per record, one column
per attribute" shape is exactly what every downstream tool — pivot tables,
SQL imports, pandas — expects, so a sheet built this way flows into the rest
of the toolkit unchanged. (Data folk call this *tidy* structure; the prep
section's wide-versus-long lesson formalises it.)

A worked contrast
-------------------

The monthly sales workbook, done badly: one tab per month, merged title rows,
totals typed between data rows, regions distinguished by cell colour. Done
well: a single ``sales`` tab — ``date, region, product, units, revenue`` —
one row per sale, plus a ``summary`` tab of formulas and a ``notes`` tab. The
first requires manual surgery for any cross-month question; the second
answers "revenue by region by quarter" with one pivot table.

The caveat
------------

These rules organise *analysis* sheets. Presentation sheets — the formatted
summary a stakeholder reads — legitimately use merged headers and colour,
because their reader is human. The discipline is keeping the two roles on
**separate tabs**: analyse in tidy structure, present in formatted views
built from it, and never let the presentation copy become the working data.
"""


MINDMAP.update({
    "Dashboards": [
        "The Difference Between Data and Metrics, and the Role of Metrics",
        "Data Dashboards: Organizing Insight for Real-Time Decision Making",
        "The Role and Importance of Data Visualization",
        "Mathematical Thinking",
    ],
    "Mathematical Thinking": [
        "Analytical Thinking and Its Core Components",
        "The Difference Between Data and Metrics, and the Role of Metrics",
        "Dashboards",
        "Spreadsheets in Data Analysis",
    ],
    "Spreadsheets in Data Analysis": [
        "The Role of Spreadsheets in Data Analysis and Basic Concepts",
        "Building and Organizing a Spreadsheet",
        "How Data Analysts Use Spreadsheets",
        "Mathematical Thinking",
    ],
    "Building and Organizing a Spreadsheet": [
        "Spreadsheets in Data Analysis",
        "Spreadsheet Calculations with Formulas",
        "Common Spreadsheet Errors and How to Fix Them",
        "Sorting and Filtering Data in Spreadsheets",
    ],
})


# ======================================================================
# Section 2 — DDD / spreadsheets (close)  (ddd 013-016)
# ======================================================================

GLOSS.update({
    "How Data Analysts Use Spreadsheets":
        "the everyday spreadsheet toolkit — sort, filter, pivot, chart — mapped to real tasks",
    "Spreadsheet Calculations with Formulas":
        "formulas, cell references, and the absolute-vs-relative distinction that makes fill-down work",
    "Common Spreadsheet Errors and How to Fix Them":
        "reading #DIV/0!, #VALUE!, #REF!, #NAME?, #N/A — and fixing causes, not symptoms",
    "Spreadsheet Functions":
        "named operations — SUM, AVERAGE, IF, VLOOKUP — the analyst's core vocabulary",
})

CONTENT["How Data Analysts Use Spreadsheets"] = r"""
The everyday toolkit
----------------------

Beyond holding data, a spreadsheet is a small analysis workbench, and analysts
lean on a compact set of its features constantly. This lesson surveys that
working toolkit — what each feature does and the task it serves — so the
formula and function lessons that follow have a home.

The core moves
----------------

- **Sorting** orders rows by a column — largest sales first, newest dates on
  top — to surface extremes and impose meaningful order. (A dedicated lesson
  in the analysis section treats sorting and filtering in depth.)
- **Filtering** hides rows that fail a condition, so you see only what matters
  right now — one region, orders above a threshold, this month.
- **Formulas** compute new values from existing cells, deriving the columns and
  totals a raw extract lacks.
- **Functions** are named, prebuilt calculations used inside formulas — the
  vocabulary the last lesson of this stage expands.
- **Pivot tables** summarise a table by grouping and aggregating —
  ``revenue by region by quarter`` from thousands of rows, with a few drags.
  They are the spreadsheet's most powerful analysis feature, and the analysis
  section returns to them repeatedly.
- **Conditional formatting** colours cells by rule (overdue in red, top
  performers in green) to make patterns jump out visually.
- **Charts** turn a range into a bar, line, or scatter for exploration or a
  stakeholder summary.

A task-to-tool map
--------------------

The features map cleanly onto everyday questions. *"Which stores
underperformed?"* — filter to the period, sort by sales, eyeball the bottom.
*"What's revenue by region?"* — pivot table. *"Which orders need attention?"* —
conditional formatting flags them. *"Show the trend"* — a line chart. Fluency
is knowing, on hearing the question, which feature answers it fastest.

Where it fits the workflow
----------------------------

These moves cover the first, exploratory pass of most analyses — fast enough
to try several angles in an afternoon, transparent enough that a stakeholder
can watch. The cleaning and analysis sections rebuild each of them at greater
depth and scale (and in SQL and Python), but the spreadsheet versions are
where the intuitions form.

The caveat
------------

Every one of these features assumes the tidy layout of the previous lesson:
sorting scrambles data if rows are not independent records, pivots break on
merged headers, filters mislead on mixed types. The toolkit is only as good as
the sheet it runs on — which is why layout came first.
"""

CONTENT["Spreadsheet Calculations with Formulas"] = r"""
Formulas: computation you can read
------------------------------------

A **formula** is an expression, beginning with ``=``, that computes a value
from other cells. Formulas are what make a spreadsheet a calculator rather than
a table: change an input and every dependent formula recomputes instantly. This
lesson covers the mechanics that trip up beginners — chiefly how cell
references behave when a formula is copied.

The anatomy of a formula
--------------------------

``=B2*C2`` multiplies two cells. ``=(B2-C2)/C2`` computes a percentage change.
Formulas combine **cell references** (``B2``), **operators**
(``+ - * /``), **numbers**, and **functions** (next lesson). The power is the
reference: ``=B2*C2`` does not mean "6" — it means "whatever is in B2 times
whatever is in C2", so the sheet stays live.

The distinction that matters most: relative vs absolute
---------------------------------------------------------

When you copy a formula down a column, its references **move** by default —
this is a **relative reference**. Copy ``=B2*C2`` from row 2 to row 3 and it
becomes ``=B3*C3``, which is usually exactly what you want: revenue per row,
computed for every row by writing the formula once and filling down.

Sometimes a reference must **not** move — a single tax rate in ``E1`` that
every row multiplies by. Freeze it with **dollar signs**: ``$E$1`` is an
**absolute reference** that stays fixed no matter where the formula is copied.
The ``$`` before the column letter locks the column; the ``$`` before the row
number locks the row; you can lock one and not the other (``$E1`` or ``E$1``)
for row- or column-only anchoring.

.. code-block:: text

   A2:  =B2*$E$1     -> fill down -> B3*$E$1, B4*$E$1, ...   (rate stays E1)
   A2:  =B2*E1       -> fill down -> B3*E2,   B4*E3,  ...    (rate drifts -- a bug)

Getting this wrong is one of the most common spreadsheet bugs: a fill-down that
silently walks the "fixed" reference down the sheet, producing numbers that
look plausible and are wrong.

Building calculations that hold up
------------------------------------

Two habits prevent trouble. **Reference, don't retype**: put the tax rate in a
cell and reference ``$E$1``, so changing it updates everything — a value typed
into fifty formulas is fifty places to miss. And **build in steps**: a column
for subtotal, another for tax, another for total, rather than one monster
formula — each step is inspectable, and the order-of-magnitude sanity check
from the mathematical-thinking lesson has somewhere to land.

The caveat
------------

Formulas are invisible once entered — the cell shows the result, not the logic
— so errors hide behind reasonable-looking numbers. That invisibility is why
the next lesson is entirely about *errors*: recognising when a formula has gone
wrong, and why.
"""

CONTENT["Common Spreadsheet Errors and How to Fix Them"] = r"""
Errors are messages, not failures
-----------------------------------

When a spreadsheet shows ``#DIV/0!`` or ``#REF!``, it is not broken — it is
**telling you something specific**. Each error code names a distinct cause, and
reading the code is the fastest route to the fix. This lesson decodes the ones
you will actually meet, then covers the errors that show *no* code at all —
which are the dangerous kind.

The coded errors, decoded
---------------------------

- ``#DIV/0!`` — **division by zero**. A formula divided by an empty or
  zero-valued cell (common in averages when the count is zero). Fix: ensure the
  denominator is non-zero, or handle the empty case explicitly.
- ``#VALUE!`` — **wrong data type**. A formula expected a number but got text —
  ``="ten"+5``, or a numeric column contaminated with ``"12 approx"``. Fix:
  correct the offending value's type.
- ``#REF!`` — **invalid reference**. The formula points at a cell that no
  longer exists, usually because a row or column it referenced was deleted, or
  a relative formula was copied off the edge of the data. Fix: restore the
  reference or repair the formula (absolute references prevent the copy
  variant).
- ``#NAME?`` — **unrecognised name**. A misspelled function or range —
  ``=SUMM(...)`` instead of ``=SUM(...)``. Fix: correct the spelling; the
  editor's autocomplete prevents most of these.
- ``#N/A`` — **not available**. A lookup (``VLOOKUP`` and friends) could not
  find its target. Often legitimate — the value genuinely is not there — and
  often caused by stray spaces or a mismatched type. Fix: confirm the lookup
  value exists and matches exactly.

Suppress with care
--------------------

Functions like ``IFERROR`` can replace an error with a blank or a message —
``=IFERROR(B2/C2, 0)``. This is right for errors you *expect and understand*
(an average over a genuinely empty group). It is dangerous as a reflex:
wrapping everything in ``IFERROR`` **hides** structural problems — a ``#REF!``
that signals broken logic gets silently turned into a zero, and the wrong
number propagates. Diagnose first; suppress only what you have understood.

The errors with no code
-------------------------

The coded errors are the *safe* ones — the spreadsheet flagged them. The
genuinely dangerous errors show a perfectly normal-looking number:

- a **relative reference that drifted** on fill-down (previous lesson),
  computing against the wrong cells;
- a **wrong range** — ``=SUM(B2:B99)`` when data runs to B100, silently
  dropping the last row;
- a **duplicate or off-by-one** in a manual selection;
- **mixed units** summed together (dollars and euros, hours and minutes).

None raises an error. All produce confident, wrong output — which is why the
order-of-magnitude sanity check and, later, systematic **verification** exist:
the spreadsheet cannot warn you about the mistakes it cannot see.

The caveat
------------

A visible error code is a gift — it points at its own cause. Train yourself to
find silent errors *before* trusting a number: does the total match a rough
estimate, does the row count look right, do spot-checked cells recompute by
hand? The coded errors teach the vocabulary; the discipline is hunting the
uncoded ones.
"""

CONTENT["Spreadsheet Functions"] = r"""
The analyst's vocabulary
--------------------------

A **function** is a named, prebuilt operation you use inside a formula:
``=SUM(B2:B100)`` instead of ``=B2+B3+...+B100``. Functions are the
spreadsheet's vocabulary — each is a verb the analyst can call — and a modest
core covers the large majority of everyday work. This lesson names that core;
the cleaning and analysis sections expand it substantially.

The core, by job
------------------

- **Aggregating** — collapse many values to one. ``SUM`` (total), ``AVERAGE``
  (mean), ``COUNT`` (how many numbers), ``COUNTA`` (how many non-empty),
  ``MIN`` / ``MAX`` (extremes), ``MEDIAN``. These answer "how much / how many /
  how typical" for a whole column.
- **Conditional aggregating** — the same, but only for rows meeting a
  condition. ``COUNTIF`` / ``SUMIF`` / ``AVERAGEIF`` ("count orders over
  $100", "sum revenue for the north region") — the workhorses the analysis
  section devotes a full lesson to.
- **Logical** — decisions inside a cell. ``IF(test, then, else)`` returns one
  value or another; ``IFERROR`` handles the errors of the previous lesson.
- **Text** — manipulate strings. ``LEN`` (length), ``LEFT`` / ``RIGHT``
  (extract characters), ``FIND`` (locate a substring), ``CONCATENATE`` /
  ``&`` (join), ``TRIM`` (strip stray spaces) — indispensable when cleaning
  messy labels.
- **Lookup** — pull a value from another table by matching a key. ``VLOOKUP``
  and its modern successors join data across sheets — powerful, error-prone,
  and given full treatment (with troubleshooting) in the analysis section.

The pattern of a function call
--------------------------------

Every function has the same shape: a **name**, then **arguments** in
parentheses — ``=FUNCTION(argument1, argument2, ...)``. ``=SUMIF(B2:B100,
">100", C2:C100)`` reads: over the range B2:B100, for rows greater than 100,
sum the matching cells of C2:C100. Learn to read the argument list and every
function — including ones you have never seen — becomes decipherable from its
name and its inputs.

.. code-block:: text

   =SUM(E2:E100)                 total revenue
   =AVERAGE(E2:E100)             mean order value
   =COUNTIF(D2:D100, "North")    number of northern orders
   =IF(E2>100, "large", "small") tag each order by size
   =LEN(A2)                      length of the text in A2

Functions across the toolkit
------------------------------

These are not spreadsheet trivia — they are the *same operations* you will meet
again as SQL aggregate functions (``SUM``, ``COUNT``, ``AVG``) and pandas
methods (``.sum()``, ``.mean()``, ``.groupby()``). Learning the vocabulary here,
where every intermediate value is visible, builds intuition that transfers
directly when the tools scale up. This closes the spreadsheet stage; the next
turns from computing on data to the human side of decisions — stakeholders and
communication.

The caveat
------------

Functions make wrong answers as fluently as right ones: ``=AVERAGE(B2:B99)``
computes a flawless mean of the wrong range. The function guarantees the
*operation*, never that it was the operation you needed on the data you meant —
so the sanity-check and verification habits apply to every function call, no
matter how simple it looks.
"""


MINDMAP.update({
    "How Data Analysts Use Spreadsheets": [
        "Building and Organizing a Spreadsheet",
        "Spreadsheet Calculations with Formulas",
        "Sorting and Filtering Data in Spreadsheets",
        "Spreadsheet Functions",
    ],
    "Spreadsheet Calculations with Formulas": [
        "How Data Analysts Use Spreadsheets",
        "Spreadsheet Functions",
        "Common Spreadsheet Errors and How to Fix Them",
        "Building and Organizing a Spreadsheet",
    ],
    "Common Spreadsheet Errors and How to Fix Them": [
        "Spreadsheet Calculations with Formulas",
        "Spreadsheet Functions",
        "Data Cleaning with Spreadsheets",
        "How Data Analysts Use Spreadsheets",
    ],
    "Spreadsheet Functions": [
        "Spreadsheet Calculations with Formulas",
        "Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets",
        "Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)",
        "Using Spreadsheet Functions for Data Cleaning",
    ],
})


# ======================================================================
# Section 2 — DDD / Stage: execution  (ddd 017-020)
# ======================================================================

GLOSS.update({
    "Defining the Problem Domain":
        "scoping before analysing: problem, stakeholders, deliverables, timeline, success",
    "Context and Bias in Data Analysis":
        "context makes data meaningful; unexamined context is where bias enters",
    "Stakeholder Expectations in Data Analysis":
        "who has a stake, what they need, and how to align before the work begins",
    "Staying Focused on the Project Objective":
        "guarding scope: keeping every step tied to the one question that matters",
})

CONTENT["Defining the Problem Domain"] = r"""
Scope before analysis
------------------------

The framing stage taught how to turn a situation into a question; this final
stage of the section is about running the resulting project without it
drifting, dying in delivery, or colliding with the people who commissioned it.
It opens with **scoping** — defining the problem domain — because the cheapest
project failures are the ones prevented before any data is pulled.

What a scope defines
----------------------

A scope statement pins down, in writing and up front, the elements a data
project needs to succeed. The standard set:

- **The problem and objective** — the specific question, stated as a SMART
  goal, and the decision waiting on it.
- **The stakeholders** — who commissioned the work, who will act on it, who
  must be consulted (the next lesson develops this).
- **The data** — what data can answer the question, and whether it exists and
  is accessible.
- **The deliverables** — the tangible outputs: a report, a dashboard, a
  recommendation, a model, each with acceptance criteria.
- **The timeline** — milestones and a deadline, aligned to the decision's
  own clock.
- **Success criteria** — how everyone will recognise "done" and "good".

Written down, these become a shared reference the whole project checks itself
against.

Why scoping prevents the expensive failures
----------------------------------------------

Every element blocks a specific disaster. An undefined **objective** yields a
precise answer to the wrong question. Unnamed **stakeholders** surface late
with veto power. Unconfirmed **data** wastes weeks before someone discovers the
needed field was never collected. Vague **deliverables** invite endless
"can you also…". A missing **timeline** lets the decision pass before the
analysis lands. Undefined **success** guarantees the argument at the end about
whether the work was any good. Scoping is an hour that routinely saves a
month.

The timeline reality check
----------------------------

One check deserves emphasis: compare the *project* timeline against the
*decision* timeline. If the decision must be made Friday and the analysis needs
two weeks, no amount of skill fixes the mismatch — the scope and expectations
must be reset now, to a smaller question answerable in time. Discovering this
at the start is planning; discovering it at the deadline is failure.

The caveat
------------

Scopes should be firm, not frozen. Real analysis genuinely discovers that the
question was mis-posed or the data richer than expected, and a good scope can
be **renegotiated** — deliberately, with stakeholders, in the open. What
scoping prevents is not change but *unmanaged* change: silent scope creep and
end-of-project surprises. The next lessons handle the two forces that most
often reshape a scope — hidden bias in the data, and the expectations of the
people around it.
"""

CONTENT["Context and Bias in Data Analysis"] = r"""
Context makes data mean something
-----------------------------------

**Context** is the condition in which data exists — the who, what, when, where,
and how of its creation. The foundations showed context is what makes a value
meaningful; this lesson shows the sharper edge: **unexamined context is where
bias enters an analysis**. The same data yields opposite conclusions depending
on context the analyst did or did not establish.

The questions that establish context
---------------------------------------

Before trusting any dataset, five questions locate it:

- **Who** collected it, and who is *in* it — which populations are represented,
  and which are silently missing?
- **What** exactly does each field measure, in what units, under what
  definition?
- **When** was it collected — and is that period representative, or a holiday,
  an outage, a boom?
- **Where** did it come from — which systems, regions, channels?
- **How** was it gathered — self-reported, sensor-logged, sampled how?

A satisfaction score of 4.5 means one thing from a representative survey and
another from one answered only by the delighted and the furious. Context is not
background; it is half the meaning.

Where context failures become bias
------------------------------------

**Bias** is a preference in favour of or against a thing, person, or group,
and unexamined context is its commonest doorway. The specific mechanisms (which
the prep section dissects) all trace to a context question skipped:

- **Sampling bias** — the data over-represents some groups (the *who* went
  unasked).
- **Historical bias** — the data faithfully records a biased past and the
  analysis projects it forward (the Amazon recruiting case from the
  foundations).
- **Selection and survivorship bias** — only certain cases made it into the
  data (the *what got captured* went unasked).
- **Confirmation bias** — the analyst's own preference, steering which
  questions get asked and which results get scrutinised.

Notice the last one is about the *analyst*, not the data: context includes
your own position and expectations, which shape the analysis as surely as the
data's origin does.

Establishing context in practice
----------------------------------

Two cheap habits. **Interrogate provenance** on arrival — walk the data-life-
cycle backward (who planned, captured, managed it?) before computing.
**Disaggregate and compare** — check results across relevant groups and time
periods, because bias hides in aggregates and surfaces in breakdowns. Neither
needs special tools; both need the decision that context is part of the job.

The caveat
------------

Perfect context is unattainable — you rarely know everything about how data
was made. The professional standard is not omniscience but **honesty about the
limits**: stating what context you established, what you could not, and how
that bounds the conclusion. An analysis that names its context gaps is trusted
far longer than one that hides them, which is exactly the fairness obligation
from the foundations, operating at the project level.
"""

CONTENT["Stakeholder Expectations in Data Analysis"] = r"""
The people with a stake
-------------------------

**Stakeholders** are the people who have invested in, or will be affected by, a
project and hold a stake in its outcome. Analysis does not happen in a vacuum:
someone requested it, someone will act on it, someone controls the data, and
someone's work will change because of it. Managing these people's expectations
is as decisive to a project's success as the analysis itself — brilliant work
that ignores its stakeholders routinely fails, while modest work aligned with
them succeeds.

Who the stakeholders are
--------------------------

Typical roles on a data project:

- **The project sponsor** — who commissioned and funds the work, and to whom
  it ultimately answers. Their question is the north star.
- **Primary stakeholders** — who will *act* on the findings: the marketer who
  reallocates budget, the operations lead who changes a process.
- **Secondary stakeholders** — consulted or affected but not deciding: the
  engineers who own the data pipeline, the team whose numbers are under study.
- **Data owners** — who control access to the data you need, and whose
  cooperation you require early.

A five-minute stakeholder map — who decides, who acts, who must be consulted,
who owns the data — prevents the classic late-project ambush by someone with
authority nobody thought to include.

What expectations to align
----------------------------

Misalignment usually hides in four unspoken assumptions, best surfaced at the
start:

- **The question** — do you and the sponsor mean the same thing by it? Play it
  back in your own words and watch for the correction.
- **The deliverable** — a one-page recommendation, or a self-serve dashboard?
  Wildly different work.
- **The timeline** — when is the answer needed to be useful, versus when it is
  merely wanted?
- **The certainty** — do they expect a definitive answer the data cannot give,
  or will directional evidence suffice? Calibrate this early or disappoint at
  the end.

The sponsor-with-an-answer problem
------------------------------------

A specific hazard: sometimes a sponsor already has a **predetermined answer**
and wants the analysis to endorse it. This is where the analyst's independence
matters most. The professional move is neither to rubber-stamp nor to pick a
fight, but to commit up front to reporting *what the data shows* — establishing,
while expectations are being set, that the analysis is a genuine test and not a
justification exercise. The fairness thread again: analysis should inform
decisions, not launder them.

The caveat
------------

Stakeholder management is not people-pleasing. The goal is *alignment* — shared
understanding of question, deliverable, timeline, and the honest limits of what
the data can say — not agreement with whatever anyone wants to hear. The next
two lessons are about holding that alignment steady once the work is underway:
staying focused on the objective, and communicating clearly.
"""

CONTENT["Staying Focused on the Project Objective"] = r"""
The objective is the compass
------------------------------

A scoped project with aligned stakeholders can still fail one slow way: by
**drifting**. Interesting tangents, mid-stream requests, and the analyst's own
curiosity gradually pull the work away from the question it was meant to
answer. Staying focused on the objective — the discipline against
**scope creep** — is what keeps a project delivering the thing it was
commissioned to deliver.

How projects lose focus
-------------------------

Drift is rarely a single bad decision; it is an accumulation of reasonable
ones:

- **The fascinating tangent** — the data reveals something intriguing but
  irrelevant, and hours vanish chasing it.
- **The creeping request** — "while you're in there, could you also…",
  repeated until the project is three projects.
- **The rabbit hole** — a small anomaly pursued far past its importance.
- **The moving target** — the objective itself quietly shifts as stakeholders'
  interest wanders, without anyone deciding to change it.

Each feels productive in the moment. Their sum is a project that consumed its
time and answered a different, larger, blurrier question than the one that
mattered.

The discipline of focus
-------------------------

Three habits hold the line. **Reread the objective** — literally return to the
scoped question at each phase and ask whether the current work serves it; the
cheapest project-management tool an analyst owns. **Park, don't chase** —
capture interesting tangents in a "later" list instead of following them now,
which honours the curiosity without derailing the work. **Renegotiate
explicitly** — when a new request has real merit, take it back to stakeholders
as a *scope change* with its own cost and timeline, rather than silently
absorbing it. The distinction is the whole game: managed change is planning,
unmanaged change is creep.

Focus versus discovery
------------------------

Focus is not blinkeredness. Genuine discovery — the data revealing that the
objective was mis-posed, or that a bigger issue lurks — is one of analysis's
highest values, and focus must not suppress it. The reconciliation is *where
the decision is made*: a discovery worth pursuing is worth taking to
stakeholders as a deliberate change of direction, in the open. What focus
forbids is not new directions but **undeliberate** ones — drifting somewhere
new without anyone choosing to.

The caveat
------------

Rigid focus on a *wrong* objective is its own failure — faithfully answering a
question that has ceased to matter because the situation moved. The objective
is a compass, not an anchor: it keeps the work oriented, and it can be reset
when reality changes, provided the reset is a decision rather than a drift. The
next lessons turn to the communication skills that make all of this — alignment,
focus, honest limits — actually land with the people who must act.
"""


MINDMAP.update({
    "Defining the Problem Domain": [
        "The Role of Business Tasks in Data Analysis",
        "Why Asking the Right Questions Matters in Data Analytics",
        "Stakeholder Expectations in Data Analysis",
        "Staying Focused on the Project Objective",
    ],
    "Context and Bias in Data Analysis": [
        "Fairness in Data Analysis",
        "Understanding Bias in Data Analysis",
        "Common Types of Data Bias",
        "Defining the Problem Domain",
    ],
    "Stakeholder Expectations in Data Analysis": [
        "Defining the Problem Domain",
        "Managing Stakeholder Expectations and Project Constraints",
        "Clear Communication with Stakeholders and Teams",
        "Staying Focused on the Project Objective",
    ],
    "Staying Focused on the Project Objective": [
        "Defining the Problem Domain",
        "Stakeholder Expectations in Data Analysis",
        "Managing Stakeholder Expectations and Project Constraints",
        "Balancing Speed and Accuracy in Data Analysis",
    ],
})


# ======================================================================
# Section 2 — DDD / execution (cont.)  (ddd 021-024)
# ======================================================================

GLOSS.update({
    "Clear Communication with Stakeholders and Teams":
        "the five Cs of clarity: making findings land with the people who act on them",
    "Adapting to Communication Expectations at Work":
        "reading the room: matching channel, register, and detail to each audience",
    "Managing Stakeholder Expectations and Project Constraints":
        "the iron triangle of scope, time, and resources — and honest trade-offs",
    "Balancing Speed and Accuracy in Data Analysis":
        "fast-enough vs right-enough: matching rigour to the decision's stakes",
})

CONTENT["Clear Communication with Stakeholders and Teams"] = r"""
Clarity is a skill, not a gift
--------------------------------

The framing stage established that data creates value only when communicated;
this lesson is about the *how* — making a finding land clearly with the people
who must act on it. Clear communication is not eloquence or charm; it is a set
of learnable habits, and for analysts, whose findings are often technical and
whose audiences often are not, it is as decisive as the analysis itself.

Five properties of clear communication
----------------------------------------

A useful checklist frames clear communication as **five Cs**:

- **Clear** — the message is unambiguous; the finding and its implication are
  stated plainly, not buried in caveats or jargon.
- **Concise** — as short as the message allows; every extra chart and sentence
  dilutes the one that matters.
- **Correct** — accurate in fact and in emphasis; a true number framed
  misleadingly still misinforms.
- **Complete** — nothing essential omitted, including the limits and the
  uncertainty; completeness is what separates honesty from spin.
- **Compelling** — engaging enough to hold attention and motivate action;
  a correct message nobody finishes changes nothing.

The five pull against each other — concise versus complete especially — and
the craft is the balance: complete enough to be honest, concise enough to be
read.

Translating for the audience
------------------------------

The analyst's specific challenge is **translation**: converting technical
findings into the audience's language. A stakeholder does not need your query
or your confidence interval's derivation; they need *what you found, what it
means for their decision, and how sure you are*. Lead with that, in their
vocabulary, and offer the technical detail as backup for those who want it —
the notebook behind the one-pager. "We should reallocate 30% of the ad budget
to search, which our data suggests would lower cost-per-customer by roughly a
fifth" communicates; "the CPA regression coefficient on the search channel
was significant" does not.

Communication runs both ways
------------------------------

Clear communication includes **listening**. Playing a stakeholder's request
back in your own words surfaces misunderstanding before it becomes wasted work;
inviting questions reveals the context you lacked. Analysts who treat
communication as broadcast learn less, and mislead more, than those who treat
it as exchange — the same two-way principle the framing stage raised, now as
a working habit.

The caveat
------------

Clarity can be weaponised into false confidence: the clearest possible
presentation of an overstated certainty does the most damage. The obligation is
to make the *true* thing clear, uncertainty included — a clear "the evidence
points this way but is not conclusive" is better communication than a crisp,
tidy overclaim. The next lesson turns from the properties of clarity to
matching them to different audiences.
"""

CONTENT["Adapting to Communication Expectations at Work"] = r"""
One message, many audiences
-----------------------------

The same finding must reach an executive, an engineer, and a marketing team —
and it should not reach them the same way. **Adapting** communication to its
audience and context is what turns a clear message into an *effective* one. The
skill is reading the room: matching the channel, the level of detail, and the
register to who is receiving the message and why.

The dimensions to adapt
-------------------------

- **Channel** — a quick answer belongs in a chat message; a decision belongs in
  a meeting or a document; a monitored metric belongs on a dashboard. Choosing
  the wrong vessel (a five-paragraph email for a yes/no, a Slack line for a
  strategic recommendation) undercuts good content.
- **Depth** — an executive wants the headline and the implication; a fellow
  analyst wants the method and the caveats; an operator wants the specific
  action. Same finding, three depths.
- **Register** — formality, tone, and vocabulary shift with audience and
  culture: the language for a board deck differs from a team stand-up. Reading
  an organisation's norms is part of the job.
- **Timing and cadence** — how often and when people expect to hear from you.
  Some stakeholders want interim updates; others want only the final answer.
  Agreeing this up front (part of the scope) prevents both silence and noise.

Reading expectations
----------------------

Expectations are partly stated and partly cultural. Ask directly where you can
— *"Would you prefer a short summary or the full analysis? A doc or a
meeting?"* — and observe where you cannot: how colleagues communicate reveals
the organisation's norms. New analysts who watch how findings are shared, and
mirror it, integrate faster than those who impose one style on everyone.

Adapting is not diluting
--------------------------

Crucially, adapting the *presentation* must not corrupt the *substance*.
Simplifying for an executive means removing technical detail, not removing
inconvenient findings or the honest uncertainty. The same true message, in
three registers — this is the standard. When "adapting to expectations"
starts to mean "telling people what they want to hear," it has crossed from
communication into its failure mode, which the next lessons on managing
expectations and resolving conflict address directly.

The caveat
------------

Over-tailoring has its own cost: maintaining wildly different versions of one
finding for many audiences is effort, and inconsistencies between versions
breed distrust when audiences compare notes. A practical balance is one honest
core message, presented at the depth and in the channel each audience needs —
varied in packaging, identical in substance.
"""

CONTENT["Managing Stakeholder Expectations and Project Constraints"] = r"""
Every project is constrained
------------------------------

No analysis has unlimited time, unlimited data, and unlimited scope at once.
Projects live inside **constraints**, and managing stakeholder expectations
means being honest about them — negotiating what is achievable rather than
promising what is not, then failing. This is the project-management core of a
data analyst's role, and it is mostly about trade-offs made in the open.

The iron triangle
-------------------

Project constraints classically form a triangle of **scope**, **time**, and
**resources** (people, data, tools), with quality at the centre. The rule is
that you cannot freely maximise all three: change one and another must give.

- Want it **faster** (less time)? Narrow the scope or add resources.
- Want it **broader** (more scope)? Allow more time or more resources.
- Fixed **resources and time**? The scope is what it is — a bigger question
  cannot be answered well in the same envelope.

Pretending the triangle does not bind is the root of most broken commitments.
A stakeholder who wants a comprehensive analysis, by Friday, from one analyst,
is asking for a corner of the triangle that does not exist — and the analyst's
job is to say so, with options, not to nod and then disappoint.

Managing expectations in practice
-----------------------------------

Three habits keep expectations aligned with reality:

- **Trade-offs as choices, not excuses.** Present constraints as decisions the
  stakeholder gets to make: "In two days I can give you a directional answer on
  the top three regions; a full national analysis needs two weeks. Which serves
  the decision?" This hands them the triangle.
- **Communicate early and often.** Surprises at the deadline destroy trust;
  a heads-up the moment a constraint bites ("the data for Q1 turns out to be
  incomplete — here's what that limits") preserves it. Interim updates are
  expectation management.
- **Under-promise, over-deliver — honestly.** Commit to what you are confident
  you can do; exceeding a realistic commitment builds credibility, while
  missing an optimistic one erodes it. This is not sandbagging but calibrated
  honesty about uncertainty.

The predetermined-answer constraint, revisited
------------------------------------------------

One "constraint" deserves resistance rather than accommodation: a stakeholder
expecting the data to confirm a foregone conclusion. Managing that expectation
means resetting it — committing, up front, to report what the analysis
actually shows. Bending the analysis to fit the desired answer is not
expectation management; it is the fairness failure the foundations warned
against, wearing a project-management costume.

The caveat
------------

Constraints are real, but they are not always fixed — sometimes the right move
is to *challenge* a constraint (secure more time, get access to better data)
rather than accept it. Good expectation management includes knowing which
constraints to work within and which to push back on, and making that case to
whoever controls the constraint. The next lesson examines the most common
trade-off of all: speed against accuracy.
"""

CONTENT["Balancing Speed and Accuracy in Data Analysis"] = r"""
The trade-off analysts make daily
-----------------------------------

The single most frequent tension in analytical work is **speed versus
accuracy**: a fast answer that might be roughly right, or a slow one that is
carefully right. Neither is universally correct. The skill is matching the
balance to the **stakes and the deadline** of the specific decision — and
knowing that "as accurate as possible, always" is itself a failure mode when it
misses the moment the answer was needed.

Why both matter, and conflict
-------------------------------

**Accuracy** is the point of analysis: a wrong answer, however fast, can be
worse than no answer, because it misleads with false confidence. But **speed**
is often what makes an answer *useful*: a perfect analysis delivered after the
decision is made has zero value, no matter how rigorous. The conflict is real
because rigour costs time — more cleaning, more validation, more cross-checks —
and time is exactly what a pending decision may not have.

Matching rigour to stakes
---------------------------

The decision's consequences set the required accuracy:

- **High-stakes, irreversible** decisions — a major investment, a public
  figure, a regulatory filing — justify slowing down for thorough validation.
  The cost of being wrong dwarfs the cost of the extra day.
- **Low-stakes, reversible** decisions — which of two email subject lines,
  a quick directional read — deserve a fast, good-enough answer; over-investing
  in precision here wastes the time high-stakes work needs.
- **Exploratory** questions — "is there anything here worth a closer look?" —
  want speed first; accuracy comes later, *if* the quick look finds something.

A rough answer, clearly labelled as rough, delivered in time to inform the
decision, frequently beats a precise answer that arrives too late — provided
its roughness is stated so no one mistakes it for more than it is.

Techniques for going fast without lying
-----------------------------------------

Speed need not mean sloppiness. **Sanity-check with estimates** (the
order-of-magnitude habit) catches the gross errors that matter most, fast.
**Sample** the data for a quick read before processing all of it. **Time-box**
exploration — an hour to see if a signal exists — before committing to full
rigour. And above all, **state the confidence level**: "this is a rough
first-pass estimate, ±20%, good enough to decide direction but not to commit
budget" is both fast and honest. The sin is not speed; it is *undisclosed*
speed presented as precision.

The caveat
------------

Some accuracy is non-negotiable regardless of deadline — a figure going into a
financial statement, a public claim, a safety decision. Recognising which
category a task falls into is itself part of the judgement: for most everyday
analysis, fast-and-labelled wins; for the irreversible and the high-stakes, no
deadline justifies a number you have not verified. This closes the
project-execution thread; the remaining lessons of the section turn to the
human situations analysts navigate — sharing impact, meetings, and conflict.
"""


MINDMAP.update({
    "Clear Communication with Stakeholders and Teams": [
        "Data Creates Value Only When It Is Communicated",
        "Adapting to Communication Expectations at Work",
        "Stakeholder Expectations in Data Analysis",
        "Data Storytelling: Giving Numbers a Clear and Convincing Voice",
    ],
    "Adapting to Communication Expectations at Work": [
        "Clear Communication with Stakeholders and Teams",
        "Managing Stakeholder Expectations and Project Constraints",
        "Sharing Data to Drive Impact",
        "Effective Meetings",
    ],
    "Managing Stakeholder Expectations and Project Constraints": [
        "Stakeholder Expectations in Data Analysis",
        "Balancing Speed and Accuracy in Data Analysis",
        "Staying Focused on the Project Objective",
        "Defining the Problem Domain",
    ],
    "Balancing Speed and Accuracy in Data Analysis": [
        "Managing Stakeholder Expectations and Project Constraints",
        "The Importance of Clean Data",
        "Verifying and Reporting Data Integrity",
        "Staying Focused on the Project Objective",
    ],
})


# ======================================================================
# Section 2 — DDD / execution (close)  (ddd 025-027)  -- SECTION 2 COMPLETE
# ======================================================================

GLOSS.update({
    "Sharing Data to Drive Impact":
        "the difference between reporting numbers and actually changing a decision",
    "Effective Meetings":
        "purpose, preparation, and follow-up: making the time analysts spend together count",
    "Conflict Resolution in the Workplace":
        "when the data disagrees with a person: disagreeing productively and separating issue from ego",
})

CONTENT["Sharing Data to Drive Impact"] = r"""
From reporting to impact
--------------------------

There is a wide gap between *sharing* data and sharing it to *drive impact*.
Reporting numbers is easy; changing what someone decides is the actual goal.
This lesson — near the end of a section that began by insisting data creates
value only when communicated — is about closing that gap: making the share
land as a decision, not just an FYI.

Report versus impact
----------------------

The distinction is concrete. A **report** presents what the data says. An
**impactful share** presents what the audience should *do* about it, and makes
doing it easy. "Churn rose 4% last quarter" is a report; "Churn rose 4%, driven
almost entirely by users who never finished onboarding — fixing the broken
onboarding email is the highest-leverage response, and here's the evidence" is
built to drive impact. Same finding; one informs, one moves.

What makes a share impactful
------------------------------

- **Lead with the takeaway.** State the finding and its implication first, then
  support it — the audience should grasp the point in the first sentence, not
  the last slide.
- **Make it about the decision.** Frame everything around what the audience
  must choose or do. A finding with no attached action is trivia; a finding
  with a clear "therefore" is a lever.
- **Right audience, right form, right time.** The impact failures from earlier
  in the section — wrong recipient, wrong format, wrong moment — are exactly
  what kills a share's impact. Deliver to the decider, in their medium, before
  the decision closes.
- **Show enough, not everything.** Include the evidence that supports belief
  and action; relegate the rest to backup. A wall of every number you computed
  buries the one that matters.
- **Make the next step obvious.** The easier you make acting on the finding —
  a clear recommendation, a ready option set — the more likely action follows.

The honest-impact obligation
------------------------------

Driving impact is not the same as *winning*. The goal is the *right* decision,
which sometimes means sharing a finding that disappoints the audience,
complicates their plan, or fails to support the answer they wanted. An
impactful share of an inconvenient truth is worth more than a persuasive share
of a convenient error — and the uncertainty must travel with the finding, so
the decision it drives is made with clear eyes. (The visualization section
returns to this as data storytelling and persuasive presentation, with the
craft of the visuals themselves.)

The caveat
------------

"Impact" can curdle into manipulation — using the tools of persuasion to push a
predetermined conclusion past a stakeholder's judgement. The line is the same
one the fairness thread has drawn throughout: communicate to help people
decide well on the evidence, not to engineer the decision you preferred before
the evidence arrived. Impact in service of truth; never truth bent for impact.
"""

CONTENT["Effective Meetings"] = r"""
Meetings are where alignment happens — or dies
------------------------------------------------

Much of an analyst's collaboration runs through meetings: kickoffs that scope
the work, check-ins that keep it aligned, and presentations that share the
result. Meetings are also where enormous time is wasted. Making them
**effective** — purposeful, prepared, and followed up — is a practical skill
that directly serves the alignment and communication this section has been
building.

Before, during, after
-----------------------

Effective meetings are made mostly outside the meeting itself.

- **Before — purpose and preparation.** Every meeting needs a reason a meeting
  is the right tool: a decision to make, a discussion that email cannot carry,
  alignment to build. If the answer could have been an email, it should have
  been. Prepare the materials, circulate context in advance, and know what a
  successful outcome looks like — for an analyst, that often means bringing the
  finding pre-packaged so the meeting spends its time on *implications*, not on
  absorbing numbers.
- **During — focus and participation.** Keep to the purpose (the same
  scope-discipline as a project), draw out the people whose input is needed,
  and surface disagreement rather than letting it go silent. An analyst's
  specific job in the room is to present findings clearly and to answer
  questions honestly — including "we don't yet know."
- **After — follow-up.** A meeting without recorded decisions and owned next
  steps often achieves nothing durable. A short summary — what was decided, who
  does what by when — is what converts a conversation into action, and it is
  frequently the analyst who should capture it.

The analyst in the meeting
----------------------------

Two roles recur. **Presenting**: lead with the takeaway, match depth to the
room (the adapting-communication lesson, applied live), and hold honest ground
under questioning — neither overclaiming nor collapsing. **Contributing**:
bringing evidence to a discussion that might otherwise run on opinion,
which is the analyst's distinctive value in any meeting — the quiet "the data
actually shows…" that redirects a debate.

The caveat
------------

Not everything needs a meeting, and analysts can lose whole days to them. Part
of meeting effectiveness is *declining* or shortening the ones that do not need
to exist, and protecting the focused time that analysis actually requires. The
best meeting is often the one replaced by a clear written update — the written
share this section has emphasised throughout.
"""

CONTENT["Conflict Resolution in the Workplace"] = r"""
When data meets disagreement
------------------------------

Analytical work generates conflict more than its tidy reputation suggests. The
data contradicts a leader's belief; two stakeholders want opposite things; a
colleague disputes your method; a finding threatens someone's project.
Handling these situations — **conflict resolution** — is the final human skill
of this section, and it rests on one move: **separating the issue from the
person**.

Where analytical conflict comes from
--------------------------------------

- **Data versus belief.** The evidence disagrees with what someone
  experienced, assumed, or hoped — the emotionally hardest case, because the
  number implicitly says they were wrong.
- **Competing stakeholders.** Two parties want the analysis to support
  incompatible conclusions, and each reads the ambiguity their way.
- **Method disputes.** A colleague challenges how you defined a metric,
  cleaned the data, or drew the inference — often legitimately.
- **Threatening findings.** A result implies someone's work underperformed or
  their plan is flawed, and the reaction is defensive.

Naming the source is half the resolution: a method dispute is solved with
evidence, a stakeholder clash with negotiation, a threatening finding with
tact — different conflicts need different responses.

Resolving productively
------------------------

- **Separate issue from ego.** Frame disagreement around the question and the
  evidence, not the people — "the data shows X" invites examination; "you're
  wrong" invites war. Make it you-and-the-stakeholder versus the problem, not
  you versus the stakeholder.
- **Lead with the evidence, hold it humbly.** State what the data shows and how
  you know, while remaining genuinely open that your method could be flawed —
  the confidence to present findings *and* the humility to have them
  challenged are not opposites.
- **Seek the shared goal.** Most workplace conflict dissolves when both sides
  return to a common objective — a better decision, a successful project — that
  is larger than the disagreement.
- **Listen for the real objection.** A challenge to your method is sometimes
  really a fear about what the finding implies; hearing the underlying concern
  resolves more than defending the surface point.

The analyst's particular obligation
--------------------------------------

There is a line an analyst must not cross to keep the peace: **changing the
findings to avoid conflict**. When the data disagrees with a powerful
stakeholder, the resolution is to present the evidence honestly and
respectfully — not to quietly soften the number until it stops offending.
Integrity under pressure is the whole point; the fairness thread that has run
through this section ends here, at its hardest test. Disagree productively,
stay open to being wrong on the *method*, but do not falsify the *result*.

The caveat and the close
--------------------------

Not all conflict should be resolved — some disagreements are healthy and
should stay live, and papering over a real methodological doubt for the sake of
harmony is its own failure. The goal is *productive* conflict: disagreement
that improves the work, conducted with enough respect that the working
relationship survives it.

This closes the Data-Driven Decisions section. You have moved from framing a
question, through metrics and spreadsheets, to the human execution that turns
analysis into decisions. The next section steps back to the raw material all of
it depends on: where data comes from, and how to prepare it.
"""


MINDMAP.update({
    "Sharing Data to Drive Impact": [
        "Data Creates Value Only When It Is Communicated",
        "Clear Communication with Stakeholders and Teams",
        "Data Storytelling: Giving Numbers a Clear and Convincing Voice",
        "Structuring a Persuasive Data Presentation: Turning Insights into Story",
    ],
    "Effective Meetings": [
        "Clear Communication with Stakeholders and Teams",
        "Adapting to Communication Expectations at Work",
        "Stakeholder Expectations in Data Analysis",
        "Presentation Skills for Data Analysts: Delivering Insights with Confidence",
    ],
    "Conflict Resolution in the Workplace": [
        "Managing Stakeholder Expectations and Project Constraints",
        "Stakeholder Expectations in Data Analysis",
        "Context and Bias in Data Analysis",
        "Clear Communication with Stakeholders and Teams",
    ],
})


# ======================================================================
# Section 3 — Data Preparation / Stage: types  (prep 001-004)
# ======================================================================

GLOSS.update({
    "How Data Is Generated and Collected":
        "where data comes from: first/second/third-party sources and how it is produced",
    "Choosing the Right Data to Collect":
        "relevance, coverage, and cost: deciding what data a question actually needs",
    "Understanding Data Types and Data Formats":
        "nominal, ordinal, discrete, continuous — and the file formats data travels in",
    "Structured Data and Data Models":
        "structured vs semi- vs unstructured, and the models that give data its shape",
})

CONTENT["How Data Is Generated and Collected"] = r"""
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
"""

CONTENT["Choosing the Right Data to Collect"] = r"""
Not all data is worth collecting
----------------------------------

Once you know where data can come from, the next decision is *which* to collect
or acquire. More data is not automatically better: the right data answers the
question at acceptable cost, while the wrong data — however voluminous — wastes
storage, effort, and attention, and can actively mislead. Choosing well is a
core Prepare-phase judgement, and it flows directly from the question the
framing section taught you to sharpen.

What "the right data" means
-----------------------------

Three properties decide whether candidate data is worth having:

- **Relevance** — does it actually bear on the question? Data that does not
  inform the decision is clutter, no matter how interesting or easy to obtain.
- **Coverage** — does it represent the whole population or period the question
  concerns? Data that covers only part (one region, one season, one channel)
  answers only a partial question, and mistaking it for the whole is how
  sampling bias enters.
- **Quality and granularity** — is it accurate, and recorded at a fine enough
  level to answer the question? Monthly totals cannot answer a question about
  daily patterns; a coarse category cannot answer one about specifics.

Balanced against these benefits is **cost**: collection effort, money, time,
and — for personal data — privacy and ethical obligation. The right choice
maximises relevance and coverage against what the decision can justify
spending.

Sample versus census
----------------------

A recurring choice is whether to collect *everything* (a census) or a
representative *sample*. A well-chosen sample is often faster, cheaper, and
entirely sufficient — the whole logic of sampling is that a representative part
can answer questions about the whole. The critical word is *representative*:
a large but skewed sample is worse than a small balanced one, because size
lends false confidence to a biased picture. (The cleaning section's sampling
lessons develop this with the mathematics.)

Choosing in practice
----------------------

Work backward from the question. State what an answer requires, list the data
that would supply it, then check each candidate for relevance, coverage, and
affordable quality. This backward pass routinely reveals two things: that some
eagerly-collected data is irrelevant to the actual decision, and that some
genuinely needed data is missing and must be collected or acquired. Better to
learn both at the Prepare stage than after weeks of analysis.

The caveat
------------

Choosing data introduces the analyst's own judgement — and therefore the
analyst's own bias — into the dataset before analysis begins. Deciding what to
collect is deciding what the analysis *can* see; excluding a source excludes a
possible finding. The discipline is to choose deliberately and *document the
choice*, so the boundaries of the data are visible to anyone who later relies
on the conclusion.
"""

CONTENT["Understanding Data Types and Data Formats"] = r"""
Two different questions
-------------------------

"What type is this data?" has two distinct meanings, and analysts need both.
One is about the **nature of the values** — are they numbers, categories,
dates? The other is about the **file format** the data travels in — CSV, JSON,
a database table. This lesson separates them, because the first governs what
analysis is *valid* and the second governs how you *load* the data at all.

Value types: the measurement levels
-------------------------------------

The nature of a value determines which operations make sense on it. The
standard classification:

- **Nominal** — named categories with no inherent order: country, product
  colour, payment method. You can count and group them, but "average colour"
  is meaningless.
- **Ordinal** — categories with a meaningful order but no fixed spacing:
  satisfaction ratings (poor/fair/good), t-shirt sizes. You can rank them, but
  the gap between "good" and "fair" is not a defined quantity.
- **Discrete** (quantitative) — countable numbers: orders placed, employees,
  defects. Whole units; you cannot have 2.5 orders.
- **Continuous** (quantitative) — measured numbers on a scale, any value in a
  range: revenue, temperature, duration. Arithmetic and averages are fully
  meaningful.

The practical payoff: the value type dictates the valid summary and chart. You
average continuous data, count nominal data, and never compute a mean of
category labels — the ``#VALUE!``-style mistakes of the spreadsheet lessons
often start as a value-type confusion.

Data formats: the containers
------------------------------

Independently, data arrives in formats — the file structures that hold it:

- **CSV / TSV** — plain text, one row per line, values separated by commas or
  tabs. Simple, universal, the lingua franca of tabular exchange.
- **JSON** — nested key–value structure, good for hierarchical data; the native
  shape of most web APIs.
- **Spreadsheet files** (``.xlsx``) — tabular data plus formatting and formulas.
- **Database tables** — structured, queryable storage (the SQL section's home).
- **XML, Parquet, and others** — further containers for specific needs.

The format determines *how you get the data in* — which tool and which step —
but not what the data *means*; a column of prices is continuous whether it
arrives as CSV or JSON.

Why keep them separate
------------------------

Confusing the two questions causes trouble. Loading a JSON file as if it were
CSV fails at the format level. Averaging a column of postal codes fails at the
value-type level — the load succeeded, the analysis is nonsense. Competent
preparation checks both: *can I read this container*, and *what may I validly
do with these values*.

The caveat
------------

Formats and types blur at the edges. A CSV stores everything as text, so a
column of numbers arrives *looking* like text and must be converted before
arithmetic — a frequent source of the type errors seen earlier. Part of the
Prepare phase is confirming that each column's *actual* type matches its
*intended* type, not merely its appearance. The next lesson goes deeper into
how data is structured overall.
"""

CONTENT["Structured Data and Data Models"] = r"""
How organised is the data?
----------------------------

Beyond individual value types, data has an overall **degree of structure** —
how regularly it is organised — and this determines which tools can work with
it and how much preparation it needs before analysis. The standard three-way
split is one of the most useful classifications in the field.

Structured, semi-structured, unstructured
-------------------------------------------

- **Structured data** is organised into a defined model — rows and columns,
  with a fixed schema saying what each field is. A database table or a tidy
  spreadsheet is structured: every record has the same fields, each of a known
  type. Structured data is the easiest to query, aggregate, and analyse, and
  it is where most of this course operates.
- **Semi-structured data** has organisational markers but no rigid table
  schema — tags or keys that give it shape while allowing records to vary.
  JSON and XML are the classic examples: hierarchical, self-describing, but not
  a fixed grid. Common from web APIs, it usually needs reshaping into
  structured form before tabular analysis.
- **Unstructured data** has no predefined model: free text, images, audio,
  video. It is the most abundant kind of data in the world and the richest, but
  also the hardest to analyse — extracting structure from it (turning reviews
  into themes, images into labels) is often a project in itself.

The progression structured → semi-structured → unstructured runs from
*easiest to analyse, least abundant* toward *hardest to analyse, most
abundant* — which is much of why data work is challenging.

What a data model is
----------------------

A **data model** is the description of how data is organised: what entities
exist, what attributes each has, and how they relate. Structured data has an
explicit model (a table's columns and types, a database's schema); giving
unstructured data a model is precisely the act of structuring it. Models are
usually described at three levels of detail — a high-level **conceptual** view
of the main entities, a **logical** view specifying attributes and
relationships, and a concrete **physical** view of how it is actually stored —
moving from *what the business cares about* down to *how the database holds
it*.

Why structure and models matter for preparation
--------------------------------------------------

The degree of structure sets the preparation workload. Structured data is close
to analysis-ready and needs mainly cleaning. Semi-structured data needs
*reshaping* into tables first. Unstructured data needs *structuring* — often
the largest task of all. Recognising which kind you face, early in the Prepare
phase, tells you how much work stands between the raw data and a valid
analysis — and whether the question is even answerable with the effort
available.

The caveat
------------

The categories are a spectrum, not sharp bins: a spreadsheet full of
free-text notes is nominally structured but practically holds unstructured
content in its cells, and "structured" guarantees a shape, not quality —
perfectly structured data can still be riddled with errors, which is exactly
why cleaning is its own section. Structure tells you how the data is
*organised*, never whether it is *right*. The following lessons move from these
broad kinds down to the concrete shapes data takes in spreadsheets and tables.
"""


MINDMAP.update({
    "How Data Is Generated and Collected": [
        "Understanding the Data Ecosystem",
        "Choosing the Right Data to Collect",
        "Accessing Data: Internal and External Sources",
        "Quantitative and Qualitative Data in Decision-Making",
    ],
    "Choosing the Right Data to Collect": [
        "How Data Is Generated and Collected",
        "Understanding Data Types and Data Formats",
        "Understanding Bias in Data Analysis",
        "The Relationship Between Data and Decision-Making",
    ],
    "Understanding Data Types and Data Formats": [
        "Choosing the Right Data to Collect",
        "Structured Data and Data Models",
        "Data Types in Spreadsheets",
        "Data Tables (Tabular Data)",
    ],
    "Structured Data and Data Models": [
        "Understanding Data Types and Data Formats",
        "Databases and Relational Database Concepts",
        "Data Tables (Tabular Data)",
        "Wide Data vs. Long Data",
    ],
})


# ======================================================================
# Section 3 — Prep / types (close) + bias_ethics (open)  (prep 005-008)
# ======================================================================

GLOSS.update({
    "Data Types in Spreadsheets":
        "text, number, date, boolean — why a spreadsheet cares what type a cell holds",
    "Data Tables (Tabular Data)":
        "the row-and-column table: the workhorse shape of analysable data",
    "Wide Data vs. Long Data":
        "two tidy layouts of the same data, and when each one serves",
    "Understanding Bias in Data Analysis":
        "what bias is, where it hides, and why unbiased data is the goal not the default",
})

CONTENT["Data Types in Spreadsheets"] = r"""
The type behind the cell
--------------------------

The spreadsheet lessons of Section 2 treated cells as holding "values"; here we
get specific about what *kind* of value a cell holds, because a spreadsheet
quietly assigns every cell a **data type**, and that type governs how the cell
behaves — what you can compute, how it sorts, and whether a formula works or
throws an error.

The core spreadsheet types
----------------------------

- **Text** (string) — letters, words, or any characters treated as label rather
  than quantity: names, categories, IDs. Text is left-aligned by default and
  cannot be summed. Numbers *stored as text* (a common import problem) look
  numeric but refuse arithmetic — a frequent cause of the ``#VALUE!`` errors
  from earlier.
- **Number** — numeric values you can calculate with: integers and decimals,
  right-aligned by default. Currency and percentage are number *formats* — the
  underlying value is a number, displayed with a symbol or scaled by 100.
- **Date and time** — stored internally as serial numbers so they can be
  subtracted and sorted chronologically, but displayed as calendar dates. This
  dual nature is why a "date" that is really text will sort alphabetically
  (wrongly) and refuse date arithmetic.
- **Boolean** — logical ``TRUE`` / ``FALSE`` values, produced by comparisons
  and consumed by ``IF`` and filters.

Why the type matters
----------------------

The type determines the valid operation, exactly as the measurement levels did:
you sum numbers, sort dates chronologically, and count text. The everyday
failures trace to a mismatch between a cell's *actual* type and its *apparent*
one — the classic being numbers or dates imported as text, which look right and
compute wrong. Recognising and fixing these is core Prepare-phase work, and it
connects directly to the value types (nominal, ordinal, discrete, continuous)
of the previous lessons: the spreadsheet type is how those abstract levels are
physically stored.

Checking and fixing types
---------------------------

Two habits catch most trouble. **Watch the alignment**: numbers and dates that
sit left-aligned are secretly text and will misbehave. **Convert deliberately**:
use the spreadsheet's type-conversion tools (or a helper column) to turn
text-numbers into real numbers and text-dates into real dates *before*
computing, rather than discovering the problem inside a broken formula. The
cleaning section builds these fixes into a systematic workflow.

The caveat
------------

Spreadsheets guess types automatically, and the guess is sometimes wrong —
famously, identifiers that look like numbers (a product code ``00123``) lose
leading zeros, and codes that look like dates get silently converted. Automatic
typing is a convenience that occasionally corrupts data on import, so part of
preparation is verifying that each column's type is the one you *intended*, not
merely the one the software chose. The next lesson steps up from individual
cells to the table they form.
"""

CONTENT["Data Tables (Tabular Data)"] = r"""
The shape analysis lives in
-----------------------------

Most data analysis happens on **tabular data** — data arranged in a table of
rows and columns. It is the shape spreadsheets display, databases store, and
the analysis tools of this course expect. Understanding what makes a table
*well-formed* is the bridge between raw data and analysis, and it formalises
the layout rules the spreadsheet-organisation lesson introduced.

The anatomy of a data table
-----------------------------

A proper data table has a consistent structure:

- **Rows are records** (also called observations) — each row is one instance of
  the thing being measured: one sale, one patient, one day.
- **Columns are variables** (also called fields or attributes) — each column is
  one measured property, holding the same type of value down its length:
  ``date``, ``region``, ``amount``.
- **A header row** names the columns, giving each variable an explicit,
  unique label.
- **Cells** hold one value each — the intersection of one record and one
  variable.

The defining discipline is consistency: every row has the same columns, and
every column holds one kind of value. That regularity is exactly what lets a
tool sort, filter, aggregate, or join the table mechanically.

Why the structure is non-negotiable
-------------------------------------

A table that breaks the pattern breaks the tools. Two variables crammed into one
column ("New York, NY" where city and state should be separate) cannot be
grouped by state. One record split across two rows double-counts under
aggregation. A column mixing types defeats sorting and math. The rules are not
aesthetic — they are the contract every downstream operation relies on, which
is why cleaning so often means *restoring* proper tabular structure before
anything else.

Tables and the relational world
---------------------------------

The tabular shape is also the foundation of **relational databases**, where
data lives in many linked tables — a topic the sources stage of this section
develops. The same "rows are records, columns are variables" discipline scales
from a single spreadsheet to a database of hundreds of related tables; learning
it here is learning the grammar of both.

The caveat
------------

Not every table you receive is well-formed — spreadsheets in the wild are full
of merged cells, sub-headers, totals rows wedged between data, and summary
layouts meant for human eyes rather than analysis. Recognising the gap between a
*presentation* table and an *analysis* table is a core preparation skill: the
next lesson examines two legitimate analysis layouts — wide and long — and when
each is the right one.
"""

CONTENT["Wide Data vs. Long Data"] = r"""
Two shapes for the same facts
-------------------------------

Well-formed tabular data still comes in two distinct layouts, and knowing the
difference — and how to move between them — is a genuinely useful preparation
skill. The same information can be arranged as **wide** data or **long** data;
neither is universally correct, and different tools and tasks prefer different
shapes.

Wide data
-----------

In **wide** format, each subject occupies a single row, and each measured
variable (or each time point) gets its own column. Sales by quarter, wide:

.. code-block:: text

   region    Q1     Q2     Q3     Q4
   North    120    135    128    150
   South     98    102    115    120

One row per region; the four quarters spread across columns. Wide data is
**compact and human-readable** — easy to scan, and the natural shape for a
report or a spreadsheet a person will read.

Long data
-----------

In **long** format, each row is a single observation, with columns identifying
*what* is measured and *its value*. The same sales data, long:

.. code-block:: text

   region   quarter   sales
   North    Q1        120
   North    Q2        135
   North    Q3        128
   ...
   South    Q4        120

Every region–quarter combination is its own row. Long data is more repetitive
to read but far more **flexible for analysis**: adding a new measure means
adding rows, not restructuring columns, and most plotting and statistical
tools expect this "one observation per row" shape (it is the *tidy* form the
tabular lesson gestured at).

Choosing and converting
-------------------------

The rule of thumb: **wide for reading, long for analysis**. A stakeholder-facing
summary is usually wide; the data feeding a chart, a pivot, or a statistical
routine is usually long. Analysts convert between them constantly — reshaping
wide to long before analysis, and often pivoting long back to wide for
presentation. Spreadsheets do this with pivot tables; the Python section does it
with pandas' reshape operations. The key realisation is that **the choice is
about convenience, not correctness** — the underlying facts are identical, only
their arrangement differs.

A worked reason it matters
----------------------------

Suppose you want a line chart of sales over quarters, one line per region. From
*long* data, the chart is a direct mapping: quarter on the x-axis, sales on the
y-axis, region as the colour — one instruction. From *wide* data, you must
first tell the tool that four separate columns are really one variable measured
four times. Same data, but the long shape matched the task and the wide shape
fought it. Recognising which shape a task wants saves exactly this friction.

The caveat
------------

Real datasets are often *neither* cleanly wide nor cleanly long — they are
half-reshaped, with some variables in columns and some in rows, or time points
mixed with attributes. Part of preparation is recognising the current shape and
deciding the target shape the analysis needs, then converting deliberately.
This closes the data-types stage; the next turns to the quality and fairness of
data, beginning with the bias that unexamined data carries.
"""

CONTENT["Understanding Bias in Data Analysis"] = r"""
Bias is the default, not the exception
----------------------------------------

Section 2 met bias through the lens of *context*; this stage confronts it
directly, because bias is one of the central threats to valid analysis. The
uncomfortable starting point: **bias is the default state of data, and
unbiased data is an achievement**, not a given. Data is collected by particular
means, from particular people, at particular times — every one of those
particulars is a chance for bias to enter. Assuming data is neutral until proven
otherwise is exactly backwards.

What bias is
--------------

**Bias** is a systematic preference for or against certain outcomes, groups, or
conclusions that distorts an analysis away from the truth. The key word is
*systematic*: bias is not random error that averages out with more data — it is
a consistent lean in one direction, which more data only makes more confidently
wrong. A biased sample of a million people misleads more reliably than a biased
sample of a hundred.

Where bias enters
-------------------

Bias can enter at every stage of the data's journey:

- **In collection** — who and what got measured (the sampling and selection
  biases the next lessons detail).
- **In the data's history** — when the data faithfully records a biased world
  and the analysis projects it forward (the Amazon recruiting case).
- **In the questions** — leading or assumption-laden questions that steer toward
  a preferred answer (the framing stage's anti-patterns).
- **In the analysis** — the analyst's own confirmation bias, scrutinising
  unwelcome results harder than welcome ones.
- **In interpretation** — reading a pattern through a preconception, seeing what
  was expected rather than what is there.

Notice bias is not only *in the data* — it is also in the *analyst* and the
*process*. Guarding against it means examining your own methods and
expectations, not just auditing the dataset.

Why it is so dangerous
------------------------

Bias is insidious precisely because it produces *plausible, confident, wrong*
results — analyses that look rigorous and pass unexamined because their
conclusions feel right. Unlike a visible error code, bias raises no flag; the
numbers compute cleanly and the story coheres. This is why bias must be
actively hunted rather than passively avoided: nothing in the tooling will warn
you.

Toward unbiased analysis
--------------------------

Because bias is the default, reducing it takes deliberate work: asking who is
represented and who is missing, seeking data that could *disconfirm* the
expected answer, disaggregating results across groups, and inviting others to
challenge the method. None eliminates bias entirely — the realistic goal is to
*recognise, reduce, and honestly disclose* it. The following lessons make this
concrete, beginning with the specific mechanisms of sampling and the varieties
of bias analysts most often meet.

The caveat
------------

Awareness of bias can curdle into paralysis or into weaponised doubt — dismissing
any unwelcome finding as "biased" while accepting congenial ones uncritically.
The discipline is even-handed: apply the same scrutiny to results you like and
results you do not, and treat "this might be biased" as the start of an
investigation, not the end of an argument. Disclosed, understood bias is
workable; it is the *unexamined* bias that ruins an analysis.
"""


MINDMAP.update({
    "Data Types in Spreadsheets": [
        "Understanding Data Types and Data Formats",
        "Building and Organizing a Spreadsheet",
        "Data Tables (Tabular Data)",
        "Data Formatting and Unit Conversion in Spreadsheets",
    ],
    "Data Tables (Tabular Data)": [
        "Data Types in Spreadsheets",
        "Wide Data vs. Long Data",
        "Structured Data and Data Models",
        "Databases and Relational Database Concepts",
    ],
    "Wide Data vs. Long Data": [
        "Data Tables (Tabular Data)",
        "Structured Data and Data Models",
        "Understanding Data Types and Data Formats",
        "Data Types in Spreadsheets",
    ],
    "Understanding Bias in Data Analysis": [
        "Context and Bias in Data Analysis",
        "Sampling Bias and Unbiased Data",
        "Common Types of Data Bias",
        "Fairness in Data Analysis",
    ],
})


# ======================================================================
# Section 3 — Prep / bias_ethics (bias + ROCCC)  (prep 009-012)
# ======================================================================

GLOSS.update({
    "Sampling Bias and Unbiased Data":
        "when the sample misrepresents the population — and how random sampling guards against it",
    "Common Types of Data Bias":
        "sampling, observer, interpretation, and confirmation bias, with how each distorts",
    "Identifying Good Data Sources (ROCCC Framework)":
        "Reliable, Original, Comprehensive, Current, Cited — the marks of trustworthy data",
    "Identifying Bad Data Sources (When Data Does Not ROCCC)":
        "reading ROCCC in reverse to spot the data you should not trust",
})

CONTENT["Sampling Bias and Unbiased Data"] = r"""
The sample stands in for the whole
------------------------------------

Analysis rarely examines an entire population; it examines a **sample** and
generalises. That move is only valid when the sample fairly represents the
whole — and when it does not, the result is **sampling bias**: a sample in
which some members of the population are systematically over- or
under-represented, so conclusions drawn from it do not hold for the population.
It is the most common and most consequential bias analysts face.

How sampling bias happens
---------------------------

The classic mechanism is a sampling method that reaches some groups more than
others:

- **Convenience sampling** — surveying whoever is easiest to reach (your own
  customers, people who answer the phone at 2 pm), which skews toward whoever
  that happens to be.
- **Self-selection** — letting people opt in (online reviews, voluntary
  surveys), which over-represents those with strong opinions and the time to
  express them.
- **Coverage gaps** — a sampling frame that silently excludes part of the
  population (a phone survey missing people without phones).

A famous historical example: a 1936 US presidential poll drew millions of
responses from telephone and automobile-registration lists, predicted the wrong
winner, and failed precisely because — in that era — phone and car owners were
wealthier and unrepresentative. Size did not save it; the sample was biased,
and more of a biased thing is still biased.

Unbiased data as the goal
---------------------------

**Unbiased data** is a sample representative of the whole population — one where
every relevant group appears in roughly its true proportion. The principal tool
for achieving it is **random sampling**: selecting members so that each has an
equal chance of being chosen, which lets representativeness arise by chance
rather than depending on a method that might lean. (The next stage's lesson on
population and sample size develops the mechanics.) Random selection does not
*guarantee* a representative sample in any single draw, but it removes the
*systematic* lean that defines bias.

Detecting it in practice
--------------------------

Two habits catch most sampling bias. **Ask how the data was collected** — the
provenance question from earlier — because the collection method is where the
bias lives: who could have been included, and who could not? **Compare the
sample to known population facts** — if your survey respondents are 70% one
demographic but the population is 50%, the skew is visible and quantifiable.
When you cannot fix a skewed sample, you can at least *state* its skew, so
conclusions are read with the right caution.

The caveat
------------

Perfectly unbiased data is largely unattainable — every real sample departs
from the population somehow. The professional standard is not perfection but
*awareness and disclosure*: knowing which groups your data over- and
under-represents, adjusting where you can, and being explicit about the limits
where you cannot. A modest analysis honest about its sampling beats an ambitious
one that assumes representativeness it never checked.
"""

CONTENT["Common Types of Data Bias"] = r"""
A field guide to bias
------------------------

Sampling bias is the most common, but not the only, way analysis goes
systematically wrong. Analysts benefit from a field guide to the recurring
types, because naming a bias is the first step to guarding against it. The
standard set covers most of what you will meet.

The common types
------------------

- **Sampling bias** — the sample does not represent the population (the previous
  lesson). Enters at *collection*.
- **Observer bias** (also experimenter or measurement bias) — the tendency for
  different observers to see or record the same thing differently, often nudged
  by what they expect. Two people rating call quality, or reading an instrument
  near a threshold, may systematically differ. Enters at *measurement*.
- **Interpretation bias** — the tendency to interpret ambiguous results in the
  way that fits a preferred or expected conclusion, when the data genuinely
  admits more than one reading. Enters at *analysis*.
- **Confirmation bias** — the tendency to search for, favour, and recall
  information that confirms what you already believe, while discounting what
  contradicts it. The analyst runs the query that supports the hunch, scrutinises
  the disconfirming result harder, and remembers the hits. Enters at *the whole
  process*.

Two of these live in the *data* (sampling, and the measurement side of
observer bias) and two live in the *analyst* (interpretation, confirmation) —
which is why guarding against bias means auditing both the dataset and your own
reasoning.

Why they are hard to catch
----------------------------

Every one of these produces internally consistent, plausible results — the
hallmark of bias from the previous lesson. Confirmation bias is especially
insidious because it feels like *diligence*: scrutinising an inconvenient result
harder than a convenient one looks responsible while being exactly the
mechanism of the bias. The tell is asymmetry — applying more skepticism to
findings you dislike than to findings you like.

Guarding against them
-----------------------

Each type has a countermeasure. Sampling bias: examine collection and compare to
population facts. Observer bias: use clear, objective definitions and, where
possible, multiple independent observers or automated measurement.
Interpretation bias: state alternative readings explicitly and check which the
data actually supports. Confirmation bias: deliberately seek *disconfirming*
evidence — ask "what would make me wrong?" and run *that* query too. The
unifying practice is inviting others to challenge the work, because another
person's biases rarely align exactly with yours.

The caveat
------------

Bias cannot be fully eliminated — it can be recognised, reduced, and disclosed.
Over-correcting has its own hazard: bending over backward to avoid one bias can
introduce another, and reflexively distrusting every finding is not rigour but
paralysis. The aim is calibrated, even-handed skepticism applied equally to
welcome and unwelcome results — the same discipline the fairness thread has
asked for throughout. The next lessons turn from bias in general to a concrete
checklist for judging whether a *source* is trustworthy.
"""

CONTENT["Identifying Good Data Sources (ROCCC Framework)"] = r"""
A checklist for trust
-----------------------

Before building on a dataset, an analyst must judge whether the *source* is
trustworthy — and a memorable checklist captures what "trustworthy" means. Good
data sources are **ROCCC**: **R**eliable, **O**riginal, **C**omprehensive,
**C**urrent, and **C**ited. Running a candidate source through the five letters
turns a vague sense of trust into specific, checkable questions.

The five criteria
-------------------

- **Reliable** — is the data accurate, complete, and unbiased, from a source
  with a track record of getting things right? Reliable data comes from
  well-regarded, vetted origins, not anonymous or casual ones.
- **Original** — can you get to the *source* of the data, rather than a copy of
  a copy? Original data comes from the party that actually collected it; each
  hand it passes through is a chance for distortion, so validating against the
  origin matters. (This echoes the first/second/third-party gradient from
  earlier.)
- **Comprehensive** — does the data contain all the information you need to
  answer the question? A source missing critical fields or covering only part
  of the population, however accurate on what it has, cannot fully answer the
  question.
- **Current** — is the data recent enough to be relevant? Data ages; a source
  from years ago may no longer reflect present reality, and the right recency
  depends on how fast the subject changes.
- **Cited** — is the source of the data documented — who created it, when,
  where, and how? Cited data can be traced and vetted; uncited data asks you to
  trust it blindly.

A source that satisfies all five is a strong foundation; the more it satisfies,
the more weight your conclusions can bear.

Using ROCCC in practice
-------------------------

ROCCC is a pre-flight check for the Prepare phase: before committing to a
dataset, walk the five letters and note where it is strong and weak. A source
need not be perfect on every criterion — few are — but the framework makes the
trade-offs *explicit*. "This data is comprehensive and current but not fully
original, since it's a third-party aggregation" is a clear-eyed basis for
proceeding with appropriate caution, and for stating that caution in the final
analysis.

Why a checklist beats a gut feeling
-------------------------------------

The value of ROCCC is that it makes source evaluation *systematic* rather than
intuitive. Under time pressure, it is tempting to grab whatever data is at hand
and start analysing — and the resulting problems (unreliable, stale, or
uncited data) surface only much later, after conclusions have been built on
sand. Five questions at the start are cheap insurance against that.

The caveat
------------

ROCCC evaluates the *source*, not the *fit to your specific question*. Data can
be impeccably reliable, original, comprehensive, current, and cited — and still
be the wrong data for your problem, or carry sampling bias the framework does
not directly test. Use ROCCC together with the relevance and coverage judgements
from the choosing-data lesson: source quality and question-fit are separate
checks, and a good analysis passes both. The next lesson applies ROCCC in
reverse, to recognise the sources you should *not* trust.
"""

CONTENT["Identifying Bad Data Sources (When Data Does Not ROCCC)"] = r"""
ROCCC in reverse
------------------

The ROCCC framework doubles as a detector for *bad* data: a source that fails
one or more criteria is one to distrust, or at least to use with open eyes. Just
as good data "ROCCCs", bad data fails to — and learning to read each failure is
how an analyst avoids building on unsound ground. This lesson runs the five
criteria backward.

The five failures
-------------------

- **Not Reliable** — from an unvetted or low-quality origin, known for errors,
  or visibly incomplete and inconsistent. Red flags: no clear methodology, a
  source with no reputation to protect, obvious internal contradictions.
- **Not Original** — a copy of a copy, with no path back to who actually
  collected it. Red flags: data quoted from a secondary article that cites no
  primary source; aggregations whose components cannot be traced. Every removed
  hand is unverifiable distortion.
- **Not Comprehensive** — missing fields, populations, or time spans the
  question requires. Red flags: gaps in coverage, suspiciously round or sparse
  data, a scope narrower than the question needs.
- **Not Current** — too old to reflect present reality, in a domain that has
  moved on. Red flags: no collection date given, or a date that predates
  relevant changes in the subject.
- **Not Cited** — no documentation of who created it, when, where, or how. Red
  flags: data with no attribution, no methodology, no provenance — asking for
  blind trust.

A source failing several of these is not merely imperfect; it is a liability,
and analysis built on it inherits every weakness.

The seductive-but-bad source
------------------------------

The dangerous case is data that is *convenient* — free, large, immediately
available — but fails ROCCC quietly. A big, easily-downloaded dataset with no
methodology and no date is tempting under deadline, and its problems do not
announce themselves: the analysis runs, the charts render, and the unreliability
surfaces only when a decision built on it goes wrong. Convenience is not a ROCCC
criterion; weigh it against the five that are.

What to do with a flawed source
---------------------------------

Failing ROCCC is not always disqualifying — sometimes flawed data is the only
data available. The professional responses, in order of preference: **find a
better source** if one exists; **supplement** the weak source with others to
cover its gaps; **validate** what you can against independent data; and, at
minimum, **disclose** the limitation prominently, so anyone relying on the
conclusion knows the ground it stands on. What is never acceptable is using
known-bad data *silently*, letting others assume a soundness you know is absent.

The caveat
------------

Source evaluation is a judgement, not a pass/fail gate — real data almost always
fails *some* criterion to *some* degree, and rejecting everything imperfect
would leave you with nothing to analyse. The skill is proportionality: matching
the required source quality to the stakes of the decision (the speed-versus-
accuracy trade-off, applied to data), and being honest about where the data
falls short of what the question deserves. This completes the bias and source-
quality core; the next lessons turn to the ethics and privacy obligations that
govern how data may be used at all.
"""


MINDMAP.update({
    "Sampling Bias and Unbiased Data": [
        "Understanding Bias in Data Analysis",
        "Common Types of Data Bias",
        "Population, Sample Size, and Random Sampling",
        "Choosing the Right Data to Collect",
    ],
    "Common Types of Data Bias": [
        "Sampling Bias and Unbiased Data",
        "Understanding Bias in Data Analysis",
        "Context and Bias in Data Analysis",
        "Identifying Good Data Sources (ROCCC Framework)",
    ],
    "Identifying Good Data Sources (ROCCC Framework)": [
        "Identifying Bad Data Sources (When Data Does Not ROCCC)",
        "Common Types of Data Bias",
        "How Data Is Generated and Collected",
        "Accessing Data: Internal and External Sources",
    ],
    "Identifying Bad Data Sources (When Data Does Not ROCCC)": [
        "Identifying Good Data Sources (ROCCC Framework)",
        "Sampling Bias and Unbiased Data",
        "Data Ethics in Data Analysis",
        "Choosing the Right Data to Collect",
    ],
})


# ======================================================================
# Section 3 — Prep / bias_ethics (close) + sources (open)  (prep 013-016)
# ======================================================================

GLOSS.update({
    "Data Ethics in Data Analysis":
        "ownership, transaction transparency, consent, and currency — the ethics of using data",
    "Data Privacy in Data Ethics":
        "protecting people's information: what privacy requires of an analyst",
    "Open Data and Openness in Data Ethics":
        "when data should be freely available — and the tension with privacy",
    "Databases and Relational Database Concepts":
        "tables, keys, and relationships: how organisational data is really stored",
})

CONTENT["Data Ethics in Data Analysis"] = r"""
Data is about people
----------------------

Much of the data analysts handle describes **people** — their purchases,
movements, health, opinions. That makes analysis an ethical activity, not just
a technical one, and **data ethics** — the well-founded standards of right and
wrong that govern how data is collected, shared, and used — is a core
professional obligation, not an optional add-on. The fairness thread from the
foundations widens here into a full account of responsible data use.

The aspects of data ethics
----------------------------

A standard framing organises data ethics around several aspects:

- **Ownership** — the principle that people own the data they generate, not the
  organisations that collect it. It is *their* information, held on their
  behalf, which reframes the collector as a custodian with duties rather than an
  owner with rights.
- **Transaction transparency** — people are entitled to know how their data is
  collected, stored, and used; the processing should be transparent, not
  hidden. No secret collection, no undisclosed uses.
- **Consent** — people have the right to know what data is gathered and to agree
  to its collection and use for a stated purpose. Consent must be *informed* (they
  understand what they are agreeing to) and *specific* (to a purpose, not a blank
  cheque).
- **Currency** — people have a right to know about the financial transactions
  involving their data: if it is being used to generate profit, and how.

Together these describe a relationship of **trust and accountability** between
the people data is about and the organisations that hold it — a relationship the
analyst operates inside and helps uphold.

Ethics in the analyst's daily work
------------------------------------

Data ethics is not only for policy-makers; it shows up in ordinary tasks. Should
this dataset be joined with that one, creating a more revealing picture than
either alone and than people consented to? Is this analysis using data for a
purpose the people it describes never agreed to? Does this "anonymised" dataset
actually protect identities, or could individuals be re-identified? An
ethically alert analyst asks these routinely — before the analysis, not after a
problem surfaces.

Why ethics protects the analysis too
--------------------------------------

Ethical lapses are also *analytical and business* risks. Data used unethically
invites legal penalty, destroys the trust that makes people willing to share
data at all, and often rests on data that was collected in ways that make it
biased or unreliable. Doing right by the people in the data and doing sound,
durable analysis point the same direction more often than they conflict.

The caveat
------------

Data ethics involves genuine judgement and real trade-offs — privacy against
usefulness, individual consent against collective benefit (public-health
research on medical data is a standing example) — and reasonable people
disagree on hard cases. The professional standard is not a single right answer
but *taking the questions seriously*: recognising the ethical dimension,
weighing the interests honestly, and being able to justify the choice. The next
lessons develop two central aspects — privacy and openness — in more depth.
"""

CONTENT["Data Privacy in Data Ethics"] = r"""
Privacy as protection
-----------------------

**Data privacy** is the aspect of data ethics concerned with protecting a
person's information — the right to control how their personal data is
collected, used, shared, and retained. Where ownership and consent establish
*whose* data it is and *whether* it may be used, privacy governs *how it is
protected* once entrusted, and it is often the most legally regulated part of
an analyst's obligations.

What privacy requires
-----------------------

Several concrete duties follow from the privacy principle:

- **Protect personal data** — secure it against unauthorised access, with the
  access controls, encryption, and careful handling the data-security lesson
  addresses. A breach is a privacy failure regardless of intent.
- **Limit use to purpose** — use personal data only for the purpose it was
  collected and consented for. Repurposing data for a new use people never
  agreed to violates privacy even when the data is held securely.
- **Minimise** — collect and retain only the personal data actually needed, and
  no longer than needed. Data you do not hold cannot be breached or misused; the
  life cycle's *destroy* stage is a privacy control.
- **Guard identity** — recognise that combining datasets or retaining detail can
  make supposedly anonymous data re-identifiable.

Personally identifiable information (PII)
-------------------------------------------

The heart of privacy is **PII** — information that can identify a specific
individual, directly (name, government ID, email) or indirectly (a combination
of attributes that together single someone out). Analysts handle PII with
particular care: accessing it only when necessary, and preferring to work with
data from which identifiers have been removed.

**Anonymisation** (removing identifying information so individuals cannot be
recognised) is the standard protective technique — but it is harder than it
looks. The recurring lesson from real re-identification cases is that stripping
obvious identifiers is often *insufficient*: a combination of "anonymous"
attributes — a postal code, a birth date, a gender — can uniquely identify a
person even with names removed. Genuine anonymisation must consider what the
remaining fields, and any dataset they could be joined with, reveal in
combination.

Privacy in the analyst's workflow
-----------------------------------

Practically: access the minimum PII the task requires; work with anonymised or
aggregated data by default; never join datasets in ways that re-identify people
without authorisation; and treat "could this analysis expose an individual?" as
a question asked *before* running it. Aggregation — reporting groups rather than
individuals — is a workhorse privacy technique, though even aggregates can leak
when a group is small enough to point at one person.

The caveat
------------

Privacy exists in tension with analytical usefulness: the same detail that makes
data powerful can make it privacy-invasive, and stronger protection often means
coarser data. There is no universal setting — the right balance depends on the
data's sensitivity, the purpose, applicable law, and what people consented to.
The obligation is to weigh usefulness against protection deliberately, default
toward protecting people, and know the legal requirements that apply. The next
lesson turns to the opposite impulse: data that *should* be open.
"""

CONTENT["Open Data and Openness in Data Ethics"] = r"""
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
"""

CONTENT["Databases and Relational Database Concepts"] = r"""
Where organisational data really lives
----------------------------------------

Spreadsheets are where analysts often *work*, but they are rarely where
organisational data *lives*. That home is the **database** — a structured
collection of data stored electronically and organised for efficient access,
management, and retrieval. Most business data of any scale sits in databases,
and the dominant kind is the **relational database**, whose concepts every
analyst needs, because SQL — the language of the coming sections — is the
language of exactly these systems.

The relational model
----------------------

A **relational database** organises data into **tables** (rows and columns,
the tabular structure from earlier) and — crucially — defines **relationships**
between those tables, so related information stored separately can be linked.
The core concepts:

- **Tables** hold one kind of entity each — a ``customers`` table, an
  ``orders`` table, a ``products`` table — rather than cramming everything into
  one giant sheet.
- **Primary key** — a column (or combination) whose value uniquely identifies
  each row in a table: a ``customer_id`` that is different for every customer.
  It is the table's guarantee that each record is distinct and addressable.
- **Foreign key** — a column in one table that refers to the primary key of
  another, creating the link between them. An ``orders`` table's
  ``customer_id`` column is a foreign key pointing at the ``customers`` table,
  recording *which customer* placed each order.
- **Relationships** — the connections foreign keys express (one customer has
  many orders; each order belongs to one customer), letting data be stored once
  and joined when needed.

Why not just one big table?
-----------------------------

Relational structure exists to avoid **redundancy** and the errors it breeds.
Storing each order alongside a full copy of its customer's details would repeat
that customer's name and address on every order, waste space, and — worse —
create inconsistency the moment one copy is updated and another is not. Keeping
customers in one table and referring to them by key means each fact is stored
**once**, in one authoritative place. This principle (roughly, *normalisation*)
is why real databases are many small linked tables rather than one sprawling
sheet, and why joining them is a core analytical skill.

Why analysts need this
------------------------

Three practical reasons. The data you will query is *shaped* this way, so
understanding tables and keys is understanding what you are querying. Answering
real questions usually means **combining** tables — customer data with order
data with product data — via their relationships, which is what SQL ``JOIN``
(the analysis section's topic) does. And the "stored once, in one place"
principle explains data you will meet: why an ``orders`` table has a bare
``customer_id`` instead of full customer details, and where to go to look them
up.

The caveat
------------

The relational model is dominant but not universal — other database kinds
(document, key-value, graph, and other "NoSQL" stores) organise data
differently for different needs, and very large or unstructured data often lives
outside neat relational tables. Relational concepts are the essential
foundation and the right starting point, but not the whole storage landscape.
The next lessons stay with this world, turning to the *metadata* that describes
what is in these databases and the governance that manages it.
"""


MINDMAP.update({
    "Data Ethics in Data Analysis": [
        "Fairness in Data Analysis",
        "Data Privacy in Data Ethics",
        "Open Data and Openness in Data Ethics",
        "Identifying Bad Data Sources (When Data Does Not ROCCC)",
    ],
    "Data Privacy in Data Ethics": [
        "Data Ethics in Data Analysis",
        "Open Data and Openness in Data Ethics",
        "Data Security in Spreadsheets",
        "Context and Bias in Data Analysis",
    ],
    "Open Data and Openness in Data Ethics": [
        "Data Ethics in Data Analysis",
        "Data Privacy in Data Ethics",
        "How Data Is Generated and Collected",
        "Accessing Data: Internal and External Sources",
    ],
    "Databases and Relational Database Concepts": [
        "Structured Data and Data Models",
        "Metadata in Databases",
        "Querying Data with SQL",
        "Accessing Data: Internal and External Sources",
    ],
})


# ======================================================================
# Section 3 — Prep / sources (close) + spreadsheets_sql (open)  (prep 017-020)
# ======================================================================

GLOSS.update({
    "Metadata in Databases":
        "data about data: descriptive, structural, and administrative context that makes data usable",
    "Metadata Repositories and Data Governance":
        "where metadata is catalogued, and the governance that keeps data trustworthy",
    "Accessing Data: Internal and External Sources":
        "getting to the data: what lives inside the organisation versus outside",
    "Importing Data into Spreadsheets":
        "getting external data into a sheet cleanly — and the type traps to watch",
})

CONTENT["Metadata in Databases"] = r"""
Data about data
-----------------

A dataset alone is often not enough to use safely; you also need to know *what
it is* — where it came from, what each field means, when it was last updated.
That describing information is **metadata**: data about data. It is what turns an
anonymous grid of values into something an analyst can trust and interpret
correctly, and it is a first-class concern of the Prepare phase.

The kinds of metadata
-----------------------

Metadata is conventionally grouped into three kinds:

- **Descriptive metadata** — identifies and describes the data: a dataset's
  title, author, a column's meaning, keywords. It answers *what is this?* — the
  difference between a column called ``val`` and knowing it holds "monthly
  revenue in USD, net of refunds."
- **Structural metadata** — describes how the data is organised and how its
  parts relate: the tables, their columns and types, the keys linking them,
  the order and grouping of elements. It answers *how is this put together?*
- **Administrative metadata** — describes management and provenance: who created
  the data and when, how and when it was last updated, who may access it, and
  usage or licensing terms. It answers *where did this come from and who owns
  it?*

Together these supply exactly the **context** the bias lessons demanded —
metadata is context, made explicit and stored alongside the data.

Why metadata is indispensable
-------------------------------

Without metadata, data is ambiguous and dangerous. A column of numbers with no
description invites the wrong interpretation; a dataset with no update date
might be dangerously stale; data with no provenance cannot be evaluated for
ROCCC. Good metadata makes data **findable** (you can locate the right dataset),
**understandable** (you interpret fields correctly), **trustworthy** (you can
assess its source and currency), and **usable** (you know how it is structured
and what you may do with it). Much of the confusion and error in real analysis
traces to metadata that was missing, wrong, or ignored.

Metadata in the analyst's workflow
------------------------------------

Practically, metadata is the *first thing to read* about an unfamiliar dataset:
the data dictionary defining each column, the documentation of how it was
collected, the last-updated timestamp. Reading it first prevents a whole class
of downstream mistakes — averaging a field that turns out to be a code,
comparing two columns that measure subtly different things, trusting a stale
extract. And as a *producer*, leaving good metadata (clear column definitions,
provenance notes, update dates) is what makes your work usable by others and by
your future self — the chain-of-custody discipline from the foundations.

The caveat
------------

Metadata is only as good as its upkeep: documentation drifts out of date as data
changes, and stale or wrong metadata can mislead more confidently than none —
a data dictionary that no longer matches the table sends you wrong with an air
of authority. Treat metadata as valuable but verify it against the data when
stakes are high, and when you maintain data, keep its metadata current as part
of the work. The next lesson turns to where metadata is catalogued
organisation-wide, and the governance around it.
"""

CONTENT["Metadata Repositories and Data Governance"] = r"""
Managing metadata at scale
----------------------------

One dataset's metadata can live in a README; an organisation's cannot. As data
multiplies across many systems, the metadata describing it needs its own
organised home — a **metadata repository** — and the whole practice of keeping
data trustworthy and well-managed becomes **data governance**. Both matter to
analysts because they determine whether the data you need can be *found*,
*understood*, and *trusted* across an organisation.

Metadata repositories
------------------------

A **metadata repository** is a central store that describes an organisation's
data — a catalogue of what datasets exist, where they live, what their fields
mean, how they relate, and how current and trustworthy they are. Sometimes
called a data catalogue, it is the "card catalogue" for an organisation's data:
instead of asking around to discover whether some dataset exists and what its
columns mean, an analyst consults the repository. It brings the descriptive,
structural, and administrative metadata of the previous lesson together in one
searchable place, and it is what makes data *discoverable* at organisational
scale rather than tribal knowledge held by whoever has been there longest.

Data governance
-----------------

**Data governance** is the set of processes, roles, policies, and standards that
ensure an organisation's data is accurate, secure, usable, and handled
responsibly throughout its life cycle. It is the management framework around the
data-life-cycle stages from the foundations, and it typically covers:

- **Quality standards** — definitions and rules for what "correct" data looks
  like, so the same term means the same thing everywhere (the reconciliation
  problem from the life-cycle lessons, solved at the policy level).
- **Access and security** — who may see and change what data, enforcing the
  privacy and security obligations from earlier.
- **Ownership and stewardship** — named people (data owners, data stewards)
  accountable for specific datasets, so responsibility is not diffuse.
- **Compliance** — ensuring data handling meets legal and regulatory
  requirements, especially for personal data.

A common governance role is the **data steward** — someone responsible for the
quality, definitions, and proper use of a particular data domain.

Why analysts should care
--------------------------

Governance and repositories shape an analyst's daily reality. Good governance
means the data you pull is defined consistently, its quality is known, and you
can find it via the catalogue — enormous time saved and errors avoided. Weak
governance means the opposite: undocumented datasets, the same metric computed
three incompatible ways, and hours lost discovering what data even exists.
Analysts are also governance *participants* — following its standards, using
agreed definitions, and often helping improve the metadata and documentation
they consume.

The caveat
------------

Governance is a means, not an end — and it can fail in both directions.
Too little, and data becomes an untrustworthy free-for-all; too much, and
bureaucratic process strangles the agility analysis needs, with catalogues so
cumbersome nobody maintains them. The goal is *enough* governance to make data
trustworthy and findable without making it inaccessible — a balance every
data-mature organisation is perpetually tuning. This closes the sources stage;
the next turns from finding and describing data to actually *getting* it, and
bringing it into the tools where analysis happens.
"""

CONTENT["Accessing Data: Internal and External Sources"] = r"""
Getting to the data
----------------------

Knowing what data you need is one thing; *getting* it is another. Data an
analyst uses comes from two broad places — **inside** the organisation and
**outside** it — and each has its own access routes, trust characteristics, and
pitfalls. Recognising which kind you are dealing with shapes how you obtain it
and how much you can trust it.

Internal (first-party) data
-----------------------------

**Internal data** is data the organisation collected and holds itself — the
first-party data from earlier. It lives in the organisation's own systems:

- **Databases** — the relational systems from the previous stage, queried with
  SQL.
- **Data warehouses** — large stores consolidating data from many internal
  systems for analysis.
- **Application and operational systems** — the CRM, the sales platform, the
  support tool, each holding its own records.
- **Internal files** — spreadsheets, exports, and documents on shared drives.

Internal data's advantages are relevance and trust: it is about your own
organisation, and you can (in principle) understand and verify how it was
collected. Its challenges are practical — knowing it exists (the metadata
repository's job), getting *access* to it (permissions and data owners), and its
frequent scattering across disconnected systems.

External (second- and third-party) data
------------------------------------------

**External data** comes from outside the organisation — the second- and
third-party data from earlier:

- **Public and open data** — government statistics, open datasets, public
  research (the open-data lesson's territory).
- **Purchased data** — datasets bought from vendors and data providers.
- **Partner data** — shared directly by another organisation.
- **APIs** — programmatic interfaces that deliver external data on request,
  often as JSON (the semi-structured format from earlier).

External data extends what internal data alone can answer — market context,
benchmarks, demographics — but demands more scrutiny: you did not collect it, so
its provenance, quality, and currency must be evaluated with full ROCCC rigour,
and its licensing and privacy terms must be respected.

Choosing and combining sources
--------------------------------

Real analysis often *combines* both: internal sales data enriched with external
demographic or economic data, joined on a common key. The internal data grounds
the analysis in your own reality; the external data supplies context your own
systems do not hold. The judgement is matching source to question — and
remembering that combining sources multiplies not only insight but also the
provenance, quality, and privacy questions you must answer for *each* source.

The caveat
------------

Access is also a *permission and ethics* matter, not merely a technical one.
Just because data is reachable does not mean you are authorised to use it, or
that using it is ethical — internal data carries access controls for reasons,
and external data carries licences and privacy obligations. Confirm you are
*permitted* to use data for your purpose, not just *able* to obtain it — the
governance and ethics threads from the previous lessons, applied at the moment
of access. The next lesson gets concrete about pulling external data into the
analyst's most common tool: the spreadsheet.
"""

CONTENT["Importing Data into Spreadsheets"] = r"""
From source to sheet
----------------------

The most common first step of a hands-on analysis is getting data *into* a
spreadsheet — importing it from a file, a database export, or an external
source. Done carelessly, import is where a surprising share of data problems are
born; done deliberately, it sets up everything after. This lesson is the
practical bridge from "data exists somewhere" to "data is in my sheet, correctly
typed and ready."

Common import routes
----------------------

- **CSV / TSV files** — the universal exchange format from earlier; spreadsheets
  open them directly, parsing rows and delimiter-separated columns.
- **Other spreadsheet files** — opening or importing an existing ``.xlsx``.
- **Database exports** — data pulled from a database (often *as* CSV) and loaded
  in.
- **Copy and paste** — quick for small data, but the most error-prone route for
  anything structured.
- **Connected imports** — some spreadsheets can pull directly from a URL, an
  API, or a database connection, refreshing as the source updates.

The import traps
------------------

Import is where the *type* problems from earlier lessons are created, so it is
where to catch them:

- **Numbers as text** — a numeric column arrives as text (common from CSV,
  where everything is text) and refuses arithmetic until converted. The
  left-alignment tell from the data-types lesson is your early warning.
- **Mangled dates** — dates parse into the wrong format or get silently
  converted, especially across regional day/month conventions.
- **Lost leading zeros** — identifier codes like ``00042`` lose their zeros when
  auto-typed as numbers, corrupting keys.
- **Delimiter and encoding issues** — a comma inside a text field splits a
  column wrongly; non-UTF-8 characters arrive garbled.
- **Header confusion** — the header row imported as data, or missing entirely.

Importing cleanly
-------------------

Three habits prevent most trouble. **Check types immediately** after import —
scan for left-aligned numbers, malformed dates, and dropped leading zeros before
doing anything else. **Control the import** rather than accepting defaults —
spreadsheets' import dialogs let you specify delimiters and column types up
front, which is far easier than fixing corruption afterward. And **keep the raw
import untouched** — paste it to its own tab and work on copies, the
raw-stays-raw rule from the spreadsheet-organisation lesson, so a botched
transformation never destroys the original.

The caveat
------------

A clean-looking import is not a verified one: data can import without error and
still be subtly wrong — a shifted column, a truncated field, an encoding that
corrupted a few characters. Import is the moment to apply the sanity checks from
the mathematical-thinking lesson — does the row count match the source, do
totals look right, do spot-checked values match the origin? Getting data into
the sheet is the *start* of trusting it, not the end. The next lessons, in the
analysis section proper, turn that imported data into answers — beginning with
sorting and filtering.
"""


MINDMAP.update({
    "Metadata in Databases": [
        "Databases and Relational Database Concepts",
        "Metadata Repositories and Data Governance",
        "Identifying Good Data Sources (ROCCC Framework)",
        "Structured Data and Data Models",
    ],
    "Metadata Repositories and Data Governance": [
        "Metadata in Databases",
        "Data Ethics in Data Analysis",
        "Accessing Data: Internal and External Sources",
        "Databases and Relational Database Concepts",
    ],
    "Accessing Data: Internal and External Sources": [
        "How Data Is Generated and Collected",
        "Importing Data into Spreadsheets",
        "Metadata Repositories and Data Governance",
        "Identifying Good Data Sources (ROCCC Framework)",
    ],
    "Importing Data into Spreadsheets": [
        "Accessing Data: Internal and External Sources",
        "Data Types in Spreadsheets",
        "Building and Organizing a Spreadsheet",
        "Sorting and Filtering Data in Spreadsheets",
    ],
})


# ======================================================================
# Section 3 — Prep / spreadsheets_sql (close)  (prep 021-025)  -- SECTION 3 COMPLETE
# ======================================================================

GLOSS.update({
    "Sorting and Filtering Data in Spreadsheets":
        "ordering and narrowing rows: the two most-used moves for making data legible",
    "BigQuery Account Types":
        "sandbox, free tier, and paid — how to access a cloud data warehouse for practice",
    "Querying Data with SQL":
        "SELECT, FROM, WHERE: retrieving exactly the rows and columns a question needs",
    "Organizing Data for Personal and Work Projects":
        "folder, file, and naming conventions that keep a data project findable and safe",
    "Data Security in Spreadsheets":
        "protecting a shared sheet: access control, protected ranges, and safe sharing",
})

CONTENT["Sorting and Filtering Data in Spreadsheets"] = r"""
The two moves you reach for first
-----------------------------------

Once data is imported and tidy, the first things an analyst does to make sense
of it are **sort** and **filter**. They are the most-used operations in any
spreadsheet, the fastest way to turn a wall of rows into something legible, and
the intuition behind their SQL equivalents (``ORDER BY`` and ``WHERE``) that the
analysis section develops. This lesson makes them precise.

Sorting: imposing order
-------------------------

**Sorting** reorders the rows of a table by the values in one or more columns —
ascending (A→Z, smallest→largest, earliest→latest) or descending. Its analytical
value is that order reveals: sort sales descending and the top performers rise to
the top; sort by date and the timeline becomes visible; sort by region then by
sales and you see the best within each group.

- **Single-column sort** orders by one column.
- **Multi-column sort** orders by one column, breaking ties with a second (region
  first, then sales within each region) — the spreadsheet applies them in
  priority order.

The one non-negotiable rule: sort the **whole table together**, so every row
moves as a unit. Sorting a single column in isolation — leaving the others in
place — silently scrambles which value belongs to which record, one of the most
destructive spreadsheet mistakes precisely because it produces no error, just
quietly corrupted data. (The one-row-one-record discipline from the organisation
lesson is what makes whole-table sorting safe.)

Filtering: narrowing the view
-------------------------------

**Filtering** temporarily hides rows that do not meet a condition, showing only
those that do — orders over $100, one region, this month's dates. Crucially,
filtering *hides* rather than *deletes*: the data is all still there, and
clearing the filter restores the full view. Filters can combine conditions
(region = "North" *and* amount > 100) to narrow to exactly the subset a question
concerns.

Filtering's analytical value is focus: most questions concern a *subset*, and
filtering isolates it so you can examine or summarise just that slice without the
rest as noise.

Sorting and filtering together
--------------------------------

The two combine constantly: filter to this quarter's northern orders, then sort
them by value to see the largest. This filter-then-sort move answers a huge range
of everyday questions ("what were our biggest northern deals this quarter?")
with two clicks and no formulas — which is exactly why it is the analyst's
reflexive first pass on new data.

The caveat
------------

Sorting and filtering change what you *see*, and it is easy to forget a filter is
active — drawing conclusions from a filtered view as though it were the whole
dataset, or exporting filtered data thinking it is complete. Always know whether
a filter is on, and remember that a *sort* permanently reorders the data (it
persists after you look away) while a *filter* only hides — different footprints,
both easy to lose track of. The next lessons move from the spreadsheet to
querying data at database scale.
"""

CONTENT["BigQuery Account Types"] = r"""
A place to practise SQL at scale
----------------------------------

To practise SQL on realistic, large datasets, you need access to a database
system — and a widely used, beginner-friendly option is **Google BigQuery**, a
cloud-based, serverless data warehouse that runs SQL queries over large datasets
without any local setup. Because BigQuery is a paid cloud service, understanding
its **account types** matters: they determine how you get started and whether you
risk any charge while learning.

The account types
-------------------

- **Sandbox** — the entry point for learners. The BigQuery **sandbox** lets you
  use BigQuery *without providing a credit card or creating a billing account*.
  It grants the same free-tier usage limits — around **1 TB of query processing
  and 10 GB of storage per month** — and lets you query BigQuery's library of
  **public datasets** immediately with just a Google account. Its main
  restriction is that tables automatically **expire after 60 days**, and some
  features (streaming inserts, DML statements, the Data Transfer Service) are
  unavailable. For learning SQL, it is ideal: zero cost, zero billing risk.
- **Free tier** — available to any account that *has* set up a billing account.
  It offers the same monthly free allowances (1 TB queries, 10 GB storage) but,
  because billing is enabled, allows **permanent storage** (no 60-day expiration)
  and the full feature set — with charges beginning only if you exceed the free
  limits.
- **Paid (full) account** — a billing account with no reliance on the free
  allowances, for production and heavier workloads; you pay for query processing
  and storage beyond the free tier.

Separately, new Google Cloud customers are typically offered a **$300 credit**
across Google Cloud products, which requires a credit card and is distinct from
the no-card sandbox.

Which to use, and why it matters
----------------------------------

For following this course and practising SQL, the **sandbox** is the right
choice: it removes the single biggest barrier for beginners — the fear of an
unexpected bill — while giving genuine access to the same query engine and real
public datasets professionals use. The 60-day expiration and monthly limits are
generous for learning and function as helpful guardrails toward efficient
queries. You can upgrade later, by attaching billing, if a real project needs
permanent storage or more capacity.

Why a cloud warehouse at all
------------------------------

BigQuery illustrates where much modern data lives: not on a laptop but in a
**cloud data warehouse** that scales to enormous datasets and is queried with
standard SQL. Learning on it means the SQL you practise is the SQL you would use
professionally, on infrastructure of the kind employers actually run — the
"learn the logic on real tools" principle from the foundations.

The caveat
------------

Cloud service details — limits, tiers, interface, even the exact free
allowances — change over time and vary by provider, so treat specific numbers as
current-at-writing rather than permanent, and check the provider's documentation
for the latest. BigQuery is one convenient environment among several (other cloud
warehouses and local databases work too); what transfers is the **SQL and the
warehouse concept**, not the particular vendor's current packaging. The next
lesson puts the environment to use with actual queries.
"""

CONTENT["Querying Data with SQL"] = r"""
Asking a database a question
------------------------------

With a database environment in hand, the core skill is the **query** — an SQL
statement that retrieves exactly the data you want. The foundations introduced
SQL's shape; this lesson is the hands-on core, the three clauses that answer the
majority of real questions and the foundation everything later in the SQL thread
builds on.

The essential query
---------------------

Every basic retrieval combines three clauses:

.. code-block:: sql

   SELECT product, region, amount
   FROM   orders
   WHERE  amount > 100;

- ``SELECT`` names the **columns** to return. ``SELECT *`` returns all columns;
  naming specific columns is tidier and, on large cloud warehouses, cheaper
  (you are billed for the columns scanned).
- ``FROM`` names the **table** the data comes from.
- ``WHERE`` sets the **condition** rows must meet to be included — the filter,
  in query form.

Read top to bottom, it is a sentence: *select* these columns *from* this table
*where* this condition holds.

Building up the WHERE clause
------------------------------

Most of a query's power lives in ``WHERE``, which supports:

- **Comparisons** — ``=``, ``<>`` (not equal), ``>``, ``<``, ``>=``, ``<=``:
  ``WHERE region = 'North'``, ``WHERE amount >= 50``.
- **Combinations** — ``AND`` (both must hold), ``OR`` (either), ``NOT``:
  ``WHERE region = 'North' AND amount > 100``.
- **Ranges and sets** — ``BETWEEN`` for ranges, ``IN`` for lists:
  ``WHERE amount BETWEEN 50 AND 200``, ``WHERE region IN ('North', 'South')``.
- **Pattern matching** — ``LIKE`` with wildcards for text:
  ``WHERE product LIKE 'Pro%'`` (products starting with "Pro").

Ordering the results
----------------------

A query can also sort its output with ``ORDER BY`` — the SQL counterpart of the
spreadsheet sort:

.. code-block:: sql

   SELECT product, amount
   FROM   orders
   WHERE  region = 'North'
   ORDER  BY amount DESC;

``ORDER BY amount DESC`` returns the rows largest-first; ``ASC`` (the default) is
smallest-first. Filtering with ``WHERE`` and ordering with ``ORDER BY`` together
are the SQL version of the spreadsheet's filter-then-sort — the same analytical
move, now at database scale.

Why queries beat exporting
----------------------------

A query goes to the data where it lives, returns only what you asked for, and is
**text** — repeatable, shareable, reviewable. Instead of exporting a million rows
to a spreadsheet and filtering there, you ask the database for the thousand that
matter. This is why SQL scales where spreadsheets strain, and why it is the most
consistently demanded analyst skill.

The caveat
------------

A query returns exactly what you specify — which is not always what you meant.
``WHERE amount > 100`` silently excludes rows where amount is *null* (missing),
and no error warns you; a ``LIKE`` pattern that is slightly off quietly returns
the wrong rows. SQL's precision demands precision from you, and the habit of
**checking results** — does the row count look right, do sampled rows match
expectations — applies to every query, exactly as the sanity-check habit applied
to spreadsheet formulas. The deeper query techniques come in the analysis
section; this is the foundation they extend.
"""

CONTENT["Organizing Data for Personal and Work Projects"] = r"""
Organisation is preparation too
---------------------------------

Preparing data is not only about the data's *content* — it is also about how the
*project* around it is organised: where files live, how they are named, which
versions are current. Poor project organisation causes a distinct and avoidable
class of problems — lost files, wrong versions analysed, work overwritten — that
have nothing to do with the analysis itself and everything to do with
discipline. This lesson closes the preparation section with the housekeeping
that keeps a data project sane.

The elements of an organised project
--------------------------------------

- **A sensible folder structure.** Separate raw data, working files, outputs,
  and documentation into clear folders (``raw/``, ``working/``, ``output/``,
  ``docs/``). The single most valuable convention is a **read-only ``raw/``
  folder** holding untouched original data — the raw-stays-raw rule from the
  spreadsheet lesson, applied at the project level, so the source is always
  recoverable.
- **Consistent file naming.** Names that are descriptive, consistent, and
  sortable: include what the file is, and a date in ``YYYY-MM-DD`` form
  (which sorts chronologically as text). ``2024-03-15_sales_north_raw.csv`` tells
  you content, date, and status at a glance; ``final_v3 (2).xlsx`` tells you
  nothing reliable.
- **Version control.** A deliberate way to track versions rather than a
  graveyard of ``final``, ``final_real``, ``final_ACTUAL``. Dated filenames help;
  proper version-control tools (as used for code) help more; the essential thing
  is *knowing which version is current* and never silently overwriting a version
  you might need.
- **Documentation.** A short README or notes file recording what the project
  is, where the data came from, what was done to it, and where things live — the
  chain-of-custody and metadata disciplines from earlier, at project scale.

Why this is analyst work, not clerical work
---------------------------------------------

Disorganisation is not a harmless untidiness; it produces *wrong analysis*.
Analysing an outdated file version, overwriting the only copy of a source, or
being unable to find the data behind a published number are real failures with
organisational, not analytical, causes — and they are entirely preventable. Good
organisation is also what makes work **reproducible** and **shareable**: a
colleague (or you in six months) can pick up a well-organised project and
understand it, where a disorganised one is a mystery even to its author.

The caveat
------------

Organisation is a means, not an end — it is possible to over-engineer a folder
structure or a naming scheme so elaborate that maintaining it costs more than it
saves, and rigid conventions nobody follows are worse than simple ones everybody
does. The goal is *enough* structure to keep the project findable, recoverable,
and reproducible, matched to its size — a quick analysis needs little, a
long-running project needs more. The final lesson of this section addresses one
more project-level concern: keeping shared data secure.
"""

CONTENT["Data Security in Spreadsheets"] = r"""
Protecting the data you hold
------------------------------

Preparing and organising data comes with an obligation to **protect** it —
especially when it is shared, and doubly so when it contains the personal
information the ethics and privacy lessons covered. Spreadsheets are among the
most shared data tools, which makes their **security** features worth knowing:
they are the practical controls that keep shared data from being seen by the
wrong people or changed by accident. This lesson closes the section, and the
preparation phase, on that protective note.

Why spreadsheet security matters
----------------------------------

Spreadsheets are easily shared, copied, and emailed — which is their
convenience and their risk. A sheet with customer data forwarded to the wrong
recipient is a privacy breach; a shared budget model where anyone can overtype a
formula is an accident waiting to happen; a public share link on sensitive data
exposes it to anyone who finds the URL. Security features exist to manage exactly
these risks.

The core protections
----------------------

- **Access control (sharing permissions).** The first line of defence: control
  *who* can open a sheet and *what* they can do — view only, comment, or edit.
  Grant the minimum access each person needs (the *minimise* principle from
  privacy), and prefer sharing with named people over open "anyone with the
  link" access for sensitive data.
- **Protected ranges and sheets.** Lock specific cells, ranges, or whole tabs so
  they cannot be edited, while leaving others open. This guards formulas and
  reference data from accidental change — the antidote to the silently-overtyped
  cell from the spreadsheet-errors lesson — while still letting collaborators
  enter data where intended.
- **Hiding versus protecting.** Hiding a sheet or column removes it from view but
  does **not** secure it — anyone can unhide it. Hiding is for tidiness;
  protection and permissions are for security. Confusing the two is a common and
  dangerous mistake.
- **Careful sharing of sensitive data.** For personal or confidential data,
  consider whether a spreadsheet is the right vessel at all, remove or mask
  identifiers where possible (aggregation and anonymisation from the privacy
  lesson), and be deliberate about link sharing and download permissions.

Security as part of preparation
---------------------------------

Security is not a separate concern bolted on at the end — it is part of handling
data responsibly throughout. Setting appropriate permissions when you share,
protecting the cells that must not change, and being careful with sensitive data
are habits that belong in every project, and they connect directly to the ethics
and privacy obligations that opened this section: protecting people's data is
both an ethical duty and a practical safeguard against costly mistakes.

The caveat
------------

Spreadsheet security has real limits: protected ranges deter accidents but are
not strong security against a determined actor, and once someone can *view*
data they can usually copy it, screenshot it, or download it regardless of other
restrictions. For genuinely sensitive data, spreadsheet features are a layer, not
a fortress — proper databases with real access controls, and organisational data
governance, are the stronger protections. Match the protection to the
sensitivity, and never assume a spreadsheet's controls make truly confidential
data safe to share widely.

This completes the Data Preparation section. You have moved from where data comes
from, through its types, structure, bias, ethics, and sources, to accessing,
querying, organising, and securing it. The data is now understood and in hand —
and the next section confronts the reality that it is almost never clean.
"""


MINDMAP.update({
    "Sorting and Filtering Data in Spreadsheets": [
        "Building and Organizing a Spreadsheet",
        "Importing Data into Spreadsheets",
        "Sorting and Filtering in Data Analysis",
        "Querying Data with SQL",
    ],
    "BigQuery Account Types": [
        "Databases and Relational Database Concepts",
        "Querying Data with SQL",
        "Introduction to SQL",
        "Accessing Data: Internal and External Sources",
    ],
    "Querying Data with SQL": [
        "Databases and Relational Database Concepts",
        "The Concept and Basic Use of SQL (Query Language)",
        "BigQuery Account Types",
        "Sorting and Filtering Data in Spreadsheets",
    ],
    "Organizing Data for Personal and Work Projects": [
        "Building and Organizing a Spreadsheet",
        "Data Security in Spreadsheets",
        "Importing Data into Spreadsheets",
        "The Importance of Clean Data",
    ],
    "Data Security in Spreadsheets": [
        "Data Privacy in Data Ethics",
        "Organizing Data for Personal and Work Projects",
        "Building and Organizing a Spreadsheet",
        "Data Ethics in Data Analysis",
    ],
})


# ======================================================================
# Section 4 — Data Cleaning / Stage: integrity  (cleaning 001-004)
# ======================================================================

GLOSS.update({
    "The Importance of Clean Data":
        "why clean data is the non-negotiable foundation every analysis stands on",
    "Data Integrity and Its Risks in Data Analysis":
        "keeping data accurate and consistent through its life — and what threatens it",
    "Aligning Data with Business Objectives":
        "checking that the data you have actually fits the question you must answer",
    "Handling Insufficient Data in Data Analysis":
        "recognising when there is not enough data, and the honest options when there isn't",
})

CONTENT["The Importance of Clean Data"] = r"""
The foundation everything stands on
-------------------------------------

Section 3 got the data understood and in hand; this section confronts the fact
that raw data is almost never ready to analyse. **Clean data** — data that is
complete, correct, consistent, and free of errors — is the non-negotiable
foundation of every trustworthy analysis, because the most sophisticated method
in the world produces wrong answers from wrong inputs. The principle has a name
old as computing: **garbage in, garbage out**.

What "clean" means
--------------------

Clean data satisfies several properties, each the absence of a specific defect:

- **Complete** — no critical values missing.
- **Accurate** — values correctly represent reality (a price of ``$1,000`` where
  reality was ``$100`` is inaccurate even though it is a valid number).
- **Consistent** — the same thing recorded the same way everywhere ("NY", "N.Y.",
  and "New York" not scattered as if three different places).
- **Unique** — no unintended duplicate records inflating the counts.
- **Valid** — values conform to their rules (a date in the date range, an age
  that is non-negative).
- **Uniform** — one unit and format throughout (all currency in dollars, all
  dates in one format).

Data failing any of these is **dirty**, and the next lessons catalogue the
specific ways.

Why it matters so much
------------------------

Dirty data does not announce itself — it produces plausible, confident, wrong
results, exactly like the bias it resembles. Duplicate records inflate totals;
inconsistent categories fragment a group so it looks smaller than it is; a
mistyped value skews an average; a missing-data pattern hides a real effect.
Because the analysis *runs* and the charts *render*, the error surfaces only when
a decision built on it goes wrong — often expensively, and long after the cause.
This is why cleaning is a first-class phase of the process, not a nuisance to
rush through.

The effort reality
--------------------

A well-known and sobering fact about real analytics work: analysts routinely
spend the **majority of a project's time** — commonly cited as most of it — on
finding and cleaning data, not on the glamorous analysis. Beginners are often
surprised; practitioners plan for it. The front-loaded-effort principle from the
foundations reaches its peak here: clean data is what makes every later step
meaningful, so the time spent securing it is the highest-leverage time in the
project.

The caveat
------------

"Clean" is relative to the *use*, not absolute — data clean enough for a rough
directional read may be too dirty for a financial report, and chasing perfect
cleanliness on data whose flaws do not affect the decision wastes the time real
problems need. The judgement (the speed-versus-accuracy trade-off, applied to
cleaning) is matching the cleaning effort to what the decision requires — and
being honest about the data's remaining flaws. The next lesson turns to the
principle that keeps data clean over time: integrity.
"""

CONTENT["Data Integrity and Its Risks in Data Analysis"] = r"""
Clean, and staying clean
--------------------------

Cleaning data once is not enough; data must *stay* accurate and consistent
through everything that happens to it. **Data integrity** is the accuracy,
completeness, consistency, and trustworthiness of data throughout its life cycle
— the property that data remains sound as it is stored, moved, transformed, and
combined. Where the previous lesson was about *achieving* clean data, integrity
is about *preserving* it against the many things that can quietly corrupt it.

What threatens integrity
--------------------------

Data integrity has recognisable enemies, and analysts meet them constantly:

- **Replication and syncing errors** — when data is copied between systems and
  the copies drift out of agreement, so the warehouse and the spreadsheet
  disagree about the same fact (the silo problem from the data-life-cycle
  lessons).
- **Transfer errors** — data corrupted or truncated while moving between systems
  or formats: a field cut short, an encoding mangled, rows dropped mid-import.
- **Transformation errors** — mistakes introduced *by processing itself*: a bad
  join that duplicates rows, a formula that drifts (the relative-reference bug),
  a conversion that loses precision. Cleaning done carelessly can *reduce*
  integrity.
- **Human error** — manual edits, mistyped corrections, accidental deletions —
  the reason the raw-stays-raw and access-control disciplines exist.
- **Storage and system issues** — corruption, incomplete writes, hardware
  faults.

The unifying theme: integrity is threatened most often not at rest but *in
motion* — every time data is copied, moved, transformed, or edited is a chance
for it to degrade.

Integrity and the cleaning process itself
-------------------------------------------

A subtle and important point: the act of cleaning data is itself a risk to
integrity. Every transformation you apply to fix one problem can introduce
another — dropping rows to remove duplicates might delete legitimate records, a
find-and-replace might over-match, a type conversion might silently fail. This is
precisely why the disciplines of this section exist: work on copies (raw stays
raw), document every change, and *verify* after each step. Cleaning without those
safeguards trades one integrity problem for another.

Why analysts must protect it
------------------------------

Integrity failures produce the same plausible-wrong results as dirty data and
bias, and they are often *harder* to spot because the data looked fine when it
arrived — the corruption happened in transit or in processing. An analyst who
assumes data kept its integrity through every hop is assuming away a major risk.
The habits that protect integrity — checking row counts before and after each
operation, validating that transformations did what was intended, keeping an
untouched original — are the same sanity-check and verification habits the course
has built throughout, applied to the data's whole journey.

The caveat
------------

Perfect integrity through a long pipeline is hard to guarantee, and some loss is
often invisible until something downstream breaks. The professional standard is
not a guarantee but *vigilance and traceability*: minimising the risky
operations, checking integrity at each step, and keeping enough of a trail
(untouched raw data, documented transformations) that when a problem does
surface, you can find where it entered. The next lessons turn to a different
integrity question — whether the data fits the objective, and whether there is
enough of it.
"""

CONTENT["Aligning Data with Business Objectives"] = r"""
Clean is not the same as useful
---------------------------------

Data can be perfectly clean and still be the *wrong data* for the question. A
flawless dataset that does not bear on the objective is a well-maintained
irrelevance. **Aligning data with business objectives** is the check that the
data you have — however clean — actually fits the decision it is meant to inform,
and it connects the cleaning phase back to the business task the framing section
made central.

The alignment questions
-------------------------

Before investing in cleaning and analysing a dataset, confirm it aligns with the
objective:

- **Relevance** — does this data actually relate to the question? The
  relevance check from the choosing-data lesson, applied now to data in hand.
- **Coverage** — does it span the population and time period the objective
  concerns? Data covering only part answers only part.
- **Granularity** — is it detailed enough? A question about daily patterns needs
  daily data; monthly totals cannot answer it no matter how clean.
- **Definitions** — do the data's fields *mean* what the objective needs? A
  "customer" field defined as "anyone who registered" cannot answer a question
  about "anyone who purchased" — the two populations differ.
- **Currency** — is it recent enough to reflect the reality the decision acts on?

A dataset can be immaculately clean and fail any of these — which is why
alignment is a separate check from cleanliness.

The misalignment trap
-----------------------

The costly failure this prevents: spending significant effort cleaning and
analysing data, producing a polished result, and only then discovering it does
not answer the actual question — the data measured a subtly different thing,
missed a key segment, or was too coarse. This is expensive precisely because the
work looked productive throughout; the misalignment was invisible until the end.
Checking alignment *before* the heavy cleaning investment is cheap insurance
against it, and it is a natural gate at the start of the cleaning phase.

Aligning when data falls short
--------------------------------

When data does not fully align, the options mirror those for a flawed source:
find better-aligned data if it exists; supplement the data to cover its gaps;
adjust the *question* to what the available data can honestly answer; or proceed
with explicit acknowledgment of the misalignment and its effect on the
conclusion. What is never acceptable is quietly analysing misaligned data as
though it answered the original question — the same honesty obligation the
fairness thread has demanded throughout.

The caveat
------------

Alignment is a matter of degree, not a binary — data rarely fits an objective
perfectly, and demanding perfect alignment would stall most analysis. The skill
is judging whether the data is *aligned enough* for the decision's stakes, and
being explicit about where it falls short, so the conclusion is read with the
right caution. This connects to the next lesson's question: sometimes the problem
is not that data is misaligned, but that there is simply not enough of it.
"""

CONTENT["Handling Insufficient Data in Data Analysis"] = r"""
When there just is not enough
-------------------------------

Sometimes the honest finding of the Prepare phase is that there is **not enough
data** to answer the question reliably — too few records, too short a time span,
too many gaps, or coverage too thin for the population involved. Recognising
insufficiency, and responding to it honestly, is a mark of a good analyst; the
failure mode is forcing a confident answer from data that cannot support one.

What insufficient data looks like
-----------------------------------

Insufficiency takes several forms:

- **Too few records** — a sample so small that results are dominated by chance
  rather than signal (the sample-size and statistical-power questions the next
  lessons formalise).
- **Too short a time span** — data covering too brief a period to reveal trends
  or account for seasonality, so a temporary blip reads as a pattern.
- **Missing data** — gaps within the dataset, whether scattered blanks or whole
  segments absent, that undermine completeness.
- **Thin coverage** — too little data about a relevant subgroup to say anything
  reliable about it, even if the overall dataset is large.

The common thread: insufficient data cannot bear the *weight* of the conclusion
the question demands — the answer would rest on too little to be trusted.

The options when data is insufficient
---------------------------------------

Standard, honest responses — roughly in order of preference:

- **Collect more data** — the direct fix when time and resources permit: extend
  the collection period, gather a larger sample, fill the gaps.
- **Find additional or alternative sources** — supplement the insufficient data
  with other data that helps answer the question.
- **Adjust the question** — narrow it to what the available data *can* answer
  reliably. A question about a thinly-covered subgroup might become a question
  about the whole, honestly scoped.
- **Wait** — sometimes the right answer is that the analysis cannot yet be done
  well, and forcing it now would mislead.
- **Proceed with explicit limitations** — when a decision must be made and no
  more data is available, provide the best analysis possible *with clear,
  prominent caveats* about what the data cannot support.

Handling missing data specifically
------------------------------------

Missing values within an otherwise sufficient dataset have their own standard
handling — removing affected records, or filling gaps with reasonable estimates
— each with trade-offs the cleaning lessons develop. The essential discipline is
to handle missing data *deliberately and transparently*, documenting what was
done, rather than letting gaps silently distort results.

The analyst's honesty obligation
----------------------------------

The hardest and most important response to insufficient data is sometimes saying
so: telling a stakeholder "the data cannot reliably answer this" when they wanted
an answer. This is not failure — it is exactly the integrity the conflict and
fairness lessons demanded. A confident answer built on insufficient data is worse
than an honest "we don't have enough to know," because it drives a decision with
false certainty. The professional move is to be clear about what the data *can*
support and what it cannot.

The caveat
------------

"Insufficient" is relative to the question and its stakes, not absolute — data
too thin for a high-stakes irreversible decision may be entirely adequate for a
low-stakes directional one. The judgement is matching the *sufficiency bar* to
what the decision requires (the speed-versus-accuracy trade-off again), and being
honest about which side of that bar the data falls on. The next lessons make
sufficiency precise, with the mathematics of populations, samples, and how much
data is enough.
"""


MINDMAP.update({
    "The Importance of Clean Data": [
        "Understanding Data Analysis",
        "Data Integrity and Its Risks in Data Analysis",
        "Dirty Data vs. Clean Data",
        "The Importance of Clean Data (revisited)",
    ],
    "Data Integrity and Its Risks in Data Analysis": [
        "The Importance of Clean Data",
        "Aligning Data with Business Objectives",
        "Verifying and Reporting Data Integrity",
        "Sample Size and Data Integrity",
    ],
    "Aligning Data with Business Objectives": [
        "The Role of Business Tasks in Data Analysis",
        "Choosing the Right Data to Collect",
        "Handling Insufficient Data in Data Analysis",
        "Data Integrity and Its Risks in Data Analysis",
    ],
    "Handling Insufficient Data in Data Analysis": [
        "Aligning Data with Business Objectives",
        "Population, Sample Size, and Random Sampling",
        "Sampling Bias and Unbiased Data",
        "Statistical Power in Data Analysis",
    ],
})


# ======================================================================
# Section 4 — Data Cleaning / integrity (statistics)  (cleaning 005-008)
# ======================================================================

GLOSS.update({
    "Population, Sample Size, and Random Sampling":
        "the whole vs. the part: population, sample, and how to sample fairly",
    "Statistical Power in Data Analysis":
        "the chance of detecting a real effect when there is one — and why 0.8 is the norm",
    "Sample Size and Data Integrity":
        "how much data is enough: confidence, precision, and the cost of too little",
    "Margin of Error":
        "the plus-or-minus around a sample estimate, and how to read it honestly",
})

CONTENT["Population, Sample Size, and Random Sampling"] = r"""
The whole and the part
------------------------

The insufficient-data lesson raised the question of *how much* data is enough;
answering it needs vocabulary. The **population** is the entire group you want to
understand — every customer, every transaction, every citizen. A **sample** is
the subset you actually examine, and the **sample size** is how many members it
contains. Almost all analysis works from samples, because examining the whole
population is usually impossible or impractical — which makes *how you sample*
one of the most consequential decisions in the whole process.

Why we sample
--------------

Studying an entire population is often infeasible: too large, too expensive, too
slow, or simply inaccessible (you cannot survey every possible future customer).
A well-chosen sample lets you draw reliable conclusions about the whole from a
manageable part — the core bargain of statistics. The bargain only holds,
though, when the sample **represents** the population, which is exactly where
sampling method matters.

Random sampling and its relatives
-----------------------------------

**Random sampling** selects members so that every member of the population has an
**equal chance** of being chosen. Its purpose is representativeness *by design*:
because selection does not depend on any characteristic, the sample tends to
mirror the population's mix, and the systematic lean that defines sampling bias
is removed. Common sampling approaches:

- **Simple random sampling** — every member equally likely; the baseline.
- **Stratified sampling** — divide the population into groups (strata) and sample
  from each, ensuring every group is represented in proportion — useful when some
  subgroups are small but important.
- **Systematic sampling** — select every *n*-th member from an ordered list; a
  practical approximation of random when the list has no hidden pattern.
- **Cluster sampling** — divide into clusters, randomly select whole clusters;
  efficient when the population is naturally grouped (e.g. by location).

The contrast case remains the biased methods from earlier — convenience and
self-selection — which do *not* give everyone an equal chance and therefore skew.

The role of sample size
-------------------------

Given a fair method, **size** governs *precision*: larger samples yield estimates
closer to the population truth, with less variability from the luck of the draw.
But size cannot fix bias — a large biased sample is confidently wrong, as the
1936-poll lesson showed. The two requirements are separate and both necessary: a
sample must be **representative** (right method) *and* **large enough** (right
size) to support reliable conclusions. The next lessons quantify "large enough."

The caveat
------------

Random sampling is the ideal, but genuinely random samples are hard to achieve
in practice — the sampling frame (the list you draw from) may itself omit part of
the population, reintroducing bias no amount of randomness within the frame can
fix. A "random" sample of phone numbers still misses people without phones.
The professional habit is to scrutinise not just *how* members were selected but
*what population the frame actually covers* — and to disclose the gap. The next
lesson turns from representativeness to a different sufficiency question: the
power to detect a real effect.
"""

CONTENT["Statistical Power in Data Analysis"] = r"""
The chance of seeing what is there
------------------------------------

Having enough representative data is not only about precision — it is also about
**sensitivity**: the ability to *detect a real effect when one exists*. That
sensitivity has a name. **Statistical power** is the probability that an analysis
will correctly identify a real effect (a genuine difference or relationship) when
it is truly present. Low power means real effects go undetected — the analysis
concludes "nothing here" when something was there — which is one of the quieter
and more dangerous failures in data work.

What power means, concretely
------------------------------

Imagine testing whether a new checkout design genuinely improves conversion. If
the improvement is real but your sample is too small, the difference can be
swamped by random noise, and the test comes back inconclusive — a **false
negative**. Statistical power is the probability of *avoiding* that outcome:
of detecting the improvement given that it is real. Power ranges from 0 to 1, and
the widely used convention is a target of **0.8 (80%)** — meaning an 80% chance
of detecting a true effect of the size you care about, and correspondingly a 20%
chance of missing it. Higher power is better but costs more data.

What power depends on
-----------------------

Four things move statistical power, and they trade off against each other:

- **Sample size** — the lever analysts most directly control. Larger samples
  give higher power; this is the main reason "enough data" matters for detecting
  effects, not just for precision.
- **Effect size** — how large the real effect is. Big effects are easy to detect
  with modest samples; subtle effects need large ones. Detecting a 0.2%
  conversion lift takes far more data than detecting a 20% lift.
- **Significance threshold** — how strict you are about false positives; stricter
  thresholds lower power for a given sample.
- **Variability** — noisier data lowers power, which is another reason clean,
  consistent data matters.

The practical upshot: to detect a small effect reliably, you need a large sample,
and if you cannot get one, you must accept that small real effects may remain
invisible to your analysis.

Why analysts should care
--------------------------

Power connects directly to the insufficient-data lesson. A common, invisible
error is running an analysis with too little data, finding "no significant
effect," and concluding there *is* no effect — when in truth the study simply
lacked the power to see it. **Absence of evidence is not evidence of absence**:
an underpowered "no effect" result means "we could not detect one," not "there is
none." Knowing roughly how much data an effect of a given size requires — before
running the analysis — is what separates a genuine null result from an
underpowered one.

The caveat
------------

Power analysis involves assumptions — chiefly a guess at the effect size you are
looking for, which you often do not know in advance — so power calculations are
estimates, not guarantees, and the conventional 0.8 target is a norm, not a law
of nature. The value of thinking about power is less the exact number than the
discipline it enforces: asking, *before* collecting or analysing, whether the
data could plausibly reveal the effect you care about. The next lesson ties
sample size directly to data integrity.
"""

CONTENT["Sample Size and Data Integrity"] = r"""
How much is enough?
--------------------

The recurring question of this stage — *how much data is enough?* — is itself a
matter of **data integrity**, because a conclusion drawn from too small a sample
is untrustworthy no matter how clean each record is. Sample size determines how
much confidence you can place in a result, and choosing it well is part of
ensuring the analysis is sound rather than merely precise-looking.

Sample size and confidence
-----------------------------

The core relationship: **larger samples give more reliable estimates** of the
population, because the random variation from the luck of the draw shrinks as the
sample grows. A tiny sample can land far from the population truth by chance
alone; a large one is pulled close. This is why a result from five customers is
suggestive at best while a result from five thousand can be trusted — the same
question, very different weight of evidence.

But — and this echoes through the whole section — **more data reduces random
error, not bias**. Increasing the size of a *biased* sample makes the estimate
more precisely wrong; it converges confidently on the wrong answer. Size and
representativeness are independent requirements: integrity needs both a fair
method *and* an adequate size.

What determines the size you need
-----------------------------------

The sample size required for a trustworthy conclusion depends on several factors:

- **The confidence you need** — higher confidence in the result requires more
  data.
- **The precision you need** — a tighter margin of error (the next lesson)
  requires a larger sample.
- **The population's variability** — more variable populations need larger
  samples to pin down.
- **The effect size** — detecting small effects requires more data than
  detecting large ones (the power lesson).
- **The stakes** — higher-stakes decisions justify the cost of collecting more.

These combine so that "enough" is not a single number but a judgement tied to the
question. Statistical tools and sample-size calculators formalise the
relationship; the analyst's job is to recognise that the question *has* an answer
and that ignoring it — grabbing whatever sample is convenient — risks conclusions
the data cannot support.

Sample size in the cleaning phase
-----------------------------------

Why does this live in a cleaning section? Because judging sample sufficiency is
part of assessing whether data is fit to analyse — the integrity check that sits
alongside checking for errors and inconsistencies. A dataset can be spotlessly
clean and still too small to answer its question reliably; recognising that
before analysis, and either gathering more or scoping the conclusion accordingly,
is part of preparing data responsibly.

The caveat
------------

Bigger is not limitlessly better: beyond the size needed for the required
confidence and precision, additional data yields diminishing returns while
costing time and money, and very large samples can make *trivial*, unimportant
differences appear "statistically significant" — significance is not the same as
importance. The goal is the *right* size for the question and its stakes — large
enough to trust, not so large that you over-invest or mistake a tiny effect for a
meaningful one. The next lesson makes the precision side of this concrete with
the margin of error.
"""

CONTENT["Margin of Error"] = r"""
The plus-or-minus on an estimate
----------------------------------

Any number computed from a sample is an *estimate* of the population's true
value, not the value itself — and honesty requires expressing how far off it
might be. The **margin of error** is the range above and below a sample estimate
within which the true population value is likely to fall. A survey reporting "48%
support, ±3%" is saying the true figure is likely between 45% and 51% — the ±3%
is the margin of error, and reporting it is the difference between an honest
estimate and a false precision.

Reading a margin of error
---------------------------

The margin of error is always paired with a **confidence level** — commonly 95% —
which states *how likely* the true value is to fall within the margin. "48% ±3%
at 95% confidence" means: if the sampling were repeated many times, about 95% of
the resulting intervals would contain the true population value. The two numbers
work together — a margin of error without a confidence level is incomplete, and a
confidence level tells you how much trust to place in the ± range.

The practical reading is a discipline against over-interpretation. If a poll
shows candidate A at 48% ±3% and candidate B at 46% ±3%, the intervals overlap
(45–51% versus 43–49%) — the "2-point lead" is **within the margin of error** and
therefore not a reliable difference at all. Treating overlapping estimates as a
real gap is one of the most common misreadings of sampled data, and knowing the
margin of error is what prevents it.

What drives the margin
------------------------

The margin of error is governed chiefly by:

- **Sample size** — the dominant factor. Larger samples shrink the margin (with
  diminishing returns — roughly, quadrupling the sample halves the margin), which
  is the precise sense in which "more data" buys precision.
- **Confidence level** — demanding higher confidence *widens* the margin for a
  given sample: to be more certain the interval contains the truth, you must make
  the interval bigger.
- **Variability** — more variable populations produce larger margins.

This ties the stage together: sample size, confidence, and precision (the margin)
are three faces of one relationship, and fixing any two constrains the third.

Why it matters for integrity and communication
-------------------------------------------------

The margin of error is where statistical honesty meets communication. Reporting a
sample estimate *without* its margin implies a precision the data does not have —
"support is 48%" sounds exact; "48% ±3%" tells the truth. An analyst who omits
the margin of error, or a stakeholder who ignores it, will read noise as signal
and make decisions on differences that are not real. Presenting estimates *with*
their uncertainty is the honest-communication obligation from Section 2, applied
to sampled numbers.

The caveat
------------

The margin of error captures only **sampling error** — the uncertainty from
examining a sample rather than the whole population. It says *nothing* about
**bias**: a biased sample can report a tight margin of error around the wrong
value, and the small ± lends false reassurance. A ±1% margin on a
self-selected online poll is precisely wrong. Margin of error quantifies one kind
of uncertainty; the biases and dirty-data problems of this section are others it
cannot see. This completes the data-integrity stage — the question of whether
there is enough sound data — and the next stage turns to the dirty data itself,
and how to clean it.
"""


MINDMAP.update({
    "Population, Sample Size, and Random Sampling": [
        "Handling Insufficient Data in Data Analysis",
        "Sampling Bias and Unbiased Data",
        "Statistical Power in Data Analysis",
        "Sample Size and Data Integrity",
    ],
    "Statistical Power in Data Analysis": [
        "Population, Sample Size, and Random Sampling",
        "Sample Size and Data Integrity",
        "Margin of Error",
        "Handling Insufficient Data in Data Analysis",
    ],
    "Sample Size and Data Integrity": [
        "Population, Sample Size, and Random Sampling",
        "Statistical Power in Data Analysis",
        "Margin of Error",
        "Data Integrity and Its Risks in Data Analysis",
    ],
    "Margin of Error": [
        "Sample Size and Data Integrity",
        "Statistical Power in Data Analysis",
        "Population, Sample Size, and Random Sampling",
        "Balancing Speed and Accuracy in Data Analysis",
    ],
})


# ======================================================================
# Section 4 — Data Cleaning / Stage: dirty  (cleaning 009-012)
# ======================================================================

GLOSS.update({
    "Dirty Data vs. Clean Data":
        "the concrete contrast: what makes data dirty, and what clean looks like beside it",
    "The Importance of Clean Data (revisited)":
        "the business cost of dirty data — why cleaning is worth the time it takes",
    "Common Issues in Dirty Data":
        "the recurring defects — duplicates, missing, inconsistent, wrong type, outliers",
    "Data Cleaning with Spreadsheets":
        "the practical spreadsheet cleaning workflow, defect by defect",
})

CONTENT["Dirty Data vs. Clean Data"] = r"""
Two states of the same table
------------------------------

The integrity stage asked whether there is *enough* sound data; this stage
confronts the data's *quality* directly. **Dirty data** is data that is
incomplete, incorrect, inconsistent, duplicated, or otherwise flawed — data with
errors that make it unreliable to analyse. **Clean data** is its opposite: the
complete, accurate, consistent, unique, valid, and uniform data the first lesson
defined. Putting the two side by side makes the abstract quality criteria
concrete and shows exactly what cleaning must fix.

The contrast, defect by defect
--------------------------------

The same customer table, dirty versus clean:

.. code-block:: text

   DIRTY                              CLEAN
   name        city        spend      name        city        spend
   Jane Smith  New York    100        Jane Smith  New York     100
   jane smith  new york    100        (duplicate removed)
   Bob Jones   NY          -50        Bob Jones   New York       0
   Ann Lee                 250        Ann Lee     Boston       250
   Tom Web     Chicago     1O0        Tom Web     Chicago      100

Reading the defects: rows 1–2 are a **duplicate** (same person, different
casing); "NY"/"New York"/"new york" is an **inconsistency**; the ``-50`` spend is
**invalid** (negative); Ann Lee's blank city is **missing**; and ``1O0`` (letter
O for zero) is a **type error** masquerading as a number. Each is a specific,
nameable defect — and each would distort an analysis in a specific way: the
duplicate inflates counts, the inconsistency fragments the New York group, the
invalid value skews the average, the missing value breaks completeness, and the
type error breaks arithmetic.

Why the distinction is the whole job
--------------------------------------

Cleaning is precisely the work of moving data from the left column to the right —
identifying each defect and correcting it so the data becomes trustworthy. Naming
the state matters because "this data is dirty" is not actionable, but "this column
has duplicates, inconsistent categories, and three negative values" is a work
list. The rest of this stage catalogues the defects and the techniques for each;
this lesson establishes the target — what clean looks like — that all of it aims
at.

The caveat
------------

The dirty/clean line is not always obvious: is a ``-50`` spend an error, or a
legitimate refund? Is a blank a missing value, or a genuine "none"? Is an unusual
value a mistake, or a real outlier worth keeping? Cleaning requires *judgement
and context*, not blind rule-following — the same value can be dirt to remove or
signal to preserve depending on what it means, which is why understanding the
data (metadata, provenance, the business) precedes cleaning it. The next lesson
revisits *why* this painstaking work is worth doing.
"""

CONTENT["The Importance of Clean Data (revisited)"] = r"""
Why this is worth the time
----------------------------

It is worth returning to the importance of clean data now that its defects are
concrete — because cleaning is tedious, time-consuming work, and an analyst under
deadline will be tempted to skip it. This lesson makes the case that the tedium
is worth it, not on principle but on **cost**: dirty data is not a cosmetic
problem but an expensive one, and cleaning is among the highest-return activities
in the whole process.

The cost of dirty data
------------------------

Dirty data is expensive in ways that compound:

- **Wrong decisions.** The direct cost: analysis on dirty data yields
  conclusions that are confidently wrong, and decisions built on them fail —
  a duplicated-record count that overstates demand, an inconsistent category that
  hides a struggling segment. The decision's cost is the data's cost.
- **Wasted downstream work.** Every hour of analysis, every chart, every
  presentation built on dirty data is wasted when the dirt surfaces — and worse,
  the rework to redo it correctly. Cleaning *first* is cheaper than discovering
  the problem *last*.
- **Eroded trust.** When a stakeholder catches an error, they distrust not just
  that number but the analyst and the whole analysis — trust that is slow to
  rebuild. One visible dirty-data mistake can discredit months of sound work.
- **Compounding downstream.** Dirty data feeding other systems, reports, or
  models propagates the error everywhere it flows — the integrity-in-motion
  problem, at organisational scale.

Industry commentary has long put large price tags on poor data quality precisely
because these costs accumulate quietly across an organisation until they surface
as failed decisions.

Why cleaning is high-return
-----------------------------

Set the cost of *not* cleaning against the cost of cleaning, and the return is
clear. Cleaning is time-consuming but *bounded* — a known, front-loaded
investment — while the cost of dirty data is *unbounded and compounding*,
surfacing unpredictably in failed decisions and lost trust. This is why the
front-loaded-effort principle holds so strongly here: the time spent cleaning is
not overhead subtracted from "real" analysis, it is the investment that makes the
real analysis *worth anything*. Clean data is the highest-leverage output of the
preparation phase.

The discipline the cost justifies
-----------------------------------

Because the stakes are real, cleaning deserves the same rigour as analysis, not a
hurried pass: work on copies so the raw survives, document every change so the
work is reproducible and reviewable, and verify after each step so a fix does not
introduce a new defect. These disciplines — developed through this section — are
what convert cleaning from risky manual surgery into a sound, trustworthy process.
The cost of dirty data is exactly what makes them worth following.

The caveat
------------

The cost argument cuts both ways: just as under-cleaning is expensive, so is
*over*-cleaning — polishing data far beyond what the decision requires, or
"correcting" values that were actually legitimate, both waste time and can
introduce errors. The return on cleaning is highest for the defects that would
*change the decision*, and lower for cosmetic imperfections that would not. Match
the cleaning effort to the stakes (the speed-versus-accuracy trade-off, once
more), clean what matters thoroughly, and do not gold-plate what does not. The
next lessons get specific about the defects worth finding.
"""

CONTENT["Common Issues in Dirty Data"] = r"""
A catalogue of defects
------------------------

Dirty data is dirty in recurring, recognisable ways, and knowing the catalogue is
what lets an analyst *hunt* defects systematically rather than stumble on them.
A handful of issue types account for the large majority of real data problems;
each has a signature and a standard remedy, previewed here and developed in the
hands-on lessons that follow.

The common issues
-------------------

- **Duplicate data** — the same record appearing more than once, whether
  identical copies or near-duplicates (the same customer with slight spelling
  differences). Duplicates inflate counts and over-weight the repeated records.
  *Remedy:* identify and remove duplicates, carefully — deciding which copy to
  keep.
- **Missing data** — absent values, from scattered blank cells to whole missing
  fields or records. Missing data breaks completeness and can bias results if the
  gaps are not random. *Remedy:* remove affected records, or fill (impute) with a
  reasonable estimate — each with trade-offs.
- **Inconsistent data** — the same thing recorded different ways: "NY"/"New
  York", "St."/"Street", mixed date formats, inconsistent capitalisation.
  Inconsistency fragments what should be one group. *Remedy:* standardise to a
  single consistent form.
- **Incorrect / invalid data** — values that violate their rules or reality: a
  negative age, a date in the future, a price of zero where zero is impossible.
  *Remedy:* identify against validity rules and correct or remove.
- **Wrong data type** — values stored as the wrong type: numbers as text, dates
  as text (the import problems from the prep section), which break calculations.
  *Remedy:* convert to the correct type.
- **Outliers** — values far outside the normal range. These are the *subtle* case:
  an outlier may be an **error** to fix or a **genuine extreme** to keep, and
  telling which requires context. *Remedy:* investigate — do not delete blindly.
- **Extra whitespace and formatting** — leading/trailing spaces and stray
  characters that make "New York " differ from "New York" to a computer.
  *Remedy:* trim and standardise.

The systematic hunt
---------------------

The value of the catalogue is that it becomes a **checklist**. Faced with a new
dataset, an analyst runs down the list — check for duplicates, scan for missing
values, look for inconsistent categories, validate ranges, confirm types, examine
outliers, trim whitespace — rather than hoping problems reveal themselves.
Systematic checking finds defects that ad-hoc glancing misses, especially the
quiet ones (a few duplicates in a million rows, one inconsistent spelling) that
never announce themselves but still distort results.

The caveat
------------

The remedies involve judgement, and each carries its own risk — removing
duplicates can delete legitimate records, imputing missing values can invent
data, "correcting" an outlier can erase a real finding. The catalogue tells you
*what to look for*; deciding *what to do* about each defect requires
understanding what the value means and what the analysis needs. Blind application
of cleaning rules is itself a source of error, which is why the hands-on lessons
that follow stress verification and documentation at every step. The next lesson
begins that hands-on work in the spreadsheet.
"""

CONTENT["Data Cleaning with Spreadsheets"] = r"""
Cleaning, hands on
--------------------

With the defects catalogued, this lesson turns to actually *fixing* them — and
the spreadsheet, the analyst's most accessible tool, is where hands-on cleaning
most often begins. Spreadsheets offer built-in features for each common defect,
and their visibility — you *see* every change — makes them an excellent place to
learn cleaning before the same operations scale up to SQL and Python.

Spreadsheet cleaning, defect by defect
----------------------------------------

- **Duplicates** — the *Remove Duplicates* feature finds and deletes identical
  rows in one operation; conditional formatting can *highlight* duplicates first
  so you see them before removing.
- **Inconsistent text** — *Find and Replace* standardises variants ("NY" →
  "New York") across the sheet; *TRIM* removes stray leading and trailing spaces;
  *UPPER*/*LOWER*/*PROPER* normalise capitalisation.
- **Missing values** — *filtering* to blanks isolates the affected rows so you
  can decide per column whether to remove or fill them; sorting brings blanks
  together for inspection.
- **Wrong types** — the type-conversion tools and format settings turn
  text-numbers into real numbers and text-dates into real dates (the import-trap
  fixes from the prep section).
- **Splitting and combining** — *Text to Columns* splits a crammed column ("New
  York, NY" → city and state); *CONCATENATE* / ``&`` combines columns when
  needed — restoring the one-variable-per-column tabular structure.
- **Outliers and invalid values** — *sorting* a column surfaces the extremes at
  top and bottom for inspection; *conditional formatting* flags values outside a
  valid range for review.

The disciplined workflow
--------------------------

The features are only half the lesson; *how* you apply them is the other half,
and it is where the section's integrity disciplines become concrete:

1. **Work on a copy.** Duplicate the raw data to a working tab and clean *there*
   — raw stays raw, so a mistake is never fatal.
2. **Clean one defect at a time.** Handle duplicates, then inconsistencies, then
   types — in steps, each verifiable, rather than a tangled all-at-once edit.
3. **Verify after each step.** Check the row count after removing duplicates,
   spot-check replaced values, confirm converted types compute — catch a bad fix
   immediately, not three steps later.
4. **Document what you did.** Record each cleaning action in a notes tab, so the
   work is reproducible and a reviewer (or you, later) can see exactly what
   changed and why.

This workflow is what separates trustworthy cleaning from risky manual surgery —
the same fix applied carelessly can corrupt data, while applied with copies,
steps, verification, and documentation it reliably improves it.

The caveat
------------

Spreadsheet cleaning is powerful for modest data but shares the spreadsheet's
limits: it strains on very large datasets, and — most dangerously — manual
cleaning is **not automatically reproducible**. A series of hand-applied
Find-and-Replaces and deletions leaves no record unless you *make* one, so
re-cleaning next month's data means redoing the work from memory, with the risk
of doing it differently. This is precisely why cleaning graduates to SQL and
Python (and documented, rerunnable scripts) as data grows and recurs — a theme
the following lessons develop. For now, the spreadsheet is where the cleaning
intuitions form, one visible fix at a time.
"""


MINDMAP.update({
    "Dirty Data vs. Clean Data": [
        "The Importance of Clean Data",
        "The Importance of Clean Data (revisited)",
        "Common Issues in Dirty Data",
        "Data Integrity and Its Risks in Data Analysis",
    ],
    "The Importance of Clean Data (revisited)": [
        "The Importance of Clean Data",
        "Dirty Data vs. Clean Data",
        "Common Issues in Dirty Data",
        "Data Cleaning with Spreadsheets",
    ],
    "Common Issues in Dirty Data": [
        "Dirty Data vs. Clean Data",
        "Data Cleaning with Spreadsheets",
        "Using Spreadsheet Functions for Data Cleaning",
        "Understanding Bias in Data Analysis",
    ],
    "Data Cleaning with Spreadsheets": [
        "Common Issues in Dirty Data",
        "Spreadsheet Tools for Data Cleaning",
        "Using Spreadsheet Functions for Data Cleaning",
        "Cleaning and Merging Multiple Datasets",
    ],
})


# ======================================================================
# Section 4 — Data Cleaning / dirty (cont.)  (cleaning 013-016)
# ======================================================================

GLOSS.update({
    "Cleaning and Merging Multiple Datasets":
        "combining data from several sources cleanly — matching keys, reconciling formats",
    "Spreadsheet Tools for Data Cleaning":
        "the built-in cleaning toolkit: dedupe, split, find/replace, validation, formatting",
    "Using Spreadsheet Functions for Data Cleaning":
        "TRIM, CLEAN, UPPER, SUBSTITUTE, VALUE and friends — cleaning by formula",
    "Viewing Data Differently for More Effective Data Cleaning":
        "sort, filter, pivot, conditional formatting as lenses that reveal hidden defects",
})

CONTENT["Cleaning and Merging Multiple Datasets"] = r"""
When cleaning means combining
-------------------------------

Real analysis rarely draws on a single tidy table; it usually **combines**
several — sales from one system, customers from another, products from a third.
Merging datasets is powerful, but it multiplies cleaning challenges, because now
the data must be consistent not only *within* each source but *across* them.
This lesson covers the distinct problems that arise when data from multiple
sources is brought together.

The merge and its prerequisite
--------------------------------

Merging combines datasets by matching records on a shared field — a **key** that
identifies the same entity across tables (a ``customer_id`` present in both an
orders table and a customers table). The relational-database concepts from the
prep section are exactly this: primary and foreign keys are what make a clean
merge possible. The prerequisite for a correct merge is that the key **means the
same thing and is formatted the same way** in every dataset — and that is where
most merge problems live.

The cross-dataset cleaning problems
-------------------------------------

- **Mismatched keys.** The join field is formatted differently across sources —
  ``"C-001"`` in one, ``"001"`` in another, or one storing IDs as numbers and
  the other as text. Records that *should* match silently do not, and rows go
  missing from the result. *Fix:* standardise the key's format in every source
  before merging.
- **Inconsistent formats.** The same field recorded differently across sources —
  dates as ``MM/DD/YYYY`` here and ``YYYY-MM-DD`` there, categories labelled
  differently, units differing. *Fix:* reconcile to one format before combining.
- **Duplicate and conflicting records.** The same entity appearing in multiple
  sources, sometimes with *disagreeing* values (two addresses for one customer).
  *Fix:* decide a rule for which source wins, and deduplicate after merging.
- **Different granularity.** One dataset is per-transaction, another is
  per-day-summary; merging them naively double-counts or misaligns. *Fix:*
  aggregate to a common grain first.
- **Structural mismatch.** Different column names for the same thing, or wide
  versus long layouts (the prep-section shapes). *Fix:* align structure and
  naming before the merge.

The disciplined merge
-----------------------

A safe merge follows a sequence: **clean each source individually first** (so
you are combining already-tidy data), **standardise the join key** across
sources, **verify the match** (how many records matched, how many did not, and
*why* the unmatched failed), and **check the result's row count** against
expectation — a merge that unexpectedly multiplies rows usually signals a
duplicate key or a many-to-many join gone wrong. Verifying the merge is as
important as performing it, because a bad merge silently drops or duplicates
data.

The caveat
------------

Merging is where the *integrity-in-motion* risk from earlier peaks: combining
datasets is a transformation, and transformations introduce errors — a join on a
non-unique key can explode row counts, a format mismatch can silently drop
records that belong. The row-count check before and after is not optional
housekeeping; it is the primary defence against a merge that quietly corrupted
the data. The deeper mechanics of combining data — VLOOKUP in spreadsheets, JOIN
in SQL — get full treatment in the analysis section; here the point is that
*cleaning must precede and follow every merge*.
"""

CONTENT["Spreadsheet Tools for Data Cleaning"] = r"""
The built-in cleaning toolkit
-------------------------------

Beyond formulas, spreadsheets ship with dedicated *features* for cleaning —
menu-driven tools that handle common defects without writing anything. Knowing
the toolkit means reaching for the right built-in instead of doing by hand what
the software does in one click, and it complements the formula-based cleaning of
the next lesson.

The core cleaning features
----------------------------

- **Remove Duplicates** — scans selected columns and deletes rows that repeat,
  in one operation. You choose *which columns* define a duplicate (all columns
  for exact copies, or a key column like email to catch near-duplicates).
- **Text to Columns** — splits a single column into several by a delimiter,
  turning ``"New York, NY"`` into separate city and state columns — restoring the
  one-variable-per-column structure.
- **Find and Replace** — standardises values across the sheet, replacing every
  ``"N/A"`` with a blank, every ``"St."`` with ``"Street"``. Its scope and
  match-case options control precisely what changes.
- **Data Validation** — sets *rules* for what a cell may contain (a date in a
  range, a value from a fixed list, a positive number), which both catches
  existing invalid data and *prevents* new bad entries — cleaning that works
  going forward, not just once.
- **Conditional Formatting** — colours cells by rule to *reveal* defects
  visually: highlight duplicates, flag values outside a valid range, mark blanks
  — making problems visible before you fix them.
- **Sort and Filter** — bring defects together for inspection: sort to surface
  extremes and group blanks, filter to isolate the rows meeting (or failing) a
  condition (the next lesson develops these as diagnostic lenses).

Tools versus formulas
-----------------------

The built-in tools and the cleaning *functions* (next lesson) overlap but suit
different jobs. Tools are **one-time, menu-driven, and immediate** — best for a
one-off clean where you apply the operation and move on. Functions are
**formula-based and live** — best when the cleaning must *recompute* as data
changes (a ``TRIM`` column that stays trimmed as new rows arrive). A common
pattern uses both: functions to build cleaned columns that update, tools for
one-time structural fixes like removing duplicates.

The caveat
------------

Menu-driven tools apply *immediately and often irreversibly* — Remove Duplicates
deletes rows then and there, Find and Replace changes everything matching at once,
and an over-broad replace (turning ``"St."`` into ``"Street"`` also mangles
``"State St. Louis"``) corrupts data in bulk with one click. Because these tools
act fast and wide, the copy-first discipline matters most here: apply them to a
working copy, check the result (especially the row count after Remove
Duplicates), and keep the raw data untouched so an over-eager tool is never
fatal. The next lesson turns to the formula-based approach that recomputes as
data changes.
"""

CONTENT["Using Spreadsheet Functions for Data Cleaning"] = r"""
Cleaning by formula
---------------------

The menu tools clean data *once*; **functions** clean it *live* — a formula that
produces a cleaned value and recomputes whenever the source changes. Building
cleaned columns with functions is the reproducible complement to one-time tools:
the cleaning logic is visible in the formula and reapplies automatically as new
data arrives. This lesson covers the functions analysts reach for most.

The cleaning functions, by defect
-----------------------------------

- **Whitespace and stray characters** — ``TRIM`` removes leading, trailing, and
  extra internal spaces (so ``" New York "`` becomes ``"New York"``); ``CLEAN``
  strips non-printable characters that sneak in from imports.
- **Capitalisation** — ``UPPER``, ``LOWER``, and ``PROPER`` normalise case, so
  ``"new york"``, ``"NEW YORK"``, and ``"New York"`` can be made uniform.
- **Substitution** — ``SUBSTITUTE(text, old, new)`` replaces specific substrings
  within a cell (removing ``"$"`` and ``","`` from ``"$1,000"`` before converting
  it to a number); ``REPLACE`` swaps by position.
- **Type conversion** — ``VALUE`` turns a text-number into a real number,
  ``DATEVALUE`` turns a text-date into a real date — fixing the import type traps
  from the prep section.
- **Extraction** — ``LEFT``, ``RIGHT``, and ``MID`` pull characters from a
  string; ``FIND`` / ``SEARCH`` locate a substring's position; ``LEN`` gives
  length. Together they extract parts of a crammed field (the string lesson in
  the analysis section develops these).
- **Conditional cleaning** — ``IF`` applies a fix only when a condition holds;
  ``IFERROR`` handles the errors a cleaning formula might throw.

.. code-block:: text

   =TRIM(A2)                          strip stray spaces
   =PROPER(TRIM(A2))                  trim, then title-case
   =VALUE(SUBSTITUTE(B2,"$",""))      strip "$", convert to number
   =IF(C2<0, 0, C2)                   replace negative values with 0

Nesting for a full clean
--------------------------

Functions compose, so one formula can apply several fixes in sequence:
``=PROPER(TRIM(A2))`` trims *then* title-cases; ``=VALUE(SUBSTITUTE(SUBSTITUTE(
B2,"$",""),",",""))`` strips both the currency symbol and the thousands
separator before converting to a number. This nesting builds a cleaning
*pipeline* in a single cell — powerful, and readable if built up step by step
rather than written as one dense expression from the start.

The reproducibility advantage
-------------------------------

The reason to clean by formula rather than by hand is **reproducibility**. A
cleaned column built from functions documents its own logic (the formula *is* the
record of what cleaning was applied) and reapplies automatically to new data —
paste next month's raw values into the source column and the cleaned column
updates. This directly answers the reproducibility weakness of manual cleaning
from earlier: formula-based cleaning is a rerunnable transformation, the first
step toward the fully scripted pipelines of SQL and Python.

The caveat
------------

Function-based cleaning has its own trap: the cleaned values live in *formula*
cells that depend on the source, so if you delete the raw column or paste the
cleaned column onto itself, the formulas break or vanish. The standard practice
is to build the cleaned columns with formulas, verify them, then — if you need
static cleaned data — *paste them as values* into a new location before removing
the source. And deeply nested formulas, while powerful, grow hard to debug: build
and test them in stages. The next lesson turns from transforming data to *seeing*
it differently, so defects reveal themselves in the first place.
"""

CONTENT["Viewing Data Differently for More Effective Data Cleaning"] = r"""
You cannot fix what you cannot see
------------------------------------

Every cleaning technique so far assumes you have *found* the defect — but the
hardest part of cleaning is often *seeing* the problem in the first place. A few
bad rows hide easily in thousands of good ones. The most effective cleaners are
skilled at **viewing data differently** — using sorting, filtering, pivoting, and
formatting as *lenses* that make hidden defects jump out. This lesson is about
detection, the step before correction.

Lenses that reveal defects
----------------------------

- **Sorting** surfaces the extremes. Sort a numeric column and outliers,
  negatives, and impossible values collect at the top and bottom where you can
  see them; sort a text column and inconsistent spellings ("New York" vs "new
  york") land *next to each other* instead of scattered, and blanks group
  together. Sorting turns "somewhere in ten thousand rows" into "at the top."
- **Filtering** isolates suspects. Filter to blanks to see all missing values at
  once; filter to a category to check whether it is recorded consistently; filter
  to a range to find values outside it. Filtering removes the noise so the subset
  you are checking stands alone.
- **Pivot tables** summarise into patterns. A pivot counting records per category
  instantly reveals inconsistency: if "New York", "NY", and "new york" appear as
  *three separate rows* with separate counts, the fragmentation is visible at a
  glance — and a pivot of counts per value is one of the fastest ways to audit a
  column's consistency.
- **Conditional formatting** flags visually. Colour rules highlight duplicates,
  mark out-of-range values, or shade blanks, so defects are *visible* while you
  scroll rather than hidden in uniform grids.

The detection mindset
-----------------------

Behind the lenses is a habit: approach a new dataset *expecting* defects and
actively hunting them, rather than assuming it is clean until an error surfaces.
Run the column-consistency pivot, sort each key numeric column to check its
extremes, filter to blanks, scan the value counts. This deliberate
multi-angle inspection — looking at the same data sorted, filtered, summarised,
and formatted — finds the quiet defects (a handful of duplicates, one stray
category, a few impossible values) that a single straight-on glance always
misses. It is the same "view it from several angles" principle behind analytical
thinking, applied to finding dirt.

Detection before correction
------------------------------

The sequence matters: **see the full scope of a defect before fixing it**. A
pivot showing that "NY" appears 400 times and "New York" 3,000 times tells you
the standardisation is worth doing *and* which form to standardise toward.
Filtering to blanks shows *how many* values are missing before you decide whether
to remove or fill them. Jumping to correction before understanding the defect's
extent leads to incomplete or wrong fixes — you standardise the two spellings you
noticed and miss the third. Look first, fully, then fix.

The caveat
------------

Viewing lenses reveal defects but can also *mislead* about them: a sort makes a
few extreme values look alarming when they may be legitimate, a filtered view can
be mistaken for the whole dataset, and a pivot's groupings depend on the data
being typed correctly (a pivot cannot group text-numbers sensibly). The lenses
are for *detection and investigation*, not automatic judgement — seeing an
anomaly is the start of asking whether it is an error or a real feature, not a
verdict. This closes the hands-on dirty-data cleaning; the next lessons step back
to the big picture of clean data and the transition to cleaning with SQL.
"""


MINDMAP.update({
    "Cleaning and Merging Multiple Datasets": [
        "Data Cleaning with Spreadsheets",
        "Using VLOOKUP to Combine Data Across Spreadsheets",
        "Spreadsheet Tools for Data Cleaning",
        "Data Mapping and the Big Picture of Clean Data",
    ],
    "Spreadsheet Tools for Data Cleaning": [
        "Data Cleaning with Spreadsheets",
        "Using Spreadsheet Functions for Data Cleaning",
        "Viewing Data Differently for More Effective Data Cleaning",
        "Common Issues in Dirty Data",
    ],
    "Using Spreadsheet Functions for Data Cleaning": [
        "Spreadsheet Tools for Data Cleaning",
        "Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)",
        "Spreadsheet Functions",
        "Data Cleaning with Spreadsheets",
    ],
    "Viewing Data Differently for More Effective Data Cleaning": [
        "Using Spreadsheet Functions for Data Cleaning",
        "Spreadsheet Tools for Data Cleaning",
        "Data Mapping and the Big Picture of Clean Data",
        "Common Issues in Dirty Data",
    ],
})


# ======================================================================
# Section 4 — Data Cleaning / dirty (close) + sql (open)  (cleaning 017-020)
# ======================================================================

GLOSS.update({
    "Data Mapping and the Big Picture of Clean Data":
        "matching fields between sources so merged data stays coherent — cleaning's big picture",
    "Introduction to SQL":
        "why SQL matters for cleaning: transformations that scale and stay reproducible",
    "Spreadsheets vs. SQL":
        "when to reach for a sheet and when for a query — strengths, limits, and the handoff",
    "Core SQL Queries for Data Cleaning and Analysis":
        "SELECT, WHERE, DISTINCT, GROUP BY — the query patterns that clean and summarise",
})

CONTENT["Data Mapping and the Big Picture of Clean Data"] = r"""
Making sources speak the same language
----------------------------------------

When data comes from several sources, cleaning each one is not enough — the
sources must be made to *fit together*. **Data mapping** is the process of
matching fields from one data source to the corresponding fields in another, so
that data from different systems can be combined coherently. It is the "big
picture" of clean data: not just tidying values within a table, but ensuring the
tables *mean the same things* so a merge produces sense rather than nonsense.

What data mapping does
------------------------

Two systems rarely describe the same reality identically. One calls a field
``customer_name``, another ``client``; one stores state as ``"NY"``, another as
``"New York"``; one records dates as ``MM/DD/YYYY``, another as ISO. Data mapping
is the explicit specification of *which field corresponds to which*, and *how
their values translate*:

- **Field mapping** — ``source.client`` ↔ ``target.customer_name``: recording
  that these differently-named columns hold the same thing.
- **Value mapping** — ``"NY"`` ↔ ``"New York"``: recording how one source's codes
  translate to another's.
- **Format mapping** — ``MM/DD/YYYY`` → ISO dates: recording the transformation
  each field needs to align.

The map is a *plan* for integration, made before combining, so the merge is
deliberate rather than improvised.

Why the big picture matters
-----------------------------

Cleaning field-by-field without the big picture produces tidy tables that still
will not combine — each internally consistent, but mutually incompatible. Data
mapping is what prevents that: it forces you to see how all the pieces relate
*before* merging, catching mismatches (a field with no counterpart, two fields
that seem to match but mean subtly different things) while they are cheap to fix.
It is the difference between cleaning as isolated janitorial work and cleaning as
preparing a *coherent whole* ready for analysis — the same big-picture-first
discipline the foundations urged, applied to data integration.

Mapping and documentation
---------------------------

A data map is also **documentation** — a record of how sources relate that
outlives the immediate merge. When next month's data arrives in the same
sources, the map tells you exactly how to combine it again; when a colleague
inherits the work, the map explains how the pieces fit. This connects data
mapping to the metadata and chain-of-custody disciplines: the map is metadata
about *relationships between* datasets, and keeping it is what makes a
multi-source pipeline reproducible.

The caveat
------------

Data mapping surfaces a hard truth: sometimes fields that *look* like they
correspond do not quite — two "revenue" columns computed under different
definitions, two "customer" fields counting different populations. Mapping them
as if identical silently corrupts the merged data. The discipline is to map on
*meaning*, not just name similarity — verifying that matched fields genuinely
measure the same thing, using the metadata and definitions from the prep
section. This completes the spreadsheet-and-concepts side of cleaning; the next
lessons scale cleaning up to SQL, where these operations run on data far too
large for a sheet.
"""

CONTENT["Introduction to SQL"] = r"""
Cleaning at database scale
----------------------------

Spreadsheet cleaning works beautifully until the data outgrows the sheet — and
organisational data usually has. **SQL** (Structured Query Language) is the tool
that scales cleaning and analysis to data of any size, running the same
operations you learned in the spreadsheet against databases holding millions of
rows. The foundations introduced SQL's shape; this lesson reintroduces it
specifically as a *cleaning* tool, opening the SQL stage of this section.

Why SQL for cleaning
----------------------

SQL brings three advantages that matter especially for cleaning:

- **Scale.** A ``TRIM`` or a duplicate-removal that a spreadsheet performs on
  ten thousand rows, SQL performs on ten million in seconds — because the
  database engine does the work, not your laptop.
- **Reproducibility.** A cleaning query is *text*. Save it, and re-cleaning next
  month's data is re-running the query — the reproducibility that manual
  spreadsheet cleaning lacked, achieved by default. The query *is* the
  documentation of what cleaning was applied.
- **Working at the source.** SQL cleans data *where it lives*, in the database,
  rather than exporting fragile copies to a spreadsheet and back — reducing the
  transfer errors that threaten integrity.

The same operations, in query form
-------------------------------------

Everything you cleaned in a spreadsheet has a SQL equivalent, and recognising the
correspondence makes SQL cleaning feel familiar rather than foreign:

- Filtering to inspect (spreadsheet filter) → ``WHERE``.
- Finding distinct values to check consistency → ``SELECT DISTINCT``.
- Counting per category to audit (pivot of counts) → ``GROUP BY`` with
  ``COUNT``.
- Trimming, changing case, substituting → SQL string functions (``TRIM``,
  ``UPPER``, ``LOWER``, ``REPLACE``).
- Converting types → ``CAST`` (a dedicated lesson ahead).
- Removing duplicates → ``DISTINCT`` or grouping (a dedicated lesson ahead).

The syntax differs, but the *operations* are the ones you already know — SQL is a
new language for familiar cleaning ideas.

What makes SQL cleaning powerful
----------------------------------

Because a cleaning query is repeatable text that runs at the source at scale, SQL
turns cleaning from a manual chore into an *engineered pipeline*. A well-written
cleaning query documents the transformation, runs identically every time, and
handles data volumes no spreadsheet could open. This is why, past a certain
scale, professional cleaning moves to SQL (and Python) — not because spreadsheets
are wrong, but because rerunnable queries are what recurring, large-scale
cleaning requires.

The caveat
------------

SQL's power raises the stakes of a mistake: a cleaning query applied to a
production database can transform millions of rows at once, and an error scales
just as fast as a fix. The disciplines from spreadsheet cleaning apply with more
force — work against a copy or in a transaction you can undo, test the query's
``SELECT`` (what it *would* affect) before running any modification, and verify
row counts before and after. The precision SQL demands is the same precision the
whole section has stressed, now operating at scale. The next lesson weighs
directly when to reach for SQL versus a spreadsheet.
"""

CONTENT["Spreadsheets vs. SQL"] = r"""
Two tools, different jobs
---------------------------

You now have two cleaning-and-analysis tools — the spreadsheet and SQL — and a
recurring practical question: which to use when? Neither is universally better;
each excels at different tasks, and knowing the trade-offs lets you reach for the
right one instead of forcing every job into your favourite. This lesson lays out
the comparison directly.

Where spreadsheets win
------------------------

- **Small to medium data** — anything that fits comfortably in a sheet (up to
  tens of thousands of rows).
- **Visibility and directness** — you *see* all the data and every change
  immediately; you can click a cell and edit it. This makes spreadsheets ideal
  for exploration, one-off fixes, and quick manual work.
- **Accessibility** — everyone has one and can open your file; no database
  access or query knowledge required, which matters for sharing with
  non-analysts.
- **Ad-hoc analysis and presentation** — quick charts, pivot tables, formatted
  summaries a stakeholder reads.

Where SQL wins
----------------

- **Large data** — millions of rows SQL handles in seconds and a spreadsheet
  cannot even open.
- **Reproducibility** — a query is repeatable text; the same clean or analysis
  reruns identically on new data, where manual spreadsheet work must be redone.
- **Working at the source** — SQL queries the database directly, avoiding fragile
  export-and-reimport cycles.
- **Complex, multi-table work** — joining several large tables and aggregating is
  SQL's home turf.
- **Automation** — queries embed in pipelines that run on a schedule.

The comparison, summarised
----------------------------

A rough rule: **spreadsheets for small, visual, one-off, shareable work; SQL for
large, reproducible, source-level, automated work.** Data size, whether the work
will *recur*, and who consumes the result usually settle the choice — the same
three questions (size, repetition, audience) the tools-overview lesson posed in
the foundations, now with hands-on experience of both sides.

They work together
--------------------

The choice is rarely exclusive; the tools *chain*. A common professional pattern:
use **SQL** to clean and aggregate large data at the source, export a manageable
result, then use a **spreadsheet** to explore it, chart it, and build the summary
a stakeholder reads. SQL does the heavy lifting on volume; the spreadsheet does
the last-mile presentation. Fluency across both — and knowing which stage each
serves — is what the coming sections build, culminating in Python, which can do
the work of both and automate the whole chain.

The caveat
------------

The spreadsheet-versus-SQL choice is a spectrum, not a wall, and the boundary
moves — modern spreadsheets connect to databases and handle larger data than they
once could, while SQL tools increasingly offer visual interfaces. The durable
guidance is not a fixed row-count threshold but the *reasoning*: match the tool
to the data's scale, the work's repetition, and the audience's needs, and expect
to use both, often on the same project. The next lessons get concrete about the
SQL cleaning operations themselves.
"""

CONTENT["Core SQL Queries for Data Cleaning and Analysis"] = r"""
The query patterns that clean
-------------------------------

SQL cleaning rests on a small set of **core query patterns** — combinations of
clauses that inspect, filter, and summarise data. These patterns do the bulk of
practical cleaning and analysis, and they extend the basic ``SELECT``/``FROM``/
``WHERE`` from the prep section with the tools that make queries genuinely
powerful. This lesson assembles the working toolkit.

Inspecting and filtering
--------------------------

The starting patterns find and isolate what needs attention:

.. code-block:: sql

   SELECT DISTINCT region          -- what distinct values exist? (consistency check)
   FROM   customers;

   SELECT *                        -- which rows have a problem?
   FROM   customers
   WHERE  spend < 0;               -- invalid negative values

   SELECT *
   FROM   customers
   WHERE  city IS NULL;            -- missing values (note: IS NULL, not = NULL)

``SELECT DISTINCT`` lists the unique values in a column — the SQL version of the
consistency-audit pivot, revealing whether "NY", "New York", and "new york" all
lurk in the data. ``WHERE`` isolates rows failing a validity rule. Note
``IS NULL`` (not ``= NULL``) is how SQL tests for missing values — a common
beginner trap, since ``= NULL`` never matches.

Summarising with GROUP BY
---------------------------

The pattern that turns rows into insight is ``GROUP BY``, which collapses rows
into groups and computes an aggregate per group:

.. code-block:: sql

   SELECT   region,
            COUNT(*)     AS n,
            AVG(spend)   AS avg_spend
   FROM     customers
   GROUP BY region
   ORDER BY n DESC;

This counts customers and averages spend *per region* — the SQL equivalent of a
pivot table, and one of the most-used analytical patterns. Grouping by a column
and counting is also a cleaning tool: ``GROUP BY email HAVING COUNT(*) > 1`` finds
duplicate emails (``HAVING`` filters *groups*, as ``WHERE`` filters rows).

The aggregate functions
-------------------------

``GROUP BY`` pairs with aggregate functions — the same operations as the
spreadsheet's: ``COUNT`` (how many), ``SUM`` (total), ``AVG`` (mean), ``MIN`` /
``MAX`` (extremes). These collapse many rows to one summary value, per group or
over the whole table. Recognising them as the familiar spreadsheet aggregates,
now in query form, is what makes SQL analysis feel like an extension of what you
already know rather than a new discipline.

Putting the patterns together
-------------------------------

Real cleaning and analysis chains these: filter to valid rows with ``WHERE``,
group with ``GROUP BY``, aggregate with ``COUNT``/``AVG``, order with
``ORDER BY``, and inspect distinct values with ``DISTINCT`` to check consistency
along the way. A handful of clauses, combined, answer a large share of real
questions — which is why SQL stays learnable despite its power, and why these
core patterns are the foundation the analysis section's advanced queries build
on.

The caveat
------------

The core patterns are precise about what they compute, which is not always what
you intend: ``COUNT(*)`` counts rows including nulls while ``COUNT(column)`` skips
nulls, ``AVG`` silently ignores nulls (changing the denominator), and a
``GROUP BY`` on an uncleaned column groups "NY" and "New York" *separately* —
producing a summary that looks authoritative but is fragmented by the very
dirtiness you were checking for. The lesson underneath: **clean before you
aggregate**, because summarising dirty data produces confident, wrong totals. The
next lessons apply these patterns to specific cleaning tasks — removing
duplicates, cleaning strings, and converting types.
"""


MINDMAP.update({
    "Data Mapping and the Big Picture of Clean Data": [
        "Cleaning and Merging Multiple Datasets",
        "Viewing Data Differently for More Effective Data Cleaning",
        "Structured Data and Data Models",
        "Introduction to SQL",
    ],
    "Introduction to SQL": [
        "The Concept and Basic Use of SQL (Query Language)",
        "Spreadsheets vs. SQL",
        "Core SQL Queries for Data Cleaning and Analysis",
        "Databases and Relational Database Concepts",
    ],
    "Spreadsheets vs. SQL": [
        "Introduction to SQL",
        "Core SQL Queries for Data Cleaning and Analysis",
        "Querying Data with SQL",
        "Data Cleaning with Spreadsheets",
    ],
    "Core SQL Queries for Data Cleaning and Analysis": [
        "Introduction to SQL",
        "Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables",
        "Using CAST to Clean and Format Data in SQL",
        "Advanced SQL Functions for Data Cleaning",
    ],
})


# ======================================================================
# Section 4 — Data Cleaning / sql (close)  (cleaning 021-024)
# ======================================================================

GLOSS.update({
    "Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables":
        "DISTINCT, GROUP BY, and string functions to dedupe and standardise text in SQL",
    "Using CAST to Clean and Format Data in SQL":
        "converting a value from one data type to another — SQL's type-fixing tool",
    "Advanced SQL Functions for Data Cleaning":
        "CASE, SUBSTR, TRIM, and pattern tools for the harder cleaning jobs",
    "COALESCE":
        "returning the first non-null value — SQL's standard handler for missing data",
})

CONTENT["Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables"] = r"""
The two most common SQL cleans
--------------------------------

Two cleaning tasks dominate SQL work: removing **duplicate** records and
standardising **string** (text) values. They are the SQL versions of the
spreadsheet's Remove Duplicates and Find-and-Replace, now at database scale, and
learning them is learning the everyday core of SQL cleaning.

Removing duplicates
---------------------

The simplest deduplication is ``SELECT DISTINCT``, which returns only unique
rows:

.. code-block:: sql

   SELECT DISTINCT customer_id, name, email
   FROM   customers;

``DISTINCT`` collapses exact duplicate rows across the selected columns. To
*find* duplicates rather than just remove them — to see which records repeat and
how often — group and count:

.. code-block:: sql

   SELECT   email, COUNT(*) AS copies
   FROM     customers
   GROUP BY email
   HAVING   COUNT(*) > 1;

This lists every email appearing more than once (``HAVING`` filters the *groups*,
keeping only those with a count above one) — the diagnostic step before deciding
how to resolve them. Which duplicate to *keep* when they disagree is a judgement,
often resolved by keeping the most recent or most complete record.

Cleaning string variables
----------------------------

Text fields accumulate the inconsistencies from earlier — stray spaces, mixed
case, unwanted characters — and SQL's string functions fix them:

.. code-block:: sql

   SELECT TRIM(name)                    AS name_trimmed,   -- remove edge spaces
          UPPER(region)                 AS region_upper,   -- uniform case
          REPLACE(phone, '-', '')       AS phone_digits    -- strip separators
   FROM   customers;

- ``TRIM`` removes leading and trailing whitespace (``LTRIM`` / ``RTRIM`` for one
  side).
- ``UPPER`` / ``LOWER`` normalise case, so ``"NY"`` and ``"ny"`` become uniform.
- ``REPLACE(text, from, to)`` substitutes one substring for another, stripping or
  swapping characters.

These are the same operations as the spreadsheet's ``TRIM``, ``UPPER``, and
``SUBSTITUTE`` — the SQL syntax differs slightly, the ideas are identical.

Cleaning into a result, not in place
--------------------------------------

Notice these queries *select* cleaned values — they produce a cleaned result set
without altering the stored table. This is the safe default: you build and verify
the cleaned output with a ``SELECT`` first, confirming it does what you intend,
before ever writing changes back. Producing a cleaned view or table from a query
is both safer and more reproducible than editing data in place.

The caveat
------------

SQL deduplication has a subtlety that catches beginners: ``SELECT DISTINCT``
removes rows that are *identical across the selected columns*, so two records for
the same customer that differ in even one field (a different timestamp, a typo'd
name) are **not** duplicates to ``DISTINCT`` and both survive. True
deduplication of near-duplicates requires deciding *which columns define
sameness* and grouping on those — not a blind ``DISTINCT *``. And string
functions are precise: ``REPLACE`` replaces *every* occurrence, so an over-broad
replacement corrupts as readily in SQL as in a spreadsheet. Verify the
``SELECT`` output before trusting it. The next lesson tackles the type-conversion
cleaning that ``CAST`` handles.
"""

CONTENT["Using CAST to Clean and Format Data in SQL"] = r"""
Fixing the type
-----------------

A recurring dirty-data defect is a value stored as the *wrong type* — a number
kept as text, a date kept as a string — which breaks the calculations and
comparisons that expect the right type. In SQL, the tool that fixes this is
**CAST**, which converts a value from one data type to another. It is the SQL
counterpart of the spreadsheet's ``VALUE`` and ``DATEVALUE``, and a core cleaning
operation.

How CAST works
----------------

``CAST`` takes a value and a target type and returns the value converted:

.. code-block:: sql

   SELECT CAST(price_text AS DECIMAL)      AS price,      -- text -> number
          CAST(order_date_text AS DATE)    AS order_date, -- text -> date
          CAST(quantity AS INTEGER)        AS quantity    -- decimal -> whole
   FROM   orders;

The syntax reads plainly: ``CAST(value AS type)``. Common target types include
``INTEGER`` and ``DECIMAL`` (numbers), ``DATE`` and ``TIMESTAMP`` (dates and
times), and ``VARCHAR`` / ``STRING`` (text). Once a text-number is cast to
``DECIMAL``, it can be summed and averaged; once a text-date is cast to ``DATE``,
it can be sorted chronologically and used in date arithmetic — the defect is
fixed.

Why type conversion matters for cleaning
------------------------------------------

Type problems are among the most common consequences of the import traps from the
prep section: data loaded from CSVs and external systems frequently arrives with
everything as text, so numeric and date columns *look* right but refuse to
compute. ``CAST`` is how you correct this at scale — one query converts a whole
column of text-numbers into real numbers, ready for the aggregation and
arithmetic the analysis will need. Without it, ``SUM`` and ``AVG`` on a
text-typed column either error or silently misbehave.

CAST in a cleaning pipeline
-----------------------------

``CAST`` composes with the string functions from the previous lesson, because raw
values often need *cleaning before conversion*. A price stored as ``"$1,000"``
cannot be cast to a number directly — the ``$`` and ``,`` must be removed first:

.. code-block:: sql

   SELECT CAST(REPLACE(REPLACE(price_text, '$', ''), ',', '') AS DECIMAL) AS price
   FROM   orders;

Strip the currency symbol and thousands separator with ``REPLACE``, *then*
``CAST`` the clean numeric string to ``DECIMAL``. This clean-then-convert pattern
is exactly the nested-function pipeline from the spreadsheet lesson, in SQL.

The caveat
------------

``CAST`` fails — or behaves unexpectedly — when a value cannot be converted to
the target type: casting ``"N/A"`` or ``"twelve"`` to a number raises an error or,
in some databases, produces a null, and a single unconvertible value can break a
query over millions of rows. This is a feature, not a bug: it forces you to
confront the values that do not fit the type, which are often *themselves*
dirty data needing attention. The safe practice is to inspect a column's distinct
values (``SELECT DISTINCT``) before casting, handle the non-conforming ones
first, and be aware that different databases treat cast failures differently
(error versus null). Type conversion reveals dirt as much as it fixes it. The
next lesson covers the more advanced functions for harder cleaning cases.
"""

CONTENT["Advanced SQL Functions for Data Cleaning"] = r"""
Beyond the basics
-------------------

The core string and conversion functions handle most cleaning, but harder cases —
conditional fixes, extracting parts of a field, pattern-based standardisation —
call for more capable tools. This lesson covers the **advanced SQL functions**
that handle the cleaning jobs the basics cannot, extending the toolkit toward the
messy realities of production data.

Conditional cleaning with CASE
--------------------------------

``CASE`` is SQL's if-then-else, applying different fixes depending on a value's
condition — the SQL counterpart of the spreadsheet's nested ``IF``:

.. code-block:: sql

   SELECT name,
          CASE
            WHEN spend < 0        THEN 0          -- invalid negatives -> 0
            WHEN spend IS NULL    THEN 0          -- missing -> 0
            ELSE spend
          END                     AS spend_clean
   FROM   customers;

``CASE`` evaluates each ``WHEN`` in order and returns the first matching
``THEN``, falling through to ``ELSE`` if none match. It handles the "different
fix for different conditions" cleaning that a single function cannot —
standardising categories, bucketing values, or applying rules that vary by row.

Extracting parts of a field
-----------------------------

When a column crams several values together, substring functions pull them
apart — the SQL version of the spreadsheet's ``LEFT``/``RIGHT``/``MID``:

.. code-block:: sql

   SELECT SUBSTR(product_code, 1, 3)   AS category_code,  -- first 3 chars
          LENGTH(name)                 AS name_length,
          POSITION('@' IN email)       AS at_position     -- find a character
   FROM   products;

``SUBSTR`` (or ``SUBSTRING``) extracts characters by position and length;
``LENGTH`` gives a string's length; ``POSITION`` (or ``INSTR``) locates a
substring. Together they split composite fields into their components — restoring
the one-value-per-column structure.

Pattern-based cleaning
------------------------

For standardising by *pattern* rather than exact match, ``LIKE`` with wildcards
finds values fitting a shape (``WHERE phone LIKE '___-___-____'`` for a phone
format), and combined with ``CASE`` it standardises variants that a simple
``REPLACE`` cannot. More advanced engines offer regular-expression functions for
complex pattern matching, though these vary by database.

Composing the advanced tools
------------------------------

The power comes from combining them: a ``CASE`` that applies different
``SUBSTR`` extractions depending on a field's pattern, or a ``TRIM`` wrapped
around a ``REPLACE`` inside a ``CAST``. Real cleaning queries nest these functions
into a pipeline that inspects, extracts, conditions, and converts in one pass —
the same composition principle as spreadsheet formula nesting, with SQL's richer
function set.

The caveat
------------

Advanced functions bring two cautions. First, **portability**: the basics
(``TRIM``, ``UPPER``, ``CASE``, ``CAST``) work almost everywhere, but the exact
names and behaviours of substring, position, and especially regular-expression
functions **differ across database systems** — ``SUBSTR`` versus ``SUBSTRING``,
``INSTR`` versus ``POSITION`` — so a query written for one database may need
adjustment for another. Second, **readability**: deeply nested advanced functions
grow hard to read and debug, so build them in stages and comment the intent.
Power without clarity is its own maintenance problem — the clarity-over-cleverness
principle, applied to SQL. The final SQL cleaning lesson covers the dedicated tool
for missing values: ``COALESCE``.
"""

CONTENT["COALESCE"] = r"""
Handling missing values directly
----------------------------------

Missing values — nulls — are one of the most common dirty-data defects, and SQL
has a dedicated function for handling them: **COALESCE**, which returns the first
non-null value from a list of arguments. It is SQL's standard, clean way to
substitute a fallback for missing data, and it earns its own lesson because
missing-value handling is so pervasive in real cleaning.

How COALESCE works
--------------------

``COALESCE`` takes any number of arguments and returns the first that is not
null:

.. code-block:: sql

   SELECT name,
          COALESCE(phone, 'no phone on file')  AS phone,
          COALESCE(nickname, first_name, 'Unknown') AS display_name
   FROM   customers;

For each row, ``COALESCE(phone, 'no phone on file')`` returns the phone if it
exists, otherwise the fallback text. The multi-argument form
``COALESCE(nickname, first_name, 'Unknown')`` tries each in turn: the nickname if
present, else the first name, else a final default — returning the first
available value down the list. This replaces scattered nulls with meaningful
values in a single, readable expression.

Why COALESCE matters
----------------------

Nulls cause trouble throughout analysis: they break calculations
(``price + tax`` is null if either is null), skew aggregates (``AVG`` ignores
them, changing the denominator), and display as blanks that confuse readers.
``COALESCE`` addresses this at the point of query, substituting a sensible value
so downstream calculations and displays behave. A common pattern fills missing
numerics with zero for summation — ``COALESCE(amount, 0)`` — so a ``SUM`` counts
the missing rows as zero rather than being thrown off by nulls.

COALESCE versus other approaches
----------------------------------

``COALESCE`` is the *standard* (cross-database) way to handle nulls; some
databases also offer ``IFNULL`` or ``NVL`` for the two-argument case, but
``COALESCE`` works everywhere and handles any number of fallbacks, so it is the
portable choice. It relates to the ``CASE`` from the previous lesson —
``COALESCE(x, y)`` is shorthand for ``CASE WHEN x IS NOT NULL THEN x ELSE y
END`` — but is far more concise for the common "use this, or a fallback if
missing" pattern.

The caveat
------------

``COALESCE`` is powerful, and that is exactly its danger: **substituting a value
for missing data is a decision with analytical consequences**, not a neutral
tidy-up. Filling missing prices with zero changes the average; filling missing
categories with "Unknown" creates a category that did not exist in reality;
replacing nulls hides the fact that data *was* missing, which may itself be a
finding (why is this field so often empty?). The honest practice is to choose the
fallback *deliberately* — is zero, the mean, or a flag the right substitute for
*this* analysis? — and to consider whether the missingness should be preserved
and reported rather than filled. ``COALESCE`` makes filling easy; whether to fill
at all is the judgement. This completes the SQL cleaning stage; the final stage
of the section turns to verifying, documenting, and reporting the cleaning work.
"""


MINDMAP.update({
    "Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables": [
        "Core SQL Queries for Data Cleaning and Analysis",
        "Using CAST to Clean and Format Data in SQL",
        "Using Spreadsheet Functions for Data Cleaning",
        "Advanced SQL Functions for Data Cleaning",
    ],
    "Using CAST to Clean and Format Data in SQL": [
        "Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables",
        "Understanding Data Types and Data Formats",
        "Data Types in Spreadsheets",
        "Advanced SQL Functions for Data Cleaning",
    ],
    "Advanced SQL Functions for Data Cleaning": [
        "Using CAST to Clean and Format Data in SQL",
        "COALESCE",
        "Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)",
        "Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables",
    ],
    "COALESCE": [
        "Advanced SQL Functions for Data Cleaning",
        "Common Issues in Dirty Data",
        "Verifying Data-Cleaning Efforts",
        "Using CAST to Clean and Format Data in SQL",
    ],
})


# ======================================================================
# Section 4 — Data Cleaning / Stage: verify  (cleaning 025-028)
# ======================================================================

GLOSS.update({
    "Verifying and Reporting Data Integrity":
        "confirming data is sound after cleaning — and telling stakeholders it is",
    "Verifying Data-Cleaning Efforts":
        "checking that each cleaning step did what it should, and nothing it shouldn't",
    "Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors":
        "concrete checks — counts, distinct values, ranges — that catch recurring defects",
    "Documenting Data-Cleaning Changes":
        "recording what was changed and why, so cleaning is reproducible and reviewable",
})

CONTENT["Verifying and Reporting Data Integrity"] = r"""
The step that makes cleaning trustworthy
------------------------------------------

Cleaning data is only half the job; **confirming it is now sound** is the other
half, and the one beginners most often skip. This final stage of the section is
about **verification** — checking that cleaning actually worked — and
**reporting** — communicating the data's integrity to the people who will rely on
it. Without verification, cleaning is an act of faith; with it, cleaning becomes a
trustworthy, defensible process.

Why verification is non-negotiable
------------------------------------

Every cleaning operation can go wrong in ways that produce plausible results,
exactly like the defects it was meant to fix. A duplicate-removal might delete
legitimate records; a find-and-replace might over-match; a type conversion might
silently null out unconvertible values; a merge might drop or multiply rows.
Because these failures produce data that *looks* fine, the only way to know
cleaning succeeded is to *check* — verification is what catches the errors
cleaning itself introduced. The integrity-in-motion principle reaches its
conclusion here: after every transformation, confirm the data is still sound.

What verifying integrity involves
-----------------------------------

Verifying data integrity after cleaning means confirming the data is now
complete, accurate, consistent, and valid — that the defects are gone and no new
ones were introduced:

- **Confirm the fix worked** — the duplicates are actually gone, the categories
  actually consistent, the types actually converted.
- **Confirm nothing broke** — the row count is what it should be (not
  mysteriously smaller from over-deletion or larger from a bad merge), the totals
  still reconcile, no legitimate data was lost.
- **Confirm against the source** — spot-check cleaned values against the original
  raw data to ensure cleaning transformed them correctly, not wrongly.
- **Confirm fitness for purpose** — the data now meets the standard the analysis
  requires (the alignment and sufficiency checks from earlier, revisited after
  cleaning).

Reporting integrity
---------------------

Verification produces something worth communicating: **confidence in the data**.
Reporting data integrity means telling stakeholders what state the data is in —
what was cleaned, what issues were found and fixed, what limitations remain, and
therefore how much the conclusions can be trusted. This is not bureaucratic
overhead; it is what lets a stakeholder weigh the analysis appropriately, and it
is the honest-communication obligation from Section 2 applied to data quality. A
brief, clear statement of the data's integrity — "duplicates removed, categories
standardised, 2% of records had missing values that were excluded, results
reliable for the top four regions" — is what turns cleaned data into *trusted*
cleaned data.

The caveat
------------

Verification can never be exhaustive — you cannot check every value in a large
dataset by hand, and some errors will inevitably escape any practical check. The
professional standard is *proportionate* verification: check thoroughly the
things most likely to be wrong and most consequential if they are (the key
fields, the counts, the transformations you are least sure of), and be honest in
reporting about what was and was not verified. Verification reduces risk to an
acceptable level and documents that it did; it does not achieve certainty, and
claiming it does is its own dishonesty. The next lessons make verification
concrete.
"""

CONTENT["Verifying Data-Cleaning Efforts"] = r"""
Checking your own work
------------------------

Verification begins with the cleaning you just did: confirming each step achieved
its goal *and* had no unintended side effects. **Verifying data-cleaning efforts**
is the disciplined review of your own transformations — the habit that separates
professional cleaning from hopeful cleaning, and the practical core of this
stage.

The two questions every cleaning step must answer
---------------------------------------------------

For each cleaning operation, verification asks two things:

1. **Did it do what I intended?** The duplicates I removed are gone; the values I
   standardised are now consistent; the column I converted is now the right type.
   Confirm the *intended* effect actually happened.
2. **Did it do anything I did not intend?** No legitimate records were deleted
   alongside the duplicates; the standardisation did not over-match and merge
   distinct values; the conversion did not silently null out data it could not
   convert. Confirm the *absence* of side effects.

The second question is the one beginners forget and professionals never do —
because the side effects are precisely what produce plausible-wrong data.

The verification techniques
-----------------------------

Concrete checks answer these questions:

- **Row counts before and after.** The single most valuable check. Know how many
  rows you started with and expect to end with; a mismatch (deduplication removed
  more than the known duplicate count, a merge changed the count unexpectedly)
  signals a problem immediately.
- **Spot-checking.** Examine specific records in detail — pick some you know, and
  confirm they cleaned correctly. Spot-checks catch errors that aggregate numbers
  hide.
- **Re-running the detection.** Apply the defect-finding checks again after
  cleaning: if you removed duplicates, re-run the duplicate query and confirm it
  now returns none; if you standardised categories, re-run the distinct-values
  check and confirm they are now uniform.
- **Comparing to the source.** Check cleaned values against the untouched raw data
  to confirm the transformation was correct, not just that *a* transformation
  happened.
- **Sanity-checking totals.** Do the aggregate numbers still make sense against
  the order-of-magnitude expectations from the mathematical-thinking lesson? A
  total that shifted implausibly after cleaning signals lost or duplicated data.

The re-run principle
----------------------

The most reliable verification is **re-running the detection that found the
defect**: the check that revealed a problem should reveal *none* after the fix.
This closes the loop — detection, correction, re-detection — and it is why the
viewing-data-differently lenses and the SQL detection queries are verification
tools as much as discovery tools. If the duplicate query still finds duplicates,
the cleaning did not work, and you know it immediately rather than discovering it
in the analysis.

The caveat
------------

Verifying your own cleaning has a blind spot: you tend to check for the problems
you *thought about*, and miss the ones you did not — the same cognitive limit as
confirmation bias, applied to your own work. Re-running your own detection
confirms the defects you looked for are fixed, but cannot reveal defects you
never checked for. This is why fresh eyes help (a colleague spotting what you
assumed), and why the systematic defect *checklist* from earlier matters — it
prompts checks you might not think to make. Verify against a list, not just
against memory. The next lesson gives the concrete spreadsheet and SQL techniques.
"""

CONTENT["Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors"] = r"""
Concrete checks in both tools
-------------------------------

Verification is only as good as the specific checks you run, and both the
spreadsheet and SQL offer concrete techniques for confirming data integrity —
especially for catching **repeated** errors, the systematic defects that recur
across many rows and matter most. This lesson assembles the practical
verification toolkit in both tools.

Spreadsheet verification techniques
-------------------------------------

- **COUNTIF / COUNTA for counts.** ``COUNTA`` counts non-empty cells (confirming
  how many records remain); ``COUNTIF`` counts cells meeting a condition
  (``=COUNTIF(spend, "<0")`` should return 0 after cleaning negatives). A
  conditional count that should be zero is a powerful integrity check.
- **Conditional formatting to re-flag.** Re-apply the formatting that highlighted
  defects; if nothing highlights now, the defect is gone — a visual re-detection.
- **Pivot tables to re-audit consistency.** Re-build the counts-per-value pivot;
  if "NY" and "New York" no longer appear as separate rows, standardisation
  worked.
- **Sorting to re-inspect extremes.** Re-sort the cleaned column; the impossible
  values that sat at the top should be gone.
- **Spot-check formulas.** Compare cleaned values to raw with a formula
  (``=A2=raw!A2``) to flag where they differ, confirming transformations were
  correct.

SQL verification techniques
-----------------------------

- **COUNT to confirm row totals.** ``SELECT COUNT(*)`` before and after; the
  difference should exactly match the intended change.
- **COUNT with WHERE to confirm fixes.** ``SELECT COUNT(*) FROM t WHERE spend <
  0`` should return 0 after cleaning negatives — a conditional count that must be
  zero.
- **SELECT DISTINCT to confirm consistency.** Re-run the distinct-values query; a
  clean column shows only the standardised forms, no variants.
- **GROUP BY ... HAVING to confirm deduplication.** Re-run
  ``GROUP BY key HAVING COUNT(*) > 1``; it should return no rows if duplicates are
  gone.
- **Aggregate reconciliation.** Compare a ``SUM`` or ``COUNT`` against a known
  expected total to confirm nothing was lost or duplicated.

Catching *repeated* errors specifically
-----------------------------------------

The emphasis on *repeated* errors is deliberate: a one-off typo affects one row,
but a systematic error — a whole column imported as text, a category
misspelled everywhere, a unit wrong throughout — affects thousands and distorts
every aggregate. These are both the most damaging defects and the most *checkable*,
because a single query or formula tests the whole column at once:
``SELECT COUNT(*) WHERE <the systematic condition>`` catches a repeated error
across a million rows instantly. Verification's greatest leverage is on exactly
these repeated, systematic defects.

The caveat
------------

These techniques verify what they *test for*, and a check returning the expected
result confirms only that specific property — a row count matching expectation
does not guarantee the *right* rows survived, and a distinct-values check
confirming consistency says nothing about whether the values are *correct*.
Layer multiple checks (count *and* spot-check *and* reconcile) rather than
trusting any single one, because each catches a different class of error and none
catches all. Verification is a net woven from several checks, not a single test.
The final lesson of the stage covers documenting all of this.
"""

CONTENT["Documenting Data-Cleaning Changes"] = r"""
The record that makes cleaning real
-------------------------------------

Cleaning that is not documented is cleaning that cannot be trusted, reproduced,
or defended. **Documenting data-cleaning changes** — keeping a clear record of
what was changed, why, and how — is what turns a series of ad-hoc fixes into a
transparent, reproducible process. It is the discipline that has run beneath this
entire section, stated now as its own practice, and it is what separates
professional data work from irreproducible manual editing.

What to document
------------------

A cleaning record captures, for each change or the cleaning as a whole:

- **What was changed** — which columns, which operations: "removed duplicate
  customer records", "standardised state abbreviations to full names", "converted
  ``price`` from text to decimal".
- **Why** — the defect that motivated it: "312 exact-duplicate rows inflated
  counts", "state recorded inconsistently across three formats".
- **How** — the method or query used, ideally the actual code: the SQL query, the
  formula, the tool applied. The *how* is what makes it reproducible.
- **What was affected** — how many records changed, and any that could not be
  cleaned and were excluded or flagged.
- **What remains** — known limitations, unresolved issues, decisions deferred
  (the missing values filled with a default, the outliers left in pending
  review).

Why documentation matters
---------------------------

Documentation serves several masters at once. **Reproducibility** — next month's
data can be cleaned the same way, because the steps are recorded (and if they are
recorded *as code*, re-running is trivial). **Reviewability** — a colleague, or an
auditor, can see exactly what was done and judge whether it was right.
**Trust** — a stakeholder who can see the cleaning log trusts the data more than
one asked to take it on faith. **Your future self** — six months on, you will not
remember why you excluded those records; the documentation will tell you. And
**error-tracing** — when a problem surfaces downstream, the cleaning log is where
you look to find whether cleaning caused it.

Documentation as reproducibility
----------------------------------

The deepest reason to document connects to the whole section's arc: the
difference between manual spreadsheet cleaning and scripted SQL/Python cleaning is
largely that the latter *documents itself* — the query or script **is** the record
of what was done. When cleaning is manual, documentation must be added
deliberately (a notes tab, a change log); when it is code, the code is the
documentation, which is a further reason professional cleaning gravitates toward
rerunnable scripts. Either way, the principle is the same: the cleaning must leave
a trail.

The caveat
------------

Documentation has a cost and a failure mode: over-documenting trivial changes
buries the important ones, and documentation that drifts out of sync with what was
actually done is worse than none — a change log that says one thing while the data
reflects another misleads confidently. The goal is documentation that is
*accurate, sufficient, and maintained* — enough to reproduce and review the
cleaning, kept truthful to what actually happened, without ceremony that no one
will sustain. This completes the verification stage; the final lessons of the
section turn to reporting cleaning results and using cleaning feedback to improve
data quality at the source.
"""


MINDMAP.update({
    "Verifying and Reporting Data Integrity": [
        "Data Integrity and Its Risks in Data Analysis",
        "Verifying Data-Cleaning Efforts",
        "Reporting Data-Cleaning Results",
        "Documenting Data-Cleaning Changes",
    ],
    "Verifying Data-Cleaning Efforts": [
        "Verifying and Reporting Data Integrity",
        "Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors",
        "Common Issues in Dirty Data",
        "Documenting Data-Cleaning Changes",
    ],
    "Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors": [
        "Verifying Data-Cleaning Efforts",
        "Core SQL Queries for Data Cleaning and Analysis",
        "COALESCE",
        "Verifying and Reporting Data Integrity",
    ],
    "Documenting Data-Cleaning Changes": [
        "Verifying and Reporting Data Integrity",
        "Reporting Data-Cleaning Results",
        "The Importance of Clean Data",
        "Verifying Data-Cleaning Efforts",
    ],
})


# ======================================================================
# Section 4 — Data Cleaning / verify (close)  (cleaning 029-032)  -- SECTION 4 COMPLETE
# note: 031-032 are job-flavored but author-filed under Section 4; kept here per inventory
# ======================================================================

GLOSS.update({
    "Reporting Data-Cleaning Results":
        "communicating what cleaning found and fixed, so stakeholders can trust the data",
    "Using Feedback from Data Cleaning to Improve Data Quality":
        "closing the loop: feeding cleaning lessons back to fix problems at the source",
    "Refining a Resume for Data Analytics Roles":
        "shaping a resume to show analytical skill — cleaning your own professional data",
    "Exploring Data Analyst Job Opportunities":
        "where analyst roles are, and how to read them against your skills and goals",
})

CONTENT["Reporting Data-Cleaning Results"] = r"""
Cleaning is not done until it is communicated
-----------------------------------------------

Documenting cleaning creates the record; **reporting** it delivers that record to
the people who need it. Reporting data-cleaning results means communicating what
the cleaning process found and fixed — clearly, honestly, and to the right
audience — so that everyone relying on the data understands its state. It is the
data-cleaning application of the Section 2 principle that value is created only
when communicated, and it is what converts private cleaning work into shared
trust.

What a cleaning report contains
---------------------------------

A good cleaning report answers the questions a data consumer will have:

- **What was the data's starting state** — the defects found: how many
  duplicates, what inconsistencies, how much missing data.
- **What was done** — the cleaning actions taken, in plain language (the *what*
  and *why* from the documentation, summarised for the audience).
- **What changed** — how many records were affected, removed, or corrected, and
  the net effect on the dataset (e.g. "12,450 rows in, 12,138 after removing
  312 duplicates").
- **What remains** — the limitations: unresolved issues, records that could not
  be cleaned, decisions made about missing values, and how these bound the
  analysis.
- **The bottom line** — whether the data is now fit for its purpose, and with
  what caveats.

Matching the report to the audience
-------------------------------------

The adapting-communication principle applies directly. A fellow analyst wants the
detail — which queries, which counts, which edge cases. A business stakeholder
wants the bottom line — "the data is reliable for the top four regions; two
smaller regions had too much missing data to include." An auditor wants the full
documented trail. Same cleaning, different reports: lead with what each audience
needs to decide how much to trust the analysis built on this data.

Why honest reporting matters most here
----------------------------------------

Reporting cleaning results is where the temptation to overstate is strong and the
cost of doing so is high. It is tempting to present data as "cleaned" full stop,
implying a completeness that no real cleaning achieves. The honest report states
the *remaining* limitations as prominently as the fixes — because a stakeholder
who knows the data excludes two regions will interpret the analysis correctly,
while one who believes it is complete and perfect will over-trust it. This is the
fairness-and-honesty thread at the data-quality level: report what you fixed
*and* what you could not, so decisions rest on an accurate picture of the data's
reliability.

The caveat
------------

A cleaning report can err toward too much as easily as too little: burying the
one limitation that matters under exhaustive detail about trivial fixes serves no
one, and a report so long nobody reads it fails as surely as no report at all.
The goal is a report *proportionate* to the audience and the stakes — enough for
them to correctly gauge the data's trustworthiness, foregrounding what actually
affects the conclusions, without ceremony. The next lesson turns cleaning results
into lasting improvement.
"""

CONTENT["Using Feedback from Data Cleaning to Improve Data Quality"] = r"""
Closing the loop
------------------

Cleaning fixes the data you have; the highest-value move is using what cleaning
*taught* you to stop the same problems from recurring. **Using feedback from data
cleaning to improve data quality** means feeding the lessons of cleaning back to
their source — the systems and processes that generated the dirty data — so future
data arrives cleaner. It is the difference between endlessly mopping the floor and
fixing the leak, and it is where a data analyst's work improves an organisation
beyond the immediate analysis.

From symptom to source
------------------------

Every recurring defect that cleaning fixes is a clue about an upstream problem.
The root-cause discipline from the foundations applies directly: ask *why* the
data was dirty, and the answer usually points to a fixable source:

- A field that is *always* inconsistently formatted → the data-entry form allows
  free text where it should offer a fixed list.
- A column that *always* imports as the wrong type → the source system or export
  is mis-configured.
- Records that are *routinely* duplicated → a process creates them twice, or lacks
  a uniqueness check.
- A field *frequently* missing → it is optional at collection when it should be
  required.

Cleaning treats the symptom (this batch's dirty data); feedback treats the cause
(why every batch is dirty this way).

How feedback improves quality at the source
----------------------------------------------

The feedback loop turns cleaning insight into upstream fixes:

- **Report patterns, not just instances** — tell the data owners not only "this
  data had these problems" but "this *kind* of problem recurs, and here is where
  it originates."
- **Suggest source improvements** — data validation at entry, required fields,
  fixed-list dropdowns, corrected export configurations, uniqueness constraints.
- **Feed data governance** — recurring quality problems are exactly what the
  governance standards and data stewards from the prep section exist to address;
  cleaning feedback is a primary input to improving those standards.

Fixing a problem at the source means it never has to be cleaned again — the
highest-leverage possible outcome, because it saves the cleaning cost on every
future batch, not just this one.

Why this is the analyst's contribution to quality
---------------------------------------------------

An analyst who only cleans is valuable; an analyst who *also* feeds back to
improve the data's source is transformative, because they reduce the total
cleaning burden of the whole organisation over time. This connects cleaning to
the data life cycle and governance: the analyst, positioned at the point where
dirty data's costs become visible, is uniquely placed to identify where quality
breaks down and advocate for fixing it. Cleaning feedback is how the person who
suffers the dirty data helps prevent it.

The caveat
------------

Source improvement is often outside the analyst's direct control — the data-entry
form, the source system, the collection process belong to other teams, and
changing them requires influence, not just insight. The realistic contribution is
to *surface* the patterns clearly and advocate for fixes, recognising that
implementation depends on others and organisational priorities. And not every
source problem is worth fixing — the cost of the upstream change must be weighed
against the recurring cost of cleaning, the same proportionality judgement as
everywhere else. Feedback is high-leverage where the problem recurs and the fix is
feasible; it is a recommendation, not a guarantee. The next lessons turn from the
data's quality to the analyst's own career.
"""

CONTENT["Refining a Resume for Data Analytics Roles"] = r"""
Cleaning your own professional data
-------------------------------------

Having learned to prepare data for analysis, it is fitting to close this section
by preparing your *own* data — the resume that represents you to employers. A
resume for a data analytics role is, in a sense, a dataset about you that must be
clean, accurate, relevant, and well-structured — the same qualities you have
learned to demand of any data. **Refining a resume for data analytics roles** is
about shaping it to demonstrate analytical capability clearly to the people
hiring for it.

What a data analytics resume must show
----------------------------------------

Employers scanning an analyst resume look for evidence of specific capabilities,
so the resume should make them easy to find:

- **Technical skills** — the tools of this course: spreadsheets, SQL, a language
  like Python or R, visualization tools, and the ability to work with databases.
  List the ones you genuinely have; be honest about level.
- **The analytical process** — evidence you can do the whole arc: ask good
  questions, prepare and clean data, analyse it, and communicate findings. A
  project described from question to insight shows this better than a skills list
  alone.
- **Impact, quantified** — results stated with numbers: "cleaned and analysed a
  dataset of 50,000 records to identify a trend that informed a staffing
  decision." Quantified impact is to a resume what a metric is to an analysis —
  concrete and credible.
- **Relevant projects** — concrete work (even course or personal projects) that
  demonstrates the skills applied to real data.

Applying the course's own principles
--------------------------------------

A resume rewards exactly the disciplines this course teaches. **Clarity over
cleverness** — a clean, scannable resume beats a cluttered, over-designed one, the
same principle as a clean chart. **Accuracy** — never overstate skills or
results; misrepresentation is the professional-integrity failure the fairness
thread warned against, and it surfaces in interviews. **Relevance** — tailor the
content to what the specific role needs, the same relevance judgement as choosing
data for a question (the next lessons, and the job-search section, develop
tailoring). **Structure** — organise for the reader's scan, as you would
structure data for analysis.

Refining as iteration
-----------------------

"Refining" is the operative word: a resume is improved through iteration, not
written once. Draft it, check it against the role, get feedback, and revise —
the same iterative loop as refining an analysis. Each application may warrant
tailoring the resume to that role's emphasis, foregrounding the skills and
projects most relevant to it.

The caveat
------------

A resume is a *representation*, and the honesty obligation is absolute: it should
present your genuine skills and experience in their best true light, never
fabricate or materially overstate them. An analyst's credibility rests on
trustworthiness with data, and a dishonest resume contradicts the very quality the
role requires — quite apart from unravelling under the interview questions the
job-search section prepares you for. Present yourself accurately and well; do not
present yourself falsely. The final lesson of this section looks at where these
resumes are sent.
"""

CONTENT["Exploring Data Analyst Job Opportunities"] = r"""
Reading the landscape
-----------------------

This section closes where a data analytics journey often points: toward the roles
themselves. **Exploring data analyst job opportunities** is about understanding
where analyst roles exist, the forms they take, and how to read them against your
own skills and goals — the practical bridge from learning the craft to practising
it professionally. It complements the foundations' look at *choosing* a role with
a look at *finding* the opportunities.

Where analyst roles are found
-------------------------------

Data analyst opportunities appear across a wide landscape:

- **Industries** — as the foundations showed, virtually every sector employs
  analysts: technology, finance, healthcare, retail, government, non-profits. The
  data differs by sector, but the analytical role is broadly transferable.
- **Role variations** — the title spans a family: general data analyst, business
  analyst, and more specialised roles leaning toward reporting, visualization, or
  particular domains. Related titles (business intelligence analyst, reporting
  analyst, junior data scientist) often want overlapping skills.
- **Company types** — large organisations offer specialisation and mentorship;
  small ones offer breadth and ownership (the size trade-off from the
  foundations), and each suits different starting points.

Reading an opportunity against yourself
------------------------------------------

The skill is matching opportunities to your situation, using the factors the
foundations laid out — industry interest, company size, specialisation,
growth, and the tools the role uses. A job posting is a description of what the
role needs; reading it well means asking which of its requirements you meet, which
you could grow into, and whether its industry, size, and focus fit what you want.
Not every posting deserves an application; the ones aligned with your skills and
goals do.

Preparing to pursue opportunities
-----------------------------------

Exploring opportunities connects to everything this section and course have built.
The **skills** you have learned — the analytical process, spreadsheets, SQL,
cleaning, and the analysis and visualization still ahead — are what the roles
require. The **resume** of the previous lesson is how you present those skills. And
the **job-search process** — tailoring applications, building a presence,
networking, and interviewing — is a craft of its own that the course's final
section develops in depth. Exploring opportunities is the first step of that
process: understanding the landscape before entering it.

The caveat
------------

Job requirements can be intimidating and are often aspirational — postings
frequently list more than any single hire is expected to have, and meeting *every*
requirement is rarely necessary to be a strong candidate. The realistic reading is
to weigh your *overall* fit and growth potential against a role, not to
self-reject over a missing checkbox, while also being honest about genuine gaps
worth closing. Opportunity exploration is about finding roles where you can
contribute and grow, which is a judgement about fit and trajectory, not a
pass/fail test against a wish list.

This completes the Data Cleaning and Preparation section. You have moved from why
clean data matters, through integrity and the statistics of sufficiency, the
defects of dirty data and how to fix them in spreadsheets and SQL, to verifying,
documenting, and reporting the work — and finally to representing your own skills
professionally. With data now understood, prepared, and clean, the next section
turns to the heart of the craft: analysing it.
"""


MINDMAP.update({
    "Reporting Data-Cleaning Results": [
        "Documenting Data-Cleaning Changes",
        "Verifying and Reporting Data Integrity",
        "Data Creates Value Only When It Is Communicated",
        "Using Feedback from Data Cleaning to Improve Data Quality",
    ],
    "Using Feedback from Data Cleaning to Improve Data Quality": [
        "Reporting Data-Cleaning Results",
        "Verifying Data-Cleaning Efforts",
        "Data Integrity and Its Risks in Data Analysis",
        "The Importance of Clean Data",
    ],
    "Refining a Resume for Data Analytics Roles": [
        "Tailoring Your Resume",
        "Using AI to Improve and Tailor Your Resume",
        "Key Factors to Consider When Choosing a Data Analytics Role",
        "Exploring Data Analyst Job Opportunities",
    ],
    "Exploring Data Analyst Job Opportunities": [
        "Choosing the Right Job Platforms",
        "Refining a Resume for Data Analytics Roles",
        "Networking for Job Search",
        "Key Factors to Consider When Choosing a Data Analytics Role",
    ],
})


# ======================================================================
# Section 5 — Analyze Data / Stage: organize  (analyze 001-004)
# ======================================================================

GLOSS.update({
    "Understanding Data Analysis":
        "what analysis actually is: the four phases that turn prepared data into insight",
    "Data Organization in Analysis":
        "arranging data so analysis is possible — the organising step before any computation",
    "Sorting and Filtering in Data Analysis":
        "the two foundational moves of analysis, and how they differ from cleaning uses",
    "Sorting Data in Spreadsheets":
        "ordering rows by one or more columns to surface structure — done safely",
})

CONTENT["Understanding Data Analysis"] = r"""
The heart of the craft
------------------------

The data is prepared and clean; now comes the phase the whole process has been
building toward — **analysis**: the work of turning organised data into insight
that answers the question. Understanding what analysis *is* — its steps and its
purpose — frames everything this section teaches. Analysis is not a single act
but a small process of its own, and knowing its shape keeps the hands-on
techniques oriented toward the goal.

What analysis is
------------------

**Data analysis** is the process of making sense of data to answer questions and
support decisions — identifying patterns, relationships, and trends that were not
visible in the raw rows. It sits inside the larger six-phase process (it is the
*Analyze* phase) but has its own internal structure, often described in four
steps:

- **Organize** — arrange the data so it can be worked with: sorted, filtered,
  formatted, structured for the question. This stage's subject.
- **Format and adjust** — get the data into consistent, analysis-ready form:
  correct types, consistent units, combined or split fields as needed.
- **Get input and combine** — bring together the data the question needs,
  including from multiple sources, and consult others where useful.
- **Transform and calculate** — the computation: aggregating, deriving, comparing
  — turning organised data into the numbers that answer the question.

These steps are not rigidly sequential — analysis loops among them — but together
they describe how prepared data becomes an answer.

Analysis versus the phases around it
--------------------------------------

Analysis is distinct from what surrounds it. It is *not* cleaning (that came
before — analysis assumes clean data) and *not* sharing (that comes after —
analysis produces the finding that sharing communicates). Its specific job is the
middle: taking data that is clean and prepared and *extracting the answer* from
it. Blurring these boundaries causes trouble — analysing dirty data, or jumping to
presentation before the analysis is sound — which is why the process separates
them.

What analysis produces
------------------------

The output of analysis is *insight* — an answer to the question, grounded in the
data: a pattern found, a comparison made, a trend identified, a relationship
revealed. Good analysis produces insight that is *correct* (the data genuinely
supports it), *relevant* (it bears on the decision), and *communicable* (it can be
conveyed to those who must act). The techniques of this section — sorting,
filtering, formatting, aggregating, combining, calculating — are all means to that
end: producing trustworthy insight from prepared data.

The caveat
------------

Analysis can produce apparent insights that are artefacts — patterns that are
noise, correlations that are coincidence, trends that are too short to be real.
The techniques ahead find patterns readily; the *judgement* to tell a real
finding from a spurious one is what makes analysis trustworthy, and it draws on
everything before it — the bias awareness, the sufficiency checks, the honest
skepticism. Analysis is not just running operations on data; it is running them
*and* judging whether what emerges is real. The next lessons begin with the first
step: organising data for analysis.
"""

CONTENT["Data Organization in Analysis"] = r"""
Arranging data to be analysed
-------------------------------

Analysis begins not with computation but with **organisation** — arranging the
data so that the patterns you seek can actually surface. Well-organised data makes
analysis fast and reliable; poorly organised data fights every operation. This
first step of analysis extends the tabular-structure and spreadsheet-organisation
disciplines from earlier into the service of finding answers.

What organising for analysis means
------------------------------------

Organising data for analysis includes several arranging activities:

- **Sorting** — ordering rows by a column to reveal structure (the next lessons'
  subject): extremes, rankings, chronology.
- **Filtering** — narrowing to the subset the question concerns, so the relevant
  data stands alone.
- **Grouping** — arranging data into the categories the analysis will summarise
  by (the aggregation lessons ahead build on this).
- **Structuring** — ensuring the data is in the right shape (tidy, correctly
  wide or long) for the analysis and tools you will use.

The goal is to get from "clean data sitting in a table" to "data arranged so the
question's answer is reachable" — the setup that makes the actual computation
straightforward.

Why organisation precedes computation
---------------------------------------

Jumping straight to calculation on unorganised data produces confusion or error:
an aggregate over unfiltered data answers the wrong question, a comparison across
inconsistently grouped data misleads, a trend sought in unsorted data stays
hidden. Organising first — sorting, filtering, grouping to match the question —
is what makes the subsequent computation both *possible* and *correct*. It is the
analysis-phase echo of the whole course's big-picture-first discipline: arrange
deliberately before you compute.

Organisation and the tools
----------------------------

Every tool in this section rewards good organisation. Spreadsheet sorting,
filtering, and pivot tables all assume the tidy structure from the prep section;
SQL's ``ORDER BY``, ``WHERE``, and ``GROUP BY`` are organisation expressed as
query. The organising *concepts* — order, subset, group, shape — are the same
across tools; only the syntax changes. Learning to think in these terms is
learning to see how to arrange any dataset toward any question, whichever tool you
reach for.

The caveat
------------

Organisation is preparation for analysis, not analysis itself — and it is possible
to over-organise, endlessly rearranging data without ever extracting the insight,
or to arrange it in a way that *presupposes* the answer (sorting and filtering
until only the data that supports a hunch remains). The discipline is to organise
in service of the *question*, not a desired conclusion, and to move on to the
actual analysis once the data is reachable. Organising is the means; the insight
is the end. The next lessons make the first organising move — sorting and
filtering — concrete.
"""

CONTENT["Sorting and Filtering in Data Analysis"] = r"""
The two foundational moves
----------------------------

Two operations underlie a huge share of all data analysis: **sorting** (ordering
data) and **filtering** (subsetting data). You met them as *cleaning* tools in the
preparation section; here they return as *analysis* tools, and the shift in
purpose is worth making explicit — the same operations, aimed now at finding
answers rather than fixing defects.

Sorting for analysis
----------------------

**Sorting** arranges rows in order by one or more columns, and in analysis its
purpose is to *reveal structure*:

- **Ranking** — sort descending to find the top (best-selling products, highest
  spenders) or ascending for the bottom.
- **Extremes** — sorting surfaces the largest and smallest values, where outliers
  and notable cases live.
- **Chronology** — sorting by date reveals the sequence and makes trends over time
  visible.
- **Grouping visually** — sorting by a category brings like records together for
  comparison.

Where cleaning used sorting to *find defects*, analysis uses it to *find answers*
— but the operation is identical.

Filtering for analysis
------------------------

**Filtering** shows only the rows meeting a condition, and in analysis its purpose
is *focus*:

- **Isolating a subset** — analyse just one segment (a region, a period, a
  customer type) by filtering to it.
- **Answering conditional questions** — "how many orders over $100?" is a filter
  plus a count.
- **Comparing subsets** — filter to one group, note the result; filter to another,
  compare.

Filtering is how you direct analysis at exactly the part of the data a question
concerns, ignoring the rest.

Sorting and filtering together
--------------------------------

Combined, they answer a vast range of questions with no further machinery: filter
to a subset, then sort it to rank — "the top ten customers in the northern region
this quarter" is one filter and one sort. This filter-then-sort pattern is the
analyst's most reflexive analytical move, and it recurs in every tool: spreadsheet
filter-and-sort, SQL ``WHERE`` plus ``ORDER BY``. Mastering it is mastering the
foundation the more advanced techniques build on.

The caveat
------------

Sorting and filtering shape what you *see*, which makes them subtly capable of
*misleading* an analysis — filtering to only the data that supports a conclusion
(dropping the inconvenient subset), or reading a sorted extreme as typical when it
is exceptional. The discipline is to filter and sort to answer the *question*
fairly, not to manufacture a desired result, and to stay aware of what a filtered
view excludes. And a reminder from the cleaning section: a filter only *hides*
rows while a sort *reorders* them — know which is active, and never mistake a
filtered view for the whole. The next lessons implement sorting in spreadsheets
and SQL.
"""

CONTENT["Sorting Data in Spreadsheets"] = r"""
Sorting, in practice
----------------------

The first hands-on analysis technique is **sorting in a spreadsheet** — ordering
rows by the values in one or more columns to surface the structure analysis
depends on. It is simple to do and immediately useful, and doing it *safely* is
the one thing that separates a helpful sort from a data-corrupting one.

Single-column sorting
-----------------------

The basic sort orders the whole table by one column, ascending or descending:

- **Ascending** — A→Z for text, smallest→largest for numbers,
  earliest→latest for dates.
- **Descending** — the reverse: Z→A, largest→smallest, latest→earliest.

Sort a sales column descending and the biggest sales rise to the top; sort a date
column ascending and the timeline orders itself. The single-column sort answers
"what are the extremes?" and "what is the order?" directly.

Multi-column sorting
----------------------

Sorting by *several* columns in priority order handles finer questions. Sort by
region first, then by sales within region: the spreadsheet groups all rows of each
region together, and within each region orders them by sales. This reveals "the
top sellers *within each* region" — a two-level structure a single sort cannot
show. The column order matters: the first sort column is primary, the second
breaks ties within it, and so on.

The critical safety rule
--------------------------

The one rule that makes sorting safe: **always sort the entire table together, so
every row moves as a complete unit.** When you sort, the spreadsheet must reorder
*all* columns' values by row, keeping each record intact. Sorting a *single column
in isolation* — reordering one column while the others stay put — silently
destroys the data by breaking the correspondence between a record's values: now a
customer's name sits beside a different customer's sales. This produces no error,
just quietly corrupted data, and it is one of the most damaging spreadsheet
mistakes. Modern spreadsheets warn you and default to whole-table sorting, but the
danger is real enough that the rule bears repeating: the record is the unit; sort
records, never lone columns.

Sorting as an analytical tool
-------------------------------

Beyond its safety mechanics, sorting is genuinely analytical: it is often the
fastest way to answer a ranking question ("who are our biggest customers?"), spot
the range of a variable (its smallest and largest values sit at the two ends), and
prepare data for a scan. Combined with filtering — filter to a subset, then sort
it — it handles a large share of everyday analytical questions with two clicks.

The caveat
------------

Sorting permanently reorders the data (unlike a filter, which only hides), so the
original row order — if it carried meaning, such as the sequence of entry — is
lost once you sort, unless you preserved it (an index column, or a copy). Before
sorting, consider whether the current order matters and needs saving; the
raw-stays-raw discipline protects you here. And sorting reveals extremes but does
not *explain* them — a value at the top of a sort is worth investigating, not
automatically a finding. The next lesson brings sorting, and filtering, into SQL.
"""


MINDMAP.update({
    "Understanding Data Analysis": [
        "Practical Application of the Data Analysis Process",
        "Data Organization in Analysis",
        "Understanding Common Problem Types in Data Analytics",
        "The Importance of Clean Data",
    ],
    "Data Organization in Analysis": [
        "Understanding Data Analysis",
        "Sorting and Filtering in Data Analysis",
        "Building and Organizing a Spreadsheet",
        "Data Validation in Spreadsheets",
    ],
    "Sorting and Filtering in Data Analysis": [
        "Data Organization in Analysis",
        "Sorting Data in Spreadsheets",
        "Sorting and Filtering Data in SQL Using ORDER BY and WHERE",
        "Sorting and Filtering Data in Spreadsheets",
    ],
    "Sorting Data in Spreadsheets": [
        "Sorting and Filtering in Data Analysis",
        "Sorting and Filtering Data in SQL Using ORDER BY and WHERE",
        "Sorting and Filtering Data in Spreadsheets",
        "Data Organization in Analysis",
    ],
})


# ======================================================================
# Section 5 — Analyze Data / organize (cont.)  (analyze 005-008)
# ======================================================================

GLOSS.update({
    "Sorting and Filtering Data in SQL Using ORDER BY and WHERE":
        "the SQL twins of sort and filter: ORDER BY to order, WHERE to subset",
    "Data Formatting and Unit Conversion in Spreadsheets":
        "consistent formats and units — making numbers comparable before analysing them",
    "Data Validation in Spreadsheets":
        "rules that constrain what a cell may hold, catching and preventing bad data",
    "Combining Data Validation and Conditional Formatting in Spreadsheets":
        "validation to enforce rules, formatting to reveal violations — used together",
})

CONTENT["Sorting and Filtering Data in SQL Using ORDER BY and WHERE"] = r"""
Sort and filter, in query form
--------------------------------

The two foundational moves — sorting and filtering — have direct SQL equivalents:
**ORDER BY** sorts, and **WHERE** filters. Everything you did with a spreadsheet's
sort and filter, SQL does with these two clauses, at database scale and in
repeatable text. This lesson makes the correspondence concrete, extending the
basic queries from the prep and cleaning sections into their analytical use.

Filtering with WHERE
----------------------

``WHERE`` restricts a query to rows meeting a condition — the SQL filter:

.. code-block:: sql

   SELECT product, region, amount
   FROM   orders
   WHERE  region = 'North'
     AND  amount > 100;

Only rows where the region is North *and* the amount exceeds 100 are returned. The
``WHERE`` toolkit from earlier applies fully: comparisons (``=``, ``<>``, ``>``,
``<``), combinations (``AND``, ``OR``, ``NOT``), ranges (``BETWEEN``), sets
(``IN``), and pattern matching (``LIKE``). ``WHERE`` is how you point analysis at
exactly the subset a question concerns.

Ordering with ORDER BY
------------------------

``ORDER BY`` sorts the result — the SQL sort:

.. code-block:: sql

   SELECT product, amount
   FROM   orders
   WHERE  region = 'North'
   ORDER  BY amount DESC;

``ORDER BY amount DESC`` returns rows largest-first (``ASC``, the default, is
smallest-first). Multi-column sorting works exactly as in a spreadsheet — list
columns in priority order:

.. code-block:: sql

   ORDER BY region ASC, amount DESC   -- by region, then by amount within region

This orders by region first, then by amount within each region — the SQL version
of the multi-column spreadsheet sort.

The two together: the analytical query
----------------------------------------

Combining ``WHERE`` and ``ORDER BY`` is the SQL filter-then-sort — the same
reflexive analytical move, now as a query. "The top ten northern orders this
quarter" becomes:

.. code-block:: sql

   SELECT   product, amount
   FROM     orders
   WHERE    region = 'North'
     AND    order_date >= '2024-01-01'
   ORDER BY amount DESC
   LIMIT    10;

``WHERE`` filters to the subset, ``ORDER BY`` ranks it, and ``LIMIT`` (a handy
companion) caps the output to the top ten. One query answers what would take
several spreadsheet steps — and reruns identically on new data.

The safety advantage
----------------------

SQL sorting sidesteps the spreadsheet's most dangerous sort mistake entirely:
``ORDER BY`` reorders the *query result*, always keeping each row's values
together, so the isolated-column corruption that plagues spreadsheets simply
cannot happen. And ``WHERE`` filters without altering the stored data — the result
is a view, the table untouched. SQL's structure makes these operations inherently
safer than their manual spreadsheet equivalents.

The caveat
------------

``WHERE`` and ``ORDER BY`` are precise about what they include and how they order,
which is not always what you intend. ``WHERE amount > 100`` silently excludes rows
where amount is *null* (the ``IS NULL`` trap from earlier), so a filter can drop
rows you meant to keep; and ``ORDER BY`` on a column with mixed or wrong types
sorts unexpectedly (text-numbers sort alphabetically, so "100" sorts before "99").
The result reflects exactly what you asked — verify it matches what you meant, the
same check-your-results habit as everywhere. The next lessons turn from ordering
and subsetting to getting the data's *format* analysis-ready.
"""

CONTENT["Data Formatting and Unit Conversion in Spreadsheets"] = r"""
Making numbers comparable
---------------------------

Organised data is not yet analysis-ready if its values are inconsistently
formatted or expressed in different units — you cannot meaningfully compare or
combine what is not on a common footing. **Data formatting and unit conversion**
is the step that puts values into consistent, comparable form, and it belongs in
the organise-and-prepare phase of analysis because analysis on mixed formats or
mixed units produces nonsense.

Consistent formatting
-----------------------

Formatting makes values *display and behave* consistently:

- **Number formats** — a consistent number of decimal places, consistent use of
  thousands separators, currency shown uniformly. Formatting affects display; the
  underlying value is unchanged, but consistent display prevents misreading.
- **Date formats** — one date format throughout (the wide/regional variation from
  the prep section), so dates sort and compare correctly and are not misread
  (is ``03/04`` March 4th or April 3rd?).
- **Text formats** — consistent capitalisation and spacing (the cleaning
  functions), so categories match.

Consistent formatting is partly cosmetic (readability) and partly functional
(correct sorting, comparison, and matching depend on it).

Unit conversion
-----------------

Unit conversion is more than cosmetic — it changes values so they are *genuinely
comparable*:

- **Currency** — amounts in different currencies must be converted to one before
  summing or comparing; adding dollars and euros as if identical is a real error.
- **Measurement units** — mixing metric and imperial (kilometres and miles,
  kilograms and pounds) produces meaningless aggregates until converted to one.
- **Time units** — durations in a mix of minutes and hours must be unified before
  arithmetic.
- **Scale** — values recorded in different scales (thousands vs. actual, per-day
  vs. per-month) must be brought to one scale to compare.

The rule: **before combining or comparing values, confirm they share a unit**, and
convert if they do not. This is one of the most common sources of the confident-
wrong result — an aggregate that summed incompatible units.

Why this belongs in analysis
------------------------------

Formatting and unit consistency is the "format and adjust" step of the analysis
process — the bridge between organised data and correct computation. Skipping it
produces the classic errors: a total that added mixed currencies, a comparison
between differently-scaled numbers, a date sort scrambled by inconsistent formats.
Getting values onto a common footing *before* calculating is what makes the
calculation meaningful.

The caveat
------------

Two cautions. First, **formatting versus value**: changing a cell's *display
format* (showing two decimals, adding a currency symbol) does not change the
underlying number, while *unit conversion* genuinely changes the value — confusing
the two leads to error, such as thinking you have converted currency when you have
only added a "$" to euro amounts. Second, **conversion introduces its own risk**:
an exchange rate applied wrongly, a conversion factor mistyped, or rounding
accumulated across many conversions can corrupt data — so conversions deserve the
same verification as any transformation. Confirm units are truly unified, not just
relabelled. The next lessons turn to *validating* data against rules.
"""

CONTENT["Data Validation in Spreadsheets"] = r"""
Rules that constrain a cell
-----------------------------

Much dirty data can be prevented at the point of entry by *constraining what a
cell may contain*. **Data validation** is a spreadsheet feature that sets rules
for the values allowed in cells — rejecting or flagging entries that violate them.
It is both a *cleaning* tool (catching existing bad data) and, more powerfully, a
*prevention* tool (stopping bad data from being entered in the first place),
which makes it a key part of preparing data for reliable analysis.

What validation rules do
--------------------------

A validation rule specifies what is acceptable in a cell:

- **Value ranges** — a number between 0 and 100, a date within a valid period,
  an amount that must be positive. Entries outside the range are rejected.
- **Lists (dropdowns)** — the cell may contain only a value from a fixed list
  (a set of valid regions, statuses, categories). This is the single most
  effective validation for preventing the inconsistency defect — if "region" can
  only be chosen from a dropdown, "NY"/"New York"/"new york" variants never arise.
- **Data types** — the cell must contain a number, a date, or text of a certain
  form.
- **Text length or format** — a code of exactly five characters, an entry
  matching a required pattern.
- **Custom rules** — formula-based conditions for more complex requirements.

When an entry violates the rule, the spreadsheet can reject it outright or warn
the user — either way, the bad value is caught at the moment it would enter.

Validation as prevention
--------------------------

The deepest value of validation is *preventing* dirty data rather than cleaning it
afterward — the source-improvement principle from the cleaning section, applied at
the cell level. A dropdown that constrains categories eliminates the inconsistency
defect before it exists; a range rule that rejects negative amounts prevents the
invalid values you would otherwise have to find and fix. Validation moves quality
control *upstream*, to the point of entry, where it is cheapest and most
effective. Every bad value prevented is one that never has to be detected,
diagnosed, and corrected.

Validation as detection
-------------------------

Applied to *existing* data, validation also *detects* violations: setting a
validation rule on a populated column flags the cells that fail it, surfacing the
invalid values already present. This makes validation a checking tool as well —
a way to audit a column against its rules and see what does not conform.

The caveat
------------

Validation is only as good as its rules, and rules have two failure modes.
**Too strict**, and legitimate values are rejected — a name-length rule that
blocks unusually long real names, a range that excludes valid extremes — which
frustrates entry and can lose real data. **Too loose**, and bad values still slip
through — a range rule that catches negatives but not implausibly large typos.
Well-designed validation reflects the data's genuine rules, neither more nor less,
and even good validation is a *filter*, not a guarantee: it enforces the
constraints you thought to specify, and cannot catch the errors you did not
anticipate. The next lesson combines validation with conditional formatting for a
stronger quality workflow.
"""

CONTENT["Combining Data Validation and Conditional Formatting in Spreadsheets"] = r"""
Two features, one quality workflow
------------------------------------

Data validation and conditional formatting are individually useful, but together
they form a stronger data-quality workflow: **validation enforces rules going
forward, conditional formatting reveals rule violations visually**. Combining them
gives both prevention and visibility — the two capabilities that keep data clean
and make remaining problems obvious.

The complementary roles
-------------------------

The two features address quality from different angles:

- **Data validation** — *constrains and prevents*. It stops bad values from being
  entered (a dropdown that only allows valid categories, a range that rejects
  negatives) and can reject or warn on violations. Its focus is the *future*:
  keeping new data clean.
- **Conditional formatting** — *reveals and highlights*. It colours cells by rule
  to make patterns and problems visible (highlighting duplicates, flagging
  out-of-range values, shading blanks). Its focus is *visibility*: making the
  state of existing data obvious at a glance.

Validation acts at the gate; conditional formatting acts on the whole field.
Together they cover both preventing and seeing.

Combining them in practice
----------------------------

A robust quality setup uses both on the same data:

- **Validation** on the input columns constrains what can be entered — dropdowns
  for categories, ranges for numbers, type rules for dates — preventing whole
  classes of dirty data.
- **Conditional formatting** on the same columns highlights anything noteworthy
  that validation does not outright prevent — values near a threshold, blanks that
  need filling, duplicates that validation alone does not catch — making them
  visible for review.

For example, a data-entry sheet might *validate* that ``status`` comes only from a
dropdown (preventing invalid statuses) while *conditionally formatting* rows where
``status`` is "overdue" in red (making the important ones visible). One feature
controls what goes in; the other draws the eye to what matters.

Why the combination is powerful
---------------------------------

Neither feature alone is complete. Validation prevents bad data but does not help
you *see* the state of what is there; conditional formatting reveals problems but
does not *prevent* them. Together they close the gap: bad data is largely stopped
at entry, and whatever slips through or needs attention is made visible. This is
the same defence-in-depth idea as the verification section — multiple checks
catching different problems — applied to spreadsheet data quality, combining a
preventive control with a detective one.

The caveat
------------

Combining features adds a maintenance burden: validation rules and formatting
rules both need to stay correct as the data and its requirements evolve, and rules
that drift out of date (a dropdown missing a newly-valid category, a formatting
threshold no longer meaningful) mislead rather than help. And neither feature, nor
both together, substitutes for actually *understanding* the data — they enforce
and reveal the rules you specified, but the judgement about what is genuinely right
or wrong in the data remains yours. This closes the spreadsheet-formatting and
validation portion of the organise stage; the next lessons turn to combining and
transforming text and data with functions and SQL.
"""


MINDMAP.update({
    "Sorting and Filtering Data in SQL Using ORDER BY and WHERE": [
        "Sorting Data in Spreadsheets",
        "Querying Data with SQL",
        "Core SQL Queries for Data Cleaning and Analysis",
        "Sorting and Filtering in Data Analysis",
    ],
    "Data Formatting and Unit Conversion in Spreadsheets": [
        "Data Types in Spreadsheets",
        "Data Validation in Spreadsheets",
        "Common Spreadsheet Errors and How to Fix Them",
        "The Importance of Clean Data",
    ],
    "Data Validation in Spreadsheets": [
        "Data Formatting and Unit Conversion in Spreadsheets",
        "Combining Data Validation and Conditional Formatting in Spreadsheets",
        "Sorting and Filtering Data in Spreadsheets",
        "The Importance of Clean Data",
    ],
    "Combining Data Validation and Conditional Formatting in Spreadsheets": [
        "Data Validation in Spreadsheets",
        "Data Formatting and Unit Conversion in Spreadsheets",
        "Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)",
        "Using CONCAT in SQL to Combine Text from Multiple Columns",
    ],
})


# ======================================================================
# Section 5 — Analyze Data / organize (close) + combine (open)  (analyze 009-012)
# ======================================================================

GLOSS.update({
    "Using CONCAT in SQL to Combine Text from Multiple Columns":
        "joining text from several columns into one — CONCAT and the concatenation operator",
    "Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)":
        "the spreadsheet string toolkit for measuring, extracting, and locating text",
    "Problem-Solving and Seeking Help in Data Analysis":
        "the analyst's debugging mindset, and how to ask for help effectively",
    "How to Effectively Search for Solutions Online as a Data Analyst":
        "turning an error or a stuck point into a search that actually finds the answer",
})

CONTENT["Using CONCAT in SQL to Combine Text from Multiple Columns"] = r"""
Joining text columns
----------------------

Analysis often needs to *combine* text from several columns into one — a full name
from first and last, an address from its parts, a label from a code and a
description. In SQL, this is **concatenation**, done with the ``CONCAT`` function
(or the concatenation operator), and it is the SQL counterpart of the
spreadsheet's ``CONCATENATE`` and ``&``.

How CONCAT works
------------------

``CONCAT`` takes several values and joins them into one string:

.. code-block:: sql

   SELECT CONCAT(first_name, ' ', last_name)        AS full_name,
          CONCAT(city, ', ', state, ' ', zip)        AS address
   FROM   customers;

Each argument is joined in order; literal text (like the space ``' '`` or the
comma-space ``', '``) is included as a separator so the result reads correctly.
Many databases also support the ``||`` operator for the same purpose
(``first_name || ' ' || last_name``), though the exact operator varies by system —
``CONCAT`` is the more portable choice.

Why concatenation matters in analysis
---------------------------------------

Combining columns serves several analytical needs:

- **Building keys** — concatenating fields to form a combined identifier for
  matching or grouping (a key from region and product code).
- **Creating labels** — assembling readable labels for output and visualisation
  from separate data columns.
- **Reassembling split data** — joining back fields that were separated (the
  inverse of the Text-to-Columns / ``SUBSTR`` splitting), when analysis needs them
  together.
- **Formatting for presentation** — producing human-readable combined strings for
  a report.

Concatenation is the "combine" counterpart to the "split" operations from earlier
— together they let you restructure text data into whatever shape the analysis or
output needs.

Concatenation with clean inputs
---------------------------------

Concatenation composes with the cleaning functions, because combining *dirty* text
produces dirty results — joining an untrimmed first name to a last name yields
``"Jane  Smith"`` with a stray space. The clean-then-combine pattern applies:
``CONCAT(TRIM(first_name), ' ', TRIM(last_name))`` trims the parts before joining,
so the combined result is clean. As with all such pipelines, the inputs' quality
determines the output's quality.

The caveat
------------

Concatenation has a null trap that catches beginners: in many databases, if *any*
argument to the older concatenation *operator* is null, the entire result becomes
null — so one missing middle name can null out an entire assembled address.
``CONCAT`` the *function* often handles nulls more gracefully (treating them as
empty strings), but behaviour **varies by database**, so combining columns that
may contain nulls requires care — often wrapping arguments in ``COALESCE`` to
substitute an empty string for nulls first. Verify how your database treats nulls
in concatenation before relying on it. The next lesson covers the parallel string
toolkit in spreadsheets.
"""

CONTENT["Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)"] = r"""
The spreadsheet string toolkit
--------------------------------

Text data — names, codes, addresses, categories — needs its own set of operations
to measure, extract, and locate parts of it, and spreadsheets provide a compact
**string function** toolkit for exactly this. Building on the cleaning functions
from earlier, these functions let you take apart, inspect, and manipulate text as
precisely as you manipulate numbers.

The core string functions
----------------------------

- ``LEN(text)`` — returns the **length** (number of characters) of a string.
  Useful for validation (is this code the right length?) and for calculations
  with other string functions.
- ``LEFT(text, n)`` — extracts the **first n characters** from the left
  (``LEFT("Product-123", 7)`` → ``"Product"``).
- ``RIGHT(text, n)`` — extracts the **last n characters** from the right
  (``RIGHT("Product-123", 3)`` → ``"123"``).
- ``MID(text, start, n)`` — extracts **n characters starting at a position**, for
  pulling from the middle.
- ``FIND(substring, text)`` — returns the **position** where a substring first
  appears (``FIND("-", "Product-123")`` → ``8``), or an error if not found.
  (``SEARCH`` is the case-insensitive variant.)

.. code-block:: text

   =LEN(A2)                    length of the text
   =LEFT(A2, 3)                first 3 characters
   =RIGHT(A2, 4)               last 4 characters
   =MID(A2, 5, 2)              2 characters starting at position 5
   =FIND("-", A2)              position of the first hyphen

Combining them to extract
---------------------------

The functions' real power comes from *combining* them to extract variable-length
parts. To pull everything before a hyphen when the hyphen's position varies, nest
``LEFT`` and ``FIND``:

.. code-block:: text

   =LEFT(A2, FIND("-", A2) - 1)    everything before the first hyphen

``FIND`` locates the hyphen, and ``LEFT`` takes everything up to just before it —
so ``"Product-123"`` yields ``"Product"`` and ``"Widget-45"`` yields ``"Widget"``,
each correctly, despite the different lengths. This ``LEFT``/``RIGHT``/``MID`` with
``FIND`` pattern is how you split text on a delimiter by formula, the flexible
counterpart to the fixed-position extractions.

Why string functions matter for analysis
-------------------------------------------

Text manipulation is constant analytical work: extracting a category code from a
product ID, pulling a domain from an email, splitting a combined field into its
parts for grouping, validating that codes have the right format. These functions
are the tools for all of it, and they mirror the SQL string functions (``SUBSTR``,
``LENGTH``, ``POSITION``) exactly — the same operations in a different syntax, so
learning them in the visible spreadsheet builds intuition that transfers to SQL.

The caveat
------------

String functions are precise about positions and counts, which makes them brittle
if the text's structure varies unexpectedly: ``LEFT(A2, 7)`` assumes the part you
want is always seven characters, and breaks when it is not; ``FIND`` returns an
error when the substring is absent, which propagates through a nested formula. The
robust approach uses *position-finding* (``FIND``) rather than fixed counts where
lengths vary, and wraps extractions in ``IFERROR`` to handle the cases where the
expected structure is missing — so a few irregular rows do not break the whole
column. Text is messier than it looks; extract defensively. This closes the
organise stage; the next lessons turn to the analyst's problem-solving craft as the
combine stage opens.
"""

CONTENT["Problem-Solving and Seeking Help in Data Analysis"] = r"""
Getting unstuck is a skill
----------------------------

Every analyst gets stuck — a formula that will not work, a query that errors, a
result that makes no sense, a technique they have never used. What separates
effective analysts is not avoiding these moments but *handling* them well:
**problem-solving** (systematically working toward a solution) and **seeking help**
(knowing when and how to ask). This lesson, opening the combine stage, is about the
debugging mindset that the hands-on techniques ahead will repeatedly demand.

The problem-solving mindset
-----------------------------

Effective problem-solving in analysis follows a rough method, echoing the root-
cause discipline from the foundations:

- **Understand the problem precisely.** What exactly is wrong? What did you expect
  versus what happened? A vague "it's broken" cannot be solved; "this query
  returns 0 rows when it should return hundreds" can.
- **Isolate it.** Narrow down where the problem is — which part of the formula,
  which clause of the query, which step of the process. Simplify until you find
  the smallest thing that reproduces the problem.
- **Form and test hypotheses.** Guess what might be causing it, then test that
  guess — change one thing, see if it helps. Systematic, one-variable-at-a-time
  testing beats random flailing.
- **Read the error.** Error messages usually say what is wrong (the spreadsheet
  error codes, the SQL error text); reading them carefully is often the whole
  solution.
- **Check your assumptions.** Frequently the problem is an assumption that turned
  out false — the data was not the type you thought, the column meant something
  different than expected.

This method — understand, isolate, hypothesise, test — is the analyst's debugging
loop, and it works on formulas, queries, and confusing results alike.

Seeking help effectively
--------------------------

When your own problem-solving stalls, seeking help is not failure — it is
efficient, provided you ask well:

- **Ask a specific, well-framed question.** "How do I do X" is weak; "I'm trying to
  do X, I tried Y, I expected A but got B — what am I missing?" gives a helper what
  they need to help.
- **Show what you tried.** Your attempt, the error, the relevant data shape — the
  context that lets someone diagnose rather than guess.
- **Ask the right source.** A colleague for domain or organisational context;
  online resources and communities for technical problems (the next lesson);
  documentation for how a tool works.

A good question often solves itself in the asking — articulating the problem
clearly frequently reveals the answer, the "rubber duck" effect.

The caveat
------------

Both halves have a failure mode. Problem-solving can become *stubbornness* —
grinding on a problem far past the point where asking would be faster, wasting
hours to avoid a five-minute question. Seeking help can become *dependence* —
asking before making any real attempt, which neither solves the immediate problem
efficiently nor builds your own capability. The balance is a reasonable attempt
first (understand, isolate, try), then ask when genuinely stuck — and always in a
way that shows your work, so the help teaches you rather than just unblocking you.
The next lesson develops the most common help-seeking skill: searching online
effectively.
"""

CONTENT["How to Effectively Search for Solutions Online as a Data Analyst"] = r"""
The analyst's most-used skill
-------------------------------

No analyst remembers every function's syntax or every error's cause — and none
needs to, because the answer is almost always a good search away. **Searching
online effectively** is, realistically, one of the most-used skills in data work:
turning a stuck point or an error into a query that finds the solution. It is not a
sign of not knowing enough; it is how professionals work, and doing it well is a
genuine skill.

Turning a problem into a good search
--------------------------------------

The key is formulating a query that matches how the answer is likely phrased:

- **Search the error message.** For an error, searching the *exact* error text
  (often minus your specific values) is the fastest route — someone has almost
  certainly hit the same error and asked about it. Paste the error, remove the
  parts unique to your data.
- **Include the tool and version.** "pivot table" alone is vague; "Excel pivot
  table group by month" or "BigQuery SQL date extract" targets the right tool and
  syntax, since solutions differ across tools.
- **Describe the goal, not just the symptom.** Search what you are *trying to do*
  ("SQL combine text from two columns") as well as what went wrong — the goal-based
  search often finds a direct how-to.
- **Use the right vocabulary.** Learning the correct term (concatenation, join,
  aggregate, VLOOKUP) makes searches far more effective; a symptom described in
  lay terms finds less than the technical name.

Reading and using results
----------------------------

Finding a candidate solution is not the end — you must evaluate and adapt it:

- **Judge the source.** Official documentation and well-regarded technical
  communities are more reliable than random posts; a heavily-upvoted, explained
  answer beats an unexplained snippet.
- **Understand before applying.** Do not paste a solution you do not understand —
  adapt it to your data, and grasp *why* it works, both to trust it and to learn.
  A copied query that happens to run is a liability if you cannot tell whether it
  did the right thing.
- **Adapt, do not just copy.** A found solution uses its author's column names and
  assumptions; adapting it to yours (and verifying the result) is the actual work.

Searching as learning
------------------------

Done well, searching is not just problem-solving but *learning*: each solved
problem, understood rather than blindly copied, adds to what you know, so you
search for the same thing less often over time. The analysts who improve fastest
are those who, having found a solution, take the extra minute to understand *why*
it works — turning a fix into knowledge.

The caveat
------------

Online solutions carry real risks. They can be **wrong**, **outdated** (a solution
for an old software version that no longer applies), or **subtly inappropriate**
for your situation — a query that works on the author's data but mishandles an edge
case in yours. And the deepest hazard is applying a solution you do not understand:
it may run without error and still produce the *wrong answer*, which is worse than
an error because nothing flags it. Always verify a found solution against your own
data and expectations (the check-your-results habit), and never let "it ran" stand
in for "it's correct." The next lessons return to hands-on combining, starting with
choosing the right tool for the task.
"""


MINDMAP.update({
    "Using CONCAT in SQL to Combine Text from Multiple Columns": [
        "Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)",
        "Advanced SQL Functions for Data Cleaning",
        "Core SQL Queries for Data Cleaning and Analysis",
        "Using Spreadsheet Functions for Data Cleaning",
    ],
    "Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)": [
        "Using CONCAT in SQL to Combine Text from Multiple Columns",
        "Spreadsheet Functions",
        "Using Spreadsheet Functions for Data Cleaning",
        "Data Validation in Spreadsheets",
    ],
    "Problem-Solving and Seeking Help in Data Analysis": [
        "How to Effectively Search for Solutions Online as a Data Analyst",
        "Analytical Thinking and Questions for Problem Solving",
        "Choosing the Right Tool in Data Analysis",
        "Troubleshooting VLOOKUP and Building a Problem-Solving Framework",
    ],
    "How to Effectively Search for Solutions Online as a Data Analyst": [
        "Problem-Solving and Seeking Help in Data Analysis",
        "Choosing the Right Tool in Data Analysis",
        "Analytical Thinking and Questions for Problem Solving",
        "Troubleshooting VLOOKUP and Building a Problem-Solving Framework",
    ],
})


# ======================================================================
# Section 5 — Analyze Data / combine (VLOOKUP arc)  (analyze 013-016)
# ======================================================================

GLOSS.update({
    "Choosing the Right Tool in Data Analysis":
        "matching the task to spreadsheet, SQL, or programming — and combining them",
    "Preparing Data for VLOOKUP in Spreadsheets":
        "the setup VLOOKUP demands: a clean lookup key in the leftmost column",
    "Using VLOOKUP to Combine Data Across Spreadsheets":
        "pulling matching values from another table by a shared key",
    "Troubleshooting VLOOKUP and Building a Problem-Solving Framework":
        "why VLOOKUP fails, how to fix it, and a reusable debugging framework",
})

CONTENT["Choosing the Right Tool in Data Analysis"] = r"""
The right tool for the task
-----------------------------

An analyst with several tools faces a recurring choice: which one for *this* task?
The spreadsheets-versus-SQL comparison from the cleaning section generalises here
into the analytical work of *combining* and *transforming* data, where the choice
between a spreadsheet, SQL, and a programming language shapes how efficiently and
reliably a job gets done. Choosing well is itself an analytical skill.

The tools and their strengths
-------------------------------

- **Spreadsheets** — best for small-to-medium data, visual and interactive work,
  quick exploration, and combining a couple of tables by key (VLOOKUP, the next
  lessons). Everyone can open the result; every value is visible.
- **SQL** — best for large data, combining *many* tables (JOINs), reproducible
  queries, and working at the database source. Scales where spreadsheets strain.
- **Programming (Python/R)** — best for complex transformations, automation,
  statistical work, repeatable pipelines, and anything the other two cannot
  express. The most powerful and the most involved (the Python section develops
  this).

Each tool has a range where it is clearly the right choice, and overlaps where
more than one would work.

The factors that decide
-------------------------

The choice turns on the same considerations as before, now applied to analytical
tasks:

- **Data size** — small favours spreadsheets; large favours SQL or programming.
- **Complexity** — simple combines suit spreadsheets; complex, multi-table, or
  multi-step work suits SQL or code.
- **Repetition** — one-off favours spreadsheets; recurring favours the
  reproducibility of SQL and code.
- **The audience and output** — a stakeholder-facing summary may need a
  spreadsheet's presentation; a data pipeline needs code.
- **What you know** — the tool you can use *well* is often the right one for a
  one-off, even if another would be marginally better in principle.

Combining tools
----------------

The choice is rarely exclusive — the tools *chain*, as the cleaning section showed.
A common analytical pattern: **SQL** joins and aggregates large data at the source,
its result exported to a **spreadsheet** for exploration and a stakeholder summary,
with **Python** automating the whole flow if it must run repeatedly. Fluency is
knowing not just each tool but *how they fit together*, so a complex task is
decomposed across the tools each part suits best.

The caveat
------------

The "right tool" is partly objective (size and complexity genuinely favour certain
tools) and partly practical (the tool you know well, the tool your team uses, the
tool the data already lives in). Dogmatically insisting on the theoretically-optimal
tool for every task — reaching for SQL on ten rows, or a spreadsheet on ten
million — serves no one; and constantly switching tools has its own cost in
context and error. Match the tool to the task's genuine demands *and* your and your
team's practical reality, and expect most real work to use more than one. The next
lessons dig into the spreadsheet's core data-combining tool: VLOOKUP.
"""

CONTENT["Preparing Data for VLOOKUP in Spreadsheets"] = r"""
Setup before lookup
----------------------

The spreadsheet's primary tool for combining data across tables is **VLOOKUP** —
but it is notoriously particular about its inputs, and most VLOOKUP failures are
really *preparation* failures. Before using VLOOKUP, the data must be set up the
way it requires, and getting that setup right is most of the battle. This lesson
covers the preparation; the next covers the lookup itself.

What VLOOKUP does, briefly
----------------------------

VLOOKUP ("vertical lookup") searches for a value in the first column of a table and
returns a value from another column in the same row — the spreadsheet's way of
pulling matching data from one table into another by a shared key. To combine an
orders sheet with a products sheet, VLOOKUP looks up each order's product code in
the products table and returns that product's name or price. It is the spreadsheet
counterpart of the SQL JOIN (lessons ahead).

The preparation VLOOKUP requires
----------------------------------

VLOOKUP imposes specific setup requirements, and violating any of them causes it to
fail or mislead:

- **The lookup key must be in the leftmost column of the lookup range.** VLOOKUP
  searches only the *first* column of the table it is given, so the key you match
  on must be that first column. This is VLOOKUP's most rigid constraint and a
  frequent cause of failure — if your key is not leftmost, you must rearrange the
  data or use a different function.
- **The key must match exactly across both tables.** The lookup value and the key
  in the lookup table must be identical — same format, same type, no stray spaces.
  ``"C-001"`` will not match ``"C-001 "`` (trailing space) or ``"c-001"`` (case) or
  a number ``1`` stored where text ``"1"`` is expected. This is where the cleaning
  functions earn their place: ``TRIM`` and consistent typing on both keys *before*
  the lookup.
- **The key should be unique in the lookup table.** VLOOKUP returns the *first*
  match it finds; if the key repeats, you get only the first, silently ignoring the
  rest — so the lookup table's key should identify rows uniquely.
- **The lookup range must cover all the needed rows and columns.** The range must
  include every row the key might match and the column you want to return.

Preparation as the real work
------------------------------

Notice that the preparation is essentially *cleaning and structuring for the
match* — the same key-matching discipline as merging datasets from the cleaning
section. Clean keys, consistent types, a unique leftmost key column, an adequate
range: get these right and VLOOKUP works; get any wrong and it fails in ways that
can be baffling until you know to check the setup. The lesson underneath is that
VLOOKUP problems are usually *data* problems, solved before the formula.

The caveat
------------

VLOOKUP's leftmost-key constraint is a genuine limitation, not just a preparation
step: sometimes the key genuinely cannot be the leftmost column without disrupting
the data, and forcing it (duplicating the key column to the left) is awkward.
Modern spreadsheets offer more flexible alternatives (INDEX/MATCH, or newer lookup
functions like XLOOKUP) that do not require the key to be leftmost and handle more
cases gracefully — worth knowing when VLOOKUP's constraints bind. But VLOOKUP
remains the most widely taught and encountered, so understanding it (and its
preparation) is essential. The next lesson performs the lookup.
"""

CONTENT["Using VLOOKUP to Combine Data Across Spreadsheets"] = r"""
Performing the lookup
-----------------------

With the data prepared — clean keys, a leftmost lookup column, an adequate range —
VLOOKUP performs the actual combination: pulling matching values from one table
into another. This lesson covers the formula itself, its four arguments, and how it
brings separate tables together by a shared key.

The VLOOKUP formula
---------------------

VLOOKUP takes four arguments:

.. code-block:: text

   =VLOOKUP(lookup_value, table_range, column_index, FALSE)

- **lookup_value** — the value to search for (the key), e.g. this row's product
  code.
- **table_range** — the table to search, whose *first column* holds the keys and
  whose other columns hold the values to return.
- **column_index** — *which column* of that range to return, counted from the left
  (1 is the key column itself, 2 the next, and so on).
- **the match type** — ``FALSE`` (or ``0``) for an **exact match**, ``TRUE`` for an
  approximate match. **Almost always use ``FALSE``**: exact match is what combining
  data by key requires; ``TRUE`` (approximate) is for a different, rarer purpose and
  is a common source of silent wrong results.

.. code-block:: text

   =VLOOKUP(A2, products!A:C, 2, FALSE)

This reads: take the key in ``A2``, search the first column of ``products!A:C``,
and return the value from the *second* column of that range (an exact match). Fill
it down the column, and every row pulls its matching product detail from the
products sheet.

Combining across sheets
-------------------------

The power is combining *separate* tables. An orders sheet and a products sheet,
linked by product code, become one enriched view: each order row gains its
product's name and price via VLOOKUP, without manually copying anything. This is
exactly the relational combine — matching on a key to bring related data together —
performed in a spreadsheet, and it is the everyday tool for the "combine data from
two sources" task the cleaning section described.

VLOOKUP and the always-FALSE rule
-----------------------------------

The single most important habit: **use ``FALSE`` (exact match) unless you have a
specific reason not to.** Approximate match (``TRUE``) assumes the lookup column is
sorted and returns the *closest* value at or below the lookup value — which for
combining data by key is almost never what you want, and produces plausible *wrong*
matches with no error. New users who omit the argument or use ``TRUE`` get
mysterious wrong results; ``FALSE`` is the safe default for combining data, and
should be your reflex.

The caveat
------------

VLOOKUP returns the *first* match and only pulls *one* column per formula, which has
consequences: if the key is not unique in the lookup table, you silently get only
the first matching row's value (a data problem the preparation lesson flagged); and
returning several columns needs several VLOOKUPs (or a different function). VLOOKUP
also breaks if columns in the lookup range are inserted or deleted, since the
``column_index`` is a fixed number — a fragility that INDEX/MATCH and XLOOKUP avoid.
And a returned value that *looks* right can still be wrong if the keys did not truly
match as intended, so verify a sample of lookups against the source. The next
lesson turns to diagnosing exactly these VLOOKUP failures.
"""

CONTENT["Troubleshooting VLOOKUP and Building a Problem-Solving Framework"] = r"""
When the lookup goes wrong
----------------------------

VLOOKUP fails often, and its failures are frequently baffling until you know the
handful of usual causes. This lesson catalogues them — and then uses VLOOKUP
troubleshooting as a worked example of a *reusable problem-solving framework* that
applies far beyond VLOOKUP, tying the combine stage's problem-solving lessons to a
concrete case.

The common VLOOKUP failures
-----------------------------

Most VLOOKUP problems trace to a short list of causes:

- **#N/A error — no match found.** The lookup value is not in the first column of
  the range. Usual reasons: a **key mismatch** (trailing spaces, different case, a
  number stored as text versus a real number), the key **genuinely absent** from
  the lookup table, or the key **not in the leftmost column** of the range.
- **Wrong value returned.** Usually **approximate match** (``TRUE`` instead of
  ``FALSE``) returning a near value, or a **wrong ``column_index``** returning the
  wrong column.
- **#REF! error.** The ``column_index`` exceeds the number of columns in the range.
- **Results break after editing.** A column was inserted or deleted, shifting what
  the fixed ``column_index`` points to.

Recognising the *symptom* (which error, or which kind of wrong result) points
quickly at the likely *cause* — the essence of efficient troubleshooting.

The diagnostic sequence
-------------------------

Troubleshooting VLOOKUP follows a systematic check, and this sequence *is* the
reusable framework:

1. **Read the symptom precisely** — ``#N/A``? wrong value? ``#REF!``? Each points at
   different causes.
2. **Check the most common cause first** — for ``#N/A``, check the key match:
   are the keys *really* identical (trim both, confirm same type)? This one cause
   explains most failures.
3. **Isolate** — test the lookup on a single row you know should match; simplify
   until the problem is cornered.
4. **Verify assumptions** — is the key truly in the leftmost column? Is
   ``column_index`` correct? Is the match type ``FALSE``?
5. **Fix at the cause** — clean the keys, rearrange the columns, correct the
   argument — not by patching around the symptom.

The reusable problem-solving framework
----------------------------------------

Notice this sequence is *not specific to VLOOKUP* — it is the general debugging loop
from the problem-solving lesson, made concrete: **read the symptom, hypothesise the
most likely cause, isolate, check assumptions, fix at the root.** The same framework
diagnoses a broken SQL query, a wrong formula, or a confusing result. VLOOKUP
troubleshooting is worth learning both for itself and as *practice of a
transferable method* — the analyst who internalises "symptom → likely cause →
isolate → verify → fix at root" can debug anything, which is why this lesson closes
the combine stage's problem-solving thread before the SQL-combining lessons.

The caveat
------------

A troubleshooting framework guides diagnosis but does not replace *understanding* —
you can follow the steps mechanically and still miss a cause you do not understand
(a locale-specific number format, a non-printing character in the key). The
framework is most powerful combined with knowledge of how the tool actually works,
so that "check assumptions" is informed by knowing which assumptions VLOOKUP makes.
And frameworks can become rote: the goal is not to recite steps but to build the
*habit of systematic diagnosis* over panic or random flailing. The next lessons move
from spreadsheet combining to its more powerful SQL counterpart: the JOIN.
"""


MINDMAP.update({
    "Choosing the Right Tool in Data Analysis": [
        "Spreadsheets vs. SQL",
        "Preparing Data for VLOOKUP in Spreadsheets",
        "Problem-Solving and Seeking Help in Data Analysis",
        "Using JOIN in SQL to Combine Tables",
    ],
    "Preparing Data for VLOOKUP in Spreadsheets": [
        "Using VLOOKUP to Combine Data Across Spreadsheets",
        "Cleaning and Merging Multiple Datasets",
        "Data Mapping and the Big Picture of Clean Data",
        "Choosing the Right Tool in Data Analysis",
    ],
    "Using VLOOKUP to Combine Data Across Spreadsheets": [
        "Preparing Data for VLOOKUP in Spreadsheets",
        "Troubleshooting VLOOKUP and Building a Problem-Solving Framework",
        "Using JOIN in SQL to Combine Tables",
        "Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)",
    ],
    "Troubleshooting VLOOKUP and Building a Problem-Solving Framework": [
        "Using VLOOKUP to Combine Data Across Spreadsheets",
        "Problem-Solving and Seeking Help in Data Analysis",
        "Common Spreadsheet Errors and How to Fix Them",
        "How to Effectively Search for Solutions Online as a Data Analyst",
    ],
})


# ======================================================================
# Section 5 — Analyze Data / combine (close)  (analyze 017-019)
# ======================================================================

GLOSS.update({
    "Using JOIN in SQL to Combine Tables":
        "combining rows from multiple tables on a matching key — SQL's core combine",
    "Subqueries in SQL":
        "a query nested inside another — using one query's result within a second",
    "Aggregating Data with Subqueries, HAVING, and CASE in SQL":
        "combining aggregation, group filtering, and conditional logic for real analysis",
})

CONTENT["Using JOIN in SQL to Combine Tables"] = r"""
The SQL way to combine tables
-------------------------------

VLOOKUP combines tables in a spreadsheet; **JOIN** does it in SQL — and far more
powerfully. A JOIN combines rows from two (or more) tables based on a related
column between them, producing a single result set that draws columns from each.
It is the operation the relational model from the prep section was built for, and
one of the most important skills in all of SQL.

How a JOIN works
------------------

A JOIN matches rows across tables on a **key** — the primary/foreign key
relationship from the prep section made operational:

.. code-block:: sql

   SELECT orders.order_id,
          orders.amount,
          customers.name,
          customers.region
   FROM   orders
   JOIN   customers
     ON   orders.customer_id = customers.customer_id;

The ``ON`` clause states the match condition — where ``orders.customer_id`` equals
``customers.customer_id`` — and the result pairs each order with its customer's
details, drawing columns from both tables. This is the relational combine: data
stored separately (orders in one table, customers in another) brought together by
their key relationship.

The types of JOIN
-------------------

Which rows appear depends on the *join type*:

- **INNER JOIN** (the default ``JOIN``) — returns only rows with a match in *both*
  tables. Orders whose customer is missing, and customers with no orders, are
  excluded. The most common join.
- **LEFT JOIN** — returns *all* rows from the left table, matched where possible,
  with nulls where the right table has no match. "All orders, with customer
  details where available" — keeps unmatched left rows.
- **RIGHT JOIN** — the mirror: all rows from the right table, nulls where the left
  has no match.
- **FULL (OUTER) JOIN** — all rows from *both* tables, matched where possible,
  nulls elsewhere.

The choice matters: an INNER JOIN silently *drops* unmatched rows (which may be
exactly what you want, or a silent loss of data), while a LEFT JOIN *keeps* them —
so choosing the wrong type changes which rows your analysis sees.

JOIN versus VLOOKUP
--------------------

JOIN is VLOOKUP's more capable relative. Where VLOOKUP pulls *one column* from
*one* lookup table by a leftmost key and returns the *first* match, a JOIN combines
*all needed columns* from *multiple* tables, on *any* matching columns (not just a
leftmost one), handling multiple matches explicitly. For combining data at any
scale or complexity, JOIN is the tool — which is why SQL is preferred over
spreadsheets for serious multi-table work.

The caveat
------------

JOINs have a signature danger: joining on a **non-unique key** multiplies rows. If a
customer has three orders and you join in a way that matches each order to each of
two addresses, you get *six* rows where you expected three — a "fan-out" that
silently inflates counts and sums. This is the SQL version of the merge row-count
explosion from the cleaning section, and the defence is the same: **check the row
count** before and after the join against expectation, and understand the *
cardinality* of the relationship (one-to-one, one-to-many, many-to-many) before
joining. A JOIN that returns more rows than expected has usually matched on a key
that was not as unique as assumed. The next lessons add nesting queries within
queries — subqueries.
"""

CONTENT["Subqueries in SQL"] = r"""
A query inside a query
------------------------

Sometimes answering a question requires the *result of one query* to be used
*inside another*. A **subquery** is exactly that — a query nested within another
query, whose result the outer query uses. Subqueries let you build analyses in
layers, answering a preliminary question and then querying against its answer, and
they are a step up in SQL expressiveness.

How subqueries work
---------------------

A subquery is a ``SELECT`` written inside another query, often in the ``WHERE``
clause:

.. code-block:: sql

   SELECT name, amount
   FROM   orders
   WHERE  amount > (SELECT AVG(amount) FROM orders);

The inner query ``(SELECT AVG(amount) FROM orders)`` computes the average amount;
the outer query then returns orders *above* that average. The subquery runs first,
produces a value, and the outer query uses it — answering "which orders are
above-average?" in one statement, without manually finding the average first.

Where subqueries appear
-------------------------

Subqueries serve in several positions:

- **In WHERE, for comparison** — comparing each row against a computed value
  (above average, above a threshold derived from the data), as above.
- **In WHERE with IN, for membership** — testing whether a value is in a set the
  subquery produces:

  .. code-block:: sql

     SELECT name FROM customers
     WHERE customer_id IN (SELECT customer_id FROM orders WHERE amount > 1000);

  This finds customers who placed a large order — the subquery lists the qualifying
  customer IDs, the outer query returns their names.
- **In FROM, as a derived table** — using a subquery's result as a table to query
  further, building a multi-step analysis in layers.
- **In SELECT, for a computed column** — producing a value per row from a related
  query.

Why subqueries matter
-----------------------

Subqueries let you express analyses that need an *intermediate result* — a
comparison against an aggregate, a filter based on another table's contents, a
calculation built in stages — without breaking the work into separate manual
steps. They keep a multi-step analysis in one repeatable query, and they are the
foundation for the layered aggregation the next lesson builds. Much of SQL's
analytical power comes from composing queries this way.

The caveat
------------

Subqueries can grow hard to read and, sometimes, slow: a deeply nested query is
difficult to follow and debug (build and test it in layers, inner query first), and
a *correlated* subquery — one that references the outer query and thus re-runs for
every outer row — can be very slow on large data. Often a JOIN accomplishes the
same result more efficiently and readably, so when a subquery grows complex, ask
whether a JOIN would serve better. And the usual precision caution applies:
subqueries involving nulls or empty results can behave unexpectedly (a ``NOT IN``
subquery that encounters a null can return no rows), so verify results against
expectation. The next lesson combines subqueries with aggregation and conditional
logic for real analytical queries.
"""

CONTENT["Aggregating Data with Subqueries, HAVING, and CASE in SQL"] = r"""
Bringing the pieces together
------------------------------

Real analytical queries combine several SQL capabilities at once: aggregating with
``GROUP BY``, filtering groups with ``HAVING``, applying conditional logic with
``CASE``, and layering with subqueries. This lesson assembles them — the
culmination of the combine stage — into the kind of query that answers a genuine
business question, closing the loop on SQL as an analytical tool.

Filtering groups with HAVING
------------------------------

``WHERE`` filters *rows*; ``HAVING`` filters *groups* after aggregation — the
distinction that trips up beginners:

.. code-block:: sql

   SELECT   region, COUNT(*) AS orders, SUM(amount) AS revenue
   FROM     orders
   GROUP BY region
   HAVING   SUM(amount) > 10000;

This groups orders by region, computes each region's count and revenue, then keeps
only regions whose *total revenue* exceeds 10,000. ``WHERE`` cannot do this — the
condition is on an aggregate (``SUM``) that does not exist until after grouping, so
it must be ``HAVING``. The rule: **``WHERE`` filters before grouping (on individual
rows), ``HAVING`` filters after grouping (on aggregated values).**

Conditional aggregation with CASE
-----------------------------------

``CASE`` inside an aggregate enables powerful conditional counting and summing —
computing different aggregates for different conditions in one query:

.. code-block:: sql

   SELECT region,
          COUNT(*)                                          AS total_orders,
          SUM(CASE WHEN amount > 100 THEN 1 ELSE 0 END)     AS large_orders,
          SUM(CASE WHEN amount > 100 THEN amount ELSE 0 END) AS large_revenue
   FROM     orders
   GROUP BY region;

Each ``CASE`` inside a ``SUM`` counts or totals only the rows meeting its condition
— "how many large orders, and their revenue, per region" — in a single grouped
query. This conditional-aggregation pattern answers segmented questions (this
category versus that, above threshold versus below) without separate queries, and
it is one of the most useful analytical techniques in SQL.

Layering with subqueries
--------------------------

Subqueries add another layer — aggregating, then querying the aggregate:

.. code-block:: sql

   SELECT region, revenue
   FROM   (SELECT region, SUM(amount) AS revenue
           FROM orders GROUP BY region) AS regional
   WHERE  revenue > (SELECT AVG(amount) * 100 FROM orders);

The inner query aggregates revenue by region; the outer query filters those
regional totals against a data-derived threshold. Composing aggregation inside a
subquery lets you analyse *summaries* — asking questions about grouped results, a
genuinely multi-step analysis in one statement.

The analytical payoff
-----------------------

Together, ``GROUP BY``, ``HAVING``, ``CASE``, and subqueries turn SQL from a
retrieval language into an *analytical* one: segment the data, aggregate it, filter
the aggregates, apply conditional logic, and layer the whole thing — answering
questions that would take many manual spreadsheet steps in one repeatable query.
This is the analytical core the section has been building toward in SQL, and it is
why SQL is the backbone of serious data analysis.

The caveat
------------

Combining these features multiplies both power and the room for error. The
``WHERE``/``HAVING`` confusion produces wrong results silently (filtering rows when
you meant groups, or vice versa); ``CASE`` conditions that do not cover all cases
leave gaps; nulls interact with aggregates subtly (``COUNT(*)`` versus
``COUNT(column)``, ``AVG`` ignoring nulls); and a complex query can be *confidently
wrong* in ways only careful checking reveals. The discipline is to **build complex
queries incrementally** — get the ``GROUP BY`` right, add ``HAVING``, add ``CASE``,
verifying at each step — rather than writing the whole thing at once, and to check
the results against expectation and simpler queries. Complexity earns power only if
it stays correct. This closes the combine stage; the next turns to calculations and
trend analysis, in spreadsheets and SQL.
"""


MINDMAP.update({
    "Using JOIN in SQL to Combine Tables": [
        "Using VLOOKUP to Combine Data Across Spreadsheets",
        "Databases and Relational Database Concepts",
        "Subqueries in SQL",
        "Cleaning and Merging Multiple Datasets",
    ],
    "Subqueries in SQL": [
        "Using JOIN in SQL to Combine Tables",
        "Aggregating Data with Subqueries, HAVING, and CASE in SQL",
        "Core SQL Queries for Data Cleaning and Analysis",
        "Advanced SQL Functions for Data Cleaning",
    ],
    "Aggregating Data with Subqueries, HAVING, and CASE in SQL": [
        "Subqueries in SQL",
        "Using JOIN in SQL to Combine Tables",
        "Core SQL Queries for Data Cleaning and Analysis",
        "Sorting and Filtering Data in SQL Using ORDER BY and WHERE",
    ],
})


# ======================================================================
# Section 5 — Analyze Data / Stage: calc  (analyze 020-023)
# ======================================================================

GLOSS.update({
    "Using Spreadsheet Formulas for Sales Trend Analysis":
        "formulas for change over time — growth rates, running totals, period comparisons",
    "Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets":
        "counting and summing only the rows that meet a condition",
    "Using SUMPRODUCT for Advanced Spreadsheet Calculations":
        "multiplying and summing across arrays in one formula — weighted and multi-condition sums",
    "Using Pivot Tables for Calculations and Trend Analysis":
        "the spreadsheet's most powerful summarising tool — grouping and aggregating with drags",
})

CONTENT["Using Spreadsheet Formulas for Sales Trend Analysis"] = r"""
Calculating change over time
------------------------------

The heart of analysis is *calculation* — turning organised data into the numbers
that answer a question — and one of the most common calculations is **trend
analysis**: how a value changes over time. Sales trend analysis is the classic
case, and the spreadsheet formulas that compute it — growth rates, running totals,
period-over-period comparisons — are foundational analytical tools, opening the
calculation stage.

The core trend calculations
-----------------------------

Trends are built from a few standard computations, each a formula pattern:

- **Period-over-period change** — the difference between consecutive periods:
  ``= this_period - last_period``. The absolute change.
- **Growth rate (percentage change)** — the change *relative* to the base:

  .. code-block:: text

     = (this_period - last_period) / last_period

  This expresses growth as a percentage (12% up, 5% down), which is comparable
  across different scales in a way absolute change is not.
- **Running (cumulative) total** — the sum accumulated up to each period, built
  with a mix of absolute and relative references so it extends down the column:

  .. code-block:: text

     = SUM($B$2:B2)     fill down: running total through each row

  The ``$B$2`` anchors the start (absolute) while ``B2`` moves (relative), so each
  row sums from the beginning through itself.
- **Comparison to a baseline** — each period against a fixed reference (a target,
  the same month last year), using an absolute reference to the baseline cell.

Reading the trend
------------------

The formulas produce numbers; *reading* them is the analysis. A growth rate turns
raw sales into a story — accelerating, steady, declining. A running total shows
progress toward a goal. A year-over-year comparison strips out seasonality that a
month-over-month view would confuse. The analyst's job is not just computing these
but interpreting what they reveal about the direction and health of what is
measured.

The absolute/relative reference connection
--------------------------------------------

Trend formulas lean heavily on the relative-versus-absolute reference distinction
from Section 2: the running total's ``SUM($B$2:B2)`` and the baseline comparison's
anchored reference both depend on getting the ``$`` right so the formula behaves
correctly when filled down. This is the earlier concept put to real analytical
work — a reminder that the foundational skills compound into the analytical ones.

The caveat
------------

Trend calculations can mislead in familiar ways. A percentage change from a *small*
base is volatile and can look dramatic while representing little (100% growth from 2
to 4 sales); a **short time span** can show a "trend" that is really noise (the
sufficiency lessons); and **seasonality** can masquerade as trend if not accounted
for (December sales are not a growth trend). The numbers compute regardless — the
judgement is whether the trend is *real*: enough data, an appropriate base, and
seasonality considered. Compute the trend, then ask whether it is signal or
artefact. The next lessons add conditional aggregation to the calculation toolkit.
"""

CONTENT["Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets"] = r"""
Aggregating with a condition
------------------------------

Plain ``SUM`` and ``COUNT`` aggregate *everything*; analysis usually needs to
aggregate only the rows meeting a *condition* — sales in one region, orders above a
threshold, customers of one type. **COUNTIF** and **SUMIF** are the spreadsheet's
conditional-aggregation functions, and they are among the most-used analytical
tools, the spreadsheet counterparts of SQL's ``COUNT``/``SUM`` with ``WHERE``.

COUNTIF and SUMIF
-------------------

- ``COUNTIF(range, condition)`` — counts the cells in a range that meet a
  condition:

  .. code-block:: text

     =COUNTIF(region, "North")        how many northern orders
     =COUNTIF(amount, ">100")         how many orders over 100

- ``SUMIF(range, condition, sum_range)`` — sums the values in ``sum_range`` for the
  rows where ``range`` meets the condition:

  .. code-block:: text

     =SUMIF(region, "North", amount)  total revenue from northern orders
     =SUMIF(amount, ">100", amount)   total of all orders over 100

``COUNTIF`` answers "how many meeting X"; ``SUMIF`` answers "the total of Y for rows
meeting X." Together they compute the conditional counts and totals that most
analytical questions reduce to.

Multiple conditions: COUNTIFS and SUMIFS
------------------------------------------

For *several* conditions at once, the plural forms ``COUNTIFS`` and ``SUMIFS`` take
multiple range-condition pairs:

.. code-block:: text

   =COUNTIFS(region, "North", amount, ">100")
   =SUMIFS(amount, region, "North", month, "January")

``COUNTIFS`` counts rows meeting *all* the conditions (northern *and* over 100);
``SUMIFS`` sums for rows meeting all conditions. These handle the segmented
questions — "January revenue in the northern region" — that a single condition
cannot express, and they are the workhorses of spreadsheet analysis.

Why conditional aggregation matters
-------------------------------------

Most analytical questions are conditional: not "total sales" but "sales *in this
segment*", not "how many orders" but "how many *of this type*". ``COUNTIF`` and
``SUMIF`` (and their plural forms) are how a spreadsheet answers these directly,
without first filtering the data by hand. They are also the conceptual bridge to
SQL's ``WHERE`` plus aggregate and to pivot tables — the same "aggregate a subset"
idea in three forms, which is why recognising the pattern here pays off repeatedly.

The caveat
------------

Conditional-aggregation functions are precise about their conditions, and small
mistakes mislead: a condition written as text must match exactly (``"North"`` will
not catch ``"north"`` or ``"North "`` with a space — the cleaning issues resurface),
and the condition syntax for comparisons (``">100"`` in quotes) trips up beginners.
The ranges must also align — ``SUMIF``'s condition range and sum range must be the
same size and correspond row-for-row, or the result is silently wrong. As always,
verify a conditional total against a hand-check or an order-of-magnitude estimate:
a ``SUMIF`` that returns an implausible number usually has a condition or range
error. The next lesson covers a more advanced calculation function: SUMPRODUCT.
"""

CONTENT["Using SUMPRODUCT for Advanced Spreadsheet Calculations"] = r"""
Multiplying and summing at once
---------------------------------

Some calculations require *multiplying* corresponding values and then *summing* the
products — a weighted average, a total from quantities and prices, a count across
multiple conditions. **SUMPRODUCT** does exactly this in one formula: it multiplies
corresponding elements of arrays and sums the results. It is a more advanced
spreadsheet tool, and a versatile one, extending the calculation toolkit beyond the
conditional aggregates.

How SUMPRODUCT works
----------------------

``SUMPRODUCT`` takes arrays (ranges) of equal size, multiplies them element by
element, and sums the products:

.. code-block:: text

   =SUMPRODUCT(quantity, price)

If ``quantity`` is {2, 5, 3} and ``price`` is {10, 4, 20}, this computes
(2×10) + (5×4) + (3×20) = 20 + 20 + 60 = 100 — the total revenue, in one formula,
without a helper column of row-by-row products. That is the core use: a sum of
products across two aligned columns.

Weighted averages and beyond
------------------------------

``SUMPRODUCT`` shines for weighted calculations. A weighted average — where each
value counts according to a weight — is a ``SUMPRODUCT`` divided by the sum of
weights:

.. code-block:: text

   = SUMPRODUCT(scores, weights) / SUM(weights)

This computes the average score weighted by importance, a calculation that is
awkward without ``SUMPRODUCT``. The function generalises the plain ``SUM`` of a
single column to a sum of *combinations* of columns.

Multi-condition counting with SUMPRODUCT
------------------------------------------

A powerful advanced use: ``SUMPRODUCT`` can count or sum across *multiple
conditions* by multiplying arrays of TRUE/FALSE tests (which evaluate as 1/0):

.. code-block:: text

   =SUMPRODUCT((region="North")*(amount>100))

Each condition produces an array of 1s and 0s; multiplying them gives 1 only where
*both* hold; summing counts the rows meeting both — a flexible alternative to
``COUNTIFS`` that can express conditions the plural functions cannot. This
array-logic capability is why ``SUMPRODUCT`` is a favourite of advanced spreadsheet
users.

The caveat
------------

``SUMPRODUCT``'s power comes with complexity and pitfalls. The arrays **must be the
same size** — mismatched ranges produce an error or, worse, a wrong result; and the
array-condition syntax (``(region="North")*(amount>100)``) is unintuitive until
learned, making the formulas hard for others to read. ``SUMPRODUCT`` can also be
slower on large ranges than purpose-built functions. Because it is powerful but
opaque, use it where its multiply-then-sum or multi-condition logic is genuinely
needed — a weighted average, a condition ``COUNTIFS`` cannot express — rather than
where a clearer ``SUMIF`` or ``COUNTIFS`` would do, and comment what a
``SUMPRODUCT`` formula computes. Clarity over cleverness applies. The next lesson
turns to the most powerful summarising tool of all: the pivot table.
"""

CONTENT["Using Pivot Tables for Calculations and Trend Analysis"] = r"""
The spreadsheet's most powerful tool
--------------------------------------

For summarising and analysing data, the **pivot table** is the single most powerful
feature a spreadsheet offers. It groups data by categories and computes aggregates
for each group — the same work as SQL's ``GROUP BY``, done with drag-and-drop —
turning thousands of rows into a compact summary in seconds. Mastering pivot tables
is one of the highest-leverage spreadsheet skills, and it brings the calculation
stage to its centre.

What a pivot table does
-------------------------

A pivot table takes a tabular dataset and lets you summarise it along dimensions
you choose:

- **Rows and columns** — the categories to group by (region as rows, month as
  columns).
- **Values** — the aggregate to compute for each group (sum of sales, count of
  orders, average amount).
- **Filters** — restricting which data the summary includes (the next lesson).

Drag ``region`` to rows, ``month`` to columns, and ``sum of sales`` to values, and
the pivot instantly produces a grid of sales by region and month — a summary that
would take many formulas to build by hand, recomputed automatically as the data
changes.

Pivot tables for trend analysis
---------------------------------

Pivot tables excel at the trend analysis the stage opened with. Grouping sales by
time period (month, quarter, year) as rows or columns produces a trend summary
directly — revenue per month, orders per quarter — and the pivot can compute the
aggregates and even percentage-of-total or running calculations. Where the
trend-formula lesson computed change period by period with formulas, a pivot table
produces the whole periodic summary at once, making it the fastest route from raw
transactions to a trend view.

Why pivot tables are so valuable
----------------------------------

The pivot table's value is *speed and flexibility*: it answers a whole class of
"summarise X by Y" questions with a few drags, and re-answers them instantly when
you change the grouping (swap region for product, month for quarter). This lets an
analyst *explore* — trying different summaries rapidly to find where the insight is
— in a way that hand-built formulas cannot match. It is the spreadsheet embodiment
of the ``GROUP BY`` aggregation that the SQL lessons covered, made interactive, and
it is why pivot tables appear in nearly every serious spreadsheet analysis.

The caveat
------------

Pivot tables are powerful but assume and demand discipline. They require **clean,
tidy source data** — the one-row-per-record, one-column-per-variable structure from
the prep section — and they *fragment* on dirty data exactly as ``GROUP BY`` does:
"NY" and "New York" appear as separate row groups, splitting what should be one.
Pivots also **do not refresh automatically** in some tools when the source changes
(you must refresh them), so a pivot can silently show stale results; and their ease
can encourage summarising without understanding, producing tidy tables of
misinterpreted numbers. Build pivots on clean data, refresh them, and interpret
them with the same care as any calculation. The next lesson extends pivot tables
with filters and calculated fields for deeper analysis.
"""


MINDMAP.update({
    "Using Spreadsheet Formulas for Sales Trend Analysis": [
        "Spreadsheet Calculations with Formulas",
        "Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets",
        "Using Pivot Tables for Calculations and Trend Analysis",
        "Mathematical Thinking",
    ],
    "Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets": [
        "Using Spreadsheet Formulas for Sales Trend Analysis",
        "Using SUMPRODUCT for Advanced Spreadsheet Calculations",
        "Spreadsheet Functions",
        "Aggregating Data with Subqueries, HAVING, and CASE in SQL",
    ],
    "Using SUMPRODUCT for Advanced Spreadsheet Calculations": [
        "Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets",
        "Using Pivot Tables for Calculations and Trend Analysis",
        "Spreadsheet Calculations with Formulas",
        "Spreadsheet Functions",
    ],
    "Using Pivot Tables for Calculations and Trend Analysis": [
        "Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets",
        "Using Pivot Table Filters and Calculated Fields for Deeper Analysis",
        "Comparing Calculations in Spreadsheets and SQL",
        "How Data Analysts Use Spreadsheets",
    ],
})


# ======================================================================
# Section 5 — Analyze Data / calc (close)  (analyze 024-027)
# ======================================================================

GLOSS.update({
    "Using Pivot Table Filters and Calculated Fields for Deeper Analysis":
        "filtering a pivot and adding computed fields — pushing pivots past basic summaries",
    "Comparing Calculations in Spreadsheets and SQL":
        "the same aggregate two ways — when each tool suits the calculation",
    "Embedding Calculations in SQL Queries":
        "computing derived values inside a query — arithmetic and expressions in SELECT",
    "Using GROUP BY and ORDER BY for Aggregated Calculations in SQL":
        "the SQL pivot: grouping to aggregate, ordering the summary",
})

CONTENT["Using Pivot Table Filters and Calculated Fields for Deeper Analysis"] = r"""
Beyond the basic pivot
------------------------

A basic pivot table summarises; two features push it toward *deeper* analysis:
**filters** (restricting what the pivot includes) and **calculated fields** (adding
computed values the source data does not contain). Together they turn the pivot
from a summarising tool into a flexible analytical instrument, extending the
previous lesson's foundation.

Pivot table filters
----------------------

Filters restrict which data a pivot summarises, letting you focus the summary
without changing the underlying data:

- **Report filters** — restrict the whole pivot to a subset (show the summary for
  one region, one year, one product line), with a control to switch the subset.
- **Slicers** — visual filter buttons that make filtering interactive and obvious,
  and can filter multiple pivots at once.
- **Row/column filters** — hide specific categories within the pivot.

Filtering a pivot answers "what does this summary look like *for this segment*" —
and switching the filter re-answers it instantly for another segment, enabling
rapid comparison across subsets.

Calculated fields
-------------------

A **calculated field** adds a new value to the pivot computed from existing fields
— a metric the source data does not store directly:

- **Ratios and rates** — profit margin from revenue and cost, conversion rate from
  conversions and visits.
- **Derived amounts** — a computed total, a per-unit value, a difference.
- **Custom metrics** — any formula combining the pivot's aggregated values.

For example, a pivot summing revenue and cost by region can gain a calculated field
``profit = revenue - cost``, or ``margin = (revenue - cost) / revenue`` — computing
the derived metric *per group* automatically. This lets the pivot present not just
raw aggregates but the *analytical* values a question actually needs.

Why these deepen analysis
---------------------------

Basic pivots answer "sum X by Y"; filters and calculated fields answer "the derived
metric M, for segment S, by Y" — a substantially richer question. They let an
analyst move from *reporting* aggregates to *analysing* them: computing the rates
and ratios that reveal what raw totals hide, and slicing to the segments where the
story lives. This is the pivot table doing genuine analytical work, not just
tabulation.

The caveat
------------

The added power adds ways to mislead. A **filter left active** shows a partial
summary that is easily mistaken for the whole — the same forgotten-filter risk as
elsewhere, now hiding in a pivot that looks complete. **Calculated fields on
aggregates** can compute subtly wrong values: a "margin" calculated as the average
of per-row margins differs from the margin of the summed totals, and the pivot's
field computes one when you may mean the other. And filters plus calculated fields
make a pivot complex enough that its logic is no longer obvious at a glance. Verify
what a filtered, calculated pivot actually computes — especially that ratios are
computed on the totals you intend — before trusting it. The next lesson steps back
to compare spreadsheet and SQL calculation directly.
"""

CONTENT["Comparing Calculations in Spreadsheets and SQL"] = r"""
The same calculation, two tools
---------------------------------

Having computed aggregates in both spreadsheets and SQL, it is worth comparing them
directly — because the *same* calculation is often expressible in either, and
knowing how they correspond (and where each suits) makes you fluent across both.
This lesson maps the correspondence, consolidating the calculation stage.

The direct correspondences
----------------------------

Most calculations translate cleanly between the two:

- **Aggregate a column** — spreadsheet ``=SUM(amount)`` ↔ SQL
  ``SELECT SUM(amount) FROM orders``.
- **Conditional aggregate** — spreadsheet ``=SUMIF(region,"North",amount)`` ↔ SQL
  ``SELECT SUM(amount) FROM orders WHERE region='North'``.
- **Group and aggregate** — spreadsheet *pivot table* (region → rows, sum of
  amount → values) ↔ SQL ``SELECT region, SUM(amount) FROM orders GROUP BY
  region``.
- **Filter then aggregate** — spreadsheet filter + ``SUM`` ↔ SQL ``WHERE`` +
  ``SUM``.

The pattern is consistent: the spreadsheet's ``SUMIF``/``COUNTIF`` and pivot tables
are the *same operations* as SQL's ``WHERE`` plus aggregates and ``GROUP BY``. The
concepts are identical; only the expression differs — a formula and a menu in one,
a query in the other.

Where each suits the calculation
----------------------------------

The correspondence does not make them interchangeable in practice:

- **Spreadsheets suit** small data, interactive exploration, calculations you want
  to *see* and adjust cell by cell, and results a stakeholder will open. A quick
  pivot to explore is often faster than writing a query.
- **SQL suits** large data, calculations that must be *reproducible* and rerun on
  new data, complex multi-table aggregation, and computation at the source. A
  calculation over millions of rows, or one that runs every week, belongs in SQL.

The deciding factors are the familiar ones — size, repetition, complexity, audience
— applied to the specific calculation.

Why the comparison matters
----------------------------

Seeing the calculations as *the same operations in different tools* is what makes an
analyst tool-fluent rather than tool-bound. It means you can prototype a calculation
in a spreadsheet where it is quick to see, then translate it to SQL when it needs to
scale or recur — and recognise that a ``GROUP BY`` query and a pivot table are the
same idea, so learning one deepens the other. The tools are different expressions of
one analytical vocabulary.

The caveat
------------

The correspondence is close but not perfect, and the gaps cause errors. The tools
can **handle edge cases differently** — nulls, blanks, text-versus-number, rounding
— so the "same" calculation can give subtly different results in each (a spreadsheet
average that skips blank cells versus a SQL ``AVG`` that ignores nulls may or may not
match, depending on the data). Translating a calculation between tools therefore
requires *verifying the results match*, not assuming they do. Use the comparison to
move fluently between tools, but check that a translated calculation reproduces the
original, especially around missing values. The next lessons go deeper into SQL
calculation.
"""

CONTENT["Embedding Calculations in SQL Queries"] = r"""
Computing inside the query
----------------------------

SQL does not only *retrieve* data — it *computes* on it, producing derived values as
part of the query result. **Embedding calculations in SQL queries** means writing
arithmetic and expressions directly in the ``SELECT`` clause, so the database
returns computed columns alongside stored ones. This is where SQL becomes a
calculation engine, not just a retrieval tool.

Calculated columns in SELECT
------------------------------

Any expression in the ``SELECT`` clause produces a computed column:

.. code-block:: sql

   SELECT product,
          quantity,
          price,
          quantity * price               AS line_total,
          price * 0.9                    AS discounted_price,
          (revenue - cost) / revenue     AS margin
   FROM   orders;

Each expression — ``quantity * price``, ``price * 0.9``, the margin formula — is
computed per row and returned as a new column (named with ``AS``). The arithmetic
operators (``+``, ``-``, ``*``, ``/``) work as expected, and expressions can combine
columns, constants, and functions. This is the SQL equivalent of a spreadsheet
formula column, computed by the database.

Calculations with functions
------------------------------

Embedded calculations extend beyond arithmetic to SQL's functions:

- **String calculations** — ``CONCAT``, ``SUBSTR``, and the string functions from
  earlier, producing derived text.
- **Date calculations** — extracting parts of dates, computing differences between
  dates (days between order and ship), which are central to trend and duration
  analysis.
- **Conditional calculations** — ``CASE`` expressions producing values that depend
  on conditions (the conditional logic from the combine stage).
- **Type conversions** — ``CAST`` within a calculation, ensuring arithmetic happens
  on the right types.

Combining these, a single ``SELECT`` can compute a rich set of derived values from
the stored data — the analytical columns a question needs, produced at query time.

Why compute in the query
--------------------------

Embedding calculations in the query, rather than retrieving raw data and computing
elsewhere, has real advantages: the computation happens *at the source* on the full
data (no exporting), the query is *reproducible* (the calculation is recorded in the
query text and reruns on new data), and only the *results* travel, not millions of
raw rows. It keeps the calculation logic with the data and the query, which is
cleaner and more scalable than pulling data out to compute on it.

The caveat
------------

Embedded calculations inherit SQL's precision demands. **Integer division** is a
classic trap — in many databases, ``5 / 2`` yields ``2`` (integer division) rather
than ``2.5``, so a ratio computed on integer columns can be silently wrong unless
you cast to a decimal type first. **Nulls propagate** — any arithmetic involving a
null yields null (``price * quantity`` is null if either is null), so calculated
columns can be unexpectedly empty. And **division by zero** errors or yields null
depending on the database. As always, the calculation returns exactly what the
expression specifies — verify that derived columns hold the values you intend,
checking especially division, nulls, and types. The next lesson combines embedded
calculation with grouping — the SQL pivot.
"""

CONTENT["Using GROUP BY and ORDER BY for Aggregated Calculations in SQL"] = r"""
The SQL pivot, complete
-------------------------

The calculation stage culminates where SQL aggregation began: **GROUP BY** to
compute aggregates per group, and **ORDER BY** to order the summary. Together they
are the SQL equivalent of the pivot table — grouping data into categories,
computing an aggregate for each, and presenting the result in a meaningful order.
This lesson assembles them into the complete aggregated-calculation pattern, closing
the calculation stage.

The core aggregated-calculation query
---------------------------------------

The pattern combines grouping, aggregating, and ordering:

.. code-block:: sql

   SELECT   region,
            COUNT(*)      AS orders,
            SUM(amount)   AS revenue,
            AVG(amount)   AS avg_order
   FROM     orders
   GROUP BY region
   ORDER BY revenue DESC;

This groups orders by region, computes each region's order count, total revenue,
and average order value, and orders the result by revenue, highest first — a
complete regional summary in one query. It is exactly a pivot table (region as the
grouping, the aggregates as values, sorted) expressed as SQL, and it answers the
"summarise X by Y, ranked" question directly.

Ordering the summary
----------------------

``ORDER BY`` on an aggregated query orders the *groups*, and can order by an
aggregate:

- ``ORDER BY revenue DESC`` — regions from highest revenue to lowest, surfacing the
  top performers.
- ``ORDER BY COUNT(*) DESC`` — groups by how many rows each contains.
- ``ORDER BY region ASC`` — the groups in category order.

Ordering by an aggregate is what turns a summary into a *ranking* — "regions by
revenue", "products by units sold" — one of the most common analytical outputs.

The full analytical query
---------------------------

Combined with the earlier clauses, the complete pattern layers filtering,
grouping, group-filtering, and ordering:

.. code-block:: sql

   SELECT   region, SUM(amount) AS revenue
   FROM     orders
   WHERE    order_date >= '2024-01-01'    -- filter rows first
   GROUP BY region                        -- group
   HAVING   SUM(amount) > 10000           -- filter groups
   ORDER BY revenue DESC;                 -- order the result

This reads as a complete analytical sentence: *from* the orders, *where* they are
recent, *grouped by* region, *keeping* high-revenue regions, *ordered by* revenue.
The clause order (``WHERE`` → ``GROUP BY`` → ``HAVING`` → ``ORDER BY``) is both the
required SQL syntax and the logical sequence of the analysis, and mastering it is
mastering SQL aggregation.

The caveat
------------

The full pattern concentrates the section's precision traps in one place: the
``WHERE``/``HAVING`` distinction (rows before grouping, aggregates after), nulls
interacting with aggregates (``COUNT(*)`` versus ``COUNT(column)``, ``AVG`` ignoring
nulls), and the requirement that every non-aggregated column in ``SELECT`` appear in
``GROUP BY`` (or the query errors or, in some databases, returns arbitrary values).
A grouped-calculation query that looks right can be subtly wrong, so the
build-incrementally-and-verify discipline is essential — get the grouping right,
add each clause, check the result against expectation. This closes the calculation
stage; the final stage of the section covers advanced analytical techniques,
including temporary tables.
"""


MINDMAP.update({
    "Using Pivot Table Filters and Calculated Fields for Deeper Analysis": [
        "Using Pivot Tables for Calculations and Trend Analysis",
        "Comparing Calculations in Spreadsheets and SQL",
        "Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets",
        "Using SUMPRODUCT for Advanced Spreadsheet Calculations",
    ],
    "Comparing Calculations in Spreadsheets and SQL": [
        "Using Pivot Tables for Calculations and Trend Analysis",
        "Embedding Calculations in SQL Queries",
        "Spreadsheets vs. SQL",
        "Using GROUP BY and ORDER BY for Aggregated Calculations in SQL",
    ],
    "Embedding Calculations in SQL Queries": [
        "Comparing Calculations in Spreadsheets and SQL",
        "Using GROUP BY and ORDER BY for Aggregated Calculations in SQL",
        "Core SQL Queries for Data Cleaning and Analysis",
        "Aggregating Data with Subqueries, HAVING, and CASE in SQL",
    ],
    "Using GROUP BY and ORDER BY for Aggregated Calculations in SQL": [
        "Embedding Calculations in SQL Queries",
        "Aggregating Data with Subqueries, HAVING, and CASE in SQL",
        "Sorting and Filtering Data in SQL Using ORDER BY and WHERE",
        "Comparing Calculations in Spreadsheets and SQL",
    ],
})


# ======================================================================
# Section 5 — Analyze Data / advanced (close)  (analyze 028-030)  -- SECTION 5 COMPLETE
# NOTE: analyze 030 title contains an em-dash (U+2014) — key written literally
# ======================================================================

GLOSS.update({
    "Data Validation as an Ongoing Analytical Process":
        "validation not as a one-time gate but a continuous check throughout analysis",
    "Temporary Tables and the WITH Clause in SQL":
        "holding intermediate results — CTEs and temp tables that structure complex queries",
    "Creating Temporary Tables in SQL \u2014 Methods, Trade-offs, and Best Practices":
        "the ways to make a temp table, when each fits, and how to use them well",
})

CONTENT["Data Validation as an Ongoing Analytical Process"] = r"""
Validation never really ends
------------------------------

Data validation appeared earlier as a spreadsheet feature and a cleaning step, but
its deepest form is a *mindset*: **validation as an ongoing analytical process**,
running continuously throughout analysis rather than once at the start. Opening the
advanced stage, this lesson reframes validation from a gate you pass to a habit you
maintain — checking that data and results remain sound at every step.

Why validation must be continuous
-----------------------------------

Data can go wrong at *any* point in analysis, not just at entry. A join can
introduce duplicates; a calculation can produce impossible values; a filter can
silently drop rows; a transformation can corrupt a field. Validating once at the
beginning cannot catch problems that arise *during* the work — so validation must
happen throughout: after each significant operation, confirm the data still makes
sense. This is the integrity-in-motion principle from the cleaning section, elevated
into a continuous analytical discipline.

What ongoing validation looks like
------------------------------------

Validation as a process weaves checks through the whole analysis:

- **After each transformation** — did the row count change as expected? Do the
  values still fall in valid ranges? (The verification techniques from the cleaning
  section, applied continuously.)
- **Against expectations** — does each result pass the order-of-magnitude sanity
  check? A number wildly off from what you expected signals a problem to
  investigate, not a finding to report.
- **Cross-checks** — does the total from one method match the total from another?
  Do the parts sum to the whole? Independent computations that should agree are a
  powerful validation.
- **Against reality** — does the result make sense in the real world? A negative
  count, a percentage over 100, a customer older than 150 — impossibilities that
  signal an error somewhere upstream.

The point is that validation is not a phase but a *reflex*, applied at every step.

Validation and trustworthy analysis
--------------------------------------

Continuous validation is what makes analysis *trustworthy* rather than merely
completed. Analysis that runs end to end without validation produces a result, but
no assurance the result is correct — and dirty data, bad joins, and calculation
errors all produce plausible-looking wrong answers that only validation catches.
The analysts whose work can be relied on are those who validate continuously, so
that by the time they present a finding, it has survived checks at every step. This
is the whole course's check-your-results habit, made into a systematic practice.

The caveat
------------

Continuous validation is essential but can tip into paralysis — validating so
obsessively that the analysis never progresses, or treating every minor anomaly as
a crisis. The judgement is *proportionate* validation: check the things most likely
to be wrong and most consequential if they are, at the points where errors most
often enter (transformations, joins, calculations), without re-checking everything
constantly. And validation confirms data is *plausible and consistent*, not
necessarily *correct* — it catches errors, but passing validation is not proof of
truth. Validate enough to trust the work, proportionate to its stakes. The next
lessons turn to a technique that makes complex, validated analysis manageable:
temporary tables.
"""

CONTENT["Temporary Tables and the WITH Clause in SQL"] = r"""
Holding intermediate results
------------------------------

Complex analysis often needs to compute an intermediate result and then work with
it — and writing everything as one deeply nested query becomes unreadable. SQL
offers ways to hold intermediate results: **temporary tables** and, most usefully,
the **WITH clause** (common table expressions). They let you break a complex
analysis into named, readable steps, which is essential for the advanced queries
this stage covers.

The WITH clause (common table expressions)
--------------------------------------------

The ``WITH`` clause defines a named, temporary result set that exists for the
duration of a single query — a **common table expression (CTE)**:

.. code-block:: sql

   WITH regional_revenue AS (
       SELECT region, SUM(amount) AS revenue
       FROM   orders
       GROUP BY region
   )
   SELECT region, revenue
   FROM   regional_revenue
   WHERE  revenue > 10000
   ORDER BY revenue DESC;

The ``WITH`` block computes ``regional_revenue`` (revenue per region); the main
query then treats it as if it were a table — filtering and ordering it. This does
the same work as a subquery in the ``FROM`` clause, but *named and readable*: the
intermediate step has a name and stands on its own, rather than being buried inside
the outer query.

Why CTEs improve complex queries
----------------------------------

CTEs make complex analysis manageable in ways that matter:

- **Readability** — each step is named and separate, so a multi-step analysis reads
  top to bottom as a sequence rather than as nested parentheses read inside-out.
- **Reuse within the query** — a CTE can be referenced multiple times in the main
  query, computing it once.
- **Chaining** — multiple CTEs can build on each other (``WITH a AS (...), b AS
  (SELECT ... FROM a)``), expressing a genuine multi-step pipeline as a readable
  sequence.
- **Debugging** — each CTE can be tested on its own by selecting from it, isolating
  where a problem is (the isolate step of the problem-solving framework).

For analytical queries of any complexity, CTEs are the standard way to keep them
comprehensible.

Temporary tables
------------------

A **temporary table** is a table that exists only for a session, holding
intermediate results that persist across *multiple* queries (unlike a CTE, which
lives only within one query). When an analysis runs several queries against the same
intermediate result, computing it once into a temporary table and querying that
repeatedly is more efficient than recomputing it each time. The next lesson covers
creating them in detail.

The caveat
------------

CTEs and temporary tables serve overlapping but distinct needs, and choosing wrongly
costs clarity or performance. A **CTE** lives for one query and is ideal for
structuring a single complex query readably; a **temporary table** persists across a
session and suits reusing an intermediate result across several queries. Overusing
either — wrapping trivial queries in CTEs, or creating temp tables for one-time use —
adds complexity without benefit. And in some databases a CTE is recomputed each time
it is referenced, which can be slow if referenced repeatedly on large data (where a
temp table would be better). Match the tool to whether the intermediate result is
used once or many times. The next lesson details creating temporary tables.
"""

CONTENT["Creating Temporary Tables in SQL \u2014 Methods, Trade-offs, and Best Practices"] = r"""
Making a temporary table
--------------------------

Temporary tables hold intermediate results across multiple queries in a session, and
there are several ways to create them — each with trade-offs. This lesson, closing
the analysis section, covers the methods, when each fits, and the practices that keep
temporary tables an asset rather than a source of confusion.

The methods
------------

SQL offers a few ways to create a temporary result:

- **CREATE TEMPORARY TABLE** — explicitly creates a temp table that persists for the
  session, then populate it with an ``INSERT`` or create it from a query:

  .. code-block:: sql

     CREATE TEMPORARY TABLE regional_revenue AS
     SELECT region, SUM(amount) AS revenue
     FROM   orders
     GROUP BY region;

  The table now exists for the session and can be queried repeatedly like any table.
- **The WITH clause (CTE)** — the previous lesson's named result set, for
  intermediate results needed within a *single* query rather than across many.
- **Creating a regular table** as a staging area — a permanent table used
  temporarily (less clean; requires explicit cleanup).
- **Views** — a saved query that behaves like a table but recomputes each time it is
  queried (not truly a stored intermediate result, but related).

The choice among them depends on scope (one query or many), persistence, and how
often the intermediate result is reused.

The trade-offs
----------------

Each method trades off differently:

- **CTEs** — cleanest and most readable for single-query steps; but scoped to one
  query, and possibly recomputed on each reference.
- **Temporary tables** — computed once and reused across a session, efficient when an
  intermediate result feeds several queries; but require creation and consume session
  resources, and add steps to the workflow.
- **Staging in permanent tables** — flexible but risky: they persist beyond the
  session, must be cleaned up, and can clutter or confuse if forgotten.

The core trade-off is *readability and simplicity* (favouring CTEs) versus *reuse and
performance* (favouring temp tables when an intermediate is queried repeatedly).

Best practices
---------------

A few practices keep temporary tables beneficial:

- **Prefer CTEs for single queries** — reach for a temp table only when an
  intermediate result is genuinely reused across multiple queries.
- **Name clearly** — a temp table's name should say what it holds, as with any good
  naming.
- **Clean up** — drop temporary tables when done (session-scoped ones clean up
  automatically, but explicit cleanup avoids surprises), and never leave staging
  tables lingering.
- **Document the pipeline** — when an analysis builds several temp tables in
  sequence, document the steps so the pipeline is reproducible and understandable
  (the documentation discipline from cleaning).

The caveat
------------

Temporary tables introduce *state* into an analysis — intermediate results that exist
outside the queries and must be tracked — which is exactly what makes them both useful
and hazardous. A temp table built early and queried later can hold *stale* data if the
source changed in between; a forgotten staging table can be reused by mistake; and a
multi-temp-table pipeline is harder to reproduce than a single query. The discipline
is to use temporary tables when reuse genuinely justifies them, keep the pipeline
documented and clean, and prefer the self-contained CTE where it suffices. Managed
well, they make complex analysis efficient; managed carelessly, they become a source
of exactly the integrity problems this course has warned against.

This completes the Analyze Data section. You have moved from what analysis is, through
organising, combining, and calculating data in both spreadsheets and SQL, to the
advanced techniques that structure complex analytical work. The data is now not only
prepared and clean but *analysed* — turned into findings. The next section addresses
the final step of making those findings land: visualising and communicating them.
"""


MINDMAP.update({
    "Data Validation as an Ongoing Analytical Process": [
        "Data Validation in Spreadsheets",
        "Combining Data Validation and Conditional Formatting in Spreadsheets",
        "Verifying and Reporting Data Integrity",
        "Temporary Tables and the WITH Clause in SQL",
    ],
    "Temporary Tables and the WITH Clause in SQL": [
        "Creating Temporary Tables in SQL \u2014 Methods, Trade-offs, and Best Practices",
        "Subqueries in SQL",
        "Aggregating Data with Subqueries, HAVING, and CASE in SQL",
        "Using GROUP BY and ORDER BY for Aggregated Calculations in SQL",
    ],
    "Creating Temporary Tables in SQL \u2014 Methods, Trade-offs, and Best Practices": [
        "Temporary Tables and the WITH Clause in SQL",
        "Subqueries in SQL",
        "Core SQL Queries for Data Cleaning and Analysis",
        "The Role and Importance of Data Visualization",
    ],
})


# ======================================================================
# Section 6 — Data Visualization / Stage: principles  (viz 001-004)
# ======================================================================

GLOSS.update({
    "Data Visualization":
        "representing data graphically so patterns and meaning become visible at a glance",
    "Connecting Data and Images":
        "how a visualization maps data values to visual properties — the encoding that carries meaning",
    "Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose":
        "what makes a visualization effective — a clear focus, sound structure, and a purpose",
    "Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity":
        "fixed images versus interactive views — when each serves, and their trade-offs",
})

CONTENT["Data Visualization"] = r"""
Seeing the data
-----------------

Analysis produces findings; **visualization** is how those findings are made
*visible* — and often how they are found in the first place. Data visualization is
the graphical representation of data: turning rows and numbers into charts, graphs,
and images that reveal patterns, trends, and relationships the raw data conceals.
Opening the visualization section, this lesson establishes what visualization is and
why it is indispensable to both analysing and communicating data.

Why visualize
--------------

Human beings read pictures far faster than tables of numbers. A column of a thousand
sales figures is opaque; the same data as a line chart shows the trend instantly. A
list of regional totals hides which is largest; a bar chart makes the ranking
obvious at a glance. Visualization works because it enlists the eye's pattern-finding
ability — we see a spike, a trend, a cluster, an outlier immediately, where reading
the underlying numbers would take minutes and might miss it entirely.

The two purposes of visualization
-----------------------------------

Visualization serves two distinct roles, and confusing them causes trouble:

- **Exploratory** — visualizing data to *find* patterns during analysis. Quick,
  rough charts for your own eyes, made to discover what the data holds (the
  viewing-data-differently lenses from cleaning were a form of this). Correctness
  and speed matter; polish does not.
- **Explanatory** — visualizing a finding to *communicate* it to others. Polished,
  focused charts made to convey a specific insight clearly to an audience. Here
  clarity, honesty, and design matter greatly (the storytelling lessons ahead).

The same chart type can serve either, but the *purpose* shapes how it is made — a
scratch chart for exploration and a presentation chart for communication are built to
different standards.

Visualization in the process
------------------------------

Visualization spans the analysis and share phases. During *analysis*, it is a tool
for seeing what the data holds — the fastest way to spot a trend, an outlier, a
relationship. During *sharing*, it is the primary vehicle for communicating findings,
because a well-chosen chart conveys an insight faster and more memorably than any
paragraph — the "data creates value only when communicated" principle, realised
visually. This section builds the skill of making visualizations that serve both.

The caveat
------------

Visualization's power to reveal is matched by its power to *mislead* — the same chart
that makes a real pattern obvious can make a spurious one look compelling, and design
choices (a truncated axis, a misleading scale, a cherry-picked range) can distort
what the data actually says, whether by accident or design. A visualization is an
*argument* about the data, not neutral truth, and the sections ahead stress honest
visualization as much as effective visualization. Seeing is believing — which is
exactly why a misleading chart is so dangerous. The next lesson examines how data
becomes an image at all.
"""

CONTENT["Connecting Data and Images"] = r"""
From numbers to visual properties
-----------------------------------

A visualization works by *mapping* data to visual properties — turning a number into
a bar's height, a category into a colour, a pair of values into a point's position.
Understanding this mapping, the connection between **data and images**, is what lets
you read visualizations critically and build them deliberately, rather than treating
charts as magic.

Visual encodings
-----------------

Every visualization *encodes* data values as visual attributes. The common encodings:

- **Position** — where a mark sits (a point's place on an axis, a dot in a scatter
  plot). Position is the most precisely readable encoding — the eye judges position
  extremely well, which is why scatter plots and dot plots are so effective.
- **Length** — how long a mark is (a bar's height or length). Also read accurately,
  which is why bar charts are excellent for comparing quantities.
- **Angle and area** — as in a pie chart's slices. Read *less* accurately by the eye
  — people struggle to compare angles and areas precisely, which is a known weakness
  of pie charts.
- **Colour** — hue (different categories) or intensity (a value's magnitude, as in a
  heat map). Effective for categories and for showing a third dimension, but read
  less precisely for exact quantities.
- **Shape and size** — distinguishing categories or encoding a value in a mark's
  size.

The choice of encoding determines how *accurately and easily* the data can be read —
some encodings the eye judges precisely, others only roughly.

Why the encoding matters
--------------------------

Because encodings differ in how accurately they are read, the *same data* is more or
less legible depending on how it is encoded. A quantity comparison is clear as bar
lengths (accurately read) but muddy as pie slices (poorly read); a trend is clear as
position over time (a line) but lost in a table. Choosing an encoding the eye reads
well for the message you want is the foundation of effective visualization — a chart
that encodes its key comparison in a hard-to-read attribute fights its own reader.

Reading visualizations critically
------------------------------------

Understanding encodings also makes you a *critical* reader of charts. Knowing that
the eye reads position and length accurately but angle and area poorly, you can spot
when a chart's design obscures rather than reveals — a pie chart where a bar chart
would be clearer, a 3D effect that distorts areas, an encoding chosen for
decoration over legibility. The connection between data and image is where both good
design and misleading design live.

The caveat
------------

The mapping from data to image is where *distortion* enters, deliberately or
carelessly. The same encoding can mislead if its scale is manipulated (a bar chart
with a truncated axis exaggerates differences), if area is used to encode a value
people will read as one-dimensional (doubling a circle's radius quadruples its area,
overstating the value fourfold), or if colour implies an order that the data does not
have. Every visualization is a set of encoding *choices*, and those choices can
serve clarity or distort it — reading and making charts both require attention to
whether the encoding represents the data faithfully. The next lesson turns to what
makes a visualization genuinely powerful.
"""

CONTENT["Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose"] = r"""
What makes a visualization powerful
-------------------------------------

Not all visualizations are equal — some convey their insight instantly and
memorably, others confuse or mislead. What separates a *powerful* visualization from a
weak one comes down to three qualities: a clear **focus**, sound **structure**, and a
genuine **analytical purpose**. This lesson names them, giving a framework for judging
and building effective charts.

Focus: one clear message
--------------------------

A powerful visualization has a **single, clear focus** — one main point it exists to
make. The most common failure of weak visualizations is trying to show *everything*,
burying the key insight under competing elements until nothing stands out. Focus
means deciding what the *one* thing the viewer should take away is, and designing the
chart so that thing is unmistakable — the important element emphasised, distractions
removed. A chart that answers "what should I notice?" with a clear "this" is
powerful; one that answers "everything" is not.

Structure: sound and honest form
----------------------------------

**Structure** is the chart's underlying form — the right chart type for the data and
message, appropriate axes and scales, logical organisation. Sound structure means the
visualization's form *fits* what it shows: a trend as a line, a comparison as bars, a
part-to-whole as a stacked bar or (sparingly) a pie, a relationship as a scatter plot.
It also means *honest* structure — scales that do not distort, axes that start where
they should, encodings that represent the data faithfully. Poor structure — the wrong
chart type, a misleading scale — undermines even a well-focused chart.

Analytical purpose: it answers a question
-------------------------------------------

A powerful visualization has a **purpose** — it answers a real question or supports a
real decision, rather than existing for decoration. The purpose is what determines
the focus and structure: knowing *what question the chart answers* tells you what to
emphasise and how to build it. A chart made without a purpose — "here is some data
as a chart" — has no basis for its choices and usually shows everything and
emphasises nothing. Purpose first, then focus and structure follow.

The three together
--------------------

The qualities reinforce each other: the *purpose* (the question) determines the
*focus* (the one message that answers it), which the *structure* (the right form)
makes clear. A visualization with all three — a genuine purpose, a single clear
focus, and sound honest structure — conveys its insight instantly. Missing any one
weakens it: no purpose and it is aimless, no focus and it is cluttered, no sound
structure and it misleads.

The caveat
------------

The pursuit of a "powerful" visualization has a dark side: the same techniques that
make a chart focused and compelling can make a *misleading* chart compelling.
Emphasising one message means de-emphasising others, which can shade into hiding
inconvenient data; a strong focus can become a cherry-picked one. Powerful and
*honest* are not the same, and the goal is both — a visualization that makes a true
insight clear, not one that makes a preferred conclusion persuasive. The focus,
structure, and purpose must serve the data's actual message, which is the ethical
line the storytelling lessons return to. The next lesson examines a structural choice:
static versus dynamic visualizations.
"""

CONTENT["Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity"] = r"""
Fixed images and interactive views
------------------------------------

A fundamental choice in visualization is whether it is **static** — a fixed image
that shows one view — or **dynamic** — interactive, letting the viewer explore,
filter, and change what they see. Each suits different situations, and understanding
the trade-offs is part of designing visualizations that fit their purpose and
audience.

Static visualizations
------------------------

A **static** visualization is a fixed image: a chart in a report, a slide, a printed
page. It shows exactly one view, chosen by its maker.

- **Strengths** — *control and clarity*. The maker decides precisely what the viewer
  sees, so the message is focused and cannot be misread through interaction. Static
  charts are universally viewable (any document, any screen, print), simple, and
  ideal for a specific point in a report or presentation.
- **Weaknesses** — *no exploration*. The viewer sees only what the maker chose; they
  cannot dig into a detail, filter to their segment, or ask a follow-up the chart
  does not answer.

Static visualization suits *communicating a specific finding* to an audience — the
explanatory purpose — where focus and control matter more than exploration.

Dynamic visualizations
------------------------

A **dynamic** (interactive) visualization lets the viewer *interact* — hovering for
detail, filtering, drilling down, changing the view. Dashboards and interactive
charts are the common forms.

- **Strengths** — *exploration and flexibility*. Viewers answer their *own*
  questions, examine the segments they care about, and see detail on demand. One
  dynamic visualization can serve many viewers with different questions.
- **Weaknesses** — *complexity and less control*. They are harder to build, require
  a platform to run (not a static document), and the maker cannot guarantee what the
  viewer will see or conclude — an interactive chart can be misread through
  interaction, or overwhelm a viewer who just wanted the answer.

Dynamic visualization suits *ongoing monitoring* and *self-service exploration* — the
dashboard's purpose — where different viewers have different questions.

Choosing between them
----------------------

The choice follows the purpose and audience. A *specific message* to a *broad or
passive audience* (a report, a presentation, a public post) usually wants a
**static** chart — focused, controlled, universally viewable. An *exploratory tool*
for an *engaged audience* who will ask varied questions (an analytics dashboard, a
data product) wants a **dynamic** one. The deciding questions are the familiar ones —
what is the message, who is the audience, and will they explore or just receive?

The caveat
------------

Interactivity is seductive and often overused — a dynamic dashboard is not
automatically better than a static chart, and building interactivity where a simple
static image would communicate more clearly wastes effort and can *dilute* the
message (a viewer given ten filters may miss the one point you needed them to see).
Conversely, forcing exploration into a static format frustrates users who need to dig
in. Match the format to the genuine need: static for a controlled message, dynamic
for genuine exploration — and default to the simpler static form unless interaction
adds real value. The next lessons turn to the visual design elements themselves.
"""


MINDMAP.update({
    "Data Visualization": [
        "The Role and Importance of Data Visualization",
        "Connecting Data and Images",
        "Data Creates Value Only When It Is Communicated",
        "Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose",
    ],
    "Connecting Data and Images": [
        "Data Visualization",
        "Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose",
        "Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement",
        "Choosing the Right Visualization: Audience-Centered Design and Chart Selection",
    ],
    "Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose": [
        "Connecting Data and Images",
        "Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity",
        "Choosing the Right Visualization: Audience-Centered Design and Chart Selection",
        "Data Storytelling: Giving Numbers a Clear and Convincing Voice",
    ],
    "Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity": [
        "Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose",
        "Dashboards",
        "Introduction to Tableau",
        "Data Visualization",
    ],
})


# ======================================================================
# Section 6 — Data Visualization / principles (close)  (viz 005-008)
# ======================================================================

GLOSS.update({
    "Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement":
        "the visual building blocks — line, shape, colour, space, movement — applied to charts",
    "Choosing the Right Visualization: Audience-Centered Design and Chart Selection":
        "matching chart type to the data, the message, and the audience",
    "Design Thinking in Data Visualization: A User-Centered Framework":
        "designing visualizations around the viewer's needs — a user-centered process",
    "Accessibility in Data Visualization: Designing for Everyone":
        "making charts readable by everyone, including people with visual differences",
})

CONTENT["Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement"] = r"""
The visual building blocks
----------------------------

Data visualization borrows from visual art, and the **elements of art** — line,
shape, colour, space, and movement — are the building blocks from which every chart
is composed. Understanding them as *tools* gives an analyst deliberate control over
how a visualization looks and reads, rather than accepting a charting tool's
defaults. This lesson applies the artist's vocabulary to the analyst's charts.

The elements and their roles
------------------------------

- **Line** — connects, directs, and shows continuity. In charts, lines carry trends
  (a line chart's path over time), define axes and gridlines, and guide the eye.
  Line weight and style signal importance — a bold line draws attention, a light
  gridline recedes.
- **Shape** — distinguishes and encodes. Different marker shapes separate
  categories; a chart's overall shape (bars, a curve, a scatter cloud) is itself
  meaningful. Shape is a channel for encoding a categorical dimension.
- **Colour** — the most powerful and most abused element. Colour distinguishes
  categories (distinct hues), encodes magnitude (intensity, as in a heat map), and
  directs attention (one highlighted colour against muted others). Used well it
  clarifies; used carelessly it confuses or misleads.
- **Space** — the arrangement and the emptiness. How elements are positioned, and the
  *white space* around them, shapes readability. Crowded charts overwhelm; generous
  space focuses. Space also encodes — position is the eye's most precise channel.
- **Movement** — how the eye travels through the visualization, and, in dynamic
  charts, literal animation. Good design directs the eye deliberately — to the key
  point first, then through the supporting detail.

Using the elements deliberately
----------------------------------

The elements are levers for the *focus* the earlier lesson demanded. To emphasise one
data series, give it a bold **line** or a saturated **colour** while muting the
rest; to separate categories, use distinct **shapes** or **hues**; to reduce clutter,
add **space** and remove non-data ink; to guide the reader, arrange for **movement**
toward the key insight. Effective visualization is the deliberate use of these
elements to make the important thing stand out and the structure read clearly.

Colour, the element to handle with care
-----------------------------------------

Colour deserves special caution because it is so easily misused. Too many colours
fragment a chart into confusion; colour used decoratively (a rainbow of bars that all
mean the same thing) adds noise without meaning; and colour that implies an order the
data lacks misleads. The discipline is to use colour *purposefully* — to encode a real
distinction or to direct attention — and sparingly, with a limited, intentional
palette. The accessibility lesson adds a further constraint: colour choices must work
for viewers who perceive colour differently.

The caveat
------------

The elements of art can beautify a chart without improving it — or while actively
harming it. Decoration that adds no information (gradients, 3D effects, background
images, superfluous colour) is "chartjunk": it competes with the data for attention
and often distorts perception (3D bars misrepresent their values). The principle,
inherited from the clarity-over-cleverness thread, is that every visual element should
*serve the data* — encode information or aid comprehension — and anything that merely
decorates should be removed. The elements are tools for clarity, not ornamentation;
a beautiful chart that obscures its data has failed. The next lesson turns to choosing
the right chart type.
"""

CONTENT["Choosing the Right Visualization: Audience-Centered Design and Chart Selection"] = r"""
Matching chart to message
---------------------------

With the principles and elements established, a concrete question recurs constantly:
*which chart type* for this data and this message? **Choosing the right
visualization** is the skill of matching chart type to what the data is and what you
want to say — and doing so with the *audience* in mind. The wrong chart type can
obscure a clear finding; the right one makes it obvious.

Chart types and what they show
--------------------------------

Common chart types each suit particular data and messages:

- **Bar chart** — comparing quantities across categories. The eye reads bar lengths
  accurately, making bars the reliable default for "compare these amounts" (sales by
  region, counts by type).
- **Line chart** — showing change over time or a continuous trend. Position along a
  path reads clearly, so lines are the standard for time series.
- **Pie chart** — showing parts of a whole, *sparingly*. Because the eye reads angles
  and areas poorly, pie charts work only for a few slices where proportions are
  roughly comparable; a bar chart is usually clearer.
- **Scatter plot** — showing the relationship between two numeric variables.
  Position (the most precise channel) reveals correlation, clusters, and outliers.
- **Histogram** — showing the distribution of one numeric variable — its shape,
  centre, and spread.
- **Heat map** — showing magnitude across two dimensions via colour intensity, for a
  grid of values.

The rule: **let the data and the message choose the chart.** A comparison wants bars;
a trend wants a line; a relationship wants a scatter plot; a distribution wants a
histogram. Forcing data into the wrong chart type (a pie chart for a comparison, a
line for unrelated categories) fights the reader.

Audience-centered selection
-----------------------------

Chart choice also depends on the *audience*. A technical audience reads a scatter plot
or box plot fluently; a general audience may need a simpler bar or line chart. A
familiar chart type communicates faster than a sophisticated one the audience must
decode — the best chart is not the most advanced but the one *this* audience reads
most easily. Matching the chart to the audience's visual literacy is as important as
matching it to the data.

Selection as a decision
-------------------------

Choosing a chart is a *deliberate decision*, not a default. The process: identify what
the data *is* (categories, a time series, two numeric variables, a distribution),
identify the *message* (a comparison, a trend, a relationship, a shape), consider the
*audience* (what they read easily), and pick the chart type that fits all three. This
turns chart selection from habit ("always a bar chart") into a reasoned choice that
serves the specific data, message, and reader.

The caveat
------------

There is rarely a single "correct" chart — often several would work, and the choice
involves judgement and trade-offs (a stacked bar versus grouped bars, a line versus a
bar for few time points). The failure to avoid is not picking the *theoretically
optimal* chart but picking an *actively wrong* one — a pie chart with fifteen slices,
a 3D chart that distorts, a chart type that mismatches the data. Aim for a chart that
clearly and honestly conveys the message to the audience; among the several that do,
the differences are refinements. The next lesson brings a structured design process
to these choices.
"""

CONTENT["Design Thinking in Data Visualization: A User-Centered Framework"] = r"""
Designing for the viewer
--------------------------

Effective visualization is a *design* problem, and design has methods. **Design
thinking** — a structured, user-centered approach to design — applies directly to
data visualization, shifting the focus from "what do I want to show" to "what does my
*viewer* need to understand." This lesson brings a deliberate design process to the
visualization choices the section has been building.

What design thinking means here
---------------------------------

Design thinking centres the *user* — here, the viewer of the visualization — and
proceeds through understanding their needs before designing. Applied to
visualization, its phases look like:

- **Empathise** — understand the viewer: who they are, what they know, what question
  they bring, what decision they must make. A visualization for executives differs
  from one for analysts because their needs differ.
- **Define** — state precisely what the visualization must accomplish: the specific
  insight it should convey to this viewer for this purpose. This is the *purpose* the
  earlier lesson demanded, made explicit.
- **Ideate** — consider multiple ways to visualize it: different chart types,
  emphases, and layouts, rather than settling on the first idea.
- **Prototype** — build a draft visualization.
- **Test** — check it against the viewer's needs: does it convey the insight clearly?
  Can the intended viewer read it? Refine based on what you learn.

The process is *iterative* — testing reveals problems that send you back to redesign,
just as refining an analysis loops.

Why user-centered design matters
----------------------------------

The core shift is from *maker-centered* to *user-centered* — from "here is the data I
have" to "here is what my viewer needs to understand." A maker-centered chart shows
what was easy or what the maker finds interesting; a user-centered one shows what the
viewer needs, in the form they can read. This is the same audience-adaptation
principle from the communication lessons, applied as a design discipline: the
visualization exists to serve its viewer's understanding, and every choice should be
judged by that standard.

Design thinking and the earlier principles
--------------------------------------------

Design thinking ties the section's principles into a process. *Empathise* and
*define* establish the **purpose** and **audience**; *ideate* explores **chart types**
and **elements**; *prototype* and *test* apply **focus** and **structure** and check
them against a real viewer. Rather than a checklist of principles, design thinking
gives a *workflow* for arriving at a visualization that embodies them — a repeatable
way to design for the viewer.

The caveat
------------

Design thinking is a framework, not a formula, and it can be over-applied — a simple
chart for a clear message does not need a five-phase design process, and treating
every visualization as a major design project wastes effort. The value scales with
the stakes: a throwaway exploratory chart needs none of this, while an important
visualization for a key decision or a broad audience rewards the full process. And
"user-centered" does not mean giving viewers whatever they *want* — sometimes the
honest, clear visualization is not the flattering one they would prefer, and serving
their genuine *understanding* takes priority over pleasing them. Match the process to
the stakes, and keep honesty above appeal. The next lesson addresses designing for
*all* viewers: accessibility.
"""

CONTENT["Accessibility in Data Visualization: Designing for Everyone"] = r"""
Visualizations for everyone
-----------------------------

A visualization only communicates to those who can *perceive* it — and a meaningful
fraction of any audience perceives colour, contrast, or detail differently.
**Accessibility** in data visualization means designing charts that everyone can read,
including people with colour vision deficiency, low vision, or other differences.
Closing the principles stage, this lesson makes clear that accessible design is not a
niche concern but part of communicating to your *whole* audience.

Why accessibility matters
---------------------------

Roughly one in twelve men and one in two hundred women has some form of colour vision
deficiency, and many viewers have low vision or read charts on small or poor screens.
A visualization that relies on distinctions these viewers cannot perceive simply fails
for them — the insight does not land. Since the purpose of an explanatory
visualization is to communicate to an audience, a chart that excludes part of that
audience has partially failed at its one job. Accessibility is effectiveness for
everyone.

Practices for accessible visualization
-----------------------------------------

Concrete practices make charts widely readable:

- **Do not rely on colour alone.** Because colour vision varies, never encode
  meaning *only* in colour — add a second cue: labels, patterns, shapes, or direct
  text. If red-versus-green is the only difference between two lines, colour-blind
  viewers cannot tell them apart; adding labels or distinct markers fixes it.
- **Use colour-blind-friendly palettes.** Certain colour combinations (notably
  red/green) are problematic; palettes designed for colour vision deficiency (and
  distinguishable by brightness, not just hue) work for far more viewers.
- **Ensure sufficient contrast.** Text and marks must contrast enough with the
  background to be readable, especially for low-vision viewers and poor displays.
- **Label directly and clearly.** Direct labels on data (rather than a legend the
  viewer must cross-reference) and legible text sizes help everyone, and are
  essential for some.
- **Provide text alternatives.** A caption or description conveying the chart's key
  insight in words serves viewers using screen readers and reinforces the message for
  all.

Accessibility as good design
------------------------------

Accessible design generally *improves* a visualization for everyone, not only for
those who need it. Not relying on colour alone, ensuring contrast, and labelling
directly make a chart clearer for *all* viewers — the same practices that serve
colour-blind or low-vision readers reduce ambiguity and effort for everyone. Designing
for accessibility is, in large part, simply designing *well*: the constraints push
toward the clarity that good visualization aims at regardless.

The caveat
------------

Accessibility is a spectrum, not a checkbox — no single design serves every possible
need perfectly, and there are trade-offs (a palette optimised for one form of colour
vision deficiency may not be ideal for another). The goal is *reasonable* inclusion:
apply the well-established practices (not colour alone, adequate contrast, clear
labels, text alternatives) that serve the large majority, rather than either ignoring
accessibility or being paralysed by the impossibility of perfection. Some
accessibility is vastly better than none, and the core practices are low-cost and
broadly beneficial. This completes the principles of visualization; the next lessons
turn to a specific tool for building them — Tableau.
"""


MINDMAP.update({
    "Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement": [
        "Connecting Data and Images",
        "Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose",
        "Choosing the Right Visualization: Audience-Centered Design and Chart Selection",
        "Accessibility in Data Visualization: Designing for Everyone",
    ],
    "Choosing the Right Visualization: Audience-Centered Design and Chart Selection": [
        "Connecting Data and Images",
        "Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement",
        "Design Thinking in Data Visualization: A User-Centered Framework",
        "Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose",
    ],
    "Design Thinking in Data Visualization: A User-Centered Framework": [
        "Choosing the Right Visualization: Audience-Centered Design and Chart Selection",
        "Accessibility in Data Visualization: Designing for Everyone",
        "Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose",
        "Data Storytelling: Giving Numbers a Clear and Convincing Voice",
    ],
    "Accessibility in Data Visualization: Designing for Everyone": [
        "Design Thinking in Data Visualization: A User-Centered Framework",
        "Choosing the Right Visualization: Audience-Centered Design and Chart Selection",
        "Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement",
        "Introduction to Tableau",
    ],
})


# ======================================================================
# Section 6 — Data Visualization / Stage: tableau  (viz 009-012)
# NOTE: viz 011 title contains subscript-2 (U+2082 ₂) — key written literally
# ======================================================================

GLOSS.update({
    "Introduction to Tableau":
        "what Tableau is and why analysts use it — visual analytics without heavy coding",
    "Getting Started with Tableau Public":
        "the free version: connecting data, building a first view, publishing to the web",
    "Creating a CO\u2082 Emissions Visualization in Tableau Public":
        "a worked Tableau example — turning an emissions dataset into a clear visualization",
    "Effective vs. Ineffective Data Visualizations in Tableau":
        "the difference in practice — what makes a Tableau chart clear versus confusing",
})

CONTENT["Introduction to Tableau"] = r"""
A tool built for visualization
--------------------------------

While spreadsheets can chart and SQL can query, **Tableau** is a tool built
specifically for *visual analytics* — creating interactive visualizations and
dashboards from data, with little or no code. Opening the Tableau stage, this lesson
introduces what Tableau is, what it does, and why it is one of the most widely used
visualization tools in data analytics.

What Tableau is
----------------

Tableau is a visual-analytics platform that connects to data sources and lets you
build visualizations by *dragging and dropping* fields rather than writing code. Its
core idea is *direct manipulation*: you place a field on an axis, a colour, or a
filter, and Tableau draws the chart, letting you explore and refine visually. This
makes sophisticated, interactive visualization accessible to analysts who are not
programmers, which is much of its appeal.

What Tableau does well
------------------------

Tableau's strengths align with the visualization principles the section established:

- **Interactive visualizations** — the dynamic charts and dashboards from the
  static-versus-dynamic lesson, letting viewers filter, hover, and drill down.
- **Rapid exploration** — drag-and-drop makes trying different views fast, supporting
  the exploratory purpose of finding patterns.
- **Connecting to many data sources** — spreadsheets, databases (via SQL), and files,
  bringing data in without manual export where possible.
- **Dashboards** — combining multiple visualizations into a single interactive view
  for monitoring and self-service (a later lesson's subject).
- **Polished, shareable output** — professional visualizations suitable for
  communicating findings to stakeholders.

Where Tableau fits
--------------------

Tableau occupies the *visualization and dashboard* niche in the analyst's toolkit,
downstream of the data preparation and analysis the earlier sections covered.
Typically, data is prepared and analysed (in spreadsheets, SQL, or code), then Tableau
visualizes the results — it is a presentation-and-exploration layer, not primarily a
cleaning or heavy-computation tool. It complements rather than replaces the other
tools: SQL or a spreadsheet shapes the data, Tableau makes it visible and interactive.

The caveat
------------

Tableau is powerful but is one option among several, and it is not always the right
one. It is a specialised (and, in its full version, commercial) tool with its own
learning curve, and for a simple static chart a spreadsheet may be faster and more
universally accessible. Tableau also does not remove the need for the *principles* —
its ease of making charts makes it just as easy to make *bad* charts (the wrong type,
misleading scales, chartjunk), so the design judgement from the principles stage
matters as much in Tableau as anywhere. The tool accelerates visualization; it does
not substitute for knowing what makes a visualization good. The next lesson gets hands
on with the free version, Tableau Public.
"""

CONTENT["Getting Started with Tableau Public"] = r"""
The free way in
----------------

**Tableau Public** is a free version of Tableau that lets anyone build and publish
interactive visualizations — the practical entry point for learning the tool. This
lesson covers getting started with it: what it is, how the workflow goes from data to
published visualization, and what to know before diving in.

What Tableau Public is
------------------------

Tableau Public is a no-cost version of Tableau with one defining characteristic: the
visualizations you create are *published publicly* to the web, to your Tableau Public
profile, where anyone can view them. This is its central trade-off — free and capable,
but not private, so it suits learning, portfolios, and public data storytelling, and
is *not* suitable for confidential or proprietary data.

The basic workflow
--------------------

Building a visualization in Tableau Public follows a consistent flow:

- **Connect to data** — load a data source: commonly a spreadsheet or CSV file for
  Tableau Public. Tableau reads the fields and classifies them as *dimensions*
  (categorical: region, date, product) and *measures* (numeric values to aggregate:
  sales, counts).
- **Build a view** — drag fields onto the *rows* and *columns* shelves to define the
  chart's structure, and onto *marks* (colour, size, label) to encode more. Tableau
  draws the chart and re-draws as you adjust — the direct-manipulation exploration.
- **Choose and refine the chart** — pick or adjust the chart type, add filters, format
  the visualization applying the design principles (clear focus, right encoding,
  accessible colour).
- **Publish** — save to your Tableau Public profile, producing a shareable,
  interactive visualization on the web.

Dimensions and measures
-------------------------

A key Tableau concept is the *dimension* versus *measure* distinction. **Dimensions**
are categorical fields you group or break data down by (the ``GROUP BY`` categories
from SQL); **measures** are numeric fields Tableau aggregates (sum, average — the
aggregates from the analysis section). Placing a dimension and a measure together —
region (dimension) and sales (measure) — produces an aggregated visualization, sales
by region, exactly the grouped aggregation the analysis section covered, now drawn
automatically. Understanding this mapping connects Tableau to the analytical concepts
already learned.

The caveat
------------

Tableau Public's defining limitation bears repeating because getting it wrong is
serious: **everything published is public.** Uploading confidential, proprietary, or
personal data to Tableau Public exposes it to the world — a data-privacy failure of
exactly the kind the ethics and PII lessons warned against. Tableau Public is for
public or non-sensitive data only; confidential work requires the paid Tableau
versions with private hosting. Beyond privacy, the free version has some feature and
data-size limits versus the commercial product. Use Tableau Public for learning,
portfolios, and public data — never for anything that must stay private. The next
lesson works through a concrete example.
"""

CONTENT["Creating a CO\u2082 Emissions Visualization in Tableau Public"] = r"""
A worked example
-----------------

Principles and workflow come together in a concrete build. This lesson walks through
creating a visualization of **CO\u2082 emissions** data in Tableau Public — a
real-world environmental dataset — showing how the abstract steps become an actual
chart, and how the design principles guide the choices along the way.

The data and the question
---------------------------

An emissions dataset typically holds, per row, a place and time and an emissions
figure — for example country, year, and CO\u2082 emissions (in tonnes or per capita).
Before building, the *question* comes first (the purpose principle): are we showing
emissions *over time* (a trend), *across countries* (a comparison), or *by region on a
map* (a geographic pattern)? The question determines everything downstream — the chart
type, the fields used, the emphasis. Suppose the question is how emissions have changed
over time for a set of countries: a *trend comparison*.

Building the visualization
----------------------------

With the question set, the Tableau workflow builds the answer:

- **Connect** the emissions dataset (a CSV). Tableau classifies ``country`` and
  ``year`` as dimensions and ``emissions`` as a measure.
- **Build the view** — for an over-time trend, place ``year`` on columns and
  ``emissions`` on rows, producing a line; place ``country`` on colour to draw one line
  per country, enabling comparison. This is a multi-series line chart — the right
  structure for a trend comparison (position over time, the precisely-read encoding).
- **Refine** — apply the principles: a clear title stating the message, a limited and
  accessible colour palette (not colour alone — label the lines directly where
  possible), appropriate axis scaling that does not distort, and removal of chartjunk.
  Focus the chart on its one message: how these countries' emissions have diverged or
  converged over time.
- **Publish** to Tableau Public as an interactive chart viewers can hover and filter.

Principles in the concrete
----------------------------

The example makes the abstract principles tangible. The **purpose** (the emissions
question) drove the **chart choice** (a line for a trend); the **encoding** used the
eye's precise channel (position over time); **colour** distinguished countries but was
kept accessible and paired with labels; and **focus** kept the chart to its single
message. A CO\u2082 visualization built this way communicates a real environmental
insight clearly — the whole section's principles applied to one concrete, meaningful
dataset.

The caveat
------------

Emissions data — like much real-world data — carries interpretation hazards the
visualization must respect. *Total* versus *per-capita* emissions tell very different
stories (a populous country leads on total but may be low per person), and choosing
which to show is a framing decision with real consequences for what the chart implies;
showing one while implying the other misleads. Time ranges, the set of countries
included, and absolute-versus-relative framing similarly shape the message. The
honest-visualization obligation is acute for consequential public data like emissions:
the chart should represent what the data genuinely says, with its framing made clear,
not shade a complex picture toward a preferred narrative. The next lesson contrasts
effective and ineffective visualizations directly.
"""

CONTENT["Effective vs. Ineffective Data Visualizations in Tableau"] = r"""
The difference in practice
----------------------------

Tableau makes it easy to build visualizations — and just as easy to build *bad* ones.
Closing the practical Tableau lessons, this one contrasts **effective and ineffective
visualizations** directly, making concrete the difference between a chart that
communicates and one that confuses, and showing the principles as a practical checklist.

What makes a visualization ineffective
-----------------------------------------

Ineffective visualizations share recognisable faults, most of them violations of the
principles:

- **Wrong chart type** — a pie chart for a comparison across many categories, a line
  chart for unrelated categories, a chart type that mismatches the data (the
  chart-selection lesson).
- **Cluttered and unfocused** — too much on one chart, no clear message, competing
  elements so nothing stands out (a failure of focus).
- **Chartjunk** — 3D effects, unnecessary colours, decorative elements that distort or
  distract (the elements-of-art caveat).
- **Misleading scales** — a truncated axis exaggerating differences, an inconsistent
  scale, a distorted encoding (a structure-and-honesty failure).
- **Poor colour and accessibility** — too many colours, colour-only encoding,
  low contrast (the accessibility lesson).
- **Missing context** — no clear title, unlabelled axes, no indication of what the
  viewer is looking at.

What makes a visualization effective
--------------------------------------

Effective visualizations are, correspondingly, the principles realised:

- **Right chart type** for the data and message.
- **Clear focus** — one message, emphasised, distractions removed.
- **Honest structure** — undistorted scales, faithful encodings.
- **Purposeful, accessible colour** — limited palette, not colour alone, good contrast.
- **Clear context** — a title stating the message, labelled axes, legible text.
- **Appropriate simplicity** — as simple as the message allows, no more.

The contrast is not about sophistication — an effective chart is often *simpler* than
an ineffective one, because it has removed everything that does not serve the message.

Effective as a checklist
--------------------------

The effective-versus-ineffective contrast turns the section's principles into a
practical review checklist for any visualization, in Tableau or elsewhere: *Is the
chart type right? Is there one clear focus? Are the scales honest? Is the colour
purposeful and accessible? Is there enough context? Is it as simple as it can be?* A
visualization that passes these communicates; one that fails them confuses. Running this
check before publishing catches the common faults while they are easy to fix.

The caveat
------------

"Effective" is judged against a *purpose and audience*, not in the abstract — a chart
effective for analysts may be ineffective for executives, and vice versa, so the
checklist is applied *relative to whom the chart is for*. And effectiveness is not
binary but a spectrum; most real charts are neither perfect nor terrible but improvable,
and the goal is a chart good enough to communicate its message clearly and honestly to
its audience, not an unattainable ideal. Use the contrast to *improve* visualizations
toward clarity and honesty, judged by their actual purpose and readers. This closes the
Tableau lessons; the next stage turns to data storytelling — weaving visualizations into
a narrative.
"""


MINDMAP.update({
    "Introduction to Tableau": [
        "Data Visualization",
        "Getting Started with Tableau Public",
        "Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity",
        "Effective vs. Ineffective Data Visualizations in Tableau",
    ],
    "Getting Started with Tableau Public": [
        "Introduction to Tableau",
        "Creating a CO\u2082 Emissions Visualization in Tableau Public",
        "Effective vs. Ineffective Data Visualizations in Tableau",
        "Choosing the Right Visualization: Audience-Centered Design and Chart Selection",
    ],
    "Creating a CO\u2082 Emissions Visualization in Tableau Public": [
        "Getting Started with Tableau Public",
        "Introduction to Tableau",
        "Effective vs. Ineffective Data Visualizations in Tableau",
        "Choosing the Right Visualization: Audience-Centered Design and Chart Selection",
    ],
    "Effective vs. Ineffective Data Visualizations in Tableau": [
        "Creating a CO\u2082 Emissions Visualization in Tableau Public",
        "Connecting Data and Images",
        "Accessibility in Data Visualization: Designing for Everyone",
        "Data Storytelling: Giving Numbers a Clear and Convincing Voice",
    ],
})


# ======================================================================
# Section 6 — Data Visualization / tableau (close) + story (open)  (viz 013-016)
# ======================================================================

GLOSS.update({
    "Using Creativity in Tableau":
        "going beyond default charts — custom, expressive visualizations that still communicate",
    "Linking Multiple Datasets in Tableau Public":
        "combining data sources inside Tableau — joins and relationships for richer views",
    "Data Storytelling: Giving Numbers a Clear and Convincing Voice":
        "wrapping data in narrative so it informs, persuades, and is remembered",
    "Engaging Your Audience in Data Storytelling: Identifying the Key Message":
        "finding the one message that matters and building the story around it",
})

CONTENT["Using Creativity in Tableau"] = r"""
Beyond the defaults
---------------------

Tableau's default charts communicate well, but the tool also supports *creative*,
custom visualizations that go beyond bars and lines — and used judiciously, creativity
can make a visualization more engaging and memorable. This lesson covers using
Tableau's flexibility expressively, while keeping creativity in service of clarity
rather than at its expense.

Where creativity fits
-----------------------

Tableau enables visualizations beyond the standard types:

- **Custom chart types** — combining marks, dual axes, and calculated fields to build
  visualizations the default menu does not offer directly.
- **Custom formatting and layout** — fonts, colours, annotations, and arrangements
  that give a visualization a distinct, polished, on-brand look.
- **Maps and geographic views** — Tableau's built-in geographic capabilities turn
  location data into maps, a naturally engaging format for spatial data.
- **Annotations and storytelling elements** — text, reference lines, and highlights
  that guide the viewer and add narrative (connecting to the storytelling stage).

Creativity here means using the tool's expressive range to make a visualization
clearer and more compelling — not decorating for its own sake.

Creativity in service of communication
----------------------------------------

The guiding principle is that creativity must *serve the message*, not compete with
it. A creative visualization that communicates its insight more clearly or memorably
than a standard chart is a success; one that is novel but harder to read is a failure,
however impressive. The test is always the same: does the creative choice help the
viewer understand faster, or does it make them work harder? Effective creativity
lowers the barrier to understanding; ineffective creativity raises it in the name of
originality.

Balancing engagement and clarity
----------------------------------

Creativity trades against the clarity and convention that aid quick reading. A
familiar chart type is instantly readable *because* it is familiar; an inventive one
asks the viewer to learn how to read it. Sometimes the engagement is worth that cost —
a memorable, distinctive visualization that draws attention to an important message —
and sometimes it is not. The judgement is whether the creative element's benefit
(engagement, memorability, a genuinely better fit) outweighs its cost (unfamiliarity,
reading effort), for this message and audience.

The caveat
------------

Creativity is where visualization most easily goes wrong, because the impulse to be
novel or impressive can override the goal of communicating. The infamous failures —
unreadable custom charts, gratuitous complexity, form over function — come from
creativity unmoored from purpose. The discipline, once more, is clarity over
cleverness: reach for a creative or custom visualization when it *communicates
better*, and default to the clear, conventional chart when it does not. A creative
visualization that people admire but cannot read has failed at the one thing
visualization is for. The next lesson covers combining data sources in Tableau.
"""

CONTENT["Linking Multiple Datasets in Tableau Public"] = r"""
Combining sources for richer views
------------------------------------

Real analysis often draws on several datasets, and Tableau can *combine* them —
linking multiple data sources so a single visualization draws on more than one table.
This lesson covers linking datasets in Tableau Public, applying the data-combining
concepts from the analysis section within the visualization tool.

How Tableau links data
------------------------

Tableau offers a few ways to combine data sources, mirroring the combining techniques
already learned:

- **Joins** — combining tables on a matching key, exactly the SQL JOIN from the
  analysis section, configured visually in Tableau. Tables are joined on a shared
  field (a key), producing a combined dataset with the join-type choices (inner, left,
  right, full) the JOIN lesson covered.
- **Relationships** — Tableau's flexible way to relate tables without a rigid
  up-front join, letting Tableau determine how to combine them per visualization.
- **Blending** — combining data from *different sources* (say a spreadsheet and a
  database) at the visualization level, aggregating each and linking on a common
  field.
- **Unions** — stacking tables with the same structure (appending rows), for
  combining like datasets (this month's and last month's data).

The concepts are the ones from the analysis section — joining on keys, combining
sources — now performed inside Tableau to feed richer visualizations.

Why link datasets
-------------------

Linking data lets a visualization draw on information spread across tables — sales
data joined to product details joined to regional information, visualized together.
Just as the analysis section combined tables to answer richer questions, linking
datasets in Tableau enables richer *visualizations*, showing relationships across data
that no single table holds. It brings the relational-combine power into the
visualization layer.

The caveat
------------

Combining data in Tableau carries exactly the hazards the analysis section flagged for
joins, now one step removed and thus easier to get wrong unnoticed. Joining on a
**non-unique key** multiplies rows and inflates the aggregates Tableau computes — the
fan-out problem, now hidden inside a chart where the wrong numbers look authoritative.
**Mismatched keys** silently drop data; **blending** aggregates before combining,
which can produce subtly different results than a join. The discipline is the same:
understand the relationship between the tables (one-to-one, one-to-many), verify that
combined visualizations show the row counts and totals you expect, and treat a chart
built on linked data with the same row-count skepticism as a SQL join. A visualization
of wrongly-combined data misleads with a confident, polished face. The next lessons
turn from building charts to telling stories with them.
"""

CONTENT["Data Storytelling: Giving Numbers a Clear and Convincing Voice"] = r"""
From charts to story
----------------------

A visualization shows data; a *story* gives it meaning, direction, and persuasive
force. **Data storytelling** is the practice of wrapping data and visualizations in a
narrative — combining the numbers, their visual representation, and a clear message
into something that informs, convinces, and is remembered. Opening the storytelling
stage, this lesson establishes why narrative matters and what data storytelling
involves.

Why storytelling matters
--------------------------

Data alone rarely moves people to act. A chart shows *what* is true; a story explains
*why it matters* and *what to do about it*, and it does so in a form the human mind is
built to receive — humans remember and are persuaded by stories far more than by
isolated facts. The "data creates value only when communicated" principle reaches its
fullest form here: the most rigorous analysis changes nothing if its findings do not
land, and storytelling is how findings land — turning a correct result into an
understood, believed, and acted-upon one.

The elements of data storytelling
-----------------------------------

Data storytelling weaves three things together:

- **The data** — the sound analysis and evidence underneath. Storytelling does not
  replace rigour; it *communicates* it, and a story on weak data is mere persuasion.
- **The visualizations** — the charts that make the data visible and the message
  clear, built with the principles the section established.
- **The narrative** — the structure and words that give the data meaning: the context
  (why this matters), the insight (what the data shows), and the implication (what it
  means and what to do). The narrative connects the visualizations into a coherent arc
  rather than a pile of charts.

Together, these turn numbers into a message with a clear and convincing voice.

The narrative arc
-------------------

A data story has structure, often resembling a classic narrative: a *setup* (the
context and question — why we are looking at this), a *development* (what the data
reveals, shown through visualizations), and a *resolution* (the insight and its
implications — what it means and what should happen). This arc gives the audience a
path through the data, building from why-they-should-care to what-they-should-do,
rather than dropping them into disconnected findings. Structuring findings as a
journey is what makes them followable and memorable.

The caveat
------------

Data storytelling sits on an ethical knife-edge, because the same narrative power that
clarifies can *manipulate*. A compelling story can make weak evidence persuasive,
smooth over inconvenient data, or lead an audience to a conclusion the data does not
support — persuasion untethered from truth. The obligation, running through the whole
course, is that the story must serve the *data's actual message*: storytelling should
make a *true* insight clear and compelling, never make a preferred conclusion
persuasive regardless of the evidence. The narrative illuminates the data; it must not
override it. This is the honest-communication principle at its most important, because
storytelling is where distortion is easiest and most effective. The next lesson finds
the key message a story is built around.
"""

CONTENT["Engaging Your Audience in Data Storytelling: Identifying the Key Message"] = r"""
The one thing they should remember
------------------------------------

A data story that tries to say everything says nothing memorable. The heart of
engaging storytelling is **identifying the key message** — the single most important
thing the audience should take away — and building the story around it. This lesson is
about finding that message and using it to focus everything else.

Why one key message
---------------------

Audiences remember little from any presentation — often just one thing, if that. If
you do not decide what that one thing should be, the audience will pick their own (or
nothing), and your most important insight may be lost among lesser ones. Identifying
*the* key message means choosing, deliberately, the single takeaway that matters most,
so the story can drive it home. This is the *focus* principle from visualization
raised to the level of the whole narrative: one clear message, emphasised, distractions
subordinated.

Finding the key message
-------------------------

The key message emerges from asking what the analysis most importantly *means for the
audience*:

- **What is the most important insight?** Among everything the analysis found, which
  finding matters most for the decision at hand?
- **What does it mean for this audience?** The key message is framed around what the
  *audience* cares about and must decide — not the most technically interesting
  finding, but the most *decision-relevant* one.
- **What should they do or think differently?** The strongest key messages point
  toward an action or a changed understanding, giving the audience something to do
  with the insight.

The key message is usually stateable in one sentence — "our newest region is
outgrowing all others and deserves more investment" — and if it cannot be, it is not
yet focused enough.

Building the story around the message
---------------------------------------

Once identified, the key message becomes the *organising principle* of the whole
story. Every visualization is chosen to support it; every piece of context sets it up;
everything that does not serve it is cut. The supporting details become supporting —
subordinate to the one message rather than competing with it. This is what makes a
story *engaging*: the audience is carried toward a single clear point, not scattered
across many, and they leave knowing exactly what they were meant to understand.

Engaging the audience
-----------------------

Engagement also comes from connecting the message to what the audience *cares about* —
framing it in terms of their goals, their decisions, their concerns. A message about
what matters to *them* engages; a technically impressive finding disconnected from
their needs does not. This is the audience-centered thread — from the communication and
design-thinking lessons — applied to storytelling: the key message must be not only
important but important *to this audience*, or it will not land however well told.

The caveat
------------

Focusing on one key message risks *oversimplification* — reducing a nuanced analysis
to a single point can omit important caveats, alternative interpretations, or the
uncertainty around the finding. The resolution is not to abandon focus but to lead
with the key message *and* honestly acknowledge its limits — a clear main point, with
its caveats stated rather than hidden. The goal is a message that is both focused *and*
honest: the audience leaves with the one thing that matters, and an accurate sense of
how much to trust it. A focused message that is misleadingly certain fails the honesty
obligation; a focused message with its limits noted informs. The next lessons turn to
dashboards and focused visuals that carry the story.
"""


MINDMAP.update({
    "Using Creativity in Tableau": [
        "Effective vs. Ineffective Data Visualizations in Tableau",
        "Linking Multiple Datasets in Tableau Public",
        "Creating a CO\u2082 Emissions Visualization in Tableau Public",
        "Introduction to Tableau",
    ],
    "Linking Multiple Datasets in Tableau Public": [
        "Using Creativity in Tableau",
        "Getting Started with Tableau Public",
        "Effective vs. Ineffective Data Visualizations in Tableau",
        "Data Storytelling: Giving Numbers a Clear and Convincing Voice",
    ],
    "Data Storytelling: Giving Numbers a Clear and Convincing Voice": [
        "Engaging Your Audience in Data Storytelling: Identifying the Key Message",
        "Data Creates Value Only When It Is Communicated",
        "Data Dashboards: Organizing Insight for Real-Time Decision Making",
        "Structuring a Persuasive Data Presentation: Turning Insights into Story",
    ],
    "Engaging Your Audience in Data Storytelling: Identifying the Key Message": [
        "Data Storytelling: Giving Numbers a Clear and Convincing Voice",
        "Data Dashboards: Organizing Insight for Real-Time Decision Making",
        "Using Filters to Create Compelling and Focused Visuals",
        "Structuring a Persuasive Data Presentation: Turning Insights into Story",
    ],
})


# ======================================================================
# Section 6 — Data Visualization / story (close) + present (open)  (viz 017-020)
# ======================================================================

GLOSS.update({
    "Data Dashboards: Organizing Insight for Real-Time Decision Making":
        "combining key visualizations into one monitored view for ongoing decisions",
    "Using Filters to Create Compelling and Focused Visuals":
        "filtering to sharpen a visualization's message and enable exploration",
    "Structuring a Persuasive Data Presentation: Turning Insights into Story":
        "arranging findings into a presentation that carries an audience to a conclusion",
    "Designing Effective Data Presentation Slides: Structure, Visuals, and Professional Impact":
        "slide design that supports the message — clear, visual, uncluttered, professional",
})

CONTENT["Data Dashboards: Organizing Insight for Real-Time Decision Making"] = r"""
Many views, one screen
------------------------

Some questions are not answered once but *monitored continuously*, and for these the
right form is a **dashboard** — a single view combining several key visualizations to
give an at-a-glance picture that supports ongoing, real-time decisions. This lesson,
within the storytelling stage, covers what dashboards are, when they fit, and how to
design them well.

What a dashboard is
--------------------

A dashboard is a curated collection of visualizations and key metrics arranged
together on one screen, updated as data changes, so a viewer can see the state of what
matters without running an analysis. Where a single chart answers one question and a
data story walks through a finding, a dashboard provides *continuous situational
awareness* — the key numbers and trends for an ongoing concern (sales performance,
operational health, project status), monitored over time.

When dashboards fit
---------------------

Dashboards suit a specific purpose distinct from one-off charts and stories:

- **Ongoing monitoring** — tracking key metrics continuously rather than answering a
  question once. A dashboard is for the questions you ask *repeatedly*.
- **Real-time or regularly-updated decisions** — where decisions depend on current
  data, and the dashboard keeps that data in view.
- **Self-service** — letting viewers check the metrics they need without asking an
  analyst each time, often with interactivity to explore (the dynamic-visualization
  purpose).

If a question is asked once, a chart or story answers it; if it is asked continuously,
a dashboard serves it. Matching the form to the frequency of the question is the key
judgement.

Designing effective dashboards
--------------------------------

Good dashboard design applies the section's principles to a *collection*:

- **Focus on what matters** — include the *key* metrics for the decision, not every
  available number. A cluttered dashboard of everything is as useless as a cluttered
  chart; a dashboard is an exercise in choosing what to leave out.
- **Organise logically** — arrange visualizations so the most important are prominent
  and related ones are grouped, guiding the eye through the information (the
  most-important-first structure).
- **Keep each visualization clear** — every chart on the dashboard follows the
  effective-visualization principles; a dashboard is only as good as its component
  charts.
- **Enable appropriate interactivity** — filters and drill-downs where viewers need to
  explore, without overwhelming.

Why dashboards matter
-----------------------

Dashboards operationalise data — turning analysis from an occasional report into a
*continuous* input to decisions. A well-designed dashboard means the right people see
the right metrics at the right time, so decisions are informed by current data as a
matter of routine rather than special request. It is the "data creates value when
communicated" principle made *ongoing*: the value is delivered continuously, whenever
the dashboard is consulted.

The caveat
------------

Dashboards fail in characteristic ways. The commonest is **cramming too much in** —
the impulse to include every metric produces a wall of charts where nothing stands
out and the key signals drown, exactly the focus failure warned against, multiplied
across a screen. Dashboards can also mislead through *stale data* (a dashboard assumed
live but not refreshing), through metrics shown without the context to interpret them
(a number with no baseline or target), and through *vanity metrics* that look
impressive but do not inform decisions. A dashboard is a design problem demanding
restraint: show the few metrics that drive decisions, keep them current and in
context, and resist the pull toward comprehensiveness. The next lesson covers filtering
to focus visuals.
"""

CONTENT["Using Filters to Create Compelling and Focused Visuals"] = r"""
Filtering for focus
----------------------

A visualization showing *all* the data is often less compelling than one showing the
*relevant* data — and **filters** are the tool for that focus. Filtering a
visualization to the subset that matters sharpens its message and, in interactive
form, lets viewers explore their own questions. This lesson covers using filters to
make visuals both focused and engaging, closing the storytelling stage.

How filters sharpen a visualization
--------------------------------------

Filtering restricts a visualization to a chosen subset, which serves focus directly:

- **Removing noise** — filtering out irrelevant categories, periods, or outliers so
  the chart shows only what bears on the message. A trend across a decade may be
  clearer filtered to the relevant few years.
- **Isolating the story** — filtering to the segment the message concerns (the one
  region, product, or period the insight is about), so the visualization makes exactly
  its point without distraction.
- **Enabling comparison** — filtering to one subset, then another, to show a
  contrast that the full data would obscure.

A focused, filtered visualization communicates its message more forcefully than a
cluttered complete one — the focus principle, achieved by subtraction.

Interactive filters for exploration
-------------------------------------

In dynamic visualizations and dashboards, filters become *interactive controls* that
let viewers choose the subset they see:

- **Viewer-driven focus** — viewers filter to *their* region, *their* period, *their*
  segment, getting a visualization focused on what they care about.
- **Self-service exploration** — one filtered visualization serves many viewers with
  different questions, each filtering to their own view (the dynamic-visualization
  strength).
- **Guided exploration** — filters can be designed to steer viewers through the data
  in a useful sequence, combining interactivity with narrative.

Interactive filtering turns a single visualization into a flexible tool that adapts to
each viewer's question.

Filters in storytelling
-------------------------

Filters connect focus to narrative. A data story can use filtering to *reveal* —
showing the whole, then filtering to the segment that carries the insight, walking the
audience from context to point. And in a dashboard, filters let each viewer focus the
shared view on their concern. Filtering is thus both a *design* tool (sharpening a
static visual's message) and an *interaction* tool (enabling exploration) — two ways of
using the same operation to serve focus.

The caveat
------------

Filtering is powerful and, precisely because it shapes what the viewer sees, a
frequent source of *distortion* — the honest-visualization concern in sharp form.
Filtering to only the data that supports a conclusion (excluding the inconvenient
period, the contradicting segment) manufactures a misleading picture while looking like
mere focus. The line between *focusing* on the relevant and *cherry-picking* the
favourable is exactly the line between honest and dishonest filtering: focus removes
what is *irrelevant to the message*; cherry-picking removes what is *inconvenient to the
conclusion*. The discipline is to filter for relevance in service of the data's true
message, to be transparent about what a visualization excludes, and to be especially
wary when a filter happens to strengthen a preferred narrative. This closes the
storytelling stage; the next lessons turn to presenting the story.
"""

CONTENT["Structuring a Persuasive Data Presentation: Turning Insights into Story"] = r"""
From findings to presentation
-------------------------------

Analysis often culminates in a *presentation* — standing before an audience to convey
what the data shows and what should be done. **Structuring a persuasive data
presentation** is the craft of arranging findings into a presentation that carries an
audience from context to conclusion. Opening the final stage of the section, this
lesson covers how to structure a data presentation as a story.

The presentation as a story
------------------------------

A persuasive presentation is not a data dump but a *narrative* — the storytelling
principles from the previous stage applied to the presentation format. It has an arc:
it establishes *why the audience should care*, develops *what the data shows*, and
arrives at *what it means and what to do*. This structure carries the audience along a
path rather than presenting disconnected findings, and it is what makes a presentation
persuasive: the audience is led to the conclusion, understanding each step, rather than
handed a result to accept.

A structure for data presentations
------------------------------------

A common, effective structure:

- **Open with the context and question** — why this matters, what question the
  analysis addressed. This earns the audience's attention and frames everything after.
- **Present the key message early** — often the strongest presentations state the main
  takeaway near the start, then support it, rather than building suspense (business
  audiences usually want the conclusion first). The key message from the storytelling
  lesson leads.
- **Develop with supporting evidence** — the visualizations and findings that support
  the message, each building the case, in a logical order.
- **Address implications and recommendations** — what the insight means for the
  audience and what they should do — the payoff that makes the analysis actionable.
- **Close with the key message and next steps** — reinforce the takeaway and make the
  call to action clear.

This arc turns findings into a persuasive journey with a clear destination.

Structuring around the audience
---------------------------------

The structure serves the *audience* and the *decision*. The context is framed around
what they care about; the key message is what matters to them; the depth of evidence
matches their needs (executives want the conclusion and confidence; analysts want the
method); the recommendations address their decision. The audience-centered thread —
from communication, design thinking, and storytelling — governs the presentation's
structure: it is built to bring *this* audience to *this* decision.

The caveat
------------

"Persuasive" carries the same ethical weight as "storytelling," intensified by the
presentation's directness and authority. A well-structured presentation can persuade an
audience of a conclusion the data does not support — through selective evidence, a
structure that hides caveats, or confident delivery outrunning the actual findings. The
obligation is that persuasion must serve the *truth*: the presentation should bring the
audience to the conclusion the *data genuinely supports*, with its uncertainties and
limitations honestly included, not engineer agreement regardless of the evidence.
Persuasive structure is for making a *true and important* message land, not for winning
assent to a predetermined position. A presentation that persuades past the evidence
betrays the analyst's core obligation to truth. The next lesson covers the slides that
carry the presentation.
"""

CONTENT["Designing Effective Data Presentation Slides: Structure, Visuals, and Professional Impact"] = r"""
Slides that serve the message
-------------------------------

Most data presentations are carried by *slides*, and slide design can strengthen a
presentation or sabotage it. **Designing effective data presentation slides** is about
making slides that support the spoken message clearly and professionally — visual,
uncluttered, and focused. This lesson covers the principles of good slide design for
data.

The purpose of a slide
------------------------

A slide *supports* the presenter; it is not the presentation itself. Its job is to
show what is better *seen* than *said* — a visualization, a key number, a short list —
while the presenter provides the narrative. The commonest slide failure is treating the
slide as the script: dense paragraphs the presenter reads aloud, which bores the
audience and splits their attention between reading and listening. A good slide shows;
the presenter tells.

Principles of effective data slides
-------------------------------------

Effective slides follow clear principles, most echoing the visualization thread:

- **One point per slide** — each slide makes a single clear point, as each
  visualization has one focus. A slide trying to make several points makes none
  memorably.
- **Visual over textual** — show a chart, a number, an image rather than text
  wherever possible; the visualization communicates faster than a paragraph.
- **Minimal text** — a few words, a short title stating the point, not sentences to
  read. Text on a slide is for labels and the key message, not narration.
- **Clean and uncluttered** — generous space, no unnecessary elements, the audience's
  eye drawn to the one thing that matters (the chartjunk principle, applied to slides).
- **Consistent and professional** — consistent fonts, colours, and layout across
  slides give a polished, credible impression; inconsistency and sloppiness undermine
  trust in the content.

The visualization on the slide
--------------------------------

Because data slides are built around visualizations, every principle from the section
applies to the chart *on* the slide: the right chart type, a clear focus, honest
scales, accessible colour, a title stating the point. A slide's visualization must also
be *readable in the room* — large enough, simple enough, and high-contrast enough to be
seen from the back, which often means simplifying a chart further for a slide than for a
report. The slide is the visualization's most demanding venue.

The caveat
------------

Slide design can consume effort disproportionate to its value, and polish can substitute
for substance. Elaborate animations, lavish design, and hours of formatting do not
improve a presentation whose *analysis* or *message* is weak — and can distract from it,
or signal that style is compensating for lack of content. The principle is that slides
serve the message: invest in making them clear, readable, and professional, but not in
decoration that adds no understanding, and never let slide-making crowd out getting the
analysis and narrative right. A beautiful deck presenting a weak insight is still a weak
presentation. The next lessons cover the frameworks and delivery that complete the
presentation craft.
"""


MINDMAP.update({
    "Data Dashboards: Organizing Insight for Real-Time Decision Making": [
        "Dashboards",
        "Using Filters to Create Compelling and Focused Visuals",
        "Engaging Your Audience in Data Storytelling: Identifying the Key Message",
        "Data Storytelling: Giving Numbers a Clear and Convincing Voice",
    ],
    "Using Filters to Create Compelling and Focused Visuals": [
        "Data Dashboards: Organizing Insight for Real-Time Decision Making",
        "Engaging Your Audience in Data Storytelling: Identifying the Key Message",
        "Data Storytelling: Giving Numbers a Clear and Convincing Voice",
        "Structuring a Persuasive Data Presentation: Turning Insights into Story",
    ],
    "Structuring a Persuasive Data Presentation: Turning Insights into Story": [
        "Data Storytelling: Giving Numbers a Clear and Convincing Voice",
        "Designing Effective Data Presentation Slides: Structure, Visuals, and Professional Impact",
        "Using a Strategic Framework to Structure Data Presentations",
        "Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method",
    ],
    "Designing Effective Data Presentation Slides: Structure, Visuals, and Professional Impact": [
        "Structuring a Persuasive Data Presentation: Turning Insights into Story",
        "Using a Strategic Framework to Structure Data Presentations",
        "Presentation Skills for Data Analysts: Delivering Insights with Confidence",
        "Data Storytelling: Giving Numbers a Clear and Convincing Voice",
    ],
})


# ======================================================================
# Section 6 — Data Visualization / present (cont.)  (viz 021-024)
# ======================================================================

GLOSS.update({
    "Using a Strategic Framework to Structure Data Presentations":
        "reusable structures that reliably organize a presentation toward its goal",
    "Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method":
        "integrating data with narrative — hypothesis, context, and a four-element method",
    "Presentation Skills for Data Analysts: Delivering Insights with Confidence":
        "the delivery half of presenting — voice, pace, presence, and handling nerves",
    "Presenting Like a Pro: Best Practices for Data Analysts":
        "the practices that mark a polished, credible data presenter",
})

CONTENT["Using a Strategic Framework to Structure Data Presentations"] = r"""
Frameworks for reliable structure
-----------------------------------

Rather than structuring every presentation from scratch, experienced presenters lean
on **strategic frameworks** — reusable structures that reliably organise findings
toward a goal. A framework provides a proven skeleton, so the presenter can focus on
the content rather than reinventing the shape. This lesson covers using frameworks to
structure data presentations effectively.

Why use a framework
---------------------

A strategic framework offers several advantages over improvising structure:

- **Proven effectiveness** — frameworks encode what reliably works, sparing you the
  risk of an untested structure that loses the audience.
- **Efficiency** — a framework gives a starting structure immediately, so effort goes
  into the content and message rather than the arrangement.
- **Completeness** — a good framework prompts you to include what matters (context,
  evidence, implications, action), so nothing essential is forgotten.
- **Audience familiarity** — audiences unconsciously recognise well-structured
  presentations and follow them more easily.

The framework is scaffolding: it holds the presentation's shape so the presenter can
concentrate on filling it well.

Common presentation frameworks
--------------------------------

Several frameworks suit data presentations, each a structured path:

- **Situation–Complication–Resolution** — establish the *situation* (the context),
  introduce the *complication* (the problem or question the data addresses), and
  present the *resolution* (the insight and recommendation). A classic, versatile
  structure that mirrors narrative.
- **Question–Answer** — pose the key question the analysis addresses, then answer it
  with the evidence — direct and clear for decision-focused audiences.
- **What / So What / Now What** — state *what* the data shows, *so what* it means, and
  *now what* should be done. A compact structure that moves from finding to
  implication to action.
- **Pyramid structure** — lead with the main conclusion, then support it with grouped
  evidence beneath — the conclusion-first approach business audiences prefer.

Each provides a reliable path from opening to call-to-action; the choice depends on the
audience and the message.

Choosing and applying a framework
-----------------------------------

The framework is a *guide*, not a cage. Choose one that fits the presentation's purpose
and audience — a decision-focused executive audience suits a conclusion-first pyramid or
What/So What/Now What; a problem-solving context suits Situation–Complication–Resolution
— then adapt it to the specific content. The framework ensures the presentation has a
sound structure; the presenter's judgement fills and adjusts it for the actual message
and audience.

The caveat
------------

Frameworks aid structure but can become *formulaic* if applied mechanically — forcing
every presentation into a rigid template regardless of fit produces generic,
going-through-the-motions presentations that do not serve their specific content. The
framework should serve the message, not the message the framework: use it as a starting
structure and adapt it, rather than contorting the content to fit the template. And no
framework substitutes for a *sound message* — a well-structured presentation of a weak
or dishonest insight is still weak or dishonest. Frameworks organise a good message
effectively; they cannot rescue a bad one. The next lesson covers weaving the data
itself into the narrative.
"""

CONTENT["Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method"] = r"""
Integrating data and narrative
--------------------------------

A data presentation must *weave* the data into the story — neither burying the audience
in numbers nor making claims the data does not back. This lesson covers integrating data
with narrative through hypotheses and context, and a specific structured approach known
as the **McCandless Method**, giving a concrete technique for presenting a data-driven
point.

Hypotheses and context
------------------------

Two elements make data land in a presentation:

- **A hypothesis** — framing a finding as a claim the data tests and supports gives the
  audience something definite to grasp. "We believe the newest region is our growth
  engine — here is the data" is more compelling than an unframed pile of regional
  numbers. The hypothesis gives the data a *point*.
- **Context** — data means nothing without a frame of reference. A number needs a
  comparison (versus last year, versus target, versus other segments) to be
  interpretable; a trend needs its baseline; a result needs the circumstances that make
  it meaningful. Providing context is what turns a bare figure into an insight the
  audience can judge.

Together, a hypothesis (what we claim) and context (against what it means) make data
communicative rather than merely present.

The McCandless Method
----------------------

The McCandless Method, associated with data journalist David McCandless, is a structured
way to present a single data visualization or point, working through four elements in
order:

- **Introduce the visualization by name** — give the chart a clear, descriptive title
  that states what it shows, so the audience knows what they are looking at before
  interpreting it.
- **Anticipate the audience's questions** — address the questions the visualization
  naturally raises ("what am I looking at? what do these axes mean?") before they become
  confusion.
- **State the insight** — say clearly what the visualization *shows* — the pattern, the
  finding, the point. Do not make the audience infer it; state it.
- **Call out the supporting evidence, and tell the audience why it matters** — direct
  attention to the specific parts of the chart that support the insight, and connect it
  to what the audience cares about.

The method ensures a visualization is *presented*, not merely *displayed* — introduced,
clarified, interpreted, and connected to the audience's concerns.

Why this integration matters
------------------------------

Weaving data into narrative this way avoids the two failure modes of data presentation:
the *data dump* (showing numbers and charts without interpretation, leaving the audience
to make sense of them) and the *unsupported claim* (asserting conclusions without showing
the data that backs them). A hypothesis frames the point, context makes it interpretable,
and a method like McCandless's presents each visualization so the audience *understands*
it. The result is a presentation where data and narrative reinforce each other — the
story guides, and the data substantiates.

The caveat
------------

Framing data with a hypothesis is powerful and therefore hazardous: a hypothesis stated
too strongly can lead the audience to *see* support the data does not really provide, and
selectively presenting the context that flatters the hypothesis (while omitting context
that complicates it) is a subtle dishonesty. The discipline is to frame a hypothesis the
data *genuinely supports*, present the context *fairly* (including what complicates the
picture), and let the visualization show what it actually shows — using these techniques
to make a *true* point clear, not to make a shaky point persuasive. The honest-analysis
obligation governs how data is woven, not just whether it is present. The next lesson
turns to delivering the presentation.
"""

CONTENT["Presentation Skills for Data Analysts: Delivering Insights with Confidence"] = r"""
The delivery half
-------------------

A well-structured presentation with clear slides can still fall flat in the *delivery* —
and a confident, clear delivery can make even modest content land. **Presentation
skills** are the delivery half of presenting: voice, pace, presence, and composure. This
lesson covers delivering data insights with confidence, the human skills that carry the
prepared content.

The core delivery skills
--------------------------

Effective delivery rests on a handful of skills:

- **Clarity of speech** — speaking clearly and at a measured pace, so the audience can
  follow. Nervous presenters rush; deliberate pacing aids comprehension and projects
  composure.
- **Voice and emphasis** — varying tone and stressing key points, so the important
  things stand out audibly, as visual emphasis makes them stand out on a slide.
- **Presence and eye contact** — engaging the audience by looking at them (not at
  slides or notes), which builds connection and conveys confidence.
- **Managing the visualizations** — guiding the audience through each chart (as the
  McCandless Method structures), directing their attention rather than assuming they
  follow.
- **Composure** — staying calm and measured, especially when nervous or challenged (the
  Q&A lessons ahead).

These skills turn prepared content into a delivery that connects with the audience.

Confidence and how it is built
--------------------------------

Confidence in presenting comes less from personality than from *preparation and
practice*. Knowing your material deeply, having rehearsed the presentation, and
anticipating questions all produce genuine confidence — you are confident because you
are prepared, not despite nerves. The most reliable route to confident delivery is
thorough preparation: rehearse the presentation, know the data cold, and anticipate what
the audience will ask. Confidence is largely preparation made visible.

Managing nervousness
----------------------

Nervousness is normal, even for experienced presenters, and the goal is to *manage* it,
not eliminate it. Preparation reduces it; so do practical techniques — a measured pace
(slowing down counters the nervous rush), focusing on the message rather than on
yourself, and treating the presentation as helping the audience understand rather than as
a performance to be judged. Reframing from "I am being evaluated" to "I am helping them
understand" shifts the focus outward and eases the nerves.

The caveat
------------

Confident delivery is a tool that can be *misused* — a polished, confident presenter can
make a weak or wrong analysis sound authoritative, and confidence that outruns the actual
evidence misleads an audience into over-trusting the conclusion. Delivery skill must not
substitute for substance or overstate certainty: the confident presentation of an
uncertain finding should still convey that uncertainty, honestly. Confidence should
reflect genuine command of *sound* material, not paper over weak material with poise. The
analyst's obligation to truth applies to *how* findings are delivered as much as to the
findings themselves — deliver a well-founded message with earned confidence, and an
uncertain one with honest calibration. The next lesson collects the best practices of
polished presenting.
"""

CONTENT["Presenting Like a Pro: Best Practices for Data Analysts"] = r"""
The marks of a polished presenter
-----------------------------------

Beyond structure, slides, and delivery basics, a set of **best practices** distinguishes
a polished, professional data presenter. This lesson collects them — the habits that
mark someone who presents data credibly and effectively, consolidating the presentation
craft before the Q&A lessons.

The best practices
-------------------

Professional data presenters consistently do the following:

- **Know the audience** — tailor content, depth, and framing to who is in the room and
  what they need to decide (the audience-centered thread throughout).
- **Lead with the message** — state the key takeaway early and clearly, then support it,
  rather than making the audience wait for the point.
- **Tell a story** — structure the presentation as a narrative with an arc, not a list of
  disconnected findings (the storytelling stage).
- **Show, don't tell** — use visualizations to convey what is better seen, keeping slides
  visual and uncluttered (the slide-design lesson).
- **Practice thoroughly** — rehearse until the delivery is smooth and the timing is
  right; preparation is the foundation of both polish and confidence.
- **Anticipate questions** — prepare for what the audience will ask, so Q&A strengthens
  rather than undermines the presentation (the next lessons).
- **Be honest about limitations** — state the data's caveats and uncertainties, which
  *builds* credibility rather than weakening it.
- **Respect time** — keep to the allotted time and the audience's attention, cutting
  what does not serve the message.

Together these are what "presenting like a pro" means in practice — not flashiness, but
disciplined clarity, preparation, and honesty.

Why honesty is a best practice
--------------------------------

It is worth emphasising that *honesty about limitations* is a professional best practice,
not a weakness. A presenter who acknowledges what the data does *not* show, where the
uncertainty lies, and what the analysis could not address comes across as more
trustworthy, not less — because a claim of certainty invites skepticism while honest
calibration invites trust. Audiences, especially sophisticated ones, trust the analyst who
volunteers the caveats over the one who oversells. Honesty is thus not only ethical but
*effective* — it is how credibility is earned and kept, which is why it appears on every
professional's list.

The professional mindset
--------------------------

Underlying the practices is a mindset: the presentation exists to help the *audience*
understand and decide, and the presenter's job is to serve that as clearly and honestly
as possible. This outward focus — on the audience's understanding rather than the
presenter's performance — is what animates all the best practices, from tailoring to
honesty to respecting time. The professional presents *for the audience*, not *at* them.

The caveat
------------

Best practices are guidelines, not guarantees, and they can be over-applied or misapplied
— rigidly following every practice regardless of context, or polishing delivery while
neglecting the analysis underneath. The practices serve a sound, honest message delivered
to a specific audience; they do not substitute for having one, and they are adapted to the
situation rather than applied by rote. A technically flawless presentation of a weak or
dishonest analysis has missed the point entirely. The practices make a *good* presentation
excellent; the goodness — sound analysis, honest message, genuine service to the audience
— must be there first. The next lessons cover the part many presenters fear most: the Q&A.
"""


MINDMAP.update({
    "Using a Strategic Framework to Structure Data Presentations": [
        "Structuring a Persuasive Data Presentation: Turning Insights into Story",
        "Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method",
        "Designing Effective Data Presentation Slides: Structure, Visuals, and Professional Impact",
        "Presenting Like a Pro: Best Practices for Data Analysts",
    ],
    "Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method": [
        "Using a Strategic Framework to Structure Data Presentations",
        "Structuring a Persuasive Data Presentation: Turning Insights into Story",
        "Presentation Skills for Data Analysts: Delivering Insights with Confidence",
        "Data Storytelling: Giving Numbers a Clear and Convincing Voice",
    ],
    "Presentation Skills for Data Analysts: Delivering Insights with Confidence": [
        "Presenting Like a Pro: Best Practices for Data Analysts",
        "Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method",
        "Preparing for Q&A: Anticipating and Responding to Stakeholder Questions",
        "Structuring a Persuasive Data Presentation: Turning Insights into Story",
    ],
    "Presenting Like a Pro: Best Practices for Data Analysts": [
        "Presentation Skills for Data Analysts: Delivering Insights with Confidence",
        "Using a Strategic Framework to Structure Data Presentations",
        "Preparing for Q&A: Anticipating and Responding to Stakeholder Questions",
        "Handling Objections in Data Presentations: Responding with Confidence and Clarity",
    ],
})


# ======================================================================
# Section 6 — Data Visualization / present (close)  (viz 025-027)  -- SECTION 6 COMPLETE
# ======================================================================

GLOSS.update({
    "Preparing for Q&A: Anticipating and Responding to Stakeholder Questions":
        "readying for the questions a presentation will draw — anticipation as preparation",
    "Handling Objections in Data Presentations: Responding with Confidence and Clarity":
        "responding to challenges and pushback without defensiveness, honestly and calmly",
    "Q&A Best Practices: Answering Questions with Clarity and Confidence":
        "the habits of fielding questions well — listen, answer directly, stay honest",
})

CONTENT["Preparing for Q&A: Anticipating and Responding to Stakeholder Questions"] = r"""
The part you cannot script
----------------------------

A presentation's prepared portion ends; then comes the **question-and-answer**, the
part many presenters fear most because it cannot be fully scripted. Yet Q&A can be
prepared for, and doing so turns it from a threat into an opportunity to reinforce the
message. This lesson, closing the visualization section, covers anticipating and
responding to stakeholder questions.

Why Q&A matters
----------------

Q&A is not an afterthought — it is often where the *real* engagement happens. It is
where stakeholders test the analysis, surface their genuine concerns, and decide
whether to trust and act on the findings. Handled well, Q&A *strengthens* a
presentation: it demonstrates the depth behind the conclusions, addresses doubts
directly, and builds the confidence that leads to action. Handled poorly, it can
unravel an otherwise strong presentation. Because so much rides on it, preparing for
Q&A is as important as preparing the presentation itself.

Anticipating questions
-----------------------

The core of Q&A preparation is *anticipation* — predicting what will be asked and
readying answers:

- **Question the analysis yourself** — what are its weak points, assumptions, and
  limitations? Stakeholders will probe exactly these, so identify them first and
  prepare honest responses.
- **Consider the audience's concerns** — what does *this* audience care about, and what
  will they want to know? Their questions follow their interests and decisions.
- **Prepare for the obvious questions** — "how confident are you?", "what about this
  other factor?", "what do you recommend?", "where did this data come from?" — the
  predictable questions deserve prepared, solid answers.
- **Ready supporting detail** — have the backup data, the methodology, and the details
  ready (often in appendix slides) for questions that go deeper than the main
  presentation.

Anticipation converts most of Q&A from improvisation into *recall* of prepared
material — you have already thought through the likely questions.

Responding well
----------------

Beyond anticipation, responding well has a method: *listen* to the whole question
before answering, *answer directly* (address what was actually asked, concisely),
*support* the answer with evidence where useful, and *stay honest* — if you do not know,
say so rather than bluffing. A direct, honest, evidence-backed answer builds trust; a
evasive or padded one erodes it. The next lessons develop responding to objections and
the finer points of answering.

The caveat
------------

Preparation cannot anticipate *every* question, and the goal is not to script Q&A
completely but to be ready for the *likely* questions and composed enough to handle the
unexpected ones. Over-preparing rigid answers can even backfire — sounding rehearsed or
failing to actually address a question that differs from the one you prepared for. The
balance is thorough anticipation of the probable, plus the composure and honesty to
handle the rest genuinely: think through the weak points and obvious questions in
advance, and meet the unforeseen ones with the same directness and honesty rather than
panic. Readiness is having thought it through, not having memorised a script. The next
lesson addresses the hardest questions: objections.
"""

CONTENT["Handling Objections in Data Presentations: Responding with Confidence and Clarity"] = r"""
When the audience pushes back
-------------------------------

Some questions are not requests for information but *objections* — challenges to the
analysis, disagreements with the conclusions, or pushback on the recommendations.
**Handling objections** with confidence and clarity, rather than defensiveness, is a
distinct and crucial skill. This lesson covers responding to challenges in a data
presentation.

Understanding objections
--------------------------

Objections come in recognisable forms, and reading which one you face guides the
response:

- **Challenges to the data or method** — "how do you know the data is reliable?", "did
  you account for X?". These question the analysis's soundness, and the response is
  evidence: explain the method, the data quality, the checks performed (the
  verification and integrity work from earlier sections).
- **Disagreement with the conclusion** — the stakeholder reads the findings
  differently, or their experience conflicts with the data. This calls for engaging the
  substance, not dismissing their view.
- **Concerns about the recommendation** — accepting the analysis but questioning the
  proposed action, often for reasons (cost, feasibility, politics) outside the data.
- **Objections rooted elsewhere** — sometimes an objection reflects a concern the
  stakeholder has not stated directly; understanding the real concern behind the
  question is often the key to addressing it.

Identifying what an objection *really* challenges is the first step to responding well.

Responding to objections
--------------------------

Effective objection-handling follows principles that keep the exchange constructive:

- **Stay calm and non-defensive** — an objection is not an attack, and treating it as
  one (becoming defensive or dismissive) escalates and undermines credibility. Composure
  signals confidence in the work.
- **Listen and acknowledge** — hear the full objection and acknowledge its validity
  where it has some; dismissing a fair point loses the room. "That's a fair concern"
  before responding shows you take it seriously.
- **Respond with evidence and reason** — address the substance with the data, method, or
  reasoning that speaks to it, rather than mere assertion.
- **Concede what is true** — if an objection identifies a genuine limitation or a point
  you had not considered, acknowledge it honestly. Conceding a valid point *strengthens*
  credibility; defending the indefensible destroys it.
- **Find common ground** — where you disagree, identify what you *do* agree on, and
  reason from there.

The goal is a constructive exchange that addresses the concern honestly, not a debate to
be "won."

Confidence without defensiveness
----------------------------------

The balance the lesson's title names — *confidence and clarity* — is the key. Confidence
means standing behind sound work and responding substantively, not caving at the first
challenge; clarity means addressing the actual objection directly and understandably.
But confidence must not become *defensiveness* (treating every objection as an attack) or
*stubbornness* (refusing to concede valid points). The confident, clear response engages
the objection on its merits, concedes what is true, and defends what is sound — all
calmly. This is what maintains credibility under challenge.

The caveat
------------

The deepest risk in handling objections is letting the *desire to defend your conclusion*
override *honesty* — arguing away a valid objection, overstating your certainty to rebuff
a challenge, or "winning" the exchange at the cost of the truth. When an objection is
right, the honest and ultimately more credible response is to concede it, even at the
cost of your conclusion — because an analyst's lasting credibility rests on being
trustworthy, not on being never wrong. Handling objections well is not about deflecting
all challenges but about engaging them honestly, conceding the valid ones, and defending
only what the evidence genuinely supports. Defending a wrong conclusion skilfully is a
failure, not a success. The final lesson consolidates Q&A best practices.
"""

CONTENT["Q&A Best Practices: Answering Questions with Clarity and Confidence"] = r"""
Fielding questions well
-------------------------

Closing the visualization and communication section, this lesson consolidates the
**best practices for Q&A** — the habits that let an analyst field questions with clarity
and confidence, turning the unscripted exchange into a strength. It gathers the threads
of the previous lessons into a practical set of principles.

The Q&A best practices
------------------------

Analysts who handle Q&A well consistently do the following:

- **Listen fully** — hear the entire question before formulating an answer; answering a
  question the person did not ask (because you jumped ahead) frustrates and misses the
  point. Pause to understand first.
- **Clarify if needed** — if a question is unclear or ambiguous, ask what the person
  means rather than guessing; answering the wrong interpretation wastes the exchange.
- **Answer directly and concisely** — address what was actually asked, get to the point,
  and avoid rambling. A direct answer respects the audience and projects command.
- **Support with evidence** — back the answer with the data, method, or reasoning when
  useful, connecting to the analysis.
- **Be honest about what you don't know** — "I don't know, but I can find out" is a
  strong, credible answer; bluffing is transparent and destroys trust. Honesty about the
  limits of your knowledge is a best practice, not a weakness.
- **Stay composed and gracious** — remain calm, respectful, and non-defensive, even with
  hostile or difficult questions. Composure under pressure signals confidence in the
  work.
- **Bridge back to the message** — where appropriate, connect an answer back to the key
  message, using questions as opportunities to reinforce the main point.

Together these turn Q&A from a feared unknown into a controlled, credibility-building
part of the presentation.

Honesty as the throughline
----------------------------

The recurring theme across Q&A — and across the whole communication section — is
*honesty*. Answering truthfully, admitting what you do not know, conceding valid
objections, and not overstating certainty are the practices that build the trust on
which an analyst's influence depends. It is tempting, under the pressure of a question,
to bluff, overstate, or defend past the evidence — and it is exactly then that honesty
matters most, because the audience is testing not just the analysis but the analyst's
trustworthiness. The analyst who is honest under questioning earns credibility that a
slick but evasive one never does.

Q&A as opportunity
-------------------

Reframing Q&A from threat to *opportunity* is the lesson's final point. Questions reveal
what the audience cares about and did not fully grasp, giving you the chance to address
their real concerns, reinforce the message, and demonstrate the depth behind the work.
Approached this way — as a chance to engage rather than a gauntlet to survive — Q&A
becomes where a good presentation earns trust and turns findings into decisions. The
prepared, honest, composed presenter welcomes it.

The caveat
------------

Q&A best practices are principles, not a script, and real Q&A is unpredictable — the
practices guide behaviour but must be applied with judgement in the moment (when to be
brief versus thorough, when to bridge versus simply answer, how to read a difficult
questioner). And the practices serve honest communication of sound work; they are not
techniques for managing an audience past a weak analysis or an inconvenient truth. Fielded
with genuine honesty and command of solid material, Q&A strengthens a presentation;
deployed to deflect and spin, the same techniques become the manipulation the whole
course warns against. The practices are for communicating truth clearly under questioning,
not for surviving scrutiny of the indefensible.

This completes the Data Visualization section — and with it, the arc from raw data to
communicated insight. You have moved through visualization principles, building charts in
Tableau, telling data stories, and presenting and defending findings. The data is now not
only prepared, cleaned, and analysed but *communicated* — turned into understanding that
drives decisions. The next section turns to the tool that ties the whole workflow together
and automates it: the Python programming language.
"""


MINDMAP.update({
    "Preparing for Q&A: Anticipating and Responding to Stakeholder Questions": [
        "Presenting Like a Pro: Best Practices for Data Analysts",
        "Handling Objections in Data Presentations: Responding with Confidence and Clarity",
        "Q&A Best Practices: Answering Questions with Clarity and Confidence",
        "Presentation Skills for Data Analysts: Delivering Insights with Confidence",
    ],
    "Handling Objections in Data Presentations: Responding with Confidence and Clarity": [
        "Preparing for Q&A: Anticipating and Responding to Stakeholder Questions",
        "Q&A Best Practices: Answering Questions with Clarity and Confidence",
        "Presenting Like a Pro: Best Practices for Data Analysts",
        "Structuring a Persuasive Data Presentation: Turning Insights into Story",
    ],
    "Q&A Best Practices: Answering Questions with Clarity and Confidence": [
        "Preparing for Q&A: Anticipating and Responding to Stakeholder Questions",
        "Handling Objections in Data Presentations: Responding with Confidence and Clarity",
        "Data Creates Value Only When It Is Communicated",
        "Presenting Like a Pro: Best Practices for Data Analysts",
    ],
})


# ======================================================================
# Section 7 — Data Analysis with Python / Stage: basics  (python 001-004)
# ======================================================================

GLOSS.update({
    "Introduction to Python and Programming Fundamentals":
        "why analysts learn Python, and the programming ideas underneath it",
    "Python Fundamentals":
        "the core building blocks of Python — values, expressions, statements, and flow",
    "Jupyter Notebook and Coding Environments":
        "where analysts write and run Python — notebooks, IDEs, and the interactive workflow",
    "Object-Oriented Programming (OOP) in Python":
        "the objects-and-methods model that shapes Python, including pandas and numpy",
})

CONTENT["Introduction to Python and Programming Fundamentals"] = r"""
Why Python
-----------

Spreadsheets and SQL take an analyst far, but a *programming language* takes them
further — automating work, handling any data size, and expressing analyses too complex
for a query. **Python** is the language most data analysts learn, and this section
builds the skill of using it for data analysis. Opening the Python section, this lesson
introduces why Python matters and the programming fundamentals underneath it.

Why Python for data analysis
------------------------------

Python has become the dominant language in data analysis for concrete reasons:

- **Readable and approachable** — Python's syntax is clean and close to plain English,
  making it one of the easier languages to learn, which matters for analysts who are not
  primarily programmers.
- **Powerful data libraries** — Python's ecosystem includes purpose-built data tools:
  ``pandas`` for tabular data, ``numpy`` for numerical computing, ``matplotlib`` and
  others for visualization (the later lessons). These do the heavy lifting.
- **Automation and reproducibility** — a Python script runs the same analysis
  identically every time and can be automated, the reproducibility that manual
  spreadsheet work lacks (a theme since the cleaning section).
- **Scale and flexibility** — Python handles data sizes and analytical complexity
  beyond spreadsheets, and can do anything the other tools can plus much they cannot.

Python is where the analyst's workflow is *automated and scaled* — the tool that ties
preparation, analysis, and visualization into repeatable pipelines.

Programming fundamentals
--------------------------

Beneath Python lie ideas common to all programming, worth naming before the syntax:

- **A program is a sequence of instructions** — code tells the computer precisely what
  to do, step by step, executed in order.
- **Data and operations** — programs hold *data* (values: numbers, text) and perform
  *operations* on it (calculations, transformations, comparisons).
- **Variables** — named places to store data for reuse (the next lessons).
- **Control flow** — deciding *which* instructions run and *how many times* (conditions
  and loops, the control stage).
- **Functions** — named, reusable blocks of instructions (a later lesson) — the
  abstraction-and-reuse principle from the foundations, in code.

These concepts underlie every language; Python is one readable way to express them.

Python in the analyst's toolkit
----------------------------------

Python does not replace spreadsheets and SQL so much as *complete* the toolkit. The
tool-choice lessons apply: spreadsheets for quick visual work, SQL for querying
databases, and Python for automation, complex transformation, and anything that must
run repeatably at scale. In practice Python often *orchestrates* the others — pulling
data via SQL, transforming it, and producing visualizations — making it the glue of a
mature analytical workflow. It is the most powerful and most general of the analyst's
tools, and correspondingly the most involved to learn.

The caveat
------------

Python's power comes with a steeper learning curve than spreadsheets, and it is not the
right tool for *every* task — a quick look at a small dataset is faster in a spreadsheet
than in code, and forcing simple work into Python is over-engineering. Learning Python
is a genuine investment, and its payoff is in automation, scale, and complexity, not in
one-off simple tasks. The judgement from the tool-choice lessons holds: reach for
Python when its strengths (reproducibility, scale, complex logic, automation) are worth
the effort, and use the simpler tools when they suffice. Python expands what you *can*
do; it does not make itself the right choice for everything. The next lesson begins with
Python's fundamentals.
"""

CONTENT["Python Fundamentals"] = r"""
The building blocks
--------------------

With the motivation established, this lesson covers the **fundamentals of Python** — the
core building blocks from which every program is made. These basics (values,
expressions, statements, and how code runs) are the vocabulary the rest of the section
builds on, introduced here in Python's clean syntax.

Values and expressions
------------------------

Python works with *values* — the data a program manipulates:

.. code-block:: python

   42              # an integer
   3.14            # a float (decimal number)
   "hello"         # a string (text)
   True            # a boolean (True or False)

Values combine into *expressions* that Python evaluates to a result:

.. code-block:: python

   2 + 3           # evaluates to 5
   "data" + "!"    # evaluates to "data!" (string concatenation)
   10 > 5          # evaluates to True

An expression is anything Python can compute a value from — the basic unit of doing
something with data.

Statements and printing
-------------------------

A *statement* is an instruction Python executes. The ``print`` function displays a
value, the standard way to see a program's output:

.. code-block:: python

   print("Hello, data analysis!")
   print(2 + 3)                       # displays 5

Programs are sequences of statements executed top to bottom, in order — the "sequence of
instructions" from the previous lesson made concrete.

Variables: naming values
--------------------------

A *variable* stores a value under a name, so it can be reused and changed (a dedicated
lesson follows):

.. code-block:: python

   sales = 1000
   tax_rate = 0.08
   total = sales + sales * tax_rate
   print(total)                       # displays 1080.0

Assigning with ``=`` binds a name to a value; the name then stands for the value
wherever used. Variables are what let a program hold and manipulate data through
meaningful names rather than raw values.

Python's readability
----------------------

A defining feature of Python is *indentation as structure* — Python uses indentation
(spaces at the start of a line) to group code into blocks, where other languages use
braces. This enforces the visual clarity that makes Python readable, and it means
consistent indentation is not optional style but *required syntax*. This readability —
clean syntax, English-like keywords, meaningful indentation — is much of why Python
suits analysts, and why code written in it is comparatively easy to read and maintain.

The caveat
------------

Python's fundamentals are simple individually but combine into subtle behaviour, and a
few basics trip up beginners. Python is *case-sensitive* (``Sales`` and ``sales`` are
different names); indentation errors (inconsistent spaces) are a common frustration
precisely because indentation is meaningful; and the distinction between *values* and
the *variables* naming them matters as programs grow. These are learned by writing and
running code, not by reading about it — which is why the environment for doing so, the
next lesson's subject, matters. The fundamentals reward practice: type the examples, run
them, and change them to see what happens. The next lesson covers where to do that.
"""

CONTENT["Jupyter Notebook and Coding Environments"] = r"""
Where analysts write Python
-----------------------------

Code needs somewhere to be written and run, and the *environment* an analyst uses shapes
how they work. For data analysis, the **Jupyter Notebook** is the most common
environment, alongside other coding tools. This lesson covers where and how analysts
write Python, and why the notebook in particular suits data work.

What a coding environment provides
------------------------------------

An environment for writing and running code offers, at minimum, a place to write code,
a way to run it, and a way to see the results. The main options for data analysts:

- **Jupyter Notebook** — an interactive, cell-based environment (below), dominant in
  data analysis.
- **IDEs (Integrated Development Environments)** — full-featured tools like VS Code or
  PyCharm, better for larger programs and software development.
- **The interactive interpreter / scripts** — running Python directly or as ``.py``
  script files, for automation and production code.

Each suits different work; for exploratory data analysis, the notebook is usually the
starting point.

The Jupyter Notebook
---------------------

Jupyter Notebook is an interactive environment that organises code into **cells** —
blocks of code you run individually, seeing each cell's output immediately below it:

- **Cells run independently** — write code in a cell, run it, see the result, then write
  the next cell building on it. This *interactive, incremental* workflow suits
  exploration: try something, see what happens, adjust.
- **Output appears inline** — results, tables, and (crucially) *visualizations* display
  right below the cell that produced them, keeping code and results together.
- **Mixes code and narrative** — notebooks combine code cells with text (Markdown)
  cells, so an analysis can be documented alongside the code that produces it — code,
  results, and explanation in one document.

This cell-based, inline-output, documented style is why Jupyter dominates data
analysis: it matches how analysis is actually done — exploring incrementally, seeing
results immediately, and documenting the reasoning.

Why the environment matters for analysis
------------------------------------------

The notebook's fit for analysis is not incidental. Exploratory analysis is iterative
(the analysis-process theme) — you try a transformation, examine the result, try
another — and the notebook's run-a-cell-see-the-result loop matches that rhythm exactly.
Inline visualizations mean charts appear where you make them; mixed narrative means the
analysis is self-documenting (the documentation discipline). The environment shapes the
work, and Jupyter shapes it toward the interactive, visual, documented style good
analysis wants.

The caveat
------------

The notebook's strengths carry matching weaknesses. Because cells can be run in *any
order*, a notebook can reach a state that is not reproducible from top to bottom — you
ran cells out of sequence, and re-running fresh gives different results, a subtle
reproducibility trap exactly opposite to the reproducibility Python promises. Notebooks
also suit exploration better than production: automated, scheduled, or large software is
usually better as ``.py`` scripts. The disciplines that address this — periodically
restarting and running the notebook top to bottom to confirm it reproduces, and moving
mature code into scripts — matter because the notebook's flexibility can quietly
undermine the reproducibility that is Python's point. Use the notebook for its
interactive strengths, but guard the reproducibility it can erode. The next lesson turns
to a model underlying Python itself: objects.
"""

CONTENT["Object-Oriented Programming (OOP) in Python"] = r"""
Everything is an object
-------------------------

Python is built on a model called **object-oriented programming**, and although an
analyst need not write elaborate object-oriented code, understanding the model *matters*
— because everything in Python is an object, including the data structures and library
objects analysis relies on. This lesson introduces OOP as the model that shapes how
Python (and pandas, and numpy) work.

Objects and methods
---------------------

In Python, a value is not just data — it is an **object** that bundles data together
with *methods* (functions that belong to it and act on it). You call a method with dot
notation:

.. code-block:: python

   name = "data analysis"
   name.upper()            # "DATA ANALYSIS" — upper() is a string method
   name.title()            # "Data Analysis"

   numbers = [3, 1, 2]
   numbers.sort()          # sorts the list in place — sort() is a list method
   numbers.append(4)       # adds to the list

``name.upper()`` calls the ``upper`` method *on* the string object ``name``. This
dot-notation — object, then method — is pervasive in Python, and recognising it as "call
this object's method" is key to reading Python code.

Classes: the blueprint
------------------------

Objects are created from **classes** — blueprints defining what data an object holds and
what methods it has. A string is an object of the ``str`` class; a list, of the ``list``
class; and the ``DataFrame`` you will use for data is an object of the ``DataFrame``
class from pandas. The class defines the type; each object is an *instance* of its class.
Analysts mostly *use* objects of existing classes (strings, lists, DataFrames) rather
than writing their own, but knowing that an object's available methods come from its
class explains why a DataFrame has different methods than a string.

Why OOP matters for analysis
------------------------------

The payoff is practical: the data tools of this section *are* objects, and using them is
calling their methods. A pandas ``DataFrame`` is an object with methods like
``.head()``, ``.groupby()``, and ``.mean()``:

.. code-block:: python

   import pandas as pd
   df = pd.DataFrame({"region": ["N", "S"], "sales": [100, 200]})
   df.head()               # a DataFrame method — shows the first rows
   df["sales"].mean()      # method on the column — computes the mean

Understanding OOP means this syntax reads naturally: ``df.groupby(...)`` is "call the
DataFrame's groupby method," exactly the object-dot-method pattern. The whole of pandas
and numpy usage is calling methods on objects, so the OOP model is the grammar of Python
data analysis.

The caveat
------------

For analysts, the risk is the *opposite* of neglect — over-investing in OOP theory that
data analysis rarely requires. Writing custom classes, inheritance hierarchies, and
elaborate object-oriented designs is software-engineering work most analysts seldom
need; the analyst's use of OOP is mostly *understanding* it well enough to use library
objects fluently, not *building* object-oriented systems. Learn the model to the depth
that makes ``df.groupby(...).mean()`` legible and the library methods sensible — the
objects-and-methods grammar — and leave the deeper OOP design to when a genuine need
arises. The next lessons return to the hands-on basics: variables, naming, and types.
"""


MINDMAP.update({
    "Introduction to Python and Programming Fundamentals": [
        "Choosing the Right Tool in Data Analysis",
        "Python Fundamentals",
        "Jupyter Notebook and Coding Environments",
        "Spreadsheets vs. SQL",
    ],
    "Python Fundamentals": [
        "Introduction to Python and Programming Fundamentals",
        "Variables in Python",
        "Data Types and Type Conversion in Python",
        "Jupyter Notebook and Coding Environments",
    ],
    "Jupyter Notebook and Coding Environments": [
        "Python Fundamentals",
        "Introduction to Python and Programming Fundamentals",
        "Object-Oriented Programming (OOP) in Python",
        "Comments, Algorithms, and Docstrings in Python",
    ],
    "Object-Oriented Programming (OOP) in Python": [
        "Python Fundamentals",
        "Functions in Python",
        "Variables in Python",
        "Introduction to Python and Programming Fundamentals",
    ],
})


# ======================================================================
# Section 7 — Data Analysis with Python / basics (cont.)  (python 005-008)
# ======================================================================

GLOSS.update({
    "Variables in Python":
        "named containers for values — assigning, reassigning, and using them",
    "Naming Conventions and Restrictions in Python":
        "the rules for valid names and the conventions for good ones (PEP 8)",
    "Data Types and Type Conversion in Python":
        "the kinds of values Python holds, and converting between them safely",
    "Functions in Python":
        "named, reusable blocks of code — defining, calling, arguments, and returns",
})

CONTENT["Variables in Python"] = r"""
Naming values for reuse
-------------------------

A program that could not *remember* values would be useless; **variables** are how
Python remembers. A variable is a named container for a value, letting you store data,
refer to it by name, and change it as the program runs. This lesson covers variables in
Python — assigning, reassigning, and using them — the foundation of holding data in code.

Assignment
-----------

A variable is created by *assigning* a value to a name with ``=``:

.. code-block:: python

   sales = 1000
   region = "North"
   tax_rate = 0.08

The name on the left is bound to the value on the right; afterwards, the name stands for
the value:

.. code-block:: python

   total = sales + sales * tax_rate
   print(total)              # 1080.0
   print(region)             # North

The ``=`` here is *assignment*, not mathematical equality — it means "let this name
refer to this value," a distinction worth keeping clear.

Reassignment
-------------

A variable's value can *change* — reassigning binds the name to a new value:

.. code-block:: python

   count = 5
   count = count + 1         # count is now 6
   count += 1                # shorthand for count = count + 1; now 7

The last form, ``+=``, is an *augmented assignment* — a common shorthand for updating a
variable based on its current value (``-=``, ``*=``, ``/=`` work similarly). Reassignment
is what lets a variable track a changing value as a program runs.

Why variables matter
----------------------

Variables serve the same purposes as naming anything: *reuse* (compute a value once,
use it many times), *clarity* (a well-named variable documents what a value means —
``tax_rate`` is clearer than ``0.08`` scattered through code), and *changeability*
(update a value in one place and everything using it updates). This is the
abstraction-and-naming principle from the foundations, in code: meaningful names for
values make programs readable and maintainable, exactly as meaningful column names make
data readable.

The caveat
------------

Variables have subtleties that catch beginners. A variable must be *assigned before it
is used* — referencing a name Python has not seen raises an error. Reassignment means a
variable's value depends on *when* you look (its value is whatever was last assigned),
which matters especially in notebooks where cells run out of order — a variable can hold
a surprising value if cells ran in an unexpected sequence. And Python variables are
*case-sensitive* (``Sales`` ≠ ``sales``), a frequent source of "undefined name" errors.
These are learned by writing code and reading the errors, which say precisely what went
wrong. The next lesson covers naming variables well.
"""

CONTENT["Naming Conventions and Restrictions in Python"] = r"""
Naming well and legally
-------------------------

Variable names are subject to *rules* (what Python allows) and *conventions* (what good
practice recommends), and both matter — the rules because breaking them is an error, the
conventions because they make code readable. This lesson covers Python's naming
**restrictions** and **conventions**, applying the "good names" principle to code.

The restrictions (what Python requires)
-----------------------------------------

Python enforces rules for valid names:

- **Allowed characters** — names may contain letters, digits, and underscores
  (``_``), and must *start* with a letter or underscore, not a digit. ``sales_2024`` is
  valid; ``2024_sales`` is not.
- **No spaces** — names cannot contain spaces (use underscores: ``tax_rate``, not
  ``tax rate``).
- **Case-sensitive** — ``sales``, ``Sales``, and ``SALES`` are three different names.
- **No reserved keywords** — Python's keywords (``if``, ``for``, ``class``, ``def``,
  ``True``, and so on) cannot be used as names, since they have special meaning.

Breaking these rules is a *syntax error* — the code will not run — so the restrictions
are non-negotiable.

The conventions (what good practice recommends)
-------------------------------------------------

Beyond legality, Python has widely-followed conventions (codified in the style guide
**PEP 8**) that make code readable:

- **snake_case for variables and functions** — lowercase words joined by underscores:
  ``total_sales``, ``tax_rate``, ``customer_count``. This is the standard Python style.
- **Descriptive names** — ``total_sales`` over ``ts`` or ``x``; a name should say what
  the value *is*. Clarity over brevity (the foundations' principle).
- **UPPER_CASE for constants** — values meant not to change (``TAX_RATE = 0.08``) by
  convention use all caps.
- **Avoid confusing names** — not single letters like ``l`` (looks like ``1``), not
  names that shadow built-ins (``list``, ``sum``, ``type``), which causes subtle bugs.

Conventions are not enforced by Python — code violating them still runs — but following
them makes code readable to yourself and others, which is why they are near-universal.

Why naming matters
-------------------

Good names are documentation that cannot go out of date. ``monthly_revenue`` needs no
comment to explain it; ``mr`` or ``x`` does. As programs grow, the difference between
well-named and cryptically-named code is the difference between maintainable and
baffling — the exact parallel to well-named versus cryptic data columns. Naming is a
small, constant discipline with a large cumulative payoff in readability, which is why it
is worth doing deliberately from the start.

The caveat
------------

The two failure modes are opposite. *Under-naming* — cryptic abbreviations, single
letters, meaningless names — makes code unreadable. But *over-naming* — names so long
they clutter (``the_total_amount_of_sales_for_the_northern_region_this_year``) — harms
readability too, and can tempt breaking the no-spaces rule. The balance is names
*descriptive enough to be clear, concise enough to read* — the same judgement as naming
anything. And a name that shadows a built-in (``sum = 5`` overwriting the ``sum``
function) is legal but causes confusing bugs, so convention rightly warns against it.
Name for the reader, within the rules. The next lesson covers the types of values names
can hold.
"""

CONTENT["Data Types and Type Conversion in Python"] = r"""
The kinds of values
--------------------

Every value in Python has a **type** — the kind of data it is — and the type determines
what can be done with the value. Understanding Python's core **data types** and how to
**convert** between them is essential, because type mismatches are among the most common
sources of errors, and type conversion is a routine part of preparing data (echoing the
``CAST`` work from SQL).

The core data types
---------------------

Python's fundamental types for analysis:

- ``int`` — whole numbers: ``42``, ``-7``, ``1000``.
- ``float`` — decimal numbers: ``3.14``, ``0.08``, ``-2.5``.
- ``str`` — text (strings): ``"North"``, ``"data"``.
- ``bool`` — boolean truth values: ``True``, ``False``.
- ``list`` — an ordered collection: ``[1, 2, 3]`` (the structures stage covers these).
- ``dict`` — key-value pairs: ``{"region": "North"}`` (also the structures stage).

The ``type()`` function reports a value's type:

.. code-block:: python

   type(42)          # <class 'int'>
   type(3.14)        # <class 'float'>
   type("North")     # <class 'str'>

Knowing a value's type explains what operations it supports — you can add ``int`` s, and
you can concatenate ``str`` s, but adding an ``int`` to a ``str`` is an error.

Type conversion
----------------

Converting a value from one type to another uses the type's conversion function — the
Python counterpart of SQL's ``CAST``:

.. code-block:: python

   int("42")         # 42     — string to integer
   float("3.14")     # 3.14   — string to float
   str(1000)         # "1000" — number to string
   int(3.9)          # 3      — float to int (truncates, does not round)

This matters constantly in data work: data read from files often arrives as *strings*
(the import type-trap, familiar from earlier sections), and must be converted to numbers
before arithmetic — exactly the text-to-number cleaning done with ``VALUE`` in
spreadsheets and ``CAST`` in SQL, now in Python.

Why types matter
-----------------

Type errors are pervasive and often puzzling until understood. ``"5" + "3"`` gives
``"53"`` (string concatenation), not ``8``, because the values are strings, not numbers;
``"5" + 3`` raises an error (cannot add string and int). Understanding types explains
these behaviours and the fixes (convert the strings to numbers first). Since data
frequently arrives with the wrong types, type awareness and conversion are a routine,
essential part of Python data analysis — the same "get the types right before computing"
discipline that ran through cleaning and analysis.

The caveat
------------

Type conversion can *fail* or behave surprisingly, exactly as ``CAST`` did. Converting a
non-numeric string to a number (``int("hello")``) raises an error, and a single bad
value can break a conversion over a whole column — which, as in SQL, usefully *forces*
you to confront the non-conforming data. ``int()`` on a float *truncates* rather than
rounds (``int(3.9)`` is ``3``, not ``4``), a common surprise. And converting can lose
information (float to int drops the decimal). As always, understand what a conversion
does to the data, verify the result, and handle the values that will not convert. The
next lesson turns to functions — reusable blocks of code.
"""

CONTENT["Functions in Python"] = r"""
Reusable blocks of code
-------------------------

As programs grow, the same operations recur — and copying code to repeat them is the
duplication the foundations warned against. **Functions** are Python's tool for reuse: a
named, reusable block of code that performs a task, defined once and called wherever
needed. This lesson covers defining and calling functions — the core of writing clean,
non-repetitive Python.

Defining and calling a function
---------------------------------

A function is defined with ``def``, given a name, parameters, and a body, and *called*
by name:

.. code-block:: python

   def add_tax(amount, rate):
       total = amount + amount * rate
       return total

   result = add_tax(1000, 0.08)      # call it; result is 1080.0
   print(add_tax(50, 0.10))          # 55.0

``def add_tax(amount, rate):`` defines a function taking two *parameters*; the indented
body is what it does; ``return`` gives back a result. Calling ``add_tax(1000, 0.08)``
runs the body with those *arguments* and evaluates to the returned value. The same
function serves any amount and rate — written once, used many times.

Parameters, arguments, and return
-----------------------------------

The pieces of a function:

- **Parameters** — the named inputs in the definition (``amount``, ``rate``) — the
  "parameterized function" idea, letting one function handle many inputs.
- **Arguments** — the actual values passed in a call (``1000``, ``0.08``).
- **Return value** — what the function gives back via ``return``, usable by the caller. A
  function without ``return`` gives back ``None``.
- **Default arguments** — parameters can have defaults used when an argument is omitted:

  .. code-block:: python

     def add_tax(amount, rate=0.08):    # rate defaults to 0.08
         return amount + amount * rate

     add_tax(1000)                      # uses default rate: 1080.0
     add_tax(1000, 0.10)               # overrides: 1100.0

Defaults are the "explicit defaults" principle from the foundations, in code.

Why functions matter
----------------------

Functions are the primary tool for the foundations' *reuse and abstraction* principles.
They eliminate duplication (write the logic once, call it many times), improve
readability (a well-named function like ``add_tax`` documents what a block of code does),
enable testing (a function can be verified in isolation), and localise change (fix the
logic in one place). A program built from well-named functions is modular, readable, and
maintainable — the same virtues good structure gives anything, achieved in code through
functions.

The caveat
------------

Functions can be misused in opposite directions. *Too little* use — repeating code
instead of writing a function — produces the duplication that makes programs
unmaintainable (change the logic and you must find every copy). *Too much or wrong*
abstraction — functions that do too many things, or are split so finely that following
the logic means jumping among many tiny functions — harms readability in the other
direction. The balance is the single-responsibility idea: a function should do *one
well-defined thing*, be named for it, and be neither a sprawling catch-all nor a
needless fragment. And functions should ideally be *pure* where practical — depending
only on their inputs and returning a result, without hidden side effects — which makes
them predictable and testable, the controlled-side-effects principle in code. The next
lessons cover writing clean, well-documented Python.
"""


MINDMAP.update({
    "Variables in Python": [
        "Python Fundamentals",
        "Naming Conventions and Restrictions in Python",
        "Data Types and Type Conversion in Python",
        "Object-Oriented Programming (OOP) in Python",
    ],
    "Naming Conventions and Restrictions in Python": [
        "Variables in Python",
        "Data Types and Type Conversion in Python",
        "Code Reusability, Modularity, and Clean Code in Python",
        "Python Fundamentals",
    ],
    "Data Types and Type Conversion in Python": [
        "Variables in Python",
        "Understanding Data Types and Data Formats",
        "Using CAST to Clean and Format Data in SQL",
        "Functions in Python",
    ],
    "Functions in Python": [
        "Variables in Python",
        "Code Reusability, Modularity, and Clean Code in Python",
        "Comments, Algorithms, and Docstrings in Python",
        "Object-Oriented Programming (OOP) in Python",
    ],
})


# ======================================================================
# Section 7 — Data Analysis with Python / basics (close) + control (open)  (python 009-012)
# NOTE: docstrings lesson uses r'''...''' delimiter — its code block contains """ docstrings
# ======================================================================

GLOSS.update({
    "Code Reusability, Modularity, and Clean Code in Python":
        "writing Python that is reusable, modular, and clean — the engineering virtues in code",
    "Comments, Algorithms, and Docstrings in Python":
        "documenting code — comments, docstrings, and thinking in algorithms",
    "Boolean Data, Comparators, and Logical Operators in Python":
        "True/False values and the comparisons and logic that produce them",
    "Branching and Conditional Statements in Python":
        "making code decide — if, elif, and else direct which code runs",
})

CONTENT["Code Reusability, Modularity, and Clean Code in Python"] = r"""
Writing code that lasts
-------------------------

Code that merely *works* is not enough; code that is **reusable, modular, and clean**
lasts — it can be understood, maintained, and built upon. These qualities, drawn straight
from the foundations' engineering principles, apply directly to Python, and this lesson
covers writing code that embodies them, closing the basics stage.

Reusability
------------

**Reusable** code is written once and used many times, rather than copied. Functions
(the previous lesson) are the primary tool: a well-designed function captures a piece of
logic that any part of a program can call. The test of reusability is the
duplication-elimination principle — if you find yourself copying code, that code should
become a function. Reusable code means a change to the logic happens in *one* place,
which is both less work and less error-prone than editing many copies.

Modularity
-----------

**Modular** code is organised into distinct, self-contained pieces — functions, and (as
programs grow) *modules* (separate files) — each responsible for one thing. The
single-responsibility principle from the foundations applies: each function or module
does one well-defined job, and the pieces combine into the whole. Modularity makes code
easier to understand (each piece is small and focused), test (verify pieces
independently), and change (modify one piece without disturbing others) — the
break-into-components principle, in code.

Clean code
-----------

**Clean** code is readable and clear — code written for humans to understand, not just
for the computer to run. Its marks are the ones this course has stressed throughout:

- **Meaningful names** — variables and functions named for what they are and do (the
  naming lesson).
- **Small, focused functions** — each doing one thing, named for it.
- **Consistent style** — following conventions (PEP 8) so code looks familiar.
- **Clarity over cleverness** — a straightforward solution over a clever, cryptic one
  (the foundations' principle, stated for code).
- **Appropriate documentation** — comments and docstrings where they aid understanding
  (the next lesson).

Clean code is a discipline of writing for the *next reader* — often your future self.

Why these matter
-----------------

Reusability, modularity, and cleanliness are what separate code that can *evolve* from
code that becomes a liability. Analysis code is rarely written once and discarded — it is
rerun, adapted, and extended, often by someone other than its author. Code with these
qualities supports that; code without them (duplicated, tangled, cryptic) resists it,
accumulating the technical debt the foundations warned against. The small disciplines —
factor out duplication, keep pieces focused, name and structure for clarity — compound
into code that remains workable as it grows.

The caveat
------------

These principles can be *over*-applied, the over-engineering the foundations cautioned
against. Factoring every two-line snippet into a function, splitting a simple script into
many modules, or adding abstraction for flexibility never needed makes code *harder* to
follow, not easier — the indirection costs more than the duplication it removes. The
judgement is proportionate: apply reusability and modularity where they genuinely aid a
program of real size and complexity, and keep a simple script simple. Clean code is not
maximally abstracted code; it is code that is as clear and simple as the problem allows,
which for a small analysis may be a single straightforward script. Match the engineering
to the scale. The next lesson covers documenting code.
"""

CONTENT["Comments, Algorithms, and Docstrings in Python"] = r'''
Documenting code
-----------------

Code expresses *what* it does; documentation explains *why* and *how*, and thinking in
**algorithms** shapes code before it is written. This lesson covers **comments**,
**docstrings**, and algorithmic thinking — the practices that make Python code
understandable and well-planned, completing the basics stage.

Comments
---------

A **comment** is text in code that Python ignores, written for human readers. In Python,
comments start with ``#``:

.. code-block:: python

   # Convert the raw price strings to numbers before summing
   prices = [float(p) for p in raw_prices]
   total = sum(prices)              # sum() adds all elements

Good comments explain *why* — the reasoning, the intent, the non-obvious — not *what* the
code plainly does. ``x = x + 1  # add one to x`` is a useless comment (the code says
that); ``# compensate for the zero-based index`` is a useful one (it explains the reason).
The documentation principle from the foundations — explain why, not just what — applies
directly.

Docstrings
-----------

A **docstring** is a string at the start of a function (or module or class) that
documents it, accessible programmatically and by tools. Python data work commonly uses
the **NumPy documentation style**, which structures a docstring into sections:

.. code-block:: python

   def add_tax(amount, rate=0.08):
       """Return the amount with tax added.

       Parameters
       ----------
       amount : float
           The pre-tax amount.
       rate : float, optional
           The tax rate as a decimal (default 0.08).

       Returns
       -------
       float
           The amount including tax.
       """
       return amount + amount * rate

The docstring states what the function does, its **Parameters**, and what it
**Returns** — so a reader (or a documentation generator) understands the function without
reading its body. For reusable functions, a docstring is the interface's documentation,
and the NumPyDoc section order (Parameters, Returns, and, as needed, Raises, Notes,
Examples) is a widely-followed convention in the data ecosystem.

Thinking in algorithms
-----------------------

An **algorithm** is a step-by-step procedure for solving a problem — and thinking
algorithmically means *planning the steps before writing code*. Before coding, an analyst
works out the logic: what steps, in what order, transform the input into the desired
output. Writing the algorithm first (even as plain-language steps or "pseudocode") clarifies
the approach before the syntax, catching logic problems early — the big-picture-first
discipline from the foundations, applied to code. Code is the *expression* of an algorithm;
getting the algorithm right first makes the code straightforward.

Why documentation matters
---------------------------

Documentation is what makes code understandable to its future readers — including its
author months later. A comment explaining a non-obvious choice, a docstring describing a
function's interface, and code that follows a clear algorithm together mean the *reasoning*
survives, not just the instructions. This is the reproducibility-and-maintainability
theme in code: undocumented code works until it must be understood or changed, at which
point its opacity becomes costly. Documentation is the small investment that keeps code
workable over time.

The caveat
------------

Documentation has the same failure modes as any: too little leaves code opaque, but too
much — comments restating obvious code, docstrings on trivial one-line functions,
narration of every step — clutters and, worse, *drifts out of sync* with the code, so a
comment says one thing while the code does another, which misleads more than no comment.
The discipline is documentation that is *accurate, useful, and maintained*: comment the
*why* and the non-obvious, docstring the *interfaces* that will be reused, keep it truthful
to the code, and skip narrating what the code plainly says. Accurate, purposeful
documentation helps; stale or redundant documentation harms. This completes the Python
basics; the next stage turns to control flow — making code decide and repeat.
'''

CONTENT["Boolean Data, Comparators, and Logical Operators in Python"] = r"""
True, false, and decisions
----------------------------

Programs make *decisions*, and every decision rests on a question with a true-or-false
answer. **Boolean** values (``True`` and ``False``), the **comparators** that produce
them, and the **logical operators** that combine them are the foundation of all control
flow. Opening the control stage, this lesson covers the boolean logic that the branching
and looping lessons build on.

Boolean values
---------------

A **boolean** is one of exactly two values, ``True`` or ``False`` — the ``bool`` type
from the types lesson. Booleans represent the answer to a yes/no question, and they are
what decisions are made from:

.. code-block:: python

   is_valid = True
   has_errors = False

Comparators
-----------

**Comparison operators** compare two values and produce a boolean:

.. code-block:: python

   5 > 3           # True   (greater than)
   5 < 3           # False  (less than)
   5 == 5          # True   (equal to — note double equals)
   5 != 3          # True   (not equal to)
   5 >= 5          # True   (greater than or equal)
   3 <= 5          # True   (less than or equal)

The critical one to note is ``==`` (equality comparison), *two* equals signs — distinct
from ``=`` (assignment), *one* equals sign. Confusing them is a classic error: ``=``
assigns a value, ``==`` asks whether two values are equal. Comparators are how a program
turns data into the true/false answers decisions need.

Logical operators
------------------

**Logical operators** combine booleans into compound conditions:

.. code-block:: python

   (age >= 18) and (age < 65)      # True only if BOTH are true
   (region == "N") or (region == "S")   # True if EITHER is true
   not is_valid                    # inverts: True becomes False

- ``and`` — true only if *both* operands are true.
- ``or`` — true if *at least one* operand is true.
- ``not`` — inverts a boolean.

These are the same logical combinations as SQL's ``AND``/``OR``/``NOT`` in ``WHERE``
clauses (the analysis section) — the identical logic, now in Python. Compound conditions
let a program ask complex questions ("is the customer an adult *and* in an eligible
region?") as a single boolean.

Why boolean logic matters
---------------------------

Boolean logic is the foundation of *control flow* — every branch and loop the next
lessons cover is directed by a boolean condition. It is also the basis of *filtering*
data (the pandas lessons will filter rows by boolean conditions, exactly as SQL's
``WHERE`` and spreadsheet filters did). Mastering comparators and logical operators is
therefore mastering the mechanism behind decisions, loops, and data filtering alike — a
small piece of logic that underlies a large share of programming.

The caveat
------------

Boolean logic has precise rules that produce surprises when misread. The ``=`` versus
``==`` confusion is the commonest (and Python catches many but not all such mistakes);
operator precedence means compound conditions sometimes need *parentheses* to group them
as intended (``a and b or c`` may not mean what you expect — parenthesise for clarity);
and comparisons involving different types or ``None`` can behave unexpectedly. Writing
compound conditions with explicit parentheses, and testing that a condition is true
exactly when it should be, guards against logic that looks right but is not — the
check-your-logic discipline applied to booleans. The next lesson uses these conditions to
make code branch.
"""

CONTENT["Branching and Conditional Statements in Python"] = r"""
Making code decide
--------------------

With boolean conditions in hand, a program can **branch** — run different code depending
on whether a condition is true. **Conditional statements** (``if``, ``elif``, ``else``)
are how Python decides, directing which instructions execute. This lesson covers
branching, the first form of control flow.

The if statement
-----------------

An ``if`` statement runs a block of code *only if* a condition is true:

.. code-block:: python

   if sales > 1000:
       print("High sales")

The condition (``sales > 1000``) is a boolean; if it is ``True``, the indented block
runs; if ``False``, it is skipped. The indentation defines the block — the code that
belongs to the ``if`` — which is why Python's meaningful indentation matters here
especially.

else and elif
--------------

``else`` provides an alternative when the condition is false, and ``elif`` (else-if)
checks further conditions:

.. code-block:: python

   if sales > 1000:
       category = "High"
   elif sales > 500:
       category = "Medium"
   else:
       category = "Low"

Python checks each condition in order: if ``sales > 1000``, category is "High" and the
rest is skipped; otherwise if ``sales > 500``, "Medium"; otherwise "Low". Only the *first*
matching branch runs. This ``if``/``elif``/``else`` chain is the direct counterpart of
SQL's ``CASE`` expression (the analysis section) — the same "different result for
different conditions" logic, now controlling which code executes.

Branching on compound conditions
----------------------------------

Branches use the full boolean logic from the previous lesson:

.. code-block:: python

   if (age >= 18) and (region in ("N", "S")):
       status = "eligible"
   else:
       status = "ineligible"

Compound conditions let a branch depend on several factors at once — the logical operators
combining into a single decision. This is how programs express real decision rules, which
usually involve multiple criteria.

Why branching matters
-----------------------

Branching is what makes a program *responsive* rather than fixed — it does different
things in different situations, which is essential to any non-trivial logic. In data work,
branching drives categorisation (bucketing values, exactly the ``CASE`` work), validation
(handling valid versus invalid data differently), and conditional processing throughout.
It is one of the two pillars of control flow (with loops, next), and the foundation of
code that adapts to its data.

The caveat
------------

Branching logic is precise and can be subtly wrong. The order of ``elif`` conditions
*matters* — since only the first match runs, mis-ordered conditions can make a branch
unreachable (checking ``sales > 500`` before ``sales > 1000`` would catch high sales in
the wrong branch); conditions must be *exhaustive* where every case should be handled (an
``else`` catching what the explicit conditions miss); and deeply *nested* branches (ifs
within ifs within ifs) grow hard to follow and are often better restructured. Test that
each branch runs exactly when intended, especially the boundaries between conditions — the
same edge-case discipline from the foundations, applied to decisions. The next lessons
cover the other pillar of control flow: loops.
"""


MINDMAP.update({
    "Code Reusability, Modularity, and Clean Code in Python": [
        "Functions in Python",
        "Comments, Algorithms, and Docstrings in Python",
        "Naming Conventions and Restrictions in Python",
        "Data Types and Type Conversion in Python",
    ],
    "Comments, Algorithms, and Docstrings in Python": [
        "Code Reusability, Modularity, and Clean Code in Python",
        "Functions in Python",
        "Branching and Conditional Statements in Python",
        "Naming Conventions and Restrictions in Python",
    ],
    "Boolean Data, Comparators, and Logical Operators in Python": [
        "Branching and Conditional Statements in Python",
        "Data Types and Type Conversion in Python",
        "Data Validation in Spreadsheets",
        "While Loops and Iteration in Python",
    ],
    "Branching and Conditional Statements in Python": [
        "Boolean Data, Comparators, and Logical Operators in Python",
        "While Loops and Iteration in Python",
        "For Loops in Python",
        "Functions in Python",
    ],
})


# ======================================================================
# Section 7 — Data Analysis with Python / control (close) + structures (open)  (python 013-016)
# NOTE: python 013 key is the exact frozen typo "...in Python" (missing final n, in title AND slug) fixed manually
# ======================================================================

GLOSS.update({
    "While Loops and Iteration in Python":
        "repeating code while a condition holds — the while loop and iteration basics",
    "For Loops in Python":
        "repeating code over the items of a collection — the for loop",
    "range() Function and Loop Control in Python":
        "generating number sequences and controlling loops with break and continue",
    "Strings in Python":
        "working with text — the string type, its operations, and its immutability",
})

CONTENT["While Loops and Iteration in Python"] = r"""
Repeating while a condition holds
-----------------------------------

The second pillar of control flow (after branching) is **iteration** — repeating code —
and the **while loop** is its most basic form: run a block *repeatedly as long as a
condition remains true*. Opening the discussion of loops, this lesson covers the while
loop and the idea of iteration that the for-loop and data-processing lessons build on.

The while loop
--------------

A ``while`` loop repeats its block while its condition is true:

.. code-block:: python

   count = 1
   while count <= 5:
       print(count)
       count += 1          # crucial: move toward ending the loop

Python checks the condition (``count <= 5``); if true, it runs the block, then checks
again — repeating until the condition becomes false. Here it prints 1 through 5, then
stops when ``count`` reaches 6. The condition uses the boolean logic from the earlier
lesson, and the loop continues until that condition fails.

The crucial detail: making progress
--------------------------------------

The single most important thing about a while loop is that *something in the loop must
eventually make the condition false* — otherwise it runs forever. In the example,
``count += 1`` is what moves toward termination; without it, ``count`` stays 1, the
condition stays true, and the loop never ends (an **infinite loop**). Every while loop
must contain something that progresses toward its exit condition, and forgetting this is
the classic while-loop bug.

Iteration as a concept
------------------------

**Iteration** — doing something repeatedly — is fundamental to programming and
especially to data work, where the same operation applies to many items (every row, every
value, every file). The while loop expresses iteration in its most general form: repeat
until a condition says stop. This general form suits situations where you do not know in
advance how many repetitions are needed — keep going until something changes (until the
data runs out, until a target is reached, until input stops). For iterating a *known*
collection, the for loop (next lesson) is usually cleaner, but the while loop handles the
open-ended cases.

The caveat
------------

The while loop's power to repeat is exactly its danger: the **infinite loop**. A
condition that never becomes false — because the loop forgets to make progress, or the
progress never reaches the exit — hangs the program indefinitely. This is the signature
while-loop failure, and guarding against it means ensuring every while loop has a clear
path to its exit condition and that something in the body advances toward it. When you do
know how many times to repeat, or you are iterating a collection, the for loop is safer
because it *cannot* loop infinitely over a finite collection. Reach for while only when
the open-ended condition genuinely calls for it, and always verify the loop can end. The
next lesson covers the for loop.
"""

CONTENT["For Loops in Python"] = r"""
Repeating over a collection
-----------------------------

When you need to do something *for each item* in a collection, the **for loop** is the
natural tool — it iterates over the items of a sequence, running its block once per item.
This is the loop analysts use most, because data work is largely "do this to every row,
value, or record." This lesson covers the for loop, the workhorse of iteration.

The for loop
------------

A ``for`` loop iterates over the items of a collection, binding each to a variable in
turn:

.. code-block:: python

   regions = ["North", "South", "East", "West"]
   for region in regions:
       print(region)

Python takes each item of ``regions`` in order, assigns it to ``region``, and runs the
block — printing all four region names. The loop variable (``region``) holds the current
item on each pass. Unlike the while loop, the for loop *automatically* stops when the
collection is exhausted — no manual progress-tracking, and no risk of an infinite loop
over a finite collection.

Iterating and accumulating
----------------------------

A common pattern combines a for loop with a variable that *accumulates* a result across
iterations:

.. code-block:: python

   sales = [100, 250, 175, 300]
   total = 0
   for amount in sales:
       total += amount        # accumulate
   print(total)               # 825

The ``total`` starts at zero and grows by each amount as the loop visits it — computing a
sum by iteration. This accumulate-across-a-loop pattern (summing, counting, collecting,
building) is one of the most useful in programming, and it is how manual aggregation is
expressed in code (though pandas, later, does it far more concisely).

For loops over different collections
--------------------------------------

For loops iterate any *iterable* — lists, strings (character by character), dictionaries
(the structures stage), and more:

.. code-block:: python

   for char in "data":        # iterates characters: d, a, t, a
       print(char)

This generality makes the for loop the standard way to process collections of any kind —
whatever the data, "for each item, do something" is a for loop.

The caveat
------------

For loops are safer than while loops (they cannot loop infinitely over a finite
collection), but they have their own pitfalls. Modifying a collection *while* iterating
over it causes subtle bugs (the collection changes underfoot) and should be avoided —
build a new collection instead. And a deeper point looms for data work: explicit Python
for loops over large datasets are *slow* compared to the vectorised operations of
``numpy`` and ``pandas`` (the libraries stage), which do the same work far faster without
an explicit loop. Loops are essential to understand and correct for general programming,
but for large-scale data the idiom shifts to vectorised operations — a for loop over a
million-row dataset is usually the wrong tool. Learn loops thoroughly, and later learn
when *not* to loop. The next lesson covers generating sequences and controlling loops.
"""

CONTENT["range() Function and Loop Control in Python"] = r"""
Sequences and finer loop control
----------------------------------

Two tools refine looping: the **range()** function, which generates sequences of numbers
to loop over, and **loop control** statements (``break`` and ``continue``), which give
finer control over how a loop proceeds. This lesson covers both, completing the control
stage's treatment of iteration.

The range() function
---------------------

``range()`` generates a sequence of numbers, most often used to repeat something a set
number of times or to loop over numeric indices:

.. code-block:: python

   for i in range(5):         # 0, 1, 2, 3, 4
       print(i)

   for i in range(1, 6):      # 1, 2, 3, 4, 5 (start, stop)
       print(i)

   for i in range(0, 10, 2):  # 0, 2, 4, 6, 8 (start, stop, step)
       print(i)

``range(n)`` produces 0 up to (but *not including*) ``n``; ``range(start, stop)`` starts
elsewhere; ``range(start, stop, step)`` steps by an interval. The *exclusive* upper bound
— ``range(5)`` stops at 4, not 5 — is the detail to remember, echoing zero-based indexing.
``range()`` is how a for loop repeats a fixed number of times or walks a numeric sequence.

Loop control: break and continue
----------------------------------

Two statements alter a loop's normal flow:

.. code-block:: python

   for amount in sales:
       if amount < 0:
           continue           # skip this iteration, go to the next
       if amount > 10000:
           break              # exit the loop entirely
       process(amount)

- ``break`` — *exits* the loop immediately, skipping any remaining iterations. Useful for
  stopping once a condition is met (found what you sought, hit a limit).
- ``continue`` — *skips the rest of the current iteration* and moves to the next. Useful
  for skipping items that should not be processed (invalid values, ones to ignore).

These give a loop finer control than "process every item to the end" — stop early, or skip
selectively.

Why these matter
-----------------

``range()`` and loop control complete the looping toolkit. ``range()`` handles
count-based and index-based repetition (repeat *n* times, iterate positions), complementing
the for loop's collection iteration. ``break`` and ``continue`` let a loop respond to
conditions — stopping when done, skipping what should be skipped — making loops precise
rather than all-or-nothing. Together with the while and for loops, they cover the iteration
patterns programming requires.

The caveat
------------

Each tool has a trap. ``range()``'s *exclusive upper bound* is a persistent
off-by-one source — ``range(1, 5)`` gives 1,2,3,4, not 1–5 — so getting the bounds right
requires care (the edge-case discipline). ``break`` and ``continue`` used heavily can make
a loop's flow *hard to follow* — a loop with several breaks and continues scattered through
it can be as tangled as deeply nested branches, so use them where they clarify (a clean
early exit) rather than as a substitute for well-structured loop logic. And the earlier
caution stands: for large data, ``range()``-driven index loops are usually slower and less
clear than the vectorised operations of pandas/numpy. Clear, correct loops first; know
their limits for scale. The next stage turns to Python's data structures, starting with
strings.
"""

CONTENT["Strings in Python"] = r"""
Working with text
------------------

Data is full of text — names, categories, codes, addresses — and Python's **string** type
is how text is represented and manipulated. Opening the data-structures stage, this lesson
covers strings in Python: creating them, their operations, and the crucial property of
immutability. It extends the string work from the spreadsheet and SQL sections into
Python.

Creating and combining strings
--------------------------------

A **string** is text, written in single or double quotes:

.. code-block:: python

   name = "North Region"
   code = 'NR-001'

Strings combine and repeat with operators:

.. code-block:: python

   greeting = "Hello, " + name       # concatenation: "Hello, North Region"
   line = "-" * 20                    # repetition: 20 dashes

The ``+`` concatenates strings (as in the spreadsheet's ``&`` and SQL's ``CONCAT``), and
``*`` repeats a string — the basic ways to build text.

String methods
--------------

Strings are objects (the OOP lesson) with many useful methods, mirroring the string
functions from earlier sections:

.. code-block:: python

   text = "  North Region  "
   text.strip()              # "North Region" — remove surrounding whitespace (like TRIM)
   text.upper()              # "  NORTH REGION  " — uppercase
   text.lower()              # lowercase
   text.replace("North", "South")   # substitute (like SUBSTITUTE / REPLACE)
   "NR-001".split("-")       # ["NR", "001"] — split on a delimiter
   len("North")              # 5 — length (like LEN)

These are the same cleaning and manipulation operations from the spreadsheet (``TRIM``,
``UPPER``, ``SUBSTITUTE``) and SQL (``TRIM``, ``UPPER``, ``REPLACE``, ``SUBSTR``) — now as
Python string methods, called with dot notation on the string object.

String immutability
--------------------

A crucial property: strings in Python are **immutable** — once created, a string cannot be
changed in place. String methods do not modify the original; they *return a new string*:

.. code-block:: python

   text = "north"
   text.upper()              # returns "NORTH", but...
   print(text)               # still "north" — unchanged!
   text = text.upper()       # to keep the result, reassign
   print(text)               # now "NORTH"

This catches many beginners: calling ``text.upper()`` does not change ``text``; you must
*assign* the result back. Immutability means string operations produce new strings, and
using the result requires capturing it — a fundamental and frequently-forgotten point.

Why strings matter
------------------

Text manipulation is constant in data work — cleaning categories, parsing codes, extracting
parts, formatting output — and Python's string methods are the tools for all of it, more
flexible than their spreadsheet and SQL counterparts. Because so much real data is text
(or arrives as text needing conversion, the type lesson), fluency with strings is
essential to Python data analysis. The following lessons go deeper into indexing,
slicing, and formatting strings.

The caveat
------------

String immutability is the pitfall to internalise: the single commonest string mistake is
calling a method and expecting the original to change — ``text.strip()`` on its own does
nothing lasting; you must write ``text = text.strip()``. Every string "modification" is
really "create a new string and (usually) reassign." Beyond that, strings carry the
encoding and special-character subtleties of all text (the Unicode considerations), and
splitting or extracting assumes a structure that real text may not consistently have (the
defensive-extraction point from the spreadsheet strings lesson applies). Capture method
results, and handle text's irregularity. The next lesson covers reaching into strings by
position: indexing and slicing.
"""


MINDMAP.update({
    "While Loops and Iteration in Python": [
        "Branching and Conditional Statements in Python",
        "For Loops in Python",
        "range() Function and Loop Control in Python",
        "Boolean Data, Comparators, and Logical Operators in Python",
    ],
    "For Loops in Python": [
        "While Loops and Iteration in Python",
        "range() Function and Loop Control in Python",
        "Data Types vs Data Structures & Introduction to Lists",
        "Branching and Conditional Statements in Python",
    ],
    "range() Function and Loop Control in Python": [
        "For Loops in Python",
        "While Loops and Iteration in Python",
        "Advanced Use of Loops, Lists, Tuples & List Comprehension",
        "Strings in Python",
    ],
    "Strings in Python": [
        "String Indexing and Slicing in Python",
        "String Formatting with .format() in Python",
        "Data Types and Type Conversion in Python",
        "Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)",
    ],
})


# ======================================================================
# Section 7 — Data Analysis with Python / structures (cont.)  (python 017-020)
# ======================================================================

GLOSS.update({
    "String Indexing and Slicing in Python":
        "reaching into strings by position — single characters and substrings",
    "String Formatting with .format() in Python":
        "building strings from values cleanly — the .format() method and f-strings",
    "Data Types vs Data Structures & Introduction to Lists":
        "the difference between a value's type and a structure that holds many, and the list",
    "Modifying Lists in Python":
        "changing lists in place — adding, removing, and updating elements",
})

CONTENT["String Indexing and Slicing in Python"] = r"""
Reaching into strings by position
-----------------------------------

A string is a *sequence* of characters, and Python lets you reach into it by position —
**indexing** to get a single character, **slicing** to get a substring. These are the
Python counterparts of the spreadsheet's ``LEFT``/``RIGHT``/``MID`` and SQL's ``SUBSTR``,
and they are essential for parsing and extracting from text. This lesson covers indexing
and slicing.

Indexing: single characters
-----------------------------

Each character in a string has an **index** — its position, counting from **zero**:

.. code-block:: python

   text = "North"
   #       01234
   text[0]           # "N" — first character (index 0)
   text[1]           # "o" — second character
   text[-1]          # "h" — last character (negative counts from the end)
   text[-2]          # "t" — second to last

The first character is at index ``0`` (not 1) — **zero-based indexing**, a fundamental and
frequently-tripped-over convention. Negative indices count from the end (``-1`` is the
last), a convenient Python feature for reaching the end without knowing the length.

Slicing: substrings
---------------------

**Slicing** extracts a range of characters with ``[start:stop]``:

.. code-block:: python

   text = "North Region"
   text[0:5]         # "North" — characters 0 through 4
   text[6:]          # "Region" — from index 6 to the end
   text[:5]          # "North" — from the start to index 4
   text[-6:]         # "Region" — the last six characters

The slice ``[start:stop]`` includes ``start`` but *excludes* ``stop`` — ``text[0:5]`` is
characters 0,1,2,3,4, not 5 — the same exclusive-upper-bound convention as ``range()``.
Omitting ``start`` means "from the beginning"; omitting ``stop`` means "to the end." Slicing
is how you extract a substring by position — the flexible text-extraction tool.

Indexing/slicing versus the earlier tools
-------------------------------------------

These operations mirror the string extraction from earlier sections exactly: ``text[0:5]``
is the spreadsheet's ``LEFT(text, 5)`` and SQL's ``SUBSTR(text, 1, 5)``; ``text[-3:]`` is
``RIGHT(text, 3)``. The concept — extract characters by position — is identical; Python's
``[start:stop]`` syntax is simply another expression of it, and one that generalises to all
sequences (lists too, as the list lessons show).

The caveat
------------

Two traps recur. **Zero-based indexing** — the first character is index 0 — means positions
are always one less than the "counting" number, a persistent source of off-by-one errors;
and the **exclusive stop** in slicing (``[0:5]`` stops at 4) compounds this. Getting an
index or slice boundary wrong extracts the wrong characters, often silently. The other trap
is indexing *past the end* of a string (``text[100]`` on a short string), which raises an
error — though slicing past the end is forgiving (it just stops at the end). As with all
position-based extraction, the discipline is care with the boundaries and awareness that
positions count from zero. The next lesson covers building strings from values.
"""

CONTENT["String Formatting with .format() in Python"] = r"""
Building strings from values
------------------------------

Analysts constantly build strings that *incorporate values* — a label with a number, a
message with a name, a formatted report line. Python's **string formatting** does this
cleanly, and this lesson covers the ``.format()`` method and the modern **f-string**, the
tools for inserting values into text without clumsy concatenation.

Why not just concatenate
--------------------------

You *can* build strings with ``+``, but it gets awkward, especially with non-string values:

.. code-block:: python

   region = "North"
   sales = 1000
   # clumsy: requires converting the number, easy to mangle spacing
   msg = "Region " + region + " had " + str(sales) + " in sales"

The conversions (``str(sales)``) and spacing make this error-prone and hard to read.
Formatting solves it.

The .format() method
---------------------

``.format()`` inserts values into *placeholders* (``{}``) in a template string:

.. code-block:: python

   msg = "Region {} had {} in sales".format(region, sales)
   # "Region North had 1000 in sales"

   # placeholders can be numbered or named for clarity:
   msg = "Region {0} had {1} in sales".format(region, sales)
   msg = "Region {r} had {s} in sales".format(r=region, s=sales)

Each ``{}`` is filled, in order, by the arguments to ``.format()`` — which handles the
type conversion automatically (the number becomes text without an explicit ``str()``). This
is cleaner and clearer than concatenation.

f-strings: the modern way
--------------------------

Modern Python offers **f-strings** — the most concise formatting, marking the string with
``f`` and putting expressions directly in the braces:

.. code-block:: python

   msg = f"Region {region} had {sales} in sales"
   # "Region North had 1000 in sales"

   total = 1234.5678
   msg = f"Total: {total:.2f}"     # "Total: 1234.57" — formatted to 2 decimals

The f-string embeds the variable (or any expression) right in the braces — the most
readable option, now the common Python idiom. It also supports *format specifiers*
(``:.2f`` for two decimal places), which control how numbers display — decimal places,
thousands separators, percentages — the presentation formatting from earlier sections,
in code.

Why formatting matters
-----------------------

Producing readable, correctly-formatted text output is constant in data work — labels for
charts, messages, report lines, formatted numbers. Formatting (especially f-strings) makes
this clean and reliable, handling conversions and controlling number display without the
fragility of manual concatenation. It is a small skill used constantly, and f-strings in
particular are worth adopting as the default.

The caveat
------------

Formatting is straightforward but has minor pitfalls. Forgetting the ``f`` prefix on an
f-string leaves the braces as *literal text* (``"{region}"`` appears verbatim rather than
the value) — a common confusion. Format specifiers (``:.2f`` and the like) have their own
small syntax to learn, and misapplying one can misformat output. And formatting controls
*display*, not the underlying value — ``f"{total:.2f}"`` shows two decimals but does not
round the stored ``total`` (the display-versus-value distinction from the spreadsheet
formatting lesson). Use formatting for clean output, remembering it shapes appearance, not
data. The next lesson turns to the first structure for holding many values: the list.
"""

CONTENT["Data Types vs Data Structures & Introduction to Lists"] = r"""
From single values to collections
-----------------------------------

So far, most values have been *single* — one number, one string. But data is usually
*many* values together, and holding collections requires **data structures**. This lesson
draws the distinction between a **data type** (the kind of a single value) and a **data
structure** (an organised collection of values), and introduces the most fundamental
structure: the **list**.

Data types versus data structures
------------------------------------

The distinction is foundational:

- A **data type** describes a *single* value's kind — ``int``, ``float``, ``str``,
  ``bool`` (the types lesson). It answers "what kind of value is this?"
- A **data structure** *organises multiple* values into a collection with a particular
  arrangement and behaviour — lists, tuples, dictionaries, sets (this stage). It answers
  "how are these many values held together?"

A single sales figure is a value of type ``float``; a *collection* of sales figures is a
data structure (a list). Types classify individual values; structures organise many. Both
matter: you choose a type for each value and a structure for how values are grouped.

The list
---------

A **list** is an ordered, changeable collection of values, written in square brackets:

.. code-block:: python

   sales = [100, 250, 175, 300]
   regions = ["North", "South", "East", "West"]
   mixed = [1, "two", 3.0, True]        # lists can hold mixed types

A list holds items *in order*, each accessible by index (the same zero-based indexing and
slicing as strings):

.. code-block:: python

   sales[0]          # 100 — first item
   sales[-1]         # 300 — last item
   sales[1:3]        # [250, 175] — a slice (a sub-list)
   len(sales)        # 4 — number of items

Lists are the workhorse structure for holding sequences of data — a column of values, a
series of records, any ordered collection.

Why lists matter
----------------

Lists are the most-used Python data structure, and the foundation for much data work. They
hold the collections that loops iterate over (the for-loop lesson), the sequences that get
transformed and aggregated, and conceptually they underlie the columns and series of
``pandas`` (the libraries stage — a DataFrame column is list-like). Understanding lists —
ordered, indexed, changeable collections — is understanding the basic shape of "many
values together" that all of data analysis works with.

The caveat
------------

Lists' flexibility invites a few missteps. They can hold *mixed types*, which is
occasionally useful but often a sign of disorganised data — a list meant to hold sales
figures should hold numbers, not a stray string, or later operations break. The zero-based
indexing and exclusive-stop slicing carry over from strings, with the same off-by-one
hazards. And a subtle point the next lessons develop: lists are **mutable** (changeable in
place), which is powerful but means a list can be modified unexpectedly if shared — the
opposite of strings' immutability, and a source of surprising bugs. Keep lists
type-consistent, mind the indexing, and be aware that lists change in place. The next lesson
covers modifying them.
"""

CONTENT["Modifying Lists in Python"] = r"""
Changing lists in place
-------------------------

Unlike strings, lists are **mutable** — they can be *changed in place* after creation:
items added, removed, or updated. This mutability makes lists the flexible, dynamic
structure they are, and this lesson covers the operations that modify them — the everyday
tools for building and updating collections.

Adding elements
---------------

Several methods add to a list:

.. code-block:: python

   sales = [100, 250]
   sales.append(300)         # add one item to the end: [100, 250, 300]
   sales.insert(0, 50)       # insert at a position: [50, 100, 250, 300]
   sales.extend([400, 500])  # add multiple items: [50, 100, 250, 300, 400, 500]

``append`` adds a single item to the end (the most common); ``insert`` places an item at a
given index; ``extend`` appends all items of another list. These grow a list as data
arrives — the accumulate pattern from the for-loop lesson often uses ``append``.

Removing elements
-----------------

Methods remove items:

.. code-block:: python

   sales.remove(50)          # remove the first matching value
   popped = sales.pop()      # remove and return the last item
   popped = sales.pop(0)     # remove and return the item at an index
   del sales[0]              # delete the item at an index

``remove`` deletes by *value* (the first match); ``pop`` deletes by *position* and returns
the removed item; ``del`` deletes by position. These shrink a list as items are consumed
or filtered.

Updating and other operations
-------------------------------

Items are updated by assigning to an index, and lists have further useful methods:

.. code-block:: python

   sales[0] = 999            # update the item at index 0
   sales.sort()              # sort the list in place (ascending)
   sales.reverse()           # reverse the order in place
   count = sales.count(250)  # count occurrences of a value

Assigning to ``sales[0]`` changes that element; ``sort`` and ``reverse`` reorder the list
*in place* (modifying the original, not returning a new list). These operations make lists
dynamic — reorderable, updatable collections.

Mutability: the key property
------------------------------

The defining feature is that these operations change the list *in place* — unlike string
methods, which return new strings. ``sales.append(300)`` modifies ``sales`` directly (no
reassignment needed); ``sales.sort()`` reorders ``sales`` itself. This in-place mutability
is what makes lists efficient for building and updating collections, and it is the direct
contrast to strings' immutability from the earlier lesson — a distinction worth holding
clearly, because it changes how you use each.

The caveat
------------

Mutability is powerful and *hazardous*, in ways that catch even experienced programmers.
Because a list is changed in place, if two variables refer to the *same* list (``b = a``
makes ``b`` another name for ``a``'s list, not a copy), modifying one changes the other —
the "shared reference" surprise, a classic source of baffling bugs. To get an independent
copy, you must explicitly copy the list (``b = a.copy()``). Also, in-place methods like
``sort()`` return ``None``, not the sorted list, so ``sales = sales.sort()`` mistakenly sets
``sales`` to ``None`` — the opposite mistake to strings (where you *must* reassign). The
disciplines: copy a list when you need an independent one, and remember in-place methods
modify rather than return. The next lessons cover tuples and further structures.
"""


MINDMAP.update({
    "String Indexing and Slicing in Python": [
        "Strings in Python",
        "String Formatting with .format() in Python",
        "Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)",
        "Data Types vs Data Structures & Introduction to Lists",
    ],
    "String Formatting with .format() in Python": [
        "String Indexing and Slicing in Python",
        "Strings in Python",
        "Data Types and Type Conversion in Python",
        "Data Types vs Data Structures & Introduction to Lists",
    ],
    "Data Types vs Data Structures & Introduction to Lists": [
        "Modifying Lists in Python",
        "Tuples in Python",
        "Data Types and Type Conversion in Python",
        "For Loops in Python",
    ],
    "Modifying Lists in Python": [
        "Data Types vs Data Structures & Introduction to Lists",
        "Advanced Use of Loops, Lists, Tuples & List Comprehension",
        "Tuples in Python",
        "Strings in Python",
    ],
})


# ======================================================================
# Section 7 — Data Analysis with Python / structures (cont.)  (python 021-024)
# ======================================================================

GLOSS.update({
    "Tuples in Python":
        "ordered, immutable collections — like lists that cannot change",
    "Advanced Use of Loops, Lists, Tuples & List Comprehension":
        "combining loops and structures, and the concise list comprehension idiom",
    "Dictionaries in Python":
        "key-value collections — looking values up by a meaningful key",
    "Advanced Dictionary Usage in Python":
        "iterating, nesting, and safely accessing dictionaries for real data",
})

CONTENT["Tuples in Python"] = r"""
Ordered and unchangeable
--------------------------

A **tuple** is like a list — an ordered collection of values — but with one crucial
difference: it is **immutable**, unchangeable after creation. Tuples serve where a
collection should be fixed, and understanding them (and why immutability is sometimes
wanted) rounds out the sequence structures. This lesson covers tuples.

Creating and using tuples
---------------------------

A tuple is written with parentheses (or just commas):

.. code-block:: python

   point = (3, 5)
   rgb = (255, 128, 0)
   record = ("North", 1000, 2024)     # a mixed tuple

Tuples are accessed exactly like lists — by index, with slicing:

.. code-block:: python

   point[0]          # 3
   record[1]         # 1000
   record[-1]        # 2024
   len(record)       # 3

Everything about *reading* a tuple mirrors a list; the difference is entirely in
*changing* it.

Immutability
------------

A tuple *cannot be changed* after creation — no adding, removing, or updating elements:

.. code-block:: python

   point = (3, 5)
   point[0] = 10     # ERROR — tuples do not support item assignment

This immutability is the tuple's defining property, and it is the same immutability strings
have. Where a list is a *changeable* ordered collection, a tuple is a *fixed* one — the two
are otherwise similar.

Why use a tuple
----------------

If tuples are just unchangeable lists, why use them? Immutability is a *feature* in the
right situations:

- **Fixed data that should not change** — coordinates, RGB colours, a fixed record — where
  accidental modification would be a bug. The immutability *protects* the data.
- **Meaning and intent** — using a tuple signals "this collection is fixed," documenting
  intent to readers.
- **Dictionary keys** — tuples can serve as dictionary keys (the next lessons) where lists
  cannot, precisely because they are immutable and stable.
- **Multiple return values** — functions often return several values as a tuple
  (``return x, y``), a common Python idiom.

The choice between list and tuple is the choice between *changeable* and *fixed*: use a
list when the collection will change, a tuple when it should not.

The caveat
------------

The tuple-versus-list choice is easy to get wrong in either direction: using a tuple for
data that *does* need to change forces awkward workarounds (you cannot modify it), while
using a list for data that should be *fixed* forgoes the protection immutability gives. The
guidance is intent-based — will this collection change during its life? Changeable → list;
fixed → tuple. And a subtle trap: a tuple's *immutability is shallow* — a tuple cannot be
reassigned, but if it *contains* a mutable object (a list inside a tuple), that inner object
can still change. For the flat collections of typical data work this rarely bites, but it is
worth knowing that immutability applies to the tuple's own structure, not necessarily to
everything within it. The next lesson combines the structures with loops and introduces list
comprehension.
"""

CONTENT["Advanced Use of Loops, Lists, Tuples & List Comprehension"] = r"""
Combining structures and loops, concisely
-------------------------------------------

With lists, tuples, and loops in hand, this lesson covers using them *together* more
powerfully — iterating structures in richer ways — and introduces **list comprehension**, a
concise Python idiom for building lists that experienced Python programmers use constantly.
It marks the transition from basic structure use to fluent, idiomatic Python.

Richer iteration
----------------

Python offers cleaner ways to iterate structures than a bare index loop:

.. code-block:: python

   sales = [100, 250, 175]

   for i, amount in enumerate(sales):     # index AND value together
       print(i, amount)

   regions = ["N", "S", "E"]
   for region, amount in zip(regions, sales):   # iterate two lists in parallel
       print(region, amount)

``enumerate`` gives both the index and the item (cleaner than tracking an index manually);
``zip`` iterates several collections in lockstep (pairing regions with sales). These make
common iteration patterns readable — and ``zip`` pairs naturally with tuple unpacking
(``for region, amount in ...`` unpacks each pair).

List comprehension
------------------

**List comprehension** builds a list concisely in a single expression, replacing a
build-with-a-loop pattern:

.. code-block:: python

   # the loop way:
   doubled = []
   for x in sales:
       doubled.append(x * 2)

   # the comprehension way — same result, one line:
   doubled = [x * 2 for x in sales]           # [200, 500, 350]

The comprehension ``[expression for item in collection]`` reads as "the expression, for
each item" — building a new list by transforming each element. It can include a *condition*
to filter:

.. code-block:: python

   large = [x for x in sales if x > 150]      # [250, 175] — only items over 150

``[x for x in sales if x > 150]`` keeps only items meeting the condition — transformation
and filtering in one concise expression.

Why comprehensions matter
---------------------------

List comprehensions are idiomatic Python — the natural, readable way to build a list by
transforming or filtering another, replacing the more verbose loop-and-append. They express
"make a new list from this one" in a single clear line, and recognising and using them is a
mark of Python fluency. The pattern also connects forward: it is conceptually the same
element-wise transformation and boolean filtering that ``numpy`` and ``pandas`` do
*vectorised* (the libraries stage), so comprehensions bridge explicit loops and the
vectorised idioms ahead.

The caveat
------------

List comprehensions are powerful and can be *overused*. A simple transformation or filter
is clearer as a comprehension than a loop; but a comprehension with multiple conditions,
nested loops, or complex logic crammed into one line becomes *harder* to read than the
equivalent loop — the clarity-over-cleverness principle warns against the dense,
show-off comprehension. The guidance: use a comprehension when it is *more* readable (a
single clear transformation or filter), and fall back to an explicit loop when the logic is
complex enough that a comprehension would obscure it. Concise is good only when it is also
clear. The next lesson turns to a different structure: the dictionary.
"""

CONTENT["Dictionaries in Python"] = r"""
Looking values up by key
--------------------------

Lists and tuples hold values in *order*, accessed by position. But often you want to look a
value up by a *meaningful key* — a customer's name, a product code, a field label — rather
than a numeric position. The **dictionary** is Python's key-value structure, and it is one
of the most important for data work. This lesson covers dictionaries.

What a dictionary is
---------------------

A **dictionary** stores **key-value pairs** — each value is associated with a key that
identifies it, written with curly braces:

.. code-block:: python

   customer = {
       "name": "Jane Smith",
       "region": "North",
       "sales": 1000,
   }

Values are looked up *by key*, not position:

.. code-block:: python

   customer["name"]          # "Jane Smith"
   customer["sales"]         # 1000

Where a list answers "what is at position 2?", a dictionary answers "what is the value for
'name'?" — access by meaningful key rather than numeric index. This makes dictionaries ideal
for representing *records* with named fields.

Modifying dictionaries
-----------------------

Dictionaries are mutable — pairs can be added, changed, and removed:

.. code-block:: python

   customer["email"] = "jane@example.com"    # add a new key-value pair
   customer["sales"] = 1200                    # update an existing value
   del customer["region"]                      # remove a pair

Assigning to a key either adds it (if new) or updates it (if it exists); ``del`` removes a
pair. Dictionaries grow and change like lists, but keyed rather than ordered.

Why dictionaries matter
------------------------

Dictionaries are fundamental to data work for several reasons. They represent *records*
naturally — a row of data as field-name-to-value pairs (``{"name": ..., "sales": ...}``),
which is exactly how structured data is often held. They enable *fast lookup* by key (far
faster than searching a list). And they are the structure behind much of Python's data
ecosystem — JSON data is dictionaries, pandas DataFrames can be built from them, and
configuration and mappings are dictionaries. Understanding key-value access is understanding
a core pattern of representing and retrieving structured data.

The caveat
------------

Dictionaries have specific rules and pitfalls. **Keys must be unique** — assigning to an
existing key *overwrites* its value rather than adding a second, so duplicate keys silently
lose data. **Keys must be immutable** — strings, numbers, and tuples can be keys, but lists
cannot (their mutability would break the dictionary's lookup), which is one reason tuples
exist. And the classic error: **accessing a key that does not exist** (``customer["phone"]``
when there is no phone) raises a ``KeyError`` and stops the program — a frequent bug when
data may be missing a field. The next lesson covers handling this and other advanced
dictionary usage safely. Keys unique and immutable, and access defensively.
"""

CONTENT["Advanced Dictionary Usage in Python"] = r"""
Dictionaries for real data
----------------------------

Basic dictionaries store and retrieve by key; *real* data work needs more — iterating over a
dictionary's contents, handling missing keys safely, and nesting dictionaries for structured
data. This lesson covers advanced dictionary usage, the techniques that make dictionaries
practical for actual data, closing the core structures.

Iterating dictionaries
-----------------------

Dictionaries are iterated by keys, values, or both:

.. code-block:: python

   customer = {"name": "Jane", "region": "North", "sales": 1000}

   for key in customer:                 # iterate keys
       print(key)

   for key, value in customer.items():  # iterate key-value pairs
       print(key, value)

   customer.keys()          # the keys
   customer.values()        # the values
   customer.items()         # the key-value pairs

``.items()`` is the common way to loop over a dictionary's contents — each iteration
unpacking a key and its value (tuple unpacking again). This is how you process every field of
a record, or every entry of a mapping.

Safe access with .get()
-------------------------

The ``KeyError`` from accessing a missing key is avoided with ``.get()``, which returns a
default instead of erroring:

.. code-block:: python

   customer["phone"]              # KeyError if 'phone' is missing — stops the program
   customer.get("phone")          # returns None if missing — safe
   customer.get("phone", "N/A")   # returns "N/A" if missing — safe with a default

``.get(key, default)`` is the *safe* way to read a dictionary when a key might be absent —
exactly the missing-data handling that ``COALESCE`` provided in SQL, in dictionary form.
Using ``.get()`` where fields may be missing prevents the crash that direct access would
cause.

Nested dictionaries
-------------------

Dictionaries can contain dictionaries (and lists), representing *structured*, hierarchical
data:

.. code-block:: python

   data = {
       "north": {"sales": 1000, "customers": 50},
       "south": {"sales": 800,  "customers": 40},
   }
   data["north"]["sales"]         # 1000 — access nested by chaining keys

Nesting represents data with structure — regions each holding their own metrics — and is
exactly the shape of JSON and much real-world data. Accessing nested data chains the keys
(``data["north"]["sales"]``), reaching down through the levels.

Why advanced usage matters
----------------------------

These techniques are what make dictionaries usable for *real* data rather than toy examples.
Iterating processes records field by field; ``.get()`` handles the missing fields real data
always has; nesting represents the hierarchical structure real data often takes (especially
data from web APIs and JSON). Together they turn the dictionary from a simple lookup into a
practical tool for structured data — a bridge toward the DataFrames of pandas, which
generalise these key-value, record-oriented ideas to full tables.

The caveat
------------

Advanced dictionary use concentrates the earlier pitfalls plus new ones. Nested access
*multiplies* the ``KeyError`` risk — ``data["west"]["sales"]`` fails if *either* "west" or
"sales" is missing, so deep access into possibly-incomplete data needs ``.get()`` at each
level (or careful checking), lest a single missing key crash the program. Deeply nested
dictionaries also grow *hard to navigate* — many levels of keys become as tangled as deeply
nested anything, and at that point a more structured representation (or a DataFrame) is often
better. Use ``.get()`` for anything that might be missing, keep nesting to what the data
genuinely requires, and reach for pandas when dictionary-of-dictionaries starts to strain.
The next lesson covers the last core structure, the set.
"""


MINDMAP.update({
    "Tuples in Python": [
        "Modifying Lists in Python",
        "Data Types vs Data Structures & Introduction to Lists",
        "Advanced Use of Loops, Lists, Tuples & List Comprehension",
        "Strings in Python",
    ],
    "Advanced Use of Loops, Lists, Tuples & List Comprehension": [
        "Tuples in Python",
        "Modifying Lists in Python",
        "For Loops in Python",
        "Dictionaries in Python",
    ],
    "Dictionaries in Python": [
        "Advanced Dictionary Usage in Python",
        "Data Types vs Data Structures & Introduction to Lists",
        "Tuples in Python",
        "Sets in Python",
    ],
    "Advanced Dictionary Usage in Python": [
        "Dictionaries in Python",
        "Sets in Python",
        "Libraries, Packages, and Modules in Python",
        "For Loops in Python",
    ],
})


# ======================================================================
# Section 7 — Data Analysis with Python / structures (close) + libraries (open)  (python 025-028)
# ======================================================================

GLOSS.update({
    "Sets in Python":
        "unordered collections of unique values — membership and set operations",
    "Libraries, Packages, and Modules in Python":
        "reusing others' code — importing modules, packages, and the library ecosystem",
    "Introduction to NumPy and Vectorization":
        "the numerical library and vectorization — fast array operations without loops",
    "NumPy Arrays (ndarray) and Core Concepts":
        "the ndarray, NumPy's core structure, and how it differs from a list",
})

CONTENT["Sets in Python"] = r"""
Unordered collections of unique values
----------------------------------------

The last core data structure is the **set** — an unordered collection of *unique* values.
Sets excel at membership testing, removing duplicates, and set operations (union,
intersection), and understanding them completes Python's built-in structures. This lesson
covers sets, closing the structures stage.

What a set is
-------------

A **set** is a collection with two defining properties: its values are **unique** (no
duplicates) and **unordered** (no positional index). Written with curly braces (like a
dictionary, but values not pairs):

.. code-block:: python

   regions = {"North", "South", "East", "North"}   # duplicate ignored
   print(regions)            # {"North", "South", "East"} — 3 unique values

Adding a duplicate has no effect (the value is already present), and there is no
``set[0]`` — sets are not indexed. Their purpose is *membership* and *uniqueness*, not
order.

Set operations
--------------

Sets support fast membership testing and mathematical set operations:

.. code-block:: python

   "North" in regions       # True — fast membership test

   a = {1, 2, 3}
   b = {2, 3, 4}
   a | b                     # {1, 2, 3, 4} — union (in either)
   a & b                     # {2, 3} — intersection (in both)
   a - b                     # {1} — difference (in a but not b)

Membership testing (``in``) is *very fast* on a set — faster than searching a list — and the
operations (union ``|``, intersection ``&``, difference ``-``) answer "in either / both /
one but not the other" directly. These are exactly the set operations of mathematics, and of
SQL's ``UNION``/``INTERSECT``/``EXCEPT``.

Why sets matter
---------------

Sets serve specific, common needs:

- **Removing duplicates** — converting a list to a set drops duplicates instantly
  (``set(my_list)``), the fastest deduplication in Python.
- **Fast membership testing** — checking whether a value is in a large collection is far
  faster with a set than a list, which matters at scale.
- **Comparing collections** — the set operations answer "what is common / different between
  these two collections?" directly (which customers are in both lists? which are only in
  one?).

For these tasks — uniqueness, membership, comparison — the set is the right tool, cleaner and
faster than working around a list.

The caveat
------------

Sets' properties are also their limitations. Being *unordered*, a set cannot be indexed or
sliced, and does not preserve insertion order — if order matters, a set is the wrong
structure (use a list). Being *unique*, a set cannot hold duplicates — which is the point
for deduplication, but means a set cannot represent data where repetition is meaningful (a
set of sales figures would collapse identical amounts into one, losing information). And set
elements must be *immutable* (like dictionary keys), so a set cannot contain lists. Use a set
precisely when uniqueness and membership are what you want, and a list when order or
repetition matters. This completes Python's core data structures; the next lessons open the
final stage — the libraries that make Python a data-analysis powerhouse.
"""

CONTENT["Libraries, Packages, and Modules in Python"] = r"""
Standing on others' code
--------------------------

Python's true power for data analysis comes not from the language alone but from its vast
ecosystem of **libraries** — reusable code others have written, imported and used rather
than reinvented. Opening the final stage, this lesson covers libraries, packages, and
modules: what they are, how to import them, and why they make Python the dominant data tool.

Modules, packages, libraries
------------------------------

The terms nest:

- A **module** is a single file of Python code (functions, classes, values) that can be
  imported and used elsewhere.
- A **package** is a collection of modules organised together (a directory of related
  modules).
- A **library** is, loosely, a body of reusable code — a package or set of packages —
  providing functionality for some purpose (numerical computing, data analysis,
  visualization).

The distinctions are technical; in practice "library" is used broadly for importable code
you use rather than write. The point is *reuse at scale* — leveraging code the community has
built, tested, and maintained.

Importing
---------

Code from a library is brought in with ``import``:

.. code-block:: python

   import math                     # import a module
   math.sqrt(16)                   # use it with the module name: 4.0

   import numpy as np              # import with an alias (convention)
   np.array([1, 2, 3])             # use via the alias

   from statistics import mean     # import a specific name
   mean([1, 2, 3])                 # use it directly: 2

``import numpy as np`` brings in the NumPy library under the short alias ``np`` (a universal
convention); ``from ... import ...`` brings a specific name into direct use. These import
forms are how every library in the coming lessons is accessed — ``import numpy as np``,
``import pandas as pd`` are the standard incantations.

The data ecosystem
-------------------

Python's data-analysis dominance rests on its libraries, the key ones being:

- **NumPy** — fast numerical computing on arrays (the next lessons).
- **pandas** — tabular data analysis, the DataFrame (the lessons after).
- **matplotlib** / others — visualization.
- and a vast constellation for statistics, machine learning, and more.

These libraries do the heavy lifting — decades of optimised, tested code — so an analyst
composes existing powerful tools rather than building from scratch. This is the reuse
principle at the ecosystem scale: the reason Python is *the* data language is largely this
library ecosystem.

The caveat
------------

Libraries are indispensable but bring their own considerations. **Dependencies** — the
libraries your code relies on — must be installed and, importantly, *versioned*: code
written for one version of a library may behave differently or break on another, so managing
which versions are used matters for reproducibility (the reproducibility theme, at the
dependency level). Importing many libraries also has a cost, and reaching for a heavy library
for a task the standard tools handle is its own over-engineering. The disciplines: use
well-established, maintained libraries; be aware of version compatibility; and import what the
task genuinely needs. The ecosystem is Python's great strength, used deliberately. The next
lessons dive into the first essential library: NumPy.
"""

CONTENT["Introduction to NumPy and Vectorization"] = r"""
Fast numbers without loops
----------------------------

The first essential data library is **NumPy** — the foundation of numerical computing in
Python, and the base that pandas itself is built on. Its central idea is **vectorization**:
performing operations on entire arrays at once, without explicit loops, far faster than
looping. This lesson introduces NumPy and vectorization, the shift from loop-based to
array-based thinking.

What NumPy provides
--------------------

NumPy provides the **array** — a fast, homogeneous collection of numbers — and a vast set of
operations on arrays. Imported conventionally as ``np``:

.. code-block:: python

   import numpy as np

   arr = np.array([1, 2, 3, 4, 5])     # a NumPy array
   arr * 2                             # array([2, 4, 6, 8, 10]) — whole array at once
   arr + 10                            # array([11, 12, 13, 14, 15])
   arr.sum()                           # 15
   arr.mean()                          # 3.0

The striking thing: ``arr * 2`` multiplies *every* element by 2 in one operation — no loop.
This is vectorization.

Vectorization
-------------

**Vectorization** means applying an operation to a whole array at once, rather than looping
over elements. Compare:

.. code-block:: python

   # loop way (slow, verbose):
   result = []
   for x in [1, 2, 3, 4, 5]:
       result.append(x * 2)

   # vectorized way (fast, concise):
   arr = np.array([1, 2, 3, 4, 5])
   result = arr * 2

Both double each element, but the vectorized version is *dramatically faster* on large data
(NumPy runs the operation in optimised low-level code, not a Python loop) and *more concise*
(one expression, no loop). This is the payoff the for-loop lesson foreshadowed — for numeric
data at scale, vectorized array operations replace explicit loops.

Why vectorization matters
--------------------------

Vectorization is the core idiom of numerical data work in Python, for two reasons. **Speed**
— vectorized NumPy operations are orders of magnitude faster than Python loops on large
arrays, which is what makes Python viable for real data sizes. **Clarity** — ``arr * 2`` or
``arr1 + arr2`` expresses an operation on all the data in one readable line, closer to the
mathematical intent than a loop. This array-and-vectorization thinking underlies pandas too
(a DataFrame column is array-like and vectorized), so learning it here is learning the idiom
of all Python data analysis.

The caveat
------------

Vectorization requires a *mental shift* that trips up those coming from loop-based
programming: you stop thinking "for each element, do X" and start thinking "do X to the whole
array." Reaching for a Python loop over a NumPy array *defeats the purpose* — it is both
slower and less clear than the vectorized operation, yet it is the instinctive move for
loop-trained programmers. The discipline is to express operations on whole arrays, using
NumPy's vectorized operations and functions rather than looping. There are genuine cases
where a loop is unavoidable, but for element-wise arithmetic and aggregation, vectorization
is the right and expected approach. Think in arrays, not elements. The next lesson covers the
array structure itself in depth.
"""

CONTENT["NumPy Arrays (ndarray) and Core Concepts"] = r"""
The array in depth
--------------------

NumPy's central structure is the **ndarray** (n-dimensional array), and understanding it —
how it differs from a list, its key properties, and its core operations — is essential to
using NumPy and pandas well. This lesson covers the ndarray and NumPy's core concepts,
deepening the array foundation.

The ndarray versus a list
---------------------------

A NumPy array (``ndarray``) looks list-like but differs crucially:

.. code-block:: python

   import numpy as np
   arr = np.array([1, 2, 3, 4, 5])

- **Homogeneous** — all elements are the *same type* (all integers, all floats), unlike a
  list's mixed types. This uniformity is what enables NumPy's speed.
- **Fixed size** — an array has a set size; you do not append to it as you do a list (you
  create new arrays instead).
- **Vectorized** — operations apply to the whole array (the previous lesson), which lists do
  not support (``[1,2,3] * 2`` repeats the list rather than doubling elements).
- **Fast and compact** — arrays store data efficiently and operate on it in optimised code,
  far outperforming lists for numeric work.

The array trades the list's flexibility (mixed types, easy resizing) for *speed and
vectorization* on uniform numeric data — the right trade for numerical analysis.

Core array properties and creation
------------------------------------

Arrays have properties describing their structure, and several creation functions:

.. code-block:: python

   arr = np.array([[1, 2, 3], [4, 5, 6]])   # a 2D array (from nested lists)
   arr.shape                 # (2, 3) — 2 rows, 3 columns
   arr.ndim                  # 2 — number of dimensions
   arr.size                  # 6 — total elements
   arr.dtype                 # the element data type (e.g. int64)

   np.zeros(5)               # array of five 0.0s
   np.arange(0, 10, 2)       # array([0, 2, 4, 6, 8]) — like range()
   np.linspace(0, 1, 5)      # 5 evenly spaced values from 0 to 1

The **shape** (dimensions) is central — arrays can be 1D (a vector), 2D (a matrix, like a
table), or more, and the shape describes that structure. NumPy's creation functions build
common arrays without manual listing.

Indexing, slicing, and operations
------------------------------------

Arrays are indexed and sliced like lists (zero-based, exclusive stop), extended to multiple
dimensions, and support rich vectorized operations:

.. code-block:: python

   arr = np.array([10, 20, 30, 40, 50])
   arr[0]                    # 10
   arr[1:3]                  # array([20, 30])
   arr[arr > 25]             # array([30, 40, 50]) — boolean indexing!

   arr.sum(); arr.mean(); arr.max(); arr.std()   # aggregate functions

The last — ``arr[arr > 25]`` — is **boolean indexing**: selecting elements by a condition,
which returns only the matching elements. This is exactly the filtering idea from SQL
``WHERE`` and spreadsheet filters, vectorized — and it is the foundation of the boolean
masking that pandas uses to filter data (an upcoming lesson).

Why the ndarray matters
------------------------

The ndarray is the foundation of numerical Python — NumPy's own operations, and *pandas*
(whose columns are essentially arrays), all rest on it. Understanding arrays — homogeneous,
vectorized, shaped, boolean-indexable — is understanding the substrate of all Python data
analysis. The concepts here (vectorization, boolean indexing, aggregation) reappear directly
in pandas, so the array is where the idioms of data analysis in Python are first and most
clearly learned.

The caveat
------------

The ndarray's constraints are real and occasionally surprising. Its **homogeneity** means
mixing types coerces them (putting a float in an int array converts, and mixing numbers and
strings makes everything strings), which can silently change data — the array assumes uniform
type. Its **fixed size** means "growing" an array actually creates a new one, so
list-style appending in a loop is both wrong-idiom and slow (build the data as a list then
convert, or use vectorized construction). And array operations require *compatible shapes* —
operating on mismatched shapes errors or triggers broadcasting rules that surprise the
unwary. For the flat, uniform numeric data of typical analysis these rarely bite, but the
array is a stricter, more structured thing than a list — which is exactly what makes it fast.
The next lessons build on the array to reach pandas, the analyst's primary Python tool.
"""


MINDMAP.update({
    "Sets in Python": [
        "Dictionaries in Python",
        "Data Types vs Data Structures & Introduction to Lists",
        "Advanced Use of Loops, Lists, Tuples & List Comprehension",
        "Libraries, Packages, and Modules in Python",
    ],
    "Libraries, Packages, and Modules in Python": [
        "Introduction to NumPy and Vectorization",
        "Object-Oriented Programming (OOP) in Python",
        "Introduction to Pandas (Data Analysis Library)",
        "Advanced Dictionary Usage in Python",
    ],
    "Introduction to NumPy and Vectorization": [
        "Libraries, Packages, and Modules in Python",
        "NumPy Arrays (ndarray) and Core Concepts",
        "Introduction to Pandas (Data Analysis Library)",
        "Advanced Use of Loops, Lists, Tuples & List Comprehension",
    ],
    "NumPy Arrays (ndarray) and Core Concepts": [
        "Introduction to NumPy and Vectorization",
        "Introduction to Pandas (Data Analysis Library)",
        "Pandas DataFrame & Series",
        "Libraries, Packages, and Modules in Python",
    ],
})


# ======================================================================
# Section 7 — Data Analysis with Python / libraries (pandas core)  (python 029-032)
# ======================================================================

GLOSS.update({
    "Introduction to Pandas (Data Analysis Library)":
        "the DataFrame library that brings spreadsheet/SQL-style tables to Python",
    "Pandas DataFrame & Series":
        "the two core pandas structures — the table (DataFrame) and the column (Series)",
    "Boolean Masking in Pandas":
        "filtering rows by a condition — the pandas equivalent of WHERE and spreadsheet filters",
    "Grouping and Aggregation in Pandas (groupby, agg)":
        "grouping rows and computing per-group aggregates — pandas' GROUP BY",
})

CONTENT["Introduction to Pandas (Data Analysis Library)"] = r"""
Tables in Python
-----------------

If NumPy is the foundation, **pandas** is the tool an analyst uses most — the library that
brings *tabular data* to Python, with the row-and-column tables familiar from spreadsheets
and SQL. Built on NumPy, pandas is the primary Python tool for data analysis, and this lesson
introduces it: what it is, and why it ties the whole course together.

What pandas provides
--------------------

Pandas provides labelled, tabular data structures and a vast toolkit for working with them.
Imported conventionally as ``pd``:

.. code-block:: python

   import pandas as pd

   df = pd.DataFrame({
       "region": ["North", "South", "East"],
       "sales":  [1000, 800, 1200],
   })

This creates a **DataFrame** — a table with named columns and indexed rows, exactly the
tabular structure the whole course has worked with. Pandas can read data from files
(``pd.read_csv("data.csv")``), databases, and more, turning external data into a DataFrame to
analyse.

Pandas as the culmination
---------------------------

Pandas is where the course's threads converge, because it does *in Python* what the earlier
tools did separately:

- **The tabular structure** from the data-preparation section — rows and columns, tidy data —
  is the DataFrame.
- **The cleaning** from Section 4 — handling duplicates, missing values, types — pandas does
  with methods (``drop_duplicates()``, ``fillna()``, ``astype()``).
- **The analysis** from Section 5 — sorting, filtering, grouping, aggregating, joining —
  pandas does with methods (``sort_values()``, boolean masks, ``groupby()``, ``merge()``).
- **The visualization** from Section 6 — pandas integrates with plotting libraries to chart
  directly.

Everything done in spreadsheets and SQL, pandas can do in Python — automated, reproducible,
and at scale. It is the single tool that spans the whole analytical workflow.

Why pandas matters
------------------

Pandas matters because it unifies the analyst's work in one powerful, programmable tool. The
spreadsheet's visibility and the SQL query's power, combined with Python's automation and
scale, come together in the DataFrame — clean it, transform it, analyse it, and visualize it,
all in reproducible code. For serious Python data analysis, pandas is *the* tool, which is why
this section builds toward it, and why the concepts from every prior section reappear here in
DataFrame form.

The caveat
------------

Pandas is powerful and correspondingly large, with a steep learning curve and many ways to do
each thing — some efficient, some not. Two cautions matter early. First, pandas rewards the
*vectorized* thinking from NumPy: iterating over a DataFrame's rows with a Python loop is slow
and un-idiomatic, where a vectorized operation or built-in method does the same far faster —
the "don't loop" lesson applies doubly to DataFrames. Second, pandas' size means there is
usually a clean built-in method for what you want, so reaching for a convoluted manual
approach often means missing the tool that does it in one call. Learn the idiomatic,
vectorized pandas — the methods and masks — rather than writing loops against it. The next
lessons cover its structures and operations. The whole course's analytical vocabulary is
about to reappear, in Python.
"""

CONTENT["Pandas DataFrame & Series"] = r"""
The two core structures
------------------------

Pandas is built on two structures: the **Series** (a single column) and the **DataFrame** (a
table of columns). Understanding them — how they relate, and how to access their data — is the
foundation of all pandas work. This lesson covers the DataFrame and Series in depth.

The Series: a column
---------------------

A **Series** is a one-dimensional labelled array — essentially a single column of data with an
index:

.. code-block:: python

   import pandas as pd
   sales = pd.Series([1000, 800, 1200], index=["North", "South", "East"])
   sales["North"]            # 1000 — access by label
   sales.mean()              # 1000.0 — vectorized aggregate

A Series is like a NumPy array (vectorized, homogeneous-ish) but with *labels* (an index) and
pandas' richer methods. Each column of a DataFrame is a Series.

The DataFrame: a table
-----------------------

A **DataFrame** is a two-dimensional labelled table — rows and named columns — the central
pandas structure:

.. code-block:: python

   df = pd.DataFrame({
       "region": ["North", "South", "East"],
       "sales":  [1000, 800, 1200],
       "customers": [50, 40, 60],
   })

   df["sales"]               # a column (a Series)
   df[["region", "sales"]]   # multiple columns (a DataFrame)
   df.head()                 # first rows
   df.shape                  # (3, 3) — rows, columns
   df.info()                 # summary of columns and types
   df.describe()             # summary statistics of numeric columns

A DataFrame is a collection of Series (columns) sharing an index (rows) — exactly the tabular,
one-column-per-variable structure from the data-preparation section, now a Python object with
methods.

Accessing rows and columns
----------------------------

Pandas accesses data by label or position:

.. code-block:: python

   df["sales"]               # a column by name
   df.loc[0]                 # a row by label (index)
   df.iloc[0]                # a row by position
   df.loc[0, "sales"]        # a specific cell by label
   df.iloc[0, 1]             # a specific cell by position

``loc`` accesses by *label*, ``iloc`` by *integer position* — the distinction to keep clear.
Columns are accessed by name (``df["sales"]``), rows by ``loc``/``iloc``. These are how you
reach any part of the table.

Why these structures matter
----------------------------

The DataFrame and Series are the objects *all* pandas analysis operates on, and they directly
embody the course's data concepts: the DataFrame is the tidy table, each column a Series (a
variable), each row an observation. Every operation ahead — filtering, grouping, aggregating,
joining — is a method on these structures, and the aggregates (``mean``, ``sum``) are
vectorized over the Series. Understanding that a DataFrame is a labelled table of Series
columns is the mental model that makes all of pandas coherent.

The caveat
------------

The DataFrame's flexibility hides subtleties that cause classic pandas confusion. The
``loc`` versus ``iloc`` distinction (label versus position) trips up beginners constantly —
they look similar but differ, and using the wrong one selects the wrong data. Pandas also has a
famous **SettingWithCopyWarning** arising from the difference between a *view* and a *copy* of
data — modifying what you think is the DataFrame but is actually a temporary slice, so the
change does not stick (or warns). And the index — pandas' row labels — behaves in ways that
surprise those expecting simple row numbers. These are learned through use and by reading
pandas' (generally helpful) warnings; the key early discipline is knowing ``loc`` from
``iloc`` and being deliberate about whether you are viewing or copying. The next lesson covers
filtering rows: boolean masking.
"""

CONTENT["Boolean Masking in Pandas"] = r"""
Filtering rows by condition
-----------------------------

The most common data operation — selecting the rows that meet a condition — pandas does with
**boolean masking**: a condition produces an array of True/False, and the DataFrame keeps the
True rows. This is the pandas equivalent of SQL's ``WHERE`` and the spreadsheet filter, and it
is fundamental. This lesson covers boolean masking.

How masking works
-----------------

A comparison on a DataFrame column produces a **boolean Series** (a mask), which then selects
rows:

.. code-block:: python

   import pandas as pd
   df = pd.DataFrame({"region": ["N","S","E"], "sales": [1000, 800, 1200]})

   df["sales"] > 900         # a boolean Series: [True, False, True]
   df[df["sales"] > 900]     # keeps rows where the mask is True

``df["sales"] > 900`` computes the condition for *every* row at once (vectorized, the NumPy
boolean-indexing idea), producing True/False per row; ``df[ ... ]`` with that mask returns only
the rows where it is True. This is filtering — "the rows where sales exceed 900" — in one
readable expression.

Combining conditions
--------------------

Multiple conditions combine with ``&`` (and), ``|`` (or), and ``~`` (not), each condition
parenthesised:

.. code-block:: python

   df[(df["sales"] > 900) & (df["region"] == "N")]   # both conditions
   df[(df["region"] == "N") | (df["region"] == "S")] # either
   df[~(df["sales"] > 900)]                            # not

The parentheses around each condition are *required* (pandas' operator precedence demands
them), and ``&``/``|`` are used, not the words ``and``/``or`` — a pandas-specific rule.
Compound masks express multi-condition filters, exactly the ``WHERE ... AND ...`` of SQL.

Masking versus the earlier filters
------------------------------------

Boolean masking is precisely the *filtering* concept from every prior section, in pandas form.
``df[df["sales"] > 900]`` is SQL's ``SELECT * FROM df WHERE sales > 900`` and the spreadsheet's
filter-to-sales-over-900 — the identical operation, vectorized in Python. Recognising this
makes masking intuitive: it is the familiar "show only the rows meeting this condition,"
expressed as a boolean Series selecting from a DataFrame. It is also used for *modifying*
subsets (``df.loc[df["sales"] > 900, "flag"] = True`` sets a value on matching rows).

Why masking matters
-------------------

Boolean masking is one of the most-used pandas operations, because filtering to relevant rows
is constant in analysis — isolating a segment, removing outliers, selecting valid data,
finding records meeting criteria. It is vectorized (fast on large data) and expressive (complex
conditions in one line), and it underlies much of data cleaning and analysis in pandas. Master
masking and a large fraction of practical pandas filtering follows.

The caveat
------------

Masking has pandas-specific traps that catch beginners predictably. The operators are ``&``,
``|``, ``~`` — *not* ``and``, ``or``, ``not`` (using the words raises an error on Series); and
each condition *must* be parenthesised (``df[df["a"]>1 & df["b"]<2]`` is wrong — precedence
mis-groups it — while ``df[(df["a"]>1) & (df["b"]<2)]`` is right). Masks involving *missing
values* (NaN) behave carefully — a comparison with NaN is False, so masking silently excludes
missing-value rows (the null-handling theme, in pandas). And modifying a masked subset invites
the view-versus-copy ``SettingWithCopyWarning``, so use ``df.loc[mask, col] = ...`` for
assignment. Parenthesise conditions, use the symbol operators, and mind NaN and assignment. The
next lesson covers grouping and aggregation.
"""

CONTENT["Grouping and Aggregation in Pandas (groupby, agg)"] = r"""
Grouping and summarising
--------------------------

The analytical workhorse — grouping rows by a category and computing an aggregate per group —
pandas does with **groupby** and aggregation. This is the direct equivalent of SQL's
``GROUP BY`` and the spreadsheet's pivot table, and it is where pandas becomes a serious
analytical tool. This lesson covers grouping and aggregation.

The groupby operation
---------------------

``groupby`` splits a DataFrame into groups by a column, then an aggregate is computed for each
group:

.. code-block:: python

   import pandas as pd
   df = pd.DataFrame({
       "region": ["N", "S", "N", "S", "E"],
       "sales":  [100, 200, 150, 250, 300],
   })

   df.groupby("region")["sales"].sum()
   # region
   # E    300
   # N    250
   # S    450

``df.groupby("region")["sales"].sum()`` groups rows by region, then sums sales within each
group — producing the per-region total. This is exactly SQL's
``SELECT region, SUM(sales) FROM df GROUP BY region`` and a pivot table of sales by region, in
pandas.

Aggregation functions and agg
-------------------------------

Any aggregate can be applied per group, and ``agg`` computes several at once:

.. code-block:: python

   df.groupby("region")["sales"].mean()     # average per region
   df.groupby("region")["sales"].count()    # count per region

   df.groupby("region")["sales"].agg(["sum", "mean", "count"])
   # multiple aggregates per group, as a table

   df.groupby("region").agg(
       total_sales=("sales", "sum"),
       avg_sales=("sales", "mean"),
   )                                          # named aggregates

The aggregate functions are the familiar ones (``sum``, ``mean``, ``count``, ``min``, ``max``,
``std``); ``agg`` applies several, optionally with names — the pandas way to build a
multi-metric summary per group, like a richer pivot table or a ``GROUP BY`` with several
aggregates.

Grouping versus the earlier tools
-----------------------------------

Groupby is the *grouped aggregation* from every prior section, in pandas. It is SQL's
``GROUP BY`` with aggregate functions (Section 5), the spreadsheet's pivot table (Section 5),
and the conditional-aggregation ideas — all expressed as ``df.groupby(col).agg(...)``. The
"split-apply-combine" it performs (split into groups, apply an aggregate, combine into a
result) is the same operation, and recognising it as the familiar pivot/GROUP BY makes it
immediately meaningful. It is the analytical core of pandas, as GROUP BY was of SQL.

The caveat
------------

Groupby is powerful and carries the aggregation subtleties seen throughout, plus pandas
specifics. Aggregates handle *missing values* by skipping them (``mean`` ignores NaN, changing
the denominator — the null-in-aggregate theme); grouping on a column with inconsistent values
fragments groups (the "NY" versus "New York" problem, so *clean before grouping*, exactly as in
SQL); and the *result* of a groupby has the grouping column as its index, which sometimes
surprises (``reset_index()`` turns it back into a column). As always, the operation computes
precisely what you specify — verify the groups are what you intend and the aggregates handle
missing data as you want. This nearly completes the section; the final lesson covers combining
DataFrames — the pandas equivalent of the JOIN.
"""


MINDMAP.update({
    "Introduction to Pandas (Data Analysis Library)": [
        "NumPy Arrays (ndarray) and Core Concepts",
        "Pandas DataFrame & Series",
        "Libraries, Packages, and Modules in Python",
        "Introduction to NumPy and Vectorization",
    ],
    "Pandas DataFrame & Series": [
        "Introduction to Pandas (Data Analysis Library)",
        "Boolean Masking in Pandas",
        "Grouping and Aggregation in Pandas (groupby, agg)",
        "NumPy Arrays (ndarray) and Core Concepts",
    ],
    "Boolean Masking in Pandas": [
        "Pandas DataFrame & Series",
        "Sorting and Filtering Data in SQL Using ORDER BY and WHERE",
        "Grouping and Aggregation in Pandas (groupby, agg)",
        "Introduction to Pandas (Data Analysis Library)",
    ],
    "Grouping and Aggregation in Pandas (groupby, agg)": [
        "Boolean Masking in Pandas",
        "Using GROUP BY and ORDER BY for Aggregated Calculations in SQL",
        "Combining Data in Pandas (concat and merge)",
        "Pandas DataFrame & Series",
    ],
})


# ======================================================================
# Section 7 — Data Analysis with Python / libraries (close)  (python 033)  -- SECTION 7 COMPLETE
# ======================================================================

GLOSS.update({
    "Combining Data in Pandas (concat and merge)":
        "bringing DataFrames together — stacking with concat, joining on keys with merge",
})

CONTENT["Combining Data in Pandas (concat and merge)"] = r"""
Bringing DataFrames together
------------------------------

Data rarely lives in one table, and the final pandas skill — and the last lesson of the
Python section — is **combining DataFrames**: stacking them with ``concat`` and joining them
on keys with ``merge``. These are the pandas equivalents of the SQL ``JOIN`` and ``UNION``,
completing the analyst's ability to do in Python everything the earlier tools did. This lesson
covers combining data in pandas.

Stacking with concat
---------------------

``pd.concat`` stacks DataFrames together — vertically (more rows) or horizontally (more
columns):

.. code-block:: python

   import pandas as pd
   jan = pd.DataFrame({"region": ["N", "S"], "sales": [100, 200]})
   feb = pd.DataFrame({"region": ["N", "S"], "sales": [150, 250]})

   pd.concat([jan, feb])                 # stack vertically: 4 rows
   pd.concat([jan, feb], ignore_index=True)   # renumber the index

Vertical ``concat`` appends rows — combining data of the *same columns* from different sources
(January and February sales, files from different periods), exactly SQL's ``UNION``. This is
how you assemble one dataset from multiple like-structured pieces.

Joining with merge
------------------

``pd.merge`` combines DataFrames by *matching keys* — the pandas ``JOIN``:

.. code-block:: python

   sales = pd.DataFrame({"region": ["N", "S", "E"], "sales": [100, 200, 300]})
   info  = pd.DataFrame({"region": ["N", "S", "E"], "manager": ["Amy", "Bo", "Cy"]})

   pd.merge(sales, info, on="region")    # join on the shared 'region' key

   # join types mirror SQL exactly:
   pd.merge(sales, info, on="region", how="inner")   # only matching keys (default)
   pd.merge(sales, info, on="region", how="left")    # all left rows, matched where possible
   pd.merge(sales, info, on="region", how="outer")   # all rows from both

``merge`` matches rows on a key column (``on="region"``) and combines their columns — and its
``how`` parameter (``inner``, ``left``, ``right``, ``outer``) is *exactly* the SQL join types
from the analysis section. Everything learned about JOINs — matching keys, the join types, what
each keeps — applies directly; ``pd.merge`` is the SQL JOIN in pandas.

Combining versus the earlier tools
------------------------------------

These operations complete the mapping of the whole analytical workflow into pandas.
``pd.concat`` (vertical) is ``UNION`` — stacking like-structured data; ``pd.merge`` is
``JOIN`` — combining related tables on keys, with the same inner/left/right/outer semantics.
The relational-combination concepts from Section 5 (the SQL section) carry over one-to-one, so
an analyst who understands JOINs already understands ``merge`` — only the syntax differs. With
combining in hand, pandas covers the entire span: import, clean, transform, filter, group,
aggregate, and join — the whole workflow, in reproducible Python.

The caveat
------------

Combining data concentrates the hazards flagged throughout the JOIN and integration lessons.
``merge`` on a key with *duplicates* can multiply rows unexpectedly (a many-to-many match
produces the cross-product, the row-explosion warned about with JOINs); a ``merge`` on keys
that do not cleanly correspond (inconsistent formatting, "N" versus "North") silently *fails to
match*, dropping rows in an inner join or producing NaN in an outer — the key-integrity
discipline from the cleaning section, now in pandas. ``concat`` assumes the pieces share
structure; stacking DataFrames with mismatched columns produces misaligned data or unexpected
NaN. As always: know your keys, verify the row count after a merge is what you expect (a sharp
change signals a join problem), and ensure concatenated pieces truly align. Combine
deliberately, checking the result.

This completes the Data Analysis with Python section — and with it, the analyst's core
technical toolkit. Across this section you have moved from Python's fundamentals through its
data structures to the libraries that make it a data powerhouse: NumPy's fast vectorized
arrays, and pandas' DataFrames, on which you can clean, transform, filter, group, aggregate,
and join data — everything the spreadsheet and SQL sections taught, now automated and
reproducible in code. Python does not replace the earlier tools so much as unify and scale
them: the concepts are the same, expressed in one programmable environment. With data skills
from foundations through Python now in place, the final section turns from *doing* the work to
*getting the role* — the job search: presenting your skills, finding opportunities, and
succeeding in interviews as a data analyst.
"""


MINDMAP.update({
    "Combining Data in Pandas (concat and merge)": [
        "Grouping and Aggregation in Pandas (groupby, agg)",
        "Using JOIN in SQL to Combine Tables",
        "Pandas DataFrame & Series",
        "Introduction to Pandas (Data Analysis Library)",
    ],
})


# ======================================================================
# Section 8 — Job Search / Stage: identity  (jobsearch 001-004)  -- SECTION 8 OPENS
# Career Dreamer facts grounded via web_search (Grow with Google / Google Labs, US-only,
# Lightcast + BLS labor data, produces Career Identity Statement, Explore Paths, Gemini handoff,
# does NOT show live listings). Written in original prose.
# ======================================================================

GLOSS.update({
    "Transferable Skills":
        "abilities that carry across roles and industries — the foundation of a career pivot",
    "Career Identity Statement":
        "a concise statement of the unique value you bring to the workforce",
    "Career Dreamer (AI Tool for Career Exploration)":
        "an experimental Grow with Google AI tool that maps your experience to career paths",
    "Job Search Plan (Using AI Tools)":
        "a structured, deliberate plan for the job search, assisted by AI tools",
})

CONTENT["Transferable Skills"] = r"""
Skills that travel
--------------------

Having built the technical toolkit of a data analyst, the final section turns to *getting
the role* — and it begins with a reframing that makes the whole job search possible:
**transferable skills**. These are abilities that carry across roles, industries, and career
changes, and recognising them is what lets someone move into data analysis from wherever they
started. Opening the job-search section, this lesson covers transferable skills and why they
matter.

What transferable skills are
------------------------------

**Transferable skills** are capabilities that are not tied to one specific job but apply
across many — communication, problem-solving, analytical thinking, attention to detail,
organisation, collaboration, and the like. They contrast with *role-specific* technical skills
(a particular software, a specific procedure) that apply narrowly. The insight is that much of
what makes someone effective is transferable: a teacher's communication and organisation, a
retail worker's customer insight and problem-solving, a scientist's analytical rigour — these
carry directly into a data-analysis role, even from an unrelated background.

Why they matter for a career in data
--------------------------------------

Transferable skills are the bridge into data analysis for career-changers and new entrants —
which describes many aspiring analysts. Data analysis rewards exactly the transferable
abilities many people already have: analytical thinking (the whole course's deductive
reasoning), attention to detail (the data-quality discipline), communication (the storytelling
and presentation section), and problem-solving (the analysis process). A career pivot into data
is rarely starting from zero; it is *repackaging* skills you already possess alongside the new
technical ones, and recognising that continuity is what makes the move feel possible rather
than impossible.

Identifying your transferable skills
--------------------------------------

The practical task is to *recognise* the transferable skills in your own background — which is
harder than it sounds, because people systematically undervalue skills that come naturally to
them. Reflecting on past roles (work, education, volunteering, side projects) and asking what
capabilities they built — how you solved problems, communicated, organised, analysed, worked
with others — surfaces the transferable skills you can carry forward. Framing these
effectively, in the language of the target role, is the next step (the career identity and
resume lessons), but it begins with *seeing* that your experience is full of relevant skills.

Why this reframing matters
----------------------------

This reframing is the psychological and practical foundation of a career change. It turns "I
have no experience in data" into "I have analytical, communication, and problem-solving skills
from my background, plus the technical data skills I am now building" — a far stronger and truer
position. Employers increasingly value demonstrated transferable skills alongside formal
credentials, precisely because capabilities like analytical thinking and communication
transfer across contexts and are hard to teach. Recognising and articulating your transferable
skills is thus both a confidence-builder and a genuine job-search advantage.

The caveat
------------

Transferable skills are powerful but must be claimed *honestly and specifically*. Vaguely
asserting "strong communication skills" or "great problem-solver" without evidence is empty —
the claim carries weight only when backed by concrete examples of *using* the skill (the
behavioural-interview and resume lessons develop this). And transferable skills complement
rather than replace the *specific technical skills* a data role requires: the analytical
thinking transfers, but you still need the actual data competencies this course teaches. The
honest position is transferable skills *plus* demonstrated technical ability, each supported by
evidence — not transferable skills asserted vaguely as a substitute for either specifics or
substance. The next lesson turns your skills into a concise statement of professional identity.
"""

CONTENT["Career Identity Statement"] = r"""
Your professional identity, in a sentence
-------------------------------------------

Once you recognise your transferable and technical skills, you need to *articulate* them — and
a **Career Identity Statement** does exactly that: a concise statement of the unique value you
bring to the workforce, shaped by your strengths, experience, and interests. It is a reusable
piece of professional self-definition, and this lesson covers what it is and how to craft one.

What a Career Identity Statement is
-------------------------------------

A **Career Identity Statement** is a short, focused statement of your professional identity —
who you are as a professional, what value you offer, and what you are aiming toward. Your
*career identity* is the unique value you bring, informed by your life and work experience and
shaped by your strengths, motivations, and interests. The statement captures that in a form you
can reuse: on a resume summary, a professional profile such as LinkedIn, or as talking points
in an interview. It answers, compactly, "what do I bring, and where am I headed?"

Crafting the statement
-----------------------

An effective Career Identity Statement weaves together several elements:

- **Who you help and what outcome you deliver** — the value you provide, framed around the
  problems you solve for an employer or audience.
- **Your key strengths** — the transferable and technical skills that define your
  contribution (the previous lesson's skills, articulated).
- **Your relevant experience** — the background that evidences those strengths, including a
  non-traditional path framed as a source of distinctive skills.
- **Your direction** — what you are now focused on or moving toward (a data-analysis role).

A common shape reads roughly: "I help [who] achieve [outcome] using [strengths], drawing on
experience across [evidence], and I am now focused on [direction]." The statement is concise —
a few sentences — and centred on *value*, not a mere list of duties.

The Career Identity Statement in the job search
-------------------------------------------------

The statement is reusable *infrastructure* for the whole search. It becomes the summary at the
top of a resume, the "about" section of a professional profile, an opener that answers "tell me
about yourself" in an interview, and the through-line that keeps your application materials
consistent. Crafting it once, well, gives you a coherent professional narrative to deploy
everywhere — which is why defining career identity is treated as a foundational step before
resumes, profiles, and interviews, each of which draws on it.

Why it matters
---------------

A clear Career Identity Statement gives a job search *focus and coherence*. Without one,
applications can read as scattered — a list of unconnected experiences with no clear
professional identity; with one, every piece of the search tells a consistent story about who
you are and the value you offer. It is especially valuable for career-changers, for whom a
non-linear path most needs a cohesive narrative to make sense to employers. The statement turns
a varied background into a deliberate professional identity, which is what employers respond to.

The caveat
------------

A Career Identity Statement must be *authentic and substantiated*, not aspirational fiction.
Claiming an identity or value you cannot back with real strengths and experience produces a
statement that collapses under the first probing interview question — the honesty principle
from the presentation section, applied to self-presentation. The statement should reflect who
you genuinely are and what you can genuinely do, framed at its best but not falsified.
Aspirational framing of real strengths is legitimate; inventing an identity is not, and it is
also self-defeating, because the gap shows. Craft a statement that is your true professional
identity stated compellingly — and note that AI tools can help draft one, the subject of the
next lesson. The next lesson covers an AI tool built to help with exactly this.
"""

CONTENT["Career Dreamer (AI Tool for Career Exploration)"] = r"""
An AI tool for career exploration
------------------------------------

Defining your skills, identity, and direction is hard to do alone — and AI tools now exist to
help. **Career Dreamer** is one such tool: an experimental, AI-powered career-exploration tool
from Grow with Google that connects your experience, skills, and interests to possible career
paths. This lesson covers what Career Dreamer is and how it fits a data-analyst job search,
grounding the section's AI-assisted approach in a concrete example.

What Career Dreamer is
-----------------------

Career Dreamer is an early-stage experiment from Grow with Google (Google's skills initiative)
that uses AI to make career exploration easier and more personalised. You provide information
about your background — current and previous roles, skills, education, and interests — and the
tool uses AI, grounded in labour-market data (drawn from sources such as Lightcast and the U.S.
Bureau of Labor Statistics), to find patterns and connect your experience to potential careers.
It is designed especially for people whose paths are non-linear — recent graduates, career
changers, adult learners, and members of the military community — for whom framing varied
experience into a coherent direction is hardest. At the time of writing it is available as a
US-only experiment.

What it helps you do
---------------------

Career Dreamer supports several of the tasks this section covers:

- **Craft a Career Identity Statement** — from the roles, tasks, and skills you enter, the tool
  (working with Google's Gemini assistant) drafts a Career Identity Statement capturing the
  value you bring, which you can reuse on a resume, profile, or in interviews (the previous
  lesson).
- **Explore career paths** — an "Explore Paths" view surfaces a range of careers that align
  with your background as a visual web of possibilities, which you can delve into to learn what
  each entails.
- **Take next steps with Gemini** — you can collaborate with Gemini to draft a cover letter,
  refine a resume, or spark further ideas — the handoff from exploration to concrete materials.

The emphasis is on *surfacing transferable skills you might undervalue* and helping you
articulate them — exactly the reframing the section opened with.

Fitting it into a data-analyst search
----------------------------------------

For someone moving into data analysis, a tool like Career Dreamer can help translate a prior
background into the language of the target role, surface the transferable skills that connect
past experience to data work, and produce a first-draft career identity to refine. Used
iteratively — first to explore broadly, then to narrow toward data roles, then to shape the
narrative — it can save hours of unfocused searching and self-doubt, giving a structured
starting point for the more concrete work (resumes, applications, interviews) that follows.

The caveat
------------

Career Dreamer, like any AI tool, is an *assistant, not an oracle* — a point its own creators
stress by calling it an experiment and its outputs a starting point rather than a prescription.
Its suggestions are drafts to validate, not truths to accept: the careers it surfaces should be
checked against real job postings and your own judgement, and the identity statement it drafts
needs your editing to be authentic and accurate. Notably, it helps you *explore and articulate*
but does not show live job listings — you still search real openings on job platforms yourself.
And as with any AI tool, avoid entering sensitive or proprietary information. Treat Career
Dreamer as a way to reflect, explore, and draft — accelerating the human work of the job search,
not replacing your own discernment. The AI-tool theme continues through the section, applied to
resumes and interview preparation. The next lesson builds a structured plan for the search
itself.
"""

CONTENT["Job Search Plan (Using AI Tools)"] = r"""
Searching with a plan
-----------------------

A job search pursued reactively — applying to whatever appears, with no structure — is
exhausting and ineffective. A **job search plan** brings the same deliberateness to finding a
role that the course brought to analysis: clear goals, a structured process, and (increasingly)
AI tools to assist. Closing the identity stage, this lesson covers planning a job search and
using AI tools within it.

Why plan a job search
----------------------

A job search is a *project*, and like any project it benefits from a plan. Without one, the
search drifts — scattershot applications, no sense of progress, effort spent inefficiently, and
the discouragement that comes from randomness. A plan gives the search structure: what roles you
are targeting, where you will look, how you will track progress, and what steps move you toward
the goal. This turns an overwhelming, open-ended undertaking into a manageable process with
direction — the big-picture-first discipline from the foundations, applied to the job hunt.

Elements of a job search plan
-------------------------------

A useful plan addresses several dimensions:

- **Target roles and criteria** — what kind of role you are seeking (data analyst, and what
  variations), and your criteria (industry, location, level), grounded in your career identity.
- **Where to search** — the platforms and channels you will use (the job-platforms and
  networking lessons ahead), rather than relying on one source.
- **Materials** — the resume, profile, and career identity statement you will tailor and deploy
  (the application-stage lessons).
- **Tracking** — a system for recording applications, their status, and follow-ups (a dedicated
  lesson), so nothing is lost and progress is visible.
- **A routine** — a sustainable rhythm of searching, applying, and following up, so the search
  is consistent rather than sporadic.

Together these make the search organised, trackable, and sustainable.

Using AI tools in the search
------------------------------

AI tools can assist at many points in the plan, and using them well is increasingly part of an
effective search:

- **Exploration and identity** — tools like Career Dreamer (the previous lesson) to explore
  directions and draft a career identity.
- **Materials** — AI assistants (such as Gemini) to help draft and tailor resumes, cover
  letters, and profiles (the resume lessons ahead).
- **Preparation** — AI tools to help prepare for interviews (the interview-stage lessons).
- **Organisation** — AI alongside spreadsheets to help track and manage applications (the
  tracking lesson).

Used well, AI accelerates the laborious parts of the search — drafting, tailoring, organising —
freeing your effort for the parts that need human judgement (targeting, relationship-building,
genuine self-presentation).

Why this matters
----------------

A structured, AI-assisted job search is more effective and less demoralising than a reactive
one. Structure provides direction and a sense of progress; AI tools handle drudgery and provide
starting points; together they make the search efficient and sustainable over the weeks or
months it may take. Approaching the job hunt as a planned project — as this course approached
analysis — is what turns it from an anxious scramble into a manageable, navigable process.

The caveat
------------

A plan and its tools serve the search; they are not the search itself, and two cautions apply.
A plan can become *procrastination* — endlessly refining the plan or tinkering with tools
instead of actually applying and reaching out; at some point the plan must translate into the
real, sometimes uncomfortable actions (applying, networking, interviewing) that produce
results. And AI tools *assist* but do not *replace* the human core of a job search: the genuine
self-presentation, the real relationships, and the actual interviews are yours to do, and
over-relying on AI to generate generic materials produces exactly the impersonal applications
that fail. Plan enough to give the search direction, use AI to accelerate the drudgery, and then
do the real human work the plan is meant to enable. This completes the identity stage; the next
stage turns to applying — resumes, profiles, platforms, and networking.
"""


MINDMAP.update({
    "Transferable Skills": [
        "Career Identity Statement",
        "Career Dreamer (AI Tool for Career Exploration)",
        "Tailoring Your Resume",
        "Job Search Plan (Using AI Tools)",
    ],
    "Career Identity Statement": [
        "Transferable Skills",
        "Career Dreamer (AI Tool for Career Exploration)",
        "Building a Professional Online Presence (Personal Brand)",
        "Tailoring Your Resume",
    ],
    "Career Dreamer (AI Tool for Career Exploration)": [
        "Career Identity Statement",
        "Transferable Skills",
        "Job Search Plan (Using AI Tools)",
        "Using AI to Improve and Tailor Your Resume",
    ],
    "Job Search Plan (Using AI Tools)": [
        "Career Dreamer (AI Tool for Career Exploration)",
        "Choosing the Right Job Platforms",
        "Networking for Job Search",
        "Career Identity Statement",
    ],
})


# ======================================================================
# Section 8 — Job Search / Stage: apply  (jobsearch 005-008)
# ======================================================================

GLOSS.update({
    "Tailoring Your Resume":
        "adapting a resume to each specific role rather than sending one generic version",
    "Using AI to Improve and Tailor Your Resume":
        "using AI assistants to draft, refine, and tailor a resume — with human judgement",
    "Building a Professional Online Presence (Personal Brand)":
        "shaping how you appear online — profile, presence, and personal brand",
    "Choosing the Right Job Platforms":
        "selecting the job platforms and channels that fit your target roles",
})

CONTENT["Tailoring Your Resume"] = r"""
One resume per role
--------------------

The single most impactful resume habit is also the most neglected: **tailoring** it to each
role, rather than sending one generic version everywhere. Opening the application stage, this
lesson covers why tailoring matters and how to do it — the practice that turns a resume from a
static document into a targeted argument for a specific job.

Why tailoring matters
----------------------

A generic resume makes a generic impression, and a tailored one makes a targeted case. Two
forces make tailoring essential. First, a hiring manager scanning a resume is asking "does this
person fit *this* role?" — and a resume aimed at the specific role answers that far more
convincingly than a one-size-fits-all document. Second, many applications pass through
*applicant tracking systems* that screen for relevance to the posting, so a resume echoing the
role's actual language is more likely to reach a human at all. Tailoring addresses both: it
speaks to the specific role for both the machine and the person.

How to tailor a resume
-----------------------

Tailoring is systematic, not guesswork:

- **Read the job posting closely** — identify the skills, tools, and qualifications it
  emphasises, and the language it uses. The posting is a specification of what they want.
- **Match your relevant experience to it** — foreground the experiences, skills, and
  accomplishments that align with the posting, and de-emphasise the irrelevant. The same
  background can be presented to highlight different aspects for different roles.
- **Mirror the key language** — use the posting's terms for skills and tools where they
  genuinely apply (if it says "data visualization," and you do that, use that phrase), which
  helps both tracking systems and human readers see the match.
- **Lead with your strongest fit** — your career identity statement (from the previous stage),
  tuned toward this role, and the most relevant qualifications, prominent.

The result is a resume that reads as *written for this role* — because it was.

Tailoring and your career identity
------------------------------------

Tailoring builds on the career identity work. Your Career Identity Statement provides the core
narrative; tailoring *tunes* that narrative toward each specific role — same underlying identity
and truthful history, emphasised differently to fit each opportunity. This is not creating a new
person for each application but presenting your genuine self in the light most relevant to each
role, which is exactly what effective tailoring is.

Why this matters
-----------------

Tailoring dramatically improves a resume's effectiveness for modest effort. A generic resume
competes poorly because it makes the reader work to see the fit; a tailored one makes the fit
obvious, clearing both the automated screen and the human scan. Given that a strong application
can hinge on this, tailoring is among the highest-return job-search habits — and the next lesson
shows how AI tools can make it faster.

The caveat
------------

Tailoring must stay *honest*, and it has a labour cost to manage. The line is firm:
emphasising and framing your genuine experience to fit a role is legitimate tailoring;
*fabricating* skills or experience you do not have to match a posting is lying, which fails at
the interview and is disqualifying if discovered — the honesty principle, at the application
stage. Tailor by *reframing truth*, never by inventing it. And tailoring every application by
hand is time-consuming, which tempts people back to generic resumes; the resolution is partly
efficiency (AI assistance, the next lesson) and partly focus (tailor thoroughly for the roles
you most want, rather than mass-applying generically). Honest tailoring, done efficiently for
the roles that matter — that is the balance. The next lesson covers using AI to tailor.
"""

CONTENT["Using AI to Improve and Tailor Your Resume"] = r"""
AI as a resume assistant
--------------------------

Tailoring a resume well takes effort, and AI tools can make it faster and often better —
drafting, refining, and adapting resume content, with your judgement steering. This lesson
covers using AI to improve and tailor a resume, applying the section's AI-assisted approach to
the central application document, while keeping the human firmly in control.

How AI can help with a resume
-------------------------------

AI assistants (such as Google's Gemini, among others) can support resume work at several points:

- **Drafting and phrasing** — turning a plain description of what you did into clear,
  accomplishment-focused resume language, and suggesting stronger action verbs and phrasing.
- **Tailoring to a posting** — given a job posting and your resume, an AI can suggest how to
  align the resume with the role's emphasised skills and language (the tailoring from the
  previous lesson, accelerated).
- **Highlighting accomplishments** — helping reframe responsibilities as quantified
  achievements ("increased X by Y") rather than mere duties, which resumes reward.
- **Catching errors** — spotting typos, awkward phrasing, and inconsistencies a tired writer
  misses.

Used this way, AI is a fast, tireless drafting-and-editing partner that raises the quality and
speed of resume work.

Keeping the human in control
------------------------------

The crucial principle is that AI *assists* the resume; it does not *author* your career. The
resume must remain *truthful* — describing your real experience and skills, not fabrications an
AI might generate to fit a posting — and it must remain *authentically yours*, in your voice and
reflecting your genuine background, not a generic AI-produced document indistinguishable from
everyone else's. You direct the AI, supply the true content, and judge and edit its output; it
speeds and sharpens the work, but the resume is yours and must be accurate. AI is the assistant;
you are the author and the source of truth.

Why this matters
-----------------

Used well, AI removes much of the friction that makes tailoring tedious — the drafting,
rephrasing, and adapting that discouraged people from tailoring at all — making thorough,
per-role tailoring practical. This directly improves applications: more roles get a genuinely
tailored resume, phrased well, in less time. For the specific, repetitive work of adapting
resume language to postings, AI is a genuine efficiency gain, which is why using it competently
is increasingly part of an effective search.

The caveat
------------

The dangers here are acute and worth stating plainly. AI will readily generate *plausible
falsehoods* — skills, achievements, or experience you do not have — if prompted to match a
posting, and including these is lying on your resume, exposed at interview and disqualifying if
found; the human must ensure every claim is *true*. AI also produces *generic, homogenised*
output if used lazily — resumes that read like every other AI-written resume, losing the
specific, authentic detail that makes a candidate distinct; the human must keep the resume in
their own voice with their real specifics. And pasting sensitive personal information into AI
tools carries privacy considerations. The discipline: use AI to *phrase and adapt your true
experience efficiently*, review every word for accuracy and authenticity, and never let it
invent or homogenise. AI sharpens an honest resume; it must not manufacture a false or faceless
one. The next lesson turns from the resume to your broader online presence.
"""

CONTENT["Building a Professional Online Presence (Personal Brand)"] = r"""
How you appear online
----------------------

A resume is no longer the only thing employers see — they look you up, and your **online
presence** shapes their impression before or after any interview. Deliberately building a
professional online presence, a **personal brand**, is now part of a serious job search. This
lesson covers shaping how you appear online as a professional.

What a professional online presence is
----------------------------------------

Your **online presence** is the picture of you that emerges from what is publicly visible —
professional profiles (such as LinkedIn), any portfolio or work you have shared, contributions
and posts, and search results for your name. Your **personal brand** is the deliberate shaping
of that picture: the consistent professional identity you present, reflecting your skills,
interests, and the value you offer. It is the online extension of your career identity — the
same professional self, expressed across your public profiles and presence.

Building it deliberately
-------------------------

A strong professional presence is built, not left to chance:

- **A complete, professional profile** — a well-crafted profile on the platform that matters
  for your field (LinkedIn for most professional roles), with your career identity, experience,
  and skills clearly presented.
- **Demonstrated work** — where relevant, a portfolio or shared examples of your work (for a
  data analyst, sample analyses, visualizations, or projects) that *show* rather than merely
  claim your skills.
- **Consistency** — a coherent professional identity across your profiles, aligned with your
  resume and career identity, so the picture is unified rather than contradictory.
- **Appropriate activity** — thoughtful engagement in your field (sharing, commenting,
  connecting) that signals genuine interest and involvement.

Together these present you, to anyone who looks, as a credible professional in your target
field.

Managing what is already there
--------------------------------

Building a presence also means *auditing* what already exists. Since employers may search your
name, it is worth reviewing what is publicly visible and ensuring nothing there undermines your
professional image — adjusting privacy on personal content, and removing or addressing anything
that conflicts with the professional identity you want to present. The goal is that what an
employer finds *supports* your candidacy, or at least does not detract from it.

Why this matters
-----------------

A professional online presence increasingly *complements the resume* — it is where employers
verify and expand their impression, where a portfolio can *demonstrate* the skills a resume
claims, and where being findable and credible can create opportunities (recruiters search for
candidates). For a data analyst especially, a portfolio of real work is powerful evidence that a
resume alone cannot provide. Neglecting online presence forgoes this, and can let an unmanaged
or inconsistent picture work against you; building it deliberately turns it into an asset.

The caveat
------------

A personal brand must be *authentic and proportionate*. Manufacturing a polished but false
professional persona — claimed expertise you lack, a portfolio misrepresenting your role in work
— is the same dishonesty as a fabricated resume, and equally exposed on contact; the presence
should reflect your genuine skills and identity. And building a presence can become *performative
over-investment* — endless profile-polishing and content-posting that substitutes for the actual
work of applying, networking, and building real skills. The presence supports a job search
grounded in genuine ability; it is not a substitute for that ability, nor should its cultivation
crowd out the substantive work. An authentic, credible presence that accurately showcases real
skills — that is the aim. The next lesson covers where to actually search for roles: job
platforms.
"""

CONTENT["Choosing the Right Job Platforms"] = r"""
Where to look
--------------

Job openings are scattered across many platforms and channels, and *where* you search shapes
what you find. **Choosing the right job platforms** — the sites and channels that fit your
target roles — makes a search more efficient and effective. This lesson covers selecting job
platforms wisely, closing much of the application stage's groundwork.

The landscape of job platforms
--------------------------------

Roles are advertised across several kinds of channel, each with strengths:

- **General job boards** — large, broad platforms listing roles across industries, useful for
  breadth and volume.
- **Professional networking platforms** — sites (such as LinkedIn) that combine job listings
  with professional networking and profiles, where being present and connected matters.
- **Company career pages** — employers' own sites, where roles are posted first or exclusively;
  applying directly to a target company often goes here.
- **Niche and specialised boards** — platforms focused on a field, industry, or role type,
  which can surface relevant roles a general board buries.
- **Networking and referrals** — not a platform but a channel (the next lesson), often the most
  effective route of all.

No single channel covers everything, so the question is which mix fits *your* search.

Choosing wisely for your search
---------------------------------

The right platforms depend on your target roles and situation:

- **Match the platform to the role** — for professional data-analyst roles, a professional
  networking platform and relevant boards typically fit better than the broadest general boards
  alone.
- **Go where your target employers post** — if specific companies interest you, their career
  pages are essential.
- **Use a focused set, well** — a few well-chosen platforms used thoroughly (with a good
  profile, saved searches, and alerts) beats a shallow presence spread across many.
- **Do not neglect networking** — the referral channel (next lesson) often outperforms any job
  board, so it belongs in the mix.

Choosing deliberately concentrates effort where the relevant roles actually are.

Why this matters
-----------------

Searching the *right* platforms makes the whole search more efficient — you see more relevant
roles and waste less effort on channels that do not surface them. Different platforms genuinely
carry different opportunities (some roles appear only on company pages, some only through
networks), so the choice of where to look directly determines what you can find and apply to.
Matching platforms to your target roles is thus a simple but consequential decision.

The caveat
------------

Two opposite mistakes await. *Spreading too thin* — maintaining a shallow presence on many
platforms, checking none properly — dilutes effort and misses roles; a focused set used well is
better. But *over-narrowing* — relying on a single platform — misses the roles that appear only
elsewhere, especially those found through company pages and networking rather than job boards.
And a deeper caution: job platforms show only the *advertised* market, while many roles are
filled through networks and referrals before or without public posting, so a platform-only
search misses a large hidden portion of opportunities. The balance is a focused set of
well-matched platforms *plus* active networking — which the next lesson addresses directly. This
advances the application stage; the next lesson covers the powerful channel of networking.
"""


MINDMAP.update({
    "Tailoring Your Resume": [
        "Using AI to Improve and Tailor Your Resume",
        "Career Identity Statement",
        "Building a Professional Online Presence (Personal Brand)",
        "Transferable Skills",
    ],
    "Using AI to Improve and Tailor Your Resume": [
        "Tailoring Your Resume",
        "Career Dreamer (AI Tool for Career Exploration)",
        "Building a Professional Online Presence (Personal Brand)",
        "Career Identity Statement",
    ],
    "Building a Professional Online Presence (Personal Brand)": [
        "Tailoring Your Resume",
        "Choosing the Right Job Platforms",
        "Networking for Job Search",
        "Career Identity Statement",
    ],
    "Choosing the Right Job Platforms": [
        "Networking for Job Search",
        "Building a Professional Online Presence (Personal Brand)",
        "Job Application Tracking (Using AI + Spreadsheets)",
        "Job Search Plan (Using AI Tools)",
    ],
})


# ======================================================================
# Section 8 — Job Search / apply (close) + interview (open)  (jobsearch 009-011)
# ======================================================================

GLOSS.update({
    "Job Application Tracking (Using AI + Spreadsheets)":
        "systematically recording applications and their status, with AI and spreadsheets",
    "Networking for Job Search":
        "building and using professional relationships to find and reach opportunities",
    "Interview Preparation":
        "readying for interviews — research, practice, and knowing your material",
})

CONTENT["Job Application Tracking (Using AI + Spreadsheets)"] = r"""
Keeping track of the search
-----------------------------

A real job search means many applications over weeks or months, and without a system they
become impossible to manage — deadlines missed, follow-ups forgotten, duplicates sent.
**Application tracking** — systematically recording applications and their status, using
spreadsheets and AI — keeps the search organised. This lesson covers tracking, a practical
discipline that brings order to the search.

Why track applications
-----------------------

The volume of a serious search overwhelms memory. Applying to dozens of roles, each with its
own status, contacts, deadlines, and follow-ups, is far more than anyone can track in their
head — and the cost of not tracking is real: missed follow-up windows, forgotten deadlines,
embarrassing duplicate applications, and no sense of what is working. A tracking system solves
this, giving a clear picture of every application's state and what needs attention — the same
"make the invisible visible" logic that dashboards brought to data, applied to the job search.

Tracking with a spreadsheet
----------------------------

The natural tool is a **spreadsheet** — fittingly, given the course — with one row per
application and columns for the key information:

- **Company and role** — what and where you applied.
- **Date applied** and **status** — when, and where it stands (applied, interviewing,
  rejected, offer).
- **Contacts** — recruiter or connection names, for follow-up.
- **Deadlines and follow-up dates** — what needs action and when.
- **Notes** — anything relevant (referral source, interview details, impressions).

This is a direct application of the spreadsheet skills from earlier sections — a simple table
that turns a chaotic search into an organised, filterable, sortable record. You can filter to
"awaiting response," sort by follow-up date, and see the whole search at a glance.

Adding AI assistance
---------------------

AI tools can complement the spreadsheet: helping set up and structure the tracker, drafting
follow-up messages, summarising a role or your notes, and prompting next actions. The spreadsheet
holds the structured record; AI can assist with the surrounding work (drafting, summarising,
organising), combining a reliable data structure with AI's help on the language and prompts — the
section's "AI plus your own tools and judgement" pattern, applied to staying organised.

Why this matters
-----------------

Tracking is unglamorous but decisive for a sustained search. It ensures nothing is dropped
(every follow-up happens, every deadline is met), prevents errors (no duplicate applications),
and provides visibility (what is in progress, what is working) that both improves the search and
sustains morale through a long process. It also embodies the professionalism the whole course
teaches — the same organisation and attention to detail that defines good analytical work,
turned on the search itself.

The caveat
------------

Tracking serves the search and can be *over-built* — an elaborate system with dozens of fields
and automations that consumes more time to maintain than it saves, or becomes a way to feel
productive without actually applying. The tracker should be as simple as it can be while doing
its job: capture what you genuinely need to stay organised, and no more. And tracking is a
*support* for the search, not the search itself — time spent perfecting the tracker is not time
spent applying, networking, or preparing, which are what actually produce results. A simple,
maintained tracker that keeps the real work on course — that is the aim, not a monument to
organisation. The next lesson covers the channel that often matters most: networking.
"""

CONTENT["Networking for Job Search"] = r"""
The most powerful channel
---------------------------

Many roles are filled through *people* — relationships, referrals, and connections — rather
than through applications to public postings. **Networking**, building and using professional
relationships, is often the single most effective job-search channel, and this lesson covers it:
what it is, how to do it, and why it matters so much. It closes the application stage on its most
important note.

Why networking matters so much
--------------------------------

A large share of roles are filled through networks — referrals, introductions, and connections
— often before or without any public posting, in what is sometimes called the *hidden job
market*. A referral from someone inside a company carries weight a cold application cannot, and
being known to the right people can surface opportunities that never reach a job board. This is
why networking so often outperforms applying to postings: it reaches the roles and the influence
that the public application channel misses. For a job seeker, relationships are frequently the
highest-return investment of effort.

What networking actually is
-----------------------------

Networking is widely misunderstood as something transactional or distasteful; it is better
understood as *building genuine professional relationships*:

- **Connecting with people in your field** — professionals, peers, people at target companies —
  through professional platforms, events, communities, and mutual contacts.
- **Building real relationships** — genuine, mutual professional connections, not extracting
  favours from strangers. The relationship comes first.
- **Informational conversations** — learning from people about their work, their company, and
  the field, which builds relationships and knowledge (and sometimes surfaces opportunities).
- **Staying connected and offering value** — maintaining relationships over time and helping
  others, so the network is mutual rather than one-directional.

Done this way, networking is professional relationship-building, not the awkward
self-promotion people dread.

Networking in practice
-----------------------

Practically, networking for a job search means: cultivating a professional presence and
connections (building on the online-presence lesson), reaching out thoughtfully to people in
your field or target companies, having genuine conversations that build relationships and
understanding, and letting people know — appropriately — that you are exploring opportunities, so
they can think of you. It also means *helping others*, which both is right and makes a network
reciprocal. The referrals and opportunities that networking produces flow from genuine
relationships, so the relationship-building is the substance.

Why this matters
-----------------

Networking reaches the large portion of opportunities that public applications miss, carries the
weight of personal referral that applications lack, and builds relationships that benefit a whole
career, not just one search. For all the effort spent on resumes and applications, relationships
often matter more to landing a role — which is why networking, despite being less comfortable for
many than submitting applications, deserves real investment in a serious search.

The caveat
------------

Networking must be *genuine and reciprocal*, not transactional or exploitative. Approaching
people purely to extract favours, contacting strangers only when you need something, or treating
relationships as mere means is both off-putting and ineffective — people sense it, and it fails.
Real networking is built on genuine professional relationships and mutual value, cultivated over
time, not switched on only when job-hunting; it also means *giving*, not only taking. And
networking complements rather than replaces a sound application effort and genuine skills — a
referral opens a door, but you must still be able to do the job. Authentic relationship-building
that reaches opportunities and carries referrals, grounded in real ability — that is networking
done right. This completes the application stage; the next stage turns to the interview.
"""

CONTENT["Interview Preparation"] = r"""
Readying for the interview
----------------------------

An interview is where applications become offers, and *preparation* is what most separates a
strong interview from a weak one — far more than raw talent or charisma. **Interview
preparation** — researching, practising, and knowing your material — is the subject that opens
the final stage. This lesson covers how to prepare for an interview so you can perform with
genuine confidence.

Why preparation is decisive
-----------------------------

Interview success comes far more from preparation than from natural ability. The candidate who
has researched the company, anticipated the questions, prepared examples, and practised their
delivery consistently outperforms the more talented but unprepared one — because preparation
produces the concrete, relevant, confident answers interviews reward, and the genuine confidence
that comes from being ready. This is the same insight as the presentation section: confidence
and quality come from preparation, not from personality, and nowhere is it truer than in the
interview.

What to prepare
---------------

Thorough interview preparation covers several fronts:

- **Research the company and role** — understand what the organisation does, its context, and
  what the role involves, so you can speak to *why this role* and tailor your answers to their
  needs. This also signals genuine interest.
- **Anticipate the questions** — prepare for the common ones ("tell me about yourself," "why
  this role," "your strengths and weaknesses") and the role-specific technical ones a data
  analyst will face (about your skills, tools, and approach).
- **Prepare your examples** — ready concrete stories of your experience and accomplishments,
  structured to answer behavioural questions (the STAR method, next lesson) — the evidence that
  makes your claims credible.
- **Know your own material** — your resume, your career identity, and your projects cold, since
  the interview will probe them; you must be able to discuss anything you have claimed.
- **Prepare questions to ask** — thoughtful questions for the interviewer, which show engagement
  and help you evaluate the role.

And, as the presentation section stressed, *practising* the delivery — rehearsing answers aloud,
ideally in a mock interview — converts preparation into fluent performance.

Why this matters
-----------------

Preparation transforms the interview from an anxious improvisation into a confident conversation
you are ready for. It produces better answers (concrete, relevant, well-structured), the
composure that comes from readiness, and the genuine engagement that researching the role
enables — all of which interviewers respond to. Given that the interview often decides the
outcome, and that preparation is the largest controllable factor in it, preparing thoroughly is
the highest-return interview investment there is.

The caveat
------------

Preparation should make you *ready*, not *rigid* or *robotic*. Over-preparing to the point of
memorising scripted answers backfires — it sounds canned, and it leaves you thrown when a
question differs from the one you rehearsed; the goal is to know your material and examples well
enough to respond *genuinely and flexibly*, not to recite. Real interviews are conversations,
and preparation should enable a natural, responsive one, not a performance of memorised lines.
And preparation cannot substitute for genuine ability and honesty — it helps you present real
skills well, not fake skills you lack, which the interview is designed to test. Prepared enough
to be confident and fluent, genuine enough to be conversational and honest — that is the balance.
The next lessons cover a specific interview technique and AI-assisted preparation.
"""


MINDMAP.update({
    "Job Application Tracking (Using AI + Spreadsheets)": [
        "Choosing the Right Job Platforms",
        "Networking for Job Search",
        "Job Search Plan (Using AI Tools)",
        "Tailoring Your Resume",
    ],
    "Networking for Job Search": [
        "Choosing the Right Job Platforms",
        "Building a Professional Online Presence (Personal Brand)",
        "Job Application Tracking (Using AI + Spreadsheets)",
        "Interview Preparation",
    ],
    "Interview Preparation": [
        "STAR Method (Behavioral Interview)",
        "Using AI (NotebookLM) for Interview Preparation",
        "Networking for Job Search",
        "Career Identity Statement",
    ],
})


# ======================================================================
# Section 8 — Job Search / interview (close)  (jobsearch 012-015)  -- SECTION 8 & COURSE COMPLETE
# NotebookLM + Gemini Live facts grounded via web_search (both Google; NotebookLM source-grounded
# research assistant with Studio artifacts / Audio Overviews; Gemini Live real-time voice mode,
# interruptible, positioned for rehearsing important moments). Written in original prose.
# ======================================================================

GLOSS.update({
    "STAR Method (Behavioral Interview)":
        "a structure for behavioural answers — Situation, Task, Action, Result",
    "Using AI (NotebookLM) for Interview Preparation":
        "using a source-grounded AI research tool to prepare from role and company material",
    "Practicing Interviews with AI (Gemini Live)":
        "rehearsing interviews aloud in real-time voice conversation with an AI",
    "Post-Interview Strategy":
        "what to do after an interview — follow up, reflect, and handle the outcome",
})

CONTENT["STAR Method (Behavioral Interview)"] = r"""
Structuring the behavioural answer
------------------------------------

Interviews increasingly use **behavioural questions** — "tell me about a time when…" — that
ask for specific past examples, on the theory that past behaviour predicts future behaviour.
The **STAR method** is a structure for answering these clearly and completely, and this lesson
covers it — a simple, powerful framework for the behavioural interview.

What behavioural questions are
--------------------------------

Behavioural questions ask you to describe a *specific situation* from your experience: "tell me
about a time you solved a difficult problem," "describe a situation where you handled
conflicting priorities," "give an example of when you used data to make a decision." They seek
concrete evidence of how you have actually behaved, rather than hypothetical or general claims —
which is why prepared, specific examples (from the preparation lesson) matter so much. The
challenge is answering them in a way that is complete and clear rather than rambling or vague,
and that is what STAR provides.

The STAR structure
------------------

**STAR** is an acronym for the four parts of a strong behavioural answer:

- **Situation** — briefly set the context: what was the situation, where and when. Enough
  background for the example to make sense.
- **Task** — what was *your* responsibility or the challenge you faced in that situation. What
  needed to be done, and your role in it.
- **Action** — what *you* specifically did — the steps you took to address the task. This is
  the heart of the answer, and it should focus on *your* actions (not "we"), showing what you
  contributed.
- **Result** — the outcome of your actions — what happened, ideally quantified ("reduced errors
  by 30%," "delivered on time"), and what you learned.

Walking through Situation, Task, Action, Result gives a complete, structured answer: the context,
your responsibility, what you did, and how it turned out — everything the interviewer needs, in a
logical order.

Why STAR works
--------------

STAR works because it ensures the answer is *complete and focused*. Without a structure,
behavioural answers tend to wander, omit the outcome, or blur what *you* did versus what the team
did; STAR prevents this by prompting each essential element in turn. It keeps the answer concise
and relevant (the interviewer follows the arc), foregrounds *your* specific contribution (the
Action), and includes the result that shows impact — which is exactly what interviewers are
assessing. For a data analyst, STAR is ideal for the "tell me about an analysis you did" questions:
the situation, your analytical task, the actions you took, and the result it drove.

The caveat
------------

STAR structures an answer but cannot supply its *substance* — you still need genuine, relevant
examples to structure, which is why preparing real STAR stories in advance (from the preparation
lesson) matters. Applied mechanically, STAR can also sound *formulaic* — a rigid recitation of
labels — so the structure should guide a natural answer, not turn it into an obvious template.
And the examples must be *honest*: STAR is for presenting real experiences clearly, not for
fabricating impressive-sounding stories, which probing follow-up questions ("what exactly did you
do?") tend to expose. Prepare genuine examples, structure them with STAR so they land, and
deliver them naturally and truthfully. The next lessons cover AI tools that help prepare and
practise exactly these answers.
"""

CONTENT["Using AI (NotebookLM) for Interview Preparation"] = r"""
Preparing with a source-grounded AI
-------------------------------------

Interview preparation means absorbing a lot of material — the job posting, the company, your own
background — and an AI tool can help organise and study it. **NotebookLM**, Google's
source-grounded research assistant, is well suited to this, because it works from *your* uploaded
material rather than general knowledge. This lesson covers using NotebookLM to prepare for
interviews, applying the section's AI-assisted approach to interview readiness.

What NotebookLM is
------------------

NotebookLM is an AI-powered research assistant from Google that is **source-grounded** — you
upload documents (the sources), and it answers questions and generates material based only on
*those* sources, with citations back to them, rather than drawing on general training data. This
grounding is its defining feature: it becomes, in effect, an expert on the specific material you
give it, which reduces the risk of it inventing information and keeps its output tied to your
actual sources. You organise material into "notebooks," and a studio panel can generate study
aids from your sources — summaries, briefing documents, study guides, and even audio "deep dive"
discussions between AI hosts that you can listen to.

Using it for interview preparation
------------------------------------

For interview preparation, NotebookLM's source-grounding fits naturally. You can gather the
relevant material — the job posting, information about the company, the role description, and your
own resume and notes — as sources, and then use the tool to prepare:

- **Understand the role and company** — ask questions grounded in the posting and company
  material to build the understanding the preparation lesson called for.
- **Generate study aids** — have it produce a briefing document or study guide from the sources,
  consolidating what you need to know.
- **Anticipate questions** — use it to think through what the role emphasises and what you might
  be asked, grounded in the actual posting.
- **Listen and review** — generate an audio overview of the material to review your preparation
  while away from the desk.

Because its answers are tied to the sources you provide, the preparation stays grounded in the
*actual* role and company rather than generic advice.

Why this helps
--------------

NotebookLM helps by *organising and synthesising* the specific material an interview requires —
turning a pile of documents about the role, company, and your background into queryable,
studyable form, grounded in those real sources. This suits interview preparation especially well
because that preparation is fundamentally about absorbing *specific* material (this role, this
company), which is exactly what a source-grounded tool is built for. It makes preparation more
organised and thorough for less effort.

The caveat
------------

NotebookLM's grounding reduces but does not eliminate the need for *your own judgement and
verification*. It works from the sources you give it, so the quality of your preparation depends
on the quality and completeness of those sources — and while grounding reduces fabrication, AI
can still misinterpret or miss nuance, so its output is a study aid to verify, not gospel. It
organises and synthesises material; it does not do the actual *preparing* — the understanding,
the practising, the internalising — which remain yours. And, as with any AI tool, be mindful of
what personal information you upload. Use NotebookLM to organise and study the real material
efficiently, while doing the genuine preparation yourself. The next lesson covers practising the
interview aloud with a different AI tool.
"""

CONTENT["Practicing Interviews with AI (Gemini Live)"] = r"""
Rehearsing out loud
--------------------

Knowing your answers is not the same as *delivering* them well — and the presentation section's
lesson holds, that fluent delivery comes from practice. **Gemini Live**, Google's real-time voice
conversation mode, lets you rehearse interviews *aloud*, in natural back-and-forth conversation,
which is the closest an AI tool comes to a mock interview. This lesson covers practising interviews
with Gemini Live, the practice half of interview readiness.

What Gemini Live is
-------------------

Gemini Live is a real-time, spoken conversation mode in Google's Gemini app: instead of typing
prompts and reading replies, you *talk* to the AI and it responds in a synthesized voice, in a
natural back-and-forth. Its defining quality is conversational flow — you can interrupt it,
change direction, and follow up mid-response, and it adapts, which makes the exchange feel more
like talking with a person than querying an assistant. It runs on the mobile app, and Google
explicitly positions it for things like rehearsing for important moments — which is exactly the
interview-practice use case.

Using it to practise interviews
---------------------------------

Gemini Live's spoken, interactive nature makes it a practice partner for the *delivery* of
interview answers:

- **Mock interview practice** — ask it to act as an interviewer for a data-analyst role and pose
  questions, then answer *aloud*, practising the actual speaking that a real interview requires.
- **Practising STAR answers** — rehearse your behavioural answers (from the STAR lesson) out
  loud, hearing how they sound and tightening them.
- **Rehearsing the common questions** — practise "tell me about yourself," "why this role," and
  the technical questions verbally, building fluency and reducing the nerves the preparation
  lesson addressed.
- **Iterating** — because it is conversational, you can practise, adjust, and practise again in a
  natural flow.

The value is *speaking practice* — converting prepared answers into fluent spoken delivery, which
only rehearsal aloud can build.

Why this helps
--------------

Gemini Live helps because it lets you practise the *spoken performance* of an interview, not just
the content — and delivery, as the presentation section stressed, comes from practice. Rehearsing
answers aloud in a conversational back-and-forth builds the fluency and composure that reading
answers silently cannot, and it is available any time, without needing another person to run a
mock interview. For the practice that turns preparation into confident delivery, a real-time voice
AI is a genuinely useful rehearsal partner.

The caveat
------------

Practising with an AI is useful rehearsal but not a full substitute for the real thing or for
human feedback. An AI interviewer approximates the experience but does not perfectly replicate a
real interviewer's judgement, follow-ups, and dynamics, and it cannot give the nuanced human
feedback a mentor or peer mock interview can — so AI practice is best complemented by practising
with real people where possible. It also cannot supply genuine preparation or real ability: it
helps you *rehearse* answers you have prepared, not manufacture competence you lack. And the usual
caution about what you share with AI tools applies. Use Gemini Live to build fluency through spoken
rehearsal, alongside human practice and genuine preparation. The final lesson covers what happens
after the interview.
"""

CONTENT["Post-Interview Strategy"] = r"""
After the interview
--------------------

The interview is not the end of the process — what you do *afterward* can affect the outcome and,
regardless of result, your longer job search. **Post-interview strategy** covers following up,
reflecting, and handling the outcome well. As the final lesson of the job-search section and of
the course, it closes the arc from learning data skills to landing the role.

Following up
------------

The immediate post-interview action is a *thank-you* and follow-up:

- **Send a thank-you** — a prompt, genuine thank-you message to your interviewer(s) expresses
  appreciation, reaffirms your interest, and keeps you memorable. It is a small courtesy that is
  often noticed, and its absence sometimes noticed too.
- **Reinforce your fit, briefly** — a good follow-up can briefly reiterate why you are a strong
  fit or add a point you wish you had made, turning the note into a light final argument.
- **Be timely and professional** — send it soon after the interview, keep it concise and
  professional, and personalise it to the conversation.

This simple step reflects the professionalism and communication the course has emphasised, applied
to the interview's aftermath.

Reflecting and improving
--------------------------

Beyond the note, reflect on the interview to *improve*:

- **What went well and what did not** — which questions you handled strongly, which you
  struggled with, so you can prepare better next time. Each interview is practice for the next.
- **Questions you were unprepared for** — note them and prepare answers, strengthening your
  readiness (the self-improvement discipline, applied to interviewing).
- **Your own read** — how the role and company felt to you, since an interview is also *your*
  chance to evaluate *them*.

Treating each interview as a source of learning steadily improves your interviewing across a
search.

Handling the outcome
--------------------

However it turns out, handle the result constructively. An offer calls for careful, professional
evaluation and negotiation; a rejection, for *not* taking it as a verdict on your worth —
rejections are a normal, frequent part of any job search, often reflecting fit or factors beyond
your control rather than a personal failing. The constructive response to rejection is to seek any
feedback offered, apply the lessons, and continue — persistence through the inevitable rejections
is part of nearly every successful search. Maintaining professionalism and resilience through
both outcomes serves your search and your reputation.

Why this matters
-----------------

Post-interview actions matter because the process continues past the interview: a thoughtful
follow-up can influence a decision, reflection improves your next interview, and resilience
sustains you through a search that will include rejections before its success. Handling the
aftermath well — professionally, reflectively, and resiliently — is the final piece of an
effective job search, turning each interview, whatever its result, into progress toward the role.

The caveat
------------

Post-interview strategy has its own balance to strike. Follow-up should be *professional, not
pestering* — a thank-you and appropriate, patient follow-up help; repeated or pushy messaging
harms, and after reasonable follow-up you must let the process take its course. Reflection should
*inform improvement, not feed rumination* — learning from an interview is constructive, but
replaying it anxiously or treating a rejection as a personal indictment is neither accurate nor
healthy, and resilience means moving forward. The aim is constructive follow-through and genuine
learning, held with the equanimity that a long search requires.

Concluding the course
----------------------

This lesson closes not only the job-search section but the entire Data Analytics course. You have
travelled a long arc: from the foundations of data-driven thinking, through preparing and cleaning
data, analysing it with spreadsheets and SQL, visualizing and communicating it, and automating the
whole workflow with Python — and finally, to presenting your skills and landing the role. The
throughline has been a set of principles as much as techniques: reason from evidence, respect the
integrity of data, communicate honestly, and pursue the work with care and rigour. These are what
make a data analyst, more than any single tool. With the technical skills built and the job-search
craft in hand, you are equipped not just to *do* data analysis but to build a career in it — and,
more importantly, to keep learning, since the field will keep changing and the capacity to learn is
the most durable skill of all. The tools will evolve; the disciplines endure.
"""


MINDMAP.update({
    "STAR Method (Behavioral Interview)": [
        "Interview Preparation",
        "Using AI (NotebookLM) for Interview Preparation",
        "Practicing Interviews with AI (Gemini Live)",
        "Career Identity Statement",
    ],
    "Using AI (NotebookLM) for Interview Preparation": [
        "Interview Preparation",
        "STAR Method (Behavioral Interview)",
        "Practicing Interviews with AI (Gemini Live)",
        "Career Dreamer (AI Tool for Career Exploration)",
    ],
    "Practicing Interviews with AI (Gemini Live)": [
        "Using AI (NotebookLM) for Interview Preparation",
        "STAR Method (Behavioral Interview)",
        "Interview Preparation",
        "Post-Interview Strategy",
    ],
    "Post-Interview Strategy": [
        "Practicing Interviews with AI (Gemini Live)",
        "Interview Preparation",
        "Networking for Job Search",
        "STAR Method (Behavioral Interview)",
    ],
})
