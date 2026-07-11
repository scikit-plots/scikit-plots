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
