:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _data-analytics-index:

:raw-html:`<div style="text-align:center"><strong>` 📈 Data Analytics
|br| |full_version| - |today|
:raw-html:`</strong></div>`

========================================================================
📈 Data Analytics
========================================================================

A hands-on data-analytics curriculum in eight sections, from first principles to the job hunt — rewritten and cross-linked for scikit-plots.

*216 of 216 lessons written across 8 of 8 sections.* Each section below is its own browsable mini-course; use the filter to search every lesson at once.

.. raw:: html

   <div style="text-align:center;margin:0.4rem 0 0.4rem">
   <input id="term-filter" type="search" autocomplete="off" spellcheck="false"
           placeholder="&#128269;&nbsp; Type to filter all lessons lessons &mdash; by title or keyword&hellip;"
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
        if(cnt){{cnt.textContent=(q&&az)?(n+' of {n_items} match'+(n===1?'':'s')):'';}}
     });
   });
   </script>


.. dropdown:: 🌱 1. Foundations
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   The case for data, the analysis process and data life cycle, analytical thinking, and the core tools of the trade.  — **27/27 lessons.** :doc:`Open the section <1_foundations/index>`

   * :doc:`001 · Why Data Analytics Matters Today — why organisations that decide with data outperform those that decide by instinct <1_foundations/001-why-data-analytics-matters-today>`
   * :doc:`002 · How Data Analytics Improves the Workplace — where analytics pays off day to day: operations, decisions, and shared facts <1_foundations/002-how-data-analytics-improves-the-workplace>`
   * :doc:`003 · Data-Driven Decision-Making — the loop from question to data to decision, and the evidence it works <1_foundations/003-data-driven-decision-making>`
   * :doc:`004 · Detectives and Data Analysts — the investigator's mindset: questions, evidence, and conclusions that hold up <1_foundations/004-detectives-and-data-analysts>`
   * :doc:`005 · The Six Phases of the Data Analysis Process — ask, prepare, process, analyze, share, act — the map for every project in this course <1_foundations/005-the-six-phases-of-the-data-analysis-process>`
   * :doc:`006 · The Origins of Data Analysis and the Many Ways to Structure It — from early statistics to EDA and CRISP-DM: one process, many framings <1_foundations/006-the-origins-of-data-analysis-and-the-many-ways-to-structure-it>`
   * :doc:`007 · Understanding the Data Ecosystem — the interlocking pieces — sources, storage, tools, people — data moves through <1_foundations/007-understanding-the-data-ecosystem>`
   * :doc:`008 · Understanding the Data Analysis Process and the Data Life Cycle — two different journeys: what the analyst does vs. what happens to the data <1_foundations/008-understanding-the-data-analysis-process-and-the-data-life-cycle>`
   * :doc:`009 · Understanding the Data Life Cycle — plan, capture, manage, analyze, archive, destroy — the data's own biography <1_foundations/009-understanding-the-data-life-cycle>`
   * :doc:`010 · A Review of the Six Stages of the Data Life Cycle — the six stages consolidated, with what can go wrong at each <1_foundations/010-a-review-of-the-six-stages-of-the-data-life-cycle>`
   * :doc:`011 · The Stages of the Data Analysis Process and Their Roles — what each analysis phase contributes, and what it hands to the next <1_foundations/011-the-stages-of-the-data-analysis-process-and-their-roles>`
   * :doc:`012 · Practical Application of the Data Analysis Process — the six phases run end-to-end on a real-shaped business case <1_foundations/012-practical-application-of-the-data-analysis-process>`
   * :doc:`013 · Analytical Skills and Their Core Components — the five skills — curiosity, context, technical mindset, data design, data strategy <1_foundations/013-analytical-skills-and-their-core-components>`
   * :doc:`014 · Applying Analytical Skills in a Business Context — the five skills at work on a real business problem, phase by phase <1_foundations/014-applying-analytical-skills-in-a-business-context>`
   * :doc:`015 · Analytical Thinking and Its Core Components — the five aspects — visualization, strategy, problem-orientation, correlation, big picture + detail <1_foundations/015-analytical-thinking-and-its-core-components>`
   * :doc:`016 · Analytical Thinking and Questions for Problem Solving — turning the aspects into questions: root causes, gaps, and the unconsidered <1_foundations/016-analytical-thinking-and-questions-for-problem-solving>`
   * :doc:`017 · Root Cause Analysis and Business Applications of the Five Whys — Toyota's why-times-five: digging past symptoms to causes worth fixing <1_foundations/017-root-cause-analysis-and-business-applications-of-the-five-whys>`
   * :doc:`018 · Data-Driven Decision-Making and the Role of Analytical Skills — how the five skills power each step of the decision loop <1_foundations/018-data-driven-decision-making-and-the-role-of-analytical-skills>`
   * :doc:`019 · Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making — the evidence in action: research findings and worked cases where data changed the outcome <1_foundations/019-case-studies-in-data-analysis-and-the-practical-impact-of-data-driven-decision-making>`
   * :doc:`020 · Overview of Core Tools Used by Data Analysts — the working toolkit — spreadsheets, SQL, visualization tools, and code — and when each fits <1_foundations/020-overview-of-core-tools-used-by-data-analysts>`
   * :doc:`021 · The Role of Spreadsheets in Data Analysis and Basic Concepts — the analyst's first instrument: cells, formulas, and functions in a visible grid <1_foundations/021-the-role-of-spreadsheets-in-data-analysis-and-basic-concepts>`
   * :doc:`022 · The Concept and Basic Use of SQL (Query Language) — asking databases questions: SELECT, FROM, WHERE and why they scale <1_foundations/022-the-concept-and-basic-use-of-sql-query-language>`
   * :doc:`023 · The Role and Importance of Data Visualization — why pictures beat tables: seeing structure that summary numbers hide <1_foundations/023-the-role-and-importance-of-data-visualization>`
   * :doc:`024 · Industries Where Data Analysts Work and How Data Is Used — the same craft in many rooms: retail to healthcare to public service <1_foundations/024-industries-where-data-analysts-work-and-how-data-is-used>`
   * :doc:`025 · The Role of Business Tasks in Data Analysis — the question behind the work: how business tasks anchor every analysis <1_foundations/025-the-role-of-business-tasks-in-data-analysis>`
   * :doc:`026 · Fairness in Data Analysis — analysis that does not create or reinforce bias — and a famous failure to learn from <1_foundations/026-fairness-in-data-analysis>`
   * :doc:`027 · Key Factors to Consider When Choosing a Data Analytics Role — industry, company size, specialisation, growth: weighing where to start <1_foundations/027-key-factors-to-consider-when-choosing-a-data-analytics-role>`

.. dropdown:: 🎯 2. Data-Driven Decisions
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Turning questions into decisions: stakeholders, metrics, and communicating results that drive action.  — **27/27 lessons.** :doc:`Open the section <2_data_driven_decisions/index>`

   * :doc:`001 · Using Data Analysis to Choose the Right Advertising Strategy — a worked prediction problem: past campaign data steering the next ad spend <2_data_driven_decisions/001-using-data-analysis-to-choose-the-right-advertising-strategy>`
   * :doc:`002 · Understanding Common Problem Types in Data Analytics — the six shapes of analyst problems, from predictions to patterns <2_data_driven_decisions/002-understanding-common-problem-types-in-data-analytics>`
   * :doc:`003 · Applying Data Analytics Problem Types in Real Business Scenarios — recognising the six types in the wild — and why naming the type speeds the work <2_data_driven_decisions/003-applying-data-analytics-problem-types-in-real-business-scenarios>`
   * :doc:`004 · Why Asking the Right Questions Matters in Data Analytics — SMART questions, and the leading/closed/assuming questions to avoid <2_data_driven_decisions/004-why-asking-the-right-questions-matters-in-data-analytics>`
   * :doc:`005 · The Relationship Between Data and Decision-Making — data-driven vs data-inspired: how evidence and judgement actually combine <2_data_driven_decisions/005-the-relationship-between-data-and-decision-making>`
   * :doc:`006 · Quantitative and Qualitative Data in Decision-Making — what/how many/how often meets why: two data kinds, one decision <2_data_driven_decisions/006-quantitative-and-qualitative-data-in-decision-making>`
   * :doc:`007 · Data Creates Value Only When It Is Communicated — an insight nobody hears changes nothing: communication as the last mile <2_data_driven_decisions/007-data-creates-value-only-when-it-is-communicated>`
   * :doc:`008 · The Difference Between Data and Metrics, and the Role of Metrics — from raw facts to quantified goals: what makes a number a metric <2_data_driven_decisions/008-the-difference-between-data-and-metrics-and-the-role-of-metrics>`
   * :doc:`009 · Dashboards — live metrics in one place: what dashboards are for, and when a report beats one <2_data_driven_decisions/009-dashboards>`
   * :doc:`010 · Mathematical Thinking — step-by-step decomposition, orders of magnitude, and choosing data sized to the decision <2_data_driven_decisions/010-mathematical-thinking>`
   * :doc:`011 · Spreadsheets in Data Analysis — where the spreadsheet sits in the decision workflow, phase by phase <2_data_driven_decisions/011-spreadsheets-in-data-analysis>`
   * :doc:`012 · Building and Organizing a Spreadsheet — one row per record, clean headers, raw kept raw: layouts that survive analysis <2_data_driven_decisions/012-building-and-organizing-a-spreadsheet>`
   * :doc:`013 · How Data Analysts Use Spreadsheets — the everyday spreadsheet toolkit — sort, filter, pivot, chart — mapped to real tasks <2_data_driven_decisions/013-how-data-analysts-use-spreadsheets>`
   * :doc:`014 · Spreadsheet Calculations with Formulas — formulas, cell references, and the absolute-vs-relative distinction that makes fill-down work <2_data_driven_decisions/014-spreadsheet-calculations-with-formulas>`
   * :doc:`015 · Common Spreadsheet Errors and How to Fix Them — reading #DIV/0!, #VALUE!, #REF!, #NAME?, #N/A — and fixing causes, not symptoms <2_data_driven_decisions/015-common-spreadsheet-errors-and-how-to-fix-them>`
   * :doc:`016 · Spreadsheet Functions — named operations — SUM, AVERAGE, IF, VLOOKUP — the analyst's core vocabulary <2_data_driven_decisions/016-spreadsheet-functions>`
   * :doc:`017 · Defining the Problem Domain — scoping before analysing: problem, stakeholders, deliverables, timeline, success <2_data_driven_decisions/017-defining-the-problem-domain>`
   * :doc:`018 · Context and Bias in Data Analysis — context makes data meaningful; unexamined context is where bias enters <2_data_driven_decisions/018-context-and-bias-in-data-analysis>`
   * :doc:`019 · Stakeholder Expectations in Data Analysis — who has a stake, what they need, and how to align before the work begins <2_data_driven_decisions/019-stakeholder-expectations-in-data-analysis>`
   * :doc:`020 · Staying Focused on the Project Objective — guarding scope: keeping every step tied to the one question that matters <2_data_driven_decisions/020-staying-focused-on-the-project-objective>`
   * :doc:`021 · Clear Communication with Stakeholders and Teams — the five Cs of clarity: making findings land with the people who act on them <2_data_driven_decisions/021-clear-communication-with-stakeholders-and-teams>`
   * :doc:`022 · Adapting to Communication Expectations at Work — reading the room: matching channel, register, and detail to each audience <2_data_driven_decisions/022-adapting-to-communication-expectations-at-work>`
   * :doc:`023 · Managing Stakeholder Expectations and Project Constraints — the iron triangle of scope, time, and resources — and honest trade-offs <2_data_driven_decisions/023-managing-stakeholder-expectations-and-project-constraints>`
   * :doc:`024 · Balancing Speed and Accuracy in Data Analysis — fast-enough vs right-enough: matching rigour to the decision's stakes <2_data_driven_decisions/024-balancing-speed-and-accuracy-in-data-analysis>`
   * :doc:`025 · Sharing Data to Drive Impact — the difference between reporting numbers and actually changing a decision <2_data_driven_decisions/025-sharing-data-to-drive-impact>`
   * :doc:`026 · Effective Meetings — purpose, preparation, and follow-up: making the time analysts spend together count <2_data_driven_decisions/026-effective-meetings>`
   * :doc:`027 · Conflict Resolution in the Workplace — when the data disagrees with a person: disagreeing productively and separating issue from ego <2_data_driven_decisions/027-conflict-resolution-in-the-workplace>`

.. dropdown:: 📦 3. Data Preparation
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Sourcing, structuring, and organising data before analysis: types, formats, databases, and sampling.  — **25/25 lessons.** :doc:`Open the section <3_data_preparation/index>`

   * :doc:`001 · How Data Is Generated and Collected — where data comes from: first/second/third-party sources and how it is produced <3_data_preparation/001-how-data-is-generated-and-collected>`
   * :doc:`002 · Choosing the Right Data to Collect — relevance, coverage, and cost: deciding what data a question actually needs <3_data_preparation/002-choosing-the-right-data-to-collect>`
   * :doc:`003 · Understanding Data Types and Data Formats — nominal, ordinal, discrete, continuous — and the file formats data travels in <3_data_preparation/003-understanding-data-types-and-data-formats>`
   * :doc:`004 · Structured Data and Data Models — structured vs semi- vs unstructured, and the models that give data its shape <3_data_preparation/004-structured-data-and-data-models>`
   * :doc:`005 · Data Types in Spreadsheets — text, number, date, boolean — why a spreadsheet cares what type a cell holds <3_data_preparation/005-data-types-in-spreadsheets>`
   * :doc:`006 · Data Tables (Tabular Data) — the row-and-column table: the workhorse shape of analysable data <3_data_preparation/006-data-tables-tabular-data>`
   * :doc:`007 · Wide Data vs. Long Data — two tidy layouts of the same data, and when each one serves <3_data_preparation/007-wide-data-vs-long-data>`
   * :doc:`008 · Understanding Bias in Data Analysis — what bias is, where it hides, and why unbiased data is the goal not the default <3_data_preparation/008-understanding-bias-in-data-analysis>`
   * :doc:`009 · Sampling Bias and Unbiased Data — when the sample misrepresents the population — and how random sampling guards against it <3_data_preparation/009-sampling-bias-and-unbiased-data>`
   * :doc:`010 · Common Types of Data Bias — sampling, observer, interpretation, and confirmation bias, with how each distorts <3_data_preparation/010-common-types-of-data-bias>`
   * :doc:`011 · Identifying Good Data Sources (ROCCC Framework) — Reliable, Original, Comprehensive, Current, Cited — the marks of trustworthy data <3_data_preparation/011-identifying-good-data-sources-roccc-framework>`
   * :doc:`012 · Identifying Bad Data Sources (When Data Does Not ROCCC) — reading ROCCC in reverse to spot the data you should not trust <3_data_preparation/012-identifying-bad-data-sources-when-data-does-not-roccc>`
   * :doc:`013 · Data Ethics in Data Analysis — ownership, transaction transparency, consent, and currency — the ethics of using data <3_data_preparation/013-data-ethics-in-data-analysis>`
   * :doc:`014 · Data Privacy in Data Ethics — protecting people's information: what privacy requires of an analyst <3_data_preparation/014-data-privacy-in-data-ethics>`
   * :doc:`015 · Open Data and Openness in Data Ethics — when data should be freely available — and the tension with privacy <3_data_preparation/015-open-data-and-openness-in-data-ethics>`
   * :doc:`016 · Databases and Relational Database Concepts — tables, keys, and relationships: how organisational data is really stored <3_data_preparation/016-databases-and-relational-database-concepts>`
   * :doc:`017 · Metadata in Databases — data about data: descriptive, structural, and administrative context that makes data usable <3_data_preparation/017-metadata-in-databases>`
   * :doc:`018 · Metadata Repositories and Data Governance — where metadata is catalogued, and the governance that keeps data trustworthy <3_data_preparation/018-metadata-repositories-and-data-governance>`
   * :doc:`019 · Accessing Data: Internal and External Sources — getting to the data: what lives inside the organisation versus outside <3_data_preparation/019-accessing-data-internal-and-external-sources>`
   * :doc:`020 · Importing Data into Spreadsheets — getting external data into a sheet cleanly — and the type traps to watch <3_data_preparation/020-importing-data-into-spreadsheets>`
   * :doc:`021 · Sorting and Filtering Data in Spreadsheets — ordering and narrowing rows: the two most-used moves for making data legible <3_data_preparation/021-sorting-and-filtering-data-in-spreadsheets>`
   * :doc:`022 · BigQuery Account Types — sandbox, free tier, and paid — how to access a cloud data warehouse for practice <3_data_preparation/022-bigquery-account-types>`
   * :doc:`023 · Querying Data with SQL — SELECT, FROM, WHERE: retrieving exactly the rows and columns a question needs <3_data_preparation/023-querying-data-with-sql>`
   * :doc:`024 · Organizing Data for Personal and Work Projects — folder, file, and naming conventions that keep a data project findable and safe <3_data_preparation/024-organizing-data-for-personal-and-work-projects>`
   * :doc:`025 · Data Security in Spreadsheets — protecting a shared sheet: access control, protected ranges, and safe sharing <3_data_preparation/025-data-security-in-spreadsheets>`

.. dropdown:: 🧽 4. Data Cleaning & Preparation
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Finding and fixing dirty data: missing values, duplicates, outliers, and validation for trustworthy inputs.  — **32/32 lessons.** :doc:`Open the section <4_data_cleaning_preparation/index>`

   * :doc:`001 · The Importance of Clean Data — why clean data is the non-negotiable foundation every analysis stands on <4_data_cleaning_preparation/001-the-importance-of-clean-data>`
   * :doc:`002 · Data Integrity and Its Risks in Data Analysis — keeping data accurate and consistent through its life — and what threatens it <4_data_cleaning_preparation/002-data-integrity-and-its-risks-in-data-analysis>`
   * :doc:`003 · Aligning Data with Business Objectives — checking that the data you have actually fits the question you must answer <4_data_cleaning_preparation/003-aligning-data-with-business-objectives>`
   * :doc:`004 · Handling Insufficient Data in Data Analysis — recognising when there is not enough data, and the honest options when there isn't <4_data_cleaning_preparation/004-handling-insufficient-data-in-data-analysis>`
   * :doc:`005 · Population, Sample Size, and Random Sampling — the whole vs. the part: population, sample, and how to sample fairly <4_data_cleaning_preparation/005-population-sample-size-and-random-sampling>`
   * :doc:`006 · Statistical Power in Data Analysis — the chance of detecting a real effect when there is one — and why 0.8 is the norm <4_data_cleaning_preparation/006-statistical-power-in-data-analysis>`
   * :doc:`007 · Sample Size and Data Integrity — how much data is enough: confidence, precision, and the cost of too little <4_data_cleaning_preparation/007-sample-size-and-data-integrity>`
   * :doc:`008 · Margin of Error — the plus-or-minus around a sample estimate, and how to read it honestly <4_data_cleaning_preparation/008-margin-of-error>`
   * :doc:`009 · Dirty Data vs. Clean Data — the concrete contrast: what makes data dirty, and what clean looks like beside it <4_data_cleaning_preparation/009-dirty-data-vs-clean-data>`
   * :doc:`010 · The Importance of Clean Data (revisited) — the business cost of dirty data — why cleaning is worth the time it takes <4_data_cleaning_preparation/010-the-importance-of-clean-data-revisited>`
   * :doc:`011 · Common Issues in Dirty Data — the recurring defects — duplicates, missing, inconsistent, wrong type, outliers <4_data_cleaning_preparation/011-common-issues-in-dirty-data>`
   * :doc:`012 · Data Cleaning with Spreadsheets — the practical spreadsheet cleaning workflow, defect by defect <4_data_cleaning_preparation/012-data-cleaning-with-spreadsheets>`
   * :doc:`013 · Cleaning and Merging Multiple Datasets — combining data from several sources cleanly — matching keys, reconciling formats <4_data_cleaning_preparation/013-cleaning-and-merging-multiple-datasets>`
   * :doc:`014 · Spreadsheet Tools for Data Cleaning — the built-in cleaning toolkit: dedupe, split, find/replace, validation, formatting <4_data_cleaning_preparation/014-spreadsheet-tools-for-data-cleaning>`
   * :doc:`015 · Using Spreadsheet Functions for Data Cleaning — TRIM, CLEAN, UPPER, SUBSTITUTE, VALUE and friends — cleaning by formula <4_data_cleaning_preparation/015-using-spreadsheet-functions-for-data-cleaning>`
   * :doc:`016 · Viewing Data Differently for More Effective Data Cleaning — sort, filter, pivot, conditional formatting as lenses that reveal hidden defects <4_data_cleaning_preparation/016-viewing-data-differently-for-more-effective-data-cleaning>`
   * :doc:`017 · Data Mapping and the Big Picture of Clean Data — matching fields between sources so merged data stays coherent — cleaning's big picture <4_data_cleaning_preparation/017-data-mapping-and-the-big-picture-of-clean-data>`
   * :doc:`018 · Introduction to SQL — why SQL matters for cleaning: transformations that scale and stay reproducible <4_data_cleaning_preparation/018-introduction-to-sql>`
   * :doc:`019 · Spreadsheets vs. SQL — when to reach for a sheet and when for a query — strengths, limits, and the handoff <4_data_cleaning_preparation/019-spreadsheets-vs-sql>`
   * :doc:`020 · Core SQL Queries for Data Cleaning and Analysis — SELECT, WHERE, DISTINCT, GROUP BY — the query patterns that clean and summarise <4_data_cleaning_preparation/020-core-sql-queries-for-data-cleaning-and-analysis>`
   * :doc:`021 · Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables — DISTINCT, GROUP BY, and string functions to dedupe and standardise text in SQL <4_data_cleaning_preparation/021-cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables>`
   * :doc:`022 · Using CAST to Clean and Format Data in SQL — converting a value from one data type to another — SQL's type-fixing tool <4_data_cleaning_preparation/022-using-cast-to-clean-and-format-data-in-sql>`
   * :doc:`023 · Advanced SQL Functions for Data Cleaning — CASE, SUBSTR, TRIM, and pattern tools for the harder cleaning jobs <4_data_cleaning_preparation/023-advanced-sql-functions-for-data-cleaning>`
   * :doc:`024 · COALESCE — returning the first non-null value — SQL's standard handler for missing data <4_data_cleaning_preparation/024-coalesce>`
   * :doc:`025 · Verifying and Reporting Data Integrity — confirming data is sound after cleaning — and telling stakeholders it is <4_data_cleaning_preparation/025-verifying-and-reporting-data-integrity>`
   * :doc:`026 · Verifying Data-Cleaning Efforts — checking that each cleaning step did what it should, and nothing it shouldn't <4_data_cleaning_preparation/026-verifying-data-cleaning-efforts>`
   * :doc:`027 · Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors — concrete checks — counts, distinct values, ranges — that catch recurring defects <4_data_cleaning_preparation/027-verification-techniques-using-spreadsheets-and-sql-to-catch-repeated-errors>`
   * :doc:`028 · Documenting Data-Cleaning Changes — recording what was changed and why, so cleaning is reproducible and reviewable <4_data_cleaning_preparation/028-documenting-data-cleaning-changes>`
   * :doc:`029 · Reporting Data-Cleaning Results — communicating what cleaning found and fixed, so stakeholders can trust the data <4_data_cleaning_preparation/029-reporting-data-cleaning-results>`
   * :doc:`030 · Using Feedback from Data Cleaning to Improve Data Quality — closing the loop: feeding cleaning lessons back to fix problems at the source <4_data_cleaning_preparation/030-using-feedback-from-data-cleaning-to-improve-data-quality>`
   * :doc:`031 · Refining a Resume for Data Analytics Roles — shaping a resume to show analytical skill — cleaning your own professional data <4_data_cleaning_preparation/031-refining-a-resume-for-data-analytics-roles>`
   * :doc:`032 · Exploring Data Analyst Job Opportunities — where analyst roles are, and how to read them against your skills and goals <4_data_cleaning_preparation/032-exploring-data-analyst-job-opportunities>`

.. dropdown:: 📊 5. Analyze Data
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Organising, formatting, aggregating, and computing on data to surface patterns and answer the question.  — **30/30 lessons.** :doc:`Open the section <5_analyze_data/index>`

   * :doc:`001 · Understanding Data Analysis — what analysis actually is: the four phases that turn prepared data into insight <5_analyze_data/001-understanding-data-analysis>`
   * :doc:`002 · Data Organization in Analysis — arranging data so analysis is possible — the organising step before any computation <5_analyze_data/002-data-organization-in-analysis>`
   * :doc:`003 · Sorting and Filtering in Data Analysis — the two foundational moves of analysis, and how they differ from cleaning uses <5_analyze_data/003-sorting-and-filtering-in-data-analysis>`
   * :doc:`004 · Sorting Data in Spreadsheets — ordering rows by one or more columns to surface structure — done safely <5_analyze_data/004-sorting-data-in-spreadsheets>`
   * :doc:`005 · Sorting and Filtering Data in SQL Using ORDER BY and WHERE — the SQL twins of sort and filter: ORDER BY to order, WHERE to subset <5_analyze_data/005-sorting-and-filtering-data-in-sql-using-order-by-and-where>`
   * :doc:`006 · Data Formatting and Unit Conversion in Spreadsheets — consistent formats and units — making numbers comparable before analysing them <5_analyze_data/006-data-formatting-and-unit-conversion-in-spreadsheets>`
   * :doc:`007 · Data Validation in Spreadsheets — rules that constrain what a cell may hold, catching and preventing bad data <5_analyze_data/007-data-validation-in-spreadsheets>`
   * :doc:`008 · Combining Data Validation and Conditional Formatting in Spreadsheets — validation to enforce rules, formatting to reveal violations — used together <5_analyze_data/008-combining-data-validation-and-conditional-formatting-in-spreadsheets>`
   * :doc:`009 · Using CONCAT in SQL to Combine Text from Multiple Columns — joining text from several columns into one — CONCAT and the concatenation operator <5_analyze_data/009-using-concat-in-sql-to-combine-text-from-multiple-columns>`
   * :doc:`010 · Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND) — the spreadsheet string toolkit for measuring, extracting, and locating text <5_analyze_data/010-working-with-strings-in-spreadsheets-len-left-right-find>`
   * :doc:`011 · Problem-Solving and Seeking Help in Data Analysis — the analyst's debugging mindset, and how to ask for help effectively <5_analyze_data/011-problem-solving-and-seeking-help-in-data-analysis>`
   * :doc:`012 · How to Effectively Search for Solutions Online as a Data Analyst — turning an error or a stuck point into a search that actually finds the answer <5_analyze_data/012-how-to-effectively-search-for-solutions-online-as-a-data-analyst>`
   * :doc:`013 · Choosing the Right Tool in Data Analysis — matching the task to spreadsheet, SQL, or programming — and combining them <5_analyze_data/013-choosing-the-right-tool-in-data-analysis>`
   * :doc:`014 · Preparing Data for VLOOKUP in Spreadsheets — the setup VLOOKUP demands: a clean lookup key in the leftmost column <5_analyze_data/014-preparing-data-for-vlookup-in-spreadsheets>`
   * :doc:`015 · Using VLOOKUP to Combine Data Across Spreadsheets — pulling matching values from another table by a shared key <5_analyze_data/015-using-vlookup-to-combine-data-across-spreadsheets>`
   * :doc:`016 · Troubleshooting VLOOKUP and Building a Problem-Solving Framework — why VLOOKUP fails, how to fix it, and a reusable debugging framework <5_analyze_data/016-troubleshooting-vlookup-and-building-a-problem-solving-framework>`
   * :doc:`017 · Using JOIN in SQL to Combine Tables — combining rows from multiple tables on a matching key — SQL's core combine <5_analyze_data/017-using-join-in-sql-to-combine-tables>`
   * :doc:`018 · Subqueries in SQL — a query nested inside another — using one query's result within a second <5_analyze_data/018-subqueries-in-sql>`
   * :doc:`019 · Aggregating Data with Subqueries, HAVING, and CASE in SQL — combining aggregation, group filtering, and conditional logic for real analysis <5_analyze_data/019-aggregating-data-with-subqueries-having-and-case-in-sql>`
   * :doc:`020 · Using Spreadsheet Formulas for Sales Trend Analysis — formulas for change over time — growth rates, running totals, period comparisons <5_analyze_data/020-using-spreadsheet-formulas-for-sales-trend-analysis>`
   * :doc:`021 · Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets — counting and summing only the rows that meet a condition <5_analyze_data/021-using-countif-and-sumif-for-conditional-aggregation-in-spreadsheets>`
   * :doc:`022 · Using SUMPRODUCT for Advanced Spreadsheet Calculations — multiplying and summing across arrays in one formula — weighted and multi-condition sums <5_analyze_data/022-using-sumproduct-for-advanced-spreadsheet-calculations>`
   * :doc:`023 · Using Pivot Tables for Calculations and Trend Analysis — the spreadsheet's most powerful summarising tool — grouping and aggregating with drags <5_analyze_data/023-using-pivot-tables-for-calculations-and-trend-analysis>`
   * :doc:`024 · Using Pivot Table Filters and Calculated Fields for Deeper Analysis — filtering a pivot and adding computed fields — pushing pivots past basic summaries <5_analyze_data/024-using-pivot-table-filters-and-calculated-fields-for-deeper-analysis>`
   * :doc:`025 · Comparing Calculations in Spreadsheets and SQL — the same aggregate two ways — when each tool suits the calculation <5_analyze_data/025-comparing-calculations-in-spreadsheets-and-sql>`
   * :doc:`026 · Embedding Calculations in SQL Queries — computing derived values inside a query — arithmetic and expressions in SELECT <5_analyze_data/026-embedding-calculations-in-sql-queries>`
   * :doc:`027 · Using GROUP BY and ORDER BY for Aggregated Calculations in SQL — the SQL pivot: grouping to aggregate, ordering the summary <5_analyze_data/027-using-group-by-and-order-by-for-aggregated-calculations-in-sql>`
   * :doc:`028 · Data Validation as an Ongoing Analytical Process — validation not as a one-time gate but a continuous check throughout analysis <5_analyze_data/028-data-validation-as-an-ongoing-analytical-process>`
   * :doc:`029 · Temporary Tables and the WITH Clause in SQL — holding intermediate results — CTEs and temp tables that structure complex queries <5_analyze_data/029-temporary-tables-and-the-with-clause-in-sql>`
   * :doc:`030 · Creating Temporary Tables in SQL — Methods, Trade-offs, and Best Practices — the ways to make a temp table, when each fits, and how to use them well <5_analyze_data/030-creating-temporary-tables-in-sql-methods-trade-offs-and-best-practices>`

.. dropdown:: 🎨 6. Data Visualization
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Turning results into visuals that inform: chart choice, design principles, and honest, accessible graphics.  — **27/27 lessons.** :doc:`Open the section <6_data_visualization/index>`

   * :doc:`001 · Data Visualization — representing data graphically so patterns and meaning become visible at a glance <6_data_visualization/001-data-visualization>`
   * :doc:`002 · Connecting Data and Images — how a visualization maps data values to visual properties — the encoding that carries meaning <6_data_visualization/002-connecting-data-and-images>`
   * :doc:`003 · Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose — what makes a visualization effective — a clear focus, sound structure, and a purpose <6_data_visualization/003-creating-powerful-data-visualizations-focus-structure-and-analytical-purpose>`
   * :doc:`004 · Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity — fixed images versus interactive views — when each serves, and their trade-offs <6_data_visualization/004-static-vs-dynamic-data-visualizations-design-tradeoffs-control-and-interactivity>`
   * :doc:`005 · Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement — the visual building blocks — line, shape, colour, space, movement — applied to charts <6_data_visualization/005-elements-of-art-in-data-visualization-line-shape-color-space-and-movement>`
   * :doc:`006 · Choosing the Right Visualization: Audience-Centered Design and Chart Selection — matching chart type to the data, the message, and the audience <6_data_visualization/006-choosing-the-right-visualization-audience-centered-design-and-chart-selection>`
   * :doc:`007 · Design Thinking in Data Visualization: A User-Centered Framework — designing visualizations around the viewer's needs — a user-centered process <6_data_visualization/007-design-thinking-in-data-visualization-a-user-centered-framework>`
   * :doc:`008 · Accessibility in Data Visualization: Designing for Everyone — making charts readable by everyone, including people with visual differences <6_data_visualization/008-accessibility-in-data-visualization-designing-for-everyone>`
   * :doc:`009 · Introduction to Tableau — what Tableau is and why analysts use it — visual analytics without heavy coding <6_data_visualization/009-introduction-to-tableau>`
   * :doc:`010 · Getting Started with Tableau Public — the free version: connecting data, building a first view, publishing to the web <6_data_visualization/010-getting-started-with-tableau-public>`
   * :doc:`011 · Creating a CO₂ Emissions Visualization in Tableau Public — a worked Tableau example — turning an emissions dataset into a clear visualization <6_data_visualization/011-creating-a-co-emissions-visualization-in-tableau-public>`
   * :doc:`012 · Effective vs. Ineffective Data Visualizations in Tableau — the difference in practice — what makes a Tableau chart clear versus confusing <6_data_visualization/012-effective-vs-ineffective-data-visualizations-in-tableau>`
   * :doc:`013 · Using Creativity in Tableau — going beyond default charts — custom, expressive visualizations that still communicate <6_data_visualization/013-using-creativity-in-tableau>`
   * :doc:`014 · Linking Multiple Datasets in Tableau Public — combining data sources inside Tableau — joins and relationships for richer views <6_data_visualization/014-linking-multiple-datasets-in-tableau-public>`
   * :doc:`015 · Data Storytelling: Giving Numbers a Clear and Convincing Voice — wrapping data in narrative so it informs, persuades, and is remembered <6_data_visualization/015-data-storytelling-giving-numbers-a-clear-and-convincing-voice>`
   * :doc:`016 · Engaging Your Audience in Data Storytelling: Identifying the Key Message — finding the one message that matters and building the story around it <6_data_visualization/016-engaging-your-audience-in-data-storytelling-identifying-the-key-message>`
   * :doc:`017 · Data Dashboards: Organizing Insight for Real-Time Decision Making — combining key visualizations into one monitored view for ongoing decisions <6_data_visualization/017-data-dashboards-organizing-insight-for-real-time-decision-making>`
   * :doc:`018 · Using Filters to Create Compelling and Focused Visuals — filtering to sharpen a visualization's message and enable exploration <6_data_visualization/018-using-filters-to-create-compelling-and-focused-visuals>`
   * :doc:`019 · Structuring a Persuasive Data Presentation: Turning Insights into Story — arranging findings into a presentation that carries an audience to a conclusion <6_data_visualization/019-structuring-a-persuasive-data-presentation-turning-insights-into-story>`
   * :doc:`020 · Designing Effective Data Presentation Slides: Structure, Visuals, and Professional Impact — slide design that supports the message — clear, visual, uncluttered, professional <6_data_visualization/020-designing-effective-data-presentation-slides-structure-visuals-and-professional-impact>`
   * :doc:`021 · Using a Strategic Framework to Structure Data Presentations — reusable structures that reliably organize a presentation toward its goal <6_data_visualization/021-using-a-strategic-framework-to-structure-data-presentations>`
   * :doc:`022 · Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method — integrating data with narrative — hypothesis, context, and a four-element method <6_data_visualization/022-weaving-data-into-presentations-hypotheses-context-and-the-mccandless-method>`
   * :doc:`023 · Presentation Skills for Data Analysts: Delivering Insights with Confidence — the delivery half of presenting — voice, pace, presence, and handling nerves <6_data_visualization/023-presentation-skills-for-data-analysts-delivering-insights-with-confidence>`
   * :doc:`024 · Presenting Like a Pro: Best Practices for Data Analysts — the practices that mark a polished, credible data presenter <6_data_visualization/024-presenting-like-a-pro-best-practices-for-data-analysts>`
   * :doc:`025 · Preparing for Q&A: Anticipating and Responding to Stakeholder Questions — readying for the questions a presentation will draw — anticipation as preparation <6_data_visualization/025-preparing-for-q-and-a-anticipating-and-responding-to-stakeholder-questions>`
   * :doc:`026 · Handling Objections in Data Presentations: Responding with Confidence and Clarity — responding to challenges and pushback without defensiveness, honestly and calmly <6_data_visualization/026-handling-objections-in-data-presentations-responding-with-confidence-and-clarity>`
   * :doc:`027 · Q&A Best Practices: Answering Questions with Clarity and Confidence — the habits of fielding questions well — listen, answer directly, stay honest <6_data_visualization/027-q-and-a-best-practices-answering-questions-with-clarity-and-confidence>`

.. dropdown:: 🐍 7. Data Analysis Using Python
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Doing the whole workflow in Python: NumPy, pandas, and plotting for real analytical tasks.  — **33/33 lessons.** :doc:`Open the section <7_data_analysis_python/index>`

   * :doc:`001 · Introduction to Python and Programming Fundamentals — why analysts learn Python, and the programming ideas underneath it <7_data_analysis_python/001-introduction-to-python-and-programming-fundamentals>`
   * :doc:`002 · Python Fundamentals — the core building blocks of Python — values, expressions, statements, and flow <7_data_analysis_python/002-python-fundamentals>`
   * :doc:`003 · Jupyter Notebook and Coding Environments — where analysts write and run Python — notebooks, IDEs, and the interactive workflow <7_data_analysis_python/003-jupyter-notebook-and-coding-environments>`
   * :doc:`004 · Object-Oriented Programming (OOP) in Python — the objects-and-methods model that shapes Python, including pandas and numpy <7_data_analysis_python/004-object-oriented-programming-oop-in-python>`
   * :doc:`005 · Variables in Python — named containers for values — assigning, reassigning, and using them <7_data_analysis_python/005-variables-in-python>`
   * :doc:`006 · Naming Conventions and Restrictions in Python — the rules for valid names and the conventions for good ones (PEP 8) <7_data_analysis_python/006-naming-conventions-and-restrictions-in-python>`
   * :doc:`007 · Data Types and Type Conversion in Python — the kinds of values Python holds, and converting between them safely <7_data_analysis_python/007-data-types-and-type-conversion-in-python>`
   * :doc:`008 · Functions in Python — named, reusable blocks of code — defining, calling, arguments, and returns <7_data_analysis_python/008-functions-in-python>`
   * :doc:`009 · Code Reusability, Modularity, and Clean Code in Python — writing Python that is reusable, modular, and clean — the engineering virtues in code <7_data_analysis_python/009-code-reusability-modularity-and-clean-code-in-python>`
   * :doc:`010 · Comments, Algorithms, and Docstrings in Python — documenting code — comments, docstrings, and thinking in algorithms <7_data_analysis_python/010-comments-algorithms-and-docstrings-in-python>`
   * :doc:`011 · Boolean Data, Comparators, and Logical Operators in Python — True/False values and the comparisons and logic that produce them <7_data_analysis_python/011-boolean-data-comparators-and-logical-operators-in-python>`
   * :doc:`012 · Branching and Conditional Statements in Python — making code decide — if, elif, and else direct which code runs <7_data_analysis_python/012-branching-and-conditional-statements-in-python>`
   * :doc:`013 · While Loops and Iteration in Python — repeating code while a condition holds — the while loop and iteration basics <7_data_analysis_python/013-while-loops-and-iteration-in-python>`
   * :doc:`014 · For Loops in Python — repeating code over the items of a collection — the for loop <7_data_analysis_python/014-for-loops-in-python>`
   * :doc:`015 · range() Function and Loop Control in Python — generating number sequences and controlling loops with break and continue <7_data_analysis_python/015-range-function-and-loop-control-in-python>`
   * :doc:`016 · Strings in Python — working with text — the string type, its operations, and its immutability <7_data_analysis_python/016-strings-in-python>`
   * :doc:`017 · String Indexing and Slicing in Python — reaching into strings by position — single characters and substrings <7_data_analysis_python/017-string-indexing-and-slicing-in-python>`
   * :doc:`018 · String Formatting with .format() in Python — building strings from values cleanly — the .format() method and f-strings <7_data_analysis_python/018-string-formatting-with-format-in-python>`
   * :doc:`019 · Data Types vs Data Structures & Introduction to Lists — the difference between a value's type and a structure that holds many, and the list <7_data_analysis_python/019-data-types-vs-data-structures-and-introduction-to-lists>`
   * :doc:`020 · Modifying Lists in Python — changing lists in place — adding, removing, and updating elements <7_data_analysis_python/020-modifying-lists-in-python>`
   * :doc:`021 · Tuples in Python — ordered, immutable collections — like lists that cannot change <7_data_analysis_python/021-tuples-in-python>`
   * :doc:`022 · Advanced Use of Loops, Lists, Tuples & List Comprehension — combining loops and structures, and the concise list comprehension idiom <7_data_analysis_python/022-advanced-use-of-loops-lists-tuples-and-list-comprehension>`
   * :doc:`023 · Dictionaries in Python — key-value collections — looking values up by a meaningful key <7_data_analysis_python/023-dictionaries-in-python>`
   * :doc:`024 · Advanced Dictionary Usage in Python — iterating, nesting, and safely accessing dictionaries for real data <7_data_analysis_python/024-advanced-dictionary-usage-in-python>`
   * :doc:`025 · Sets in Python — unordered collections of unique values — membership and set operations <7_data_analysis_python/025-sets-in-python>`
   * :doc:`026 · Libraries, Packages, and Modules in Python — reusing others' code — importing modules, packages, and the library ecosystem <7_data_analysis_python/026-libraries-packages-and-modules-in-python>`
   * :doc:`027 · Introduction to NumPy and Vectorization — the numerical library and vectorization — fast array operations without loops <7_data_analysis_python/027-introduction-to-numpy-and-vectorization>`
   * :doc:`028 · NumPy Arrays (ndarray) and Core Concepts — the ndarray, NumPy's core structure, and how it differs from a list <7_data_analysis_python/028-numpy-arrays-ndarray-and-core-concepts>`
   * :doc:`029 · Introduction to Pandas (Data Analysis Library) — the DataFrame library that brings spreadsheet/SQL-style tables to Python <7_data_analysis_python/029-introduction-to-pandas-data-analysis-library>`
   * :doc:`030 · Pandas DataFrame & Series — the two core pandas structures — the table (DataFrame) and the column (Series) <7_data_analysis_python/030-pandas-dataframe-and-series>`
   * :doc:`031 · Boolean Masking in Pandas — filtering rows by a condition — the pandas equivalent of WHERE and spreadsheet filters <7_data_analysis_python/031-boolean-masking-in-pandas>`
   * :doc:`032 · Grouping and Aggregation in Pandas (groupby, agg) — grouping rows and computing per-group aggregates — pandas' GROUP BY <7_data_analysis_python/032-grouping-and-aggregation-in-pandas-groupby-agg>`
   * :doc:`033 · Combining Data in Pandas (concat and merge) — bringing DataFrames together — stacking with concat, joining on keys with merge <7_data_analysis_python/033-combining-data-in-pandas-concat-and-merge>`

.. dropdown:: 💼 8. Job Search
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   From portfolio to offer: resumes, the analyst interview, case studies, and landing the role.  — **15/15 lessons.** :doc:`Open the section <8_job_search/index>`

   * :doc:`001 · Transferable Skills — abilities that carry across roles and industries — the foundation of a career pivot <8_job_search/001-transferable-skills>`
   * :doc:`002 · Career Identity Statement — a concise statement of the unique value you bring to the workforce <8_job_search/002-career-identity-statement>`
   * :doc:`003 · Career Dreamer (AI Tool for Career Exploration) — an experimental Grow with Google AI tool that maps your experience to career paths <8_job_search/003-career-dreamer-ai-tool-for-career-exploration>`
   * :doc:`004 · Job Search Plan (Using AI Tools) — a structured, deliberate plan for the job search, assisted by AI tools <8_job_search/004-job-search-plan-using-ai-tools>`
   * :doc:`005 · Tailoring Your Resume — adapting a resume to each specific role rather than sending one generic version <8_job_search/005-tailoring-your-resume>`
   * :doc:`006 · Using AI to Improve and Tailor Your Resume — using AI assistants to draft, refine, and tailor a resume — with human judgement <8_job_search/006-using-ai-to-improve-and-tailor-your-resume>`
   * :doc:`007 · Building a Professional Online Presence (Personal Brand) — shaping how you appear online — profile, presence, and personal brand <8_job_search/007-building-a-professional-online-presence-personal-brand>`
   * :doc:`008 · Choosing the Right Job Platforms — selecting the job platforms and channels that fit your target roles <8_job_search/008-choosing-the-right-job-platforms>`
   * :doc:`009 · Job Application Tracking (Using AI + Spreadsheets) — systematically recording applications and their status, with AI and spreadsheets <8_job_search/009-job-application-tracking-using-ai-spreadsheets>`
   * :doc:`010 · Networking for Job Search — building and using professional relationships to find and reach opportunities <8_job_search/010-networking-for-job-search>`
   * :doc:`011 · Interview Preparation — readying for interviews — research, practice, and knowing your material <8_job_search/011-interview-preparation>`
   * :doc:`012 · STAR Method (Behavioral Interview) — a structure for behavioural answers — Situation, Task, Action, Result <8_job_search/012-star-method-behavioral-interview>`
   * :doc:`013 · Using AI (NotebookLM) for Interview Preparation — using a source-grounded AI research tool to prepare from role and company material <8_job_search/013-using-ai-notebooklm-for-interview-preparation>`
   * :doc:`014 · Practicing Interviews with AI (Gemini Live) — rehearsing interviews aloud in real-time voice conversation with an AI <8_job_search/014-practicing-interviews-with-ai-gemini-live>`
   * :doc:`015 · Post-Interview Strategy — what to do after an interview — follow up, reflect, and handle the outcome <8_job_search/015-post-interview-strategy>`

🔤 Every lesson, A–Z
---------------------

.. dropdown:: 🔠 A–Z index
   :animate: fade-in-slide-down
   :class-container: term-az

   .. hlist::
      :columns: 2

   * :doc:`A Review of the Six Stages of the Data Life Cycle  (Foundations) <1_foundations/010-a-review-of-the-six-stages-of-the-data-life-cycle>`
   * :doc:`Accessibility in Data Visualization: Designing for Everyone  (Data Visualization) <6_data_visualization/008-accessibility-in-data-visualization-designing-for-everyone>`
   * :doc:`Accessing Data: Internal and External Sources  (Data Preparation) <3_data_preparation/019-accessing-data-internal-and-external-sources>`
   * :doc:`Adapting to Communication Expectations at Work  (Data-Driven Decisions) <2_data_driven_decisions/022-adapting-to-communication-expectations-at-work>`
   * :doc:`Advanced Dictionary Usage in Python  (Data Analysis Using Python) <7_data_analysis_python/024-advanced-dictionary-usage-in-python>`
   * :doc:`Advanced SQL Functions for Data Cleaning  (Data Cleaning & Preparation) <4_data_cleaning_preparation/023-advanced-sql-functions-for-data-cleaning>`
   * :doc:`Advanced Use of Loops, Lists, Tuples & List Comprehension  (Data Analysis Using Python) <7_data_analysis_python/022-advanced-use-of-loops-lists-tuples-and-list-comprehension>`
   * :doc:`Aggregating Data with Subqueries, HAVING, and CASE in SQL  (Analyze Data) <5_analyze_data/019-aggregating-data-with-subqueries-having-and-case-in-sql>`
   * :doc:`Aligning Data with Business Objectives  (Data Cleaning & Preparation) <4_data_cleaning_preparation/003-aligning-data-with-business-objectives>`
   * :doc:`Analytical Skills and Their Core Components  (Foundations) <1_foundations/013-analytical-skills-and-their-core-components>`
   * :doc:`Analytical Thinking and Its Core Components  (Foundations) <1_foundations/015-analytical-thinking-and-its-core-components>`
   * :doc:`Analytical Thinking and Questions for Problem Solving  (Foundations) <1_foundations/016-analytical-thinking-and-questions-for-problem-solving>`
   * :doc:`Applying Analytical Skills in a Business Context  (Foundations) <1_foundations/014-applying-analytical-skills-in-a-business-context>`
   * :doc:`Applying Data Analytics Problem Types in Real Business Scenarios  (Data-Driven Decisions) <2_data_driven_decisions/003-applying-data-analytics-problem-types-in-real-business-scenarios>`
   * :doc:`Balancing Speed and Accuracy in Data Analysis  (Data-Driven Decisions) <2_data_driven_decisions/024-balancing-speed-and-accuracy-in-data-analysis>`
   * :doc:`BigQuery Account Types  (Data Preparation) <3_data_preparation/022-bigquery-account-types>`
   * :doc:`Boolean Data, Comparators, and Logical Operators in Python  (Data Analysis Using Python) <7_data_analysis_python/011-boolean-data-comparators-and-logical-operators-in-python>`
   * :doc:`Boolean Masking in Pandas  (Data Analysis Using Python) <7_data_analysis_python/031-boolean-masking-in-pandas>`
   * :doc:`Branching and Conditional Statements in Python  (Data Analysis Using Python) <7_data_analysis_python/012-branching-and-conditional-statements-in-python>`
   * :doc:`Building a Professional Online Presence (Personal Brand)  (Job Search) <8_job_search/007-building-a-professional-online-presence-personal-brand>`
   * :doc:`Building and Organizing a Spreadsheet  (Data-Driven Decisions) <2_data_driven_decisions/012-building-and-organizing-a-spreadsheet>`
   * :doc:`Career Dreamer (AI Tool for Career Exploration)  (Job Search) <8_job_search/003-career-dreamer-ai-tool-for-career-exploration>`
   * :doc:`Career Identity Statement  (Job Search) <8_job_search/002-career-identity-statement>`
   * :doc:`Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making  (Foundations) <1_foundations/019-case-studies-in-data-analysis-and-the-practical-impact-of-data-driven-decision-making>`
   * :doc:`Choosing the Right Data to Collect  (Data Preparation) <3_data_preparation/002-choosing-the-right-data-to-collect>`
   * :doc:`Choosing the Right Job Platforms  (Job Search) <8_job_search/008-choosing-the-right-job-platforms>`
   * :doc:`Choosing the Right Tool in Data Analysis  (Analyze Data) <5_analyze_data/013-choosing-the-right-tool-in-data-analysis>`
   * :doc:`Choosing the Right Visualization: Audience-Centered Design and Chart Selection  (Data Visualization) <6_data_visualization/006-choosing-the-right-visualization-audience-centered-design-and-chart-selection>`
   * :doc:`Cleaning and Merging Multiple Datasets  (Data Cleaning & Preparation) <4_data_cleaning_preparation/013-cleaning-and-merging-multiple-datasets>`
   * :doc:`Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables  (Data Cleaning & Preparation) <4_data_cleaning_preparation/021-cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables>`
   * :doc:`Clear Communication with Stakeholders and Teams  (Data-Driven Decisions) <2_data_driven_decisions/021-clear-communication-with-stakeholders-and-teams>`
   * :doc:`COALESCE  (Data Cleaning & Preparation) <4_data_cleaning_preparation/024-coalesce>`
   * :doc:`Code Reusability, Modularity, and Clean Code in Python  (Data Analysis Using Python) <7_data_analysis_python/009-code-reusability-modularity-and-clean-code-in-python>`
   * :doc:`Combining Data in Pandas (concat and merge)  (Data Analysis Using Python) <7_data_analysis_python/033-combining-data-in-pandas-concat-and-merge>`
   * :doc:`Combining Data Validation and Conditional Formatting in Spreadsheets  (Analyze Data) <5_analyze_data/008-combining-data-validation-and-conditional-formatting-in-spreadsheets>`
   * :doc:`Comments, Algorithms, and Docstrings in Python  (Data Analysis Using Python) <7_data_analysis_python/010-comments-algorithms-and-docstrings-in-python>`
   * :doc:`Common Issues in Dirty Data  (Data Cleaning & Preparation) <4_data_cleaning_preparation/011-common-issues-in-dirty-data>`
   * :doc:`Common Spreadsheet Errors and How to Fix Them  (Data-Driven Decisions) <2_data_driven_decisions/015-common-spreadsheet-errors-and-how-to-fix-them>`
   * :doc:`Common Types of Data Bias  (Data Preparation) <3_data_preparation/010-common-types-of-data-bias>`
   * :doc:`Comparing Calculations in Spreadsheets and SQL  (Analyze Data) <5_analyze_data/025-comparing-calculations-in-spreadsheets-and-sql>`
   * :doc:`Conflict Resolution in the Workplace  (Data-Driven Decisions) <2_data_driven_decisions/027-conflict-resolution-in-the-workplace>`
   * :doc:`Connecting Data and Images  (Data Visualization) <6_data_visualization/002-connecting-data-and-images>`
   * :doc:`Context and Bias in Data Analysis  (Data-Driven Decisions) <2_data_driven_decisions/018-context-and-bias-in-data-analysis>`
   * :doc:`Core SQL Queries for Data Cleaning and Analysis  (Data Cleaning & Preparation) <4_data_cleaning_preparation/020-core-sql-queries-for-data-cleaning-and-analysis>`
   * :doc:`Creating a CO₂ Emissions Visualization in Tableau Public  (Data Visualization) <6_data_visualization/011-creating-a-co-emissions-visualization-in-tableau-public>`
   * :doc:`Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose  (Data Visualization) <6_data_visualization/003-creating-powerful-data-visualizations-focus-structure-and-analytical-purpose>`
   * :doc:`Creating Temporary Tables in SQL — Methods, Trade-offs, and Best Practices  (Analyze Data) <5_analyze_data/030-creating-temporary-tables-in-sql-methods-trade-offs-and-best-practices>`
   * :doc:`Dashboards  (Data-Driven Decisions) <2_data_driven_decisions/009-dashboards>`
   * :doc:`Data Cleaning with Spreadsheets  (Data Cleaning & Preparation) <4_data_cleaning_preparation/012-data-cleaning-with-spreadsheets>`
   * :doc:`Data Creates Value Only When It Is Communicated  (Data-Driven Decisions) <2_data_driven_decisions/007-data-creates-value-only-when-it-is-communicated>`
   * :doc:`Data Dashboards: Organizing Insight for Real-Time Decision Making  (Data Visualization) <6_data_visualization/017-data-dashboards-organizing-insight-for-real-time-decision-making>`
   * :doc:`Data Ethics in Data Analysis  (Data Preparation) <3_data_preparation/013-data-ethics-in-data-analysis>`
   * :doc:`Data Formatting and Unit Conversion in Spreadsheets  (Analyze Data) <5_analyze_data/006-data-formatting-and-unit-conversion-in-spreadsheets>`
   * :doc:`Data Integrity and Its Risks in Data Analysis  (Data Cleaning & Preparation) <4_data_cleaning_preparation/002-data-integrity-and-its-risks-in-data-analysis>`
   * :doc:`Data Mapping and the Big Picture of Clean Data  (Data Cleaning & Preparation) <4_data_cleaning_preparation/017-data-mapping-and-the-big-picture-of-clean-data>`
   * :doc:`Data Organization in Analysis  (Analyze Data) <5_analyze_data/002-data-organization-in-analysis>`
   * :doc:`Data Privacy in Data Ethics  (Data Preparation) <3_data_preparation/014-data-privacy-in-data-ethics>`
   * :doc:`Data Security in Spreadsheets  (Data Preparation) <3_data_preparation/025-data-security-in-spreadsheets>`
   * :doc:`Data Storytelling: Giving Numbers a Clear and Convincing Voice  (Data Visualization) <6_data_visualization/015-data-storytelling-giving-numbers-a-clear-and-convincing-voice>`
   * :doc:`Data Tables (Tabular Data)  (Data Preparation) <3_data_preparation/006-data-tables-tabular-data>`
   * :doc:`Data Types and Type Conversion in Python  (Data Analysis Using Python) <7_data_analysis_python/007-data-types-and-type-conversion-in-python>`
   * :doc:`Data Types in Spreadsheets  (Data Preparation) <3_data_preparation/005-data-types-in-spreadsheets>`
   * :doc:`Data Types vs Data Structures & Introduction to Lists  (Data Analysis Using Python) <7_data_analysis_python/019-data-types-vs-data-structures-and-introduction-to-lists>`
   * :doc:`Data Validation as an Ongoing Analytical Process  (Analyze Data) <5_analyze_data/028-data-validation-as-an-ongoing-analytical-process>`
   * :doc:`Data Validation in Spreadsheets  (Analyze Data) <5_analyze_data/007-data-validation-in-spreadsheets>`
   * :doc:`Data Visualization  (Data Visualization) <6_data_visualization/001-data-visualization>`
   * :doc:`Data-Driven Decision-Making  (Foundations) <1_foundations/003-data-driven-decision-making>`
   * :doc:`Data-Driven Decision-Making and the Role of Analytical Skills  (Foundations) <1_foundations/018-data-driven-decision-making-and-the-role-of-analytical-skills>`
   * :doc:`Databases and Relational Database Concepts  (Data Preparation) <3_data_preparation/016-databases-and-relational-database-concepts>`
   * :doc:`Defining the Problem Domain  (Data-Driven Decisions) <2_data_driven_decisions/017-defining-the-problem-domain>`
   * :doc:`Design Thinking in Data Visualization: A User-Centered Framework  (Data Visualization) <6_data_visualization/007-design-thinking-in-data-visualization-a-user-centered-framework>`
   * :doc:`Designing Effective Data Presentation Slides: Structure, Visuals, and Professional Impact  (Data Visualization) <6_data_visualization/020-designing-effective-data-presentation-slides-structure-visuals-and-professional-impact>`
   * :doc:`Detectives and Data Analysts  (Foundations) <1_foundations/004-detectives-and-data-analysts>`
   * :doc:`Dictionaries in Python  (Data Analysis Using Python) <7_data_analysis_python/023-dictionaries-in-python>`
   * :doc:`Dirty Data vs. Clean Data  (Data Cleaning & Preparation) <4_data_cleaning_preparation/009-dirty-data-vs-clean-data>`
   * :doc:`Documenting Data-Cleaning Changes  (Data Cleaning & Preparation) <4_data_cleaning_preparation/028-documenting-data-cleaning-changes>`
   * :doc:`Effective Meetings  (Data-Driven Decisions) <2_data_driven_decisions/026-effective-meetings>`
   * :doc:`Effective vs. Ineffective Data Visualizations in Tableau  (Data Visualization) <6_data_visualization/012-effective-vs-ineffective-data-visualizations-in-tableau>`
   * :doc:`Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement  (Data Visualization) <6_data_visualization/005-elements-of-art-in-data-visualization-line-shape-color-space-and-movement>`
   * :doc:`Embedding Calculations in SQL Queries  (Analyze Data) <5_analyze_data/026-embedding-calculations-in-sql-queries>`
   * :doc:`Engaging Your Audience in Data Storytelling: Identifying the Key Message  (Data Visualization) <6_data_visualization/016-engaging-your-audience-in-data-storytelling-identifying-the-key-message>`
   * :doc:`Exploring Data Analyst Job Opportunities  (Data Cleaning & Preparation) <4_data_cleaning_preparation/032-exploring-data-analyst-job-opportunities>`
   * :doc:`Fairness in Data Analysis  (Foundations) <1_foundations/026-fairness-in-data-analysis>`
   * :doc:`For Loops in Python  (Data Analysis Using Python) <7_data_analysis_python/014-for-loops-in-python>`
   * :doc:`Functions in Python  (Data Analysis Using Python) <7_data_analysis_python/008-functions-in-python>`
   * :doc:`Getting Started with Tableau Public  (Data Visualization) <6_data_visualization/010-getting-started-with-tableau-public>`
   * :doc:`Grouping and Aggregation in Pandas (groupby, agg)  (Data Analysis Using Python) <7_data_analysis_python/032-grouping-and-aggregation-in-pandas-groupby-agg>`
   * :doc:`Handling Insufficient Data in Data Analysis  (Data Cleaning & Preparation) <4_data_cleaning_preparation/004-handling-insufficient-data-in-data-analysis>`
   * :doc:`Handling Objections in Data Presentations: Responding with Confidence and Clarity  (Data Visualization) <6_data_visualization/026-handling-objections-in-data-presentations-responding-with-confidence-and-clarity>`
   * :doc:`How Data Analysts Use Spreadsheets  (Data-Driven Decisions) <2_data_driven_decisions/013-how-data-analysts-use-spreadsheets>`
   * :doc:`How Data Analytics Improves the Workplace  (Foundations) <1_foundations/002-how-data-analytics-improves-the-workplace>`
   * :doc:`How Data Is Generated and Collected  (Data Preparation) <3_data_preparation/001-how-data-is-generated-and-collected>`
   * :doc:`How to Effectively Search for Solutions Online as a Data Analyst  (Analyze Data) <5_analyze_data/012-how-to-effectively-search-for-solutions-online-as-a-data-analyst>`
   * :doc:`Identifying Bad Data Sources (When Data Does Not ROCCC)  (Data Preparation) <3_data_preparation/012-identifying-bad-data-sources-when-data-does-not-roccc>`
   * :doc:`Identifying Good Data Sources (ROCCC Framework)  (Data Preparation) <3_data_preparation/011-identifying-good-data-sources-roccc-framework>`
   * :doc:`Importing Data into Spreadsheets  (Data Preparation) <3_data_preparation/020-importing-data-into-spreadsheets>`
   * :doc:`Industries Where Data Analysts Work and How Data Is Used  (Foundations) <1_foundations/024-industries-where-data-analysts-work-and-how-data-is-used>`
   * :doc:`Interview Preparation  (Job Search) <8_job_search/011-interview-preparation>`
   * :doc:`Introduction to NumPy and Vectorization  (Data Analysis Using Python) <7_data_analysis_python/027-introduction-to-numpy-and-vectorization>`
   * :doc:`Introduction to Pandas (Data Analysis Library)  (Data Analysis Using Python) <7_data_analysis_python/029-introduction-to-pandas-data-analysis-library>`
   * :doc:`Introduction to Python and Programming Fundamentals  (Data Analysis Using Python) <7_data_analysis_python/001-introduction-to-python-and-programming-fundamentals>`
   * :doc:`Introduction to SQL  (Data Cleaning & Preparation) <4_data_cleaning_preparation/018-introduction-to-sql>`
   * :doc:`Introduction to Tableau  (Data Visualization) <6_data_visualization/009-introduction-to-tableau>`
   * :doc:`Job Application Tracking (Using AI + Spreadsheets)  (Job Search) <8_job_search/009-job-application-tracking-using-ai-spreadsheets>`
   * :doc:`Job Search Plan (Using AI Tools)  (Job Search) <8_job_search/004-job-search-plan-using-ai-tools>`
   * :doc:`Jupyter Notebook and Coding Environments  (Data Analysis Using Python) <7_data_analysis_python/003-jupyter-notebook-and-coding-environments>`
   * :doc:`Key Factors to Consider When Choosing a Data Analytics Role  (Foundations) <1_foundations/027-key-factors-to-consider-when-choosing-a-data-analytics-role>`
   * :doc:`Libraries, Packages, and Modules in Python  (Data Analysis Using Python) <7_data_analysis_python/026-libraries-packages-and-modules-in-python>`
   * :doc:`Linking Multiple Datasets in Tableau Public  (Data Visualization) <6_data_visualization/014-linking-multiple-datasets-in-tableau-public>`
   * :doc:`Managing Stakeholder Expectations and Project Constraints  (Data-Driven Decisions) <2_data_driven_decisions/023-managing-stakeholder-expectations-and-project-constraints>`
   * :doc:`Margin of Error  (Data Cleaning & Preparation) <4_data_cleaning_preparation/008-margin-of-error>`
   * :doc:`Mathematical Thinking  (Data-Driven Decisions) <2_data_driven_decisions/010-mathematical-thinking>`
   * :doc:`Metadata in Databases  (Data Preparation) <3_data_preparation/017-metadata-in-databases>`
   * :doc:`Metadata Repositories and Data Governance  (Data Preparation) <3_data_preparation/018-metadata-repositories-and-data-governance>`
   * :doc:`Modifying Lists in Python  (Data Analysis Using Python) <7_data_analysis_python/020-modifying-lists-in-python>`
   * :doc:`Naming Conventions and Restrictions in Python  (Data Analysis Using Python) <7_data_analysis_python/006-naming-conventions-and-restrictions-in-python>`
   * :doc:`Networking for Job Search  (Job Search) <8_job_search/010-networking-for-job-search>`
   * :doc:`NumPy Arrays (ndarray) and Core Concepts  (Data Analysis Using Python) <7_data_analysis_python/028-numpy-arrays-ndarray-and-core-concepts>`
   * :doc:`Object-Oriented Programming (OOP) in Python  (Data Analysis Using Python) <7_data_analysis_python/004-object-oriented-programming-oop-in-python>`
   * :doc:`Open Data and Openness in Data Ethics  (Data Preparation) <3_data_preparation/015-open-data-and-openness-in-data-ethics>`
   * :doc:`Organizing Data for Personal and Work Projects  (Data Preparation) <3_data_preparation/024-organizing-data-for-personal-and-work-projects>`
   * :doc:`Overview of Core Tools Used by Data Analysts  (Foundations) <1_foundations/020-overview-of-core-tools-used-by-data-analysts>`
   * :doc:`Pandas DataFrame & Series  (Data Analysis Using Python) <7_data_analysis_python/030-pandas-dataframe-and-series>`
   * :doc:`Population, Sample Size, and Random Sampling  (Data Cleaning & Preparation) <4_data_cleaning_preparation/005-population-sample-size-and-random-sampling>`
   * :doc:`Post-Interview Strategy  (Job Search) <8_job_search/015-post-interview-strategy>`
   * :doc:`Practical Application of the Data Analysis Process  (Foundations) <1_foundations/012-practical-application-of-the-data-analysis-process>`
   * :doc:`Practicing Interviews with AI (Gemini Live)  (Job Search) <8_job_search/014-practicing-interviews-with-ai-gemini-live>`
   * :doc:`Preparing Data for VLOOKUP in Spreadsheets  (Analyze Data) <5_analyze_data/014-preparing-data-for-vlookup-in-spreadsheets>`
   * :doc:`Preparing for Q&A: Anticipating and Responding to Stakeholder Questions  (Data Visualization) <6_data_visualization/025-preparing-for-q-and-a-anticipating-and-responding-to-stakeholder-questions>`
   * :doc:`Presentation Skills for Data Analysts: Delivering Insights with Confidence  (Data Visualization) <6_data_visualization/023-presentation-skills-for-data-analysts-delivering-insights-with-confidence>`
   * :doc:`Presenting Like a Pro: Best Practices for Data Analysts  (Data Visualization) <6_data_visualization/024-presenting-like-a-pro-best-practices-for-data-analysts>`
   * :doc:`Problem-Solving and Seeking Help in Data Analysis  (Analyze Data) <5_analyze_data/011-problem-solving-and-seeking-help-in-data-analysis>`
   * :doc:`Python Fundamentals  (Data Analysis Using Python) <7_data_analysis_python/002-python-fundamentals>`
   * :doc:`Q&A Best Practices: Answering Questions with Clarity and Confidence  (Data Visualization) <6_data_visualization/027-q-and-a-best-practices-answering-questions-with-clarity-and-confidence>`
   * :doc:`Quantitative and Qualitative Data in Decision-Making  (Data-Driven Decisions) <2_data_driven_decisions/006-quantitative-and-qualitative-data-in-decision-making>`
   * :doc:`Querying Data with SQL  (Data Preparation) <3_data_preparation/023-querying-data-with-sql>`
   * :doc:`range() Function and Loop Control in Python  (Data Analysis Using Python) <7_data_analysis_python/015-range-function-and-loop-control-in-python>`
   * :doc:`Refining a Resume for Data Analytics Roles  (Data Cleaning & Preparation) <4_data_cleaning_preparation/031-refining-a-resume-for-data-analytics-roles>`
   * :doc:`Reporting Data-Cleaning Results  (Data Cleaning & Preparation) <4_data_cleaning_preparation/029-reporting-data-cleaning-results>`
   * :doc:`Root Cause Analysis and Business Applications of the Five Whys  (Foundations) <1_foundations/017-root-cause-analysis-and-business-applications-of-the-five-whys>`
   * :doc:`Sample Size and Data Integrity  (Data Cleaning & Preparation) <4_data_cleaning_preparation/007-sample-size-and-data-integrity>`
   * :doc:`Sampling Bias and Unbiased Data  (Data Preparation) <3_data_preparation/009-sampling-bias-and-unbiased-data>`
   * :doc:`Sets in Python  (Data Analysis Using Python) <7_data_analysis_python/025-sets-in-python>`
   * :doc:`Sharing Data to Drive Impact  (Data-Driven Decisions) <2_data_driven_decisions/025-sharing-data-to-drive-impact>`
   * :doc:`Sorting and Filtering Data in Spreadsheets  (Data Preparation) <3_data_preparation/021-sorting-and-filtering-data-in-spreadsheets>`
   * :doc:`Sorting and Filtering Data in SQL Using ORDER BY and WHERE  (Analyze Data) <5_analyze_data/005-sorting-and-filtering-data-in-sql-using-order-by-and-where>`
   * :doc:`Sorting and Filtering in Data Analysis  (Analyze Data) <5_analyze_data/003-sorting-and-filtering-in-data-analysis>`
   * :doc:`Sorting Data in Spreadsheets  (Analyze Data) <5_analyze_data/004-sorting-data-in-spreadsheets>`
   * :doc:`Spreadsheet Calculations with Formulas  (Data-Driven Decisions) <2_data_driven_decisions/014-spreadsheet-calculations-with-formulas>`
   * :doc:`Spreadsheet Functions  (Data-Driven Decisions) <2_data_driven_decisions/016-spreadsheet-functions>`
   * :doc:`Spreadsheet Tools for Data Cleaning  (Data Cleaning & Preparation) <4_data_cleaning_preparation/014-spreadsheet-tools-for-data-cleaning>`
   * :doc:`Spreadsheets in Data Analysis  (Data-Driven Decisions) <2_data_driven_decisions/011-spreadsheets-in-data-analysis>`
   * :doc:`Spreadsheets vs. SQL  (Data Cleaning & Preparation) <4_data_cleaning_preparation/019-spreadsheets-vs-sql>`
   * :doc:`Stakeholder Expectations in Data Analysis  (Data-Driven Decisions) <2_data_driven_decisions/019-stakeholder-expectations-in-data-analysis>`
   * :doc:`STAR Method (Behavioral Interview)  (Job Search) <8_job_search/012-star-method-behavioral-interview>`
   * :doc:`Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity  (Data Visualization) <6_data_visualization/004-static-vs-dynamic-data-visualizations-design-tradeoffs-control-and-interactivity>`
   * :doc:`Statistical Power in Data Analysis  (Data Cleaning & Preparation) <4_data_cleaning_preparation/006-statistical-power-in-data-analysis>`
   * :doc:`Staying Focused on the Project Objective  (Data-Driven Decisions) <2_data_driven_decisions/020-staying-focused-on-the-project-objective>`
   * :doc:`String Formatting with .format() in Python  (Data Analysis Using Python) <7_data_analysis_python/018-string-formatting-with-format-in-python>`
   * :doc:`String Indexing and Slicing in Python  (Data Analysis Using Python) <7_data_analysis_python/017-string-indexing-and-slicing-in-python>`
   * :doc:`Strings in Python  (Data Analysis Using Python) <7_data_analysis_python/016-strings-in-python>`
   * :doc:`Structured Data and Data Models  (Data Preparation) <3_data_preparation/004-structured-data-and-data-models>`
   * :doc:`Structuring a Persuasive Data Presentation: Turning Insights into Story  (Data Visualization) <6_data_visualization/019-structuring-a-persuasive-data-presentation-turning-insights-into-story>`
   * :doc:`Subqueries in SQL  (Analyze Data) <5_analyze_data/018-subqueries-in-sql>`
   * :doc:`Tailoring Your Resume  (Job Search) <8_job_search/005-tailoring-your-resume>`
   * :doc:`Temporary Tables and the WITH Clause in SQL  (Analyze Data) <5_analyze_data/029-temporary-tables-and-the-with-clause-in-sql>`
   * :doc:`The Concept and Basic Use of SQL (Query Language)  (Foundations) <1_foundations/022-the-concept-and-basic-use-of-sql-query-language>`
   * :doc:`The Difference Between Data and Metrics, and the Role of Metrics  (Data-Driven Decisions) <2_data_driven_decisions/008-the-difference-between-data-and-metrics-and-the-role-of-metrics>`
   * :doc:`The Importance of Clean Data  (Data Cleaning & Preparation) <4_data_cleaning_preparation/001-the-importance-of-clean-data>`
   * :doc:`The Importance of Clean Data (revisited)  (Data Cleaning & Preparation) <4_data_cleaning_preparation/010-the-importance-of-clean-data-revisited>`
   * :doc:`The Origins of Data Analysis and the Many Ways to Structure It  (Foundations) <1_foundations/006-the-origins-of-data-analysis-and-the-many-ways-to-structure-it>`
   * :doc:`The Relationship Between Data and Decision-Making  (Data-Driven Decisions) <2_data_driven_decisions/005-the-relationship-between-data-and-decision-making>`
   * :doc:`The Role and Importance of Data Visualization  (Foundations) <1_foundations/023-the-role-and-importance-of-data-visualization>`
   * :doc:`The Role of Business Tasks in Data Analysis  (Foundations) <1_foundations/025-the-role-of-business-tasks-in-data-analysis>`
   * :doc:`The Role of Spreadsheets in Data Analysis and Basic Concepts  (Foundations) <1_foundations/021-the-role-of-spreadsheets-in-data-analysis-and-basic-concepts>`
   * :doc:`The Six Phases of the Data Analysis Process  (Foundations) <1_foundations/005-the-six-phases-of-the-data-analysis-process>`
   * :doc:`The Stages of the Data Analysis Process and Their Roles  (Foundations) <1_foundations/011-the-stages-of-the-data-analysis-process-and-their-roles>`
   * :doc:`Transferable Skills  (Job Search) <8_job_search/001-transferable-skills>`
   * :doc:`Troubleshooting VLOOKUP and Building a Problem-Solving Framework  (Analyze Data) <5_analyze_data/016-troubleshooting-vlookup-and-building-a-problem-solving-framework>`
   * :doc:`Tuples in Python  (Data Analysis Using Python) <7_data_analysis_python/021-tuples-in-python>`
   * :doc:`Understanding Bias in Data Analysis  (Data Preparation) <3_data_preparation/008-understanding-bias-in-data-analysis>`
   * :doc:`Understanding Common Problem Types in Data Analytics  (Data-Driven Decisions) <2_data_driven_decisions/002-understanding-common-problem-types-in-data-analytics>`
   * :doc:`Understanding Data Analysis  (Analyze Data) <5_analyze_data/001-understanding-data-analysis>`
   * :doc:`Understanding Data Types and Data Formats  (Data Preparation) <3_data_preparation/003-understanding-data-types-and-data-formats>`
   * :doc:`Understanding the Data Analysis Process and the Data Life Cycle  (Foundations) <1_foundations/008-understanding-the-data-analysis-process-and-the-data-life-cycle>`
   * :doc:`Understanding the Data Ecosystem  (Foundations) <1_foundations/007-understanding-the-data-ecosystem>`
   * :doc:`Understanding the Data Life Cycle  (Foundations) <1_foundations/009-understanding-the-data-life-cycle>`
   * :doc:`Using a Strategic Framework to Structure Data Presentations  (Data Visualization) <6_data_visualization/021-using-a-strategic-framework-to-structure-data-presentations>`
   * :doc:`Using AI (NotebookLM) for Interview Preparation  (Job Search) <8_job_search/013-using-ai-notebooklm-for-interview-preparation>`
   * :doc:`Using AI to Improve and Tailor Your Resume  (Job Search) <8_job_search/006-using-ai-to-improve-and-tailor-your-resume>`
   * :doc:`Using CAST to Clean and Format Data in SQL  (Data Cleaning & Preparation) <4_data_cleaning_preparation/022-using-cast-to-clean-and-format-data-in-sql>`
   * :doc:`Using CONCAT in SQL to Combine Text from Multiple Columns  (Analyze Data) <5_analyze_data/009-using-concat-in-sql-to-combine-text-from-multiple-columns>`
   * :doc:`Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets  (Analyze Data) <5_analyze_data/021-using-countif-and-sumif-for-conditional-aggregation-in-spreadsheets>`
   * :doc:`Using Creativity in Tableau  (Data Visualization) <6_data_visualization/013-using-creativity-in-tableau>`
   * :doc:`Using Data Analysis to Choose the Right Advertising Strategy  (Data-Driven Decisions) <2_data_driven_decisions/001-using-data-analysis-to-choose-the-right-advertising-strategy>`
   * :doc:`Using Feedback from Data Cleaning to Improve Data Quality  (Data Cleaning & Preparation) <4_data_cleaning_preparation/030-using-feedback-from-data-cleaning-to-improve-data-quality>`
   * :doc:`Using Filters to Create Compelling and Focused Visuals  (Data Visualization) <6_data_visualization/018-using-filters-to-create-compelling-and-focused-visuals>`
   * :doc:`Using GROUP BY and ORDER BY for Aggregated Calculations in SQL  (Analyze Data) <5_analyze_data/027-using-group-by-and-order-by-for-aggregated-calculations-in-sql>`
   * :doc:`Using JOIN in SQL to Combine Tables  (Analyze Data) <5_analyze_data/017-using-join-in-sql-to-combine-tables>`
   * :doc:`Using Pivot Table Filters and Calculated Fields for Deeper Analysis  (Analyze Data) <5_analyze_data/024-using-pivot-table-filters-and-calculated-fields-for-deeper-analysis>`
   * :doc:`Using Pivot Tables for Calculations and Trend Analysis  (Analyze Data) <5_analyze_data/023-using-pivot-tables-for-calculations-and-trend-analysis>`
   * :doc:`Using Spreadsheet Formulas for Sales Trend Analysis  (Analyze Data) <5_analyze_data/020-using-spreadsheet-formulas-for-sales-trend-analysis>`
   * :doc:`Using Spreadsheet Functions for Data Cleaning  (Data Cleaning & Preparation) <4_data_cleaning_preparation/015-using-spreadsheet-functions-for-data-cleaning>`
   * :doc:`Using SUMPRODUCT for Advanced Spreadsheet Calculations  (Analyze Data) <5_analyze_data/022-using-sumproduct-for-advanced-spreadsheet-calculations>`
   * :doc:`Using VLOOKUP to Combine Data Across Spreadsheets  (Analyze Data) <5_analyze_data/015-using-vlookup-to-combine-data-across-spreadsheets>`
   * :doc:`Variables in Python  (Data Analysis Using Python) <7_data_analysis_python/005-variables-in-python>`
   * :doc:`Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors  (Data Cleaning & Preparation) <4_data_cleaning_preparation/027-verification-techniques-using-spreadsheets-and-sql-to-catch-repeated-errors>`
   * :doc:`Verifying and Reporting Data Integrity  (Data Cleaning & Preparation) <4_data_cleaning_preparation/025-verifying-and-reporting-data-integrity>`
   * :doc:`Verifying Data-Cleaning Efforts  (Data Cleaning & Preparation) <4_data_cleaning_preparation/026-verifying-data-cleaning-efforts>`
   * :doc:`Viewing Data Differently for More Effective Data Cleaning  (Data Cleaning & Preparation) <4_data_cleaning_preparation/016-viewing-data-differently-for-more-effective-data-cleaning>`
   * :doc:`Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method  (Data Visualization) <6_data_visualization/022-weaving-data-into-presentations-hypotheses-context-and-the-mccandless-method>`
   * :doc:`While Loops and Iteration in Python  (Data Analysis Using Python) <7_data_analysis_python/013-while-loops-and-iteration-in-python>`
   * :doc:`Why Asking the Right Questions Matters in Data Analytics  (Data-Driven Decisions) <2_data_driven_decisions/004-why-asking-the-right-questions-matters-in-data-analytics>`
   * :doc:`Why Data Analytics Matters Today  (Foundations) <1_foundations/001-why-data-analytics-matters-today>`
   * :doc:`Wide Data vs. Long Data  (Data Preparation) <3_data_preparation/007-wide-data-vs-long-data>`
   * :doc:`Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)  (Analyze Data) <5_analyze_data/010-working-with-strings-in-spreadsheets-len-left-right-find>`

.. toctree::
   :hidden:
   :maxdepth: 1

   1_foundations/001-why-data-analytics-matters-today
   1_foundations/002-how-data-analytics-improves-the-workplace
   1_foundations/003-data-driven-decision-making
   1_foundations/004-detectives-and-data-analysts
   1_foundations/005-the-six-phases-of-the-data-analysis-process
   1_foundations/006-the-origins-of-data-analysis-and-the-many-ways-to-structure-it
   1_foundations/007-understanding-the-data-ecosystem
   1_foundations/008-understanding-the-data-analysis-process-and-the-data-life-cycle
   1_foundations/009-understanding-the-data-life-cycle
   1_foundations/010-a-review-of-the-six-stages-of-the-data-life-cycle
   1_foundations/011-the-stages-of-the-data-analysis-process-and-their-roles
   1_foundations/012-practical-application-of-the-data-analysis-process
   1_foundations/013-analytical-skills-and-their-core-components
   1_foundations/014-applying-analytical-skills-in-a-business-context
   1_foundations/015-analytical-thinking-and-its-core-components
   1_foundations/016-analytical-thinking-and-questions-for-problem-solving
   1_foundations/017-root-cause-analysis-and-business-applications-of-the-five-whys
   1_foundations/018-data-driven-decision-making-and-the-role-of-analytical-skills
   1_foundations/019-case-studies-in-data-analysis-and-the-practical-impact-of-data-driven-decision-making
   1_foundations/020-overview-of-core-tools-used-by-data-analysts
   1_foundations/021-the-role-of-spreadsheets-in-data-analysis-and-basic-concepts
   1_foundations/022-the-concept-and-basic-use-of-sql-query-language
   1_foundations/023-the-role-and-importance-of-data-visualization
   1_foundations/024-industries-where-data-analysts-work-and-how-data-is-used
   1_foundations/025-the-role-of-business-tasks-in-data-analysis
   1_foundations/026-fairness-in-data-analysis
   1_foundations/027-key-factors-to-consider-when-choosing-a-data-analytics-role
   2_data_driven_decisions/001-using-data-analysis-to-choose-the-right-advertising-strategy
   2_data_driven_decisions/002-understanding-common-problem-types-in-data-analytics
   2_data_driven_decisions/003-applying-data-analytics-problem-types-in-real-business-scenarios
   2_data_driven_decisions/004-why-asking-the-right-questions-matters-in-data-analytics
   2_data_driven_decisions/005-the-relationship-between-data-and-decision-making
   2_data_driven_decisions/006-quantitative-and-qualitative-data-in-decision-making
   2_data_driven_decisions/007-data-creates-value-only-when-it-is-communicated
   2_data_driven_decisions/008-the-difference-between-data-and-metrics-and-the-role-of-metrics
   2_data_driven_decisions/009-dashboards
   2_data_driven_decisions/010-mathematical-thinking
   2_data_driven_decisions/011-spreadsheets-in-data-analysis
   2_data_driven_decisions/012-building-and-organizing-a-spreadsheet
   2_data_driven_decisions/013-how-data-analysts-use-spreadsheets
   2_data_driven_decisions/014-spreadsheet-calculations-with-formulas
   2_data_driven_decisions/015-common-spreadsheet-errors-and-how-to-fix-them
   2_data_driven_decisions/016-spreadsheet-functions
   2_data_driven_decisions/017-defining-the-problem-domain
   2_data_driven_decisions/018-context-and-bias-in-data-analysis
   2_data_driven_decisions/019-stakeholder-expectations-in-data-analysis
   2_data_driven_decisions/020-staying-focused-on-the-project-objective
   2_data_driven_decisions/021-clear-communication-with-stakeholders-and-teams
   2_data_driven_decisions/022-adapting-to-communication-expectations-at-work
   2_data_driven_decisions/023-managing-stakeholder-expectations-and-project-constraints
   2_data_driven_decisions/024-balancing-speed-and-accuracy-in-data-analysis
   2_data_driven_decisions/025-sharing-data-to-drive-impact
   2_data_driven_decisions/026-effective-meetings
   2_data_driven_decisions/027-conflict-resolution-in-the-workplace
   3_data_preparation/001-how-data-is-generated-and-collected
   3_data_preparation/002-choosing-the-right-data-to-collect
   3_data_preparation/003-understanding-data-types-and-data-formats
   3_data_preparation/004-structured-data-and-data-models
   3_data_preparation/005-data-types-in-spreadsheets
   3_data_preparation/006-data-tables-tabular-data
   3_data_preparation/007-wide-data-vs-long-data
   3_data_preparation/008-understanding-bias-in-data-analysis
   3_data_preparation/009-sampling-bias-and-unbiased-data
   3_data_preparation/010-common-types-of-data-bias
   3_data_preparation/011-identifying-good-data-sources-roccc-framework
   3_data_preparation/012-identifying-bad-data-sources-when-data-does-not-roccc
   3_data_preparation/013-data-ethics-in-data-analysis
   3_data_preparation/014-data-privacy-in-data-ethics
   3_data_preparation/015-open-data-and-openness-in-data-ethics
   3_data_preparation/016-databases-and-relational-database-concepts
   3_data_preparation/017-metadata-in-databases
   3_data_preparation/018-metadata-repositories-and-data-governance
   3_data_preparation/019-accessing-data-internal-and-external-sources
   3_data_preparation/020-importing-data-into-spreadsheets
   3_data_preparation/021-sorting-and-filtering-data-in-spreadsheets
   3_data_preparation/022-bigquery-account-types
   3_data_preparation/023-querying-data-with-sql
   3_data_preparation/024-organizing-data-for-personal-and-work-projects
   3_data_preparation/025-data-security-in-spreadsheets
   4_data_cleaning_preparation/001-the-importance-of-clean-data
   4_data_cleaning_preparation/002-data-integrity-and-its-risks-in-data-analysis
   4_data_cleaning_preparation/003-aligning-data-with-business-objectives
   4_data_cleaning_preparation/004-handling-insufficient-data-in-data-analysis
   4_data_cleaning_preparation/005-population-sample-size-and-random-sampling
   4_data_cleaning_preparation/006-statistical-power-in-data-analysis
   4_data_cleaning_preparation/007-sample-size-and-data-integrity
   4_data_cleaning_preparation/008-margin-of-error
   4_data_cleaning_preparation/009-dirty-data-vs-clean-data
   4_data_cleaning_preparation/010-the-importance-of-clean-data-revisited
   4_data_cleaning_preparation/011-common-issues-in-dirty-data
   4_data_cleaning_preparation/012-data-cleaning-with-spreadsheets
   4_data_cleaning_preparation/013-cleaning-and-merging-multiple-datasets
   4_data_cleaning_preparation/014-spreadsheet-tools-for-data-cleaning
   4_data_cleaning_preparation/015-using-spreadsheet-functions-for-data-cleaning
   4_data_cleaning_preparation/016-viewing-data-differently-for-more-effective-data-cleaning
   4_data_cleaning_preparation/017-data-mapping-and-the-big-picture-of-clean-data
   4_data_cleaning_preparation/018-introduction-to-sql
   4_data_cleaning_preparation/019-spreadsheets-vs-sql
   4_data_cleaning_preparation/020-core-sql-queries-for-data-cleaning-and-analysis
   4_data_cleaning_preparation/021-cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables
   4_data_cleaning_preparation/022-using-cast-to-clean-and-format-data-in-sql
   4_data_cleaning_preparation/023-advanced-sql-functions-for-data-cleaning
   4_data_cleaning_preparation/024-coalesce
   4_data_cleaning_preparation/025-verifying-and-reporting-data-integrity
   4_data_cleaning_preparation/026-verifying-data-cleaning-efforts
   4_data_cleaning_preparation/027-verification-techniques-using-spreadsheets-and-sql-to-catch-repeated-errors
   4_data_cleaning_preparation/028-documenting-data-cleaning-changes
   4_data_cleaning_preparation/029-reporting-data-cleaning-results
   4_data_cleaning_preparation/030-using-feedback-from-data-cleaning-to-improve-data-quality
   4_data_cleaning_preparation/031-refining-a-resume-for-data-analytics-roles
   4_data_cleaning_preparation/032-exploring-data-analyst-job-opportunities
   5_analyze_data/001-understanding-data-analysis
   5_analyze_data/002-data-organization-in-analysis
   5_analyze_data/003-sorting-and-filtering-in-data-analysis
   5_analyze_data/004-sorting-data-in-spreadsheets
   5_analyze_data/005-sorting-and-filtering-data-in-sql-using-order-by-and-where
   5_analyze_data/006-data-formatting-and-unit-conversion-in-spreadsheets
   5_analyze_data/007-data-validation-in-spreadsheets
   5_analyze_data/008-combining-data-validation-and-conditional-formatting-in-spreadsheets
   5_analyze_data/009-using-concat-in-sql-to-combine-text-from-multiple-columns
   5_analyze_data/010-working-with-strings-in-spreadsheets-len-left-right-find
   5_analyze_data/011-problem-solving-and-seeking-help-in-data-analysis
   5_analyze_data/012-how-to-effectively-search-for-solutions-online-as-a-data-analyst
   5_analyze_data/013-choosing-the-right-tool-in-data-analysis
   5_analyze_data/014-preparing-data-for-vlookup-in-spreadsheets
   5_analyze_data/015-using-vlookup-to-combine-data-across-spreadsheets
   5_analyze_data/016-troubleshooting-vlookup-and-building-a-problem-solving-framework
   5_analyze_data/017-using-join-in-sql-to-combine-tables
   5_analyze_data/018-subqueries-in-sql
   5_analyze_data/019-aggregating-data-with-subqueries-having-and-case-in-sql
   5_analyze_data/020-using-spreadsheet-formulas-for-sales-trend-analysis
   5_analyze_data/021-using-countif-and-sumif-for-conditional-aggregation-in-spreadsheets
   5_analyze_data/022-using-sumproduct-for-advanced-spreadsheet-calculations
   5_analyze_data/023-using-pivot-tables-for-calculations-and-trend-analysis
   5_analyze_data/024-using-pivot-table-filters-and-calculated-fields-for-deeper-analysis
   5_analyze_data/025-comparing-calculations-in-spreadsheets-and-sql
   5_analyze_data/026-embedding-calculations-in-sql-queries
   5_analyze_data/027-using-group-by-and-order-by-for-aggregated-calculations-in-sql
   5_analyze_data/028-data-validation-as-an-ongoing-analytical-process
   5_analyze_data/029-temporary-tables-and-the-with-clause-in-sql
   5_analyze_data/030-creating-temporary-tables-in-sql-methods-trade-offs-and-best-practices
   6_data_visualization/001-data-visualization
   6_data_visualization/002-connecting-data-and-images
   6_data_visualization/003-creating-powerful-data-visualizations-focus-structure-and-analytical-purpose
   6_data_visualization/004-static-vs-dynamic-data-visualizations-design-tradeoffs-control-and-interactivity
   6_data_visualization/005-elements-of-art-in-data-visualization-line-shape-color-space-and-movement
   6_data_visualization/006-choosing-the-right-visualization-audience-centered-design-and-chart-selection
   6_data_visualization/007-design-thinking-in-data-visualization-a-user-centered-framework
   6_data_visualization/008-accessibility-in-data-visualization-designing-for-everyone
   6_data_visualization/009-introduction-to-tableau
   6_data_visualization/010-getting-started-with-tableau-public
   6_data_visualization/011-creating-a-co-emissions-visualization-in-tableau-public
   6_data_visualization/012-effective-vs-ineffective-data-visualizations-in-tableau
   6_data_visualization/013-using-creativity-in-tableau
   6_data_visualization/014-linking-multiple-datasets-in-tableau-public
   6_data_visualization/015-data-storytelling-giving-numbers-a-clear-and-convincing-voice
   6_data_visualization/016-engaging-your-audience-in-data-storytelling-identifying-the-key-message
   6_data_visualization/017-data-dashboards-organizing-insight-for-real-time-decision-making
   6_data_visualization/018-using-filters-to-create-compelling-and-focused-visuals
   6_data_visualization/019-structuring-a-persuasive-data-presentation-turning-insights-into-story
   6_data_visualization/020-designing-effective-data-presentation-slides-structure-visuals-and-professional-impact
   6_data_visualization/021-using-a-strategic-framework-to-structure-data-presentations
   6_data_visualization/022-weaving-data-into-presentations-hypotheses-context-and-the-mccandless-method
   6_data_visualization/023-presentation-skills-for-data-analysts-delivering-insights-with-confidence
   6_data_visualization/024-presenting-like-a-pro-best-practices-for-data-analysts
   6_data_visualization/025-preparing-for-q-and-a-anticipating-and-responding-to-stakeholder-questions
   6_data_visualization/026-handling-objections-in-data-presentations-responding-with-confidence-and-clarity
   6_data_visualization/027-q-and-a-best-practices-answering-questions-with-clarity-and-confidence
   7_data_analysis_python/001-introduction-to-python-and-programming-fundamentals
   7_data_analysis_python/002-python-fundamentals
   7_data_analysis_python/003-jupyter-notebook-and-coding-environments
   7_data_analysis_python/004-object-oriented-programming-oop-in-python
   7_data_analysis_python/005-variables-in-python
   7_data_analysis_python/006-naming-conventions-and-restrictions-in-python
   7_data_analysis_python/007-data-types-and-type-conversion-in-python
   7_data_analysis_python/008-functions-in-python
   7_data_analysis_python/009-code-reusability-modularity-and-clean-code-in-python
   7_data_analysis_python/010-comments-algorithms-and-docstrings-in-python
   7_data_analysis_python/011-boolean-data-comparators-and-logical-operators-in-python
   7_data_analysis_python/012-branching-and-conditional-statements-in-python
   7_data_analysis_python/013-while-loops-and-iteration-in-python
   7_data_analysis_python/014-for-loops-in-python
   7_data_analysis_python/015-range-function-and-loop-control-in-python
   7_data_analysis_python/016-strings-in-python
   7_data_analysis_python/017-string-indexing-and-slicing-in-python
   7_data_analysis_python/018-string-formatting-with-format-in-python
   7_data_analysis_python/019-data-types-vs-data-structures-and-introduction-to-lists
   7_data_analysis_python/020-modifying-lists-in-python
   7_data_analysis_python/021-tuples-in-python
   7_data_analysis_python/022-advanced-use-of-loops-lists-tuples-and-list-comprehension
   7_data_analysis_python/023-dictionaries-in-python
   7_data_analysis_python/024-advanced-dictionary-usage-in-python
   7_data_analysis_python/025-sets-in-python
   7_data_analysis_python/026-libraries-packages-and-modules-in-python
   7_data_analysis_python/027-introduction-to-numpy-and-vectorization
   7_data_analysis_python/028-numpy-arrays-ndarray-and-core-concepts
   7_data_analysis_python/029-introduction-to-pandas-data-analysis-library
   7_data_analysis_python/030-pandas-dataframe-and-series
   7_data_analysis_python/031-boolean-masking-in-pandas
   7_data_analysis_python/032-grouping-and-aggregation-in-pandas-groupby-agg
   7_data_analysis_python/033-combining-data-in-pandas-concat-and-merge
   8_job_search/001-transferable-skills
   8_job_search/002-career-identity-statement
   8_job_search/003-career-dreamer-ai-tool-for-career-exploration
   8_job_search/004-job-search-plan-using-ai-tools
   8_job_search/005-tailoring-your-resume
   8_job_search/006-using-ai-to-improve-and-tailor-your-resume
   8_job_search/007-building-a-professional-online-presence-personal-brand
   8_job_search/008-choosing-the-right-job-platforms
   8_job_search/009-job-application-tracking-using-ai-spreadsheets
   8_job_search/010-networking-for-job-search
   8_job_search/011-interview-preparation
   8_job_search/012-star-method-behavioral-interview
   8_job_search/013-using-ai-notebooklm-for-interview-preparation
   8_job_search/014-practicing-interviews-with-ai-gemini-live
   8_job_search/015-post-interview-strategy

.. tags:: purpose: reference, topic: data analytics
