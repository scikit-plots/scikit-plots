:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-ddd-index:

:raw-html:`<div style="text-align:center"><strong>` 🎯 Data-Driven Decisions
|br| |full_version| - |today|
:raw-html:`</strong></div>`

========================================================================
🎯 Data-Driven Decisions
========================================================================

*Section 2 of the Data Analytics hub — 27 of 27 lessons.*

Turning questions into decisions: stakeholders, metrics, and communicating results that drive action.

:doc:`↑ Back to the Data Analytics hub <../index>`

.. raw:: html

   <div style="text-align:center;margin:0.4rem 0 0.4rem">
   <input id="term-filter" type="search" autocomplete="off" spellcheck="false"
           placeholder="&#128269;&nbsp; Type to filter this section lessons &mdash; by title or keyword&hellip;"
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


.. dropdown:: 🧭 Framing the Problem
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Problem types, asking the right questions, and how data drives decisions.

   * :doc:`001 · Using Data Analysis to Choose the Right Advertising Strategy — a worked prediction problem: past campaign data steering the next ad spend <001-using-data-analysis-to-choose-the-right-advertising-strategy>`
   * :doc:`002 · Understanding Common Problem Types in Data Analytics — the six shapes of analyst problems, from predictions to patterns <002-understanding-common-problem-types-in-data-analytics>`
   * :doc:`003 · Applying Data Analytics Problem Types in Real Business Scenarios — recognising the six types in the wild — and why naming the type speeds the work <003-applying-data-analytics-problem-types-in-real-business-scenarios>`
   * :doc:`004 · Why Asking the Right Questions Matters in Data Analytics — SMART questions, and the leading/closed/assuming questions to avoid <004-why-asking-the-right-questions-matters-in-data-analytics>`
   * :doc:`005 · The Relationship Between Data and Decision-Making — data-driven vs data-inspired: how evidence and judgement actually combine <005-the-relationship-between-data-and-decision-making>`
   * :doc:`006 · Quantitative and Qualitative Data in Decision-Making — what/how many/how often meets why: two data kinds, one decision <006-quantitative-and-qualitative-data-in-decision-making>`
   * :doc:`007 · Data Creates Value Only When It Is Communicated — an insight nobody hears changes nothing: communication as the last mile <007-data-creates-value-only-when-it-is-communicated>`

.. dropdown:: 📐 Metrics & Dashboards
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Data versus metrics, dashboards, and quantitative thinking.

   * :doc:`008 · The Difference Between Data and Metrics, and the Role of Metrics — from raw facts to quantified goals: what makes a number a metric <008-the-difference-between-data-and-metrics-and-the-role-of-metrics>`
   * :doc:`009 · Dashboards — live metrics in one place: what dashboards are for, and when a report beats one <009-dashboards>`
   * :doc:`010 · Mathematical Thinking — step-by-step decomposition, orders of magnitude, and choosing data sized to the decision <010-mathematical-thinking>`

.. dropdown:: 📗 Spreadsheets for Analysis
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Organising, calculating, and troubleshooting analysis in spreadsheets.

   * :doc:`011 · Spreadsheets in Data Analysis — where the spreadsheet sits in the decision workflow, phase by phase <011-spreadsheets-in-data-analysis>`
   * :doc:`012 · Building and Organizing a Spreadsheet — one row per record, clean headers, raw kept raw: layouts that survive analysis <012-building-and-organizing-a-spreadsheet>`
   * :doc:`013 · How Data Analysts Use Spreadsheets — the everyday spreadsheet toolkit — sort, filter, pivot, chart — mapped to real tasks <013-how-data-analysts-use-spreadsheets>`
   * :doc:`014 · Spreadsheet Calculations with Formulas — formulas, cell references, and the absolute-vs-relative distinction that makes fill-down work <014-spreadsheet-calculations-with-formulas>`
   * :doc:`015 · Common Spreadsheet Errors and How to Fix Them — reading #DIV/0!, #VALUE!, #REF!, #NAME?, #N/A — and fixing causes, not symptoms <015-common-spreadsheet-errors-and-how-to-fix-them>`
   * :doc:`016 · Spreadsheet Functions — named operations — SUM, AVERAGE, IF, VLOOKUP — the analyst's core vocabulary <016-spreadsheet-functions>`

.. dropdown:: 🗣 Stakeholders, Communication & Execution
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Scoping with stakeholders and communicating results that drive action.

   * :doc:`017 · Defining the Problem Domain — scoping before analysing: problem, stakeholders, deliverables, timeline, success <017-defining-the-problem-domain>`
   * :doc:`018 · Context and Bias in Data Analysis — context makes data meaningful; unexamined context is where bias enters <018-context-and-bias-in-data-analysis>`
   * :doc:`019 · Stakeholder Expectations in Data Analysis — who has a stake, what they need, and how to align before the work begins <019-stakeholder-expectations-in-data-analysis>`
   * :doc:`020 · Staying Focused on the Project Objective — guarding scope: keeping every step tied to the one question that matters <020-staying-focused-on-the-project-objective>`
   * :doc:`021 · Clear Communication with Stakeholders and Teams — the five Cs of clarity: making findings land with the people who act on them <021-clear-communication-with-stakeholders-and-teams>`
   * :doc:`022 · Adapting to Communication Expectations at Work — reading the room: matching channel, register, and detail to each audience <022-adapting-to-communication-expectations-at-work>`
   * :doc:`023 · Managing Stakeholder Expectations and Project Constraints — the iron triangle of scope, time, and resources — and honest trade-offs <023-managing-stakeholder-expectations-and-project-constraints>`
   * :doc:`024 · Balancing Speed and Accuracy in Data Analysis — fast-enough vs right-enough: matching rigour to the decision's stakes <024-balancing-speed-and-accuracy-in-data-analysis>`
   * :doc:`025 · Sharing Data to Drive Impact — the difference between reporting numbers and actually changing a decision <025-sharing-data-to-drive-impact>`
   * :doc:`026 · Effective Meetings — purpose, preparation, and follow-up: making the time analysts spend together count <026-effective-meetings>`
   * :doc:`027 · Conflict Resolution in the Workplace — when the data disagrees with a person: disagreeing productively and separating issue from ego <027-conflict-resolution-in-the-workplace>`

🔤 Every lesson, A–Z
---------------------

.. dropdown:: 🔠 A–Z index
   :animate: fade-in-slide-down
   :class-container: term-az

   .. hlist::
      :columns: 2

      * :doc:`Adapting to Communication Expectations at Work <022-adapting-to-communication-expectations-at-work>`
      * :doc:`Applying Data Analytics Problem Types in Real Business Scenarios <003-applying-data-analytics-problem-types-in-real-business-scenarios>`
      * :doc:`Balancing Speed and Accuracy in Data Analysis <024-balancing-speed-and-accuracy-in-data-analysis>`
      * :doc:`Building and Organizing a Spreadsheet <012-building-and-organizing-a-spreadsheet>`
      * :doc:`Clear Communication with Stakeholders and Teams <021-clear-communication-with-stakeholders-and-teams>`
      * :doc:`Common Spreadsheet Errors and How to Fix Them <015-common-spreadsheet-errors-and-how-to-fix-them>`
      * :doc:`Conflict Resolution in the Workplace <027-conflict-resolution-in-the-workplace>`
      * :doc:`Context and Bias in Data Analysis <018-context-and-bias-in-data-analysis>`
      * :doc:`Dashboards <009-dashboards>`
      * :doc:`Data Creates Value Only When It Is Communicated <007-data-creates-value-only-when-it-is-communicated>`
      * :doc:`Defining the Problem Domain <017-defining-the-problem-domain>`
      * :doc:`Effective Meetings <026-effective-meetings>`
      * :doc:`How Data Analysts Use Spreadsheets <013-how-data-analysts-use-spreadsheets>`
      * :doc:`Managing Stakeholder Expectations and Project Constraints <023-managing-stakeholder-expectations-and-project-constraints>`
      * :doc:`Mathematical Thinking <010-mathematical-thinking>`
      * :doc:`Quantitative and Qualitative Data in Decision-Making <006-quantitative-and-qualitative-data-in-decision-making>`
      * :doc:`Sharing Data to Drive Impact <025-sharing-data-to-drive-impact>`
      * :doc:`Spreadsheet Calculations with Formulas <014-spreadsheet-calculations-with-formulas>`
      * :doc:`Spreadsheet Functions <016-spreadsheet-functions>`
      * :doc:`Spreadsheets in Data Analysis <011-spreadsheets-in-data-analysis>`
      * :doc:`Stakeholder Expectations in Data Analysis <019-stakeholder-expectations-in-data-analysis>`
      * :doc:`Staying Focused on the Project Objective <020-staying-focused-on-the-project-objective>`
      * :doc:`The Difference Between Data and Metrics, and the Role of Metrics <008-the-difference-between-data-and-metrics-and-the-role-of-metrics>`
      * :doc:`The Relationship Between Data and Decision-Making <005-the-relationship-between-data-and-decision-making>`
      * :doc:`Understanding Common Problem Types in Data Analytics <002-understanding-common-problem-types-in-data-analytics>`
      * :doc:`Using Data Analysis to Choose the Right Advertising Strategy <001-using-data-analysis-to-choose-the-right-advertising-strategy>`
      * :doc:`Why Asking the Right Questions Matters in Data Analytics <004-why-asking-the-right-questions-matters-in-data-analytics>`

.. toctree::
   :hidden:
   :includehidden:
   :maxdepth: 1

   001-using-data-analysis-to-choose-the-right-advertising-strategy
   002-understanding-common-problem-types-in-data-analytics
   003-applying-data-analytics-problem-types-in-real-business-scenarios
   004-why-asking-the-right-questions-matters-in-data-analytics
   005-the-relationship-between-data-and-decision-making
   006-quantitative-and-qualitative-data-in-decision-making
   007-data-creates-value-only-when-it-is-communicated
   008-the-difference-between-data-and-metrics-and-the-role-of-metrics
   009-dashboards
   010-mathematical-thinking
   011-spreadsheets-in-data-analysis
   012-building-and-organizing-a-spreadsheet
   013-how-data-analysts-use-spreadsheets
   014-spreadsheet-calculations-with-formulas
   015-common-spreadsheet-errors-and-how-to-fix-them
   016-spreadsheet-functions
   017-defining-the-problem-domain
   018-context-and-bias-in-data-analysis
   019-stakeholder-expectations-in-data-analysis
   020-staying-focused-on-the-project-objective
   021-clear-communication-with-stakeholders-and-teams
   022-adapting-to-communication-expectations-at-work
   023-managing-stakeholder-expectations-and-project-constraints
   024-balancing-speed-and-accuracy-in-data-analysis
   025-sharing-data-to-drive-impact
   026-effective-meetings
   027-conflict-resolution-in-the-workplace

.. tags:: purpose: reference, topic: data analytics
