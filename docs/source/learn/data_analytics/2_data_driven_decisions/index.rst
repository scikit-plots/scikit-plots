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
   <input type="text" id="term-filter" placeholder="🔍 Type to filter this section &mdash; by title or keyword…"
          style="width:100%;padding:.6em .8em;margin:.4em 0 1em;font-size:1em;
                 border:1px solid var(--pst-color-border,#ccc);border-radius:6px;box-sizing:border-box;">
   <div id="term-filter-count" style="margin:-.6em 0 1em;font-size:.85em;opacity:.7;"></div>
   </div>
   <script>
   (function(){
     var box=document.getElementById('term-filter');
     if(!box)return;
     var count=document.getElementById('term-filter-count');
     function norm(s){return (s||'').toLowerCase();}
     function run(){
       var q=norm(box.value), shown=0, total=0;
       document.querySelectorAll('.da-row').forEach(function(row){
         total++;
         var hit=q===''||norm(row.getAttribute('data-k')).indexOf(q)>=0;
         row.style.display=hit?'':'none';
         if(hit)shown++;
       });
       document.querySelectorAll('details.sd-dropdown, details.term-az').forEach(function(d){
         var any=d.querySelectorAll('.da-row:not([style*="none"])').length>0;
         d.style.display=any?'':'none';
         if(q!==''&&any)d.setAttribute('open','');
       });
       count.textContent=q===''?'':(shown+' of '+total+' match');
     }
     box.addEventListener('input',run);
   })();
   </script>


.. dropdown:: 🧭 Framing the Problem
   :class-container: sd-dropdown

   Problem types, asking the right questions, and how data drives decisions.

   .. raw:: html

         <div class="da-row" data-k="Using Data Analysis to Choose the Right Advertising Strategy a worked prediction problem: past campaign data steering the next ad spend"><a href="001-using-data-analysis-to-choose-the-right-advertising-strategy.html">001 · Using Data Analysis to Choose the Right Advertising Strategy — a worked prediction problem: past campaign data steering the next ad spend</a></div>
         <div class="da-row" data-k="Understanding Common Problem Types in Data Analytics the six shapes of analyst problems, from predictions to patterns"><a href="002-understanding-common-problem-types-in-data-analytics.html">002 · Understanding Common Problem Types in Data Analytics — the six shapes of analyst problems, from predictions to patterns</a></div>
         <div class="da-row" data-k="Applying Data Analytics Problem Types in Real Business Scenarios recognising the six types in the wild — and why naming the type speeds the work"><a href="003-applying-data-analytics-problem-types-in-real-business-scenarios.html">003 · Applying Data Analytics Problem Types in Real Business Scenarios — recognising the six types in the wild — and why naming the type speeds the work</a></div>
         <div class="da-row" data-k="Why Asking the Right Questions Matters in Data Analytics SMART questions, and the leading/closed/assuming questions to avoid"><a href="004-why-asking-the-right-questions-matters-in-data-analytics.html">004 · Why Asking the Right Questions Matters in Data Analytics — SMART questions, and the leading/closed/assuming questions to avoid</a></div>
         <div class="da-row" data-k="The Relationship Between Data and Decision-Making data-driven vs data-inspired: how evidence and judgement actually combine"><a href="005-the-relationship-between-data-and-decision-making.html">005 · The Relationship Between Data and Decision-Making — data-driven vs data-inspired: how evidence and judgement actually combine</a></div>
         <div class="da-row" data-k="Quantitative and Qualitative Data in Decision-Making what/how many/how often meets why: two data kinds, one decision"><a href="006-quantitative-and-qualitative-data-in-decision-making.html">006 · Quantitative and Qualitative Data in Decision-Making — what/how many/how often meets why: two data kinds, one decision</a></div>
         <div class="da-row" data-k="Data Creates Value Only When It Is Communicated an insight nobody hears changes nothing: communication as the last mile"><a href="007-data-creates-value-only-when-it-is-communicated.html">007 · Data Creates Value Only When It Is Communicated — an insight nobody hears changes nothing: communication as the last mile</a></div>

.. dropdown:: 📐 Metrics & Dashboards
   :class-container: sd-dropdown

   Data versus metrics, dashboards, and quantitative thinking.

   .. raw:: html

         <div class="da-row" data-k="The Difference Between Data and Metrics, and the Role of Metrics from raw facts to quantified goals: what makes a number a metric"><a href="008-the-difference-between-data-and-metrics-and-the-role-of-metrics.html">008 · The Difference Between Data and Metrics, and the Role of Metrics — from raw facts to quantified goals: what makes a number a metric</a></div>
         <div class="da-row" data-k="Dashboards live metrics in one place: what dashboards are for, and when a report beats one"><a href="009-dashboards.html">009 · Dashboards — live metrics in one place: what dashboards are for, and when a report beats one</a></div>
         <div class="da-row" data-k="Mathematical Thinking step-by-step decomposition, orders of magnitude, and choosing data sized to the decision"><a href="010-mathematical-thinking.html">010 · Mathematical Thinking — step-by-step decomposition, orders of magnitude, and choosing data sized to the decision</a></div>

.. dropdown:: 📗 Spreadsheets for Analysis
   :class-container: sd-dropdown

   Organising, calculating, and troubleshooting analysis in spreadsheets.

   .. raw:: html

         <div class="da-row" data-k="Spreadsheets in Data Analysis where the spreadsheet sits in the decision workflow, phase by phase"><a href="011-spreadsheets-in-data-analysis.html">011 · Spreadsheets in Data Analysis — where the spreadsheet sits in the decision workflow, phase by phase</a></div>
         <div class="da-row" data-k="Building and Organizing a Spreadsheet one row per record, clean headers, raw kept raw: layouts that survive analysis"><a href="012-building-and-organizing-a-spreadsheet.html">012 · Building and Organizing a Spreadsheet — one row per record, clean headers, raw kept raw: layouts that survive analysis</a></div>
         <div class="da-row" data-k="How Data Analysts Use Spreadsheets the everyday spreadsheet toolkit — sort, filter, pivot, chart — mapped to real tasks"><a href="013-how-data-analysts-use-spreadsheets.html">013 · How Data Analysts Use Spreadsheets — the everyday spreadsheet toolkit — sort, filter, pivot, chart — mapped to real tasks</a></div>
         <div class="da-row" data-k="Spreadsheet Calculations with Formulas formulas, cell references, and the absolute-vs-relative distinction that makes fill-down work"><a href="014-spreadsheet-calculations-with-formulas.html">014 · Spreadsheet Calculations with Formulas — formulas, cell references, and the absolute-vs-relative distinction that makes fill-down work</a></div>
         <div class="da-row" data-k="Common Spreadsheet Errors and How to Fix Them reading #DIV/0!, #VALUE!, #REF!, #NAME?, #N/A — and fixing causes, not symptoms"><a href="015-common-spreadsheet-errors-and-how-to-fix-them.html">015 · Common Spreadsheet Errors and How to Fix Them — reading #DIV/0!, #VALUE!, #REF!, #NAME?, #N/A — and fixing causes, not symptoms</a></div>
         <div class="da-row" data-k="Spreadsheet Functions named operations — SUM, AVERAGE, IF, VLOOKUP — the analyst's core vocabulary"><a href="016-spreadsheet-functions.html">016 · Spreadsheet Functions — named operations — SUM, AVERAGE, IF, VLOOKUP — the analyst's core vocabulary</a></div>

.. dropdown:: 🗣 Stakeholders, Communication & Execution
   :class-container: sd-dropdown

   Scoping with stakeholders and communicating results that drive action.

   .. raw:: html

         <div class="da-row" data-k="Defining the Problem Domain scoping before analysing: problem, stakeholders, deliverables, timeline, success"><a href="017-defining-the-problem-domain.html">017 · Defining the Problem Domain — scoping before analysing: problem, stakeholders, deliverables, timeline, success</a></div>
         <div class="da-row" data-k="Context and Bias in Data Analysis context makes data meaningful; unexamined context is where bias enters"><a href="018-context-and-bias-in-data-analysis.html">018 · Context and Bias in Data Analysis — context makes data meaningful; unexamined context is where bias enters</a></div>
         <div class="da-row" data-k="Stakeholder Expectations in Data Analysis who has a stake, what they need, and how to align before the work begins"><a href="019-stakeholder-expectations-in-data-analysis.html">019 · Stakeholder Expectations in Data Analysis — who has a stake, what they need, and how to align before the work begins</a></div>
         <div class="da-row" data-k="Staying Focused on the Project Objective guarding scope: keeping every step tied to the one question that matters"><a href="020-staying-focused-on-the-project-objective.html">020 · Staying Focused on the Project Objective — guarding scope: keeping every step tied to the one question that matters</a></div>
         <div class="da-row" data-k="Clear Communication with Stakeholders and Teams the five Cs of clarity: making findings land with the people who act on them"><a href="021-clear-communication-with-stakeholders-and-teams.html">021 · Clear Communication with Stakeholders and Teams — the five Cs of clarity: making findings land with the people who act on them</a></div>
         <div class="da-row" data-k="Adapting to Communication Expectations at Work reading the room: matching channel, register, and detail to each audience"><a href="022-adapting-to-communication-expectations-at-work.html">022 · Adapting to Communication Expectations at Work — reading the room: matching channel, register, and detail to each audience</a></div>
         <div class="da-row" data-k="Managing Stakeholder Expectations and Project Constraints the iron triangle of scope, time, and resources — and honest trade-offs"><a href="023-managing-stakeholder-expectations-and-project-constraints.html">023 · Managing Stakeholder Expectations and Project Constraints — the iron triangle of scope, time, and resources — and honest trade-offs</a></div>
         <div class="da-row" data-k="Balancing Speed and Accuracy in Data Analysis fast-enough vs right-enough: matching rigour to the decision's stakes"><a href="024-balancing-speed-and-accuracy-in-data-analysis.html">024 · Balancing Speed and Accuracy in Data Analysis — fast-enough vs right-enough: matching rigour to the decision's stakes</a></div>
         <div class="da-row" data-k="Sharing Data to Drive Impact the difference between reporting numbers and actually changing a decision"><a href="025-sharing-data-to-drive-impact.html">025 · Sharing Data to Drive Impact — the difference between reporting numbers and actually changing a decision</a></div>
         <div class="da-row" data-k="Effective Meetings purpose, preparation, and follow-up: making the time analysts spend together count"><a href="026-effective-meetings.html">026 · Effective Meetings — purpose, preparation, and follow-up: making the time analysts spend together count</a></div>
         <div class="da-row" data-k="Conflict Resolution in the Workplace when the data disagrees with a person: disagreeing productively and separating issue from ego"><a href="027-conflict-resolution-in-the-workplace.html">027 · Conflict Resolution in the Workplace — when the data disagrees with a person: disagreeing productively and separating issue from ego</a></div>

🔤 Every lesson, A–Z
---------------------

.. dropdown:: 🔠 A–Z index
   :class-container: term-az

   .. raw:: html

         <div class="da-row" data-k="Adapting to Communication Expectations at Work"><a href="022-adapting-to-communication-expectations-at-work.html">Adapting to Communication Expectations at Work</a></div>
         <div class="da-row" data-k="Applying Data Analytics Problem Types in Real Business Scenarios"><a href="003-applying-data-analytics-problem-types-in-real-business-scenarios.html">Applying Data Analytics Problem Types in Real Business Scenarios</a></div>
         <div class="da-row" data-k="Balancing Speed and Accuracy in Data Analysis"><a href="024-balancing-speed-and-accuracy-in-data-analysis.html">Balancing Speed and Accuracy in Data Analysis</a></div>
         <div class="da-row" data-k="Building and Organizing a Spreadsheet"><a href="012-building-and-organizing-a-spreadsheet.html">Building and Organizing a Spreadsheet</a></div>
         <div class="da-row" data-k="Clear Communication with Stakeholders and Teams"><a href="021-clear-communication-with-stakeholders-and-teams.html">Clear Communication with Stakeholders and Teams</a></div>
         <div class="da-row" data-k="Common Spreadsheet Errors and How to Fix Them"><a href="015-common-spreadsheet-errors-and-how-to-fix-them.html">Common Spreadsheet Errors and How to Fix Them</a></div>
         <div class="da-row" data-k="Conflict Resolution in the Workplace"><a href="027-conflict-resolution-in-the-workplace.html">Conflict Resolution in the Workplace</a></div>
         <div class="da-row" data-k="Context and Bias in Data Analysis"><a href="018-context-and-bias-in-data-analysis.html">Context and Bias in Data Analysis</a></div>
         <div class="da-row" data-k="Dashboards"><a href="009-dashboards.html">Dashboards</a></div>
         <div class="da-row" data-k="Data Creates Value Only When It Is Communicated"><a href="007-data-creates-value-only-when-it-is-communicated.html">Data Creates Value Only When It Is Communicated</a></div>
         <div class="da-row" data-k="Defining the Problem Domain"><a href="017-defining-the-problem-domain.html">Defining the Problem Domain</a></div>
         <div class="da-row" data-k="Effective Meetings"><a href="026-effective-meetings.html">Effective Meetings</a></div>
         <div class="da-row" data-k="How Data Analysts Use Spreadsheets"><a href="013-how-data-analysts-use-spreadsheets.html">How Data Analysts Use Spreadsheets</a></div>
         <div class="da-row" data-k="Managing Stakeholder Expectations and Project Constraints"><a href="023-managing-stakeholder-expectations-and-project-constraints.html">Managing Stakeholder Expectations and Project Constraints</a></div>
         <div class="da-row" data-k="Mathematical Thinking"><a href="010-mathematical-thinking.html">Mathematical Thinking</a></div>
         <div class="da-row" data-k="Quantitative and Qualitative Data in Decision-Making"><a href="006-quantitative-and-qualitative-data-in-decision-making.html">Quantitative and Qualitative Data in Decision-Making</a></div>
         <div class="da-row" data-k="Sharing Data to Drive Impact"><a href="025-sharing-data-to-drive-impact.html">Sharing Data to Drive Impact</a></div>
         <div class="da-row" data-k="Spreadsheet Calculations with Formulas"><a href="014-spreadsheet-calculations-with-formulas.html">Spreadsheet Calculations with Formulas</a></div>
         <div class="da-row" data-k="Spreadsheet Functions"><a href="016-spreadsheet-functions.html">Spreadsheet Functions</a></div>
         <div class="da-row" data-k="Spreadsheets in Data Analysis"><a href="011-spreadsheets-in-data-analysis.html">Spreadsheets in Data Analysis</a></div>
         <div class="da-row" data-k="Stakeholder Expectations in Data Analysis"><a href="019-stakeholder-expectations-in-data-analysis.html">Stakeholder Expectations in Data Analysis</a></div>
         <div class="da-row" data-k="Staying Focused on the Project Objective"><a href="020-staying-focused-on-the-project-objective.html">Staying Focused on the Project Objective</a></div>
         <div class="da-row" data-k="The Difference Between Data and Metrics, and the Role of Metrics"><a href="008-the-difference-between-data-and-metrics-and-the-role-of-metrics.html">The Difference Between Data and Metrics, and the Role of Metrics</a></div>
         <div class="da-row" data-k="The Relationship Between Data and Decision-Making"><a href="005-the-relationship-between-data-and-decision-making.html">The Relationship Between Data and Decision-Making</a></div>
         <div class="da-row" data-k="Understanding Common Problem Types in Data Analytics"><a href="002-understanding-common-problem-types-in-data-analytics.html">Understanding Common Problem Types in Data Analytics</a></div>
         <div class="da-row" data-k="Using Data Analysis to Choose the Right Advertising Strategy"><a href="001-using-data-analysis-to-choose-the-right-advertising-strategy.html">Using Data Analysis to Choose the Right Advertising Strategy</a></div>
         <div class="da-row" data-k="Why Asking the Right Questions Matters in Data Analytics"><a href="004-why-asking-the-right-questions-matters-in-data-analytics.html">Why Asking the Right Questions Matters in Data Analytics</a></div>
