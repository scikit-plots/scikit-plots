:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-foundations-index:

:raw-html:`<div style="text-align:center"><strong>` 🌱 Foundations
|br| |full_version| - |today|
:raw-html:`</strong></div>`

========================================================================
🌱 Foundations
========================================================================

*Section 1 of the Data Analytics hub — 27 of 27 lessons.*

The case for data, the analysis process and data life cycle, analytical thinking, and the core tools of the trade.

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


.. dropdown:: 🌟 Why Data Analytics
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Why data matters today and how it reshapes work and decision-making.

   * :doc:`001 · Why Data Analytics Matters Today — why organisations that decide with data outperform those that decide by instinct <001-why-data-analytics-matters-today>`
   * :doc:`002 · How Data Analytics Improves the Workplace — where analytics pays off day to day: operations, decisions, and shared facts <002-how-data-analytics-improves-the-workplace>`
   * :doc:`003 · Data-Driven Decision-Making — the loop from question to data to decision, and the evidence it works <003-data-driven-decision-making>`
   * :doc:`004 · Detectives and Data Analysts — the investigator's mindset: questions, evidence, and conclusions that hold up <004-detectives-and-data-analysts>`

.. dropdown:: 🔄 The Analysis Process & Data Life Cycle
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   The six phases of analysis and the life cycle data moves through.

   * :doc:`005 · The Six Phases of the Data Analysis Process — ask, prepare, process, analyze, share, act — the map for every project in this course <005-the-six-phases-of-the-data-analysis-process>`
   * :doc:`006 · The Origins of Data Analysis and the Many Ways to Structure It — from early statistics to EDA and CRISP-DM: one process, many framings <006-the-origins-of-data-analysis-and-the-many-ways-to-structure-it>`
   * :doc:`007 · Understanding the Data Ecosystem — the interlocking pieces — sources, storage, tools, people — data moves through <007-understanding-the-data-ecosystem>`
   * :doc:`008 · Understanding the Data Analysis Process and the Data Life Cycle — two different journeys: what the analyst does vs. what happens to the data <008-understanding-the-data-analysis-process-and-the-data-life-cycle>`
   * :doc:`009 · Understanding the Data Life Cycle — plan, capture, manage, analyze, archive, destroy — the data's own biography <009-understanding-the-data-life-cycle>`
   * :doc:`010 · A Review of the Six Stages of the Data Life Cycle — the six stages consolidated, with what can go wrong at each <010-a-review-of-the-six-stages-of-the-data-life-cycle>`
   * :doc:`011 · The Stages of the Data Analysis Process and Their Roles — what each analysis phase contributes, and what it hands to the next <011-the-stages-of-the-data-analysis-process-and-their-roles>`
   * :doc:`012 · Practical Application of the Data Analysis Process — the six phases run end-to-end on a real-shaped business case <012-practical-application-of-the-data-analysis-process>`

.. dropdown:: 🧠 Analytical Skills & Thinking
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Analytical skills, structured questioning, and root-cause methods.

   * :doc:`013 · Analytical Skills and Their Core Components — the five skills — curiosity, context, technical mindset, data design, data strategy <013-analytical-skills-and-their-core-components>`
   * :doc:`014 · Applying Analytical Skills in a Business Context — the five skills at work on a real business problem, phase by phase <014-applying-analytical-skills-in-a-business-context>`
   * :doc:`015 · Analytical Thinking and Its Core Components — the five aspects — visualization, strategy, problem-orientation, correlation, big picture + detail <015-analytical-thinking-and-its-core-components>`
   * :doc:`016 · Analytical Thinking and Questions for Problem Solving — turning the aspects into questions: root causes, gaps, and the unconsidered <016-analytical-thinking-and-questions-for-problem-solving>`
   * :doc:`017 · Root Cause Analysis and Business Applications of the Five Whys — Toyota's why-times-five: digging past symptoms to causes worth fixing <017-root-cause-analysis-and-business-applications-of-the-five-whys>`
   * :doc:`018 · Data-Driven Decision-Making and the Role of Analytical Skills — how the five skills power each step of the decision loop <018-data-driven-decision-making-and-the-role-of-analytical-skills>`
   * :doc:`019 · Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making — the evidence in action: research findings and worked cases where data changed the outcome <019-case-studies-in-data-analysis-and-the-practical-impact-of-data-driven-decision-making>`

.. dropdown:: 🧰 Tools, Applications & Ethics
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Spreadsheets, SQL, visualization, industry uses, and fairness.

   * :doc:`020 · Overview of Core Tools Used by Data Analysts — the working toolkit — spreadsheets, SQL, visualization tools, and code — and when each fits <020-overview-of-core-tools-used-by-data-analysts>`
   * :doc:`021 · The Role of Spreadsheets in Data Analysis and Basic Concepts — the analyst's first instrument: cells, formulas, and functions in a visible grid <021-the-role-of-spreadsheets-in-data-analysis-and-basic-concepts>`
   * :doc:`022 · The Concept and Basic Use of SQL (Query Language) — asking databases questions: SELECT, FROM, WHERE and why they scale <022-the-concept-and-basic-use-of-sql-query-language>`
   * :doc:`023 · The Role and Importance of Data Visualization — why pictures beat tables: seeing structure that summary numbers hide <023-the-role-and-importance-of-data-visualization>`
   * :doc:`024 · Industries Where Data Analysts Work and How Data Is Used — the same craft in many rooms: retail to healthcare to public service <024-industries-where-data-analysts-work-and-how-data-is-used>`
   * :doc:`025 · The Role of Business Tasks in Data Analysis — the question behind the work: how business tasks anchor every analysis <025-the-role-of-business-tasks-in-data-analysis>`
   * :doc:`026 · Fairness in Data Analysis — analysis that does not create or reinforce bias — and a famous failure to learn from <026-fairness-in-data-analysis>`
   * :doc:`027 · Key Factors to Consider When Choosing a Data Analytics Role — industry, company size, specialisation, growth: weighing where to start <027-key-factors-to-consider-when-choosing-a-data-analytics-role>`

🔤 Every lesson, A–Z index
---------------------------

.. dropdown:: 🔠 Open the full alphabetical index
   :animate: fade-in-slide-down
   :class-container: term-az

   .. hlist::
      :columns: 2

      * :doc:`A Review of the Six Stages of the Data Life Cycle <010-a-review-of-the-six-stages-of-the-data-life-cycle>`
      * :doc:`Analytical Skills and Their Core Components <013-analytical-skills-and-their-core-components>`
      * :doc:`Analytical Thinking and Its Core Components <015-analytical-thinking-and-its-core-components>`
      * :doc:`Analytical Thinking and Questions for Problem Solving <016-analytical-thinking-and-questions-for-problem-solving>`
      * :doc:`Applying Analytical Skills in a Business Context <014-applying-analytical-skills-in-a-business-context>`
      * :doc:`Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making <019-case-studies-in-data-analysis-and-the-practical-impact-of-data-driven-decision-making>`
      * :doc:`Data-Driven Decision-Making <003-data-driven-decision-making>`
      * :doc:`Data-Driven Decision-Making and the Role of Analytical Skills <018-data-driven-decision-making-and-the-role-of-analytical-skills>`
      * :doc:`Detectives and Data Analysts <004-detectives-and-data-analysts>`
      * :doc:`Fairness in Data Analysis <026-fairness-in-data-analysis>`
      * :doc:`How Data Analytics Improves the Workplace <002-how-data-analytics-improves-the-workplace>`
      * :doc:`Industries Where Data Analysts Work and How Data Is Used <024-industries-where-data-analysts-work-and-how-data-is-used>`
      * :doc:`Key Factors to Consider When Choosing a Data Analytics Role <027-key-factors-to-consider-when-choosing-a-data-analytics-role>`
      * :doc:`Overview of Core Tools Used by Data Analysts <020-overview-of-core-tools-used-by-data-analysts>`
      * :doc:`Practical Application of the Data Analysis Process <012-practical-application-of-the-data-analysis-process>`
      * :doc:`Root Cause Analysis and Business Applications of the Five Whys <017-root-cause-analysis-and-business-applications-of-the-five-whys>`
      * :doc:`The Concept and Basic Use of SQL (Query Language) <022-the-concept-and-basic-use-of-sql-query-language>`
      * :doc:`The Origins of Data Analysis and the Many Ways to Structure It <006-the-origins-of-data-analysis-and-the-many-ways-to-structure-it>`
      * :doc:`The Role and Importance of Data Visualization <023-the-role-and-importance-of-data-visualization>`
      * :doc:`The Role of Business Tasks in Data Analysis <025-the-role-of-business-tasks-in-data-analysis>`
      * :doc:`The Role of Spreadsheets in Data Analysis and Basic Concepts <021-the-role-of-spreadsheets-in-data-analysis-and-basic-concepts>`
      * :doc:`The Six Phases of the Data Analysis Process <005-the-six-phases-of-the-data-analysis-process>`
      * :doc:`The Stages of the Data Analysis Process and Their Roles <011-the-stages-of-the-data-analysis-process-and-their-roles>`
      * :doc:`Understanding the Data Analysis Process and the Data Life Cycle <008-understanding-the-data-analysis-process-and-the-data-life-cycle>`
      * :doc:`Understanding the Data Ecosystem <007-understanding-the-data-ecosystem>`
      * :doc:`Understanding the Data Life Cycle <009-understanding-the-data-life-cycle>`
      * :doc:`Why Data Analytics Matters Today <001-why-data-analytics-matters-today>`

.. toctree::
   :hidden:
   :includehidden:
   :maxdepth: 1

   001-why-data-analytics-matters-today
   002-how-data-analytics-improves-the-workplace
   003-data-driven-decision-making
   004-detectives-and-data-analysts
   005-the-six-phases-of-the-data-analysis-process
   006-the-origins-of-data-analysis-and-the-many-ways-to-structure-it
   007-understanding-the-data-ecosystem
   008-understanding-the-data-analysis-process-and-the-data-life-cycle
   009-understanding-the-data-life-cycle
   010-a-review-of-the-six-stages-of-the-data-life-cycle
   011-the-stages-of-the-data-analysis-process-and-their-roles
   012-practical-application-of-the-data-analysis-process
   013-analytical-skills-and-their-core-components
   014-applying-analytical-skills-in-a-business-context
   015-analytical-thinking-and-its-core-components
   016-analytical-thinking-and-questions-for-problem-solving
   017-root-cause-analysis-and-business-applications-of-the-five-whys
   018-data-driven-decision-making-and-the-role-of-analytical-skills
   019-case-studies-in-data-analysis-and-the-practical-impact-of-data-driven-decision-making
   020-overview-of-core-tools-used-by-data-analysts
   021-the-role-of-spreadsheets-in-data-analysis-and-basic-concepts
   022-the-concept-and-basic-use-of-sql-query-language
   023-the-role-and-importance-of-data-visualization
   024-industries-where-data-analysts-work-and-how-data-is-used
   025-the-role-of-business-tasks-in-data-analysis
   026-fairness-in-data-analysis
   027-key-factors-to-consider-when-choosing-a-data-analytics-role

.. tags:: purpose: reference, topic: data analytics
