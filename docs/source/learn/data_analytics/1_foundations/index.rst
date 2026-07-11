.. _da-foundations-index:

========================================================================
🌱 Foundations
========================================================================

*Section 1 of the Data Analytics hub — 27 of 27 lessons.*

The case for data, the analysis process and data life cycle, analytical thinking, and the core tools of the trade.

:doc:`↑ Back to the Data Analytics hub <../index>`

.. raw:: html

   <input type="text" id="term-filter" placeholder="🔍 Filter this section by title or keyword…"
          style="width:100%;padding:.6em .8em;margin:.4em 0 1em;font-size:1em;
                 border:1px solid var(--pst-color-border,#ccc);border-radius:6px;box-sizing:border-box;">
   <div id="term-filter-count" style="margin:-.6em 0 1em;font-size:.85em;opacity:.7;"></div>
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


.. dropdown:: 🌟 Why Data Analytics
   :class-container: sd-dropdown

   Why data matters today and how it reshapes work and decision-making.

   .. raw:: html

         <div class="da-row" data-k="Why Data Analytics Matters Today why organisations that decide with data outperform those that decide by instinct"><a href="001-why-data-analytics-matters-today.html">001 · Why Data Analytics Matters Today — why organisations that decide with data outperform those that decide by instinct</a></div>
         <div class="da-row" data-k="How Data Analytics Improves the Workplace where analytics pays off day to day: operations, decisions, and shared facts"><a href="002-how-data-analytics-improves-the-workplace.html">002 · How Data Analytics Improves the Workplace — where analytics pays off day to day: operations, decisions, and shared facts</a></div>
         <div class="da-row" data-k="Data-Driven Decision-Making the loop from question to data to decision, and the evidence it works"><a href="003-data-driven-decision-making.html">003 · Data-Driven Decision-Making — the loop from question to data to decision, and the evidence it works</a></div>
         <div class="da-row" data-k="Detectives and Data Analysts the investigator's mindset: questions, evidence, and conclusions that hold up"><a href="004-detectives-and-data-analysts.html">004 · Detectives and Data Analysts — the investigator's mindset: questions, evidence, and conclusions that hold up</a></div>

.. dropdown:: 🔄 The Analysis Process & Data Life Cycle
   :class-container: sd-dropdown

   The six phases of analysis and the life cycle data moves through.

   .. raw:: html

         <div class="da-row" data-k="The Six Phases of the Data Analysis Process ask, prepare, process, analyze, share, act — the map for every project in this course"><a href="005-the-six-phases-of-the-data-analysis-process.html">005 · The Six Phases of the Data Analysis Process — ask, prepare, process, analyze, share, act — the map for every project in this course</a></div>
         <div class="da-row" data-k="The Origins of Data Analysis and the Many Ways to Structure It from early statistics to EDA and CRISP-DM: one process, many framings"><a href="006-the-origins-of-data-analysis-and-the-many-ways-to-structure-it.html">006 · The Origins of Data Analysis and the Many Ways to Structure It — from early statistics to EDA and CRISP-DM: one process, many framings</a></div>
         <div class="da-row" data-k="Understanding the Data Ecosystem the interlocking pieces — sources, storage, tools, people — data moves through"><a href="007-understanding-the-data-ecosystem.html">007 · Understanding the Data Ecosystem — the interlocking pieces — sources, storage, tools, people — data moves through</a></div>
         <div class="da-row" data-k="Understanding the Data Analysis Process and the Data Life Cycle two different journeys: what the analyst does vs. what happens to the data"><a href="008-understanding-the-data-analysis-process-and-the-data-life-cycle.html">008 · Understanding the Data Analysis Process and the Data Life Cycle — two different journeys: what the analyst does vs. what happens to the data</a></div>
         <div class="da-row" data-k="Understanding the Data Life Cycle plan, capture, manage, analyze, archive, destroy — the data's own biography"><a href="009-understanding-the-data-life-cycle.html">009 · Understanding the Data Life Cycle — plan, capture, manage, analyze, archive, destroy — the data's own biography</a></div>
         <div class="da-row" data-k="A Review of the Six Stages of the Data Life Cycle the six stages consolidated, with what can go wrong at each"><a href="010-a-review-of-the-six-stages-of-the-data-life-cycle.html">010 · A Review of the Six Stages of the Data Life Cycle — the six stages consolidated, with what can go wrong at each</a></div>
         <div class="da-row" data-k="The Stages of the Data Analysis Process and Their Roles what each analysis phase contributes, and what it hands to the next"><a href="011-the-stages-of-the-data-analysis-process-and-their-roles.html">011 · The Stages of the Data Analysis Process and Their Roles — what each analysis phase contributes, and what it hands to the next</a></div>
         <div class="da-row" data-k="Practical Application of the Data Analysis Process the six phases run end-to-end on a real-shaped business case"><a href="012-practical-application-of-the-data-analysis-process.html">012 · Practical Application of the Data Analysis Process — the six phases run end-to-end on a real-shaped business case</a></div>

.. dropdown:: 🧠 Analytical Skills & Thinking
   :class-container: sd-dropdown

   Analytical skills, structured questioning, and root-cause methods.

   .. raw:: html

         <div class="da-row" data-k="Analytical Skills and Their Core Components the five skills — curiosity, context, technical mindset, data design, data strategy"><a href="013-analytical-skills-and-their-core-components.html">013 · Analytical Skills and Their Core Components — the five skills — curiosity, context, technical mindset, data design, data strategy</a></div>
         <div class="da-row" data-k="Applying Analytical Skills in a Business Context the five skills at work on a real business problem, phase by phase"><a href="014-applying-analytical-skills-in-a-business-context.html">014 · Applying Analytical Skills in a Business Context — the five skills at work on a real business problem, phase by phase</a></div>
         <div class="da-row" data-k="Analytical Thinking and Its Core Components the five aspects — visualization, strategy, problem-orientation, correlation, big picture + detail"><a href="015-analytical-thinking-and-its-core-components.html">015 · Analytical Thinking and Its Core Components — the five aspects — visualization, strategy, problem-orientation, correlation, big picture + detail</a></div>
         <div class="da-row" data-k="Analytical Thinking and Questions for Problem Solving turning the aspects into questions: root causes, gaps, and the unconsidered"><a href="016-analytical-thinking-and-questions-for-problem-solving.html">016 · Analytical Thinking and Questions for Problem Solving — turning the aspects into questions: root causes, gaps, and the unconsidered</a></div>
         <div class="da-row" data-k="Root Cause Analysis and Business Applications of the Five Whys "><a href="017-root-cause-analysis-and-business-applications-of-the-five-whys.html">017 · Root Cause Analysis and Business Applications of the Five Whys</a></div>
         <div class="da-row" data-k="Data-Driven Decision-Making and the Role of Analytical Skills "><a href="018-data-driven-decision-making-and-the-role-of-analytical-skills.html">018 · Data-Driven Decision-Making and the Role of Analytical Skills</a></div>
         <div class="da-row" data-k="Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making "><a href="019-case-studies-in-data-analysis-and-the-practical-impact-of-data-driven-decision-making.html">019 · Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making</a></div>

.. dropdown:: 🧰 Tools, Applications & Ethics
   :class-container: sd-dropdown

   Spreadsheets, SQL, visualization, industry uses, and fairness.

   .. raw:: html

         <div class="da-row" data-k="Overview of Core Tools Used by Data Analysts "><a href="020-overview-of-core-tools-used-by-data-analysts.html">020 · Overview of Core Tools Used by Data Analysts</a></div>
         <div class="da-row" data-k="The Role of Spreadsheets in Data Analysis and Basic Concepts "><a href="021-the-role-of-spreadsheets-in-data-analysis-and-basic-concepts.html">021 · The Role of Spreadsheets in Data Analysis and Basic Concepts</a></div>
         <div class="da-row" data-k="The Concept and Basic Use of SQL (Query Language) "><a href="022-the-concept-and-basic-use-of-sql-query-language.html">022 · The Concept and Basic Use of SQL (Query Language)</a></div>
         <div class="da-row" data-k="The Role and Importance of Data Visualization "><a href="023-the-role-and-importance-of-data-visualization.html">023 · The Role and Importance of Data Visualization</a></div>
         <div class="da-row" data-k="Industries Where Data Analysts Work and How Data Is Used "><a href="024-industries-where-data-analysts-work-and-how-data-is-used.html">024 · Industries Where Data Analysts Work and How Data Is Used</a></div>
         <div class="da-row" data-k="The Role of Business Tasks in Data Analysis "><a href="025-the-role-of-business-tasks-in-data-analysis.html">025 · The Role of Business Tasks in Data Analysis</a></div>
         <div class="da-row" data-k="Fairness in Data Analysis "><a href="026-fairness-in-data-analysis.html">026 · Fairness in Data Analysis</a></div>
         <div class="da-row" data-k="Key Factors to Consider When Choosing a Data Analytics Role "><a href="027-key-factors-to-consider-when-choosing-a-data-analytics-role.html">027 · Key Factors to Consider When Choosing a Data Analytics Role</a></div>

.. dropdown:: 🔠 A–Z index
   :class-container: term-az

   .. raw:: html

         <div class="da-row" data-k="A Review of the Six Stages of the Data Life Cycle"><a href="010-a-review-of-the-six-stages-of-the-data-life-cycle.html">A Review of the Six Stages of the Data Life Cycle</a></div>
         <div class="da-row" data-k="Analytical Skills and Their Core Components"><a href="013-analytical-skills-and-their-core-components.html">Analytical Skills and Their Core Components</a></div>
         <div class="da-row" data-k="Analytical Thinking and Its Core Components"><a href="015-analytical-thinking-and-its-core-components.html">Analytical Thinking and Its Core Components</a></div>
         <div class="da-row" data-k="Analytical Thinking and Questions for Problem Solving"><a href="016-analytical-thinking-and-questions-for-problem-solving.html">Analytical Thinking and Questions for Problem Solving</a></div>
         <div class="da-row" data-k="Applying Analytical Skills in a Business Context"><a href="014-applying-analytical-skills-in-a-business-context.html">Applying Analytical Skills in a Business Context</a></div>
         <div class="da-row" data-k="Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making"><a href="019-case-studies-in-data-analysis-and-the-practical-impact-of-data-driven-decision-making.html">Case Studies in Data Analysis and the Practical Impact of Data-Driven Decision-Making</a></div>
         <div class="da-row" data-k="Data-Driven Decision-Making"><a href="003-data-driven-decision-making.html">Data-Driven Decision-Making</a></div>
         <div class="da-row" data-k="Data-Driven Decision-Making and the Role of Analytical Skills"><a href="018-data-driven-decision-making-and-the-role-of-analytical-skills.html">Data-Driven Decision-Making and the Role of Analytical Skills</a></div>
         <div class="da-row" data-k="Detectives and Data Analysts"><a href="004-detectives-and-data-analysts.html">Detectives and Data Analysts</a></div>
         <div class="da-row" data-k="Fairness in Data Analysis"><a href="026-fairness-in-data-analysis.html">Fairness in Data Analysis</a></div>
         <div class="da-row" data-k="How Data Analytics Improves the Workplace"><a href="002-how-data-analytics-improves-the-workplace.html">How Data Analytics Improves the Workplace</a></div>
         <div class="da-row" data-k="Industries Where Data Analysts Work and How Data Is Used"><a href="024-industries-where-data-analysts-work-and-how-data-is-used.html">Industries Where Data Analysts Work and How Data Is Used</a></div>
         <div class="da-row" data-k="Key Factors to Consider When Choosing a Data Analytics Role"><a href="027-key-factors-to-consider-when-choosing-a-data-analytics-role.html">Key Factors to Consider When Choosing a Data Analytics Role</a></div>
         <div class="da-row" data-k="Overview of Core Tools Used by Data Analysts"><a href="020-overview-of-core-tools-used-by-data-analysts.html">Overview of Core Tools Used by Data Analysts</a></div>
         <div class="da-row" data-k="Practical Application of the Data Analysis Process"><a href="012-practical-application-of-the-data-analysis-process.html">Practical Application of the Data Analysis Process</a></div>
         <div class="da-row" data-k="Root Cause Analysis and Business Applications of the Five Whys"><a href="017-root-cause-analysis-and-business-applications-of-the-five-whys.html">Root Cause Analysis and Business Applications of the Five Whys</a></div>
         <div class="da-row" data-k="The Concept and Basic Use of SQL (Query Language)"><a href="022-the-concept-and-basic-use-of-sql-query-language.html">The Concept and Basic Use of SQL (Query Language)</a></div>
         <div class="da-row" data-k="The Origins of Data Analysis and the Many Ways to Structure It"><a href="006-the-origins-of-data-analysis-and-the-many-ways-to-structure-it.html">The Origins of Data Analysis and the Many Ways to Structure It</a></div>
         <div class="da-row" data-k="The Role and Importance of Data Visualization"><a href="023-the-role-and-importance-of-data-visualization.html">The Role and Importance of Data Visualization</a></div>
         <div class="da-row" data-k="The Role of Business Tasks in Data Analysis"><a href="025-the-role-of-business-tasks-in-data-analysis.html">The Role of Business Tasks in Data Analysis</a></div>
         <div class="da-row" data-k="The Role of Spreadsheets in Data Analysis and Basic Concepts"><a href="021-the-role-of-spreadsheets-in-data-analysis-and-basic-concepts.html">The Role of Spreadsheets in Data Analysis and Basic Concepts</a></div>
         <div class="da-row" data-k="The Six Phases of the Data Analysis Process"><a href="005-the-six-phases-of-the-data-analysis-process.html">The Six Phases of the Data Analysis Process</a></div>
         <div class="da-row" data-k="The Stages of the Data Analysis Process and Their Roles"><a href="011-the-stages-of-the-data-analysis-process-and-their-roles.html">The Stages of the Data Analysis Process and Their Roles</a></div>
         <div class="da-row" data-k="Understanding the Data Analysis Process and the Data Life Cycle"><a href="008-understanding-the-data-analysis-process-and-the-data-life-cycle.html">Understanding the Data Analysis Process and the Data Life Cycle</a></div>
         <div class="da-row" data-k="Understanding the Data Ecosystem"><a href="007-understanding-the-data-ecosystem.html">Understanding the Data Ecosystem</a></div>
         <div class="da-row" data-k="Understanding the Data Life Cycle"><a href="009-understanding-the-data-life-cycle.html">Understanding the Data Life Cycle</a></div>
         <div class="da-row" data-k="Why Data Analytics Matters Today"><a href="001-why-data-analytics-matters-today.html">Why Data Analytics Matters Today</a></div>
