:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-index:

:raw-html:`<div style="text-align:center"><strong>` 🧽 Data Cleaning & Preparation
|br| |full_version| - |today|
:raw-html:`</strong></div>`

========================================================================
🧽 Data Cleaning & Preparation
========================================================================

*Section 4 of the Data Analytics hub — 32 of 32 lessons.*

Finding and fixing dirty data: missing values, duplicates, outliers, and validation for trustworthy inputs.

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


.. dropdown:: 🧱 Data Integrity & Sampling
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Why clean data matters, integrity risks, sampling, power, and margin of error.

   * :doc:`001 · The Importance of Clean Data — why clean data is the non-negotiable foundation every analysis stands on <001-the-importance-of-clean-data>`
   * :doc:`002 · Data Integrity and Its Risks in Data Analysis — keeping data accurate and consistent through its life — and what threatens it <002-data-integrity-and-its-risks-in-data-analysis>`
   * :doc:`003 · Aligning Data with Business Objectives — checking that the data you have actually fits the question you must answer <003-aligning-data-with-business-objectives>`
   * :doc:`004 · Handling Insufficient Data in Data Analysis — recognising when there is not enough data, and the honest options when there isn't <004-handling-insufficient-data-in-data-analysis>`
   * :doc:`005 · Population, Sample Size, and Random Sampling — the whole vs. the part: population, sample, and how to sample fairly <005-population-sample-size-and-random-sampling>`
   * :doc:`006 · Statistical Power in Data Analysis — the chance of detecting a real effect when there is one — and why 0.8 is the norm <006-statistical-power-in-data-analysis>`
   * :doc:`007 · Sample Size and Data Integrity — how much data is enough: confidence, precision, and the cost of too little <007-sample-size-and-data-integrity>`
   * :doc:`008 · Margin of Error — the plus-or-minus around a sample estimate, and how to read it honestly <008-margin-of-error>`

.. dropdown:: 🧹 Dirty Data & Spreadsheet Cleaning
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Recognising dirty data and cleaning it with spreadsheet tools and functions.

   * :doc:`009 · Dirty Data vs. Clean Data — the concrete contrast: what makes data dirty, and what clean looks like beside it <009-dirty-data-vs-clean-data>`
   * :doc:`010 · The Importance of Clean Data (revisited) — the business cost of dirty data — why cleaning is worth the time it takes <010-the-importance-of-clean-data-revisited>`
   * :doc:`011 · Common Issues in Dirty Data — the recurring defects — duplicates, missing, inconsistent, wrong type, outliers <011-common-issues-in-dirty-data>`
   * :doc:`012 · Data Cleaning with Spreadsheets — the practical spreadsheet cleaning workflow, defect by defect <012-data-cleaning-with-spreadsheets>`
   * :doc:`013 · Cleaning and Merging Multiple Datasets — combining data from several sources cleanly — matching keys, reconciling formats <013-cleaning-and-merging-multiple-datasets>`
   * :doc:`014 · Spreadsheet Tools for Data Cleaning — the built-in cleaning toolkit: dedupe, split, find/replace, validation, formatting <014-spreadsheet-tools-for-data-cleaning>`
   * :doc:`015 · Using Spreadsheet Functions for Data Cleaning — TRIM, CLEAN, UPPER, SUBSTITUTE, VALUE and friends — cleaning by formula <015-using-spreadsheet-functions-for-data-cleaning>`
   * :doc:`016 · Viewing Data Differently for More Effective Data Cleaning — sort, filter, pivot, conditional formatting as lenses that reveal hidden defects <016-viewing-data-differently-for-more-effective-data-cleaning>`
   * :doc:`017 · Data Mapping and the Big Picture of Clean Data — matching fields between sources so merged data stays coherent — cleaning's big picture <017-data-mapping-and-the-big-picture-of-clean-data>`

.. dropdown:: 🐬 Cleaning with SQL
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   SQL from first queries to CAST, COALESCE, and advanced cleaning functions.

   * :doc:`018 · Introduction to SQL — why SQL matters for cleaning: transformations that scale and stay reproducible <018-introduction-to-sql>`
   * :doc:`019 · Spreadsheets vs. SQL — when to reach for a sheet and when for a query — strengths, limits, and the handoff <019-spreadsheets-vs-sql>`
   * :doc:`020 · Core SQL Queries for Data Cleaning and Analysis — SELECT, WHERE, DISTINCT, GROUP BY — the query patterns that clean and summarise <020-core-sql-queries-for-data-cleaning-and-analysis>`
   * :doc:`021 · Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables — DISTINCT, GROUP BY, and string functions to dedupe and standardise text in SQL <021-cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables>`
   * :doc:`022 · Using CAST to Clean and Format Data in SQL — converting a value from one data type to another — SQL's type-fixing tool <022-using-cast-to-clean-and-format-data-in-sql>`
   * :doc:`023 · Advanced SQL Functions for Data Cleaning — CASE, SUBSTR, TRIM, and pattern tools for the harder cleaning jobs <023-advanced-sql-functions-for-data-cleaning>`
   * :doc:`024 · COALESCE — returning the first non-null value — SQL's standard handler for missing data <024-coalesce>`

.. dropdown:: ✅ Verification, Documentation & Next Steps
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Verifying and reporting cleaning work, documenting changes, and moving forward.

   * :doc:`025 · Verifying and Reporting Data Integrity — confirming data is sound after cleaning — and telling stakeholders it is <025-verifying-and-reporting-data-integrity>`
   * :doc:`026 · Verifying Data-Cleaning Efforts — checking that each cleaning step did what it should, and nothing it shouldn't <026-verifying-data-cleaning-efforts>`
   * :doc:`027 · Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors — concrete checks — counts, distinct values, ranges — that catch recurring defects <027-verification-techniques-using-spreadsheets-and-sql-to-catch-repeated-errors>`
   * :doc:`028 · Documenting Data-Cleaning Changes — recording what was changed and why, so cleaning is reproducible and reviewable <028-documenting-data-cleaning-changes>`
   * :doc:`029 · Reporting Data-Cleaning Results — communicating what cleaning found and fixed, so stakeholders can trust the data <029-reporting-data-cleaning-results>`
   * :doc:`030 · Using Feedback from Data Cleaning to Improve Data Quality — closing the loop: feeding cleaning lessons back to fix problems at the source <030-using-feedback-from-data-cleaning-to-improve-data-quality>`
   * :doc:`031 · Refining a Resume for Data Analytics Roles — shaping a resume to show analytical skill — cleaning your own professional data <031-refining-a-resume-for-data-analytics-roles>`
   * :doc:`032 · Exploring Data Analyst Job Opportunities — where analyst roles are, and how to read them against your skills and goals <032-exploring-data-analyst-job-opportunities>`

🔤 Every lesson, A–Z
---------------------

.. dropdown:: 🔠 A–Z index
   :animate: fade-in-slide-down
   :class-container: term-az

   .. hlist::
      :columns: 2

   * :doc:`Advanced SQL Functions for Data Cleaning <023-advanced-sql-functions-for-data-cleaning>`
   * :doc:`Aligning Data with Business Objectives <003-aligning-data-with-business-objectives>`
   * :doc:`Cleaning and Merging Multiple Datasets <013-cleaning-and-merging-multiple-datasets>`
   * :doc:`Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables <021-cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables>`
   * :doc:`COALESCE <024-coalesce>`
   * :doc:`Common Issues in Dirty Data <011-common-issues-in-dirty-data>`
   * :doc:`Core SQL Queries for Data Cleaning and Analysis <020-core-sql-queries-for-data-cleaning-and-analysis>`
   * :doc:`Data Cleaning with Spreadsheets <012-data-cleaning-with-spreadsheets>`
   * :doc:`Data Integrity and Its Risks in Data Analysis <002-data-integrity-and-its-risks-in-data-analysis>`
   * :doc:`Data Mapping and the Big Picture of Clean Data <017-data-mapping-and-the-big-picture-of-clean-data>`
   * :doc:`Dirty Data vs. Clean Data <009-dirty-data-vs-clean-data>`
   * :doc:`Documenting Data-Cleaning Changes <028-documenting-data-cleaning-changes>`
   * :doc:`Exploring Data Analyst Job Opportunities <032-exploring-data-analyst-job-opportunities>`
   * :doc:`Handling Insufficient Data in Data Analysis <004-handling-insufficient-data-in-data-analysis>`
   * :doc:`Introduction to SQL <018-introduction-to-sql>`
   * :doc:`Margin of Error <008-margin-of-error>`
   * :doc:`Population, Sample Size, and Random Sampling <005-population-sample-size-and-random-sampling>`
   * :doc:`Refining a Resume for Data Analytics Roles <031-refining-a-resume-for-data-analytics-roles>`
   * :doc:`Reporting Data-Cleaning Results <029-reporting-data-cleaning-results>`
   * :doc:`Sample Size and Data Integrity <007-sample-size-and-data-integrity>`
   * :doc:`Spreadsheet Tools for Data Cleaning <014-spreadsheet-tools-for-data-cleaning>`
   * :doc:`Spreadsheets vs. SQL <019-spreadsheets-vs-sql>`
   * :doc:`Statistical Power in Data Analysis <006-statistical-power-in-data-analysis>`
   * :doc:`The Importance of Clean Data <001-the-importance-of-clean-data>`
   * :doc:`The Importance of Clean Data (revisited) <010-the-importance-of-clean-data-revisited>`
   * :doc:`Using CAST to Clean and Format Data in SQL <022-using-cast-to-clean-and-format-data-in-sql>`
   * :doc:`Using Feedback from Data Cleaning to Improve Data Quality <030-using-feedback-from-data-cleaning-to-improve-data-quality>`
   * :doc:`Using Spreadsheet Functions for Data Cleaning <015-using-spreadsheet-functions-for-data-cleaning>`
   * :doc:`Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors <027-verification-techniques-using-spreadsheets-and-sql-to-catch-repeated-errors>`
   * :doc:`Verifying and Reporting Data Integrity <025-verifying-and-reporting-data-integrity>`
   * :doc:`Verifying Data-Cleaning Efforts <026-verifying-data-cleaning-efforts>`
   * :doc:`Viewing Data Differently for More Effective Data Cleaning <016-viewing-data-differently-for-more-effective-data-cleaning>`

.. toctree::
   :hidden:
   :maxdepth: 1

   001-the-importance-of-clean-data
   002-data-integrity-and-its-risks-in-data-analysis
   003-aligning-data-with-business-objectives
   004-handling-insufficient-data-in-data-analysis
   005-population-sample-size-and-random-sampling
   006-statistical-power-in-data-analysis
   007-sample-size-and-data-integrity
   008-margin-of-error
   009-dirty-data-vs-clean-data
   010-the-importance-of-clean-data-revisited
   011-common-issues-in-dirty-data
   012-data-cleaning-with-spreadsheets
   013-cleaning-and-merging-multiple-datasets
   014-spreadsheet-tools-for-data-cleaning
   015-using-spreadsheet-functions-for-data-cleaning
   016-viewing-data-differently-for-more-effective-data-cleaning
   017-data-mapping-and-the-big-picture-of-clean-data
   018-introduction-to-sql
   019-spreadsheets-vs-sql
   020-core-sql-queries-for-data-cleaning-and-analysis
   021-cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables
   022-using-cast-to-clean-and-format-data-in-sql
   023-advanced-sql-functions-for-data-cleaning
   024-coalesce
   025-verifying-and-reporting-data-integrity
   026-verifying-data-cleaning-efforts
   027-verification-techniques-using-spreadsheets-and-sql-to-catch-repeated-errors
   028-documenting-data-cleaning-changes
   029-reporting-data-cleaning-results
   030-using-feedback-from-data-cleaning-to-improve-data-quality
   031-refining-a-resume-for-data-analytics-roles
   032-exploring-data-analyst-job-opportunities

.. tags:: purpose: reference, topic: data analytics
