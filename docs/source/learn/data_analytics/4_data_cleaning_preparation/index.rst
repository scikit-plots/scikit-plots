.. _da-cleaning-index:

========================================================================
🧽 Data Cleaning & Preparation
========================================================================

*Section 4 of the Data Analytics hub — 32 of 32 lessons.*

Finding and fixing dirty data: missing values, duplicates, outliers, and validation for trustworthy inputs.

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


.. dropdown:: 🧱 Data Integrity & Sampling
   :class-container: sd-dropdown

   Why clean data matters, integrity risks, sampling, power, and margin of error.

   .. raw:: html

         <div class="da-row" data-k="The Importance of Clean Data "><a href="001-the-importance-of-clean-data.html">001 · The Importance of Clean Data</a></div>
         <div class="da-row" data-k="Data Integrity and Its Risks in Data Analysis "><a href="002-data-integrity-and-its-risks-in-data-analysis.html">002 · Data Integrity and Its Risks in Data Analysis</a></div>
         <div class="da-row" data-k="Aligning Data with Business Objectives "><a href="003-aligning-data-with-business-objectives.html">003 · Aligning Data with Business Objectives</a></div>
         <div class="da-row" data-k="Handling Insufficient Data in Data Analysis "><a href="004-handling-insufficient-data-in-data-analysis.html">004 · Handling Insufficient Data in Data Analysis</a></div>
         <div class="da-row" data-k="Population, Sample Size, and Random Sampling "><a href="005-population-sample-size-and-random-sampling.html">005 · Population, Sample Size, and Random Sampling</a></div>
         <div class="da-row" data-k="Statistical Power in Data Analysis "><a href="006-statistical-power-in-data-analysis.html">006 · Statistical Power in Data Analysis</a></div>
         <div class="da-row" data-k="Sample Size and Data Integrity "><a href="007-sample-size-and-data-integrity.html">007 · Sample Size and Data Integrity</a></div>
         <div class="da-row" data-k="Margin of Error "><a href="008-margin-of-error.html">008 · Margin of Error</a></div>

.. dropdown:: 🧹 Dirty Data & Spreadsheet Cleaning
   :class-container: sd-dropdown

   Recognising dirty data and cleaning it with spreadsheet tools and functions.

   .. raw:: html

         <div class="da-row" data-k="Dirty Data vs. Clean Data "><a href="009-dirty-data-vs-clean-data.html">009 · Dirty Data vs. Clean Data</a></div>
         <div class="da-row" data-k="The Importance of Clean Data (revisited) "><a href="010-the-importance-of-clean-data-revisited.html">010 · The Importance of Clean Data (revisited)</a></div>
         <div class="da-row" data-k="Common Issues in Dirty Data "><a href="011-common-issues-in-dirty-data.html">011 · Common Issues in Dirty Data</a></div>
         <div class="da-row" data-k="Data Cleaning with Spreadsheets "><a href="012-data-cleaning-with-spreadsheets.html">012 · Data Cleaning with Spreadsheets</a></div>
         <div class="da-row" data-k="Cleaning and Merging Multiple Datasets "><a href="013-cleaning-and-merging-multiple-datasets.html">013 · Cleaning and Merging Multiple Datasets</a></div>
         <div class="da-row" data-k="Spreadsheet Tools for Data Cleaning "><a href="014-spreadsheet-tools-for-data-cleaning.html">014 · Spreadsheet Tools for Data Cleaning</a></div>
         <div class="da-row" data-k="Using Spreadsheet Functions for Data Cleaning "><a href="015-using-spreadsheet-functions-for-data-cleaning.html">015 · Using Spreadsheet Functions for Data Cleaning</a></div>
         <div class="da-row" data-k="Viewing Data Differently for More Effective Data Cleaning "><a href="016-viewing-data-differently-for-more-effective-data-cleaning.html">016 · Viewing Data Differently for More Effective Data Cleaning</a></div>
         <div class="da-row" data-k="Data Mapping and the Big Picture of Clean Data "><a href="017-data-mapping-and-the-big-picture-of-clean-data.html">017 · Data Mapping and the Big Picture of Clean Data</a></div>

.. dropdown:: 🐬 Cleaning with SQL
   :class-container: sd-dropdown

   SQL from first queries to CAST, COALESCE, and advanced cleaning functions.

   .. raw:: html

         <div class="da-row" data-k="Introduction to SQL "><a href="018-introduction-to-sql.html">018 · Introduction to SQL</a></div>
         <div class="da-row" data-k="Spreadsheets vs. SQL "><a href="019-spreadsheets-vs-sql.html">019 · Spreadsheets vs. SQL</a></div>
         <div class="da-row" data-k="Core SQL Queries for Data Cleaning and Analysis "><a href="020-core-sql-queries-for-data-cleaning-and-analysis.html">020 · Core SQL Queries for Data Cleaning and Analysis</a></div>
         <div class="da-row" data-k="Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables "><a href="021-cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables.html">021 · Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables</a></div>
         <div class="da-row" data-k="Using CAST to Clean and Format Data in SQL "><a href="022-using-cast-to-clean-and-format-data-in-sql.html">022 · Using CAST to Clean and Format Data in SQL</a></div>
         <div class="da-row" data-k="Advanced SQL Functions for Data Cleaning "><a href="023-advanced-sql-functions-for-data-cleaning.html">023 · Advanced SQL Functions for Data Cleaning</a></div>
         <div class="da-row" data-k="COALESCE "><a href="024-coalesce.html">024 · COALESCE</a></div>

.. dropdown:: ✅ Verification, Documentation & Next Steps
   :class-container: sd-dropdown

   Verifying and reporting cleaning work, documenting changes, and moving forward.

   .. raw:: html

         <div class="da-row" data-k="Verifying and Reporting Data Integrity "><a href="025-verifying-and-reporting-data-integrity.html">025 · Verifying and Reporting Data Integrity</a></div>
         <div class="da-row" data-k="Verifying Data-Cleaning Efforts "><a href="026-verifying-data-cleaning-efforts.html">026 · Verifying Data-Cleaning Efforts</a></div>
         <div class="da-row" data-k="Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors "><a href="027-verification-techniques-using-spreadsheets-and-sql-to-catch-repeated-errors.html">027 · Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors</a></div>
         <div class="da-row" data-k="Documenting Data-Cleaning Changes "><a href="028-documenting-data-cleaning-changes.html">028 · Documenting Data-Cleaning Changes</a></div>
         <div class="da-row" data-k="Reporting Data-Cleaning Results "><a href="029-reporting-data-cleaning-results.html">029 · Reporting Data-Cleaning Results</a></div>
         <div class="da-row" data-k="Using Feedback from Data Cleaning to Improve Data Quality "><a href="030-using-feedback-from-data-cleaning-to-improve-data-quality.html">030 · Using Feedback from Data Cleaning to Improve Data Quality</a></div>
         <div class="da-row" data-k="Refining a Resume for Data Analytics Roles "><a href="031-refining-a-resume-for-data-analytics-roles.html">031 · Refining a Resume for Data Analytics Roles</a></div>
         <div class="da-row" data-k="Exploring Data Analyst Job Opportunities "><a href="032-exploring-data-analyst-job-opportunities.html">032 · Exploring Data Analyst Job Opportunities</a></div>

.. dropdown:: 🔠 A–Z index
   :class-container: term-az

   .. raw:: html

         <div class="da-row" data-k="Advanced SQL Functions for Data Cleaning"><a href="023-advanced-sql-functions-for-data-cleaning.html">Advanced SQL Functions for Data Cleaning</a></div>
         <div class="da-row" data-k="Aligning Data with Business Objectives"><a href="003-aligning-data-with-business-objectives.html">Aligning Data with Business Objectives</a></div>
         <div class="da-row" data-k="Cleaning and Merging Multiple Datasets"><a href="013-cleaning-and-merging-multiple-datasets.html">Cleaning and Merging Multiple Datasets</a></div>
         <div class="da-row" data-k="Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables"><a href="021-cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables.html">Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables</a></div>
         <div class="da-row" data-k="COALESCE"><a href="024-coalesce.html">COALESCE</a></div>
         <div class="da-row" data-k="Common Issues in Dirty Data"><a href="011-common-issues-in-dirty-data.html">Common Issues in Dirty Data</a></div>
         <div class="da-row" data-k="Core SQL Queries for Data Cleaning and Analysis"><a href="020-core-sql-queries-for-data-cleaning-and-analysis.html">Core SQL Queries for Data Cleaning and Analysis</a></div>
         <div class="da-row" data-k="Data Cleaning with Spreadsheets"><a href="012-data-cleaning-with-spreadsheets.html">Data Cleaning with Spreadsheets</a></div>
         <div class="da-row" data-k="Data Integrity and Its Risks in Data Analysis"><a href="002-data-integrity-and-its-risks-in-data-analysis.html">Data Integrity and Its Risks in Data Analysis</a></div>
         <div class="da-row" data-k="Data Mapping and the Big Picture of Clean Data"><a href="017-data-mapping-and-the-big-picture-of-clean-data.html">Data Mapping and the Big Picture of Clean Data</a></div>
         <div class="da-row" data-k="Dirty Data vs. Clean Data"><a href="009-dirty-data-vs-clean-data.html">Dirty Data vs. Clean Data</a></div>
         <div class="da-row" data-k="Documenting Data-Cleaning Changes"><a href="028-documenting-data-cleaning-changes.html">Documenting Data-Cleaning Changes</a></div>
         <div class="da-row" data-k="Exploring Data Analyst Job Opportunities"><a href="032-exploring-data-analyst-job-opportunities.html">Exploring Data Analyst Job Opportunities</a></div>
         <div class="da-row" data-k="Handling Insufficient Data in Data Analysis"><a href="004-handling-insufficient-data-in-data-analysis.html">Handling Insufficient Data in Data Analysis</a></div>
         <div class="da-row" data-k="Introduction to SQL"><a href="018-introduction-to-sql.html">Introduction to SQL</a></div>
         <div class="da-row" data-k="Margin of Error"><a href="008-margin-of-error.html">Margin of Error</a></div>
         <div class="da-row" data-k="Population, Sample Size, and Random Sampling"><a href="005-population-sample-size-and-random-sampling.html">Population, Sample Size, and Random Sampling</a></div>
         <div class="da-row" data-k="Refining a Resume for Data Analytics Roles"><a href="031-refining-a-resume-for-data-analytics-roles.html">Refining a Resume for Data Analytics Roles</a></div>
         <div class="da-row" data-k="Reporting Data-Cleaning Results"><a href="029-reporting-data-cleaning-results.html">Reporting Data-Cleaning Results</a></div>
         <div class="da-row" data-k="Sample Size and Data Integrity"><a href="007-sample-size-and-data-integrity.html">Sample Size and Data Integrity</a></div>
         <div class="da-row" data-k="Spreadsheet Tools for Data Cleaning"><a href="014-spreadsheet-tools-for-data-cleaning.html">Spreadsheet Tools for Data Cleaning</a></div>
         <div class="da-row" data-k="Spreadsheets vs. SQL"><a href="019-spreadsheets-vs-sql.html">Spreadsheets vs. SQL</a></div>
         <div class="da-row" data-k="Statistical Power in Data Analysis"><a href="006-statistical-power-in-data-analysis.html">Statistical Power in Data Analysis</a></div>
         <div class="da-row" data-k="The Importance of Clean Data"><a href="001-the-importance-of-clean-data.html">The Importance of Clean Data</a></div>
         <div class="da-row" data-k="The Importance of Clean Data (revisited)"><a href="010-the-importance-of-clean-data-revisited.html">The Importance of Clean Data (revisited)</a></div>
         <div class="da-row" data-k="Using CAST to Clean and Format Data in SQL"><a href="022-using-cast-to-clean-and-format-data-in-sql.html">Using CAST to Clean and Format Data in SQL</a></div>
         <div class="da-row" data-k="Using Feedback from Data Cleaning to Improve Data Quality"><a href="030-using-feedback-from-data-cleaning-to-improve-data-quality.html">Using Feedback from Data Cleaning to Improve Data Quality</a></div>
         <div class="da-row" data-k="Using Spreadsheet Functions for Data Cleaning"><a href="015-using-spreadsheet-functions-for-data-cleaning.html">Using Spreadsheet Functions for Data Cleaning</a></div>
         <div class="da-row" data-k="Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors"><a href="027-verification-techniques-using-spreadsheets-and-sql-to-catch-repeated-errors.html">Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors</a></div>
         <div class="da-row" data-k="Verifying and Reporting Data Integrity"><a href="025-verifying-and-reporting-data-integrity.html">Verifying and Reporting Data Integrity</a></div>
         <div class="da-row" data-k="Verifying Data-Cleaning Efforts"><a href="026-verifying-data-cleaning-efforts.html">Verifying Data-Cleaning Efforts</a></div>
         <div class="da-row" data-k="Viewing Data Differently for More Effective Data Cleaning"><a href="016-viewing-data-differently-for-more-effective-data-cleaning.html">Viewing Data Differently for More Effective Data Cleaning</a></div>
