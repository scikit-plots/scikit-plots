.. _da-analyze-index:

:raw-html:`<div style="text-align:center"><strong>` 📊 Analyze Data
|br| |full_version| - |today|
:raw-html:`</strong></div>`

========================================================================
📊 Analyze Data
========================================================================

*Section 5 of the Data Analytics hub — 30 of 30 lessons.*

Organising, formatting, aggregating, and computing on data to surface patterns and answer the question.

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


.. dropdown:: 🗂️ Organizing & Formatting Data
   :class-container: sd-dropdown

   Sorting, filtering, formatting, validating, and string work in sheets and SQL.

   .. raw:: html

         <div class="da-row" data-k="Understanding Data Analysis what analysis actually is: the four phases that turn prepared data into insight"><a href="001-understanding-data-analysis.html">001 · Understanding Data Analysis — what analysis actually is: the four phases that turn prepared data into insight</a></div>
         <div class="da-row" data-k="Data Organization in Analysis arranging data so analysis is possible — the organising step before any computation"><a href="002-data-organization-in-analysis.html">002 · Data Organization in Analysis — arranging data so analysis is possible — the organising step before any computation</a></div>
         <div class="da-row" data-k="Sorting and Filtering in Data Analysis the two foundational moves of analysis, and how they differ from cleaning uses"><a href="003-sorting-and-filtering-in-data-analysis.html">003 · Sorting and Filtering in Data Analysis — the two foundational moves of analysis, and how they differ from cleaning uses</a></div>
         <div class="da-row" data-k="Sorting Data in Spreadsheets ordering rows by one or more columns to surface structure — done safely"><a href="004-sorting-data-in-spreadsheets.html">004 · Sorting Data in Spreadsheets — ordering rows by one or more columns to surface structure — done safely</a></div>
         <div class="da-row" data-k="Sorting and Filtering Data in SQL Using ORDER BY and WHERE the SQL twins of sort and filter: ORDER BY to order, WHERE to subset"><a href="005-sorting-and-filtering-data-in-sql-using-order-by-and-where.html">005 · Sorting and Filtering Data in SQL Using ORDER BY and WHERE — the SQL twins of sort and filter: ORDER BY to order, WHERE to subset</a></div>
         <div class="da-row" data-k="Data Formatting and Unit Conversion in Spreadsheets consistent formats and units — making numbers comparable before analysing them"><a href="006-data-formatting-and-unit-conversion-in-spreadsheets.html">006 · Data Formatting and Unit Conversion in Spreadsheets — consistent formats and units — making numbers comparable before analysing them</a></div>
         <div class="da-row" data-k="Data Validation in Spreadsheets rules that constrain what a cell may hold, catching and preventing bad data"><a href="007-data-validation-in-spreadsheets.html">007 · Data Validation in Spreadsheets — rules that constrain what a cell may hold, catching and preventing bad data</a></div>
         <div class="da-row" data-k="Combining Data Validation and Conditional Formatting in Spreadsheets validation to enforce rules, formatting to reveal violations — used together"><a href="008-combining-data-validation-and-conditional-formatting-in-spreadsheets.html">008 · Combining Data Validation and Conditional Formatting in Spreadsheets — validation to enforce rules, formatting to reveal violations — used together</a></div>
         <div class="da-row" data-k="Using CONCAT in SQL to Combine Text from Multiple Columns joining text from several columns into one — CONCAT and the concatenation operator"><a href="009-using-concat-in-sql-to-combine-text-from-multiple-columns.html">009 · Using CONCAT in SQL to Combine Text from Multiple Columns — joining text from several columns into one — CONCAT and the concatenation operator</a></div>
         <div class="da-row" data-k="Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND) the spreadsheet string toolkit for measuring, extracting, and locating text"><a href="010-working-with-strings-in-spreadsheets-len-left-right-find.html">010 · Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND) — the spreadsheet string toolkit for measuring, extracting, and locating text</a></div>

.. dropdown:: 🔗 Problem-Solving & Combining Data
   :class-container: sd-dropdown

   Getting unstuck, choosing tools, and joining data with VLOOKUP, JOIN, and subqueries.

   .. raw:: html

         <div class="da-row" data-k="Problem-Solving and Seeking Help in Data Analysis the analyst's debugging mindset, and how to ask for help effectively"><a href="011-problem-solving-and-seeking-help-in-data-analysis.html">011 · Problem-Solving and Seeking Help in Data Analysis — the analyst's debugging mindset, and how to ask for help effectively</a></div>
         <div class="da-row" data-k="How to Effectively Search for Solutions Online as a Data Analyst turning an error or a stuck point into a search that actually finds the answer"><a href="012-how-to-effectively-search-for-solutions-online-as-a-data-analyst.html">012 · How to Effectively Search for Solutions Online as a Data Analyst — turning an error or a stuck point into a search that actually finds the answer</a></div>
         <div class="da-row" data-k="Choosing the Right Tool in Data Analysis matching the task to spreadsheet, SQL, or programming — and combining them"><a href="013-choosing-the-right-tool-in-data-analysis.html">013 · Choosing the Right Tool in Data Analysis — matching the task to spreadsheet, SQL, or programming — and combining them</a></div>
         <div class="da-row" data-k="Preparing Data for VLOOKUP in Spreadsheets the setup VLOOKUP demands: a clean lookup key in the leftmost column"><a href="014-preparing-data-for-vlookup-in-spreadsheets.html">014 · Preparing Data for VLOOKUP in Spreadsheets — the setup VLOOKUP demands: a clean lookup key in the leftmost column</a></div>
         <div class="da-row" data-k="Using VLOOKUP to Combine Data Across Spreadsheets pulling matching values from another table by a shared key"><a href="015-using-vlookup-to-combine-data-across-spreadsheets.html">015 · Using VLOOKUP to Combine Data Across Spreadsheets — pulling matching values from another table by a shared key</a></div>
         <div class="da-row" data-k="Troubleshooting VLOOKUP and Building a Problem-Solving Framework why VLOOKUP fails, how to fix it, and a reusable debugging framework"><a href="016-troubleshooting-vlookup-and-building-a-problem-solving-framework.html">016 · Troubleshooting VLOOKUP and Building a Problem-Solving Framework — why VLOOKUP fails, how to fix it, and a reusable debugging framework</a></div>
         <div class="da-row" data-k="Using JOIN in SQL to Combine Tables combining rows from multiple tables on a matching key — SQL's core combine"><a href="017-using-join-in-sql-to-combine-tables.html">017 · Using JOIN in SQL to Combine Tables — combining rows from multiple tables on a matching key — SQL's core combine</a></div>
         <div class="da-row" data-k="Subqueries in SQL a query nested inside another — using one query's result within a second"><a href="018-subqueries-in-sql.html">018 · Subqueries in SQL — a query nested inside another — using one query's result within a second</a></div>
         <div class="da-row" data-k="Aggregating Data with Subqueries, HAVING, and CASE in SQL combining aggregation, group filtering, and conditional logic for real analysis"><a href="019-aggregating-data-with-subqueries-having-and-case-in-sql.html">019 · Aggregating Data with Subqueries, HAVING, and CASE in SQL — combining aggregation, group filtering, and conditional logic for real analysis</a></div>

.. dropdown:: 🧮 Calculations & Aggregation
   :class-container: sd-dropdown

   Formulas, conditional aggregation, pivot tables, and SQL GROUP BY calculations.

   .. raw:: html

         <div class="da-row" data-k="Using Spreadsheet Formulas for Sales Trend Analysis formulas for change over time — growth rates, running totals, period comparisons"><a href="020-using-spreadsheet-formulas-for-sales-trend-analysis.html">020 · Using Spreadsheet Formulas for Sales Trend Analysis — formulas for change over time — growth rates, running totals, period comparisons</a></div>
         <div class="da-row" data-k="Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets counting and summing only the rows that meet a condition"><a href="021-using-countif-and-sumif-for-conditional-aggregation-in-spreadsheets.html">021 · Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets — counting and summing only the rows that meet a condition</a></div>
         <div class="da-row" data-k="Using SUMPRODUCT for Advanced Spreadsheet Calculations multiplying and summing across arrays in one formula — weighted and multi-condition sums"><a href="022-using-sumproduct-for-advanced-spreadsheet-calculations.html">022 · Using SUMPRODUCT for Advanced Spreadsheet Calculations — multiplying and summing across arrays in one formula — weighted and multi-condition sums</a></div>
         <div class="da-row" data-k="Using Pivot Tables for Calculations and Trend Analysis the spreadsheet's most powerful summarising tool — grouping and aggregating with drags"><a href="023-using-pivot-tables-for-calculations-and-trend-analysis.html">023 · Using Pivot Tables for Calculations and Trend Analysis — the spreadsheet's most powerful summarising tool — grouping and aggregating with drags</a></div>
         <div class="da-row" data-k="Using Pivot Table Filters and Calculated Fields for Deeper Analysis filtering a pivot and adding computed fields — pushing pivots past basic summaries"><a href="024-using-pivot-table-filters-and-calculated-fields-for-deeper-analysis.html">024 · Using Pivot Table Filters and Calculated Fields for Deeper Analysis — filtering a pivot and adding computed fields — pushing pivots past basic summaries</a></div>
         <div class="da-row" data-k="Comparing Calculations in Spreadsheets and SQL the same aggregate two ways — when each tool suits the calculation"><a href="025-comparing-calculations-in-spreadsheets-and-sql.html">025 · Comparing Calculations in Spreadsheets and SQL — the same aggregate two ways — when each tool suits the calculation</a></div>
         <div class="da-row" data-k="Embedding Calculations in SQL Queries computing derived values inside a query — arithmetic and expressions in SELECT"><a href="026-embedding-calculations-in-sql-queries.html">026 · Embedding Calculations in SQL Queries — computing derived values inside a query — arithmetic and expressions in SELECT</a></div>
         <div class="da-row" data-k="Using GROUP BY and ORDER BY for Aggregated Calculations in SQL the SQL pivot: grouping to aggregate, ordering the summary"><a href="027-using-group-by-and-order-by-for-aggregated-calculations-in-sql.html">027 · Using GROUP BY and ORDER BY for Aggregated Calculations in SQL — the SQL pivot: grouping to aggregate, ordering the summary</a></div>

.. dropdown:: 🚀 Validation & Temporary Tables
   :class-container: sd-dropdown

   Ongoing validation and temporary tables with WITH for cleaner SQL workflows.

   .. raw:: html

         <div class="da-row" data-k="Data Validation as an Ongoing Analytical Process validation not as a one-time gate but a continuous check throughout analysis"><a href="028-data-validation-as-an-ongoing-analytical-process.html">028 · Data Validation as an Ongoing Analytical Process — validation not as a one-time gate but a continuous check throughout analysis</a></div>
         <div class="da-row" data-k="Temporary Tables and the WITH Clause in SQL holding intermediate results — CTEs and temp tables that structure complex queries"><a href="029-temporary-tables-and-the-with-clause-in-sql.html">029 · Temporary Tables and the WITH Clause in SQL — holding intermediate results — CTEs and temp tables that structure complex queries</a></div>
         <div class="da-row" data-k="Creating Temporary Tables in SQL — Methods, Trade-offs, and Best Practices the ways to make a temp table, when each fits, and how to use them well"><a href="030-creating-temporary-tables-in-sql-methods-trade-offs-and-best-practices.html">030 · Creating Temporary Tables in SQL — Methods, Trade-offs, and Best Practices — the ways to make a temp table, when each fits, and how to use them well</a></div>

🔤 Every lesson, A–Z
---------------------

.. dropdown:: 🔠 A–Z index
   :class-container: term-az

   .. raw:: html

         <div class="da-row" data-k="Aggregating Data with Subqueries, HAVING, and CASE in SQL"><a href="019-aggregating-data-with-subqueries-having-and-case-in-sql.html">Aggregating Data with Subqueries, HAVING, and CASE in SQL</a></div>
         <div class="da-row" data-k="Choosing the Right Tool in Data Analysis"><a href="013-choosing-the-right-tool-in-data-analysis.html">Choosing the Right Tool in Data Analysis</a></div>
         <div class="da-row" data-k="Combining Data Validation and Conditional Formatting in Spreadsheets"><a href="008-combining-data-validation-and-conditional-formatting-in-spreadsheets.html">Combining Data Validation and Conditional Formatting in Spreadsheets</a></div>
         <div class="da-row" data-k="Comparing Calculations in Spreadsheets and SQL"><a href="025-comparing-calculations-in-spreadsheets-and-sql.html">Comparing Calculations in Spreadsheets and SQL</a></div>
         <div class="da-row" data-k="Creating Temporary Tables in SQL — Methods, Trade-offs, and Best Practices"><a href="030-creating-temporary-tables-in-sql-methods-trade-offs-and-best-practices.html">Creating Temporary Tables in SQL — Methods, Trade-offs, and Best Practices</a></div>
         <div class="da-row" data-k="Data Formatting and Unit Conversion in Spreadsheets"><a href="006-data-formatting-and-unit-conversion-in-spreadsheets.html">Data Formatting and Unit Conversion in Spreadsheets</a></div>
         <div class="da-row" data-k="Data Organization in Analysis"><a href="002-data-organization-in-analysis.html">Data Organization in Analysis</a></div>
         <div class="da-row" data-k="Data Validation as an Ongoing Analytical Process"><a href="028-data-validation-as-an-ongoing-analytical-process.html">Data Validation as an Ongoing Analytical Process</a></div>
         <div class="da-row" data-k="Data Validation in Spreadsheets"><a href="007-data-validation-in-spreadsheets.html">Data Validation in Spreadsheets</a></div>
         <div class="da-row" data-k="Embedding Calculations in SQL Queries"><a href="026-embedding-calculations-in-sql-queries.html">Embedding Calculations in SQL Queries</a></div>
         <div class="da-row" data-k="How to Effectively Search for Solutions Online as a Data Analyst"><a href="012-how-to-effectively-search-for-solutions-online-as-a-data-analyst.html">How to Effectively Search for Solutions Online as a Data Analyst</a></div>
         <div class="da-row" data-k="Preparing Data for VLOOKUP in Spreadsheets"><a href="014-preparing-data-for-vlookup-in-spreadsheets.html">Preparing Data for VLOOKUP in Spreadsheets</a></div>
         <div class="da-row" data-k="Problem-Solving and Seeking Help in Data Analysis"><a href="011-problem-solving-and-seeking-help-in-data-analysis.html">Problem-Solving and Seeking Help in Data Analysis</a></div>
         <div class="da-row" data-k="Sorting and Filtering Data in SQL Using ORDER BY and WHERE"><a href="005-sorting-and-filtering-data-in-sql-using-order-by-and-where.html">Sorting and Filtering Data in SQL Using ORDER BY and WHERE</a></div>
         <div class="da-row" data-k="Sorting and Filtering in Data Analysis"><a href="003-sorting-and-filtering-in-data-analysis.html">Sorting and Filtering in Data Analysis</a></div>
         <div class="da-row" data-k="Sorting Data in Spreadsheets"><a href="004-sorting-data-in-spreadsheets.html">Sorting Data in Spreadsheets</a></div>
         <div class="da-row" data-k="Subqueries in SQL"><a href="018-subqueries-in-sql.html">Subqueries in SQL</a></div>
         <div class="da-row" data-k="Temporary Tables and the WITH Clause in SQL"><a href="029-temporary-tables-and-the-with-clause-in-sql.html">Temporary Tables and the WITH Clause in SQL</a></div>
         <div class="da-row" data-k="Troubleshooting VLOOKUP and Building a Problem-Solving Framework"><a href="016-troubleshooting-vlookup-and-building-a-problem-solving-framework.html">Troubleshooting VLOOKUP and Building a Problem-Solving Framework</a></div>
         <div class="da-row" data-k="Understanding Data Analysis"><a href="001-understanding-data-analysis.html">Understanding Data Analysis</a></div>
         <div class="da-row" data-k="Using CONCAT in SQL to Combine Text from Multiple Columns"><a href="009-using-concat-in-sql-to-combine-text-from-multiple-columns.html">Using CONCAT in SQL to Combine Text from Multiple Columns</a></div>
         <div class="da-row" data-k="Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets"><a href="021-using-countif-and-sumif-for-conditional-aggregation-in-spreadsheets.html">Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets</a></div>
         <div class="da-row" data-k="Using GROUP BY and ORDER BY for Aggregated Calculations in SQL"><a href="027-using-group-by-and-order-by-for-aggregated-calculations-in-sql.html">Using GROUP BY and ORDER BY for Aggregated Calculations in SQL</a></div>
         <div class="da-row" data-k="Using JOIN in SQL to Combine Tables"><a href="017-using-join-in-sql-to-combine-tables.html">Using JOIN in SQL to Combine Tables</a></div>
         <div class="da-row" data-k="Using Pivot Table Filters and Calculated Fields for Deeper Analysis"><a href="024-using-pivot-table-filters-and-calculated-fields-for-deeper-analysis.html">Using Pivot Table Filters and Calculated Fields for Deeper Analysis</a></div>
         <div class="da-row" data-k="Using Pivot Tables for Calculations and Trend Analysis"><a href="023-using-pivot-tables-for-calculations-and-trend-analysis.html">Using Pivot Tables for Calculations and Trend Analysis</a></div>
         <div class="da-row" data-k="Using Spreadsheet Formulas for Sales Trend Analysis"><a href="020-using-spreadsheet-formulas-for-sales-trend-analysis.html">Using Spreadsheet Formulas for Sales Trend Analysis</a></div>
         <div class="da-row" data-k="Using SUMPRODUCT for Advanced Spreadsheet Calculations"><a href="022-using-sumproduct-for-advanced-spreadsheet-calculations.html">Using SUMPRODUCT for Advanced Spreadsheet Calculations</a></div>
         <div class="da-row" data-k="Using VLOOKUP to Combine Data Across Spreadsheets"><a href="015-using-vlookup-to-combine-data-across-spreadsheets.html">Using VLOOKUP to Combine Data Across Spreadsheets</a></div>
         <div class="da-row" data-k="Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)"><a href="010-working-with-strings-in-spreadsheets-len-left-right-find.html">Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)</a></div>
