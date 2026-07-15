:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

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


.. dropdown:: 🗂️ Organizing & Formatting Data
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Sorting, filtering, formatting, validating, and string work in sheets and SQL.

   * :doc:`001 · Understanding Data Analysis — what analysis actually is: the four phases that turn prepared data into insight <001-understanding-data-analysis>`
   * :doc:`002 · Data Organization in Analysis — arranging data so analysis is possible — the organising step before any computation <002-data-organization-in-analysis>`
   * :doc:`003 · Sorting and Filtering in Data Analysis — the two foundational moves of analysis, and how they differ from cleaning uses <003-sorting-and-filtering-in-data-analysis>`
   * :doc:`004 · Sorting Data in Spreadsheets — ordering rows by one or more columns to surface structure — done safely <004-sorting-data-in-spreadsheets>`
   * :doc:`005 · Sorting and Filtering Data in SQL Using ORDER BY and WHERE — the SQL twins of sort and filter: ORDER BY to order, WHERE to subset <005-sorting-and-filtering-data-in-sql-using-order-by-and-where>`
   * :doc:`006 · Data Formatting and Unit Conversion in Spreadsheets — consistent formats and units — making numbers comparable before analysing them <006-data-formatting-and-unit-conversion-in-spreadsheets>`
   * :doc:`007 · Data Validation in Spreadsheets — rules that constrain what a cell may hold, catching and preventing bad data <007-data-validation-in-spreadsheets>`
   * :doc:`008 · Combining Data Validation and Conditional Formatting in Spreadsheets — validation to enforce rules, formatting to reveal violations — used together <008-combining-data-validation-and-conditional-formatting-in-spreadsheets>`
   * :doc:`009 · Using CONCAT in SQL to Combine Text from Multiple Columns — joining text from several columns into one — CONCAT and the concatenation operator <009-using-concat-in-sql-to-combine-text-from-multiple-columns>`
   * :doc:`010 · Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND) — the spreadsheet string toolkit for measuring, extracting, and locating text <010-working-with-strings-in-spreadsheets-len-left-right-find>`

.. dropdown:: 🔗 Problem-Solving & Combining Data
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Getting unstuck, choosing tools, and joining data with VLOOKUP, JOIN, and subqueries.

   * :doc:`011 · Problem-Solving and Seeking Help in Data Analysis — the analyst's debugging mindset, and how to ask for help effectively <011-problem-solving-and-seeking-help-in-data-analysis>`
   * :doc:`012 · How to Effectively Search for Solutions Online as a Data Analyst — turning an error or a stuck point into a search that actually finds the answer <012-how-to-effectively-search-for-solutions-online-as-a-data-analyst>`
   * :doc:`013 · Choosing the Right Tool in Data Analysis — matching the task to spreadsheet, SQL, or programming — and combining them <013-choosing-the-right-tool-in-data-analysis>`
   * :doc:`014 · Preparing Data for VLOOKUP in Spreadsheets — the setup VLOOKUP demands: a clean lookup key in the leftmost column <014-preparing-data-for-vlookup-in-spreadsheets>`
   * :doc:`015 · Using VLOOKUP to Combine Data Across Spreadsheets — pulling matching values from another table by a shared key <015-using-vlookup-to-combine-data-across-spreadsheets>`
   * :doc:`016 · Troubleshooting VLOOKUP and Building a Problem-Solving Framework — why VLOOKUP fails, how to fix it, and a reusable debugging framework <016-troubleshooting-vlookup-and-building-a-problem-solving-framework>`
   * :doc:`017 · Using JOIN in SQL to Combine Tables — combining rows from multiple tables on a matching key — SQL's core combine <017-using-join-in-sql-to-combine-tables>`
   * :doc:`018 · Subqueries in SQL — a query nested inside another — using one query's result within a second <018-subqueries-in-sql>`
   * :doc:`019 · Aggregating Data with Subqueries, HAVING, and CASE in SQL — combining aggregation, group filtering, and conditional logic for real analysis <019-aggregating-data-with-subqueries-having-and-case-in-sql>`

.. dropdown:: 🧮 Calculations & Aggregation
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Formulas, conditional aggregation, pivot tables, and SQL GROUP BY calculations.

   * :doc:`020 · Using Spreadsheet Formulas for Sales Trend Analysis — formulas for change over time — growth rates, running totals, period comparisons <020-using-spreadsheet-formulas-for-sales-trend-analysis>`
   * :doc:`021 · Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets — counting and summing only the rows that meet a condition <021-using-countif-and-sumif-for-conditional-aggregation-in-spreadsheets>`
   * :doc:`022 · Using SUMPRODUCT for Advanced Spreadsheet Calculations — multiplying and summing across arrays in one formula — weighted and multi-condition sums <022-using-sumproduct-for-advanced-spreadsheet-calculations>`
   * :doc:`023 · Using Pivot Tables for Calculations and Trend Analysis — the spreadsheet's most powerful summarising tool — grouping and aggregating with drags <023-using-pivot-tables-for-calculations-and-trend-analysis>`
   * :doc:`024 · Using Pivot Table Filters and Calculated Fields for Deeper Analysis — filtering a pivot and adding computed fields — pushing pivots past basic summaries <024-using-pivot-table-filters-and-calculated-fields-for-deeper-analysis>`
   * :doc:`025 · Comparing Calculations in Spreadsheets and SQL — the same aggregate two ways — when each tool suits the calculation <025-comparing-calculations-in-spreadsheets-and-sql>`
   * :doc:`026 · Embedding Calculations in SQL Queries — computing derived values inside a query — arithmetic and expressions in SELECT <026-embedding-calculations-in-sql-queries>`
   * :doc:`027 · Using GROUP BY and ORDER BY for Aggregated Calculations in SQL — the SQL pivot: grouping to aggregate, ordering the summary <027-using-group-by-and-order-by-for-aggregated-calculations-in-sql>`

.. dropdown:: 🚀 Validation & Temporary Tables
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Ongoing validation and temporary tables with WITH for cleaner SQL workflows.

   * :doc:`028 · Data Validation as an Ongoing Analytical Process — validation not as a one-time gate but a continuous check throughout analysis <028-data-validation-as-an-ongoing-analytical-process>`
   * :doc:`029 · Temporary Tables and the WITH Clause in SQL — holding intermediate results — CTEs and temp tables that structure complex queries <029-temporary-tables-and-the-with-clause-in-sql>`
   * :doc:`030 · Creating Temporary Tables in SQL — Methods, Trade-offs, and Best Practices — the ways to make a temp table, when each fits, and how to use them well <030-creating-temporary-tables-in-sql-methods-trade-offs-and-best-practices>`

🔤 Every lesson, A–Z index
---------------------------

.. dropdown:: 🔠 Open the full alphabetical index
   :animate: fade-in-slide-down
   :class-container: term-az

   .. hlist::
      :columns: 2

      * :doc:`Aggregating Data with Subqueries, HAVING, and CASE in SQL <019-aggregating-data-with-subqueries-having-and-case-in-sql>`
      * :doc:`Choosing the Right Tool in Data Analysis <013-choosing-the-right-tool-in-data-analysis>`
      * :doc:`Combining Data Validation and Conditional Formatting in Spreadsheets <008-combining-data-validation-and-conditional-formatting-in-spreadsheets>`
      * :doc:`Comparing Calculations in Spreadsheets and SQL <025-comparing-calculations-in-spreadsheets-and-sql>`
      * :doc:`Creating Temporary Tables in SQL — Methods, Trade-offs, and Best Practices <030-creating-temporary-tables-in-sql-methods-trade-offs-and-best-practices>`
      * :doc:`Data Formatting and Unit Conversion in Spreadsheets <006-data-formatting-and-unit-conversion-in-spreadsheets>`
      * :doc:`Data Organization in Analysis <002-data-organization-in-analysis>`
      * :doc:`Data Validation as an Ongoing Analytical Process <028-data-validation-as-an-ongoing-analytical-process>`
      * :doc:`Data Validation in Spreadsheets <007-data-validation-in-spreadsheets>`
      * :doc:`Embedding Calculations in SQL Queries <026-embedding-calculations-in-sql-queries>`
      * :doc:`How to Effectively Search for Solutions Online as a Data Analyst <012-how-to-effectively-search-for-solutions-online-as-a-data-analyst>`
      * :doc:`Preparing Data for VLOOKUP in Spreadsheets <014-preparing-data-for-vlookup-in-spreadsheets>`
      * :doc:`Problem-Solving and Seeking Help in Data Analysis <011-problem-solving-and-seeking-help-in-data-analysis>`
      * :doc:`Sorting and Filtering Data in SQL Using ORDER BY and WHERE <005-sorting-and-filtering-data-in-sql-using-order-by-and-where>`
      * :doc:`Sorting and Filtering in Data Analysis <003-sorting-and-filtering-in-data-analysis>`
      * :doc:`Sorting Data in Spreadsheets <004-sorting-data-in-spreadsheets>`
      * :doc:`Subqueries in SQL <018-subqueries-in-sql>`
      * :doc:`Temporary Tables and the WITH Clause in SQL <029-temporary-tables-and-the-with-clause-in-sql>`
      * :doc:`Troubleshooting VLOOKUP and Building a Problem-Solving Framework <016-troubleshooting-vlookup-and-building-a-problem-solving-framework>`
      * :doc:`Understanding Data Analysis <001-understanding-data-analysis>`
      * :doc:`Using CONCAT in SQL to Combine Text from Multiple Columns <009-using-concat-in-sql-to-combine-text-from-multiple-columns>`
      * :doc:`Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets <021-using-countif-and-sumif-for-conditional-aggregation-in-spreadsheets>`
      * :doc:`Using GROUP BY and ORDER BY for Aggregated Calculations in SQL <027-using-group-by-and-order-by-for-aggregated-calculations-in-sql>`
      * :doc:`Using JOIN in SQL to Combine Tables <017-using-join-in-sql-to-combine-tables>`
      * :doc:`Using Pivot Table Filters and Calculated Fields for Deeper Analysis <024-using-pivot-table-filters-and-calculated-fields-for-deeper-analysis>`
      * :doc:`Using Pivot Tables for Calculations and Trend Analysis <023-using-pivot-tables-for-calculations-and-trend-analysis>`
      * :doc:`Using Spreadsheet Formulas for Sales Trend Analysis <020-using-spreadsheet-formulas-for-sales-trend-analysis>`
      * :doc:`Using SUMPRODUCT for Advanced Spreadsheet Calculations <022-using-sumproduct-for-advanced-spreadsheet-calculations>`
      * :doc:`Using VLOOKUP to Combine Data Across Spreadsheets <015-using-vlookup-to-combine-data-across-spreadsheets>`
      * :doc:`Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND) <010-working-with-strings-in-spreadsheets-len-left-right-find>`

.. toctree::
   :hidden:
   :includehidden:
   :maxdepth: 1

   001-understanding-data-analysis
   002-data-organization-in-analysis
   003-sorting-and-filtering-in-data-analysis
   004-sorting-data-in-spreadsheets
   005-sorting-and-filtering-data-in-sql-using-order-by-and-where
   006-data-formatting-and-unit-conversion-in-spreadsheets
   007-data-validation-in-spreadsheets
   008-combining-data-validation-and-conditional-formatting-in-spreadsheets
   009-using-concat-in-sql-to-combine-text-from-multiple-columns
   010-working-with-strings-in-spreadsheets-len-left-right-find
   011-problem-solving-and-seeking-help-in-data-analysis
   012-how-to-effectively-search-for-solutions-online-as-a-data-analyst
   013-choosing-the-right-tool-in-data-analysis
   014-preparing-data-for-vlookup-in-spreadsheets
   015-using-vlookup-to-combine-data-across-spreadsheets
   016-troubleshooting-vlookup-and-building-a-problem-solving-framework
   017-using-join-in-sql-to-combine-tables
   018-subqueries-in-sql
   019-aggregating-data-with-subqueries-having-and-case-in-sql
   020-using-spreadsheet-formulas-for-sales-trend-analysis
   021-using-countif-and-sumif-for-conditional-aggregation-in-spreadsheets
   022-using-sumproduct-for-advanced-spreadsheet-calculations
   023-using-pivot-tables-for-calculations-and-trend-analysis
   024-using-pivot-table-filters-and-calculated-fields-for-deeper-analysis
   025-comparing-calculations-in-spreadsheets-and-sql
   026-embedding-calculations-in-sql-queries
   027-using-group-by-and-order-by-for-aggregated-calculations-in-sql
   028-data-validation-as-an-ongoing-analytical-process
   029-temporary-tables-and-the-with-clause-in-sql
   030-creating-temporary-tables-in-sql-methods-trade-offs-and-best-practices

.. tags:: purpose: reference, topic: data analytics
