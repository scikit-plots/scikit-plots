.. _da-prep-index:

========================================================================
📦 Data Preparation
========================================================================

*Section 3 of the Data Analytics hub — 25 of 25 lessons.*

Sourcing, structuring, and organising data before analysis: types, formats, databases, and sampling.

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


.. dropdown:: 🧬 Data Types & Structure
   :class-container: sd-dropdown

   How data is generated, its types and formats, and structured vs tabular shapes.

   .. raw:: html

         <div class="da-row" data-k="How Data Is Generated and Collected "><a href="001-how-data-is-generated-and-collected.html">001 · How Data Is Generated and Collected</a></div>
         <div class="da-row" data-k="Choosing the Right Data to Collect "><a href="002-choosing-the-right-data-to-collect.html">002 · Choosing the Right Data to Collect</a></div>
         <div class="da-row" data-k="Understanding Data Types and Data Formats "><a href="003-understanding-data-types-and-data-formats.html">003 · Understanding Data Types and Data Formats</a></div>
         <div class="da-row" data-k="Structured Data and Data Models "><a href="004-structured-data-and-data-models.html">004 · Structured Data and Data Models</a></div>
         <div class="da-row" data-k="Data Types in Spreadsheets "><a href="005-data-types-in-spreadsheets.html">005 · Data Types in Spreadsheets</a></div>
         <div class="da-row" data-k="Data Tables (Tabular Data) "><a href="006-data-tables-tabular-data.html">006 · Data Tables (Tabular Data)</a></div>
         <div class="da-row" data-k="Wide Data vs. Long Data "><a href="007-wide-data-vs-long-data.html">007 · Wide Data vs. Long Data</a></div>

.. dropdown:: ⚖️ Bias & Data Ethics
   :class-container: sd-dropdown

   Recognising bias, judging sources with ROCCC, and the ethics of data use.

   .. raw:: html

         <div class="da-row" data-k="Understanding Bias in Data Analysis "><a href="008-understanding-bias-in-data-analysis.html">008 · Understanding Bias in Data Analysis</a></div>
         <div class="da-row" data-k="Sampling Bias and Unbiased Data "><a href="009-sampling-bias-and-unbiased-data.html">009 · Sampling Bias and Unbiased Data</a></div>
         <div class="da-row" data-k="Common Types of Data Bias "><a href="010-common-types-of-data-bias.html">010 · Common Types of Data Bias</a></div>
         <div class="da-row" data-k="Identifying Good Data Sources (ROCCC Framework) "><a href="011-identifying-good-data-sources-roccc-framework.html">011 · Identifying Good Data Sources (ROCCC Framework)</a></div>
         <div class="da-row" data-k="Identifying Bad Data Sources (When Data Does Not ROCCC) "><a href="012-identifying-bad-data-sources-when-data-does-not-roccc.html">012 · Identifying Bad Data Sources (When Data Does Not ROCCC)</a></div>
         <div class="da-row" data-k="Data Ethics in Data Analysis "><a href="013-data-ethics-in-data-analysis.html">013 · Data Ethics in Data Analysis</a></div>
         <div class="da-row" data-k="Data Privacy in Data Ethics "><a href="014-data-privacy-in-data-ethics.html">014 · Data Privacy in Data Ethics</a></div>
         <div class="da-row" data-k="Open Data and Openness in Data Ethics "><a href="015-open-data-and-openness-in-data-ethics.html">015 · Open Data and Openness in Data Ethics</a></div>

.. dropdown:: 🗄️ Databases & Data Sources
   :class-container: sd-dropdown

   Relational databases, metadata and governance, and accessing data.

   .. raw:: html

         <div class="da-row" data-k="Databases and Relational Database Concepts "><a href="016-databases-and-relational-database-concepts.html">016 · Databases and Relational Database Concepts</a></div>
         <div class="da-row" data-k="Metadata in Databases "><a href="017-metadata-in-databases.html">017 · Metadata in Databases</a></div>
         <div class="da-row" data-k="Metadata Repositories and Data Governance "><a href="018-metadata-repositories-and-data-governance.html">018 · Metadata Repositories and Data Governance</a></div>
         <div class="da-row" data-k="Accessing Data: Internal and External Sources "><a href="019-accessing-data-internal-and-external-sources.html">019 · Accessing Data: Internal and External Sources</a></div>

.. dropdown:: 🔢 Spreadsheets, SQL & Organization
   :class-container: sd-dropdown

   Importing, sorting, querying with SQL, organising, and securing data.

   .. raw:: html

         <div class="da-row" data-k="Importing Data into Spreadsheets "><a href="020-importing-data-into-spreadsheets.html">020 · Importing Data into Spreadsheets</a></div>
         <div class="da-row" data-k="Sorting and Filtering Data in Spreadsheets "><a href="021-sorting-and-filtering-data-in-spreadsheets.html">021 · Sorting and Filtering Data in Spreadsheets</a></div>
         <div class="da-row" data-k="BigQuery Account Types "><a href="022-bigquery-account-types.html">022 · BigQuery Account Types</a></div>
         <div class="da-row" data-k="Querying Data with SQL "><a href="023-querying-data-with-sql.html">023 · Querying Data with SQL</a></div>
         <div class="da-row" data-k="Organizing Data for Personal and Work Projects "><a href="024-organizing-data-for-personal-and-work-projects.html">024 · Organizing Data for Personal and Work Projects</a></div>
         <div class="da-row" data-k="Data Security in Spreadsheets "><a href="025-data-security-in-spreadsheets.html">025 · Data Security in Spreadsheets</a></div>

.. dropdown:: 🔠 A–Z index
   :class-container: term-az

   .. raw:: html

         <div class="da-row" data-k="Accessing Data: Internal and External Sources"><a href="019-accessing-data-internal-and-external-sources.html">Accessing Data: Internal and External Sources</a></div>
         <div class="da-row" data-k="BigQuery Account Types"><a href="022-bigquery-account-types.html">BigQuery Account Types</a></div>
         <div class="da-row" data-k="Choosing the Right Data to Collect"><a href="002-choosing-the-right-data-to-collect.html">Choosing the Right Data to Collect</a></div>
         <div class="da-row" data-k="Common Types of Data Bias"><a href="010-common-types-of-data-bias.html">Common Types of Data Bias</a></div>
         <div class="da-row" data-k="Data Ethics in Data Analysis"><a href="013-data-ethics-in-data-analysis.html">Data Ethics in Data Analysis</a></div>
         <div class="da-row" data-k="Data Privacy in Data Ethics"><a href="014-data-privacy-in-data-ethics.html">Data Privacy in Data Ethics</a></div>
         <div class="da-row" data-k="Data Security in Spreadsheets"><a href="025-data-security-in-spreadsheets.html">Data Security in Spreadsheets</a></div>
         <div class="da-row" data-k="Data Tables (Tabular Data)"><a href="006-data-tables-tabular-data.html">Data Tables (Tabular Data)</a></div>
         <div class="da-row" data-k="Data Types in Spreadsheets"><a href="005-data-types-in-spreadsheets.html">Data Types in Spreadsheets</a></div>
         <div class="da-row" data-k="Databases and Relational Database Concepts"><a href="016-databases-and-relational-database-concepts.html">Databases and Relational Database Concepts</a></div>
         <div class="da-row" data-k="How Data Is Generated and Collected"><a href="001-how-data-is-generated-and-collected.html">How Data Is Generated and Collected</a></div>
         <div class="da-row" data-k="Identifying Bad Data Sources (When Data Does Not ROCCC)"><a href="012-identifying-bad-data-sources-when-data-does-not-roccc.html">Identifying Bad Data Sources (When Data Does Not ROCCC)</a></div>
         <div class="da-row" data-k="Identifying Good Data Sources (ROCCC Framework)"><a href="011-identifying-good-data-sources-roccc-framework.html">Identifying Good Data Sources (ROCCC Framework)</a></div>
         <div class="da-row" data-k="Importing Data into Spreadsheets"><a href="020-importing-data-into-spreadsheets.html">Importing Data into Spreadsheets</a></div>
         <div class="da-row" data-k="Metadata in Databases"><a href="017-metadata-in-databases.html">Metadata in Databases</a></div>
         <div class="da-row" data-k="Metadata Repositories and Data Governance"><a href="018-metadata-repositories-and-data-governance.html">Metadata Repositories and Data Governance</a></div>
         <div class="da-row" data-k="Open Data and Openness in Data Ethics"><a href="015-open-data-and-openness-in-data-ethics.html">Open Data and Openness in Data Ethics</a></div>
         <div class="da-row" data-k="Organizing Data for Personal and Work Projects"><a href="024-organizing-data-for-personal-and-work-projects.html">Organizing Data for Personal and Work Projects</a></div>
         <div class="da-row" data-k="Querying Data with SQL"><a href="023-querying-data-with-sql.html">Querying Data with SQL</a></div>
         <div class="da-row" data-k="Sampling Bias and Unbiased Data"><a href="009-sampling-bias-and-unbiased-data.html">Sampling Bias and Unbiased Data</a></div>
         <div class="da-row" data-k="Sorting and Filtering Data in Spreadsheets"><a href="021-sorting-and-filtering-data-in-spreadsheets.html">Sorting and Filtering Data in Spreadsheets</a></div>
         <div class="da-row" data-k="Structured Data and Data Models"><a href="004-structured-data-and-data-models.html">Structured Data and Data Models</a></div>
         <div class="da-row" data-k="Understanding Bias in Data Analysis"><a href="008-understanding-bias-in-data-analysis.html">Understanding Bias in Data Analysis</a></div>
         <div class="da-row" data-k="Understanding Data Types and Data Formats"><a href="003-understanding-data-types-and-data-formats.html">Understanding Data Types and Data Formats</a></div>
         <div class="da-row" data-k="Wide Data vs. Long Data"><a href="007-wide-data-vs-long-data.html">Wide Data vs. Long Data</a></div>
