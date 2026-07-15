:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-index:

:raw-html:`<div style="text-align:center"><strong>` 📦 Data Preparation
|br| |full_version| - |today|
:raw-html:`</strong></div>`

========================================================================
📦 Data Preparation
========================================================================

*Section 3 of the Data Analytics hub — 25 of 25 lessons.*

Sourcing, structuring, and organising data before analysis: types, formats, databases, and sampling.

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


.. dropdown:: 🧬 Data Types & Structure
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   How data is generated, its types and formats, and structured vs tabular shapes.

   * :doc:`001 · How Data Is Generated and Collected — where data comes from: first/second/third-party sources and how it is produced <001-how-data-is-generated-and-collected>`
   * :doc:`002 · Choosing the Right Data to Collect — relevance, coverage, and cost: deciding what data a question actually needs <002-choosing-the-right-data-to-collect>`
   * :doc:`003 · Understanding Data Types and Data Formats — nominal, ordinal, discrete, continuous — and the file formats data travels in <003-understanding-data-types-and-data-formats>`
   * :doc:`004 · Structured Data and Data Models — structured vs semi- vs unstructured, and the models that give data its shape <004-structured-data-and-data-models>`
   * :doc:`005 · Data Types in Spreadsheets — text, number, date, boolean — why a spreadsheet cares what type a cell holds <005-data-types-in-spreadsheets>`
   * :doc:`006 · Data Tables (Tabular Data) — the row-and-column table: the workhorse shape of analysable data <006-data-tables-tabular-data>`
   * :doc:`007 · Wide Data vs. Long Data — two tidy layouts of the same data, and when each one serves <007-wide-data-vs-long-data>`

.. dropdown:: ⚖️ Bias & Data Ethics
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Recognising bias, judging sources with ROCCC, and the ethics of data use.

   * :doc:`008 · Understanding Bias in Data Analysis — what bias is, where it hides, and why unbiased data is the goal not the default <008-understanding-bias-in-data-analysis>`
   * :doc:`009 · Sampling Bias and Unbiased Data — when the sample misrepresents the population — and how random sampling guards against it <009-sampling-bias-and-unbiased-data>`
   * :doc:`010 · Common Types of Data Bias — sampling, observer, interpretation, and confirmation bias, with how each distorts <010-common-types-of-data-bias>`
   * :doc:`011 · Identifying Good Data Sources (ROCCC Framework) — Reliable, Original, Comprehensive, Current, Cited — the marks of trustworthy data <011-identifying-good-data-sources-roccc-framework>`
   * :doc:`012 · Identifying Bad Data Sources (When Data Does Not ROCCC) — reading ROCCC in reverse to spot the data you should not trust <012-identifying-bad-data-sources-when-data-does-not-roccc>`
   * :doc:`013 · Data Ethics in Data Analysis — ownership, transaction transparency, consent, and currency — the ethics of using data <013-data-ethics-in-data-analysis>`
   * :doc:`014 · Data Privacy in Data Ethics — protecting people's information: what privacy requires of an analyst <014-data-privacy-in-data-ethics>`
   * :doc:`015 · Open Data and Openness in Data Ethics — when data should be freely available — and the tension with privacy <015-open-data-and-openness-in-data-ethics>`

.. dropdown:: 🗄️ Databases & Data Sources
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Relational databases, metadata and governance, and accessing data.

   * :doc:`016 · Databases and Relational Database Concepts — tables, keys, and relationships: how organisational data is really stored <016-databases-and-relational-database-concepts>`
   * :doc:`017 · Metadata in Databases — data about data: descriptive, structural, and administrative context that makes data usable <017-metadata-in-databases>`
   * :doc:`018 · Metadata Repositories and Data Governance — where metadata is catalogued, and the governance that keeps data trustworthy <018-metadata-repositories-and-data-governance>`
   * :doc:`019 · Accessing Data: Internal and External Sources — getting to the data: what lives inside the organisation versus outside <019-accessing-data-internal-and-external-sources>`

.. dropdown:: 🔢 Spreadsheets, SQL & Organization
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Importing, sorting, querying with SQL, organising, and securing data.

   * :doc:`020 · Importing Data into Spreadsheets — getting external data into a sheet cleanly — and the type traps to watch <020-importing-data-into-spreadsheets>`
   * :doc:`021 · Sorting and Filtering Data in Spreadsheets — ordering and narrowing rows: the two most-used moves for making data legible <021-sorting-and-filtering-data-in-spreadsheets>`
   * :doc:`022 · BigQuery Account Types — sandbox, free tier, and paid — how to access a cloud data warehouse for practice <022-bigquery-account-types>`
   * :doc:`023 · Querying Data with SQL — SELECT, FROM, WHERE: retrieving exactly the rows and columns a question needs <023-querying-data-with-sql>`
   * :doc:`024 · Organizing Data for Personal and Work Projects — folder, file, and naming conventions that keep a data project findable and safe <024-organizing-data-for-personal-and-work-projects>`
   * :doc:`025 · Data Security in Spreadsheets — protecting a shared sheet: access control, protected ranges, and safe sharing <025-data-security-in-spreadsheets>`

🔤 Every lesson, A–Z index
---------------------------

.. dropdown:: 🔠 Open the full alphabetical index
   :animate: fade-in-slide-down
   :class-container: term-az

   .. hlist::
      :columns: 2

      * :doc:`Accessing Data: Internal and External Sources <019-accessing-data-internal-and-external-sources>`
      * :doc:`BigQuery Account Types <022-bigquery-account-types>`
      * :doc:`Choosing the Right Data to Collect <002-choosing-the-right-data-to-collect>`
      * :doc:`Common Types of Data Bias <010-common-types-of-data-bias>`
      * :doc:`Data Ethics in Data Analysis <013-data-ethics-in-data-analysis>`
      * :doc:`Data Privacy in Data Ethics <014-data-privacy-in-data-ethics>`
      * :doc:`Data Security in Spreadsheets <025-data-security-in-spreadsheets>`
      * :doc:`Data Tables (Tabular Data) <006-data-tables-tabular-data>`
      * :doc:`Data Types in Spreadsheets <005-data-types-in-spreadsheets>`
      * :doc:`Databases and Relational Database Concepts <016-databases-and-relational-database-concepts>`
      * :doc:`How Data Is Generated and Collected <001-how-data-is-generated-and-collected>`
      * :doc:`Identifying Bad Data Sources (When Data Does Not ROCCC) <012-identifying-bad-data-sources-when-data-does-not-roccc>`
      * :doc:`Identifying Good Data Sources (ROCCC Framework) <011-identifying-good-data-sources-roccc-framework>`
      * :doc:`Importing Data into Spreadsheets <020-importing-data-into-spreadsheets>`
      * :doc:`Metadata in Databases <017-metadata-in-databases>`
      * :doc:`Metadata Repositories and Data Governance <018-metadata-repositories-and-data-governance>`
      * :doc:`Open Data and Openness in Data Ethics <015-open-data-and-openness-in-data-ethics>`
      * :doc:`Organizing Data for Personal and Work Projects <024-organizing-data-for-personal-and-work-projects>`
      * :doc:`Querying Data with SQL <023-querying-data-with-sql>`
      * :doc:`Sampling Bias and Unbiased Data <009-sampling-bias-and-unbiased-data>`
      * :doc:`Sorting and Filtering Data in Spreadsheets <021-sorting-and-filtering-data-in-spreadsheets>`
      * :doc:`Structured Data and Data Models <004-structured-data-and-data-models>`
      * :doc:`Understanding Bias in Data Analysis <008-understanding-bias-in-data-analysis>`
      * :doc:`Understanding Data Types and Data Formats <003-understanding-data-types-and-data-formats>`
      * :doc:`Wide Data vs. Long Data <007-wide-data-vs-long-data>`

.. toctree::
   :hidden:
   :includehidden:
   :maxdepth: 1

   001-how-data-is-generated-and-collected
   002-choosing-the-right-data-to-collect
   003-understanding-data-types-and-data-formats
   004-structured-data-and-data-models
   005-data-types-in-spreadsheets
   006-data-tables-tabular-data
   007-wide-data-vs-long-data
   008-understanding-bias-in-data-analysis
   009-sampling-bias-and-unbiased-data
   010-common-types-of-data-bias
   011-identifying-good-data-sources-roccc-framework
   012-identifying-bad-data-sources-when-data-does-not-roccc
   013-data-ethics-in-data-analysis
   014-data-privacy-in-data-ethics
   015-open-data-and-openness-in-data-ethics
   016-databases-and-relational-database-concepts
   017-metadata-in-databases
   018-metadata-repositories-and-data-governance
   019-accessing-data-internal-and-external-sources
   020-importing-data-into-spreadsheets
   021-sorting-and-filtering-data-in-spreadsheets
   022-bigquery-account-types
   023-querying-data-with-sql
   024-organizing-data-for-personal-and-work-projects
   025-data-security-in-spreadsheets

.. tags:: purpose: reference, topic: data analytics
