:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-index:

:raw-html:`<div style="text-align:center"><strong>` 🐍 Data Analysis Using Python
|br| |full_version| - |today|
:raw-html:`</strong></div>`

========================================================================
🐍 Data Analysis Using Python
========================================================================

*Section 7 of the Data Analytics hub — 33 of 33 lessons.*

Doing the whole workflow in Python: NumPy, pandas, and plotting for real analytical tasks.

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


.. dropdown:: 🐍 Python Fundamentals
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   From first program and Jupyter to variables, types, functions, and clean code.

   * :doc:`001 · Introduction to Python and Programming Fundamentals — why analysts learn Python, and the programming ideas underneath it <001-introduction-to-python-and-programming-fundamentals>`
   * :doc:`002 · Python Fundamentals — the core building blocks of Python — values, expressions, statements, and flow <002-python-fundamentals>`
   * :doc:`003 · Jupyter Notebook and Coding Environments — where analysts write and run Python — notebooks, IDEs, and the interactive workflow <003-jupyter-notebook-and-coding-environments>`
   * :doc:`004 · Object-Oriented Programming (OOP) in Python — the objects-and-methods model that shapes Python, including pandas and numpy <004-object-oriented-programming-oop-in-python>`
   * :doc:`005 · Variables in Python — named containers for values — assigning, reassigning, and using them <005-variables-in-python>`
   * :doc:`006 · Naming Conventions and Restrictions in Python — the rules for valid names and the conventions for good ones (PEP 8) <006-naming-conventions-and-restrictions-in-python>`
   * :doc:`007 · Data Types and Type Conversion in Python — the kinds of values Python holds, and converting between them safely <007-data-types-and-type-conversion-in-python>`
   * :doc:`008 · Functions in Python — named, reusable blocks of code — defining, calling, arguments, and returns <008-functions-in-python>`
   * :doc:`009 · Code Reusability, Modularity, and Clean Code in Python — writing Python that is reusable, modular, and clean — the engineering virtues in code <009-code-reusability-modularity-and-clean-code-in-python>`
   * :doc:`010 · Comments, Algorithms, and Docstrings in Python — documenting code — comments, docstrings, and thinking in algorithms <010-comments-algorithms-and-docstrings-in-python>`

.. dropdown:: 🔀 Control Flow
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Booleans, branching, and iteration with while, for, and range.

   * :doc:`011 · Boolean Data, Comparators, and Logical Operators in Python — True/False values and the comparisons and logic that produce them <011-boolean-data-comparators-and-logical-operators-in-python>`
   * :doc:`012 · Branching and Conditional Statements in Python — making code decide — if, elif, and else direct which code runs <012-branching-and-conditional-statements-in-python>`
   * :doc:`013 · While Loops and Iteration in Python — repeating code while a condition holds — the while loop and iteration basics <013-while-loops-and-iteration-in-python>`
   * :doc:`014 · For Loops in Python — repeating code over the items of a collection — the for loop <014-for-loops-in-python>`
   * :doc:`015 · range() Function and Loop Control in Python — generating number sequences and controlling loops with break and continue <015-range-function-and-loop-control-in-python>`

.. dropdown:: 📚 Strings & Data Structures
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Strings, lists, tuples, comprehensions, dictionaries, and sets in practice.

   * :doc:`016 · Strings in Python — working with text — the string type, its operations, and its immutability <016-strings-in-python>`
   * :doc:`017 · String Indexing and Slicing in Python — reaching into strings by position — single characters and substrings <017-string-indexing-and-slicing-in-python>`
   * :doc:`018 · String Formatting with .format() in Python — building strings from values cleanly — the .format() method and f-strings <018-string-formatting-with-format-in-python>`
   * :doc:`019 · Data Types vs Data Structures & Introduction to Lists — the difference between a value's type and a structure that holds many, and the list <019-data-types-vs-data-structures-and-introduction-to-lists>`
   * :doc:`020 · Modifying Lists in Python — changing lists in place — adding, removing, and updating elements <020-modifying-lists-in-python>`
   * :doc:`021 · Tuples in Python — ordered, immutable collections — like lists that cannot change <021-tuples-in-python>`
   * :doc:`022 · Advanced Use of Loops, Lists, Tuples & List Comprehension — combining loops and structures, and the concise list comprehension idiom <022-advanced-use-of-loops-lists-tuples-and-list-comprehension>`
   * :doc:`023 · Dictionaries in Python — key-value collections — looking values up by a meaningful key <023-dictionaries-in-python>`
   * :doc:`024 · Advanced Dictionary Usage in Python — iterating, nesting, and safely accessing dictionaries for real data <024-advanced-dictionary-usage-in-python>`
   * :doc:`025 · Sets in Python — unordered collections of unique values — membership and set operations <025-sets-in-python>`

.. dropdown:: 🐼 NumPy & pandas
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Vectorized arrays and DataFrames: masking, groupby, and combining data.

   * :doc:`026 · Libraries, Packages, and Modules in Python — reusing others' code — importing modules, packages, and the library ecosystem <026-libraries-packages-and-modules-in-python>`
   * :doc:`027 · Introduction to NumPy and Vectorization — the numerical library and vectorization — fast array operations without loops <027-introduction-to-numpy-and-vectorization>`
   * :doc:`028 · NumPy Arrays (ndarray) and Core Concepts — the ndarray, NumPy's core structure, and how it differs from a list <028-numpy-arrays-ndarray-and-core-concepts>`
   * :doc:`029 · Introduction to Pandas (Data Analysis Library) — the DataFrame library that brings spreadsheet/SQL-style tables to Python <029-introduction-to-pandas-data-analysis-library>`
   * :doc:`030 · Pandas DataFrame & Series — the two core pandas structures — the table (DataFrame) and the column (Series) <030-pandas-dataframe-and-series>`
   * :doc:`031 · Boolean Masking in Pandas — filtering rows by a condition — the pandas equivalent of WHERE and spreadsheet filters <031-boolean-masking-in-pandas>`
   * :doc:`032 · Grouping and Aggregation in Pandas (groupby, agg) — grouping rows and computing per-group aggregates — pandas' GROUP BY <032-grouping-and-aggregation-in-pandas-groupby-agg>`
   * :doc:`033 · Combining Data in Pandas (concat and merge) — bringing DataFrames together — stacking with concat, joining on keys with merge <033-combining-data-in-pandas-concat-and-merge>`

🔤 Every lesson, A–Z
---------------------

.. dropdown:: 🔠 A–Z index
   :animate: fade-in-slide-down
   :class-container: term-az

   .. hlist::
      :columns: 2

      * :doc:`Advanced Dictionary Usage in Python <024-advanced-dictionary-usage-in-python>`
      * :doc:`Advanced Use of Loops, Lists, Tuples & List Comprehension <022-advanced-use-of-loops-lists-tuples-and-list-comprehension>`
      * :doc:`Boolean Data, Comparators, and Logical Operators in Python <011-boolean-data-comparators-and-logical-operators-in-python>`
      * :doc:`Boolean Masking in Pandas <031-boolean-masking-in-pandas>`
      * :doc:`Branching and Conditional Statements in Python <012-branching-and-conditional-statements-in-python>`
      * :doc:`Code Reusability, Modularity, and Clean Code in Python <009-code-reusability-modularity-and-clean-code-in-python>`
      * :doc:`Combining Data in Pandas (concat and merge) <033-combining-data-in-pandas-concat-and-merge>`
      * :doc:`Comments, Algorithms, and Docstrings in Python <010-comments-algorithms-and-docstrings-in-python>`
      * :doc:`Data Types and Type Conversion in Python <007-data-types-and-type-conversion-in-python>`
      * :doc:`Data Types vs Data Structures & Introduction to Lists <019-data-types-vs-data-structures-and-introduction-to-lists>`
      * :doc:`Dictionaries in Python <023-dictionaries-in-python>`
      * :doc:`For Loops in Python <014-for-loops-in-python>`
      * :doc:`Functions in Python <008-functions-in-python>`
      * :doc:`Grouping and Aggregation in Pandas (groupby, agg) <032-grouping-and-aggregation-in-pandas-groupby-agg>`
      * :doc:`Introduction to NumPy and Vectorization <027-introduction-to-numpy-and-vectorization>`
      * :doc:`Introduction to Pandas (Data Analysis Library) <029-introduction-to-pandas-data-analysis-library>`
      * :doc:`Introduction to Python and Programming Fundamentals <001-introduction-to-python-and-programming-fundamentals>`
      * :doc:`Jupyter Notebook and Coding Environments <003-jupyter-notebook-and-coding-environments>`
      * :doc:`Libraries, Packages, and Modules in Python <026-libraries-packages-and-modules-in-python>`
      * :doc:`Modifying Lists in Python <020-modifying-lists-in-python>`
      * :doc:`Naming Conventions and Restrictions in Python <006-naming-conventions-and-restrictions-in-python>`
      * :doc:`NumPy Arrays (ndarray) and Core Concepts <028-numpy-arrays-ndarray-and-core-concepts>`
      * :doc:`Object-Oriented Programming (OOP) in Python <004-object-oriented-programming-oop-in-python>`
      * :doc:`Pandas DataFrame & Series <030-pandas-dataframe-and-series>`
      * :doc:`Python Fundamentals <002-python-fundamentals>`
      * :doc:`range() Function and Loop Control in Python <015-range-function-and-loop-control-in-python>`
      * :doc:`Sets in Python <025-sets-in-python>`
      * :doc:`String Formatting with .format() in Python <018-string-formatting-with-format-in-python>`
      * :doc:`String Indexing and Slicing in Python <017-string-indexing-and-slicing-in-python>`
      * :doc:`Strings in Python <016-strings-in-python>`
      * :doc:`Tuples in Python <021-tuples-in-python>`
      * :doc:`Variables in Python <005-variables-in-python>`
      * :doc:`While Loops and Iteration in Python <013-while-loops-and-iteration-in-python>`

.. toctree::
   :hidden:
   :includehidden:
   :maxdepth: 1

   001-introduction-to-python-and-programming-fundamentals
   002-python-fundamentals
   003-jupyter-notebook-and-coding-environments
   004-object-oriented-programming-oop-in-python
   005-variables-in-python
   006-naming-conventions-and-restrictions-in-python
   007-data-types-and-type-conversion-in-python
   008-functions-in-python
   009-code-reusability-modularity-and-clean-code-in-python
   010-comments-algorithms-and-docstrings-in-python
   011-boolean-data-comparators-and-logical-operators-in-python
   012-branching-and-conditional-statements-in-python
   013-while-loops-and-iteration-in-python
   014-for-loops-in-python
   015-range-function-and-loop-control-in-python
   016-strings-in-python
   017-string-indexing-and-slicing-in-python
   018-string-formatting-with-format-in-python
   019-data-types-vs-data-structures-and-introduction-to-lists
   020-modifying-lists-in-python
   021-tuples-in-python
   022-advanced-use-of-loops-lists-tuples-and-list-comprehension
   023-dictionaries-in-python
   024-advanced-dictionary-usage-in-python
   025-sets-in-python
   026-libraries-packages-and-modules-in-python
   027-introduction-to-numpy-and-vectorization
   028-numpy-arrays-ndarray-and-core-concepts
   029-introduction-to-pandas-data-analysis-library
   030-pandas-dataframe-and-series
   031-boolean-masking-in-pandas
   032-grouping-and-aggregation-in-pandas-groupby-agg
   033-combining-data-in-pandas-concat-and-merge

.. tags:: purpose: reference, topic: data analytics
