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


.. dropdown:: 🐍 Python Fundamentals
   :class-container: sd-dropdown

   From first program and Jupyter to variables, types, functions, and clean code.

   .. raw:: html

         <div class="da-row" data-k="Introduction to Python and Programming Fundamentals "><a href="001-introduction-to-python-and-programming-fundamentals.html">001 · Introduction to Python and Programming Fundamentals</a></div>
         <div class="da-row" data-k="Python Fundamentals "><a href="002-python-fundamentals.html">002 · Python Fundamentals</a></div>
         <div class="da-row" data-k="Jupyter Notebook and Coding Environments "><a href="003-jupyter-notebook-and-coding-environments.html">003 · Jupyter Notebook and Coding Environments</a></div>
         <div class="da-row" data-k="Object-Oriented Programming (OOP) in Python "><a href="004-object-oriented-programming-oop-in-python.html">004 · Object-Oriented Programming (OOP) in Python</a></div>
         <div class="da-row" data-k="Variables in Python "><a href="005-variables-in-python.html">005 · Variables in Python</a></div>
         <div class="da-row" data-k="Naming Conventions and Restrictions in Python "><a href="006-naming-conventions-and-restrictions-in-python.html">006 · Naming Conventions and Restrictions in Python</a></div>
         <div class="da-row" data-k="Data Types and Type Conversion in Python "><a href="007-data-types-and-type-conversion-in-python.html">007 · Data Types and Type Conversion in Python</a></div>
         <div class="da-row" data-k="Functions in Python "><a href="008-functions-in-python.html">008 · Functions in Python</a></div>
         <div class="da-row" data-k="Code Reusability, Modularity, and Clean Code in Python "><a href="009-code-reusability-modularity-and-clean-code-in-python.html">009 · Code Reusability, Modularity, and Clean Code in Python</a></div>
         <div class="da-row" data-k="Comments, Algorithms, and Docstrings in Python "><a href="010-comments-algorithms-and-docstrings-in-python.html">010 · Comments, Algorithms, and Docstrings in Python</a></div>

.. dropdown:: 🔀 Control Flow
   :class-container: sd-dropdown

   Booleans, branching, and iteration with while, for, and range.

   .. raw:: html

         <div class="da-row" data-k="Boolean Data, Comparators, and Logical Operators in Python "><a href="011-boolean-data-comparators-and-logical-operators-in-python.html">011 · Boolean Data, Comparators, and Logical Operators in Python</a></div>
         <div class="da-row" data-k="Branching and Conditional Statements in Python "><a href="012-branching-and-conditional-statements-in-python.html">012 · Branching and Conditional Statements in Python</a></div>
         <div class="da-row" data-k="While Loops and Iteration in Python "><a href="013-while-loops-and-iteration-in-pytho.html">013 · While Loops and Iteration in Python</a></div>
         <div class="da-row" data-k="For Loops in Python "><a href="014-for-loops-in-python.html">014 · For Loops in Python</a></div>
         <div class="da-row" data-k="range() Function and Loop Control in Python "><a href="015-range-function-and-loop-control-in-python.html">015 · range() Function and Loop Control in Python</a></div>

.. dropdown:: 📚 Strings & Data Structures
   :class-container: sd-dropdown

   Strings, lists, tuples, comprehensions, dictionaries, and sets in practice.

   .. raw:: html

         <div class="da-row" data-k="Strings in Python "><a href="016-strings-in-python.html">016 · Strings in Python</a></div>
         <div class="da-row" data-k="String Indexing and Slicing in Python "><a href="017-string-indexing-and-slicing-in-python.html">017 · String Indexing and Slicing in Python</a></div>
         <div class="da-row" data-k="String Formatting with .format() in Python "><a href="018-string-formatting-with-format-in-python.html">018 · String Formatting with .format() in Python</a></div>
         <div class="da-row" data-k="Data Types vs Data Structures & Introduction to Lists "><a href="019-data-types-vs-data-structures-and-introduction-to-lists.html">019 · Data Types vs Data Structures & Introduction to Lists</a></div>
         <div class="da-row" data-k="Modifying Lists in Python "><a href="020-modifying-lists-in-python.html">020 · Modifying Lists in Python</a></div>
         <div class="da-row" data-k="Tuples in Python "><a href="021-tuples-in-python.html">021 · Tuples in Python</a></div>
         <div class="da-row" data-k="Advanced Use of Loops, Lists, Tuples & List Comprehension "><a href="022-advanced-use-of-loops-lists-tuples-and-list-comprehension.html">022 · Advanced Use of Loops, Lists, Tuples & List Comprehension</a></div>
         <div class="da-row" data-k="Dictionaries in Python "><a href="023-dictionaries-in-python.html">023 · Dictionaries in Python</a></div>
         <div class="da-row" data-k="Advanced Dictionary Usage in Python "><a href="024-advanced-dictionary-usage-in-python.html">024 · Advanced Dictionary Usage in Python</a></div>
         <div class="da-row" data-k="Sets in Python "><a href="025-sets-in-python.html">025 · Sets in Python</a></div>

.. dropdown:: 🐼 NumPy & pandas
   :class-container: sd-dropdown

   Vectorized arrays and DataFrames: masking, groupby, and combining data.

   .. raw:: html

         <div class="da-row" data-k="Libraries, Packages, and Modules in Python "><a href="026-libraries-packages-and-modules-in-python.html">026 · Libraries, Packages, and Modules in Python</a></div>
         <div class="da-row" data-k="Introduction to NumPy and Vectorization "><a href="027-introduction-to-numpy-and-vectorization.html">027 · Introduction to NumPy and Vectorization</a></div>
         <div class="da-row" data-k="NumPy Arrays (ndarray) and Core Concepts "><a href="028-numpy-arrays-ndarray-and-core-concepts.html">028 · NumPy Arrays (ndarray) and Core Concepts</a></div>
         <div class="da-row" data-k="Introduction to Pandas (Data Analysis Library) "><a href="029-introduction-to-pandas-data-analysis-library.html">029 · Introduction to Pandas (Data Analysis Library)</a></div>
         <div class="da-row" data-k="Pandas DataFrame & Series "><a href="030-pandas-dataframe-and-series.html">030 · Pandas DataFrame & Series</a></div>
         <div class="da-row" data-k="Boolean Masking in Pandas "><a href="031-boolean-masking-in-pandas.html">031 · Boolean Masking in Pandas</a></div>
         <div class="da-row" data-k="Grouping and Aggregation in Pandas (groupby, agg) "><a href="032-grouping-and-aggregation-in-pandas-groupby-agg.html">032 · Grouping and Aggregation in Pandas (groupby, agg)</a></div>
         <div class="da-row" data-k="Combining Data in Pandas (concat and merge) "><a href="033-combining-data-in-pandas-concat-and-merge.html">033 · Combining Data in Pandas (concat and merge)</a></div>

🔤 Every lesson, A–Z
---------------------

.. dropdown:: 🔠 A–Z index
   :class-container: term-az

   .. raw:: html

         <div class="da-row" data-k="Advanced Dictionary Usage in Python"><a href="024-advanced-dictionary-usage-in-python.html">Advanced Dictionary Usage in Python</a></div>
         <div class="da-row" data-k="Advanced Use of Loops, Lists, Tuples & List Comprehension"><a href="022-advanced-use-of-loops-lists-tuples-and-list-comprehension.html">Advanced Use of Loops, Lists, Tuples & List Comprehension</a></div>
         <div class="da-row" data-k="Boolean Data, Comparators, and Logical Operators in Python"><a href="011-boolean-data-comparators-and-logical-operators-in-python.html">Boolean Data, Comparators, and Logical Operators in Python</a></div>
         <div class="da-row" data-k="Boolean Masking in Pandas"><a href="031-boolean-masking-in-pandas.html">Boolean Masking in Pandas</a></div>
         <div class="da-row" data-k="Branching and Conditional Statements in Python"><a href="012-branching-and-conditional-statements-in-python.html">Branching and Conditional Statements in Python</a></div>
         <div class="da-row" data-k="Code Reusability, Modularity, and Clean Code in Python"><a href="009-code-reusability-modularity-and-clean-code-in-python.html">Code Reusability, Modularity, and Clean Code in Python</a></div>
         <div class="da-row" data-k="Combining Data in Pandas (concat and merge)"><a href="033-combining-data-in-pandas-concat-and-merge.html">Combining Data in Pandas (concat and merge)</a></div>
         <div class="da-row" data-k="Comments, Algorithms, and Docstrings in Python"><a href="010-comments-algorithms-and-docstrings-in-python.html">Comments, Algorithms, and Docstrings in Python</a></div>
         <div class="da-row" data-k="Data Types and Type Conversion in Python"><a href="007-data-types-and-type-conversion-in-python.html">Data Types and Type Conversion in Python</a></div>
         <div class="da-row" data-k="Data Types vs Data Structures & Introduction to Lists"><a href="019-data-types-vs-data-structures-and-introduction-to-lists.html">Data Types vs Data Structures & Introduction to Lists</a></div>
         <div class="da-row" data-k="Dictionaries in Python"><a href="023-dictionaries-in-python.html">Dictionaries in Python</a></div>
         <div class="da-row" data-k="For Loops in Python"><a href="014-for-loops-in-python.html">For Loops in Python</a></div>
         <div class="da-row" data-k="Functions in Python"><a href="008-functions-in-python.html">Functions in Python</a></div>
         <div class="da-row" data-k="Grouping and Aggregation in Pandas (groupby, agg)"><a href="032-grouping-and-aggregation-in-pandas-groupby-agg.html">Grouping and Aggregation in Pandas (groupby, agg)</a></div>
         <div class="da-row" data-k="Introduction to NumPy and Vectorization"><a href="027-introduction-to-numpy-and-vectorization.html">Introduction to NumPy and Vectorization</a></div>
         <div class="da-row" data-k="Introduction to Pandas (Data Analysis Library)"><a href="029-introduction-to-pandas-data-analysis-library.html">Introduction to Pandas (Data Analysis Library)</a></div>
         <div class="da-row" data-k="Introduction to Python and Programming Fundamentals"><a href="001-introduction-to-python-and-programming-fundamentals.html">Introduction to Python and Programming Fundamentals</a></div>
         <div class="da-row" data-k="Jupyter Notebook and Coding Environments"><a href="003-jupyter-notebook-and-coding-environments.html">Jupyter Notebook and Coding Environments</a></div>
         <div class="da-row" data-k="Libraries, Packages, and Modules in Python"><a href="026-libraries-packages-and-modules-in-python.html">Libraries, Packages, and Modules in Python</a></div>
         <div class="da-row" data-k="Modifying Lists in Python"><a href="020-modifying-lists-in-python.html">Modifying Lists in Python</a></div>
         <div class="da-row" data-k="Naming Conventions and Restrictions in Python"><a href="006-naming-conventions-and-restrictions-in-python.html">Naming Conventions and Restrictions in Python</a></div>
         <div class="da-row" data-k="NumPy Arrays (ndarray) and Core Concepts"><a href="028-numpy-arrays-ndarray-and-core-concepts.html">NumPy Arrays (ndarray) and Core Concepts</a></div>
         <div class="da-row" data-k="Object-Oriented Programming (OOP) in Python"><a href="004-object-oriented-programming-oop-in-python.html">Object-Oriented Programming (OOP) in Python</a></div>
         <div class="da-row" data-k="Pandas DataFrame & Series"><a href="030-pandas-dataframe-and-series.html">Pandas DataFrame & Series</a></div>
         <div class="da-row" data-k="Python Fundamentals"><a href="002-python-fundamentals.html">Python Fundamentals</a></div>
         <div class="da-row" data-k="range() Function and Loop Control in Python"><a href="015-range-function-and-loop-control-in-python.html">range() Function and Loop Control in Python</a></div>
         <div class="da-row" data-k="Sets in Python"><a href="025-sets-in-python.html">Sets in Python</a></div>
         <div class="da-row" data-k="String Formatting with .format() in Python"><a href="018-string-formatting-with-format-in-python.html">String Formatting with .format() in Python</a></div>
         <div class="da-row" data-k="String Indexing and Slicing in Python"><a href="017-string-indexing-and-slicing-in-python.html">String Indexing and Slicing in Python</a></div>
         <div class="da-row" data-k="Strings in Python"><a href="016-strings-in-python.html">Strings in Python</a></div>
         <div class="da-row" data-k="Tuples in Python"><a href="021-tuples-in-python.html">Tuples in Python</a></div>
         <div class="da-row" data-k="Variables in Python"><a href="005-variables-in-python.html">Variables in Python</a></div>
         <div class="da-row" data-k="While Loops and Iteration in Python"><a href="013-while-loops-and-iteration-in-pytho.html">While Loops and Iteration in Python</a></div>
