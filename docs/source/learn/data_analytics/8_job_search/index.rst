.. _da-jobsearch-index:

:raw-html:`<div style="text-align:center"><strong>` 💼 Job Search
|br| |full_version| - |today|
:raw-html:`</strong></div>`

========================================================================
💼 Job Search
========================================================================

*Section 8 of the Data Analytics hub — 15 of 15 lessons.*

From portfolio to offer: resumes, the analyst interview, case studies, and landing the role.

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


.. dropdown:: 🧭 Career Identity & Planning
   :class-container: sd-dropdown

   Transferable skills, a career identity statement, and an AI-assisted search plan.

   .. raw:: html

         <div class="da-row" data-k="Transferable Skills "><a href="001-transferable-skills.html">001 · Transferable Skills</a></div>
         <div class="da-row" data-k="Career Identity Statement "><a href="002-career-identity-statement.html">002 · Career Identity Statement</a></div>
         <div class="da-row" data-k="Career Dreamer (AI Tool for Career Exploration) "><a href="003-career-dreamer-ai-tool-for-career-exploration.html">003 · Career Dreamer (AI Tool for Career Exploration)</a></div>
         <div class="da-row" data-k="Job Search Plan (Using AI Tools) "><a href="004-job-search-plan-using-ai-tools.html">004 · Job Search Plan (Using AI Tools)</a></div>

.. dropdown:: 📄 Resume, Brand & Applications
   :class-container: sd-dropdown

   Tailoring resumes, building an online presence, platforms, tracking, networking.

   .. raw:: html

         <div class="da-row" data-k="Tailoring Your Resume "><a href="005-tailoring-your-resume.html">005 · Tailoring Your Resume</a></div>
         <div class="da-row" data-k="Using AI to Improve and Tailor Your Resume "><a href="006-using-ai-to-improve-and-tailor-your-resume.html">006 · Using AI to Improve and Tailor Your Resume</a></div>
         <div class="da-row" data-k="Building a Professional Online Presence (Personal Brand) "><a href="007-building-a-professional-online-presence-personal-brand.html">007 · Building a Professional Online Presence (Personal Brand)</a></div>
         <div class="da-row" data-k="Choosing the Right Job Platforms "><a href="008-choosing-the-right-job-platforms.html">008 · Choosing the Right Job Platforms</a></div>
         <div class="da-row" data-k="Job Application Tracking (Using AI + Spreadsheets) "><a href="009-job-application-tracking-using-ai-spreadsheets.html">009 · Job Application Tracking (Using AI + Spreadsheets)</a></div>
         <div class="da-row" data-k="Networking for Job Search "><a href="010-networking-for-job-search.html">010 · Networking for Job Search</a></div>

.. dropdown:: 🎯 Interviews & Follow-Up
   :class-container: sd-dropdown

   Interview prep, the STAR method, AI practice tools, and post-interview strategy.

   .. raw:: html

         <div class="da-row" data-k="Interview Preparation "><a href="011-interview-preparation.html">011 · Interview Preparation</a></div>
         <div class="da-row" data-k="STAR Method (Behavioral Interview) "><a href="012-star-method-behavioral-interview.html">012 · STAR Method (Behavioral Interview)</a></div>
         <div class="da-row" data-k="Using AI (NotebookLM) for Interview Preparation "><a href="013-using-ai-notebooklm-for-interview-preparation.html">013 · Using AI (NotebookLM) for Interview Preparation</a></div>
         <div class="da-row" data-k="Practicing Interviews with AI (Gemini Live) "><a href="014-practicing-interviews-with-ai-gemini-live.html">014 · Practicing Interviews with AI (Gemini Live)</a></div>
         <div class="da-row" data-k="Post-Interview Strategy "><a href="015-post-interview-strategy.html">015 · Post-Interview Strategy</a></div>

🔤 Every lesson, A–Z
---------------------

.. dropdown:: 🔠 A–Z index
   :class-container: term-az

   .. raw:: html

         <div class="da-row" data-k="Building a Professional Online Presence (Personal Brand)"><a href="007-building-a-professional-online-presence-personal-brand.html">Building a Professional Online Presence (Personal Brand)</a></div>
         <div class="da-row" data-k="Career Dreamer (AI Tool for Career Exploration)"><a href="003-career-dreamer-ai-tool-for-career-exploration.html">Career Dreamer (AI Tool for Career Exploration)</a></div>
         <div class="da-row" data-k="Career Identity Statement"><a href="002-career-identity-statement.html">Career Identity Statement</a></div>
         <div class="da-row" data-k="Choosing the Right Job Platforms"><a href="008-choosing-the-right-job-platforms.html">Choosing the Right Job Platforms</a></div>
         <div class="da-row" data-k="Interview Preparation"><a href="011-interview-preparation.html">Interview Preparation</a></div>
         <div class="da-row" data-k="Job Application Tracking (Using AI + Spreadsheets)"><a href="009-job-application-tracking-using-ai-spreadsheets.html">Job Application Tracking (Using AI + Spreadsheets)</a></div>
         <div class="da-row" data-k="Job Search Plan (Using AI Tools)"><a href="004-job-search-plan-using-ai-tools.html">Job Search Plan (Using AI Tools)</a></div>
         <div class="da-row" data-k="Networking for Job Search"><a href="010-networking-for-job-search.html">Networking for Job Search</a></div>
         <div class="da-row" data-k="Post-Interview Strategy"><a href="015-post-interview-strategy.html">Post-Interview Strategy</a></div>
         <div class="da-row" data-k="Practicing Interviews with AI (Gemini Live)"><a href="014-practicing-interviews-with-ai-gemini-live.html">Practicing Interviews with AI (Gemini Live)</a></div>
         <div class="da-row" data-k="STAR Method (Behavioral Interview)"><a href="012-star-method-behavioral-interview.html">STAR Method (Behavioral Interview)</a></div>
         <div class="da-row" data-k="Tailoring Your Resume"><a href="005-tailoring-your-resume.html">Tailoring Your Resume</a></div>
         <div class="da-row" data-k="Transferable Skills"><a href="001-transferable-skills.html">Transferable Skills</a></div>
         <div class="da-row" data-k="Using AI (NotebookLM) for Interview Preparation"><a href="013-using-ai-notebooklm-for-interview-preparation.html">Using AI (NotebookLM) for Interview Preparation</a></div>
         <div class="da-row" data-k="Using AI to Improve and Tailor Your Resume"><a href="006-using-ai-to-improve-and-tailor-your-resume.html">Using AI to Improve and Tailor Your Resume</a></div>
