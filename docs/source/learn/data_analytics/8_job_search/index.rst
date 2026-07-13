:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

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


.. dropdown:: 🧭 Career Identity & Planning
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Transferable skills, a career identity statement, and an AI-assisted search plan.

   * :doc:`001 · Transferable Skills — abilities that carry across roles and industries — the foundation of a career pivot <001-transferable-skills>`
   * :doc:`002 · Career Identity Statement — a concise statement of the unique value you bring to the workforce <002-career-identity-statement>`
   * :doc:`003 · Career Dreamer (AI Tool for Career Exploration) — an experimental Grow with Google AI tool that maps your experience to career paths <003-career-dreamer-ai-tool-for-career-exploration>`
   * :doc:`004 · Job Search Plan (Using AI Tools) — a structured, deliberate plan for the job search, assisted by AI tools <004-job-search-plan-using-ai-tools>`

.. dropdown:: 📄 Resume, Brand & Applications
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Tailoring resumes, building an online presence, platforms, tracking, networking.

   * :doc:`005 · Tailoring Your Resume — adapting a resume to each specific role rather than sending one generic version <005-tailoring-your-resume>`
   * :doc:`006 · Using AI to Improve and Tailor Your Resume — using AI assistants to draft, refine, and tailor a resume — with human judgement <006-using-ai-to-improve-and-tailor-your-resume>`
   * :doc:`007 · Building a Professional Online Presence (Personal Brand) — shaping how you appear online — profile, presence, and personal brand <007-building-a-professional-online-presence-personal-brand>`
   * :doc:`008 · Choosing the Right Job Platforms — selecting the job platforms and channels that fit your target roles <008-choosing-the-right-job-platforms>`
   * :doc:`009 · Job Application Tracking (Using AI + Spreadsheets) — systematically recording applications and their status, with AI and spreadsheets <009-job-application-tracking-using-ai-spreadsheets>`
   * :doc:`010 · Networking for Job Search — building and using professional relationships to find and reach opportunities <010-networking-for-job-search>`

.. dropdown:: 🎯 Interviews & Follow-Up
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Interview prep, the STAR method, AI practice tools, and post-interview strategy.

   * :doc:`011 · Interview Preparation — readying for interviews — research, practice, and knowing your material <011-interview-preparation>`
   * :doc:`012 · STAR Method (Behavioral Interview) — a structure for behavioural answers — Situation, Task, Action, Result <012-star-method-behavioral-interview>`
   * :doc:`013 · Using AI (NotebookLM) for Interview Preparation — using a source-grounded AI research tool to prepare from role and company material <013-using-ai-notebooklm-for-interview-preparation>`
   * :doc:`014 · Practicing Interviews with AI (Gemini Live) — rehearsing interviews aloud in real-time voice conversation with an AI <014-practicing-interviews-with-ai-gemini-live>`
   * :doc:`015 · Post-Interview Strategy — what to do after an interview — follow up, reflect, and handle the outcome <015-post-interview-strategy>`

🔤 Every lesson, A–Z
---------------------

.. dropdown:: 🔠 A–Z index
   :animate: fade-in-slide-down
   :class-container: term-az

   .. hlist::
      :columns: 2

   * :doc:`Building a Professional Online Presence (Personal Brand) <007-building-a-professional-online-presence-personal-brand>`
   * :doc:`Career Dreamer (AI Tool for Career Exploration) <003-career-dreamer-ai-tool-for-career-exploration>`
   * :doc:`Career Identity Statement <002-career-identity-statement>`
   * :doc:`Choosing the Right Job Platforms <008-choosing-the-right-job-platforms>`
   * :doc:`Interview Preparation <011-interview-preparation>`
   * :doc:`Job Application Tracking (Using AI + Spreadsheets) <009-job-application-tracking-using-ai-spreadsheets>`
   * :doc:`Job Search Plan (Using AI Tools) <004-job-search-plan-using-ai-tools>`
   * :doc:`Networking for Job Search <010-networking-for-job-search>`
   * :doc:`Post-Interview Strategy <015-post-interview-strategy>`
   * :doc:`Practicing Interviews with AI (Gemini Live) <014-practicing-interviews-with-ai-gemini-live>`
   * :doc:`STAR Method (Behavioral Interview) <012-star-method-behavioral-interview>`
   * :doc:`Tailoring Your Resume <005-tailoring-your-resume>`
   * :doc:`Transferable Skills <001-transferable-skills>`
   * :doc:`Using AI (NotebookLM) for Interview Preparation <013-using-ai-notebooklm-for-interview-preparation>`
   * :doc:`Using AI to Improve and Tailor Your Resume <006-using-ai-to-improve-and-tailor-your-resume>`

.. toctree::
   :hidden:
   :maxdepth: 1

   001-transferable-skills
   002-career-identity-statement
   003-career-dreamer-ai-tool-for-career-exploration
   004-job-search-plan-using-ai-tools
   005-tailoring-your-resume
   006-using-ai-to-improve-and-tailor-your-resume
   007-building-a-professional-online-presence-personal-brand
   008-choosing-the-right-job-platforms
   009-job-application-tracking-using-ai-spreadsheets
   010-networking-for-job-search
   011-interview-preparation
   012-star-method-behavioral-interview
   013-using-ai-notebooklm-for-interview-preparation
   014-practicing-interviews-with-ai-gemini-live
   015-post-interview-strategy

.. tags:: purpose: reference, topic: data analytics
