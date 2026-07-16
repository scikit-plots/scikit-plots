:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-viz-index:

:raw-html:`<div style="text-align:center"><strong>` 🎨 Data Visualization
|br| |full_version| - |today|
:raw-html:`</strong></div>`

========================================================================
🎨 Data Visualization
========================================================================

*Section 6 of the Data Analytics hub — 27 of 27 lessons.*

Turning results into visuals that inform: chart choice, design principles, and honest, accessible graphics.

:doc:`↑ Back to the Data Analytics hub <../index>`

.. raw:: html

   <div style="text-align:center;margin:0.4rem 0 0.4rem">
   <input id="term-filter" type="search" autocomplete="off" spellcheck="false"
           placeholder="&#128269;&nbsp; Type to filter this section 27 lessons &mdash; by title or keyword&hellip;"
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
        if(cnt){{cnt.textContent=(q&&az)?(n+' of 27 match'+(n===1?'':'s')):'';}}
     });
   });
   </script>


.. dropdown:: 🎨 Visualization Principles & Design
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   What makes a visual work: purpose, art elements, chart choice, and accessibility.

   * :doc:`001 · Data Visualization — representing data graphically so patterns and meaning become visible at a glance <001-data-visualization>`
   * :doc:`002 · Connecting Data and Images — how a visualization maps data values to visual properties — the encoding that carries meaning <002-connecting-data-and-images>`
   * :doc:`003 · Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose — what makes a visualization effective — a clear focus, sound structure, and a purpose <003-creating-powerful-data-visualizations-focus-structure-and-analytical-purpose>`
   * :doc:`004 · Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity — fixed images versus interactive views — when each serves, and their trade-offs <004-static-vs-dynamic-data-visualizations-design-tradeoffs-control-and-interactivity>`
   * :doc:`005 · Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement — the visual building blocks — line, shape, colour, space, movement — applied to charts <005-elements-of-art-in-data-visualization-line-shape-color-space-and-movement>`
   * :doc:`006 · Choosing the Right Visualization: Audience-Centered Design and Chart Selection — matching chart type to the data, the message, and the audience <006-choosing-the-right-visualization-audience-centered-design-and-chart-selection>`
   * :doc:`007 · Design Thinking in Data Visualization: A User-Centered Framework — designing visualizations around the viewer's needs — a user-centered process <007-design-thinking-in-data-visualization-a-user-centered-framework>`
   * :doc:`008 · Accessibility in Data Visualization: Designing for Everyone — making charts readable by everyone, including people with visual differences <008-accessibility-in-data-visualization-designing-for-everyone>`

.. dropdown:: 📊 Tableau
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   From first steps in Tableau Public to creative, linked, effective visuals.

   * :doc:`009 · Introduction to Tableau — what Tableau is and why analysts use it — visual analytics without heavy coding <009-introduction-to-tableau>`
   * :doc:`010 · Getting Started with Tableau Public — the free version: connecting data, building a first view, publishing to the web <010-getting-started-with-tableau-public>`
   * :doc:`011 · Creating a CO₂ Emissions Visualization in Tableau Public — a worked Tableau example — turning an emissions dataset into a clear visualization <011-creating-a-co-emissions-visualization-in-tableau-public>`
   * :doc:`012 · Effective vs. Ineffective Data Visualizations in Tableau — the difference in practice — what makes a Tableau chart clear versus confusing <012-effective-vs-ineffective-data-visualizations-in-tableau>`
   * :doc:`013 · Using Creativity in Tableau — going beyond default charts — custom, expressive visualizations that still communicate <013-using-creativity-in-tableau>`
   * :doc:`014 · Linking Multiple Datasets in Tableau Public — combining data sources inside Tableau — joins and relationships for richer views <014-linking-multiple-datasets-in-tableau-public>`

.. dropdown:: 📖 Storytelling & Dashboards
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Turning findings into narrative, key messages, dashboards, and focused views.

   * :doc:`015 · Data Storytelling: Giving Numbers a Clear and Convincing Voice — wrapping data in narrative so it informs, persuades, and is remembered <015-data-storytelling-giving-numbers-a-clear-and-convincing-voice>`
   * :doc:`016 · Engaging Your Audience in Data Storytelling: Identifying the Key Message — finding the one message that matters and building the story around it <016-engaging-your-audience-in-data-storytelling-identifying-the-key-message>`
   * :doc:`017 · Data Dashboards: Organizing Insight for Real-Time Decision Making — combining key visualizations into one monitored view for ongoing decisions <017-data-dashboards-organizing-insight-for-real-time-decision-making>`
   * :doc:`018 · Using Filters to Create Compelling and Focused Visuals — filtering to sharpen a visualization's message and enable exploration <018-using-filters-to-create-compelling-and-focused-visuals>`

.. dropdown:: 🎤 Presentations & Q&A
   :animate: fade-in-slide-down
   :class-container: sd-dropdown

   Structuring persuasive presentations, slide design, delivery, and handling Q&A.

   * :doc:`019 · Structuring a Persuasive Data Presentation: Turning Insights into Story — arranging findings into a presentation that carries an audience to a conclusion <019-structuring-a-persuasive-data-presentation-turning-insights-into-story>`
   * :doc:`020 · Designing Effective Data Presentation Slides: Structure, Visuals, and Professional Impact — slide design that supports the message — clear, visual, uncluttered, professional <020-designing-effective-data-presentation-slides-structure-visuals-and-professional-impact>`
   * :doc:`021 · Using a Strategic Framework to Structure Data Presentations — reusable structures that reliably organize a presentation toward its goal <021-using-a-strategic-framework-to-structure-data-presentations>`
   * :doc:`022 · Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method — integrating data with narrative — hypothesis, context, and a four-element method <022-weaving-data-into-presentations-hypotheses-context-and-the-mccandless-method>`
   * :doc:`023 · Presentation Skills for Data Analysts: Delivering Insights with Confidence — the delivery half of presenting — voice, pace, presence, and handling nerves <023-presentation-skills-for-data-analysts-delivering-insights-with-confidence>`
   * :doc:`024 · Presenting Like a Pro: Best Practices for Data Analysts — the practices that mark a polished, credible data presenter <024-presenting-like-a-pro-best-practices-for-data-analysts>`
   * :doc:`025 · Preparing for Q&A: Anticipating and Responding to Stakeholder Questions — readying for the questions a presentation will draw — anticipation as preparation <025-preparing-for-q-and-a-anticipating-and-responding-to-stakeholder-questions>`
   * :doc:`026 · Handling Objections in Data Presentations: Responding with Confidence and Clarity — responding to challenges and pushback without defensiveness, honestly and calmly <026-handling-objections-in-data-presentations-responding-with-confidence-and-clarity>`
   * :doc:`027 · Q&A Best Practices: Answering Questions with Clarity and Confidence — the habits of fielding questions well — listen, answer directly, stay honest <027-q-and-a-best-practices-answering-questions-with-clarity-and-confidence>`

🔤 Every lesson, A–Z index
---------------------------

.. dropdown:: 🔠 Open the full alphabetical index
   :animate: fade-in-slide-down
   :class-container: term-az

   .. hlist::
      :columns: 2

      * :doc:`Accessibility in Data Visualization: Designing for Everyone <008-accessibility-in-data-visualization-designing-for-everyone>`
      * :doc:`Choosing the Right Visualization: Audience-Centered Design and Chart Selection <006-choosing-the-right-visualization-audience-centered-design-and-chart-selection>`
      * :doc:`Connecting Data and Images <002-connecting-data-and-images>`
      * :doc:`Creating a CO₂ Emissions Visualization in Tableau Public <011-creating-a-co-emissions-visualization-in-tableau-public>`
      * :doc:`Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose <003-creating-powerful-data-visualizations-focus-structure-and-analytical-purpose>`
      * :doc:`Data Dashboards: Organizing Insight for Real-Time Decision Making <017-data-dashboards-organizing-insight-for-real-time-decision-making>`
      * :doc:`Data Storytelling: Giving Numbers a Clear and Convincing Voice <015-data-storytelling-giving-numbers-a-clear-and-convincing-voice>`
      * :doc:`Data Visualization <001-data-visualization>`
      * :doc:`Design Thinking in Data Visualization: A User-Centered Framework <007-design-thinking-in-data-visualization-a-user-centered-framework>`
      * :doc:`Designing Effective Data Presentation Slides: Structure, Visuals, and Professional Impact <020-designing-effective-data-presentation-slides-structure-visuals-and-professional-impact>`
      * :doc:`Effective vs. Ineffective Data Visualizations in Tableau <012-effective-vs-ineffective-data-visualizations-in-tableau>`
      * :doc:`Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement <005-elements-of-art-in-data-visualization-line-shape-color-space-and-movement>`
      * :doc:`Engaging Your Audience in Data Storytelling: Identifying the Key Message <016-engaging-your-audience-in-data-storytelling-identifying-the-key-message>`
      * :doc:`Getting Started with Tableau Public <010-getting-started-with-tableau-public>`
      * :doc:`Handling Objections in Data Presentations: Responding with Confidence and Clarity <026-handling-objections-in-data-presentations-responding-with-confidence-and-clarity>`
      * :doc:`Introduction to Tableau <009-introduction-to-tableau>`
      * :doc:`Linking Multiple Datasets in Tableau Public <014-linking-multiple-datasets-in-tableau-public>`
      * :doc:`Preparing for Q&A: Anticipating and Responding to Stakeholder Questions <025-preparing-for-q-and-a-anticipating-and-responding-to-stakeholder-questions>`
      * :doc:`Presentation Skills for Data Analysts: Delivering Insights with Confidence <023-presentation-skills-for-data-analysts-delivering-insights-with-confidence>`
      * :doc:`Presenting Like a Pro: Best Practices for Data Analysts <024-presenting-like-a-pro-best-practices-for-data-analysts>`
      * :doc:`Q&A Best Practices: Answering Questions with Clarity and Confidence <027-q-and-a-best-practices-answering-questions-with-clarity-and-confidence>`
      * :doc:`Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity <004-static-vs-dynamic-data-visualizations-design-tradeoffs-control-and-interactivity>`
      * :doc:`Structuring a Persuasive Data Presentation: Turning Insights into Story <019-structuring-a-persuasive-data-presentation-turning-insights-into-story>`
      * :doc:`Using a Strategic Framework to Structure Data Presentations <021-using-a-strategic-framework-to-structure-data-presentations>`
      * :doc:`Using Creativity in Tableau <013-using-creativity-in-tableau>`
      * :doc:`Using Filters to Create Compelling and Focused Visuals <018-using-filters-to-create-compelling-and-focused-visuals>`
      * :doc:`Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method <022-weaving-data-into-presentations-hypotheses-context-and-the-mccandless-method>`

.. toctree::
   :hidden:
   :includehidden:
   :maxdepth: 1

   001-data-visualization
   002-connecting-data-and-images
   003-creating-powerful-data-visualizations-focus-structure-and-analytical-purpose
   004-static-vs-dynamic-data-visualizations-design-tradeoffs-control-and-interactivity
   005-elements-of-art-in-data-visualization-line-shape-color-space-and-movement
   006-choosing-the-right-visualization-audience-centered-design-and-chart-selection
   007-design-thinking-in-data-visualization-a-user-centered-framework
   008-accessibility-in-data-visualization-designing-for-everyone
   009-introduction-to-tableau
   010-getting-started-with-tableau-public
   011-creating-a-co-emissions-visualization-in-tableau-public
   012-effective-vs-ineffective-data-visualizations-in-tableau
   013-using-creativity-in-tableau
   014-linking-multiple-datasets-in-tableau-public
   015-data-storytelling-giving-numbers-a-clear-and-convincing-voice
   016-engaging-your-audience-in-data-storytelling-identifying-the-key-message
   017-data-dashboards-organizing-insight-for-real-time-decision-making
   018-using-filters-to-create-compelling-and-focused-visuals
   019-structuring-a-persuasive-data-presentation-turning-insights-into-story
   020-designing-effective-data-presentation-slides-structure-visuals-and-professional-impact
   021-using-a-strategic-framework-to-structure-data-presentations
   022-weaving-data-into-presentations-hypotheses-context-and-the-mccandless-method
   023-presentation-skills-for-data-analysts-delivering-insights-with-confidence
   024-presenting-like-a-pro-best-practices-for-data-analysts
   025-preparing-for-q-and-a-anticipating-and-responding-to-stakeholder-questions
   026-handling-objections-in-data-presentations-responding-with-confidence-and-clarity
   027-q-and-a-best-practices-answering-questions-with-clarity-and-confidence

.. tags:: purpose: reference, topic: data analytics
