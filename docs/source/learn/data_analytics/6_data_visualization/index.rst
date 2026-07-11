.. _da-viz-index:

========================================================================
🎨 Data Visualization
========================================================================

*Section 6 of the Data Analytics hub — 27 of 27 lessons.*

Turning results into visuals that inform: chart choice, design principles, and honest, accessible graphics.

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


.. dropdown:: 🎨 Visualization Principles & Design
   :class-container: sd-dropdown

   What makes a visual work: purpose, art elements, chart choice, and accessibility.

   .. raw:: html

         <div class="da-row" data-k="Data Visualization "><a href="001-data-visualization.html">001 · Data Visualization</a></div>
         <div class="da-row" data-k="Connecting Data and Images "><a href="002-connecting-data-and-images.html">002 · Connecting Data and Images</a></div>
         <div class="da-row" data-k="Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose "><a href="003-creating-powerful-data-visualizations-focus-structure-and-analytical-purpose.html">003 · Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose</a></div>
         <div class="da-row" data-k="Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity "><a href="004-static-vs-dynamic-data-visualizations-design-tradeoffs-control-and-interactivity.html">004 · Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity</a></div>
         <div class="da-row" data-k="Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement "><a href="005-elements-of-art-in-data-visualization-line-shape-color-space-and-movement.html">005 · Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement</a></div>
         <div class="da-row" data-k="Choosing the Right Visualization: Audience-Centered Design and Chart Selection "><a href="006-choosing-the-right-visualization-audience-centered-design-and-chart-selection.html">006 · Choosing the Right Visualization: Audience-Centered Design and Chart Selection</a></div>
         <div class="da-row" data-k="Design Thinking in Data Visualization: A User-Centered Framework "><a href="007-design-thinking-in-data-visualization-a-user-centered-framework.html">007 · Design Thinking in Data Visualization: A User-Centered Framework</a></div>
         <div class="da-row" data-k="Accessibility in Data Visualization: Designing for Everyone "><a href="008-accessibility-in-data-visualization-designing-for-everyone.html">008 · Accessibility in Data Visualization: Designing for Everyone</a></div>

.. dropdown:: 📊 Tableau
   :class-container: sd-dropdown

   From first steps in Tableau Public to creative, linked, effective visuals.

   .. raw:: html

         <div class="da-row" data-k="Introduction to Tableau "><a href="009-introduction-to-tableau.html">009 · Introduction to Tableau</a></div>
         <div class="da-row" data-k="Getting Started with Tableau Public "><a href="010-getting-started-with-tableau-public.html">010 · Getting Started with Tableau Public</a></div>
         <div class="da-row" data-k="Creating a CO₂ Emissions Visualization in Tableau Public "><a href="011-creating-a-co-emissions-visualization-in-tableau-public.html">011 · Creating a CO₂ Emissions Visualization in Tableau Public</a></div>
         <div class="da-row" data-k="Effective vs. Ineffective Data Visualizations in Tableau "><a href="012-effective-vs-ineffective-data-visualizations-in-tableau.html">012 · Effective vs. Ineffective Data Visualizations in Tableau</a></div>
         <div class="da-row" data-k="Using Creativity in Tableau "><a href="013-using-creativity-in-tableau.html">013 · Using Creativity in Tableau</a></div>
         <div class="da-row" data-k="Linking Multiple Datasets in Tableau Public "><a href="014-linking-multiple-datasets-in-tableau-public.html">014 · Linking Multiple Datasets in Tableau Public</a></div>

.. dropdown:: 📖 Storytelling & Dashboards
   :class-container: sd-dropdown

   Turning findings into narrative, key messages, dashboards, and focused views.

   .. raw:: html

         <div class="da-row" data-k="Data Storytelling: Giving Numbers a Clear and Convincing Voice "><a href="015-data-storytelling-giving-numbers-a-clear-and-convincing-voice.html">015 · Data Storytelling: Giving Numbers a Clear and Convincing Voice</a></div>
         <div class="da-row" data-k="Engaging Your Audience in Data Storytelling: Identifying the Key Message "><a href="016-engaging-your-audience-in-data-storytelling-identifying-the-key-message.html">016 · Engaging Your Audience in Data Storytelling: Identifying the Key Message</a></div>
         <div class="da-row" data-k="Data Dashboards: Organizing Insight for Real-Time Decision Making "><a href="017-data-dashboards-organizing-insight-for-real-time-decision-making.html">017 · Data Dashboards: Organizing Insight for Real-Time Decision Making</a></div>
         <div class="da-row" data-k="Using Filters to Create Compelling and Focused Visuals "><a href="018-using-filters-to-create-compelling-and-focused-visuals.html">018 · Using Filters to Create Compelling and Focused Visuals</a></div>

.. dropdown:: 🎤 Presentations & Q&A
   :class-container: sd-dropdown

   Structuring persuasive presentations, slide design, delivery, and handling Q&A.

   .. raw:: html

         <div class="da-row" data-k="Structuring a Persuasive Data Presentation: Turning Insights into Story "><a href="019-structuring-a-persuasive-data-presentation-turning-insights-into-story.html">019 · Structuring a Persuasive Data Presentation: Turning Insights into Story</a></div>
         <div class="da-row" data-k="Designing Effective Data Presentation Slides: Structure, Visuals, and Professional Impact "><a href="020-designing-effective-data-presentation-slides-structure-visuals-and-professional-impact.html">020 · Designing Effective Data Presentation Slides: Structure, Visuals, and Professional Impact</a></div>
         <div class="da-row" data-k="Using a Strategic Framework to Structure Data Presentations "><a href="021-using-a-strategic-framework-to-structure-data-presentations.html">021 · Using a Strategic Framework to Structure Data Presentations</a></div>
         <div class="da-row" data-k="Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method "><a href="022-weaving-data-into-presentations-hypotheses-context-and-the-mccandless-method.html">022 · Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method</a></div>
         <div class="da-row" data-k="Presentation Skills for Data Analysts: Delivering Insights with Confidence "><a href="023-presentation-skills-for-data-analysts-delivering-insights-with-confidence.html">023 · Presentation Skills for Data Analysts: Delivering Insights with Confidence</a></div>
         <div class="da-row" data-k="Presenting Like a Pro: Best Practices for Data Analysts "><a href="024-presenting-like-a-pro-best-practices-for-data-analysts.html">024 · Presenting Like a Pro: Best Practices for Data Analysts</a></div>
         <div class="da-row" data-k="Preparing for Q&A: Anticipating and Responding to Stakeholder Questions "><a href="025-preparing-for-q-and-a-anticipating-and-responding-to-stakeholder-questions.html">025 · Preparing for Q&A: Anticipating and Responding to Stakeholder Questions</a></div>
         <div class="da-row" data-k="Handling Objections in Data Presentations: Responding with Confidence and Clarity "><a href="026-handling-objections-in-data-presentations-responding-with-confidence-and-clarity.html">026 · Handling Objections in Data Presentations: Responding with Confidence and Clarity</a></div>
         <div class="da-row" data-k="Q&A Best Practices: Answering Questions with Clarity and Confidence "><a href="027-q-and-a-best-practices-answering-questions-with-clarity-and-confidence.html">027 · Q&A Best Practices: Answering Questions with Clarity and Confidence</a></div>

.. dropdown:: 🔠 A–Z index
   :class-container: term-az

   .. raw:: html

         <div class="da-row" data-k="Accessibility in Data Visualization: Designing for Everyone"><a href="008-accessibility-in-data-visualization-designing-for-everyone.html">Accessibility in Data Visualization: Designing for Everyone</a></div>
         <div class="da-row" data-k="Choosing the Right Visualization: Audience-Centered Design and Chart Selection"><a href="006-choosing-the-right-visualization-audience-centered-design-and-chart-selection.html">Choosing the Right Visualization: Audience-Centered Design and Chart Selection</a></div>
         <div class="da-row" data-k="Connecting Data and Images"><a href="002-connecting-data-and-images.html">Connecting Data and Images</a></div>
         <div class="da-row" data-k="Creating a CO₂ Emissions Visualization in Tableau Public"><a href="011-creating-a-co-emissions-visualization-in-tableau-public.html">Creating a CO₂ Emissions Visualization in Tableau Public</a></div>
         <div class="da-row" data-k="Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose"><a href="003-creating-powerful-data-visualizations-focus-structure-and-analytical-purpose.html">Creating Powerful Data Visualizations: Focus, Structure, and Analytical Purpose</a></div>
         <div class="da-row" data-k="Data Dashboards: Organizing Insight for Real-Time Decision Making"><a href="017-data-dashboards-organizing-insight-for-real-time-decision-making.html">Data Dashboards: Organizing Insight for Real-Time Decision Making</a></div>
         <div class="da-row" data-k="Data Storytelling: Giving Numbers a Clear and Convincing Voice"><a href="015-data-storytelling-giving-numbers-a-clear-and-convincing-voice.html">Data Storytelling: Giving Numbers a Clear and Convincing Voice</a></div>
         <div class="da-row" data-k="Data Visualization"><a href="001-data-visualization.html">Data Visualization</a></div>
         <div class="da-row" data-k="Design Thinking in Data Visualization: A User-Centered Framework"><a href="007-design-thinking-in-data-visualization-a-user-centered-framework.html">Design Thinking in Data Visualization: A User-Centered Framework</a></div>
         <div class="da-row" data-k="Designing Effective Data Presentation Slides: Structure, Visuals, and Professional Impact"><a href="020-designing-effective-data-presentation-slides-structure-visuals-and-professional-impact.html">Designing Effective Data Presentation Slides: Structure, Visuals, and Professional Impact</a></div>
         <div class="da-row" data-k="Effective vs. Ineffective Data Visualizations in Tableau"><a href="012-effective-vs-ineffective-data-visualizations-in-tableau.html">Effective vs. Ineffective Data Visualizations in Tableau</a></div>
         <div class="da-row" data-k="Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement"><a href="005-elements-of-art-in-data-visualization-line-shape-color-space-and-movement.html">Elements of Art in Data Visualization: Line, Shape, Color, Space, and Movement</a></div>
         <div class="da-row" data-k="Engaging Your Audience in Data Storytelling: Identifying the Key Message"><a href="016-engaging-your-audience-in-data-storytelling-identifying-the-key-message.html">Engaging Your Audience in Data Storytelling: Identifying the Key Message</a></div>
         <div class="da-row" data-k="Getting Started with Tableau Public"><a href="010-getting-started-with-tableau-public.html">Getting Started with Tableau Public</a></div>
         <div class="da-row" data-k="Handling Objections in Data Presentations: Responding with Confidence and Clarity"><a href="026-handling-objections-in-data-presentations-responding-with-confidence-and-clarity.html">Handling Objections in Data Presentations: Responding with Confidence and Clarity</a></div>
         <div class="da-row" data-k="Introduction to Tableau"><a href="009-introduction-to-tableau.html">Introduction to Tableau</a></div>
         <div class="da-row" data-k="Linking Multiple Datasets in Tableau Public"><a href="014-linking-multiple-datasets-in-tableau-public.html">Linking Multiple Datasets in Tableau Public</a></div>
         <div class="da-row" data-k="Preparing for Q&A: Anticipating and Responding to Stakeholder Questions"><a href="025-preparing-for-q-and-a-anticipating-and-responding-to-stakeholder-questions.html">Preparing for Q&A: Anticipating and Responding to Stakeholder Questions</a></div>
         <div class="da-row" data-k="Presentation Skills for Data Analysts: Delivering Insights with Confidence"><a href="023-presentation-skills-for-data-analysts-delivering-insights-with-confidence.html">Presentation Skills for Data Analysts: Delivering Insights with Confidence</a></div>
         <div class="da-row" data-k="Presenting Like a Pro: Best Practices for Data Analysts"><a href="024-presenting-like-a-pro-best-practices-for-data-analysts.html">Presenting Like a Pro: Best Practices for Data Analysts</a></div>
         <div class="da-row" data-k="Q&A Best Practices: Answering Questions with Clarity and Confidence"><a href="027-q-and-a-best-practices-answering-questions-with-clarity-and-confidence.html">Q&A Best Practices: Answering Questions with Clarity and Confidence</a></div>
         <div class="da-row" data-k="Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity"><a href="004-static-vs-dynamic-data-visualizations-design-tradeoffs-control-and-interactivity.html">Static vs. Dynamic Data Visualizations: Design Tradeoffs, Control, and Interactivity</a></div>
         <div class="da-row" data-k="Structuring a Persuasive Data Presentation: Turning Insights into Story"><a href="019-structuring-a-persuasive-data-presentation-turning-insights-into-story.html">Structuring a Persuasive Data Presentation: Turning Insights into Story</a></div>
         <div class="da-row" data-k="Using a Strategic Framework to Structure Data Presentations"><a href="021-using-a-strategic-framework-to-structure-data-presentations.html">Using a Strategic Framework to Structure Data Presentations</a></div>
         <div class="da-row" data-k="Using Creativity in Tableau"><a href="013-using-creativity-in-tableau.html">Using Creativity in Tableau</a></div>
         <div class="da-row" data-k="Using Filters to Create Compelling and Focused Visuals"><a href="018-using-filters-to-create-compelling-and-focused-visuals.html">Using Filters to Create Compelling and Focused Visuals</a></div>
         <div class="da-row" data-k="Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method"><a href="022-weaving-data-into-presentations-hypotheses-context-and-the-mccandless-method.html">Weaving Data into Presentations: Hypotheses, Context, and the McCandless Method</a></div>
