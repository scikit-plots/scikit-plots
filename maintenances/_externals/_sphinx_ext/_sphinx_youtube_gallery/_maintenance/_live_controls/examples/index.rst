:html_theme.sidebar_secondary.remove:

.. _resource-youtube-index:

YouTube Resources
=================

Search channels or explore videos by topic. Both sections use ``youtube-gallery``
as a typed YouTube adapter over the same ``gallery-grid`` UI engine.

Channels
--------

.. youtube-gallery::
   :grid-columns: 1 2 3 4
   :class-card: downstream-project-links
   :sort: title
   :interactive:
   :collection-id: youtube-channels
   :search-label: Search channels
   :search-fields: description

   channels:
     - handle: aiDotEngineer
       title: '@aiDotEngineer'
       description: AI engineering and LLM applications
     - handle: anthropic-ai
       title: '@anthropic-ai'
       description: Official Anthropic channel
     - handle: claude
       title: '@claude'
       description: Claude Channel
     - handle: codebasics
       title: '@codebasics'
       description: Data structures, Python, and ML fundamentals
     - handle: DataTalksClub
       title: '@DataTalksClub'
       description: Data engineering, ML ops, and LLM engineering
     - handle: EmergentMindAI
       title: '@EmergentMindAI'
       description: AI research summaries and trends
     - handle: mattpocockuk
       title: '@mattpocockuk'
       description: TypeScript, web development, and AI tools
     - handle: StatisticsGlobe
       title: '@StatisticsGlobe'
       description: Statistics and data visualization tutorials

Videos
------

.. youtube-gallery::
   :grid-columns: 1 1 2 2
   :group-by: category
   :interactive:
   :filter-fields: category
   :sort-fields: title,category
   :collection-id: youtube-videos
   :search-label: Search videos

   videos:
     - id: UNzCG3lw6O0
       title: Building Great Agent Skills
       fields:
         category: Agent Skills & Design Patterns
     - id: CEvIs9y1uog
       title: Don't Build Agents, Build Skills Instead
       fields:
         category: Agent Skills & Design Patterns
     - id: TRjq7t2Ms5I
       title: Building Production-Ready RAG Applications
       fields:
         category: AI Engineering & Production
     - id: knDDGYHnnSI
       title: Marriage of Knowledge Graphs and RAG
       fields:
         category: AI Engineering & Production
     - id: 5HP4DQZJkNQ
       title: Ship Production Code 2x Faster With AI
       fields:
         category: AI Engineering & Production
     - id: rxy_9I8967Q
       title: Can Language Models Rebuild Production Systems?
       fields:
         category: AI Strategy & Thought Leadership
     - id: 4szRHy_CT7s
       title: Why Do We Need AI Fluency?
       fields:
         category: AI Strategy & Thought Leadership
     - id: wB9C0Mz9gSo
       title: Advanced Matplotlib Techniques
       fields:
         category: Data Science & Visualization
     - id: 3Fp1zn5ao2M
       title: Introduction to NumPy and Matplotlib
       fields:
         category: Data Science & Visualization
     - id: qqwf4Vuj8oM
       title: Matplotlib Fundamentals Tutorial
       fields:
         category: Data Science & Visualization
     - id: aL_zQfi7lDI
       title: Machine Learning Design Patterns
       fields:
         category: Data Science & Visualization
     - id: JXtISpdDPNY
       title: Principal Component Analysis (PCA) in Python
       fields:
         category: Machine Learning & Algorithms
     - id: mNpBrHwOCt4
       title: Principal Component Analysis (PCA) in R
       fields:
         category: Machine Learning & Algorithms
