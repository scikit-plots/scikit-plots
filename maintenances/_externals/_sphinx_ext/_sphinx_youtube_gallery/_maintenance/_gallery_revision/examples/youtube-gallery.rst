YouTube gallery examples
========================

Enable ``_sphinx_ext._sphinx_youtube_gallery`` or its fully qualified
``scikitplot._externals`` equivalent. Preview over HTTP.

Standalone player
-----------------

.. youtube:: JXtISpdDPNY

Arbitrary authored card body
----------------------------

Use raw ``gallery-grid`` when the body markup itself is custom content rather
than typed YouTube data.

.. gallery-grid::
   :grid-columns: 1 1 2 2

   - title: PCA tutorial
     content: |
       .. youtube:: JXtISpdDPNY
   - title: PCA in a dropdown
     content: |
       .. admonition:: Watch the lesson
          :class: dropdown

          .. youtube:: JXtISpdDPNY

Typed video gallery
-------------------

``fields:`` supplies site-specific gallery metadata without adding visible card
prose. ``channel`` is the human label; ``handle`` is linkable channel identity.

.. youtube-gallery::
   :mode: embed
   :interactive:
   :filter-fields: category
   :sort-fields: title,published

   videos:
     - id: JXtISpdDPNY
       title: Principal Component Analysis in Python
       description: A searchable tutorial description from Statistics Globe.
       channel: Statistics Globe
       handle: StatisticsGlobe
       published: 2024-03-01T10:00:00Z
       tags: [pca, dimensionality-reduction]
       fields:
         category: Statistics

Channel exploration from video data
-----------------------------------

The same typed video shape can be projected to title-only clickable channel
cards offline. No API lookup is required when ``channel_id`` or ``handle`` is
already reviewed in the catalog.

.. youtube-gallery::
   :view: channels
   :interactive:
   :sort-fields: title,video_count,published

   videos:
     - id: JXtISpdDPNY
       title: Principal Component Analysis in Python
       channel: Statistics Globe
       handle: StatisticsGlobe
       published: 2024-03-01T10:00:00Z
     - id: dQw4w9WgXcQ
       title: Second reviewed video
       channel: Statistics Globe
       handle: StatisticsGlobe
       published: 2025-01-15T10:00:00Z
