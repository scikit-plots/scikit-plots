:orphan:

Inline YouTube gallery
======================

.. youtube-gallery::
   :collection-id: inline-videos
   :grid-columns: 1 1 2 2
   :interactive:
   :grid-gutter: 3
   :card-shadow: sm
   :video-width: 100%
   :video-aspect: 16:9
   :filter-fields: channel,tags
   :sort-fields: title,published,duration

   videos:
     - id: JXtISpdDPNY
       title: Principal Component Analysis in Python
       channel: Statistics Globe
       tags: [pca, dimensionality-reduction]
     - id: UNzCG3lw6O0
       title: Building Great Agent Skills
       tags: [agents, skills]

File-backed YouTube gallery
---------------------------

.. youtube-gallery:: youtube.yaml
   :collection-id: catalog-videos
   :grid-columns: 1 1 2 2
   :group-by: channel
   :sort: -published
   :interactive:
   :grid-gutter: 3
   :card-shadow: sm
   :video-width: 100%
   :video-aspect: 16:9
   :filter-fields: channel,tags
