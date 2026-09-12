:orphan:

Control edge cases
==================

.. gallery-grid::
   :interactive:
   :filter-fields: tags
   :sort-fields: title,duration,published
   :grid-reverse:

   - title: Café Z
     tags: [one, one]
     duration: 1.2
     published: '2026-01-01T01:00:00+02:00'
   - title: Alpha
     tags: [two]
     duration: 1.11
     published: '2026-01-01T00:00:00+00:00'
   - title: Missing

.. gallery-grid::
   :interactive:

   - title: Outer
     content: |
       .. gallery-grid::
          :interactive:

          - title: Inner one
          - title: Inner two
