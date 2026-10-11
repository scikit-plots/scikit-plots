.. _seaborn-data-semantics-and-styling:

Data semantics and styling
==========================

These functions deliberately accept seaborn-style arguments, but the wrapper
owns model-specific preprocessing before delegating to plotting machinery.
Keep data semantics separate from visual style:

* first validate x/y/hue and class/score meaning;
* then choose ``kind`` and visual options;
* retain the returned Matplotlib object when later composition/customization is
  required; and
* save figures at the application boundary rather than burying file writes in
  reusable data-preparation code.
