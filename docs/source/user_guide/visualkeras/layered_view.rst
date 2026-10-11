.. _visualkeras-layered-view:

Layered view
============

:func:`layered_view` represents layers as scaled boxes and is especially useful
for convolutional architectures. Controls include minimum/maximum rendered
sizes, x/y/z scaling, layer type/index ignore lists, color maps, spacing,
padding, legends/text, and output file handling.

Visual dimensions are explanatory encodings, not literal physical tensor
sizes. Extreme tensor dimensions are intentionally clipped/scaled to keep the
figure usable. When exact shapes matter, include text labels or a separate model
summary.
