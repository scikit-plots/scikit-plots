.. _visualkeras-troubleshooting:

Troubleshooting
===============

Unknown or missing shape
   Some custom/lazy layers do not expose a usable output shape at visualization
   time. Build/call the model as appropriate or provide labels that do not
   require unavailable shape metadata.

Graph edges missing after framework upgrade
   Verify Keras/TensorFlow compatibility and layer-node traversal rather than
   changing graph layout parameters first.

Text/font errors
   Confirm the selected font exists in the documentation/runtime environment.
   Avoid embedding machine-specific font paths in reusable examples.

Image too large
   Reduce scale, ignore non-essential layers, use a graph view, or render a
   selected sub-model.
