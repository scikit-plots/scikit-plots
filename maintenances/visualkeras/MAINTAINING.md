# Maintaining `scikitplot.visualkeras`

This domain owns architecture visualization behavior in `scikitplot/visualkeras/`, including graph/layered rendering semantics, Keras/TensorFlow layer traversal, spacing helpers, drawing utilities, focused tests, and optional dependency behavior. It does not own TensorFlow/Keras internals, Pillow/aggdraw, root image-saving infrastructure, or upstream visualkeras.

Read `MAINTENANCE.json` in the documented order, then run the contract checker, subsystem reviewer, and maintenance regression suite. Current release status is blocked by open graph-shape, ellipsize, optional-backend, default-deprecation, and test-matrix findings. Do not bless fingerprints merely because a harness can render representative models.
