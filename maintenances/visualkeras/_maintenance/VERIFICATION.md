# Verification

Evidence ladder:

1. Static contract + mutation suite.
2. Pure utility tests with real Pillow.
3. Keras-only package-root harness with an explicitly fake aggdraw adapter to exercise model traversal/rendering logic.
4. Negative probes for single-output flattening, ellipsis threshold, missing backend error semantics, and default warnings.
5. Complete-project run with real root helpers and real aggdraw.
6. TensorFlow lane and standalone Keras lane across supported versions/backends.
7. Platform/rendering matrix plus save/display behavior from the real root decorator.

Never promote level 3 into level 5: fake aggdraw proves VisualKeras control flow, not real aggdraw integration.
