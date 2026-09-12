# Ownership family

`scikitplot.visualkeras` owns visualization and framework-adaptation semantics. TensorFlow/Keras own model/layer internals; Pillow and aggdraw own raster/vector drawing mechanics; root `scikitplot.utils._pil` owns save/display decorator behavior. Do not copy backend semantics into maintenance code.
