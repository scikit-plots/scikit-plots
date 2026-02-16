.. _visualkeras_examples:


Visualkeras
----------------------------------------------------------------------

Examples related to the :py:mod:`~scikitplot.visualkeras` submodule with
e.g. a DL (ANN, CNN, NLP) :py:class:`~tensorflow.keras.Model` model instance.

.. https://www.linkedin.com/posts/lysandredebut_i-have-bittersweet-news-to-share-yesterday-activity-7338966863403528192-om5p
.. important::

    * ⚠️ Hugging Face Deprecated Transformers models are not supported in TensorFlow — use KerasNLP or KerasHub instead.
    * `🚫 transformers deprecated models <https://www.linkedin.com/feed/update/urn:li:activity:7338966863403528192/>`_.

.. prompt:: bash $

    # 💡visualkeras Need aggdraw tensorflow or tensorflow-cpu
    pip install scikitplot[core, cpu]

    # (Recommended)
    # !pip install aggdraw
    # !pip install tensorflow

    python -c "import tensorflow as tf, google.protobuf as pb; print('tf', tf.__version__); print('protobuf', pb.__version__)"
    python -m pip check

    # If Needed
    # pip install -U "protobuf<6"
    # pip install protobuf==5.29.4
    import tensorflow as tf
