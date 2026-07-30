:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TB
        U[User callables and factories]
        W[Contract wrappers]
        C[CustomChunker, Filter, Normalizer, Enricher]
        H[HookableCorpusPipeline]
        P[PipelineHooks before, after, and error callbacks]
        B[FactoryCorpusBuilder]
        S[CustomSimilarityIndex]
        K[Core corpus contracts]
        R[Pipeline or build result]

        U --> W
        W --> C
        W --> H
        W --> B
        W --> S
        H --> P
        C --> K
        P --> K
        B --> K
        S --> K
        K --> R
