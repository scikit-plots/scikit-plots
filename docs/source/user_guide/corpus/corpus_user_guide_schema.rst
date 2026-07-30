Corpus User Guide Diagram Schema
================================

This file is an integration template. Copy the relevant sections into the
actual ``scikitplot.corpus`` user-guide pages and adjust relative paths if the
diagram directory is placed elsewhere.

At a glance
-----------

Use the logical view first. It answers what happens to user data without
exposing package internals.

.. include:: _diagrams/00_corpus_at_a_glance_logical.rst

Architecture for maintainers
----------------------------

Keep the physical package layout separate from the user journey.

.. include:: _diagrams/01_corpus_physical_module_map.rst

Public API
----------

.. include:: _diagrams/02_public_api_facade_flow.rst

CorpusPipeline
--------------

.. include:: _diagrams/03_pipeline_execution_flow.rst

.. include:: _diagrams/04_pipeline_lifecycle_state.rst

CorpusBuilder
-------------

.. include:: _diagrams/05_corpus_builder_flow.rst

.. include:: _diagrams/06_corpus_builder_lifecycle_state.rst

Sources, URLs, and downloads
----------------------------

.. include:: _diagrams/07_source_resolution_flow.rst

.. include:: _diagrams/08_url_handler_flow.rst

.. include:: _diagrams/09_downloader_dispatch_flow.rst

Readers and archives
--------------------

.. include:: _diagrams/10_document_reader_factory_flow.rst

.. include:: _diagrams/11_reader_family_flow.rst

.. include:: _diagrams/12_archive_processing_flow.rst

Chunking
--------

.. include:: _diagrams/13_chunker_family_flow.rst

.. include:: _diagrams/14_multilingual_semantic_chunking_flow.rst

Text transformation
-------------------

.. include:: _diagrams/15_normalization_flow.rst

.. include:: _diagrams/16_nlp_enrichment_flow.rst

Embeddings and search
---------------------

.. include:: _diagrams/17_embedding_flow.rst

.. include:: _diagrams/18_similarity_search_flow.rst

Persistence and export
----------------------

.. include:: _diagrams/19_storage_backend_flow.rst

.. include:: _diagrams/20_sqlite_storage_lifecycle_state.rst

.. include:: _diagrams/21_export_flow.rst

Integrations and metadata
-------------------------

.. include:: _diagrams/22_adapter_flow.rst

.. include:: _diagrams/23_metadata_flow.rst

Extensibility
-------------

.. include:: _diagrams/24_component_registry_flow.rst

.. include:: _diagrams/25_custom_hooks_flow.rst

Reliability and contracts
-------------------------

.. include:: _diagrams/26_pipeline_guard_state.rst

.. include:: _diagrams/27_schema_and_types_flow.rst

.. include:: _diagrams/28_error_propagation_flow.rst

Security and platform evolution
-------------------------------

The security-policy diagram is explicitly a target architecture. Label it as
proposed or planned until the corresponding components exist in source.

.. include:: _diagrams/29_security_and_resource_policy_target.rst

.. include:: _diagrams/30_platform_capability_flow.rst

.. include:: _diagrams/31_compatibility_layer_flow.rst
