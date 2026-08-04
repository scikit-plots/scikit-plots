Cython Architecture
===================

The runtime architecture separates public API handling, policy validation,
build orchestration, cache publication, loading, and maintenance.

Architecture overview
---------------------

.. include:: _diagrams/architecture_overview.rst
  :start-after: :orphan:

Public API path
---------------

.. include:: _diagrams/public_api_flow.rst
  :start-after: :orphan:

Security boundary
-----------------

.. include:: _diagrams/security_validation_flow.rst
  :start-after: :orphan:

Template resolution
-------------------

.. include:: _diagrams/templates_flow.rst
  :start-after: :orphan:
