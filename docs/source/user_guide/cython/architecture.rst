Cython Architecture
===================

The runtime architecture separates public API handling, policy validation,
build orchestration, cache publication, loading, and maintenance.

Architecture overview
---------------------

.. include:: _diagrams/architecture_overview.rst

Public API path
---------------

.. include:: _diagrams/public_api_flow.rst

Security boundary
-----------------

.. include:: _diagrams/security_validation_flow.rst

Template resolution
-------------------

.. include:: _diagrams/templates_flow.rst
