.. _mcp-deployment-and-operations:

Deployment and operations
=========================

Local stdio
-----------

The default CLI transport is suitable for a local MCP client that launches the
server as a subprocess::

   python -m scikitplot.mcp

In this mode the process waits for an MCP client on standard input/output. Use
``--self-test`` when you only want to validate corpus/retrieval behavior.

Local HTTP
----------

Use Streamable HTTP when a local service needs an HTTP endpoint::

   python -m scikitplot.mcp --transport streamable-http

The default local endpoints are ``http://127.0.0.1:8000/mcp`` and
``http://127.0.0.1:8000/healthz``. ``--probe`` checks the lightweight health
endpoint; ``--self-test`` is the stronger retrieval/corpus check. A healthy HTTP
route does not prove that the intended corpus is loaded or that search quality is
correct.

Docker
------

``--docker`` selects container-oriented defaults. For local development, publish
the container port only on host loopback when possible::

   docker run --rm -p 127.0.0.1:8000:8000 scikitplot-mcp:latest

The Docker profile does **not** provide authentication, authorization, TLS,
firewall policy, or per-user quotas. For production images install a built wheel
in a clean runtime stage rather than relying on an editable source-tree install.

Remote/team clients
-------------------

Treat remote MCP like any other network service:

.. code-block:: text

   MCP client
      |
      v
   authenticated TLS proxy / gateway
      |
      v
   private scikitplot.mcp service
      |
      v
   read-only retriever

Use identity, authorization, TLS, firewall/private-network policy, request and
concurrency limits, secret-safe logging, and an explicit corpus/runtime update
process. The CLI refuses an ordinary unauthenticated non-local bind unless the
operator makes that choice explicitly; that override is not a substitute for a
production security layer.

Operational controls
--------------------

Common controls include host/port/path, health path, maximum concurrency,
request-body limits, stateful HTTP, and log level. Keep those deployment controls
separate from retrieval parameters such as ``k`` or hybrid-fusion weights.
