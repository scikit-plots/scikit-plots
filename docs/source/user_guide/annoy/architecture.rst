.. _annoy-architecture:

Architecture
============

``scikitplot.annoy`` has two compiled relationships that should not be confused.

.. code-block:: text

   scikitplot.cexternals._annoy
       low-level compiled Annoy implementation
                    |
                    +---- imported/inherited by ----> scikitplot.annoy.Index

   scikitplot.annoy._annoy
       separate Cython consumer of shared Annoy headers

The public high-level :class:`Index` is mixin-composed. The mixins isolate
orthogonal concerns such as persistence, metadata, NumPy interoperability,
pickling, vector helpers, and plotting. This makes it possible to use the core
ANN behavior without importing plotting or array libraries merely to import the
package.

Low-level versus high-level
---------------------------

Use ``Annoy`` / ``AnnoyIndex`` when compatibility with the low-level compiled
surface is the primary goal. Use :class:`Index` when you need explicit
``save_index``/``load_index`` naming, bundle metadata, safer Python-object
persistence, array conversion, or thread-resolution helpers.

Thread policy
-------------

:func:`threads_info` exposes the package's build-thread policy. Thread count is
an execution choice; it does not change the meaning of an already-built index.
Keep reproducibility concerns such as seed and build parameters separate from
resource allocation.
