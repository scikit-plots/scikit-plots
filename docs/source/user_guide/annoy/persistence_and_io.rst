.. _annoy-persistence-and-io:

Persistence and I/O
===================

Annoy's native strength is a compact index that can be memory-mapped after it
is built. The high-level facade adds several persistence choices; use the least
powerful one that matches your requirement.

Native index files
------------------

Use ``save_index`` / ``load_index`` (or the corresponding low-level
``save`` / ``load`` methods) when you only need the Annoy index artifact. This
is the most direct representation and is appropriate for process-shared,
read-mostly retrieval.

Bundles and metadata
--------------------

The I/O and metadata mixins can publish a bundle containing the index plus
versioned metadata. Prefer this when an index must travel with enough parameters
to validate how it should be reconstructed.

Byte serialization and pickle
-----------------------------

Byte serialization is useful for controlled application transport. Pickle is a
Python object persistence mechanism and must be treated as executable input:
**never unpickle data from an untrusted source**. Compression modes affect
storage size, not the trust boundary.

Mmap lifetime
-------------

Loading may memory-map index data. Keep the underlying artifact available for
as long as the mapped index is in use. Replacing or truncating a mapped file in
place is not a safe publication strategy; publish a new artifact and switch
readers to it.
