.. _memmap-protection-sync-locking:

Protection, sync, and locking
=============================

The module exposes native-style protection and synchronization operations
through a Python wrapper.

``mprotect``
   Changes protection for the mapped region. Invalid combinations or unsupported
   operations raise rather than silently weakening the requested policy.

``msync``
   Requests synchronization of modified file-backed pages. ``MS_SYNC`` and
   ``MS_ASYNC`` express different durability/latency trade-offs.

``mlock`` / ``munlock``
   Request that pages stay resident in physical memory. These calls are subject
   to operating-system policy and resource limits such as Linux
   ``RLIMIT_MEMLOCK``; failure is not exceptional evidence of a broken mapping.

Protection flags are capability requests, not security sandboxes. A writable
mapping still gives code in the process access to the mapped bytes.
