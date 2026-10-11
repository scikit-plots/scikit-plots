.. _impute-index-access-and-privacy:

Index access and privacy
========================

``ANNImputer`` supports different index-access/ownership modes. The distinction
matters because a nearest-neighbor index is derived from training data and may
still reveal information about it.

Private ownership
   Keep the index encapsulated by the estimator when callers should not receive
   direct index access.

External/public ownership
   Useful when an application intentionally manages or shares the index, but the
   caller becomes responsible for storage, lifetime, permissions and exposure.

On-disk indexes are artifacts and should inherit the sensitivity of the data
used to build them. Memory mapping reduces duplicate RAM use; it does not make
an index anonymous or safe to publish.
