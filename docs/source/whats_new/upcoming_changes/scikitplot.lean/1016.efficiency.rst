- Forked name-resolving workers read each reference and each context
  (namespace and opens) as text from one shared buffer, decoded privately,
  and the resolution cache is keyed by context number: a worker resolving
  every name of mathlib4 copies 62 MB instead of 125 MB. With two workers
  a summary's peak memory falls from 550 MB to 451 MB on mathlib4 and from
  3.0 GB to 2.6 GB on a project of a million declarations; one worker is
  unchanged.
