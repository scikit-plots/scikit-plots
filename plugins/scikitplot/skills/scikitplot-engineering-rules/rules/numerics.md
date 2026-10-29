# Numerics and scoring

Categories: Numerics, Exactness, Scoring, Native kernels

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)).

### Rule 78: Native kernels - elements are compared by equality, never by their hash
- **When:** C, C++ or Cython code compares Python elements through an integer encoding.
- **Then:** encode `str` by code point; encode anything else through one `dict` shared by both operands, so two elements get one code exactly when the container would call them equal (identity, then `==`). A test feeds colliding values: `[-1]`/`[-2]`, `[0]`/`[2**61 - 1]`, and tuples of them.
- **Verified by:** `_editdistance/test/test_bycython.py::test_equal_hashes_do_not_make_elements_equal`; the levenshtein collision tests run on every available backend.
- **Added:** 2026-10-12. Root cause: `bycython` passed `hash(element)` to the kernel. CPython gives `hash(-1) == hash(-2)`, so `levenshtein.distance([-1], [-2])` was 0 on the default backend and 1 with `backend="python"`.

### Rule 83: Scoring - one formula per recipe, and every scoring path calls it
- **When:** a ranker has more than one scoring path (dense, batch, sparse, cached).
- **Then:** keep one per-term function per recipe. A sparse path adds every query term's contribution at frequency 0, unless the recipe proves that contribution is 0. A randomised differential test holds the ranks and the scores of all paths bit-identical.
- **Verified by:** `test_sparse_ranking_and_scores_equal_the_dense_path`, which runs 200 random corpora for each of the 3 recipes, and `test_batch_scores_equal_the_dense_path`.
- **Added:** 2026-10-12. Root cause: the sparse path assumed that a document without a query term scores 0. BM25L and BM25+ add a delta lower bound, so rankings differed in 34 and 35 of 300 random queries. The one agreement test used a single-term Okapi query.

### Rule 90: Numerics - compute in float64, cast to the storage dtype last, and let non-finite values reach their guard
- **When:** normalising, summing or dividing vectors whose output dtype may be narrow (`float16`), or whose input may be non-finite.
- **Then:** do the arithmetic in `float64` and cast at the end. Never map NaN or inf to zeros: a zero row passes the finiteness check that a NaN row would fail.
- **Verified by:** `test_float16_is_normalised_before_the_cast` (768 components of 10 gave rows of zeros before) and `test_a_non_finite_vector_is_not_hidden`.
- **Added:** 2026-10-12. Root cause: Run 1 normalised after `astype(self.dtype)`; in `float16` the sum of squares overflowed above 65504.

### Rule 97: Numerics - compare floating sums with a tolerance scaled to the terms, not to the result
- **When:** deciding equality or "at least as extreme" between sums of floats (permutation and sign-flip tests, ties in scores).
- **Then:** use an absolute tolerance of the order `n * eps * sum(|terms|)`, or exact arithmetic (`fractions`, `math.fsum` plus a bound).
- **Verified by:** `test_flips_equal_for_the_represented_values_count` (`0.3 + 0.4` against `0.7`).
- **Added:** 2026-10-12. Root cause: a tolerance relative to the observed statistic missed rounding error that scales with the per-query differences; an exact test reported 0.875 instead of 1.0.

### Rule 113: Exactness - sort on floats, decide near ties exactly
- **When:** an order must follow exact arithmetic (rank fusion, ties by id) and exact arithmetic is slow.
- **Then:** sort by the float value; within runs whose floats differ by less than the rounding bound (`4 * n * 2**-53 * max`), sort by the exact value (`fractions.Fraction`), then the tie-break key. Report a float computed order-independently (`math.fsum`).
- **Verified by:** `test_float_close_scores_are_ordered_exactly`, `test_an_exact_tie_is_a_tie_whatever_the_rounding`, the randomised reference tests.
- **Added:** 2026-10-13. Root cause: the first contract summed `Fraction`s for every term, 30-40x slower than the float code it replaced; the review measured it.
