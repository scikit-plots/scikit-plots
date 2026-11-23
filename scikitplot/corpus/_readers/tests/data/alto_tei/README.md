# ALTO and TEI test fixtures

Seven files copied unchanged from
https://github.com/Princeton-CDH/remarx/tree/94f1f049788b54b501b469bf994a5c9af7c1fa8d/tests/test_sentence/test_corpus/fixtures
(commit 94f1f04, 2026-07-21). They are licensed under the Apache License,
Version 2.0 (`LICENSE` in this folder), not under scikit-plots' BSD-3-Clause
license. The source repository has no NOTICE file. The texts they carry
(Marx; *Die Neue Zeit*, 1896-97) are in the public domain.

Do not edit them: the tests compare against these exact bytes. To replace
them, copy the new files from the source and update the hashes below.

| File | What it tests |
|---|---|
| `alto_page.xml` | one ALTO v4 page (eScriptorium export, words per line in `CONTENT`) |
| `alto_metadata.xml` | an ALTO page with title and author blocks |
| `alto_page_with_footnote.xml` | an ALTO page with a footnote block |
| `alto_sample.zip` | five ALTO pages stored out of order, an empty page, an unsorted page |
| `alto_hyphenated_test.zip` | a word broken across two lines |
| `sample_tei.xml` | a TEI document |
| `sample_tei_with_footnotes.xml` | a TEI document with footnotes |

SHA-256:

```text
d6197a25b5cb896f70138991a6ad7235999e5d4aa5fa3cc1f8fed853673c7b54  alto_hyphenated_test.zip
d9fb20cc785af43342d235f6bec24bf2a19cc3e7c7c59d7c0a663951ddce5cc7  alto_metadata.xml
0baf62e91faf9a43a436b63552c8338ea67a19f5402daed7ea62937f46901c3f  alto_page.xml
f85bfa91131ffe792bbe7061ef6200bd5592d9686db991cea643b7e92d30b3cb  alto_page_with_footnote.xml
c74f5ace0a04c7f10ca600d8dd64d0935c1abe0372ecee54594eb358dce8cfb2  alto_sample.zip
8fd415061934aedd25c27a63f2dbee910ad886ad7a115fae551cf0cc05f3ebdb  sample_tei.xml
5308dc685c63c43ae5b273db5580894c5fdef13f509678692545c3a5ccbb3b25  sample_tei_with_footnotes.xml
```
