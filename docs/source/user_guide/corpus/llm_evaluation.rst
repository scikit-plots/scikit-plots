..
  docs/source/user_guide/corpus/llm_evaluation.rst

.. currentmodule:: scikitplot.corpus

.. _corpus-llm-evaluation:

Judging answers with language models
====================================

Retrieval metrics (:ref:`corpus-evaluation`) say whether the right context was
found. They cannot say whether a generated answer stays true to that context,
answers the question, or beats another model's answer. For that a model reads
the answer and gives a verdict: an *LLM judge*.

Here the judge labels and a fixed formula scores. The judge says which claims
are supported, which contexts are relevant, or which answer is better, as
strict JSON. The score is computed from those labels, so a judge can never
return a number that the formula did not produce.

Run the deterministic metrics first. A judge's scores vary between runs and
between judges, and a regression in retrieval should not hide behind that
variance.

Connect a model
---------------

:func:`make_llm` puts every provider behind one call,
``llm.complete(prompt, system=...)``:

.. code-block:: python

   from scikitplot.corpus import make_llm

   gpt = make_llm("openai", "your-openai-model", max_tokens=1024)
   claude = make_llm("anthropic", "your-claude-model", max_tokens=1024)
   gemini = make_llm("gemini", "your-gemini-model", max_tokens=1024)

* Each provider needs its own SDK: ``pip install openai``, ``anthropic`` or
  ``google-genai``. The SDK is imported when the model is made, never when
  :mod:`scikitplot.corpus` is imported.
* With no ``client=``, the SDK's default client reads its API key from the
  environment (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``
  or ``GOOGLE_API_KEY``). A key is never written to an identity, a cache key
  or a report.
* ``client=`` takes any client with the same surface, such as
  ``openai.OpenAI(base_url=...)`` for an OpenAI-compatible server.
* Any callable ``fn(prompt, system) -> str`` is a model too: a local model, a
  gateway, or a test double.
* Values inside ``options["extra_headers"]`` and ``options["extra_query"]``
  (where gateways and Azure put keys) are written as ``"<redacted>"`` in the
  model's identity and ``repr``.
* Nothing is sent that you did not ask for. No temperature, seed or response
  format is added, because models differ in which they accept. Pass them in
  ``options`` where your model takes them, for example
  ``options={"temperature": 0}``.
* Anthropic's Messages API requires ``max_tokens``. For OpenAI it is sent as
  ``max_completion_tokens`` (``openai`` 1.45 and later; with an older SDK, or
  a server that does not know it, pass ``options={"max_tokens": n}``
  instead), and for Gemini as ``max_output_tokens``. A request argument that
  the client's signature does not take is refused when the model is made,
  not on every call.
* A response that stopped early (cut off at the token limit, or filtered)
  raises :class:`IncompleteCompletionError`. A cut-off verdict is not a
  verdict.
* ``cache={}`` (or any mutable mapping, such as a ``shelve`` file) reuses a
  response for an identical request to the same endpoint. The key covers the
  provider, model, options, the client's ``base_url``, the system prompt and
  the prompt. A mapping that works only in the thread that opened it (a
  ``shelve`` file on Python 3.13 or later, which uses SQLite) cannot be used
  with ``n_jobs > 1``. Its failure raises :class:`CacheError`, and is not
  mistaken for a model failure.

Judge one answer
----------------

The examples on this page use a stand-in judge, so they run without an API
key. Swap in a model from :func:`make_llm` for real use:

.. code-block:: python

   import json
   from scikitplot.corpus import LLMJudge

   def toy_model(prompt, system):
       """A stand-in for a real model: it supports every claim and rates 4 of 5."""
       if '"claims"' in prompt:
           return json.dumps({"claims": [
               {"claim": "Paris is the capital of France.", "supported": True},
               {"claim": "Paris has 30 million inhabitants.", "supported": False},
           ]})
       return json.dumps({"score": 4, "reason": "Direct answer."})

   judge = LLMJudge(toy_model)
   contexts = ["Paris is the capital and largest city of France."]
   result = judge.faithfulness(
       "What is the capital of France?",
       "Paris, a city of 30 million people, is the capital of France.",
       contexts,
   )
   result.score        # 0.5
   result.details      # {'n_claims': 2, 'n_supported': 1}
   judge.answer_relevance("What is the capital of France?", "Paris.").score   # 0.75

Each method returns a :class:`JudgeResult`. It holds the score, the parsed
verdict, every completion with its token counts, and ``error`` or
``undefined`` when there is no score.

==========================================  ==================================================
Method (metric)                             Score
==========================================  ==================================================
:meth:`LLMJudge.faithfulness`               supported claims / claims made by the answer
:meth:`LLMJudge.answer_relevance`           ``(rating - 1) / 4`` for a 1-5 rating
:meth:`LLMJudge.context_precision`          average precision of the per-context relevance
                                            verdicts, in retrieval order
:meth:`LLMJudge.context_recall`             reference statements attributable to the
                                            contexts / reference statements
:meth:`LLMJudge.answer_correctness`         ``TP / (TP + (FP + FN) / 2)`` over statements
:meth:`LLMJudge.criteria_score`             ``(rating - 1) / (scale - 1)`` for your rubric
:meth:`LLMJudge.preference`                 1, 0.5 or 0 for answer A, asked in both orders
==========================================  ==================================================

A verdict must be a single JSON object, optionally inside one fenced code
block. Every required field must have its exact JSON type: booleans are
``true``/``false``, and ratings are integers in range. Duplicate keys and
``NaN`` are refused. Nothing is guessed from prose. :func:`parse_verdict`
applies this rule, and ``on_error`` decides whether a malformed verdict
raises (``"raise"``, the default) or is recorded with ``score=NaN``
(``"record"``).

``undefined`` covers a valid verdict that has no score. An answer that makes
no claim has no faithfulness, for example. Contexts may be strings, search
hits (``hit.doc.text``), documents (``doc.text``) or mappings with ``"text"``
(:func:`context_texts`).

Evaluate a system
-----------------

:func:`evaluate_answers` asks your system every question of a dataset and
scores every answer in two ways: with the deterministic text metrics against
the references, and with each judge.

.. code-block:: python

   from scikitplot.corpus import evaluate_answers

   dataset = [
       {"id": "q1", "question": "What is the capital of France?", "reference": "Paris",
        "contexts": ["Paris is the capital and largest city of France."]},
       {"id": "q2", "question": "Which planet is the largest?", "reference": ["Jupiter", "planet Jupiter"],
        "contexts": ["Jupiter is the largest planet in the Solar System."]},
   ]

   def my_system(question):
       return {"Which planet is the largest?": "Jupiter"}.get(question, "Paris")

   report = evaluate_answers(my_system, dataset, judges={"toy": toy_model})
   report.metrics
   # {'exact_match': 1.0, 'token_f1': 1.0, 'faithfulness/toy': 0.5, 'faithfulness': 0.5,
   #  'answer_relevance/toy': 0.75, 'answer_relevance': 0.75}

* ``system`` may return a ``str``, a mapping with ``"answer"`` (and the
  ``"contexts"`` it retrieved, which then replace the dataset's), or an object
  with ``.answer``.
* With ``system=None`` each item's stored ``"answer"`` is evaluated. Use this
  to score answers that your model, GPT, Claude or Gemini produced elsewhere.
* A metric whose input an item lacks (no reference, no contexts) is ``NaN``
  for that item and costs no call. ``report.coverage`` counts the defined
  values.
* A judge sees a reference only when the item has exactly one. Several
  acceptable references go to the text metrics only.
* ``on_error="record"`` (the default) keeps going when the system or a judge
  fails, and counts the failures in ``report.errors``. ``n_jobs`` runs calls
  on threads, and results stay in dataset order.

Several judges form a panel. ``<metric>/<judge>`` is each judge's score and
``<metric>`` is their mean. ``report.agreement`` gives Krippendorff's alpha
between the judges for each metric. An alpha well below 1 means the judges
do not measure the same thing, and their mean should not be trusted.

.. code-block:: python

   panel = evaluate_answers(None, [dict(item, answer="Paris") for item in dataset[:1]],
                            judges={"first": toy_model, "second": toy_model},
                            metrics=["answer_relevance"], text_metrics=[])
   panel.metrics      # {'answer_relevance/first': 0.75, 'answer_relevance/second': 0.75, 'answer_relevance': 0.75}
   panel.usage["first"]["responses"]   # 1

The report records every judge's identity: provider, model, options, prompt
version and a hash of the prompts. ``report.usage`` counts responses, cached
responses and the tokens of the responses that were not cached, so a rerun
answered from the cache costs nothing. ``report.errors`` counts failures,
whether or not the verdicts are kept. It is written and read like a retrieval report, as JSON,
JSON Lines (one row per item), YAML or TOML.
:meth:`AnswerEvaluation.compare` runs the same paired sign-flip test as
retrieval reports. Compare reports that were scored by the same judges.

.. code-block:: python

   report.write("answers.json")
   again = type(report).read("answers.json")
   comparison = report.compare(again)
   [(m.metric, m.delta) for m in comparison.metrics][:2]   # [('exact_match', 0.0), ('token_f1', 0.0)]

Head to head
------------

:func:`evaluate_preference` asks each judge which of two answers is better,
once in each order. Some models prefer whichever option comes first. A judge
that changes its verdict with the order contributes a tie, and lowers
``position_consistency``.

.. code-block:: python

   from scikitplot.corpus import evaluate_preference

   def always_first(prompt, system):
       return '{"winner": "answer_a", "reason": "The first one."}'

   duel = evaluate_preference(dataset, ["Paris", "Jupiter"], ["Lyon", "Saturn"],
                              judges={"biased": always_first}, names=("mine", "baseline"))
   duel.summary["biased"]["position_consistency"]   # 0.0
   duel.summary["biased"]["ties"]                   # 2

Agreement statistics
--------------------

The agreement functions work on any ratings, from models or from people:

* :func:`cohen_kappa`: two raters, with optional linear or quadratic weights.
* :func:`fleiss_kappa`: any number of raters, nominal labels.
* :func:`krippendorff_alpha`: any number of raters, missing ratings allowed,
  at the nominal, ordinal, interval or ratio level.

They agree with scikit-learn, statsmodels and the ``krippendorff`` package,
none of which they need:

.. code-block:: python

   from scikitplot.corpus import cohen_kappa, krippendorff_alpha

   round(cohen_kappa([1, 2, 3, 3], [1, 2, 3, 2], weights="quadratic"), 3)   # 0.8
   round(krippendorff_alpha([[1, 2, 3, None], [1, 2, 2, 4]], level="interval"), 3)   # 0.706

Limits
------

* **Prompt injection.** A judge reads untrusted text. Answers and contexts are
  placed in tags, any closing tag inside them is escaped, and the system
  instruction says that tagged text is data. This lowers the risk; it does
  not remove it.
* **Bias.** A judge may favour longer answers, or answers from its own model
  family. Use judges from several providers, read the agreement, and check a
  sample of verdicts (``keep_verdicts=True`` keeps them in the report).
* **Cost.** Each judge metric makes one call per item and judge, and
  preference makes two. ``report.usage`` counts responses and tokens, and
  ``cache=`` avoids paying twice for the same request.

.. seealso::

   * :ref:`corpus-evaluation` for retrieval metrics and scikit-learn scorers.
