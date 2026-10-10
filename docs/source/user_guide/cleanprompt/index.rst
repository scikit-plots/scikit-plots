..
  docs/source/user_guide/cleanprompt/index.rst

..
  https://devguide.python.org/documentation/markup/#sections
  https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections
  # with overline, for parts    : ######################################################################
  * with overline, for chapters : **********************************************************************
  = for sections                : ======================================================================
  - for subsections             : ----------------------------------------------------------------------
  ^ for subsubsections          : ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  " for paragraphs              : """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. currentmodule:: scikitplot.cleanprompt

.. _cleanprompt-index:

======================================================================
CleanPrompt User Guide
======================================================================

CleanPrompt is the local privacy boundary in :mod:`scikitplot.cleanprompt`.
It replaces values that its active detectors identify with stable placeholders
(or, when explicitly requested, selected surrogate values) before text leaves
your process, then restores the original values after a reply comes back.

The base tier is implemented with the Python standard library. Optional tiers
add named-entity detection, a local web interface, and the Fernet vault cipher.
The base package can therefore be imported and used without loading those
third-party packages.

.. important::

   CleanPrompt protects values that it **detects or that you explicitly tell it
   to hide**. It is not a proof that arbitrary text contains no sensitive
   information. Run ``doctor`` to see which detection surfaces are active and
   ``inspect`` when you need to review what a particular input would expose.
   :ref:`cleanprompt-security` lists what is and is not covered.

The whole idea in one picture
-----------------------------

.. code-block:: text

   original text
       |
       v
   detect -> assign stable labels -> rewrite
       |                              |
       |                              +---- safe text ----> model / tool / agent
       |
       +---- handle / vault stays local
                                      |
                           model reply v
                                 restore
                                      |
                                      v
                               local clear text

Two kinds of state come out of every redaction, and they go to different
places. The **redacted text** is meant to cross the model boundary. The
**vault** (inside a :class:`Handle` in the high-level API) holds the removed
values; it is the authority to restore them and **must not be sent with the
prompt**. Treat a :class:`Handle` or :class:`Vault` like a secret.

Thirty seconds
--------------

.. prompt:: python >>>

   from scikitplot.cleanprompt import encode, decode

   encoded = encode("Mail ada@example.com about the Acme renewal", hide=["Acme"])
   encoded.text
   # 'Mail [EMAIL-1] about the [CUSTOM-1] renewal'
   decode("I will contact [EMAIL-1] about [CUSTOM-1].", encoded.handle)
   # 'I will contact ada@example.com about Acme.'

From a terminal, the same pair needs no paths at all:

.. code-block:: bash

   python -m scikitplot.cleanprompt encode "Mail ada@example.com about the renewal"
   python -m scikitplot.cleanprompt decode "I will contact [EMAIL-1]."

Which surface should I use?
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 39 61

   * - Goal
     - Start with
   * - One prompt/reply exchange in Python
     - :func:`encode` and :func:`decode` (:ref:`cleanprompt-python-api`)
   * - A multi-turn conversation with stable labels
     - :class:`Session` or :func:`session`
   * - Full control over policy and detectors
     - :class:`Redactor`, :class:`RedactionPolicy`, :class:`DetectorRegistry`
   * - Paste text into a chat by hand
     - ``cleanprompt encode`` and ``cleanprompt decode``
       (:ref:`cleanprompt-command-line`)
   * - Preview what would be removed
     - ``cleanprompt inspect``
   * - Fail CI when findings are present
     - ``cleanprompt scan``
   * - Names, organisations and places
     - the entity engines (:ref:`cleanprompt-entity-detection`)
   * - Files, records, folders, Office documents, or archives
     - :class:`FluentCleanPrompt` / ``cleanprompt batch``
       (:ref:`cleanprompt-files-and-packs`)
   * - Put a privacy gate in front of a model client
     - :class:`Guard` (:ref:`cleanprompt-agents`)
   * - Guard an arbitrary command-line model
     - ``cleanprompt ask --via "COMMAND"``
   * - Give an MCP agent a local redaction boundary
     - ``cleanprompt mcp``
   * - Pin team configuration and detect definition drift
     - ``cleanprompt plan``
   * - A browser page, locally or in a container
     - ``cleanprompt flask`` / ``cleanprompt docker``
       (:ref:`cleanprompt-web-and-containers`)
   * - Check installation and optional capabilities
     - :func:`capabilities` / ``cleanprompt doctor``

How this guide is organised
---------------------------

Read the first two pages in order; after that, go to the page for the job in
front of you.

.. toctree::
   :maxdepth: 2

   getting_started
   how_it_works
   python_api
   command_line
   entity_detection
   files_and_packs
   agents_and_models
   web_and_containers
   security_and_limits
   troubleshooting

The executable gallery, :ref:`cleanprompt_examples`, follows the same path with
code you can run: basics, the command line, the Python API, notebooks, then
moderate and advanced topics, recipes, packs and formats, and agents.

.. seealso::

   * :ref:`cleanprompt_examples`
   * :ref:`corpus-index`
   * :mod:`scikitplot.cleanprompt`
