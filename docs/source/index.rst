.. https://docs.open-mpi.org/en/v5.0.x/developers/rst-for-markdown-expats.html#whitespace-and-indenting
.. https://docutils.sourceforge.io/docs/user/rst/quickref.html
.. https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#hyperlink-targets
.. Used for setting metadata like the document's title for processing tools.
.. title:: Homepage

.. https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#comments
.. Adding Comments
.. Headings and Sections
.. = Main title or top-level heading.
.. - Second-level heading.
.. ~ Third-level heading.
.. ^ Fourth-level heading.
.. " Fifth-level heading.

.. External Links
.. `Link text <http://example.com>`_
.. Internal Links (to sections within the document)
.. See the :ref:`section_name` for more details.
.. Cross-referencing Sections
.. :ref:`section_name`

.. Define the overall structure, that affects the prev-next buttons and the order
   of the sections in the top navbar.

.. https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directives
.. toctree::
   :maxdepth: 2
   :caption: Introduction
   :hidden:

   Introduction <introduction/introduction.rst>

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   User Guide <user_guide/user_guide.rst>
   API Reference <api/index.rst>

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   Tutorials <auto_examples/index>

.. toctree::
   :maxdepth: 1
   :caption: Community
   :hidden:

   Release Notes <whats_new/whats_new.rst>
   Community <https://zenodo.org/communities/scikit-plots>
   
   
   Developer’s Guide <https://scikit-plots.github.io/dev/developers/index.html>
   
   FAQ <community/faq.rst>
   Glossary <community/scikiplot_glossary.rst>
   About us <teams/about.rst>