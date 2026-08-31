..
  # Used for setting metadata like the document's title for processing tools.

.. title:: Homepage

..
  # Label (target) used for internal referencing, use underscores or hyphens (no spaces allowed)
  # Underscores are the official recommended separator in reST labels.
  # Dashes are allowed but sometimes can conflict with word hyphens.
  # See the section on `whatever <#root-index>`_ or See :ref:`root-index`.

.. _root-index:

.. Define the overall structure, that affects the prev-next buttons and the order
   of the sections in the top navbar.

.. toctree::
	:maxdepth: 2
	:caption: Introduction
	:hidden:

	Introduction <introduction/index.rst>

.. toctree::
	:maxdepth: 2
	:caption: Documentation
	:hidden:

	User Guide <user_guide/index.rst>
	APIs Reference <apis/index.rst>

.. toctree::
	:maxdepth: 2
	:caption: Tutorials
	:hidden:

	Tutorials <auto_examples/index>
	Learn <https://scikit-plots-learn.readthedocs.io/en/latest/learn/index.html>
	Tags <_tags/tagsindex.rst>

.. toctree::
	:maxdepth: 1
	:caption: Project
	:hidden:

	Code of Conduct <project/code_of_conduct.rst>
	Community <project/community.rst>
	Developer's Guide <devel/index>
	Governance Process <project/governance.rst>
	Release History <whats_new/index.rst>
	Roadmap <project/roadmap.rst>
	About Us | Project <project/index.rst>
