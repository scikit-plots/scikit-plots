
.. currentmodule:: scikitplot._logo

.. _logo-index:

Logo API
========

Scikit-plots includes a small brand helper that generates the official
logo using **pure Matplotlib primitives**.
The default output is deterministic and requires **no image assets**.

This makes the logo easy to:

* ship inside the library,
* regenerate at any size,
* theme for dark/light docs,
* export as SVG/PNG without external dependencies.

.. note::

   The public interface is exposed as :mod:`~scikitplot._logo`.
   The internal implementation may live in :mod:`~scikitplot._logo`.

   If you are developing inside the repository you may see examples using
   ``sp._logo``; user code should prefer ``sp._logo`` when available.


Quickstart
----------

.. .. jupyter-execute
.. .. code-block:: python
.. prompt:: python >>>

   import scikitplot as sp

   # Show primary icon
   sp._logo.show()

   # Save a single icon
   sp._logo.save("scikit-plots.svg")

   # Save multiple variants using a template
   sp._logo.save(
       "assets/scikit-plots-{variant}.svg",
       variants=sp._logo.list_variants()
   )


Variants
--------

Scikit-plots provides several icon variants tuned for different contexts:

* **primary**: full brand mark for docs, README, websites.
* **small**: simplified icon for tiny sizes.
* **metrics**: highlights classic ML evaluation concepts.
* **knn**: subtle nod to Annoy/KNN features.

.. code-block:: python

   sp._logo.list_variants()
   # ('primary', 'small', 'metrics', 'knn')

   # Save all variants with auto suffix
   sp._logo.save("assets/scikit-plots.svg", variants=sp._logo.list_variants())


Themes & Monochrome
-------------------

Use themes for dark backgrounds and monochrome for badges or watermarks.

.. code-block:: python

   # Dark theme for docs landing pages
   sp._logo.save("assets/scikit-plots-dark.svg", theme="dark")

   # Monochrome for badges
   sp._logo.save("assets/scikit-plots-mono.svg", mono=True)


Decorative Dots
---------------

The primary mark supports a small dot field.
For brand-stable output, keep ``dots="fixed"`` (the default).

.. code-block:: python

   # Brand-stable default
   sp._logo.save("assets/scikit-plots-primary.svg", dots="fixed")

   # Turn off dots for extra minimal output
   sp._logo.save("assets/scikit-plots-clean.svg", dots="none")

   # Experimental random layout (not recommended for official assets)
   sp._logo.save("assets/scikit-plots-random.svg", dots="random", seed=42)


Presets
-------

Presets provide convenient defaults for common contexts such as favicons,
avatars, and documentation hero images.

.. code-block:: python

   sp._logo.list_size_presets()
   # ('favicon', 'avatar', 'docs-hero')

   # A crisp small icon for browser/app use
   sp._logo.save("assets/favicon.png", preset="favicon")

   # A balanced default for social/GitHub avatars
   sp._logo.save("assets/avatar.svg", preset="avatar")

   # A large icon for docs landing pages
   sp._logo.save("assets/hero.svg", preset="docs-hero")


Wordmark
--------

The wordmark combines the icon with a text lockup.
This is useful for website headers, slides, and banners.

.. code-block:: python

   # Show wordmark
   sp._logo.wordmark.show()

   # Save wordmark lockup
   sp._logo.wordmark.save("assets/scikit-plots-lockup.svg")


Saving Behavior
---------------

The save API follows Matplotlib-style conventions:

- Format is inferred from filename suffix.
- Override inference with ``ext=`` or ``format=``.
- Extra keyword arguments are forwarded to ``Figure.savefig``.

.. code-block:: python

   # Infer format from suffix
   sp._logo.save("assets/scikit-plots.png", dpi=300)

   # Force SVG even without suffix
   sp._logo.save("assets/scikit-plots", format="svg")

   # Save multiple variants using a template
   sp._logo.save(
       "assets/scikit-plots-{variant}.png",
       variants=["primary", "small"],
       dpi=300
   )


Command Line
------------

A lightweight CLI is available for batch generation.

.. code-block:: bash

   # Export all icon variants
   python -m scikitplot._logo --all --out assets

   # Generate a favicon preset
   python -m scikitplot._logo --all --preset favicon --out assets --format png
   python -m scikitplot._logo --preset favicon --out assets --format png

   # Generate the wordmark lockup
   python -m scikitplot._logo --wordmark --out assets --format svg
   python -m scikitplot._logo --wordmark --out assets --format png


Developing & Testing Logos
--------------------------

When extending or adjusting the brand system:

- Prefer testing **artist-level determinism** rather than raw SVG bytes.
- Keep ``dots="fixed"`` for official outputs.
- Add new variants by composing existing primitives (bars, wedges, lines).

If you need a backend-stability check for SVG serialization,
compare **normalized** SVG output rather than byte-for-byte equality.

.. seealso::

   * :meth:`matplotlib.figure.Figure.savefig`
   * :func:`scikitplot._logo.create`
   * :func:`scikitplot._logo.save`
   * :func:`scikitplot._logo.list_variants`
