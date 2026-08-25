"""Sphinx configuration for the mimcs documentation.

Builds one site from two sources: the hand-written Markdown in ``design/`` and ``reference/``
(via MyST), and an API reference generated from the docstrings (via autodoc). Keeping them in one
tree is what lets the ~30 "see doc NN" citations in the source become real links.

    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build -W --keep-going
"""

import mimcs

project = "mimcs"
author = "Miika Kailas"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # the ~20% of modules that use Google-style Args: blocks
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Importing mimcs attaches a stream handler at INFO, so a build would otherwise carry the
# library's own log output interleaved with Sphinx's.
mimcs.set_log_level("WARNING")

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# `from __future__ import annotations` is used throughout, so signatures are already strings.
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Emit :ivar: fields instead of `.. attribute::` directives: the latter collide with
# autodoc's own documentation of the same dataclass fields (DrawComponent, OptimizeResult).
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = "mimcs"
