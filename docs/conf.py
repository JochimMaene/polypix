"""Sphinx configuration for the Polypix documentation."""

from __future__ import annotations

from importlib.metadata import version as package_version

project = "Polypix"
author = "Jochim Maene"
copyright = "2026, Jochim Maene"
release = package_version("polypix")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["assets/generated/*.html"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 4

autosummary_generate = True
autodoc_typehints = "description"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "Polypix"
html_baseurl = "https://jochimmaene.github.io/polypix/"
html_extra_path = ["assets"]
html_theme_options = {
    "github_url": "https://github.com/JochimMaene/polypix",
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_depth": 3,
}
