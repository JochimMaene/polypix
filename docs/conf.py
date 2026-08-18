"""Sphinx configuration for the Polypix documentation."""

from __future__ import annotations

from importlib.metadata import version as package_version

project = "Polypix"
author = "Jochim Maene"
copyright = "2026, Jochim Maene"
release = package_version("polypix")

extensions = [
    "myst_parser",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["assets/generated/*.html"]

myst_enable_extensions = [
    "attrs_block",
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 4

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "sphinx_book_theme"
html_title = "Polypix"
html_baseurl = "https://jochimmaene.github.io/polypix/"
html_extra_path = ["assets"]
html_static_path = ["_static"]
html_css_files = ["polypix.css"]
html_favicon = "_static/polypix-mark-square.svg"
html_theme_options = {
    "logo": {
        "image_light": "polypix.svg",
        "image_dark": "polypix-dark.svg",
        "alt_text": "Polypix",
        "text": "Polypix",
    },
    "repository_url": "https://github.com/JochimMaene/polypix",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": False,
    "use_fullscreen_button": False,
    "home_page_in_toc": False,
    "show_navbar_depth": 1,
    "max_navbar_depth": 3,
    "show_toc_level": 2,
    "toc_title": "On this page",
    "navigation_with_keys": False,
    # The sidebar already carries a search field; pydata's persistent navbar
    # search would render a second one on desktop.
    "navbar_persistent": [],
}
