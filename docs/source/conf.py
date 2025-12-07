"""Sphinx configuration for multistat package documentation."""

import os
import sys

# Add src directory to path for module imports
src_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "src"))
sys.path.insert(0, src_path)

repo_root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, repo_root)

# -- Project information -----------------------------------------------------

project = "02409 Multivariate Statistics - Exam Tools"
copyright = "2025, Philip Korsager Nickel"
author = "Philip Korsager Nickel"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

root_doc = "index"
source_suffix = {".rst": "restructuredtext"}

# -- MathJax configuration ---------------------------------------------------

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}

# -- Autodoc configuration ---------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}

# -- Numpydoc configuration --------------------------------------------------

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"optional", "default", "of"}

templates_path = ["_templates"]
numpydoc_use_plots = False

# -- Sphinx Gallery configuration --------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": "../../exams/2019",  # Path to solved exam (only 2019 has solutions)
    "gallery_dirs": "exam_gallery/2019",  # Output directory for gallery
    "filename_pattern": "/plot_",  # Pattern to match which scripts to execute
    "download_all_examples": False,
    "remove_config_comments": True,
    "abort_on_example_error": False,
    "plot_gallery": True,
    "capture_repr": ("_repr_html_", "__repr__"),
    "matplotlib_animations": True,
    "first_notebook_cell": None,
    "last_notebook_cell": None,
    "notebook_images": False,
    "nested_sections": False,
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("multistat",),
    "inspect_global_variables": True,
    "reference_url": {
        "multistat": None,
    },
}

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
}

# -- HTML output options -----------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "02409 Multivariate Statistics - Exam Tools"
html_static_path = ["_static"]
html_show_sourcelink = False
html_css_files = ["custom.css"]

html_theme_options = {
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "show_toc_level": 2,
    "collapse_navigation": True,
    "navigation_depth": 4,
    "show_nav_level": 2,
    "secondary_sidebar_items": ["page-toc"],
}
