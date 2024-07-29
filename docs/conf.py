#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# RE-Emission documentation build configuration file, created by T. Janus
# 11-07-2024
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
import shlex
import reemission

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    'sphinx_math_dollar',
    "sphinx.ext.mathjax",
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_tabs.tabs',
    'sphinx_copybutton',
    'sphinx_toggleprompt',
    'sphinxcontrib.pdfembed',
    'sphinxcontrib.bibtex',
    "sphinx.ext.githubpages",  # Creates a .nojekyll to allow the `_static`, etc. folders to work
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.napoleon",
    #"myst_nb",
    "nbsphinx", # For adding jupyter notebooks into documentation
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

bibtex_bibfiles = ['_static/references.bib']
bibtex_encoding = 'utf-8'
bibtex_default_style = 'unsrt'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_sphinx_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

#autodoc_default_options = {
#    "members": True,
#    "undoc-members": True,
#    "private-members": True
#}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown'
} #     '.ipynb': '' entry not needed

# The master toctree document.
master_doc = "index"

project = 'ReEmission'
copyright = '2024, Tomasz Janus, University of Manchester, United Kingdom'
author = 'Tomasz Janus'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = reemission.__version__
# The full version, including alpha/beta/rc tags.
release = version

language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False
#html_theme = 'alabaster'
#html_theme = "sphinx_rtd_theme"
#html_theme = "pydata_sphinx_theme"
#html_theme = "sphinxdoc"
html_theme = "sphinxawesome_theme"
html_static_path = ['_static']
htmlhelp_basename = "ReEmissionDoc"

from dataclasses import asdict
from sphinxawesome_theme import ThemeOptions
from sphinxawesome_theme.postprocess import Icons
html_permalinks_icon = Icons.permalinks_icon
html_permalinks = False
theme_options = ThemeOptions(
   # Add your theme options. For example:
   show_breadcrumbs=True,
)
html_theme_options = asdict(theme_options)

# The name of the Pygments (syntax highlighting) style to use.
# Select theme for both light and dark mode
pygments_style = "sphinx"
# Select a different theme for dark mode
pygments_style_dark = "github-dark"

html_sidebars = {
  "**": ["sidebar_main_nav_links.html", "sidebar_toc.html"]
}

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #'preamble': '',
    # Latex figure (float) alignment
    #'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "Reemission.tex", "ReEmission Documentation", "Tomasz Janus", "manual"),
]

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "reemission", "ReEmission Documentation", [author], 1)]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "ReEmission",
        "ReEmission Documentation",
        author,
        "ReEmission",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# A string of reStructuredText that will be included at the end of every source 
# file that is read.
rst_epilog = """

.. |under-construction| image:: https://i.gifer.com/1Kj6.gif
           :alt: Page Under Construction

"""

