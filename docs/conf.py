import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

project = 'dialz'
copyright = f"Zara Siddique {datetime.today().year}"
author = 'Zara Siddique'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    'myst_nb',
    'sphinx_copybutton'
]

pygments_style = "sphinx"

templates_path = ['_templates']
exclude_patterns = []

nb_execution_mode = "off"

autodoc_typehints = "none"

html_theme = "furo"
# html_static_path = ["_static"]