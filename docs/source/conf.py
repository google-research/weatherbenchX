# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# https://stackoverflow.com/a/62613202


# import subprocess
# import sys

import weatherbenchAfrica  # verify this works

from weatherbenchAfrica.data_loaders.xarray_loaders import PredictionsFromXarray
# print("python exec:", sys.executable)
# print("pip environment:")
# subprocess.run([sys.executable, "-m", "pip", "list"])

import os
import sys

sys.path.insert(0, os.path.abspath("../../weatherbenchX"))  # Source code dir relative to this file
print("sys.path:", sys.path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "WeatherBench-X"
copyright = "2025, WeatherBench-X authors"
author = "WeatherBench-X authors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    # "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "sphinx_rtd_theme"
# html_theme = "pydata_sphinx_theme"
html_theme = 'furo'  # https://pradyunsg.me/furo/quickstart/
html_logo = "_static/wbx-logo.png"
html_theme_options = {
    "source_repository": "https://github.com/google-research/weatherbenchX/",
    "sidebar_hide_name": True,
}
html_static_path = ["_static"]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
# html_show_sourcelink = (
#     False  # Remove 'view source code' from top of page (for html, not python)
# )
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
# set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
# nbsphinx_allow_errors = True  # Continue through Jupyter errors
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_typehints
autodoc_typehints = "signature"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"
# add_module_names = False  # Remove namespaces from class/method signatures

nb_execution_mode = "off"
