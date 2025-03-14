# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
# Hardcode the absolute path (replace with your actual path)
PROJECT_ROOT = "C:\\Users\\HaochengLin\\Documents\\GitHub\\GEO-BOT-PROTOTYPE"
CLASSES_PATH = os.path.join(PROJECT_ROOT, "classes")

# Force-add these paths to sys.path
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CLASSES_PATH)

# Debugging: Print sys.path
print(f"✅ sys.path now includes:\n{sys.path}")

project = 'GEO-chatbot'
copyright = '2025, Haocheng Lin'
author = 'Haocheng Lin'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",     # Auto-generate documentation from docstrings
    "sphinx.ext.napoleon",    # Support for Google-style docstrings
    "sphinx.ext.doctest"      # Enable documentation testing. 
]

templates_path = ['_templates']
exclude_patterns = []

# Automatically include special/private methods and __init__
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__",
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
