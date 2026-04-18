# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mypyutils'
copyright = 'CC BY 4.0, 2025, lunaria'
author = 'lunaria'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, str(Path(__file__).parent.parent))

extensions = [
    'autoapi.extension',
    'sphinx_design',
    'sphinx.ext.intersphinx',
]

# autoapi.extension
autoapi_type = 'python'
autoapi_dirs = ['..']
autoapi_ignore = [
    '*/.*/*',
    '*/docs/*',
    '*/__pycache__/*',
]
autoapi_add_toctree_entry = False
autoapi_python_use_implicit_namespaces = True
autoapi_template_dir = './_templates/autoapi/'
autoapi_python_class_content = 'both'
autoapi_options =  [
    'members',
    'inherited-members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'imported-members',
]

# sphinx.ext.intersphinx
intersphinx_mapping = {
    'huggingface_hub': ('https://huggingface.co/docs/huggingface_hub/master/en/', None),
    'python': ('https://docs.python.org/3/', None),
    'transformers': ('https://huggingface.co/docs/transformers/master/en/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_title = "mypyutils"
html_theme_options = {
    'secondary_sidebar_items': [],
    'footer_start': ['copyleft', 'sphinx-version', 'theme-version'],
    'footer_end': ['sourcelink'],
}
html_sidebars = {
    '**': ['page-toc'],
}
