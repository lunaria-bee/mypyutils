#!/usr/bin/env python

'''Initialize Sphinx docs with the settings I like.'''


import argparse
from pathlib import Path
import re
import shutil
import subprocess as subp
import sys


EXTENSIONS = '''extensions = [
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
    'private-members',
    'show-inheritance',
    'show-module-summary',
]

# sphinx.ext.intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
}
'''

HTML_OPTS = '''html_theme_options = {
    'secondary_sidebar_items': [],
    'footer_start': ['copyleft', 'sphinx-version', 'theme-version'],
    'footer_end': ['sourcelink'],
}
html_sidebars = {
    '**': ['page-toc'],
}
'''

GITIGNORE = '''# Documentation output
autoapi/
_build/
'''

README = '''To build documentation, run the following in this directory:

1. `$ pip -r requirements.txt`
2. `$ make html`

Documentation can now be accessed at `docs/_build/html/index.html`.
'''

REQUIREMENTS = '''pydata-sphinx-theme
Sphinx
sphinx-autoapi
sphinx-design
'''


parser = argparse.ArgumentParser()
parser.add_argument(
    'dir',
    type=Path,
    default='./docs/',
    nargs='?',
    help="Passed to sphinx-quickstart's PROJECT_DIR argument. Defaults to 'docs/'.",
)


def _main(argv):
    if '--' in argv:
        split_idx = argv.index('--')
        main_argv = argv[:split_idx]
        sphinx_argv = argv[split_idx+1:]
    else:
        main_argv = argv
        sphinx_argv = []

    args, sphinx_args = parser.parse_known_args(main_argv)

    subp.run(['sphinx-quickstart', args.dir, '--no-sep'] + sphinx_argv)

    ## index.rst ##
    with open(args.dir / 'index.rst') as index_file:
        index_lines = list(index_file.readlines())
    with open(args.dir / 'index.rst', 'w') as index_file:
        for line in index_lines:
            # Remove trailing documentation from title.
            match = re.fullmatch(r'(.+?) documentation\n', line)
            if match:
                line = f"{match.group(1)}\n"

            # Remove unwanted lines.
            if (
                    line.startswith("Add your content")
                    or line.startswith("`reStructuredText ")
                    or line.startswith("documentation for details.")
                    or re.fullmatch(r' *:caption: Contents:\n', line)
            ):
                line = ""

            index_file.write(line)

    ## conf.py ##
    write_on_delay = []
    with open(args.dir / 'conf.py') as conf_file:
        conf_lines = list(conf_file.readlines())
    with open(args.dir / 'conf.py', 'w') as conf_file:
        for line in conf_lines:
            delay_written = set()
            for i in range(len(write_on_delay)):
                write_on_delay[i][0] -= 1
                if write_on_delay[i][0] == 0:
                    conf_file.write(write_on_delay[i][1])
                    delay_written.add(i)
                for i in delay_written:
                    write_on_delay.pop(i)

            # Save project title.
            match = re.fullmatch(r"project = '(.*?)'\n", line)
            if match:
                project_title = match.group(1)

            # Fix copyright info.
            match = re.fullmatch(r"copyright = '([0-9]+?), (.+?)'\n", line)
            if match:
                year, author = match.groups()
                line = f"copyright = 'CC BY 4.0, {year}, {author}'\n"

            # Add top-level project dir to path.
            if line.startswith("# -- General configuration --"):
                write_on_delay.append([
                    3,
                    "from pathlib import Path\n"
                    "import sys\n"
                    "sys.path.insert(0, str(Path(__file__).parent.parent))\n\n",
                ])

            # Add extensions.
            if line == "extensions = []\n":
                line = EXTENSIONS

            # Set HTML theme.
            if line.startswith("html_theme"):
                line = "html_theme = 'pydata_sphinx_theme'\n"

            conf_file.write(line)

        conf_file.write(f"html_title='{project_title}'\n")
        conf_file.write(HTML_OPTS)

    ## .gitignore ##
    with open(args.dir / '.gitignore', 'w') as gitignore_file:
        gitignore_file.write(GITIGNORE)

    ## README.md ##
    with open(args.dir / 'README.md', 'w') as readme_file:
        readme_file.write(README)

    ## requirements.txt ##
    with open(args.dir / 'requirements.txt', 'w') as requirements_file:
        requirements_file.write(REQUIREMENTS)

    ## _templates ##
    shutil.copytree(
        Path(__file__).parent / 'resources/sphinx_templates/',
        args.dir / '_templates/',
        dirs_exist_ok=True,
    )


if __name__ == '__main__': _main(sys.argv[1:])
