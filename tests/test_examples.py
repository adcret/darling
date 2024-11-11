"""Parse python files and look for inline sphinx comments that represents examples. Check that these runs without errors.
"""

import os
import re
import tempfile

import matplotlib.pyplot as plt
import numpy as np


def test_code():
    _root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", ".."))
    directory_path = os.path.join(_root_path, "darling")
    code_snippets = extract_sphinx_code(directory_path)
    # simply write examples to tmp file and import.
    tmp_file = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "..", "..", "tests", "tmp.py")
    )
    for snippet in code_snippets:
        with open(tmp_file, "w") as f:
            f.write(snippet)
            import tmp
    os.remove(tmp_file)


def extract_sphinx_code(directory):
    # Pattern to capture entire Sphinx docstrings (triple-quoted strings)
    sphinx_docstring_pattern = re.compile(r"\"\"\"(.*?)\"\"\"", re.DOTALL)
    # Updated pattern to capture Python code blocks within a Sphinx docstring, stopping at a line with reduced indentation
    code_block_pattern = re.compile(
        r"\.\. code-block:: python\n((?:\n| {4}.*\n)+)", re.DOTALL
    )

    code_snippets = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                    # Find all Sphinx-style docstrings
                    for docstring in sphinx_docstring_pattern.findall(content):
                        reading = False
                        for line in docstring.split("\n"):
                            # extract example code
                            if ".. code-block:: python" in line:
                                reading = True
                                example = ""
                            if reading and ".. code-block:: python" not in line:
                                if line.startswith("        ") and "Args:" not in line:
                                    example += line[8:] + "\n"
                                elif bool(re.match(r"^ {4}\S", line)):
                                    reading = False
                                    code_snippets.append(example)
    return code_snippets


if __name__ == "__main__":
    pass