"""Parse python files and look for inline sphinx comments that represents examples.

I.e comments that starts with ".. code-block:: python" are parsed and the python code is returned
as a strings.

We then write example code to a temp file module and import to check for any obvious errors.
"""
import os
import re
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np


class TestSphinxExamples(unittest.TestCase):

    def setUp(self):
        self.debug = False

    def test_sphinx_examples(self):
        """execute strings of python code and check for obvious errors.
        """
        _root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", ".."))
        directory_path = os.path.join(_root_path, "darling")
        python_files, code_snippets = self.extract_sphinx_code(directory_path)
        # simply write examples to tmp file and import.
        tmp_file = os.path.abspath(
            os.path.join(os.path.abspath(__file__), "..", "..", "tests", "tmp.py")
        )
        for pyfile,snippet in zip(python_files, code_snippets):

            try:
                exec(snippet)
            except:
                errmsg = "Sphinx docs example in "+pyfile+" failed to execute: \n>>\n" + snippet + '>>'
                raise ValueError(errmsg)

    def extract_sphinx_code(self, directory):
        """Grab all python docstrings in all files in a directory and parse inline python code.

        I.e comments that starts with ".. code-block:: python" are parsed and the python code is returned
        as a strings.r

        Args:
            directory (:obj: `str`): path to directory to traverse and parse.

        Returns:
            (:obj:`list` of `str`):: The python code snippets.
        """
        # Pattern to capture entire Sphinx docstrings (triple-quoted strings)
        sphinx_docstring_pattern = re.compile(r"\"\"\"(.*?)\"\"\"", re.DOTALL)
        # Updated pattern to capture Python code blocks within a Sphinx docstring, stopping at a line with reduced indentation
        code_block_pattern = re.compile(
            r"\.\. code-block:: python\n((?:\n| {4}.*\n)+)", re.DOTALL
        )

        code_snippets = []
        python_files = []

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                        # Find all Sphinx-style docstrings
                        for docstring in sphinx_docstring_pattern.findall(content):
                            reading = False
                            doc_lines = docstring.split("\n")
                            for i,line in enumerate(doc_lines):
                                # extract example code
                                if ".. code-block:: python" in line:
                                    reading = True
                                    example = ""
                                    func_comment = line.startswith("    ")
                                elif reading:
                                    if func_comment:
                                        if line.startswith("        ") and "Args:" not in line:
                                            example += line[8:] + "\n"
                                        elif bool(re.match(r"^ {4}\S", line)) or i==len(doc_lines)-1:
                                            reading = False
                                            code_snippets.append(example)
                                            python_files.append(file)
                                    else:
                                        if line.startswith("    "):
                                            example += line[4:] + "\n"
                                        elif bool(re.match(r"^ {0}\S", line)) or i==len(doc_lines)-1:
                                            reading = False
                                            code_snippets.append(example)
                                            python_files.append(file)
        return python_files, code_snippets


if __name__ == "__main__":
    unittest.main()
