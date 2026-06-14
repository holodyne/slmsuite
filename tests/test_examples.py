import os
import importlib.util
import logging

import pytest

pytest_notebook = pytest.importorskip("pytest_notebook", reason="pytest_notebook not installed")
execute_notebook = pytest_notebook.execution.execute_notebook
nbformat = pytest.importorskip("nbformat", reason="nbformat not installed")

# Suppress verbose debug logging from notebook execution (kernel messages, etc.).
logging.getLogger("pytest_notebook").setLevel(logging.WARNING)

# Load docs/source/examples.py by file path (it's not an installed package).
_examples_module_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "docs", "source", "examples.py"
)
_spec = importlib.util.spec_from_file_location("docs_examples", _examples_module_path)
_examples_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_examples_mod)

download_example_notebooks = _examples_mod.download_example_notebooks
get_sphinx_examples = _examples_mod.get_sphinx_examples

# FUTURE: add a fixture to test notebooks with or without cupy.
@pytest.mark.slow
def test_examples(subtests):

    # First download the example notebooks (if not already downloaded)
    examples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_examples")
    download_example_notebooks(
        examples_path=examples_path,
        images_path=None, # Don't download images
    )

    # Get the list of example notebooks.
    notebooks = get_sphinx_examples()

    for nb_name in notebooks:
        nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", nb_name + ".ipynb")
        nb_path_run = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", nb_name + "_run.ipynb")

        with subtests.test(f"testing {nb_name}"):
            assert os.path.isfile(nb_path), "Notebook not found."

            with open(nb_path, "r", encoding="utf8") as f:
                nb = nbformat.read(f, as_version=4)

            # Remove cells marked as requiring hardware not available in CI.
            nb.cells = [cell for cell in nb.cells if "slmsuite_experimental" not in cell.metadata]

            # Execute the notebook.
            exec_result = execute_notebook(
                nb,
                cwd=os.path.dirname(nb_path),
                timeout=1200,
            )

            # Always save the executed notebook for inspection.
            with open(nb_path_run, "w", encoding="utf8") as f:
                nbformat.write(exec_result.notebook, f)

            # Fail the subtest if a cell raised an error.
            if exec_result.exec_error is not None:
                raise exec_result.exec_error