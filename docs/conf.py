"""Sphinx configuration for the okapi documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

project = "okapi"
author = "Alexander Dittrich"
copyright = "2026, Alexander Dittrich"
release = "0.2.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
myst_enable_extensions = ["colon_fence", "dollarmath"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "*.json"]

# The heavy GPU stack is mocked so the docs build without CUDA wheels.
autodoc_mock_imports = [
    "jax",
    "flax",
    "optax",
    "orbax",
    "mujoco",
    "mujoco_playground",
    "brax",
    "warp",
    "wandb",
]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True

html_theme = "furo"
html_title = "okapi"
html_logo = "_static/okapi.png"
html_static_path = ["_static"]
