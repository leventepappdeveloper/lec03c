"""
CSCI E-89 Assignment 03, Problem 2 -- Wrap the Problem 1 code in a notebook
Author: Levente Papp

Builds HW03_Prob2_notebook.ipynb in the repo root: one labeled Markdown cell
plus one code cell per Problem 1 script, in executable order (steps 1-8).

Usage (from the repo root):
    python3 prob2/build_notebook.py
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_ROOT = Path(__file__).resolve().parent.parent
PROB1_DIR = REPO_ROOT / "prob1"
NOTEBOOK_PATH = REPO_ROOT / "HW03_Prob2_notebook.ipynb"

STEPS = [
    ("01_setup.py", "Setup",
     "Imports, random seed (42), and device selection (CUDA, MPS, or CPU)."),
    ("02_load_data.py", "Load the data",
     "Downloads Fashion MNIST, scales pixels to [0, 1], splits the 60,000 "
     "training images into 55,000 training / 5,000 validation, and builds "
     "DataLoaders with batch size 32."),
    ("03_show_samples.py", "Show sample images",
     "A 4×8 grid of training images labeled with their class names."),
    ("04_helpers.py", "Training helpers",
     "`evaluate_tm()` computes a torchmetrics metric over a DataLoader; "
     "`train()` trains for n epochs and records training loss, training "
     "accuracy, and validation accuracy per epoch. Ends with a quick smoke "
     "test on a small subset."),
    ("05_model.py", "The model",
     "`ImageClassifier` (Flatten → 300 → ReLU → 100 → ReLU → 10), "
     "cross-entropy loss, SGD with learning rate 0.1, and multiclass accuracy."),
    ("06_train.py", "Train the model",
     "Trains for 20 epochs and saves the history and weights to "
     "`prob1/outputs/`. Takes about 4 minutes on a CPU."),
    ("07_plot_accuracy.py", "Plot training accuracy",
     "Training accuracy per epoch, with validation accuracy for comparison."),
    ("08_evaluate.py", "Evaluate",
     "Test-set accuracy, then predicted vs. true classes, class "
     "probabilities, and top-4 predictions for 3 validation images."),
]

INTRO = """# CSCI E-89 Assignment 03: Building an Image Classifier with PyTorch
**Author:** Levente Papp

This notebook wraps the code generated with Claude Code in Problem 1. It
follows the "Building an Image Classifier with PyTorch" section of
`10_neural_nets_with_pytorch.ipynb` and trains an MLP on Fashion MNIST.

Each code cell is one script from the `prob1/` folder of the repository,
in executable order. The Markdown cell above each code cell names its script.

**Before running:**
- Open this notebook from the **repository root** (next to the `prob1/`
  folder). The cells use paths like `prob1/outputs/`.
- Install the requirements: `pip install torch torchvision torchmetrics matplotlib numpy`
- Run the cells top to bottom (**Kernel → Restart & Run All**). Each cell
  uses names defined by the cells before it."""


def build():
    cells = [new_markdown_cell(INTRO)]
    for n, (filename, title, description) in enumerate(STEPS, start=1):
        cells.append(new_markdown_cell(
            f"## Step {n}: {title}\n"
            f"**Script:** `prob1/{filename}`\n\n"
            f"{description}"))
        cells.append(new_code_cell((PROB1_DIR / filename).read_text().rstrip()))
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.validate(nb)
    nbformat.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH} ({len(cells)} cells, {len(STEPS)} code cells)")


if __name__ == "__main__":
    build()
