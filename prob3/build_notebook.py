"""
CSCI E-89 Assignment 03, Problem 3 -- build the notebook from the step scripts
Author: Levente Papp

Puts each script 01-08 into its own code cell, in execution order, with a
markdown cell above it naming the source script. The standalone bootstrap
block is dropped, since notebook cells already share one namespace.

Usage (from the repo root):  python prob3/build_notebook.py
"""

import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROB3_DIR = Path(__file__).resolve().parent
NOTEBOOK = PROB3_DIR / "e89_Papp_Levente_HW03_Prob3.ipynb"
BOOTSTRAP = re.compile(
    r"# --- standalone bootstrap.*?# --- end bootstrap ---\n+", re.DOTALL)

TITLE = """\
# CSCI E-89 Deep Learning — Assignment 03, Problem 3

* **Name:** Levente Papp
* **Course:** CSCI E-89 Deep Learning, Harvard Extension School
* **Assignment:** Assignment 03, Problem 3

This notebook rebuilds the *Building an Image Classifier with PyTorch* pipeline
from Aurélien Géron's chapter 10 notebook (`10_neural_nets_with_pytorch.ipynb`)
in a single Claude Code session. It trains a multilayer perceptron on
Fashion MNIST and adds a plot of training and validation accuracy.

Each code cell below is one standalone script from the `prob3/` folder of the
repository, in execution order. The only change from the scripts is that their
short "standalone bootstrap" block is removed: in the scripts it runs the earlier
steps when a script is launched on its own, which a notebook doesn't need."""

DESCRIPTIONS = {
    "01_setup.py": """\
Imports PyTorch, torchvision, torchmetrics, Matplotlib and NumPy, fixes the
random seed (42) for Python, NumPy and PyTorch so results are reproducible, and
picks the device: a CUDA GPU if present, else an Apple Silicon (MPS) GPU, else
the CPU.""",
    "02_load_data.py": """\
Downloads Fashion MNIST with torchvision and converts each image to a
`float32` tensor scaled to [0, 1]. The 60,000 training images are split
randomly into 55,000 for training and 5,000 for validation; the 10,000 test
images are kept separate. Each set is wrapped in a `DataLoader` with batch
size 32 (only the training loader shuffles). The cell prints the dataset sizes
and the shape of one batch: `[32, 1, 28, 28]` = batch, channel, rows, columns.""",
    "03_show_samples.py": """\
Shows a 4 × 8 grid of the first 32 training images with their class names, as
a quick check that images and labels line up.""",
    "04_helpers.py": """\
Defines two helpers:

* `evaluate_tm()` runs the model over a `DataLoader` in evaluation mode without
  gradients and returns a torchmetrics metric (here, accuracy).
* `train()` trains for `n_epochs`. For each batch it does the forward pass,
  computes the loss, backpropagates, and takes an optimizer step. After each
  epoch it records the mean training loss, the training accuracy and the
  validation accuracy in a `history` dict, and prints them.""",
    "05_model.py": """\
Defines `ImageClassifier`, a multilayer perceptron: `Flatten` (28 × 28 → 784),
a hidden layer of 300 ReLU units, one of 100 ReLU units, and 10 output logits,
one per class. The cell also creates the cross-entropy loss, an SGD optimizer
with learning rate 0.1, and a 10-class accuracy metric, then prints the model
and its parameter count:
784·300+300 + 300·100+100 + 100·10+10 = 266,610.""",
    "06_train.py": """\
Trains the model for `n_epochs = 20` and keeps the per-epoch results in
`history`. This takes about 3 minutes on a CPU.""",
    "07_plot_accuracy.py": """\
Plots training and validation accuracy per epoch on the same chart. Training
accuracy is averaged over each epoch while the weights are still changing, so
it is drawn half an epoch to the left, as in Géron's notebook. The gap between
the curves (training above validation) shows the model starting to overfit.""",
    "08_evaluate.py": """\
Reports the final training and validation accuracy and the accuracy on the
test set. Then, for the first 3 validation images, it prints the predicted and
true classes, the softmax probabilities for all 10 classes, and the top-4
predictions with their probabilities (renormalized over those 4 classes, as in
Géron's notebook).""",
}


def main():
    cells = [new_markdown_cell(TITLE)]
    for script in sorted(PROB3_DIR.glob("0[1-8]_*.py")):
        step = int(script.name[:2])
        code = BOOTSTRAP.sub("", script.read_text()).strip()
        cells.append(new_markdown_cell(
            f"## Step {step}: `prob3/{script.name}`\n\n"
            + DESCRIPTIONS[script.name]))
        cells.append(new_code_cell(code))
    notebook = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    })
    nbformat.write(notebook, NOTEBOOK)
    print(f"Wrote {NOTEBOOK} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
