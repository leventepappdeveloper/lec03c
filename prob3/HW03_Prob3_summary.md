# CSCI E-89 Assignment 03, Problem 3 — Session Summary

**Author:** Levente Papp
**Task:** In one new Claude Code session and a single long prompt, rebuild the
*Building an Image Classifier with PyTorch* section of Géron's chapter 10
notebook (`10_neural_nets_with_pytorch.ipynb`) as standalone scripts, add a
training-accuracy plot, package the scripts as a notebook, execute it, and
export it to HTML. Everything lives in `prob3/`, so the Problem 1 and 2 files
are unchanged.

## What was built

| File | Purpose |
|---|---|
| `01_setup.py` | Imports (torch, torchvision, torchmetrics, matplotlib, numpy), seed 42, device selection (CUDA → MPS → CPU) |
| `02_load_data.py` | Downloads Fashion MNIST, converts to `float32` in [0, 1], splits 55,000 / 5,000 train/validation, DataLoaders with batch size 32, prints sizes and one batch's shape |
| `03_show_samples.py` | 4 × 8 grid of training images with class names |
| `04_helpers.py` | `evaluate_tm()` (a torchmetrics metric over a DataLoader) and `train()` (n epochs, records training loss, training accuracy and validation accuracy per epoch in `history`, prints progress) |
| `05_model.py` | `ImageClassifier` (Flatten → 300 ReLU → 100 ReLU → 10), cross-entropy loss, SGD with lr = 0.1, multiclass accuracy; prints the parameter count (266,610) |
| `06_train.py` | `n_epochs = 20`, trains and keeps `history` |
| `07_plot_accuracy.py` | Training and validation accuracy on one chart, with axis labels, a title and a legend |
| `08_evaluate.py` | Test accuracy, then predicted/true classes, class probabilities and top-4 predictions for 3 validation images |
| `_pipeline.py`, `run_all.py` | Let each step run on its own (`python prob3/06_train.py` runs 01–05 first), or all eight in order |
| `build_notebook.py` | Builds the notebook: a title cell, then a markdown cell and a code cell per script |
| `e89_Papp_Levente_HW03_Prob3.ipynb` | The executed notebook, with all outputs and both plots saved |
| `e89_Papp_Levente_HW03_Prob3.html` | HTML export (`jupyter nbconvert e89_Papp_Levente_HW03_Prob3.ipynb --to html`) |

The code follows Géron's notebook (`evaluate_tm`, `train2`, `ImageClassifier`,
the same seeds and split). Differences from it:

* `train()` records accuracy under clearer key names (`train_accuracy`,
  `valid_accuracy`) and calls `model.train()` once per epoch rather than once
  per batch.
* Step 8 moves predictions to the CPU before printing them, so it works on
  CUDA as well as MPS and CPU.
* Step 7 is new: it plots training and validation accuracy per epoch.

## Problems hit and how they were fixed

1. **PyTorch's wheel index was blocked.** Installing the CPU-only build from
   `download.pytorch.org` was refused by the session's network policy.
   Installed `torch`/`torchvision` from PyPI instead. There is no GPU here, so
   everything ran on the CPU.
2. **The scripts share state, but each had to be runnable on its own.** Like
   notebook cells, `06_train.py` needs the `model` from `05_model.py`, and so
   on. Each script starts with a short bootstrap block that, when the script is
   launched on its own, runs the earlier steps into the same namespace
   (`_pipeline.py`), and does nothing when the steps already share a namespace.
   `build_notebook.py` removes that block from the notebook cells.
3. **Paths had to work from any directory and inside the notebook.** A notebook
   has no `__file__`, so `01_setup.py` uses the script's folder when there is
   one and the working directory otherwise. The dataset (`prob3/datasets/`) and
   the plots the scripts save (`prob3/figures/`) are in `.gitignore`; the
   notebook embeds its own copies of the plots.
4. **Title cell rendering.** In the first build, the Name / Course / Assignment
   lines would have rendered as one run-on line in Markdown. Changed them to a
   bullet list, then rebuilt and re-executed the notebook.

The notebook ran end-to-end without errors on the first execution. Its numbers
match the standalone scripts exactly, because all seeds are fixed.

## Final results (20 epochs, CPU, seed 42)

| Metric | Accuracy |
|---|---|
| Training (epoch 20) | **0.9286** |
| Validation (epoch 20) | **0.8788** (best: 0.8886 at epoch 16) |
| Test | **0.8824** |

Training accuracy keeps rising while validation accuracy levels off around
0.87–0.89 after about epoch 9, so the model is starting to overfit. All 3
validation images checked in step 8 (Sneaker, Coat, Pullover) were predicted
correctly. The Pullover was the least certain: 0.59, with Coat and Shirt
next.
