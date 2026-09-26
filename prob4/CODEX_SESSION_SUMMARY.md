# CSCI E-89 Assignment 03 — Problem 4

## OpenAI Codex Session Summary

**Student:** Levente Papp

**Objective:** Reproduce the **Building an Image Classifier with PyTorch** workflow from `10_neural_nets_with_pytorch.ipynb` through an incremental series of prompts, scripts, and a final Jupyter notebook.

## Development Approach

The solution was developed incrementally with OpenAI Codex. Each prompt requested one focused step, and each resulting Python script was placed in the `prob4/` directory. After all eight scripts were created, their code was organized into a single notebook in executable order.

## Prompt and Implementation History

### Step 1 — Environment Setup

**Prompt:** Create `prob4/01_setup.py` with the required imports, a random seed, and automatic CUDA, MPS, or CPU device selection.

**Result:** The script imports PyTorch, TorchVision, TorchMetrics, Matplotlib, and NumPy; sets the NumPy and PyTorch random seeds to 42; seeds CUDA when available; and chooses CUDA first, MPS second, and CPU otherwise.

### Step 2 — Load and Prepare the Data

**Prompt:** Create `prob4/02_load_data.py` to download Fashion-MNIST, scale images to floating-point tensors in `[0, 1]`, split the 60,000 training examples into 55,000 training and 5,000 validation examples, create data loaders with batch size 32, and print dataset and batch sizes.

**Result:** The script creates a reproducible training/validation split and training, validation, and test data loaders. Only the training loader shuffles its examples.

### Step 3 — Display Sample Images

**Prompt:** Create `prob4/03_show_samples.py` to display a grid of training images with their class names.

**Result:** The script creates a 4-by-8 grid of Fashion-MNIST examples, labels each image with its human-readable class name, displays the plot, and saves it as `prob4/figures/samples.png`.

### Step 4 — Evaluation and Training Helpers

**Prompt:** Create `prob4/04_helpers.py` with an evaluation function and a training function that records training loss, training accuracy, and validation accuracy for each epoch.

**Result:** The `evaluate()` function computes a TorchMetrics metric over a complete data loader without gradient tracking. The `train()` function performs the optimization loop, evaluates validation accuracy after every epoch, stores all requested measurements in a history dictionary, and prints progress.

### Step 5 — Define the Model

**Prompt:** Create `prob4/05_model.py` with an `ImageClassifier` containing a flattening layer, hidden layers of 300 and 100 ReLU units, and 10 outputs. Also create cross-entropy loss, an SGD optimizer with learning rate 0.1, and a multiclass accuracy metric, and print the parameter count.

**Result:** The requested multilayer perceptron and training objects were created. The model has 266,610 trainable parameters.

### Step 6 — Train the Model

**Prompt:** Create `prob4/06_train.py`, train the model for 20 epochs with the training helper, and retain the returned history.

**Result:** The script sets `n_epochs = 20`, invokes `train()`, and stores the per-epoch measurements in `history`.

### Step 7 — Plot Accuracy

**Prompt:** Create `prob4/07_plot_accuracy.py` to plot training and validation accuracy on the same labeled chart.

**Result:** The script plots both accuracy series by epoch with a title, axis labels, grid, and legend. It displays the chart and saves it as `prob4/figures/accuracy.png`.

### Step 8 — Evaluate Predictions

**Prompt:** Create `prob4/08_evaluate.py` to report test accuracy and, for three validation images, show predicted classes, true classes, class probabilities, and the top four predictions.

**Result:** The script evaluates the full test loader, applies softmax to three validation predictions, and reports the requested class information and probabilities.

### Final Notebook Assembly

**Prompt:** Combine all scripts into a Jupyter notebook with cells in the proper executable order.

**Result:** `prob4/e89_Papp_Levente_HW03_Prob4.ipynb` was created with the following sequence:

1. Setup and device selection
2. Fashion-MNIST loading and preparation
3. Sample-image visualization
4. Evaluation and training helper functions
5. Model, loss, optimizer, and metric creation
6. Twenty-epoch training
7. Training and validation accuracy plot
8. Test evaluation and detailed validation predictions

Explanatory Markdown cells were added between the code sections, and the student's name appears at the beginning of the notebook.

## Verification Performed

- Every standalone Python script was syntax-checked with `python -m py_compile` during incremental development.
- Static checks confirmed that the requested constants, functions, classes, layers, metrics, training operations, and plot labels were present.
- The notebook was parsed as valid notebook JSON (`nbformat` 4), and the source of every code cell was compiled successfully.
- Git whitespace checks were performed before commits.

## Execution-Environment Limitation

The final notebook could not be fully executed in the Codex container because PyTorch, TorchVision, TorchMetrics, Matplotlib, NumPy, Jupyter, and nbconvert were not installed. Attempts to install them with `pip` were rejected by the environment's outbound network proxy with HTTP 403 errors, and the Ubuntu package repositories were also unreachable.

For accuracy and academic integrity, Codex did **not** fabricate training metrics, predictions, plots, execution counts, or notebook outputs. The notebook should be run from top to bottom in a Python environment containing the required packages and with network access for the initial Fashion-MNIST download. After successful execution, the notebook can be saved with its genuine outputs and exported to HTML.

## Files Produced

- `prob4/01_setup.py`
- `prob4/02_load_data.py`
- `prob4/03_show_samples.py`
- `prob4/04_helpers.py`
- `prob4/05_model.py`
- `prob4/06_train.py`
- `prob4/07_plot_accuracy.py`
- `prob4/08_evaluate.py`
- `prob4/e89_Papp_Levente_HW03_Prob4.ipynb`
- `prob4/CODEX_SESSION_SUMMARY.md`

Together, these files document both the incremental Codex interaction and the resulting Fashion-MNIST classification workflow for Assignment 03, Problem 4.
