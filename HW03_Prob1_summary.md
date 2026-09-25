# CSCI E-89 Assignment 03, Problem 1: Summary

**Student:** Levente Papp
**Objective:** Reproduce the "Building an Image Classifier with PyTorch" section of
`10_neural_nets_with_pytorch.ipynb`, working with Claude Code one script per step.
Every script was saved in `prob1/`, run to confirm it works, and committed and pushed to `main`.

## Final results

| Metric | Value |
|---|---|
| Model | MLP: Flatten → 300 (ReLU) → 100 (ReLU) → 10 |
| Trainable parameters | 266,610 |
| Training | SGD, learning rate 0.1, batch size 32, 20 epochs, seed 42, CPU |
| Final training accuracy | 92.94% |
| Final validation accuracy | 88.90% |
| Best validation accuracy | 89.18% (epoch 19) |
| **Test accuracy** | **88.51%** (10,000 images) |

## Files

| File | Purpose |
|---|---|
| `prob1/01_setup.py` | Imports, random seed, device selection |
| `prob1/02_load_data.py` | Fashion MNIST download, train/validation split, DataLoaders |
| `prob1/03_show_samples.py` | Grid of sample training images, saved as `prob1/figures/samples.png` |
| `prob1/04_helpers.py` | `evaluate_tm()` and `train()` functions |
| `prob1/05_model.py` | `ImageClassifier`, loss, optimizer, accuracy metric |
| `prob1/06_train.py` | 20-epoch training; saves `prob1/outputs/history.json` and `model.pt` |
| `prob1/07_plot_accuracy.py` | Learning curves, saved as `prob1/figures/accuracy.png` |
| `prob1/08_evaluate.py` | Test accuracy and predictions for 3 validation images |
| `prob1/run_all.py` | Runs steps 1–8 in order from the terminal |

Each script is written as a Jupyter notebook cell. Paste them into a notebook saved in
the repo root, one script per cell, in order 1–8. Each cell uses names defined by the cells
before it, such as `torch`, `train_loader` and `model`. From the terminal, run
`python3 prob1/run_all.py` from the repo root; it runs the 8 scripts in order in one shared
namespace, the same way a notebook does. Steps 7 and 8 read the files saved by step 6. The
Fashion MNIST files download into `datasets/`, which `.gitignore` keeps out of git.

## Step by step

### Step 1: Setup (`01_setup.py`)
**Request:** Import torch, torchvision, torchmetrics, matplotlib and numpy, set a random
seed, and pick the device (CUDA, MPS or CPU).

**Done:** Seeds Python's `random`, NumPy and PyTorch (and CUDA if present) with 42.
Picks the device in the notebook's order: CUDA, then MPS, then CPU. Also imports `torch.nn`,
`torchvision.transforms.v2 as T` and `DataLoader` for the later steps. Running the
file prints the library versions and the device.

**Result:** PyTorch 2.14.0, TorchVision 0.29.0, TorchMetrics 1.9.0, NumPy 2.4.6,
Matplotlib 3.11.2. Device: `cpu` (the cloud container has no GPU).

**Notes:** The repo had no `main` branch, so the first push created it. A `.gitignore` was
added for `__pycache__/` and `datasets/`. PyTorch was installed from PyPI because the
container's network blocks download.pytorch.org.

### Step 2: Load the data (`02_load_data.py`)
**Request:** Download Fashion MNIST, convert the images to float tensors scaled to
[0, 1], split the 60,000 training images into 55,000 training and 5,000 validation, create
DataLoaders with batch size 32, and print the dataset sizes and the shape of one batch.

**Done:** `T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])`, then
`random_split` with seed 42. Train, validation and test loaders, with only the training
loader shuffled.

**Result:**
```
Training set:   55,000 images
Validation set: 5,000 images
Test set:       10,000 images
Batch images shape: (32, 1, 28, 28)  dtype: torch.float32
Batch labels shape: (32,)  dtype: torch.int64
Pixel value range:  [0.0, 1.0]
```

### Step 3: Show sample images (`03_show_samples.py`)
**Request:** Show a grid of sample training images with their class names.

**Done:** A 4×8 grid of the first 32 training images, each titled with its class name.
The figure is saved to `prob1/figures/samples.png` and opened in a window when a display is
available. The image was checked by eye, and every label matched its picture.

### Step 4: Training helpers (`04_helpers.py`)
**Request:** An evaluation function that computes a torchmetrics metric over a DataLoader,
and a training function that trains for n epochs and records training loss, training accuracy
and validation accuracy per epoch in a history dict, printing progress each epoch.

**Done:** `evaluate_tm()` (reset → update per batch → compute, in eval mode with no
gradients) and `train()`, based on the notebook's `evaluate_tm` and `train2`. The history
keys are `train_losses`, `train_metrics` and `valid_metrics`. Changes from the notebook:
the function is called `train`, the progress line says "accuracy", and the model is put back
in training mode once per epoch instead of once per batch.

**Result:** The end of the script trains a simple one-layer model for 3 epochs on 2,000
training and 500 validation images as a quick test. Training loss fell each epoch (1.22 →
0.73), and `evaluate_tm` gave the same validation accuracy as the last history entry (0.7460).

### Step 5: The model (`05_model.py`)
**Request:** Define `ImageClassifier` (Flatten, hidden layers of 300 and 100 units with
ReLU, 10 outputs). Create the model, cross-entropy loss, SGD with learning rate 0.1, and a
multiclass accuracy metric. Print the number of parameters.

**Result:** 266,610 parameters, which matches working it out by hand:
(784·300 + 300) + (300·100 + 100) + (100·10 + 10). A dummy batch of 32 gives
output shape `(32, 10)`.

### Step 6: Train (`06_train.py`)
**Request:** Set `n_epochs = 20` and train the model with the training function,
keeping the history.

**Done:** Resets the seed to 42 before training so the batch order is the same on every run.
Saves the history to `prob1/outputs/history.json` and the weights to
`prob1/outputs/model.pt` (about 1 MB), so steps 7 and 8 don't have to retrain.

**Result (selected epochs):**
```
Epoch 1/20,  train loss: 0.6057, train accuracy: 0.7810, valid accuracy: 0.7872
Epoch 5/20,  train loss: 0.3132, train accuracy: 0.8836, valid accuracy: 0.8762
Epoch 10/20, train loss: 0.2516, train accuracy: 0.9053, valid accuracy: 0.8708
Epoch 15/20, train loss: 0.2149, train accuracy: 0.9193, valid accuracy: 0.8900
Epoch 20/20, train loss: 0.1863, train accuracy: 0.9294, valid accuracy: 0.8890
```
Training took 223.6 s on the CPU.

### Step 7: Plot accuracy (`07_plot_accuracy.py`)
**Request:** Plot training accuracy per epoch with validation accuracy on the same chart,
with axis labels, a title and a legend.

**Done:** Training accuracy is a blue dashed line with circles and validation accuracy is an
orange solid line with squares, so the lines can be told apart without color. The y-axis is in
percent, and a label with an arrow marks the best validation epoch. Saved to
`prob1/figures/accuracy.png`.

**What the chart shows:** Both curves rise quickly for the first 4–5 epochs. After that,
training accuracy keeps climbing to 92.9%, while validation accuracy levels off around 88–89%
and moves up and down from epoch to epoch. The gap of about 4 points at epoch 20 shows mild
overfitting. The notebook plots training accuracy half an epoch to the left, because it is
averaged while the weights are still changing during the epoch. This plot uses whole epochs
for both lines.

### Step 8: Evaluate (`08_evaluate.py`)
**Request:** Report test-set accuracy, then for 3 validation images show the predicted
classes, true classes, class probabilities and top-4 predictions.

**Result:** Test accuracy was 88.51%, against 88.90% on validation, so validation was a fair
estimate of performance on new images. All 3 validation images were classified correctly:

| Image | True | Predicted | Top-4 predictions (probability) |
|---|---|---|---|
| 1 | Sneaker | Sneaker | Sneaker 0.962, Ankle boot 0.038, Sandal 0.000, Trouser 0.000 |
| 2 | Coat | Coat | Coat 0.982, Pullover 0.017, Shirt 0.001, T-shirt/top 0.000 |
| 3 | Pullover | Pullover | Pullover 0.685, Coat 0.243, Shirt 0.072, Dress 0.000 |

As in the notebook, the top-4 probabilities come from a softmax over only the 4 highest
scores, so they add up to 1 for each image. The model is confident on the sneaker and the coat
and much less sure about the pullover. Pullover, coat and shirt look alike in 28×28 grayscale,
and Fashion MNIST models usually confuse them most.

## Observations
- A simple 2-hidden-layer MLP reaches about 88.5% test accuracy on Fashion MNIST in 20 epochs.
- Validation accuracy levels off after about 8 epochs while training accuracy keeps rising,
  which is mild overfitting. Stopping early at the best validation epoch, or adding
  regularization, would be the natural next steps.
- In the 3 sample predictions, the probability the model gave to wrong classes went to
  classes that look alike: pullover, coat and shirt (upper-body garments), and sneaker and
  ankle boot (footwear). A confusion matrix would show whether that holds across the test set.
- All randomness is seeded with 42: weight initialization, the train/validation split and the
  batch order. Re-running the scripts on the same hardware and library versions gives the same
  numbers.
