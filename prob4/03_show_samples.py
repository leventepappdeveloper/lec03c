"""
CSCI E-89 Assignment 03, Problem 4 -- Step 3: Show sample images
Author: Levente Papp

Display a grid of Fashion-MNIST training images with their class names.
"""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Load Step 2 automatically when this file is run as a standalone script. When
# the scripts are combined into a notebook and run in order, reuse its data.
if "train_data" not in globals() or "train_and_valid_data" not in globals():
    step_2 = runpy.run_path(BASE_DIR / "02_load_data.py")
    train_data = step_2["train_data"]
    train_and_valid_data = step_2["train_and_valid_data"]

N_ROWS = 4
N_COLUMNS = 8
FIGURE_PATH = BASE_DIR / "figures" / "samples.png"
class_names = train_and_valid_data.classes

figure, axes = plt.subplots(
    N_ROWS,
    N_COLUMNS,
    figsize=(N_COLUMNS * 1.5, N_ROWS * 1.8),
)

for index, axis in enumerate(axes.flat):
    image, label = train_data[index]
    # Fashion-MNIST images have shape [1, 28, 28]; remove the channel axis so
    # Matplotlib can render each grayscale image.
    axis.imshow(image.squeeze(0), cmap="binary")
    axis.set_title(class_names[label], fontsize=9)
    axis.axis("off")

figure.suptitle("Fashion-MNIST sample training images")
figure.tight_layout()

FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(FIGURE_PATH, dpi=100, bbox_inches="tight")
print(f"Displayed {N_ROWS * N_COLUMNS} samples and saved the grid to {FIGURE_PATH}")
plt.show()
