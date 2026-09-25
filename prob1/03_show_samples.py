"""
CSCI E-89 Assignment 03, Problem 1 -- Step 3: Show sample images
Author: Levente Papp

Displays a grid of sample training images labeled with their class names.
The figure is saved to prob1/figures/samples.png and also shown on screen
when a display is available.
"""

import importlib
from pathlib import Path

setup = importlib.import_module("01_setup")
data = importlib.import_module("02_load_data")
plt = setup.plt

N_ROWS, N_COLS = 4, 8
FIG_PATH = Path(__file__).resolve().parent / "figures" / "samples.png"

class_names = data.train_and_valid_data.classes

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 1.5, N_ROWS * 1.8))
for i, ax in enumerate(axes.flat):
    image, label = data.train_data[i]
    # image has shape [1, 28, 28]; drop the channel dim for imshow
    ax.imshow(image.squeeze(0), cmap="binary")
    ax.set_title(class_names[label], fontsize=9)
    ax.axis("off")
fig.suptitle("Fashion MNIST: sample training images")
fig.tight_layout()

FIG_PATH.parent.mkdir(exist_ok=True)
fig.savefig(FIG_PATH, dpi=100)
print(f"Saved {N_ROWS * N_COLS} sample images to {FIG_PATH}")

# plt.show() only has an effect with an interactive backend (not on a headless machine)
if plt.get_backend().lower() != "agg":
    plt.show()
