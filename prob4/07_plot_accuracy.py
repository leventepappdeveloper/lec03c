"""
CSCI E-89 Assignment 03, Problem 4 -- Step 7: Plot accuracy
Author: Levente Papp

Plot the training and validation accuracy recorded after each epoch.
"""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Running this script directly also runs the training step. In the final
# notebook, reuse the history produced by the preceding training cell.
if "history" not in globals():
    training_state = runpy.run_path(BASE_DIR / "06_train.py")
    history = training_state["history"]

training_accuracy = history["train_accuracy"]
validation_accuracy = history["valid_accuracy"]
epochs = range(1, len(training_accuracy) + 1)

figure, axis = plt.subplots(figsize=(8, 5))
axis.plot(
    epochs,
    training_accuracy,
    marker="o",
    linestyle="--",
    label="Training accuracy",
)
axis.plot(
    epochs,
    validation_accuracy,
    marker="s",
    linestyle="-",
    label="Validation accuracy",
)

axis.set_xlabel("Epoch")
axis.set_ylabel("Accuracy")
axis.set_title("Fashion-MNIST training and validation accuracy")
axis.set_xticks(list(epochs))
axis.grid(alpha=0.3)
axis.legend()
figure.tight_layout()

figure_path = BASE_DIR / "figures" / "accuracy.png"
figure_path.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(figure_path, dpi=120, bbox_inches="tight")
print(f"Saved accuracy plot to {figure_path}")
plt.show()
