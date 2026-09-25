"""
CSCI E-89 Assignment 03, Problem 4 -- Step 6: Train the classifier
Author: Levente Papp

Train the Fashion-MNIST classifier for 20 epochs and retain its history.
"""

import runpy
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Load the earlier steps when this file is executed directly. In the final
# notebook, the variables already exist because its cells are run in order.
required_names = {
    "model",
    "optimizer",
    "loss_function",
    "accuracy_metric",
    "train_loader",
    "valid_loader",
    "train",
}
if not required_names.issubset(globals()):
    data_state = runpy.run_path(BASE_DIR / "02_load_data.py")
    helper_state = runpy.run_path(BASE_DIR / "04_helpers.py")
    model_state = runpy.run_path(BASE_DIR / "05_model.py")

    train_loader = data_state["train_loader"]
    valid_loader = data_state["valid_loader"]
    train = helper_state["train"]
    model = model_state["model"]
    optimizer = model_state["optimizer"]
    loss_function = model_state["loss_function"]
    accuracy_metric = model_state["accuracy_metric"]

n_epochs = 20
history = train(
    model,
    optimizer,
    loss_function,
    accuracy_metric,
    train_loader,
    valid_loader,
    n_epochs,
)

print(f"Training complete; history contains {len(history['train_loss'])} epochs")
