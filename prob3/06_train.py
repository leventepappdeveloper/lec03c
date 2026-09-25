# CSCI E-89 Assignment 03, Problem 3 -- Step 6: train the model
# Author: Levente Papp
# --- standalone bootstrap (removed in the notebook) ---
from _pipeline import run_prerequisites
run_prerequisites(__file__, globals())
# --- end bootstrap ---

n_epochs = 20
# history holds per-epoch train_loss, train_accuracy and valid_accuracy
history = train(model, optimizer, xentropy, accuracy, train_loader,
                valid_loader, n_epochs)
