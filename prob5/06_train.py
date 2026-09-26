# CSCI E-89 Assignment 03, Problem 5 -- Step 6: train for 20 epochs
# Author: Levente Papp

n_epochs = 20
# Retain all three per-epoch series for reporting and visualization.
history = train(model, optimizer, loss_function, accuracy_metric, train_loader,
                valid_loader, n_epochs)
