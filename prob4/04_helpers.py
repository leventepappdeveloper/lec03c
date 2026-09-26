"""
CSCI E-89 Assignment 03, Problem 4 -- Step 4: Training helpers
Author: Levente Papp

Define reusable functions for evaluating and training a PyTorch classifier.
"""

import torch


def evaluate(model, data_loader, metric):
    """Compute a torchmetrics metric over every batch in ``data_loader``."""
    device = next(model.parameters()).device
    metric = metric.to(device)
    model.eval()
    metric.reset()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            metric.update(logits, labels)

    return metric.compute()


def train(
    model,
    optimizer,
    loss_function,
    accuracy_metric,
    train_loader,
    valid_loader,
    n_epochs,
):
    """Train ``model`` and return loss and accuracy values for each epoch."""
    device = next(model.parameters()).device
    accuracy_metric = accuracy_metric.to(device)
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "valid_accuracy": [],
    }

    for epoch in range(n_epochs):
        model.train()
        accuracy_metric.reset()
        total_loss = 0.0
        sample_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            accuracy_metric.update(logits.detach(), labels)

        train_loss = total_loss / sample_count
        train_accuracy = accuracy_metric.compute().item()
        valid_accuracy = evaluate(model, valid_loader, accuracy_metric).item()

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["valid_accuracy"].append(valid_accuracy)

        print(
            f"Epoch {epoch + 1:2d}/{n_epochs}: "
            f"loss={train_loss:.4f}, "
            f"train_accuracy={train_accuracy:.4f}, "
            f"valid_accuracy={valid_accuracy:.4f}"
        )

    return history


print("Defined evaluate() and train()")
