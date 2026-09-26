# CSCI E-89 Assignment 03, Problem 5 -- Step 4: helper functions
# Author: Levente Papp


def evaluate(model, data_loader, metric):
    """Compute a TorchMetrics metric over every batch in a DataLoader."""
    model.eval()
    metric.reset()
    with torch.no_grad():  # Evaluation needs neither gradients nor a graph.
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            metric.update(model(X_batch), y_batch)
    return metric.compute()


def train(model, optimizer, criterion, metric, train_loader, valid_loader,
          n_epochs):
    """Train for n_epochs and return loss/train/validation histories."""
    history = {"train_loss": [], "train_accuracy": [], "valid_accuracy": []}
    for epoch in range(n_epochs):
        model.train()
        metric.reset()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)               # Forward pass.
            loss = criterion(logits, y_batch)
            loss.backward()                       # Compute gradients.
            optimizer.step()                      # Update model parameters.
            total_loss += loss.item()
            metric.update(logits.detach(), y_batch)

        history["train_loss"].append(total_loss / len(train_loader))
        history["train_accuracy"].append(metric.compute().item())
        history["valid_accuracy"].append(
            evaluate(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1:2d}/{n_epochs}, "
              f"train loss: {history['train_loss'][-1]:.4f}, "
              f"train accuracy: {history['train_accuracy'][-1]:.4f}, "
              f"valid accuracy: {history['valid_accuracy'][-1]:.4f}")
    return history


print("Defined evaluate() and train()")
