# CSCI E-89 Assignment 03, Problem 3 -- Step 4: evaluation and training helpers
# Author: Levente Papp
# --- standalone bootstrap (removed in the notebook) ---
from _pipeline import run_prerequisites
run_prerequisites(__file__, globals())
# --- end bootstrap ---


def evaluate_tm(model, data_loader, metric):
    """Compute a torchmetrics metric for `model` over a whole DataLoader."""
    model.eval()     # evaluation mode (matters for dropout, batch norm, ...)
    metric.reset()   # forget results from any previous call
    with torch.no_grad():  # no gradients needed: faster, uses less memory
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)  # accumulate batch statistics
    return metric.compute()  # combine the batches into one value


def train(model, optimizer, criterion, metric, train_loader, valid_loader,
          n_epochs):
    """Train for n_epochs; return a history dict of per-epoch results."""
    history = {"train_loss": [], "train_accuracy": [], "valid_accuracy": []}
    for epoch in range(n_epochs):
        model.train()  # training mode (evaluate_tm switched to eval mode)
        total_loss = 0.
        metric.reset()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)            # forward pass: logits
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()                    # backprop: compute gradients
            optimizer.step()                   # gradient descent step
            optimizer.zero_grad()              # reset gradients for next batch
            metric.update(y_pred, y_batch)     # running training accuracy
        history["train_loss"].append(total_loss / len(train_loader))
        history["train_accuracy"].append(metric.compute().item())
        history["valid_accuracy"].append(
            evaluate_tm(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1:2d}/{n_epochs}, "
              f"train loss: {history['train_loss'][-1]:.4f}, "
              f"train accuracy: {history['train_accuracy'][-1]:.4f}, "
              f"valid accuracy: {history['valid_accuracy'][-1]:.4f}")
    return history


print("Defined evaluate_tm() and train()")
