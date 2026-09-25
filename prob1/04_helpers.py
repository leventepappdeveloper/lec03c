"""
CSCI E-89 Assignment 03, Problem 1 -- Step 4: Training helpers
Author: Levente Papp

evaluate_tm(): computes a torchmetrics metric over a whole DataLoader.
train():       trains a model for n epochs, recording the training loss,
               training metric, and validation metric for each epoch.

The end of the cell does a quick smoke test on a small data subset.

Notebook cell 4. Run the cells in order (1-8) from a notebook in the
repo root; each cell uses names defined by the earlier cells.
"""


def evaluate_tm(model, data_loader, metric):
    """Return metric computed over every batch in data_loader."""
    model.eval()
    metric.reset()  # clear state left over from any previous use
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)  # accumulate batch statistics
    return metric.compute()  # aggregate over the whole dataset


def train(model, optimizer, criterion, metric, train_loader, valid_loader,
          n_epochs):
    """Train for n_epochs and return a history dict of per-epoch values."""
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        model.train()  # evaluate_tm() switches to eval mode, so reset each epoch
        total_loss = 0.
        metric.reset()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        history["train_losses"].append(total_loss / len(train_loader))
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train accuracy: {history['train_metrics'][-1]:.4f}, "
              f"valid accuracy: {history['valid_metrics'][-1]:.4f}")
    return history


# Smoke test: a throwaway linear model on 2,000 training / 500 validation
# images, just to check that the helpers run end to end. The smoke_ prefix
# keeps these names separate from the real model built in step 5.
smoke_train = torch.utils.data.Subset(train_data, range(2_000))
smoke_valid = torch.utils.data.Subset(valid_data, range(500))
smoke_train_loader = DataLoader(smoke_train, batch_size=32, shuffle=True)
smoke_valid_loader = DataLoader(smoke_valid, batch_size=32)

torch.manual_seed(SEED)
smoke_model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10)).to(device)
smoke_optimizer = torch.optim.SGD(smoke_model.parameters(), lr=0.1)
smoke_accuracy = torchmetrics.Accuracy(
    task="multiclass", num_classes=10).to(device)

smoke_history = train(smoke_model, smoke_optimizer, nn.CrossEntropyLoss(),
                      smoke_accuracy, smoke_train_loader, smoke_valid_loader,
                      n_epochs=3)
print("History keys:", list(smoke_history))
print("Valid accuracy via evaluate_tm:",
      f"{evaluate_tm(smoke_model, smoke_valid_loader, smoke_accuracy).item():.4f}")
