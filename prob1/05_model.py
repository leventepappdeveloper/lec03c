"""
CSCI E-89 Assignment 03, Problem 1 -- Step 5: The model
Author: Levente Papp

Defines the ImageClassifier MLP (Flatten -> 300 -> ReLU -> 100 -> ReLU -> 10)
and creates the model, cross-entropy loss, SGD optimizer (lr=0.1), and a
multiclass accuracy metric.

Notebook cell 5. Run the cells in order (1-8) from a notebook in the
repo root; each cell uses names defined by the earlier cells.
"""

N_INPUTS = 1 * 28 * 28  # channels * rows * columns
N_HIDDEN1, N_HIDDEN2 = 300, 100
N_CLASSES = 10
LEARNING_RATE = 0.1


class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes)
        )

    def forward(self, X):
        return self.mlp(X)  # raw logits; CrossEntropyLoss applies softmax


torch.manual_seed(SEED)
model = ImageClassifier(n_inputs=N_INPUTS, n_hidden1=N_HIDDEN1,
                        n_hidden2=N_HIDDEN2, n_classes=N_CLASSES).to(device)
xentropy = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
accuracy = torchmetrics.Accuracy(
    task="multiclass", num_classes=N_CLASSES).to(device)

print(model)
print()
for name, param in model.named_parameters():
    print(f"{name:15s} {str(tuple(param.shape)):12s} {param.numel():>8,}")
n_params = sum(param.numel() for param in model.parameters())
print(f"Total parameters: {n_params:,}")

# Sanity check: a dummy batch should produce one logit per class
X_dummy = torch.zeros(32, 1, 28, 28, device=device)
print(f"Output shape for a batch of 32: {tuple(model(X_dummy).shape)}")
