"""
CSCI E-89 Assignment 03, Problem 4 -- Step 5: Image classifier model
Author: Levente Papp

Define and initialize the Fashion-MNIST multilayer perceptron, its loss,
optimizer, and accuracy metric.
"""

import torch
import torch.nn as nn
import torchmetrics


SEED = globals().get("SEED", 42)
if "device" not in globals():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"


class ImageClassifier(nn.Module):
    """A fully connected classifier that returns logits for ten classes."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, images):
        """Return unnormalized class scores for a batch of images."""
        return self.network(images)


torch.manual_seed(SEED)
model = ImageClassifier().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(
    device
)

parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(model)
print(f"Number of parameters: {parameter_count:,}")
