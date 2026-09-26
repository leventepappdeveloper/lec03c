# CSCI E-89 Assignment 03, Problem 5 -- Step 5: classifier model
# Author: Levente Papp


class ImageClassifier(nn.Module):
    """A 784 -> 300 -> 100 -> 10 multilayer perceptron."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),                 # [batch, 1, 28, 28] -> [batch, 784]
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),           # Raw logits; the loss applies softmax.
        )

    def forward(self, images):
        return self.network(images)


torch.manual_seed(SEED)  # Make initial weights reproducible.
model = ImageClassifier().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
accuracy_metric = torchmetrics.Accuracy(
    task="multiclass", num_classes=10).to(device)

parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(model)
print(f"Number of parameters: {parameter_count:,}")
