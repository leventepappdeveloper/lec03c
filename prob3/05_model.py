# CSCI E-89 Assignment 03, Problem 3 -- Step 5: model, loss, optimizer, metric
# Author: Levente Papp
# --- standalone bootstrap (removed in the notebook) ---
from _pipeline import run_prerequisites
run_prerequisites(__file__, globals())
# --- end bootstrap ---


class ImageClassifier(nn.Module):
    """MLP: 28x28 image -> 300 -> 100 -> 10 class logits."""
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),                    # [batch, 1, 28, 28] -> [batch, 784]
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes)  # logits, no softmax (the loss does it)
        )

    def forward(self, X):
        return self.mlp(X)


torch.manual_seed(42)  # reproducible initial weights
model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                        n_classes=10).to(device)
xentropy = nn.CrossEntropyLoss()  # applies log-softmax to the logits itself
# Create the optimizer after moving the model to the device
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

print(model)
n_params = sum(param.numel() for param in model.parameters())
print(f"Number of parameters: {n_params:,}")
