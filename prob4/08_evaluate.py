"""
CSCI E-89 Assignment 03, Problem 4 -- Step 8: Evaluate the classifier
Author: Levente Papp

Report test accuracy and inspect predictions for three validation images.
"""

import runpy
from pathlib import Path

import torch
import torch.nn.functional as F


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Load and train the pipeline when this script is run directly. In the final
# notebook, reuse the trained model and datasets created by the earlier cells.
required_names = {
    "model",
    "accuracy_metric",
    "evaluate",
    "test_loader",
    "valid_loader",
    "train_and_valid_data",
}
if not required_names.issubset(globals()):
    training_state = runpy.run_path(BASE_DIR / "06_train.py")
    data_state = training_state["data_state"]
    helper_state = training_state["helper_state"]

    model = training_state["model"]
    accuracy_metric = training_state["accuracy_metric"]
    evaluate = helper_state["evaluate"]
    test_loader = data_state["test_loader"]
    valid_loader = data_state["valid_loader"]
    train_and_valid_data = data_state["train_and_valid_data"]

class_names = train_and_valid_data.classes
test_accuracy = evaluate(model, test_loader, accuracy_metric).item()
print(f"Test-set accuracy: {test_accuracy:.4f}")

# Make predictions for the first three images in the unshuffled validation
# loader. Softmax converts the model's ten output logits to probabilities.
model.eval()
images, true_labels = next(iter(valid_loader))
images = images[:3].to(next(model.parameters()).device)
true_labels = true_labels[:3]

with torch.no_grad():
    logits = model(images)
    probabilities = F.softmax(logits, dim=1).cpu()

predicted_labels = probabilities.argmax(dim=1)
top_probabilities, top_indices = probabilities.topk(k=4, dim=1)

print("\nPredictions for 3 validation images:")
for index in range(3):
    predicted_class = class_names[predicted_labels[index].item()]
    true_class = class_names[true_labels[index].item()]
    print(f"\nImage {index + 1}")
    print(f"  Predicted class: {predicted_class}")
    print(f"  True class:      {true_class}")

    class_probabilities = ", ".join(
        f"{class_name}={probability.item():.4f}"
        for class_name, probability in zip(class_names, probabilities[index])
    )
    print(f"  Class probabilities: {class_probabilities}")

    top_four = ", ".join(
        f"{class_names[class_index.item()]} ({probability.item():.4f})"
        for probability, class_index in zip(
            top_probabilities[index], top_indices[index]
        )
    )
    print(f"  Top-4 predictions: {top_four}")
