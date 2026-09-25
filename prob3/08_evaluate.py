# CSCI E-89 Assignment 03, Problem 3 -- Step 8: test accuracy and predictions
# Author: Levente Papp
# --- standalone bootstrap (removed in the notebook) ---
from _pipeline import run_prerequisites
run_prerequisites(__file__, globals())
# --- end bootstrap ---

# Accuracy on the 10,000 test images, which the model has never seen
test_accuracy = evaluate_tm(model, test_loader, accuracy).item()
print(f"Final training accuracy:   {history['train_accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history['valid_accuracy'][-1]:.4f}")
print(f"Test accuracy:             {test_accuracy:.4f}")

# Predict the first 3 validation images
model.eval()
X_new, y_new = next(iter(valid_loader))
X_new, y_new = X_new[:3].to(device), y_new[:3]
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1).cpu()  # index of the largest logit

print(f"\nPredicted class indices: {y_pred.tolist()}")
print(f"Predicted classes: {[class_names[i] for i in y_pred]}")
print(f"True class indices:      {y_new.tolist()}")
print(f"True classes:      {[class_names[i] for i in y_new]}")

# Softmax turns the logits into class probabilities (each row sums to 1)
y_proba = F.softmax(y_pred_logits, dim=1).cpu()
print("\nClass probabilities (one row per image, one column per class):")
print(y_proba.round(decimals=3))

# The 4 most likely classes per image, with probabilities renormalized over
# those 4 classes (as in Geron's notebook)
y_top4_values, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
y_top4_probas = F.softmax(y_top4_values, dim=1).cpu()
y_top4_indices = y_top4_indices.cpu()
print("\nTop-4 predictions:")
for i in range(len(X_new)):
    top4 = ", ".join(f"{class_names[c]} ({p:.3f})"
                     for c, p in zip(y_top4_indices[i], y_top4_probas[i]))
    print(f"  Image {i} (true: {class_names[y_new[i]]}): {top4}")
