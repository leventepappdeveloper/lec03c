"""
CSCI E-89 Assignment 03, Problem 1 -- Step 8: Evaluate the trained model
Author: Levente Papp

Loads the weights saved by step 6, reports test-set accuracy, then for the
first 3 validation images shows the predicted and true classes, the full
class probabilities, and the top-4 predictions.

Notebook cell 8. Run the cells in order (1-8) from a notebook in the
repo root; each cell uses names defined by the earlier cells.
"""

WEIGHTS_PATH = Path("prob1/outputs/model.pt")
N_SAMPLES = 3
TOP_K = 4

if not WEIGHTS_PATH.exists():
    raise SystemExit(f"{WEIGHTS_PATH} not found -- run the step 6 cell (06_train.py) first.")
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
classes = train_and_valid_data.classes

# --- Test-set accuracy ---
test_acc = evaluate_tm(model, test_loader, accuracy).item()
valid_acc = evaluate_tm(model, valid_loader, accuracy).item()
print(f"Validation accuracy: {valid_acc:.4f}")
print(f"Test accuracy:       {test_acc:.4f}  ({len(test_data):,} images)")

# --- Predictions for 3 validation images ---
model.eval()
X_new, y_new = next(iter(valid_loader))  # valid_loader is not shuffled
X_new, y_new = X_new[:N_SAMPLES].to(device), y_new[:N_SAMPLES]
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1).cpu()  # index of the largest logit

print(f"\nPredicted class indices: {y_pred.tolist()}")
print(f"True class indices:      {y_new.tolist()}")
print(f"Predicted classes: {[classes[i] for i in y_pred]}")
print(f"True classes:      {[classes[i] for i in y_new]}")
print(f"Correct: {(y_pred == y_new).sum().item()}/{N_SAMPLES}")

# Class probabilities (softmax over all 10 logits); move to CPU for printing
y_proba = F.softmax(y_pred_logits, dim=1).cpu()
print("\nClass probabilities:")
header = "".join(f"{name[:11]:>12s}" for name in classes)
print(f"{'':8s}{header}")
for i, row in enumerate(y_proba):
    print(f"Image {i + 1:<2d}" + "".join(f"{p:>12.3f}" for p in row))

# Top-4: softmax over only the 4 largest logits, as in the notebook
y_top_values, y_top_indices = torch.topk(y_pred_logits, k=TOP_K, dim=1)
y_top_probas = F.softmax(y_top_values, dim=1).cpu()
y_top_indices = y_top_indices.cpu()
print(f"\nTop-{TOP_K} probabilities (renormalized over the top {TOP_K}):")
print(y_top_probas.round(decimals=3))
print(f"Top-{TOP_K} class indices:")
print(y_top_indices)

print(f"\nTop-{TOP_K} predictions per image:")
for i in range(N_SAMPLES):
    ranked = ", ".join(f"{classes[j]} ({p:.3f})"
                       for j, p in zip(y_top_indices[i], y_top_probas[i]))
    print(f"Image {i + 1} (true: {classes[y_new[i]]}): {ranked}")
