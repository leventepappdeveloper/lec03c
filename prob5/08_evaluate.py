# CSCI E-89 Assignment 03, Problem 5 -- Step 8: evaluate and inspect predictions
# Author: Levente Papp

# Measure generalization on all 10,000 previously unseen test examples.
test_accuracy = evaluate(model, test_loader, accuracy_metric).item()
print(f"Final training accuracy:   {history['train_accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history['valid_accuracy'][-1]:.4f}")
print(f"Test-set accuracy:         {test_accuracy:.4f}")

# Predict the first three images from the deterministic validation loader.
model.eval()
X_new, y_new = next(iter(valid_loader))
X_new, y_new = X_new[:3].to(device), y_new[:3]
with torch.no_grad():
    logits = model(X_new)
probabilities = F.softmax(logits, dim=1).cpu()
predicted_labels = probabilities.argmax(dim=1)

print(f"\nPredicted classes: {[class_names[i] for i in predicted_labels]}")
print(f"True classes:      {[class_names[i] for i in y_new]}")
print("\nClass probabilities (rows are images; columns follow class_names):")
print(probabilities.round(decimals=3))

# Report each image's four most probable classes using full-model probabilities.
top_probabilities, top_indices = probabilities.topk(k=4, dim=1)
print("\nTop-4 predictions:")
for index in range(3):
    top_four = ", ".join(
        f"{class_names[class_index]} ({probability:.3f})"
        for class_index, probability in zip(top_indices[index],
                                            top_probabilities[index]))
    print(f"  Image {index + 1} (true: {class_names[y_new[index]]}): {top_four}")
