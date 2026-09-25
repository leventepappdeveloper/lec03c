# CSCI E-89 Assignment 03, Problem 3 -- Step 7: plot the accuracy curves
# Author: Levente Papp
# --- standalone bootstrap (removed in the notebook) ---
from _pipeline import run_prerequisites
run_prerequisites(__file__, globals())
# --- end bootstrap ---

epochs = np.arange(1, n_epochs + 1)
fig, ax = plt.subplots(figsize=(8, 5))
# Training accuracy is averaged over each epoch while the model is still
# learning, so (as in Geron's notebook) it is plotted half an epoch earlier;
# validation accuracy is measured at the end of each epoch.
ax.plot(epochs - 0.5, history["train_accuracy"], ".--", label="Training")
ax.plot(epochs, history["valid_accuracy"], ".-", label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Fashion MNIST MLP: training and validation accuracy")
ax.set_xticks(epochs)
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "accuracy.png", dpi=100)
plt.show()
