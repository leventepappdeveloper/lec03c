# CSCI E-89 Assignment 03, Problem 5 -- Step 7: plot accuracy
# Author: Levente Papp

epochs = np.arange(1, n_epochs + 1)
fig, axis = plt.subplots(figsize=(8, 5))
axis.plot(epochs, history["train_accuracy"], ".--", label="Training accuracy")
axis.plot(epochs, history["valid_accuracy"], ".-", label="Validation accuracy")
axis.set_xlabel("Epoch")
axis.set_ylabel("Accuracy")
axis.set_title("Fashion-MNIST training and validation accuracy")
axis.set_xticks(epochs)
axis.grid(alpha=0.3)
axis.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "accuracy.png", dpi=100)
plt.show()
