# CSCI E-89 Assignment 03, Problem 3 -- Step 3: show sample training images
# Author: Levente Papp
# --- standalone bootstrap (removed in the notebook) ---
from _pipeline import run_prerequisites
run_prerequisites(__file__, globals())
# --- end bootstrap ---

n_rows, n_cols = 4, 8
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.4, n_rows * 1.7))
for index, ax in enumerate(axes.flat):
    image, label = train_data[index]      # image: [1, 28, 28], label: int
    ax.imshow(image.squeeze(0), cmap="binary")  # drop the channel dimension
    ax.set_title(class_names[label], fontsize=9)
    ax.axis("off")
fig.suptitle("Fashion MNIST: sample training images")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "samples.png", dpi=100)
plt.show()
