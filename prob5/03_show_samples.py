# CSCI E-89 Assignment 03, Problem 5 -- Step 3: show training samples
# Author: Levente Papp

n_rows, n_cols = 4, 8
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.4, n_rows * 1.7))
for index, axis in enumerate(axes.flat):
    image, label = train_data[index]
    # Remove the one-channel axis so Matplotlib renders a 28 x 28 image.
    axis.imshow(image.squeeze(0), cmap="binary")
    axis.set_title(class_names[label], fontsize=9)
    axis.axis("off")
fig.suptitle("Fashion-MNIST sample training images")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "samples.png", dpi=100)
plt.show()
