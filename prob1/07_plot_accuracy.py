"""
CSCI E-89 Assignment 03, Problem 1 -- Step 7: Plot accuracy per epoch
Author: Levente Papp

Plots training and validation accuracy per epoch from the history saved by
step 6. The figure is saved to prob1/figures/accuracy.png and also shown.

Notebook cell 7. Run the cells in order (1-8) from a notebook in the
repo root; each cell uses names defined by the earlier cells.
"""

HISTORY_PATH = Path("prob1/outputs/history.json")
FIG_PATH = Path("prob1/figures/accuracy.png")

TRAIN_COLOR = "#2a78d6"  # blue
VALID_COLOR = "#eb6834"  # orange

if not HISTORY_PATH.exists():
    raise SystemExit(f"{HISTORY_PATH} not found -- run the step 6 cell (06_train.py) first.")
history = json.loads(HISTORY_PATH.read_text())

train_acc = np.array(history["train_metrics"])
valid_acc = np.array(history["valid_metrics"])
epochs = np.arange(1, len(train_acc) + 1)
best = valid_acc.argmax()

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(epochs, train_acc, "o--", color=TRAIN_COLOR, linewidth=2,
        markersize=5, label="Training accuracy")
ax.plot(epochs, valid_acc, "s-", color=VALID_COLOR, linewidth=2,
        markersize=5, label="Validation accuracy")

# Mark the best validation epoch
ax.annotate(f"Best validation: {valid_acc[best]:.2%} (epoch {epochs[best]})",
            xy=(epochs[best], valid_acc[best]),
            xytext=(epochs[best] + 0.3, valid_acc[best] - 0.035),
            ha="right", color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.4", relpos=(1, 1)))

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("ImageClassifier on Fashion MNIST: training vs. validation accuracy")
ax.set_xticks(epochs)
ax.set_xlim(0.5, epochs[-1] + 0.5)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.grid(color="0.9")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PATH, dpi=120)
print(f"Final training accuracy:   {train_acc[-1]:.4f}")
print(f"Final validation accuracy: {valid_acc[-1]:.4f}")
print(f"Gap at final epoch:        {train_acc[-1] - valid_acc[-1]:.4f}")
print(f"Saved plot to {FIG_PATH}")

# plt.show() only has an effect with an interactive backend (not on a headless machine)
if plt.get_backend().lower() != "agg":
    plt.show()
