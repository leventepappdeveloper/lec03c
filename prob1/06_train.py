"""
CSCI E-89 Assignment 03, Problem 1 -- Step 6: Train the model
Author: Levente Papp

Trains the ImageClassifier for 20 epochs with the step 4 train() helper and
keeps the per-epoch history. The history and trained weights are saved to
prob1/outputs/.

Notebook cell 6. Run the cells in order (1-8) from a notebook in the
repo root; each cell uses names defined by the earlier cells.
"""

OUTPUT_DIR = Path("prob1/outputs")
HISTORY_PATH = OUTPUT_DIR / "history.json"
WEIGHTS_PATH = OUTPUT_DIR / "model.pt"

n_epochs = 20

print(f"Training on {device} for {n_epochs} epochs "
      f"({len(train_data):,} training images)\n")
torch.manual_seed(SEED)  # fixes the shuffle order of train_loader
start = time.time()
history = train(model, optimizer, xentropy, accuracy,
                train_loader, valid_loader, n_epochs)
print(f"\nTraining time: {time.time() - start:.1f} s")

best_epoch = max(range(n_epochs), key=lambda i: history["valid_metrics"][i])
print(f"Final valid accuracy: {history['valid_metrics'][-1]:.4f}")
print(f"Best valid accuracy:  {history['valid_metrics'][best_epoch]:.4f} "
      f"(epoch {best_epoch + 1})")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH.write_text(json.dumps(history, indent=2))
torch.save(model.state_dict(), WEIGHTS_PATH)
print(f"Saved history to {HISTORY_PATH}")
print(f"Saved model weights to {WEIGHTS_PATH}")
