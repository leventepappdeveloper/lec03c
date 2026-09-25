"""
CSCI E-89 Assignment 03, Problem 1 -- Step 6: Train the model
Author: Levente Papp

Trains the ImageClassifier for 20 epochs with the step 4 train() helper and
keeps the per-epoch history. The history and trained weights are saved to
prob1/outputs/ so later steps can use them without retraining.
"""

import importlib
import json
import time
from pathlib import Path

setup = importlib.import_module("01_setup")
data = importlib.import_module("02_load_data")
helpers = importlib.import_module("04_helpers")
m = importlib.import_module("05_model")
torch = setup.torch

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
HISTORY_PATH = OUTPUT_DIR / "history.json"
WEIGHTS_PATH = OUTPUT_DIR / "model.pt"

n_epochs = 20

if __name__ == "__main__":
    print(f"Training on {setup.device} for {n_epochs} epochs "
          f"({len(data.train_data):,} training images)\n")
    torch.manual_seed(setup.SEED)  # fixes the shuffle order of train_loader
    start = time.time()
    history = helpers.train(m.model, m.optimizer, m.xentropy, m.accuracy,
                            data.train_loader, data.valid_loader, n_epochs)
    print(f"\nTraining time: {time.time() - start:.1f} s")

    best_epoch = max(range(n_epochs), key=lambda i: history["valid_metrics"][i])
    print(f"Final valid accuracy: {history['valid_metrics'][-1]:.4f}")
    print(f"Best valid accuracy:  {history['valid_metrics'][best_epoch]:.4f} "
          f"(epoch {best_epoch + 1})")

    OUTPUT_DIR.mkdir(exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))
    torch.save(m.model.state_dict(), WEIGHTS_PATH)
    print(f"Saved history to {HISTORY_PATH}")
    print(f"Saved model weights to {WEIGHTS_PATH}")
