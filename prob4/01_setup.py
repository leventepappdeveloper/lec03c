"""
CSCI E-89 Assignment 03, Problem 4 -- Step 1: Setup
Author: Levente Papp

Import the libraries used by the Fashion-MNIST image classifier, seed the
random number generators for reproducibility, and select the best available
PyTorch device.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
import torchvision


SEED = 42

# Seed NumPy and PyTorch so later data preparation and model training steps are
# reproducible. Seed every CUDA device too when CUDA is available.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Prefer an NVIDIA GPU, then Apple Silicon acceleration, and otherwise use CPU.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(
    f"PyTorch {torch.__version__}, torchvision {torchvision.__version__}, "
    f"torchmetrics {torchmetrics.__version__}"
)
print(f"NumPy {np.__version__}, Matplotlib {plt.matplotlib.__version__}")
print(f"Random seed: {SEED}")
print(f"Using device: {device}")
