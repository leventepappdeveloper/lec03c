# CSCI E-89 Assignment 03, Problem 5 -- Step 1: setup
# Author: Levente Papp
# Import dependencies, seed random generators, and select an accelerator.

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

# Fix all random number generators used by this pipeline for reproducibility.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Prefer CUDA, then Apple Metal Performance Shaders, then the CPU.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Keep downloaded data and generated figures inside this problem's folder.
BASE_DIR = Path.cwd() / "prob5"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"PyTorch {torch.__version__}, torchvision {torchvision.__version__}, "
      f"torchmetrics {torchmetrics.__version__}")
print(f"Random seed: {SEED}")
print(f"Using device: {device}")
