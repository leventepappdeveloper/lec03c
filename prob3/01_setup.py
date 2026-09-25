# CSCI E-89 Assignment 03, Problem 3 -- Step 1: setup
# Author: Levente Papp
# Imports, random seeds, and device selection (CUDA, MPS, or CPU).

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

# Fix every random number generator so that runs are reproducible
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Pick the fastest available device: NVIDIA GPU, Apple Silicon GPU, else CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Folder that holds this pipeline (prob3/). A notebook has no __file__, so it
# falls back to the working directory, which is the notebook's own folder.
BASE_DIR = (Path(__file__).resolve().parent if "__file__" in globals()
            else Path.cwd())
FIGURES_DIR = BASE_DIR / "figures"  # where the scripts save their plots
FIGURES_DIR.mkdir(exist_ok=True)

print(f"PyTorch {torch.__version__}, torchvision {torchvision.__version__}, "
      f"torchmetrics {torchmetrics.__version__}")
print(f"Random seed: {SEED}")
print(f"Using device: {device}")
