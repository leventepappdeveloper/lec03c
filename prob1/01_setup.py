"""
CSCI E-89 Assignment 03, Problem 1 -- Step 1: Setup
Author: Levente Papp

Building an Image Classifier with PyTorch (Fashion MNIST).
This step handles imports, reproducibility (random seed), and device selection.

Notebook cell 1. Run the cells in order (1-8) from a notebook in the
repo root; each cell uses names defined by the earlier cells.
"""

import json
import random
import time
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

SEED = 42


def set_seed(seed=SEED):
    """Seed Python, NumPy, and PyTorch RNGs for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


set_seed()
device = get_device()

print(f"PyTorch version:      {torch.__version__}")
print(f"TorchVision version:  {torchvision.__version__}")
print(f"TorchMetrics version: {torchmetrics.__version__}")
print(f"NumPy version:        {np.__version__}")
print(f"Matplotlib version:   {plt.matplotlib.__version__}")
print(f"Random seed:          {SEED}")
print(f"Device:               {device}")
