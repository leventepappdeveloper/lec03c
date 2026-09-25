"""
CSCI E-89 Assignment 03, Problem 1 -- Step 1: Setup
Author: Levente Papp

Building an Image Classifier with PyTorch (Fashion MNIST).
This step handles imports, reproducibility (random seed), and device selection.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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

if __name__ == "__main__":
    print(f"PyTorch version:      {torch.__version__}")
    print(f"TorchVision version:  {torchvision.__version__}")
    print(f"TorchMetrics version: {torchmetrics.__version__}")
    print(f"NumPy version:        {np.__version__}")
    print(f"Matplotlib version:   {plt.matplotlib.__version__}")
    print(f"Random seed:          {SEED}")
    print(f"Device:               {device}")
