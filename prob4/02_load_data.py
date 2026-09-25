"""
CSCI E-89 Assignment 03, Problem 4 -- Step 2: Load Fashion-MNIST
Author: Levente Papp

Download Fashion-MNIST, convert its images to float tensors in [0, 1],
create reproducible training and validation splits, and batch the datasets.
"""

from pathlib import Path

import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, random_split


SEED = 42
BATCH_SIZE = 32
DATA_DIR = Path(__file__).resolve().parent / "datasets"

# ToImage creates a [channels, height, width] tensor. Scaling while converting
# to float32 maps the original byte values from [0, 255] into [0, 1].
transform = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]
)

train_and_valid_data = torchvision.datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform,
)
test_data = torchvision.datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform,
)

# Use a dedicated generator so the 55,000/5,000 split is reproducible without
# changing the random state used by other parts of the classifier pipeline.
split_generator = torch.Generator().manual_seed(SEED)
train_data, valid_data = random_split(
    train_and_valid_data,
    [55_000, 5_000],
    generator=split_generator,
)

loader_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=loader_generator,
)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

print(f"Training set:   {len(train_data):,} images")
print(f"Validation set: {len(valid_data):,} images")
print(f"Test set:       {len(test_data):,} images")

images, labels = next(iter(train_loader))
print(f"Image batch shape: {tuple(images.shape)}")
print(f"Label batch shape: {tuple(labels.shape)}")
