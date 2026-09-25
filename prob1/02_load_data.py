"""
CSCI E-89 Assignment 03, Problem 1 -- Step 2: Load the data
Author: Levente Papp

Downloads Fashion MNIST, converts images to float tensors in [0, 1],
splits the 60,000 training images into 55,000 train / 5,000 validation,
and wraps everything in DataLoaders with batch size 32.

Notebook cell 2. Run the cells in order (1-8) from a notebook in the
repo root; each cell uses names defined by the earlier cells.
"""

DATA_DIR = Path("datasets")
BATCH_SIZE = 32

# ToImage() -> tensor of shape [C, H, W]; ToDtype(..., scale=True) -> float32 in [0, 1]
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

train_and_valid_data = torchvision.datasets.FashionMNIST(
    root=DATA_DIR, train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(
    root=DATA_DIR, train=False, download=True, transform=toTensor)

torch.manual_seed(SEED)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])

torch.manual_seed(SEED)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

print(f"Training set:   {len(train_data):,} images")
print(f"Validation set: {len(valid_data):,} images")
print(f"Test set:       {len(test_data):,} images")
print(f"Classes:        {train_and_valid_data.classes}")

X_batch, y_batch = next(iter(train_loader))
print(f"Batch images shape: {tuple(X_batch.shape)}  dtype: {X_batch.dtype}")
print(f"Batch labels shape: {tuple(y_batch.shape)}  dtype: {y_batch.dtype}")
print(f"Pixel value range:  [{X_batch.min():.1f}, {X_batch.max():.1f}]")
