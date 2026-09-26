# CSCI E-89 Assignment 03, Problem 5 -- Step 2: load Fashion-MNIST
# Author: Levente Papp

# ToImage creates a [channel, height, width] tensor; ToDtype converts the
# original bytes to float32 and scales their range from [0, 255] to [0, 1].
to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

DATA_DIR = BASE_DIR / "datasets"
train_and_valid_data = torchvision.datasets.FashionMNIST(
    root=DATA_DIR, train=True, download=True, transform=to_tensor)
test_data = torchvision.datasets.FashionMNIST(
    root=DATA_DIR, train=False, download=True, transform=to_tensor)
class_names = train_and_valid_data.classes

# Split the original 60,000 training examples reproducibly into 55k and 5k.
split_generator = torch.Generator().manual_seed(SEED)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000], generator=split_generator)

# Shuffle only training batches; validation and test order stays deterministic.
loader_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                          generator=loader_generator)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

print(f"Training set:   {len(train_data):,} images")
print(f"Validation set: {len(valid_data):,} images")
print(f"Test set:       {len(test_data):,} images")
X_batch, y_batch = next(iter(train_loader))
print(f"One batch of images: {tuple(X_batch.shape)}, dtype {X_batch.dtype}, "
      f"values in [{X_batch.min():.1f}, {X_batch.max():.1f}]")
print(f"One batch of labels: {tuple(y_batch.shape)}, dtype {y_batch.dtype}")
