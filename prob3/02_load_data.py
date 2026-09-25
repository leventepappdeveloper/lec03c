# CSCI E-89 Assignment 03, Problem 3 -- Step 2: load Fashion MNIST
# Author: Levente Papp
# --- standalone bootstrap (removed in the notebook) ---
from _pipeline import run_prerequisites
run_prerequisites(__file__, globals())
# --- end bootstrap ---

# Convert each PIL image to a tensor, then to float32 scaled from [0, 255]
# to [0, 1]
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

# Download (first run only) the 60,000 training and 10,000 test images
DATA_DIR = BASE_DIR / "datasets"
train_and_valid_data = torchvision.datasets.FashionMNIST(
    root=DATA_DIR, train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(
    root=DATA_DIR, train=False, download=True, transform=toTensor)
class_names = train_and_valid_data.classes  # e.g. "T-shirt/top", "Trouser"

# Randomly split the 60,000 training images into 55,000 train + 5,000 valid
torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])

# Batch the data; only the training set is shuffled (reshuffled every epoch)
torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

print(f"Training set:   {len(train_data):,} images")
print(f"Validation set: {len(valid_data):,} images")
print(f"Test set:       {len(test_data):,} images")

# One batch: images are [batch, channels, rows, columns], labels are [batch]
X_batch, y_batch = next(iter(train_loader))
print(f"One batch of images: {tuple(X_batch.shape)}, dtype {X_batch.dtype}, "
      f"values in [{X_batch.min():.1f}, {X_batch.max():.1f}]")
print(f"One batch of labels: {tuple(y_batch.shape)}, dtype {y_batch.dtype}")
