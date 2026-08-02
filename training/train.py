import torch

from torch.utils.data import DataLoader, random_split

from training.dataset import DeepfakeDataset

dataset = DeepfakeDataset("dataset/processed/faces")

print(f"Total Images : {len(dataset)}")

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print(f"Training Images : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")
print(f"Testing Images : {len(test_dataset)}")