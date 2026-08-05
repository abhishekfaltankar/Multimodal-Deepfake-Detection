import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from training.dataset import DeepfakeDataset
from models.efficientnet import DeepfakeDetector

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using Device: {device}")

# Dataset
dataset = DeepfakeDataset("dataset/processed/faces")

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


# Model
model = DeepfakeDetector().to(device)

# Loss Function
criterion = nn.CrossEntropyLoss()


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

print("\nModel Loaded Successfully!")
print(f"Loss Function : {criterion.__class__.__name__}")
print(f"Optimizer     : {optimizer.__class__.__name__}")

print(model)

images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)

outputs = model(images)

print(f"\nBatch Shape : {images.shape}")
print(f"Output Shape: {outputs.shape}")


# Training (1 Epoch)
model.train()

running_loss = 0.0
correct = 0
total = 0

for images, labels in train_loader:

    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    running_loss += loss.item()

    _, predicted = torch.max(outputs, 1)

    total += labels.size(0)

    correct += (predicted == labels).sum().item()

epoch_loss = running_loss / len(train_loader)
epoch_accuracy = 100 * correct / total

print("\nTraining Completed!")
print(f"Loss     : {epoch_loss:.4f}")
print(f"Accuracy : {epoch_accuracy:.2f}%")