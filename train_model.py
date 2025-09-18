


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import os

# ======================
# 1. Paths & Transforms
# ======================
data_dir = r"C:\Users\agrani\Downloads\Indian Bovine Breed Recognition.v1i.folder"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======================
# 2. Load Dataset (Full)
# ======================
train_data = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
valid_data = datasets.ImageFolder(f"{data_dir}/valid", transform=test_transform)
test_data  = datasets.ImageFolder(f"{data_dir}/test", transform=test_transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=16, shuffle=False)

classes = train_data.classes
print("Detected Breeds:", classes)

# ======================
# 3. Load Model
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet50(weights=None)  # no pre-trained weights
for param in model.parameters():
    param.requires_grad = False  # freeze backbone

num_classes = len(classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ======================
# 4. Loss & Optimizer
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ======================
# 5. Evaluation Function
# ======================
def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ======================
# 6. Training Function
# ======================
def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        # Training loop with progress bar
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # Validation loop with progress bar
        val_acc = 0
        val_correct, val_total = 0, 0
        model.eval()
        for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.3f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    return model

# ======================
# 7. Train Model
# ======================
trained_model = train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=5)

# ======================
# 8. Evaluate on Test
# ======================
test_acc = evaluate_model(trained_model, test_loader)
print(f"Final Test Accuracy: {test_acc:.2f}%")

# ======================
# 9. Save Model & Classes
# ======================
torch.save(trained_model.state_dict(), "indian_cattle_breed_model.pth")
with open("breed_classes.json", "w") as f:
    json.dump(classes, f)

print("Model and class names saved!")
