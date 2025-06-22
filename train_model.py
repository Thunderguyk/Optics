import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import time

# ───────────────────────────────────────────
# ⚙️ System & GPU Check
# ───────────────────────────────────────────
DISABLE_CUDNN = False
if DISABLE_CUDNN:
    torch.backends.cudnn.enabled = False
    print("⚠️ cuDNN disabled for debugging.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)
print("🧠 CUDA available:", torch.cuda.is_available())

# ───────────────────────────────────────────
# 🧹 Preprocessing & Augmentation
# ───────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ───────────────────────────────────────────
# 📂 Dataset Loading
# ───────────────────────────────────────────
data_path = r"C:\Users\thund\Downloads\archive\asl_alphabet_train\asl_alphabet_train"

print("📂 Loading dataset...")
start_time = time.time()
train_data = datasets.ImageFolder(data_path, transform=transform)
print(f"✅ Dataset loaded. Classes: {len(train_data.classes)} — {train_data.classes}")
print(f"⏱ Took {time.time() - start_time:.2f} seconds")

train_size = int(0.9 * len(train_data))
val_size = len(train_data) - train_size
train_ds, val_ds = torch.utils.data.random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# ───────────────────────────────────────────
# 🧠 Model Definition
# ───────────────────────────────────────────
class ASLCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_block = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc_block(x)

model = ASLCNN(num_classes=len(train_data.classes)).to(device)
print("✅ Model loaded to device.")

# ───────────────────────────────────────────
# 🧪 Dry Run
# ───────────────────────────────────────────
print("🧪 Running a dry batch test...")
sample_batch = next(iter(train_loader))
images, labels = sample_batch
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
    outputs = model(images)
print("✅ Forward pass successful on GPU!")

# ───────────────────────────────────────────
# 🎯 Loss, Optimizer & Scheduler
# ───────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ───────────────────────────────────────────
# 🏋️ Training Loop
# ───────────────────────────────────────────
print("🚀 Starting training...")

def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = 100 * correct / total
    print(f"🎯 Validation Accuracy: {acc:.2f}%")
    return acc

num_epochs = 10
for epoch in range(num_epochs):
    start = time.time()
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:
            print(f"🔁 Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"\n✅ Epoch {epoch+1} completed — Avg Loss: {avg_loss:.4f}, Time: {time.time() - start:.2f}s")
    evaluate_model(model, val_loader)
    scheduler.step()

# ───────────────────────────────────────────
# 💾 Save Model to Desktop
# ───────────────────────────────────────────
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "asl_model_pytorch.pth")
torch.save(model.state_dict(), desktop_path)
print(f"💾 Model saved to: {desktop_path}")
