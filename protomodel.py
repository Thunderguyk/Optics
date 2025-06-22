import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time
import os

class FastASLCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout(x)
        return self.fc2(x)

def main():
    # ==== SETUP ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Using device:", device)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_path = r"C:\Users\thund\Downloads\archive\asl_alphabet_train\asl_alphabet_train"
    print("📂 Loading dataset...")
    train_data = datasets.ImageFolder(data_path, transform=transform)
    print(f"✅ Loaded {len(train_data)} images with {len(train_data.classes)} classes")

    train_size = int(0.9 * len(train_data))
    val_size = len(train_data) - train_size
    train_ds, val_ds = torch.utils.data.random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=4, pin_memory=True)

    model = FastASLCNN(num_classes=len(train_data.classes)).to(device)

    # Dry test
    print("🧪 Dry test...")
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
    print("✅ Dry pass OK")

    # ==== TRAIN ====
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)

    print("🚀 Training begins")
    EPOCHS = 3
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        start = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"[{epoch+1}/{EPOCHS}] Batch {i+1}/{len(train_loader)} Loss: {loss.item():.4f}")
        print(f"✅ Epoch {epoch+1} complete — Avg Loss: {epoch_loss / len(train_loader):.4f}, Time: {round(time.time() - start, 2)}s")

    # ==== SAVE ====
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    model_path = os.path.join(desktop, "asl_model_prototype.pth")
    try:
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved at {model_path}")
    except Exception as e:
        print("❌ Save failed:", e)

if __name__ == '__main__':
    main()
