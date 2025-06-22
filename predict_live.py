import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os

# 🧠 Your trained model architecture
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

# ✅ Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastASLCNN(num_classes=29)  # 29 = A-Z + space + nothing + delete (adjust based on your dataset)
model_path = os.path.join(os.path.expanduser("~"), "Desktop", "asl_model_prototype.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)
print("✅ Model loaded and ready!")

# ✂️ Classes
classes = sorted(os.listdir(r"C:\Users\thund\Downloads\archive\asl_alphabet_train\asl_alphabet_train"))

# 🧪 Image transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 🎥 OpenCV webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not detected!")
    exit()

print("🎬 Starting webcam ASL detection... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 👁️ Draw ROI (Region of Interest)
    x, y, w, h = 100, 100, 200, 200

    roi = frame[y:y+w, x:x+w]
    img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = output.argmax(1).item()
        pred_class = classes[pred_idx]

    # 🖼️ Display the frame with prediction
    cv2.rectangle(frame, (x, y), (x+w, y+w), (0, 255, 0), 2)
    cv2.putText(frame, f"Predicted: {pred_class}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.imshow("ASL Live Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exiting ASL Live Detection.")
