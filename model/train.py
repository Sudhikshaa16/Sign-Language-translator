# model/train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .gesture_dataset import GestureDataset
from .mlp import MLP

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
dataset = GestureDataset()  # auto-detect ../dataset/real
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# -------------------------------
# 2️⃣ Model, loss, optimizer
# -------------------------------
input_size = dataset.num_features
num_classes = len(dataset.class_map)

model = MLP(input_size=input_size, num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 3️⃣ Training loop
# -------------------------------
EPOCHS = 50
for epoch in range(EPOCHS):
    running_loss = 0.0
    for x, y in dataloader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {running_loss/len(dataloader):.4f}")

# -------------------------------
# 4️⃣ Save model
# -------------------------------
model_path = os.path.join(os.path.dirname(__file__), "gesture_mlp.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Training complete! Model saved at {model_path}")
