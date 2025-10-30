import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_DIR = "aligned_dataset"     # or aligned_dataset_noface
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------
class AttentionDataset(Dataset):
    def __init__(self, folder):
        self.files = glob.glob(os.path.join(folder, "*.npz"))
        self.samples = []
        for f in self.files:
            d = np.load(f)
            X, y = d["X"], d["y"]
            # Normalize features
            X = (X - X.mean(0)) / (X.std(0) + 1e-6)
            self.samples.append((X.astype(np.float32), y.astype(np.float32)))
        print(f"Loaded {len(self.samples)} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X), torch.tensor(y)

dataset = AttentionDataset(DATA_DIR)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------
class CNN1DAttention(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
        )

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)      # (B, F, T)
        y_pred = self.conv(x)        # (B, 1, T)
        return y_pred.squeeze(1)     # (B, T)

model = CNN1DAttention(in_ch=2).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------------------------------------------------------------
# TRAIN LOOP
# ---------------------------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(X)
        # Match lengths
        min_len = min(y_pred.shape[1], y.shape[1])
        loss = criterion(y_pred[:, :min_len], y[:, :min_len])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X)
            min_len = min(y_pred.shape[1], y.shape[1])
            val_loss += criterion(y_pred[:, :min_len], y[:, :min_len]).item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), "cnn_attention_model.pt")
print("✅ Model saved as cnn_attention_model.pt")

# ---------------------------------------------------------------------
# VISUALIZATION ON ONE VIDEO
# ---------------------------------------------------------------------
X, y = dataset[0]
X, y = X.unsqueeze(0).to(DEVICE), y.numpy()
model.eval()
with torch.no_grad():
    y_pred = model(X).cpu().numpy().squeeze()

plt.figure(figsize=(10,4))
plt.plot(y, label="Human Attention", color='orange')
plt.plot(y_pred, label="Predicted Attention", color='blue')
plt.title("Micro-Attention Prediction")
plt.xlabel("Seconds")
plt.ylabel("Attention Intensity")
plt.legend()
plt.tight_layout()
plt.savefig("attention_prediction_curve.png", dpi=150)
plt.show()
print("✅ Saved attention_prediction_curve.png")
