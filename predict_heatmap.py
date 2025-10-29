# predict_heatmap.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

FEATURES_PATH = "features.npy"
MODEL_DIR     = "model"
MODEL_PATH    = os.path.join(MODEL_DIR, "microattention.pth")
SCALER_PATH   = os.path.join(MODEL_DIR, "feature_scaler.npz")
OUT_DIR       = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Model (must match training)
# -----------------------------
class TemporalAttentionCNN(nn.Module):
    def __init__(self, in_dim=3, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):  # (B,T,D)
        x = x.permute(0, 2, 1)         # (B,D,T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)         # (B,T,H)
        x = torch.sigmoid(self.out(x)) # (B,T,1)
        return x.squeeze(-1)

# -----------------------------
# Load features & scaler
# -----------------------------
if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("features.npy not found. Run extract_features.py first.")

X = np.load(FEATURES_PATH)  # (T, 3)
scaler = np.load(SCALER_PATH)
mean, std = scaler["mean"], scaler["std"]
Xn = (X - mean) / (std + 1e-8)

# -----------------------------
# Load model & predict
# -----------------------------
model = TemporalAttentionCNN(in_dim=3, hidden=64)
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()

Xt = torch.tensor(Xn, dtype=torch.float32).unsqueeze(0)  # (1,T,3)
with torch.no_grad():
    attn = model(Xt).squeeze(0).cpu().numpy()  # (T,)

# -----------------------------
# Save + visualize
# -----------------------------
# CSV
csv_path = os.path.join(OUT_DIR, "attention_scores.csv")
np.savetxt(csv_path, attn, delimiter=",")
print(f"Saved per-second attention → {csv_path}")

# Plot
plt.figure(figsize=(10, 3))
plt.plot(attn)
plt.ylim(0, 1)
plt.title("Predicted Micro-Attention (per second)")
plt.xlabel("Time (s)")
plt.ylabel("Attention [0–1]")
plt.tight_layout()
png_path = os.path.join(OUT_DIR, "attention_curve.png")
plt.savefig(png_path, dpi=150)
plt.close()
print(f"Saved plot → {png_path}")
