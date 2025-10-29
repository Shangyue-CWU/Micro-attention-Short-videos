# train_model.py
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Config
# -----------------------------
FEATURES_PATH = "features.npy"       # (T, 3) = [flow, audio, face]
LABELS_PATH   = "labels.npy"         # optional (T,) in [0,1]
SAVE_DIR      = "model"
EPOCHS        = 300
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
HIDDEN        = 64
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# Load features
# -----------------------------
if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("features.npy not found. Run extract_features.py first.")

X = np.load(FEATURES_PATH)  # shape (T, 3)
T, D = X.shape
if D != 3:
    raise ValueError(f"Expected feature dim=3, got {D}")

# -----------------------------
# Labels: load if provided, else create pseudo-labels
# -----------------------------
if os.path.exists(LABELS_PATH):
    y = np.load(LABELS_PATH).astype(np.float32)
    assert len(y) == T, "labels.npy length must equal T (seconds) in features.npy"
    print(f"Loaded labels.npy with shape {y.shape}")
else:
    # Heuristic: attention ~ normalized (flow + face_motion + 0.3 * audio_energy)
    # Safe even when audio was zero-filled.
    flow  = X[:, 0]
    audio = X[:, 1]
    face  = X[:, 2]

    def norm(v):
        v = v - v.min() if v.size > 0 else v
        vmax = v.max() if v.size > 0 else 1.0
        return v / (vmax + 1e-8)

    attn_proxy = 0.5 * norm(flow) + 0.4 * norm(face) + 0.1 * norm(audio)
    y = attn_proxy.astype(np.float32)
    print("No labels.npy found. Generated pseudo-labels from motion/face/audio.")

# Ensure y in [0,1]
y = np.clip(y, 0.0, 1.0)

# -----------------------------
# Train/Val split (time-aware; small T -> keep simple)
# -----------------------------
split = max(1, int(0.8 * T))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# If validation is empty (very short clips), duplicate last frame for stability
if len(X_val) == 0:
    X_val = X_train[-1:].copy()
    y_val = y_train[-1:].copy()

# -----------------------------
# Normalize features (fit on train)
# -----------------------------
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-8
X_train_n = (X_train - mean) / std
X_val_n   = (X_val   - mean) / std

# Save scaler
np.savez(os.path.join(SAVE_DIR, "feature_scaler.npz"), mean=mean, std=std)

# -----------------------------
# Build simple temporal CNN
# -----------------------------
class TemporalAttentionCNN(nn.Module):
    def __init__(self, in_dim=3, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B,T,D)
        x = x.permute(0, 2, 1)         # (B,D,T)
        x = F.relu(self.conv1(x))      # (B,H,T)
        x = F.relu(self.conv2(x))      # (B,H,T)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)         # (B,T,H)
        x = torch.sigmoid(self.out(x)) # (B,T,1)
        return x.squeeze(-1)           # (B,T)

model = TemporalAttentionCNN(in_dim=3, hidden=HIDDEN).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

# To tensors
Xtr = torch.tensor(X_train_n, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, Ttr, D)
ytr = torch.tensor(y_train,  dtype=torch.float32, device=DEVICE).unsqueeze(0)   # (1, Ttr)
Xva = torch.tensor(X_val_n,  dtype=torch.float32, device=DEVICE).unsqueeze(0)   # (1, Tva, D)
yva = torch.tensor(y_val,    dtype=torch.float32, device=DEVICE).unsqueeze(0)   # (1, Tva)

best_val = float("inf")
best_state = None

print(f"Training on {Xtr.shape[1]} seconds, validating on {Xva.shape[1]} seconds.")
for epoch in range(1, EPOCHS + 1):
    model.train()
    opt.zero_grad()
    pred = model(Xtr)
    loss = loss_fn(pred, ytr)
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(Xva)
        val_loss = loss_fn(val_pred, yva).item()

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | train {loss.item():.4f} | val {val_loss:.4f}")

# Save best model
if best_state is not None:
    model.load_state_dict(best_state)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "microattention.pth"))

# Also dump a small training report
report = {
    "epochs": EPOCHS,
    "best_val_mse": float(best_val),
    "train_T": int(Xtr.shape[1]),
    "val_T": int(Xva.shape[1]),
    "device": DEVICE,
}
with open(os.path.join(SAVE_DIR, "train_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"Saved model → {os.path.join(SAVE_DIR, 'microattention.pth')}")
print(f"Saved scaler → {os.path.join(SAVE_DIR, 'feature_scaler.npz')}")
print(f"ℹBest val MSE: {best_val:.4f}")
