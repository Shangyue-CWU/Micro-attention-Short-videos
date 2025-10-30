import os, glob
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# =========================
# Config
# =========================
DATA_DIR   = "aligned_dataset"      # or aligned_dataset_noface / _safe
EPOCHS     = 30
LR         = 2e-4
BATCH_SIZE = 1                          # sequences have variable length; keep 1 for simplicity
D_MODEL    = 128                        # Transformer model dim
N_HEADS    = 4
NLAYERS    = 3
FFN_DIM    = 256
DROPOUT    = 0.1
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 1
print(f"Using device: {DEVICE}")

# =========================
# Dataset
# =========================
class AttentionDataset(Dataset):
    """
    Each item returns:
      X: (T, 2)  -> [optical_flow, audio_rms]
      y: (T,)    -> human attention
    We z-score X across features and min-max y per video for stability.
    """
    def __init__(self, folder):
        self.files = glob.glob(os.path.join(folder, "*.npz"))
        self.samples = []
        for f in self.files:
            d = np.load(f, allow_pickle=True)
            X, y = d["X"].astype(np.float32), d["y"].astype(np.float32)
            # Normalize features (per-video)
            X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
            # Normalize labels (per-video) -> [0,1]
            y = (y - y.min()) / (y.max() - y.min() + 1e-6)
            self.samples.append((X, y))
        print(f"Loaded {len(self.samples)} videos from {folder}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.from_numpy(X), torch.from_numpy(y)

dataset = AttentionDataset(DATA_DIR)
n_train = max(1, int(0.8 * len(dataset)))
n_val   = max(1, len(dataset) - n_train)
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# =========================
# Positional Encoding
# =========================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 12000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]

# =========================
# Transformer Model
# =========================
class TransformerAttentionPredictor(nn.Module):
    def __init__(self, in_dim=2, d_model=128, nheads=4, nlayers=3, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc    = SinusoidalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head    = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: (B, T, F=2)
        z = self.input_proj(x)              # (B, T, D)
        z = self.pos_enc(z)                 # (B, T, D)
        z = self.encoder(z)                 # (B, T, D)
        y_hat = self.head(z).squeeze(-1)    # (B, T)
        return y_hat

model = TransformerAttentionPredictor(
    in_dim=2, d_model=D_MODEL, nheads=N_HEADS, nlayers=NLAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# =========================
# Eval helper (MSE + Spearman)
# =========================
@torch.no_grad()
def evaluate(loader):
    model.eval()
    mse_list, rho_list = [], []
    for X, y in loader:
        X = X.to(DEVICE)               # (B=1, T, 2)
        y = y.to(DEVICE)               # (B=1, T)
        y_pred = model(X)              # (B=1, T)
        T = min(y_pred.shape[1], y.shape[1])
        mse = criterion(y_pred[:, :T], y[:, :T]).item()
        mse_list.append(mse)
        # Spearman on CPU numpy
        yt = y[:, :T].detach().cpu().numpy().flatten()
        yp = y_pred[:, :T].detach().cpu().numpy().flatten()
        rho, _ = spearmanr(yt, yp)
        rho_list.append(0.0 if np.isnan(rho) else float(rho))
    return float(np.mean(mse_list)), float(np.mean(rho_list))

# =========================
# Train
# =========================
best_val = 1e9
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for X, y in train_loader:
        X = X.to(DEVICE); y = y.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(X)
        T = min(y_pred.shape[1], y.shape[1])
        loss = criterion(y_pred[:, :T], y[:, :T])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())
    scheduler.step()

    val_mse, val_rho = evaluate(val_loader)
    if epoch % PRINT_EVERY == 1 or epoch == EPOCHS:
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train MSE {np.mean(train_losses):.4f} | Val MSE {val_mse:.4f} | Val Spearman {val_rho:.3f}")

    # simple checkpointing
    if val_mse < best_val:
        best_val = val_mse
        torch.save(model.state_dict(), "transformer_attention_best.pt")

print("✅ Best model saved to transformer_attention_best.pt")

# =========================
# Visualization on one sample
# =========================
model.load_state_dict(torch.load("transformer_attention_best.pt", map_location=DEVICE))
model.eval()
X0, y0 = dataset[0]
with torch.no_grad():
    yp0 = model(X0.unsqueeze(0).to(DEVICE)).cpu().numpy().squeeze()

y0 = y0.numpy()
plt.figure(figsize=(10,4))
plt.plot(y0,  label="Human Attention", color="orange")
plt.plot(yp0, label="Predicted Attention (Transformer)", color="blue")
plt.title("Transformer-based Micro-Attention Prediction")
plt.xlabel("Seconds"); plt.ylabel("Attention (normalized)")
plt.legend(); plt.tight_layout()
plt.savefig("attention_prediction_transformer.png", dpi=150)
plt.show()
print("✅ Saved attention_prediction_transformer.png")
