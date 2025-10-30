import os, glob
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

DATA_DIR   = "aligned_dataset"      # or aligned_dataset_noface/_safe
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 25
LR         = 2e-4
BATCH_SIZE = 1
D_MODEL    = 128
N_HEADS    = 4
NLAYERS    = 2
FFN_DIM    = 256
DROPOUT    = 0.1
ENC_WEIGHTS= "transformer_simclr_encoder.pt"  # from pretraining

# ---------- Data ----------
class AttentionDataset(Dataset):
    def __init__(self, folder):
        self.files = glob.glob(os.path.join(folder, "*.npz"))
        self.samples = []
        for f in self.files:
            d = np.load(f, allow_pickle=True)
            X = d["X"].astype(np.float32)
            y = d["y"].astype(np.float32)
            X = (X - X.mean(0)) / (X.std(0)+1e-6)
            y = (y - y.min()) / (y.max() - y.min() + 1e-6)
            self.samples.append((X, y))
        print(f"Loaded {len(self.samples)} videos")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        X, y = self.samples[i]
        return torch.from_numpy(X), torch.from_numpy(y)

ds = AttentionDataset(DATA_DIR)
n_tr = max(1, int(0.8*len(ds))); n_val = len(ds)-n_tr
tr, va = random_split(ds, [n_tr, n_val])
tr_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True)
va_loader = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False)

# ---------- Model ----------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 12000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class TransformerEncoder(nn.Module):
    def __init__(self, in_dim, d_model, nheads, nlayers, ffn_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos  = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nheads, ffn_dim, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
    def forward(self, x):
        z = self.pos(self.proj(x))
        return self.enc(z)

class AttentionRegressor(nn.Module):
    def __init__(self, in_dim=2):
        super().__init__()
        self.encoder = TransformerEncoder(in_dim, D_MODEL, N_HEADS, NLAYERS, FFN_DIM, DROPOUT)
        self.head    = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL//2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(D_MODEL//2, 1)
        )
    def forward(self, x):     # x:(B,T,F)
        z = self.encoder(x)   # (B,T,D)
        y = self.head(z).squeeze(-1)  # (B,T)
        return y

# load encoder from pretraining
model = AttentionRegressor(in_dim=ds[0][0].shape[-1]).to(DEVICE)
if os.path.exists(ENC_WEIGHTS):
    state = torch.load(ENC_WEIGHTS, map_location=DEVICE)
    model.encoder.load_state_dict(state, strict=False)
    print(f"Loaded pretrained encoder weights from {ENC_WEIGHTS}")
else:
    print("⚠️ Pretrained weights not found, training encoder from scratch.")

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    mses, rhos = [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        yp = model(X)
        T = min(yp.shape[1], y.shape[1])
        mses.append(criterion(yp[:,:T], y[:,:T]).item())
        yt = y[:,:T].cpu().numpy().flatten()
        yq = yp[:,:T].cpu().numpy().flatten()
        r, _ = spearmanr(yt, yq)
        rhos.append(0.0 if np.isnan(r) else float(r))
    return float(np.mean(mses)), float(np.mean(rhos))

best = 1e9
for epoch in range(1, EPOCHS+1):
    model.train()
    tr_loss = 0.0
    for X, y in tr_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        yp = model(X)
        T = min(yp.shape[1], y.shape[1])
        loss = criterion(yp[:,:T], y[:,:T])
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item()
    tr_loss /= max(1,len(tr_loader))
    va_mse, va_rho = evaluate(va_loader)
    print(f"Epoch {epoch:02d}/{EPOCHS} | Train MSE {tr_loss:.4f} | Val MSE {va_mse:.4f} | Val Spearman {va_rho:.3f}")
    if va_mse < best:
        best = va_mse
        torch.save(model.state_dict(), "transformer_finetuned.pt")

print("Saved best model → transformer_finetuned.pt")

# visualize one video
X0, y0 = ds[0]
with torch.no_grad():
    yp0 = model(X0.unsqueeze(0).to(DEVICE)).cpu().numpy().squeeze()
y0 = y0.numpy()
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(y0,  label="Human", color="orange")
plt.plot(yp0, label="Predicted (FT)", color="blue")
plt.title("Transformer (SimCLR-pretrained) → Fine-tuned Attention")
plt.xlabel("Seconds"); plt.ylabel("Attention (normalized)")
plt.legend(); plt.tight_layout()
plt.savefig("attention_prediction_transformer_ft.png", dpi=150)
plt.show()
