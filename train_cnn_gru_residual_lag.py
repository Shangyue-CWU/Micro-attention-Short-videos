# train_cnn_gru_residual_lag.py
# Lightweight CNN+GRU with residual skip and lag-aware loss for micro-attention regression.

import os, glob, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------
# Config
# ------------------
DATA_DIR   = "aligned_dataset_shifted"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 200
LR         = 2e-3
BATCH_SIZE = 1
WIN_LEN    = 128
STRIDE     = 64
AMP        = True
MAX_SHIFT  = 2      # ± frames tolerance in lag-aware MSE

W_MSE      = 0.40
W_DER      = 0.40
W_LAG      = 0.20   # set 0.10–0.30; increase if small residual lag persists

print(f"Running on {DEVICE} | DATA_DIR={DATA_DIR}")

# ------------------
# Dataset
# ------------------
class AttentionDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i], allow_pickle=True)
        X, y = d["X"].astype(np.float32), d["y"].astype(np.float32)
        # Per-clip feature z-score across time (feature-wise)
        X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
        # Keep labels in [0,1] for stable visualization/metrics
        y = (y - y.min()) / (y.max() - y.min() + 1e-6)
        return torch.from_numpy(X), torch.from_numpy(y)

def make_windows(x, y, win=128, stride=64):
    T = x.size(0)
    if T <= win:
        return [(x.unsqueeze(0), y.unsqueeze(0))]
    out = []
    for s in range(0, T - win + 1, stride):
        out.append((x[s:s+win].unsqueeze(0), y[s:s+win].unsqueeze(0)))
    # (optional) tail window can be added if you want full coverage
    return out

# ------------------
# Model
# ------------------
class CNN_GRU_Residual(nn.Module):
    def __init__(self, in_dim, hidden=96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU()
        )
        self.gru  = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.fc   = nn.Linear(hidden, 1)
        self.skip = nn.Linear(in_dim, 1)  # residual trend from raw features
        self.drop = nn.Dropout(0.1)

    def forward(self, x):                 # x: (B,T,F)
        z = self.conv(x.transpose(1,2)).transpose(1,2)  # (B,T,H)
        z, _ = self.gru(z)                               # (B,T,H)
        z = self.drop(z)
        y = self.fc(z).squeeze(-1)                       # (B,T)
        skip = self.skip(x).squeeze(-1)                  # (B,T)
        return 0.85 * y + 0.15 * skip                    # small residual blend

# ------------------
# Losses & Metrics
# ------------------
mse = nn.MSELoss()

def derivative_loss(y_pred, y_true):
    return F.smooth_l1_loss(y_pred[:,1:] - y_pred[:,:-1],
                            y_true[:,1:] - y_true[:,:-1])

def shift_like(y, s):
    """Edge-replicated shift by s frames (positive = shift right / delay)."""
    if s == 0:
        return y
    if s > 0:
        pad = y[:, :1].repeat(1, s)
        return torch.cat([pad, y[:, :-s]], dim=1)
    else:
        s = -s
        pad = y[:, -1:].repeat(1, s)
        return torch.cat([y[:, s:], pad], dim=1)

def lag_aware_mse(y_pred, y_true, max_shift=2):
    best = None
    for s in range(-max_shift, max_shift + 1):
        y_shift = shift_like(y_true, s)
        loss = ((y_pred - y_shift)**2).mean()
        best = loss if best is None else torch.minimum(best, loss)
    return best

def peak_f1(y_true, y_pred, tol=3, topk=0.1):
    yt, yp = y_true.flatten(), y_pred.flatten()
    k = max(1, int(len(yt) * topk))
    thr_t = torch.topk(yt, k).values.min()
    thr_p = torch.topk(yp, k).values.min()
    idx_t = torch.where(yt >= thr_t)[0]
    idx_p = torch.where(yp >= thr_p)[0]
    tp, used = 0, set()
    for i in idx_p.tolist():
        lo, hi = max(0, i - tol), min(len(yt) - 1, i + tol)
        cand = idx_t[(idx_t >= lo) & (idx_t <= hi)].tolist()
        if cand:
            j = min(cand, key=lambda c: abs(c - i))
            if j not in used:
                used.add(j); tp += 1
    fp = len(idx_p) - tp; fn = len(idx_t) - tp
    prec = tp / (tp + fp + 1e-6); rec = tp / (tp + fn + 1e-6)
    return 2 * prec * rec / (prec + rec + 1e-6)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mses, rhos, d_mae, pf1s = [], [], [], []
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        yp = model(X.unsqueeze(0) if X.dim() == 2 else X)
        if yp.dim() == 3: yp = yp.squeeze(0)
        T = min(Y.size(-1), yp.size(-1))
        yt, yq = Y[..., :T], yp[..., :T]
        mses.append(((yq - yt)**2).mean().item())
        r, _ = spearmanr(yt.cpu().numpy().flatten(), yq.cpu().numpy().flatten())
        rhos.append(0 if np.isnan(r) else float(r))
        d_mae.append((torch.abs(torch.diff(yq) - torch.diff(yt))).mean().item())
        pf1s.append(float(peak_f1(yt.cpu(), yq.cpu(), tol=3, topk=0.10)))
    return float(np.mean(mses)), float(np.mean(rhos)), float(np.mean(d_mae)), float(np.mean(pf1s))

# ------------------
# Train
# ------------------
ds = AttentionDataset(DATA_DIR)
n_tr = max(1, int(0.85 * len(ds))); n_va = len(ds) - n_tr
tr, va = random_split(ds, [n_tr, n_va])
tr_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True)
va_loader = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False)

example_X, _ = ds[5]
model = CNN_GRU_Residual(example_X.shape[-1]).to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler= torch.cuda.amp.GradScaler(enabled=AMP)

best = 1e9
for epoch in range(1, EPOCHS + 1):
    model.train(); total = 0.0; steps = 0
    for X, Y in tr_loader:
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        for xw, yw in make_windows(X.squeeze(0), Y.squeeze(0), WIN_LEN, STRIDE):
            with torch.cuda.amp.autocast(enabled=AMP):
                yp = model(xw)  # (1,T)
                loss = 0.0
                loss += W_MSE * mse(yp, yw)
                loss += W_DER * derivative_loss(yp, yw)
                if W_LAG > 1e-9:
                    loss += W_LAG * lag_aware_mse(yp, yw, max_shift=MAX_SHIFT)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total += float(loss.detach().item()); steps += 1
    sch.step()

    val_mse, val_rho, val_dmae, val_pf1 = evaluate(model, va_loader, DEVICE)
    print(f"Epoch {epoch:02d}/{EPOCHS} | Train {total/max(1,steps):.4f} | "
          f"Val MSE {val_mse:.4f} | ρ {val_rho:.3f} | dMAE {val_dmae:.4f} | PeakF1 {val_pf1:.3f}")

    if val_mse < best:
        best = val_mse
        torch.save(model.state_dict(), "cnn_gru_residual_lag_best.pt")

print("Saved → cnn_gru_residual_lag_best.pt")

# ------------------
# Visualization
# ------------------
X0, Y0 = ds[5]
with torch.no_grad():
    pred = model(X0.unsqueeze(0).to(DEVICE)).cpu().numpy().squeeze()
Y0 = Y0.numpy()
pred_plot = np.clip(pred, 0, 1)
plt.figure(figsize=(10,4))
plt.plot(Y0,        label="Human",     color="orange")
plt.plot(pred_plot, label="Predicted", color="blue")
plt.title("CNN-GRU (Residual) + Lag-aware loss")
plt.xlabel("Frames"); plt.ylabel("Attention (normalized)")
plt.legend(); plt.tight_layout()
plt.savefig("attention_prediction_cnn_gru_residual_lag5_100.png", dpi=150)
print("Saved attention_prediction_cnn_gru_residual_lag5_100.png")


for i in range(len(ds)):
    X0, Y0 = ds[i]
    with torch.no_grad():
        pred = model(X0.unsqueeze(0).to(DEVICE)).cpu().numpy().squeeze()
    Y0 = Y0.numpy()
    pred_plot = np.clip(pred, 0, 1)
    plt.figure(figsize=(10,4))
    plt.plot(Y0,        label="Human",     color="orange")
    plt.plot(pred_plot, label="Predicted", color="blue")
    plt.title("CNN-GRU (Residual) + Lag-aware loss")
    plt.xlabel("Frames"); plt.ylabel("Attention (normalized)")
    plt.legend(); plt.tight_layout()
    plt.savefig("eval_results/attention_prediction_cnn_gru_residual_lag0%d_100.png"%i, dpi=150)
    print("Saved attention_prediction_cnn_gru_residual_lag0%d_100.png"%i)
