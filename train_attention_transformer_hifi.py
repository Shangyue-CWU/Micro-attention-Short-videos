import os, glob
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")  # avoid blocking
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
DATA_DIR    = "aligned_dataset"   # as requested
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS      = 40
LR          = 1e-4
BATCH_SIZE  = 1                    # variable-length; keep 1
D_MODEL     = 160
N_HEADS     = 4
NLAYERS     = 4
FFN_DIM     = 320
DROPOUT     = 0.1
WIN_LEN     = 256                  # seconds per training window
STRIDE      = 128                  # overlap stride
ALPHA_MSE   = 0.40                 # value match
BETA_CORR   = 0.30                 # rank/shape
GAMMA_DER   = 0.20                 # slope match
DELTA_PEAK  = 0.10                 # peak-weighted mse
AMP         = True                 # mixed precision
PRINT_EVERY = 1

print(f"Running on {DEVICE} | AMP={AMP}")

# ======================
# DATA
# ======================
class AttentionDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        self.samples=[]
        for f in self.files:
            d=np.load(f, allow_pickle=True)
            X=d["X"].astype(np.float32)      # (T, F)
            y=d["y"].astype(np.float32)      # (T,)
            # per-video normalization
            X=(X-X.mean(0))/(X.std(0)+1e-6)
            y=(y-y.min())/(y.max()-y.min()+1e-6)
            self.samples.append((X,y))
        print(f"Loaded {len(self.samples)} videos from {folder}")
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        X,y=self.samples[i]
        return torch.from_numpy(X), torch.from_numpy(y)

ds = AttentionDataset(DATA_DIR)
n_tr = max(1, int(0.85*len(ds))); n_val = len(ds)-n_tr
tr, va = random_split(ds, [n_tr, n_val])
tr_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True)
va_loader = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False)

# ======================
# MODEL
# ======================
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2)*(-np.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x):
        return x + self.pe[:,:x.size(1)]

class TransformerRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, D_MODEL), nn.LayerNorm(D_MODEL))
        self.pos  = PosEnc(D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            dropout=DROPOUT, batch_first=True, norm_first=True
        )
        self.enc  = nn.TransformerEncoder(layer, NLAYERS)  # bidirectional
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL//2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(D_MODEL//2, 1)
        )
        self.scale = nn.Parameter(torch.tensor(1.0))        # learn amplitude

    def forward(self, x):
        z = self.pos(self.proj(x))   # (B,T,D)
        z = self.enc(z)              # (B,T,D)
        y = self.head(z).squeeze(-1) * self.scale
        return torch.sigmoid(y)      # (B,T) in [0,1]

model = TransformerRegressor(in_dim=ds[0][0].shape[-1]).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=AMP)

# ======================
# LOSSES (vectorized & fast)
# ======================
mse = nn.MSELoss()

def corr_loss(y_pred, y_true, eps=1e-8):
    # y_pred, y_true: (B,T)
    y_pred = y_pred - y_pred.mean(dim=1, keepdim=True)
    y_true = y_true - y_true.mean(dim=1, keepdim=True)
    num = (y_pred * y_true).sum(dim=1)
    den = torch.sqrt((y_pred**2).sum(dim=1) * (y_true**2).sum(dim=1) + eps)
    corr = num / den
    return (1 - corr).mean()

def derivative_loss(y_pred, y_true):
    # encourage matching slopes (first differences)
    dy_p = y_pred[:,1:] - y_pred[:,:-1]
    dy_t = y_true[:,1:] - y_true[:,:-1]
    return nn.functional.smooth_l1_loss(dy_p, dy_t)

def peak_weighted_mse(y_pred, y_true, k=3.0):
    # weight regions where ground truth changes rapidly
    # w in [1, 1+k]; multiply MSE by w (stop gradient through w)
    with torch.no_grad():
        dy = y_true[:,1:] - y_true[:,:-1]
        w = 1.0 + k * (torch.abs(dy) / (torch.abs(dy).max(dim=1, keepdim=True).values + 1e-6))
        w = torch.cat([w[:, :1], w], dim=1)  # align length T (simple edge pad)
    return ((w * (y_pred - y_true)**2).mean())

# ======================
# WINDOWING UTILS
# ======================
def make_windows(x, y, win_len=WIN_LEN, stride=STRIDE):
    # x:(T,F), y:(T,) -> list of (xw, yw) windows
    T = x.size(0)
    if T <= win_len:
        return [(x.unsqueeze(0), y.unsqueeze(0))]
    ws = []
    for s in range(0, T - win_len + 1, stride):
        ws.append((x[s:s+win_len].unsqueeze(0), y[s:s+win_len].unsqueeze(0)))
    if (T - win_len) % stride != 0:
        # tail window
        ws.append((x[-win_len:].unsqueeze(0), y[-win_len:].unsqueeze(0)))
    return ws

# ======================
# EVALUATION
# ======================
@torch.no_grad()
def evaluate(loader):
    model.eval()
    mses, rhos = [], []
    for X, Y in loader:
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        # run full sequence (may be long; ok for eval)
        y_pred = model(X.unsqueeze(0) if X.dim()==2 else X)
        if y_pred.dim() == 3: y_pred = y_pred.squeeze(0)  # (T,)
        T = min(Y.size(-1), y_pred.size(-1))
        yt = Y[..., :T].detach().cpu().numpy().flatten()
        yp = y_pred[..., :T].detach().cpu().numpy().flatten()
        mses.append(np.mean((yp - yt)**2))
        r, _ = spearmanr(yt, yp)
        rhos.append(0 if np.isnan(r) else r)
    return float(np.mean(mses)), float(np.mean(rhos))

# ======================
# TRAIN
# ======================
best = 1e9
for epoch in range(1, EPOCHS+1):
    model.train()
    total = 0.0
    for X, Y in tr_loader:
        X, Y = X.to(DEVICE), Y.to(DEVICE)  # X:(T,F) Y:(T,)
        # sliding windows per video
        windows = make_windows(X.squeeze(0), Y.squeeze(0), WIN_LEN, STRIDE)

        for xw, yw in windows:
            with torch.cuda.amp.autocast(enabled=AMP):
                yp = model(xw)      # (1, win_len)
                # compute fast composite loss
                loss_val  = ALPHA_MSE * mse(yp, yw)
                loss_val += BETA_CORR * corr_loss(yp, yw)
                loss_val += GAMMA_DER * derivative_loss(yp, yw)
                loss_val += DELTA_PEAK * peak_weighted_mse(yp, yw)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss_val).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss_val.detach().item())

    sched.step()
    val_mse, val_rho = evaluate(va_loader)
    if epoch % PRINT_EVERY == 0:
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train {total/max(1,len(tr_loader)):.4f} | Val MSE {val_mse:.4f} | Val Spearman {val_rho:.3f}")

    if val_mse < best:
        best = val_mse
        torch.save(model.state_dict(), "transformer_best_fast.pt")

print("✅ Training complete — transformer_best_fast.pt saved.")

# ======================
# VISUALIZE one example
# ======================
X0, Y0 = ds[0]
X0, Y0 = X0.to(DEVICE), Y0.numpy()
model.eval()
with torch.no_grad():
    yp0 = model(X0.unsqueeze(0)).cpu().numpy().squeeze()

plt.figure(figsize=(10,4))
plt.plot(Y0,  label="Human",    color="orange")
plt.plot(yp0, label="Predicted", color="blue")
plt.title("Fast Transformer Micro-Attention (MSE+Corr+Deriv+Peak)")
plt.xlabel("Seconds"); plt.ylabel("Attention (normalized)")
plt.legend(); plt.tight_layout()
plt.savefig("attention_prediction_fast.png", dpi=150)
print("Saved attention_prediction_fast.png")
