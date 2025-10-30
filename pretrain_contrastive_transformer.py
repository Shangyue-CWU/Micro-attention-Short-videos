import os, glob, math, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------- Config ----------------
DATA_DIR    = "aligned_dataset"   # or aligned_dataset_noface/_safe
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS      = 40
BATCH_SIZE  = 64                      # this is number of windows (not videos)
LR          = 2e-4
D_MODEL     = 128
N_HEADS     = 4
NLAYERS     = 2
FFN_DIM     = 256
DROPOUT     = 0.1
WIN_LEN     = 32                      # seconds per window
TAU         = 0.07                    # NT-Xent temperature
PRINT_EVERY = 1

# --------- Utils: temporal augmentations ----------
def time_crop(X, win_len):
    if X.shape[0] <= win_len:  # pad by repeat if too short
        k = math.ceil(win_len / X.shape[0])
        X = np.tile(X, (k,1))
    start = np.random.randint(0, X.shape[0]-win_len+1)
    return X[start:start+win_len].copy()

def jitter(X, sigma=0.02):
    return X + np.random.normal(0, sigma, X.shape).astype(np.float32)

def scaling(X, low=0.9, high=1.1):
    s = np.random.uniform(low, high, (1, X.shape[1])).astype(np.float32)
    return (X * s).astype(np.float32)

def time_mask(X, p=0.1):
    X = X.copy()
    m = np.random.rand(*X.shape[:1]) < p
    X[m] = 0
    return X

def aug_pair(X, win_len):
    x = time_crop(X, win_len)
    a1 = jitter(scaling(time_mask(x, p=0.15),), sigma=0.03)
    a2 = jitter(scaling(time_mask(x, p=0.15),), sigma=0.03)
    return a1, a2

# ---------------- Dataset ----------------
class ContrastiveWindows(Dataset):
    def __init__(self, folder, win_len=WIN_LEN):
        self.files = glob.glob(os.path.join(folder, "*.npz"))
        self.samples = []
        for f in self.files:
            d = np.load(f, allow_pickle=True)
            X = d["X"].astype(np.float32)  # (T, F=2/3)
            # per-video z-score for stability
            X = (X - X.mean(0)) / (X.std(0) + 1e-6)
            self.samples.append(X)
        self.win_len = win_len

    def __len__(self):
        return sum(max(1, len(x)//self.win_len) for x in self.samples)

    def __getitem__(self, idx):
        # sample a video uniformly, then two views of a window
        X = random.choice(self.samples)
        x1, x2 = aug_pair(X, self.win_len)
        return torch.from_numpy(x1), torch.from_numpy(x2)

# --------------- Model -------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 12000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,T,D)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerEncoder(nn.Module):
    def __init__(self, in_dim, d_model, nheads, nlayers, ffn_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos  = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nheads, ffn_dim, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
    def forward(self, x):  # x:(B,T,F)
        z = self.pos(self.proj(x))
        return self.enc(z)  # (B,T,D)

class SimCLRHead(nn.Module):
    def __init__(self, d_model, proj_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, proj_dim)
        )
    def forward(self, z):  # z:(B,T,D)
        z = z.transpose(1,2)                # (B,D,T)
        g = self.pool(z).squeeze(-1)        # (B,D)
        h = self.mlp(g)                     # (B,P)
        return nn.functional.normalize(h, dim=1)

def nt_xent_loss(z1, z2, tau=TAU):
    z = torch.cat([z1, z2], dim=0)                       # (2B, P)
    sim = torch.matmul(z, z.T) / tau                     # cosine since z normalized
    B = z1.size(0)
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    # positives: (i, i+B) and (i+B, i)
    pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)
    logsumexp = torch.logsumexp(sim, dim=1)
    loss = (-pos + logsumexp).mean()
    return loss

# --------------- Training ----------------
def main():
    ds = ContrastiveWindows(DATA_DIR, WIN_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    # infer feature dim from one sample
    x1, _ = next(iter(dl))
    F = x1.shape[-1]

    enc  = TransformerEncoder(F, D_MODEL, N_HEADS, NLAYERS, FFN_DIM, DROPOUT).to(DEVICE)
    head = SimCLRHead(D_MODEL, proj_dim=128).to(DEVICE)
    opt  = torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), lr=LR, weight_decay=1e-4)

    for epoch in range(1, EPOCHS+1):
        enc.train(); head.train()
        losses = []
        for a, b in tqdm(dl, desc=f"Pretrain {epoch}/{EPOCHS}"):
            a, b = a.to(DEVICE), b.to(DEVICE)
            za = head(enc(a))
            zb = head(enc(b))
            loss = nt_xent_loss(za, zb, TAU)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        if epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch} | SimCLR loss: {np.mean(losses):.4f}")

    torch.save(enc.state_dict(), "transformer_simclr_encoder.pt")
    print("Saved pretrained encoder → transformer_simclr_encoder.pt")

if __name__ == "__main__":
    main()
