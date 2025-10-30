import os, glob, numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

SRC = "aligned_dataset"
DST = "aligned_dataset_shifted"
os.makedirs(DST, exist_ok=True)

def best_lag(x, y, max_lag=5):
    # x, y are 1D arrays (len T)
    T = min(len(x), len(y))
    x, y = x[:T], y[:T]
    x = (x - x.mean()) / (x.std() + 1e-6)
    y = (y - y.mean()) / (y.std() + 1e-6)
    lags = range(-max_lag, max_lag + 1)
    best_L, best_r = 0, -1e9
    for L in lags:
        if L < 0:
            r = np.corrcoef(x[:T+L], y[-L:T])[0,1]
        elif L > 0:
            r = np.corrcoef(x[L:T], y[:T-L])[0,1]
        else:
            r = np.corrcoef(x, y)[0,1]
        if np.isnan(r): r = -1e9
        if r > best_r:
            best_r, best_L = r, L
    return best_L, best_r

def apply_shift(y, L):
    if L < 0:
        return np.concatenate([y[-L:], np.repeat(y[-1], -L)])
    elif L > 0:
        return np.concatenate([np.repeat(y[0], L), y[:-L]])
    return y

files = sorted(glob.glob(os.path.join(SRC, "*.npz")))
lags = []
base_rs, best_rs = [], []
for f in files:
    d = np.load(f, allow_pickle=True)
    X, y = d["X"].astype(np.float32), d["y"].astype(np.float32)
    # feature fuse: motion + audio (expand if you added more features)
    x_fuse = X.mean(axis=1) if X.shape[1] == 1 else X[:, :2].mean(axis=1)
    # normalize y to [0,1] but keep shape
    y = (y - y.min()) / (y.max() - y.min() + 1e-6)
    # baseline corr (no shift)
    T = min(len(x_fuse), len(y))
    base = np.corrcoef(x_fuse[:T], y[:T])[0,1]
    L, r = best_lag(x_fuse, y, max_lag=5)
    y_shift = apply_shift(y, L)
    out = os.path.join(DST, os.path.basename(f))
    np.savez_compressed(out, X=X, y=y_shift, best_lag=L, base_corr=base, best_corr=r)
    lags.append(L); base_rs.append(base); best_rs.append(r)
    print(f"{os.path.basename(f)}  lag={L:+d}s  base_r={base:.3f}  best_r={r:.3f}")

# quick summary
print("\nSummary:")
print("mean lag (s):", np.mean(lags))
print("median lag (s):", np.median(lags))
print("mean base_r:", np.mean(base_rs), " → mean best_r:", np.mean(best_rs))
plt.hist(lags, bins=11); plt.title("Estimated best lags (s)"); plt.savefig("lag_hist.png", dpi=140)
print("Saved lag_hist.png  and shifted dataset to", DST)
