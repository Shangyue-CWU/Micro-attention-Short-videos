# train_attention_local_fixed.py
import os, glob, math, warnings, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =======================
# Config
# =======================
DATA_DIR   = "aligned_dataset_shifted"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 100
LR         = 1e-4
BATCH_SIZE = 1
D_MODEL    = 160; N_HEADS = 4; NLAYERS = 3; FFN_DIM = 320; DROPOUT = 0.1
WIN_LEN    = 256; STRIDE = 128
W_MSE=0.25; W_CORR=0.25; W_DER=0.25; W_PEAK=0.25
W_GRAD=0.10
W_ATTEN = 64
USE_GLOBAL_NORM = True
AMP = True

print(f"Running on {DEVICE} | DATA_DIR={DATA_DIR}")

# =======================
# Dataset + global stats
# =======================
class AttentionDataset(Dataset):
    def __init__(self, folder, feat_mean=None, feat_std=None):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        self.feat_mean = feat_mean
        self.feat_std  = feat_std

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i], allow_pickle=True)
        X, y = d["X"].astype(np.float32), d["y"].astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0)+1e-6)
        if self.feat_mean is not None and self.feat_std is not None:
            X = (X - self.feat_mean) / (self.feat_std + 1e-6)
        y = (y - y.min()) / (y.max() - y.min() + 1e-6)
        return torch.from_numpy(X), torch.from_numpy(y)

def compute_global_stats(files, max_files=500):
    feats = []
    for fp in files[:max_files]:
        d = np.load(fp, allow_pickle=True)
        X = d["X"].astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        feats.append(X)
    A = np.concatenate(feats, axis=0)
    return A.mean(0).astype(np.float32), A.std(0).astype(np.float32)

_tmp = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
gmean, gstd = (None, None)
if USE_GLOBAL_NORM and len(_tmp):
    print("Computing global feature mean/std (subset)…")
    gmean, gstd = compute_global_stats(_tmp, max_files=min(800,len(_tmp)))

ds = AttentionDataset(DATA_DIR, feat_mean=gmean, feat_std=gstd)
n_tr = max(1,int(0.85*len(ds))); n_va = len(ds)-n_tr
tr, va = random_split(ds,[n_tr,n_va])
tr_loader = DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True)
va_loader = DataLoader(va,batch_size=BATCH_SIZE,shuffle=False)

# =======================
# Modules
# =======================
class PosEnc(nn.Module):
    def __init__(self,d_model,max_len=20000):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x):
        T=x.size(1)
        return x+self.pe[:,:T]

# ----- FIXED MULTISCALE TCN -----
class MultiScaleTCN(nn.Module):
    """Multi-scale (dilated) front-end with consistent temporal size."""
    def __init__(self,in_dim,hidden=128):
        super().__init__()
        self.cfgs=[(3,1),(5,2),(7,4)]
        self.branches=nn.ModuleList()
        for k,d in self.cfgs:
            pad=(k+(k-1)*(d-1))//2   # 'same' padding
            branch=nn.Sequential(
                nn.Conv1d(in_dim,hidden,k,padding=pad,dilation=d,groups=min(in_dim,16)),
                nn.ReLU(),
                nn.Conv1d(hidden,hidden,1),
                nn.ReLU(),
            )
            self.branches.append(branch)
        self.merge=nn.Sequential(
            nn.Conv1d(len(self.cfgs)*hidden,in_dim,1),nn.ReLU(),
            nn.Conv1d(in_dim,in_dim,1)
        )
    def forward(self,x):
        z=x.transpose(1,2)
        outs=[b(z) for b in self.branches]
        T=min(o.size(-1) for o in outs)
        outs=[o[...,:T] for o in outs]
        feats=torch.cat(outs,dim=1)
        z_out=self.merge(feats)[...,:T]
        return z_out.transpose(1,2)

def build_local_band_mask(T,w):
    """adaptive local attention mask"""
    idx=torch.arange(T)
    dist=(idx[None,:]-idx[:,None]).abs()
    mask=dist>w
    return mask

class LocalTransformerEncoder(nn.Module):
    def __init__(self,d_model,nhead,ffn_dim,nlayers,dropout):
        super().__init__()
        layer=nn.TransformerEncoderLayer(
            d_model=d_model,nhead=nhead,dim_feedforward=ffn_dim,
            dropout=dropout,batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(layer,nlayers)
    def forward(self,x,mask):
        return self.enc(x,mask=mask)

class Model(nn.Module):
    def __init__(self,in_dim,d_model=D_MODEL,w_attn=W_ATTEN):
        super().__init__()
        self.w_attn=w_attn
        self.front=MultiScaleTCN(in_dim)
        self.proj=nn.Sequential(nn.Linear(in_dim,d_model),nn.LayerNorm(d_model))
        self.pos=PosEnc(d_model)
        self.temporal=LocalTransformerEncoder(d_model,N_HEADS,FFN_DIM,NLAYERS,DROPOUT)
        self.head=nn.Sequential(
            nn.Linear(d_model,d_model//2),nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2,1)
        )
        self.peak=nn.Sequential(
            nn.Conv1d(d_model,d_model//2,3,padding=1),nn.ReLU(),
            nn.Conv1d(d_model//2,1,3,padding=1),nn.Sigmoid()
        )
        self.scale=nn.Parameter(torch.tensor(1.0))
    def forward(self,x):
        x=self.front(x)
        z=self.proj(x)
        z=self.pos(z)
        T=z.size(1)
        mask=build_local_band_mask(T,self.w_attn).to(z.device)
        z=self.temporal(z,mask)
        y_main=self.head(z).squeeze(-1)
        p=self.peak(z.transpose(1,2)).squeeze(1)
        y=(y_main+0.3*p)*self.scale
        return y

# =======================
# Losses & helpers
# =======================
mse=nn.MSELoss()
huber=nn.SmoothL1Loss(beta=0.5)

def corr_loss(y_pred,y_true,eps=1e-8):
    yp=y_pred-y_pred.mean(dim=1,keepdim=True)
    yt=y_true-y_true.mean(dim=1,keepdim=True)
    num=(yp*yt).sum(dim=1)
    den=torch.sqrt((yp**2).sum(dim=1)*(yt**2).sum(dim=1)+eps)
    return (1-num/den).mean()

def derivative_loss(y_pred,y_true):
    dy_p=y_pred[:,1:]-y_pred[:,:-1]
    dy_t=y_true[:,1:]-y_true[:,:-1]
    return nn.functional.smooth_l1_loss(dy_p,dy_t)

def grad_loss(y_pred,y_true):
    dy_p=y_pred[:,1:]-y_pred[:,:-1]
    dy_t=y_true[:,1:]-y_true[:,:-1]
    return ((dy_p-dy_t)**2).mean()

def peak_weighted_mse(y_pred,y_true,k=3.0):
    with torch.no_grad():
        dy=y_true[:,1:]-y_true[:,:-1]
        w=1.0+k*(torch.abs(dy)/(torch.abs(dy).max(dim=1,keepdim=True).values+1e-6))
        w=torch.cat([w[:,:1],w],dim=1)
    return (w*(y_pred-y_true)**2).mean()

def make_windows(x,y,win=256,stride=128):
    T=x.size(0)
    if T<=win: return [(x.unsqueeze(0),y.unsqueeze(0))]
    out=[]
    for s in range(0,T-win+1,stride):
        out.append((x[s:s+win].unsqueeze(0),y[s:s+win].unsqueeze(0)))
    if (T-win)%stride!=0:
        out.append((x[-win:].unsqueeze(0),y[-win:].unsqueeze(0)))
    return out

def peak_f1(y_true,y_pred,tol=3,topk=0.1):
    yt,yp=y_true.flatten(),y_pred.flatten()
    k=max(1,int(len(yt)*topk))
    thr_t=torch.topk(yt,k).values.min()
    thr_p=torch.topk(yp,k).values.min()
    idx_t=torch.where(yt>=thr_t)[0]
    idx_p=torch.where(yp>=thr_p)[0]
    tp,used=0,set()
    for i in idx_p.tolist():
        lo,hi=max(0,i-tol),min(len(yt)-1,i+tol)
        cand=idx_t[(idx_t>=lo)&(idx_t<=hi)].tolist()
        if cand:
            j=min(cand,key=lambda c:abs(c-i))
            if j not in used:
                used.add(j); tp+=1
    fp=len(idx_p)-tp; fn=len(idx_t)-tp
    prec=tp/(tp+fp+1e-6); rec=tp/(tp+fn+1e-6)
    return 2*prec*rec/(prec+rec+1e-6)

# =======================
# Build model & train
# =======================
example_X,_=ds[0]
model=Model(in_dim=example_X.shape[-1]).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
scaler=torch.cuda.amp.GradScaler(enabled=AMP)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    mses,rhos,dmae,pf1=[],[],[],[]
    for X,Y in loader:
        X,Y=X.to(DEVICE),Y.to(DEVICE)
        yp=model(X.unsqueeze(0) if X.dim()==2 else X)
        if yp.dim()==3: yp=yp.squeeze(0)
        T=min(Y.size(-1),yp.size(-1))
        yt, yq=Y[...,:T], yp[...,:T]
        mses.append(((yq-yt)**2).mean().item())
        r,_=spearmanr(yt.cpu().numpy().flatten(),yq.cpu().numpy().flatten())
        rhos.append(0 if np.isnan(r) else float(r))
        dmae.append((torch.abs(torch.diff(yq)-torch.diff(yt))).mean().item())
        pf1.append(float(peak_f1(yt.cpu(),yq.cpu(),tol=3,topk=0.1)))
    return np.mean(mses),np.mean(rhos),np.mean(dmae),np.mean(pf1)

best=1e9
for epoch in range(1,EPOCHS+1):
    model.train(); total=0; steps=0
    for X,Y in tr_loader:
        X,Y=X.to(DEVICE),Y.to(DEVICE)
        for xw,yw in make_windows(X.squeeze(0),Y.squeeze(0),WIN_LEN,STRIDE):
            with torch.cuda.amp.autocast(enabled=AMP):
                yp=model(xw)
                loss=W_MSE*huber(yp,yw)
                loss+=W_CORR*corr_loss(yp,yw)
                loss+=W_DER*derivative_loss(yp,yw)
                loss+=W_PEAK*peak_weighted_mse(yp,yw)
                loss+=W_GRAD*grad_loss(yp,yw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
            total+=float(loss.detach()); steps+=1
    sch.step()
    val_mse,val_rho,val_dmae,val_pf1=evaluate(va_loader)
    print(f"Epoch {epoch:02d}/{EPOCHS} | Train {total/max(1,steps):.4f} | "
          f"Val MSE {val_mse:.4f} | ρ {val_rho:.3f} | dMAE {val_dmae:.4f} | PeakF1 {val_pf1:.3f}")
    if val_mse<best:
        best=val_mse
        torch.save(model.state_dict(),"local_transformer_multiscale_peak_best.pt")

print("Saved → local_transformer_multiscale_peak_best.pt")

# =======================
# Visualization
# =======================
X0,Y0=ds[0]
with torch.no_grad():
    pred=model(X0.unsqueeze(0).to(DEVICE)).cpu().numpy().squeeze()
Y0=Y0.numpy()
pred_plot=np.clip(pred,0,1)
plt.figure(figsize=(10,4))
plt.plot(Y0,label="Human",color="orange")
plt.plot(pred_plot,label="Predicted",color="blue")
plt.title("Local-Attention Transformer + MultiScale TCN (fixed)")
plt.xlabel("Frames"); plt.ylabel("Attention (normalized)")
plt.legend(); plt.tight_layout()
plt.savefig("attention_prediction_local_fixed2.png",dpi=150)
print("Saved attention_prediction_local_fixed2.png")
