# train_cnn_gru.py
import os, glob, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR="aligned_dataset_shifted"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS=200; LR=1e-3; BATCH_SIZE=1; WIN_LEN=128; STRIDE=64; AMP=True
print(f"Running on {DEVICE}")

class AttentionDataset(Dataset):
    def __init__(self,folder):
        self.files=sorted(glob.glob(os.path.join(folder,"*.npz")))
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        d=np.load(self.files[i],allow_pickle=True)
        X,y=d["X"].astype(np.float32),d["y"].astype(np.float32)
        X=(X-X.mean(0))/(X.std(0)+1e-6)
        y=(y-y.min())/(y.max()-y.min()+1e-6)
        return torch.from_numpy(X),torch.from_numpy(y)

def make_windows(x,y,win,stride):
    T=x.size(0)
    if T<=win: return [(x.unsqueeze(0),y.unsqueeze(0))]
    out=[]
    for s in range(0,T-win+1,stride):
        out.append((x[s:s+win].unsqueeze(0),y[s:s+win].unsqueeze(0)))
    return out

class CNN_GRU(nn.Module):
    def __init__(self,in_dim,hidden=64):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(in_dim,hidden,5,padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden,hidden,3,padding=1),
            nn.ReLU()
        )
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.fc=nn.Linear(hidden,1)
    def forward(self,x):
        z=self.conv(x.transpose(1,2)).transpose(1,2)
        z,_=self.gru(z)
        return self.fc(z).squeeze(-1)

ds=AttentionDataset(DATA_DIR)
n_tr=int(0.85*len(ds)); n_va=len(ds)-n_tr
tr,va=random_split(ds,[n_tr,n_va])
tr_loader=DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True)
va_loader=DataLoader(va,batch_size=BATCH_SIZE,shuffle=False)

example_X,_=ds[0]
model=CNN_GRU(example_X.shape[-1]).to(DEVICE)
opt=torch.optim.Adam(model.parameters(),lr=LR)
scaler=torch.cuda.amp.GradScaler(enabled=AMP)
mse=nn.MSELoss()

def derivative_loss(y_pred,y_true):
    return nn.functional.smooth_l1_loss(
        y_pred[:,1:]-y_pred[:,:-1],
        y_true[:,1:]-y_true[:,:-1]
    )

@torch.no_grad()
def evaluate(loader):
    model.eval(); mses=[]; rhos=[]; dmae=[]
    for X,Y in loader:
        X,Y=X.to(DEVICE),Y.to(DEVICE)
        yp=model(X.unsqueeze(0) if X.dim()==2 else X)
        if yp.dim()==3: yp=yp.squeeze(0)
        T=min(Y.size(-1),yp.size(-1))
        yt,yq=Y[...,:T],yp[...,:T]
        mses.append(((yq-yt)**2).mean().item())
        r,_=spearmanr(yt.cpu().numpy().flatten(),yq.cpu().numpy().flatten())
        rhos.append(0 if np.isnan(r) else float(r))
        dmae.append((torch.abs(torch.diff(yq)-torch.diff(yt))).mean().item())
    return np.mean(mses),np.mean(rhos),np.mean(dmae)

for epoch in range(1,EPOCHS+1):
    model.train(); total=0.0
    for X,Y in tr_loader:
        X,Y=X.to(DEVICE),Y.to(DEVICE)
        for xw,yw in make_windows(X.squeeze(0),Y.squeeze(0),WIN_LEN,STRIDE):
            with torch.cuda.amp.autocast(enabled=AMP):
                yp=model(xw)
                loss=0.6*mse(yp,yw)+0.4*derivative_loss(yp,yw)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total+=loss.item()
    val_mse,val_rho,val_dmae=evaluate(va_loader)
    print(f"Epoch {epoch:02d}/{EPOCHS} | Train {total:.4f} | Val MSE {val_mse:.4f} | ρ {val_rho:.3f} | dMAE {val_dmae:.4f}")

X0,Y0=ds[0]
with torch.no_grad():
    pred=model(X0.unsqueeze(0).to(DEVICE)).cpu().numpy().squeeze()
plt.figure(figsize=(10,4))
plt.plot(Y0.numpy(),label="Human",color="orange")
plt.plot(np.clip(pred,0,1),label="Predicted",color="blue")
plt.legend(); plt.tight_layout()
plt.savefig("attention_prediction_cnn_gru200.png",dpi=150)
print("Saved attention_prediction_cnn_gru200.png")
