import numpy as np, glob, matplotlib.pyplot as plt

files = glob.glob("aligned_dataset/*.npz")
X_all, y_all = [], []
for f in files:
    d = np.load(f)
    X_all.append(d['X'])
    y_all.append(d['y'])
X = np.vstack(X_all)
y = np.concatenate(y_all)

plt.scatter(X[:,0], y, alpha=0.3, label='Optical Flow')
plt.scatter(X[:,1], y, alpha=0.3, label='Audio RMS')
plt.legend(); plt.xlabel('Feature Value'); plt.ylabel('Attention')
plt.show()


