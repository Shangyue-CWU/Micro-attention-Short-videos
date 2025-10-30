"""
load_tvsum_final.py
-------------------
For TVSum v1.1 MATLAB v7.3 (.mat) where each field is (50,1)
and user_anno[i][0] is (20, n_frames).
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

ANNOT_PATH = "matlab/tvsum50_annotations/ydata-tvsum50.mat"
OUTPUT_DIR = "processed_tvsum"
FPS = 30  # approximate
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading annotations from {ANNOT_PATH} ...")
f = h5py.File(ANNOT_PATH, "r")
root = f["tvsum50"]

def decode_str(ref):
    arr = f[ref][()]
    return "".join(chr(int(c[0])) for c in arr)

video_refs = root["video"]
category_refs = root["category"]
user_refs = root["user_anno"]

n_videos = video_refs.shape[0]
print(f"Detected {n_videos} video entries.")
print("Extracting per-second averaged importance scores...")

saved = 0
for i in tqdm(range(n_videos)):
    try:
        vid_name = decode_str(video_refs[i][0])
        category = decode_str(category_refs[i][0])
        user_scores = np.array(f[user_refs[i][0]])  # (20, n_frames)

        # Average over 20 annotators
        avg_score = user_scores.mean(axis=0)

        n_frames = len(avg_score)
        n_seconds = n_frames // FPS
        if n_seconds == 0:
            continue

        avg_score = avg_score[: n_seconds * FPS]
        y = avg_score.reshape(n_seconds, FPS).mean(axis=1)
        X = np.zeros((n_seconds, 1))  # placeholder for multimodal features

        np.savez(Path(OUTPUT_DIR) / f"{vid_name}.npz", X=X, y=y, category=category)
        saved += 1
    except Exception as e:
        print(f"⚠️ Skipping {i}: {e}")

f.close()
print(f"\n✅ Done! {saved} label files saved in {OUTPUT_DIR}/")
