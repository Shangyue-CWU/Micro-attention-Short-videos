import os
import cv2
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
VIDEO_DIR = Path("matlab/tvsum50_videos/video")   # ✅ folder containing your .mp4 videos
LABEL_DIR = Path("processed_tvsum")               # ✅ folder containing per-video labels
OUT_DIR = Path("aligned_dataset")          # ✅ where to save aligned .npz
OUT_DIR.mkdir(exist_ok=True, parents=True)

FPS_LIMIT = 10    # limit FPS for faster processing

# ---------------------------------------------------------------------
# Command line arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")
parser.add_argument("--debug", action="store_true", help="Print debug info per video")
args = parser.parse_args()

# ---------------------------------------------------------------------
# Feature extraction (motion + audio only)
# ---------------------------------------------------------------------
def extract_features_simple(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = min(fps, FPS_LIMIT)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    n_seconds = int(duration)

    # --- Optical flow ---
    flows = []
    prev_gray = None
    for sec in range(n_seconds):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * fps)
        ret, frame = cap.read()
        if not ret:
            flows.append(0)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flows.append(np.mean(mag))
        else:
            flows.append(0)
        prev_gray = gray
    cap.release()

    # --- Audio RMS ---
    tmp_audio = str(video_path).replace(".mp4", "_temp.wav")
    os.system(f"ffmpeg -y -i {video_path} -ar 16000 -ac 1 -vn {tmp_audio} -hide_banner -loglevel error")
    try:
        y, sr = librosa.load(tmp_audio, sr=None)
        rms = librosa.feature.rms(y=y, frame_length=sr, hop_length=sr)
        rms = rms.flatten()[:n_seconds]
    except Exception:
        rms = np.zeros(n_seconds)
    if os.path.exists(tmp_audio):
        os.remove(tmp_audio)

    # --- Align lengths ---
    min_len = min(len(flows), len(rms))
    flows, rms = np.array(flows[:min_len]), np.array(rms[:min_len])
    features = np.stack([flows, rms], axis=1)
    return features


# ---------------------------------------------------------------------
# Main alignment loop
# ---------------------------------------------------------------------
video_files = sorted(VIDEO_DIR.glob("*.mp4"))
if args.limit:
    video_files = video_files[:args.limit]

for video_file in tqdm(video_files, desc=f"Aligning videos from {VIDEO_DIR}"):
    vid_id = video_file.stem
    label_path = LABEL_DIR / f"{vid_id}.npz"
    if not label_path.exists():
        if args.debug:
            print(f"⚠️ Missing label for {vid_id}")
        continue

    # Load labels
    data = np.load(label_path, allow_pickle=True)
    y = data["y"]
    category = data["category"]

    if args.debug:
        print(f"Processing {vid_id} ({category}) ...")

    # Extract features (motion + audio)
    X = extract_features_simple(video_file)

    # Align length
    min_len = min(len(X), len(y))
    X, y = X[:min_len], y[:min_len]

    # Save aligned dataset
    out_path = OUT_DIR / f"{vid_id}_aligned.npz"
    np.savez(out_path, X=X, y=y, category=category)
    if args.debug:
        print(f"Saved: {out_path.name}  shape={X.shape}")

print(f"\nDone! {len(list(OUT_DIR.glob('*.npz')))} videos aligned and saved in {OUT_DIR}/")
