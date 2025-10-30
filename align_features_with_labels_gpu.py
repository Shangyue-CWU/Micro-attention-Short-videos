import os
import cv2
import numpy as np
import librosa
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
VIDEO_DIR = Path("matlab/tvsum50_videos/video")
LABEL_DIR = Path("processed_tvsum")
OUT_DIR = Path("aligned_dataset_gpu")
OUT_DIR.mkdir(exist_ok=True, parents=True)

FPS_LIMIT = 10   # process up to 10 FPS for speed
AUDIO_SR = 16000 # audio sample rate

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

# ---------------------------------------------------------------------
# HELPER: extract audio safely
# ---------------------------------------------------------------------
def extract_audio_rms(video_path, n_seconds):
    """Extracts per-second RMS energy using ffmpeg + librosa."""
    tmp_audio = str(video_path).replace(".mp4", "_temp.wav")
    cmd = ["ffmpeg", "-y", "-i", str(video_path),
           "-ar", str(AUDIO_SR), "-ac", "1", "-vn", tmp_audio,
           "-hide_banner", "-loglevel", "error"]
    try:
        subprocess.run(cmd, timeout=30, check=True)
        y, sr = librosa.load(tmp_audio, sr=None)
        rms = librosa.feature.rms(y=y, frame_length=sr, hop_length=sr)
        rms = rms.flatten()[:n_seconds]
    except Exception:
        rms = np.zeros(n_seconds)
    finally:
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)
    return rms

# ---------------------------------------------------------------------
# FEATURE EXTRACTION: optical flow (GPU if available) + audio
# ---------------------------------------------------------------------
def extract_features_gpu(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    fps = min(fps, FPS_LIMIT)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / max(fps, 1)
    n_seconds = int(duration)

    # Detect CUDA
    has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if has_cuda:
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(
            numLevels=5, pyrScale=0.5, fastPyramids=False,
            winSize=15, numIters=3, polyN=5, polySigma=1.1, flags=0)
        gpu_prev = None
    else:
        prev_gray = None

    flows = []
    for sec in range(n_seconds):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * fps)
        ret, frame = cap.read()
        if not ret or frame is None:
            flows.append(0)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if has_cuda:
            gpu_gray = cv2.cuda_GpuMat()
            gpu_gray.upload(gray)
            if gpu_prev is not None:
                flow_gpu = gpu_flow.calc(gpu_prev, gpu_gray, None)
                fx, fy = cv2.cuda.split(flow_gpu)
                mag, _ = cv2.cuda.cartToPolar(fx, fy)
                mag_host = mag.download()
                flows.append(float(np.mean(mag_host)))
            else:
                flows.append(0)
            gpu_prev = gpu_gray
        else:
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
                flows.append(float(np.mean(mag)))
            else:
                flows.append(0)
            prev_gray = gray

    cap.release()

    # --- Audio RMS ---
    rms = extract_audio_rms(video_path, n_seconds)

    # --- Align lengths ---
    min_len = min(len(flows), len(rms))
    flows, rms = np.array(flows[:min_len]), np.array(rms[:min_len])
    features = np.stack([flows, rms], axis=1)
    return features

# ---------------------------------------------------------------------
# MAIN ALIGNMENT LOOP
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

    try:
        data = np.load(label_path, allow_pickle=True)
        y = data["y"]
        category = data["category"]
    except Exception as e:
        print(f"⚠️ Error loading labels for {vid_id}: {e}")
        continue

    if args.debug:
        print(f"🎬 Processing {vid_id} ({category}) ...")

    try:
        X = extract_features_gpu(video_file)
    except Exception as e:
        print(f"⚠️ Error extracting {vid_id}: {e}")
        continue

    min_len = min(len(X), len(y))
    X, y = X[:min_len], y[:min_len]

    out_path = OUT_DIR / f"{vid_id}_aligned.npz"
    np.savez(out_path, X=X, y=y, category=category)
    if args.debug:
        print(f"✅ Saved: {out_path.name}  shape={X.shape}")

print(f"\n✅ Done! {len(list(OUT_DIR.glob('*.npz')))} videos aligned and saved in {OUT_DIR}/")
