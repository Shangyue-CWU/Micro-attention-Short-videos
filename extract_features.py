import cv2
import numpy as np
import librosa
import mediapipe as mp
from moviepy.editor import VideoFileClip
import os

# ==========================================================
# CONFIGURATION
# ==========================================================
video_path = "sample_video.mp4"
tmp_audio = "temp_audio.wav"

# ==========================================================
# 1. AUDIO FEATURE EXTRACTION (safe + optional)
# ==========================================================
audio_energy = []
try:
    clip = VideoFileClip(video_path)
    duration = clip.duration
    if clip.audio is not None:
        print("🎧 Extracting audio track...")
        clip.audio.write_audiofile(tmp_audio, verbose=False, logger=None)
        audio, sr = librosa.load(tmp_audio, sr=None)
        hop = sr  # 1-second hop
        for i in range(0, len(audio), hop):
            seg = audio[i:i + hop]
            rms = np.sqrt(np.mean(seg ** 2))
            audio_energy.append(rms)
        os.remove(tmp_audio)
        print(f"✅ Extracted audio energy for {len(audio_energy)} seconds.")
    else:
        print("⚠️ No audio track found — filling with zeros.")
        audio_energy = [0.0] * int(duration)
except Exception as e:
    print(f"⚠️ Audio extraction failed: {e}")
    audio_energy = [0.0] * 1

# ==========================================================
# 2. VISUAL MOTION (Optical Flow)
# ==========================================================
print("🎥 Computing optical flow features...")
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30  # fallback
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_est = n_frames / fps
duration = int(duration or duration_est)

prev_gray = None
flow_magnitudes = []

for sec in range(duration):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_magnitudes.append(np.mean(mag))
    prev_gray = gray
cap.release()
print(f"✅ Optical flow extracted for {len(flow_magnitudes)} seconds.")

# ==========================================================
# 3. FACIAL LANDMARK MOTION
# ==========================================================
print("😀 Extracting facial motion features...")
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
cap = cv2.VideoCapture(video_path)
face_motion = []
prev_landmarks = None

for sec in range(duration):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if res.multi_face_landmarks:
        lm = np.array([(p.x, p.y) for p in res.multi_face_landmarks[0].landmark])
        if prev_landmarks is not None:
            dist = np.linalg.norm(lm - prev_landmarks, axis=1)
            face_motion.append(np.mean(dist))
        prev_landmarks = lm
    else:
        face_motion.append(0.0)
cap.release()
face_mesh.close()
print(f"✅ Facial motion extracted for {len(face_motion)} seconds.")

# ==========================================================
# 4. ALIGN AND COMBINE MULTIMODAL FEATURES
# ==========================================================
print("🧠 Combining features...")
n = min(len(flow_magnitudes), len(audio_energy), len(face_motion))
if n == 0:
    print("❌ Error: No features extracted — check video file.")
    exit()

features = np.stack([
    flow_magnitudes[:n],
    audio_energy[:n],
    face_motion[:n]
], axis=1)

np.save("features.npy", features)
print(f"Feature matrix saved: features.npy (shape = {features.shape})")
print("Feature dimensions → [optical_flow, audio_energy, face_motion]")
