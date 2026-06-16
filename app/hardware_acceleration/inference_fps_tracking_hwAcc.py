"""
inference_fps_tracking.py
─────────────────────────────────────────────────────────────────────────────
Real-time ASL classification pipeline (Odroid N2 optimised) with FPS logging.
Runs for RUN_DURATION seconds then prints a summary.

Optimisations applied:
  - model_complexity=0          : lighter MediaPipe hand model
  - OMP_NUM_THREADS=4           : use 4 of 6 cores for XNNPACK
  - Frame skipping               : MediaPipe runs every 3rd frame
  - Reduced capture resolution   : 640×480
  - TorchScript JIT              : compiled MLP inference
  - torch.set_num_threads(1)     : avoids threading overhead on tiny model

Controls:  Q = quit early
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.environ["OMP_NUM_THREADS"]        = "4"
os.environ["OPENBLAS_NUM_THREADS"]   = "4"
os.environ["MEDIAPIPE_DISABLE_GPU"]  = "0"

import pickle
import time
import collections
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)


# ══════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════
class GestureMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.0):
        super().__init__()
        layers = []
        prev   = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════
checkpoint = pickle.load(open('app/model_mlp.p', 'rb'))
cfg        = checkpoint['model_config']

model = GestureMLP(
    input_size   = cfg['input_size'],
    hidden_sizes = cfg['hidden_sizes'],
    num_classes  = cfg['num_classes'],
    dropout      = 0.0,
)
model.load_state_dict(checkpoint['model_state'])
model.eval()

dummy = torch.zeros(1, cfg['input_size'])
model = torch.jit.trace(model, dummy)

le = checkpoint['label_encoder']
print(f"Model loaded  |  classes: {list(le.classes_)}  |  "
      f"test accuracy: {checkpoint['test_accuracy']*100:.1f}%")


# ══════════════════════════════════════════════
# MEDIAPIPE
# ══════════════════════════════════════════════
mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    model_complexity        = 0,
    static_image_mode       = False,
    max_num_hands           = 1,
    min_detection_confidence= 0.5,
    min_tracking_confidence = 0.5,
)


# ══════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.45

def extract_features(hand_landmarks):
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]
    min_x, min_y = min(x_), min(y_)
    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x - min_x)
        features.append(lm.y - min_y)
    return np.asarray(features, dtype=np.float32)


def predict(features):
    tensor = torch.tensor(features).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    idx        = probs.argmax(1).item()
    confidence = probs[0, idx].item()
    label      = le.inverse_transform([idx])[0]
    return label, confidence


# ══════════════════════════════════════════════
# FPS TRACKER & TIMER
# ══════════════════════════════════════════════
FPS_WINDOW       = 30
RUN_DURATION     = 30
FPS_LOG_INTERVAL = 5
frame_times      = collections.deque(maxlen=FPS_WINDOW)
fps_log          = []
last_log_time    = None


# ══════════════════════════════════════════════
# CAMERA
# ══════════════════════════════════════════════
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open camera.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"Starting webcam — running for {RUN_DURATION}s. Press Q to quit early.")

# ══════════════════════════════════════════════
# FRAME-SKIP CONFIG
# ══════════════════════════════════════════════
MEDIAPIPE_EVERY_N = 3
frame_idx         = 0
last_landmarks    = None

run_start     = time.perf_counter()
last_log_time = run_start


# ══════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════
while True:
    loop_start = time.perf_counter()
    elapsed    = loop_start - run_start

    if elapsed >= RUN_DURATION:
        print(f"\nTime's up! ({RUN_DURATION}s elapsed)")
        break

    ret, frame = cap.read()
    if not ret:
        print("Camera error — exiting.")
        break

    H, W, _ = frame.shape

    # ── MediaPipe (skipped on non-keyframes) ──────────────────────────────
    if frame_idx % MEDIAPIPE_EVERY_N == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            last_landmarks = results.multi_hand_landmarks[0]
        else:
            last_landmarks = None
    frame_idx += 1

    if last_landmarks is not None:
        hand_landmarks = last_landmarks

        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]
        x1 = max(int(min(x_) * W) - 20, 0)
        y1 = max(int(min(y_) * H) - 20, 0)
        x2 = min(int(max(x_) * W) + 20, W)
        y2 = min(int(max(y_) * H) + 20, H)

        features          = extract_features(hand_landmarks)
        label, confidence = predict(features)

        if confidence >= CONFIDENCE_THRESHOLD:
            colour = (0, 220, 0)
            text   = f"{label}  {confidence*100:.0f}%"
        else:
            colour = (0, 165, 255)
            text   = f"?  {confidence*100:.0f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(frame, text, (x1, max(y1 - 12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2, cv2.LINE_AA)

    # ── FPS ───────────────────────────────────────────────────────────────
    frame_times.append(time.perf_counter() - loop_start)
    fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

    if loop_start - last_log_time >= FPS_LOG_INTERVAL:
        fps_log.append((elapsed, fps))
        print(f"  t={elapsed:5.1f}s  →  FPS: {fps:.1f}")
        last_log_time = loop_start

    fps_text    = f"FPS: {fps:.1f}"
    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = 0.75
    thickness   = 2
    (tw, th), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
    tx = W - tw - 12
    ty = th + 12
    cv2.putText(frame, fps_text, (tx + 1, ty + 1),
                font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (tx, ty),
                font, font_scale, (0, 220, 0), thickness, cv2.LINE_AA)

    cv2.imshow('ASL Classifier  [Q = quit]', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nQuit early by user.")
        break

cap.release()
cv2.destroyAllWindows()

# ── Final FPS summary ──────────────────────────────────────────────────────
if fps_log:
    print("\n── FPS Summary ──────────────────────")
    for t, f in fps_log:
        print(f"  t={t:5.1f}s  →  {f:.1f} FPS")
    avg = sum(f for _, f in fps_log) / len(fps_log)
    print(f"  Average across snapshots: {avg:.1f} FPS")
    print("─────────────────────────────────────")
