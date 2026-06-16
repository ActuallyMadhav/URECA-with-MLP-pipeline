"""
inference_classifier_mlp.py
─────────────────────────────────────────────────────────────────────────────
Real-time ASL classification pipeline (Odroid N2 optimised):

  Webcam frame
    → MediaPipe  (model_complexity=0, XNNPACK multi-threaded)
    → Normalise  (42 features, position-invariant)
    → MLP        (TorchScript JIT compiled)
    → Letter + confidence displayed on screen

Optimisations applied:
  - model_complexity=0          : lighter MediaPipe hand model
  - OMP_NUM_THREADS=4           : use 4 of 6 cores for XNNPACK
  - Frame skipping               : MediaPipe runs every 3rd frame
  - Reduced capture resolution   : 640×480
  - TorchScript JIT              : compiled MLP inference
  - torch.set_num_threads(1)     : avoids threading overhead on tiny model

Controls:  Q = quit
─────────────────────────────────────────────────────────────────────────────
"""

import os
# ── Set before importing mediapipe/torch ──────────────────────────────────
os.environ["OMP_NUM_THREADS"]        = "4"
os.environ["OPENBLAS_NUM_THREADS"]   = "4"
os.environ["MEDIAPIPE_DISABLE_GPU"]  = "0"

import pickle
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)   # tiny model — threading overhead not worth it


# ══════════════════════════════════════════════
# MODEL  (must match train_classifier_mlp.py)
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

# JIT compile for faster repeated inference
dummy  = torch.zeros(1, cfg['input_size'])
model  = torch.jit.trace(model, dummy)

le = checkpoint['label_encoder']
print(f"Model loaded  |  classes: {list(le.classes_)}  |  "
      f"test accuracy: {checkpoint['test_accuracy']*100:.1f}%")


# ══════════════════════════════════════════════
# MEDIAPIPE  (model_complexity=0 — faster)
# ══════════════════════════════════════════════
mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    model_complexity        = 0,     # lighter model — biggest single win
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
# CAMERA  (lower resolution = less data to move)
# ══════════════════════════════════════════════
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open camera.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting webcam — press Q to quit.")

# ══════════════════════════════════════════════
# FRAME-SKIP CONFIG
# ══════════════════════════════════════════════
MEDIAPIPE_EVERY_N = 3   # run MediaPipe 1-in-N frames; reuse landmarks otherwise
frame_idx         = 0
last_landmarks    = None


# ══════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════
while True:
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

    cv2.imshow('ASL Classifier  [Q = quit]', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
