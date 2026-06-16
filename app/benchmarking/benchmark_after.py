"""
benchmark_after.py
─────────────────────────────────────────────────────────────────────────────
Benchmarks your OPTIMISED pipeline for RUN_DURATION seconds and saves
results to results_after.json.

Run benchmark_before.py first, then this, then compare.py.

Usage:
    python3 benchmark_after.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
# ── Must be set before importing mediapipe / torch ────────────────────────
os.environ["OMP_NUM_THREADS"]       = "4"
os.environ["OPENBLAS_NUM_THREADS"]  = "4"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "0"

import pickle
import time
import collections
import json
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

RUN_DURATION = 30

# ══════════════════════════════════════════════
# MODEL  — JIT compiled
# ══════════════════════════════════════════════
class GestureMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.0):
        super().__init__()
        layers, prev = [], input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

checkpoint = pickle.load(open('../model_mlp.p', 'rb'))
cfg        = checkpoint['model_config']
model      = GestureMLP(cfg['input_size'], cfg['hidden_sizes'], cfg['num_classes'], dropout=0.0)
model.load_state_dict(checkpoint['model_state'])
model.eval()
model = torch.jit.trace(model, torch.zeros(1, cfg['input_size']))   # JIT
le    = checkpoint['label_encoder']
print(f"Model loaded  |  classes: {list(le.classes_)}  |  "
      f"test accuracy: {checkpoint['test_accuracy']*100:.1f}%")

# ══════════════════════════════════════════════
# MEDIAPIPE  — model_complexity=0
# ══════════════════════════════════════════════
mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    model_complexity        = 0,     # lighter model
    static_image_mode       = False,
    max_num_hands           = 1,
    min_detection_confidence= 0.5,
    min_tracking_confidence = 0.5,
)

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
# CAMERA  — capped at 640×480
# ══════════════════════════════════════════════
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open camera.")
    exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ══════════════════════════════════════════════
# FRAME-SKIP CONFIG
# ══════════════════════════════════════════════
MEDIAPIPE_EVERY_N = 3
frame_idx         = 0
last_landmarks    = None

# ══════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════
stats = {
    'latency_pre':  [],
    'latency_inf':  [],
    'latency_draw': [],
    'total_loop':   [],
    'fps_timeline': [],
    'time_stamps':  [],
}
frame_times = collections.deque(maxlen=30)
run_start   = time.perf_counter()

print(f"\nBenchmarking OPTIMISED pipeline for {RUN_DURATION}s...\n")

# ══════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════
while True:
    t0      = time.perf_counter()
    elapsed = t0 - run_start
    if elapsed >= RUN_DURATION:
        break

    ret, frame = cap.read()
    if not ret:
        break
    H, W, _ = frame.shape

    # Pre-processing — MediaPipe runs every N frames only
    t1 = time.perf_counter()
    if frame_idx % MEDIAPIPE_EVERY_N == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(frame_rgb)
        last_landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None
    stats['latency_pre'].append(time.perf_counter() - t1)
    frame_idx += 1

    if last_landmarks is not None:
        hand_landmarks = last_landmarks

        # Inference
        t2                = time.perf_counter()
        features          = extract_features(hand_landmarks)
        label, confidence = predict(features)
        stats['latency_inf'].append(time.perf_counter() - t2)

        # Drawing
        t3 = time.perf_counter()
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
        colour = (0, 220, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        text   = f"{label}  {confidence*100:.0f}%" if confidence >= CONFIDENCE_THRESHOLD else f"?  {confidence*100:.0f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(frame, text, (x1, max(y1 - 12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2, cv2.LINE_AA)
        stats['latency_draw'].append(time.perf_counter() - t3)
    else:
        stats['latency_inf'].append(0)
        stats['latency_draw'].append(0)

    loop_time = time.perf_counter() - t0
    frame_times.append(loop_time)
    fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
    stats['total_loop'].append(loop_time)
    stats['fps_timeline'].append(fps)
    stats['time_stamps'].append(elapsed)

    cv2.imshow('Benchmark OPTIMISED  [Q=quit]', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ══════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════
inf_vals  = [x for x in stats['latency_inf']  if x > 0]
draw_vals = [x for x in stats['latency_draw'] if x > 0]

summary = {
    'label':           'after',
    'avg_fps':         float(np.mean(stats['fps_timeline'])),
    'std_fps':         float(np.std(stats['fps_timeline'])),
    'min_fps':         float(np.min(stats['fps_timeline'])),
    'max_fps':         float(np.max(stats['fps_timeline'])),
    'avg_pre_ms':      float(np.mean(stats['latency_pre'])  * 1000),
    'avg_inf_ms':      float(np.mean(inf_vals)              * 1000) if inf_vals  else 0,
    'avg_draw_ms':     float(np.mean(draw_vals)             * 1000) if draw_vals else 0,
    'avg_total_ms':    float(np.mean(stats['total_loop'])   * 1000),
    'frames_captured': len(stats['fps_timeline']),
    'fps_timeline':    stats['fps_timeline'],
    'time_stamps':     stats['time_stamps'],
}

with open('results_after.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"""
── Results [OPTIMISED] ───────────────────────────
  Avg FPS        : {summary['avg_fps']:.1f}
  FPS std dev    : {summary['std_fps']:.2f}
  FPS range      : {summary['min_fps']:.1f} – {summary['max_fps']:.1f}
  Pre-process    : {summary['avg_pre_ms']:.2f} ms
  MLP Inference  : {summary['avg_inf_ms']:.2f} ms
  Drawing        : {summary['avg_draw_ms']:.2f} ms
  Total latency  : {summary['avg_total_ms']:.2f} ms
  Frames captured: {summary['frames_captured']}
──────────────────────────────────────────────────
Saved → results_after.json
Run compare.py next.
""")
