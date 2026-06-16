"""
inference_performance_metrics1.py
─────────────────────────────────────────────────────────────────────────────
Benchmarks the ASL pipeline for RUN_DURATION seconds and saves a
performance plot. Odroid N2 optimised.

Optimisations applied:
  - model_complexity=0          : lighter MediaPipe hand model
  - OMP_NUM_THREADS=4           : use 4 of 6 cores for XNNPACK
  - Frame skipping               : MediaPipe runs every 3rd frame
  - Reduced capture resolution   : 640×480
  - TorchScript JIT              : compiled MLP inference
  - torch.set_num_threads(1)     : avoids threading overhead on tiny model
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.environ["OMP_NUM_THREADS"]        = "4"
os.environ["OPENBLAS_NUM_THREADS"]   = "4"
os.environ["MEDIAPIPE_DISABLE_GPU"]  = "0"

import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_num_threads(1)


# ══════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════
class GestureMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_size
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
cfg = checkpoint['model_config']
model = GestureMLP(cfg['input_size'], cfg['hidden_sizes'], cfg['num_classes'])
model.load_state_dict(checkpoint['model_state'])
model.eval()

dummy = torch.zeros(1, cfg['input_size'])
model = torch.jit.trace(model, dummy)

le = checkpoint['label_encoder']


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
)

CONFIDENCE_THRESHOLD = 0.45
RUN_DURATION         = 30

# ══════════════════════════════════════════════
# PERFORMANCE TRACKING
# ══════════════════════════════════════════════
stats = {
    'pre_processing': [],
    'inference':      [],
    'drawing':        [],
    'total_loop':     [],
    'fps_timeline':   [],
    'time_stamps':    [],
}

def extract_features(hand_landmarks):
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]
    min_x, min_y = min(x_), min(y_)
    features = [val for lm in hand_landmarks.landmark for val in (lm.x - min_x, lm.y - min_y)]
    return np.asarray(features, dtype=np.float32)

def predict(features):
    tensor = torch.tensor(features).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    idx = probs.argmax(1).item()
    return le.inverse_transform([idx])[0], probs[0, idx].item()


# ══════════════════════════════════════════════
# CAMERA
# ══════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ══════════════════════════════════════════════
# FRAME-SKIP CONFIG
# ══════════════════════════════════════════════
MEDIAPIPE_EVERY_N = 3
frame_idx         = 0
last_landmarks    = None

print(f"Benchmarking started for {RUN_DURATION}s...")
run_start = time.perf_counter()


# ══════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════
while True:
    t_start = time.perf_counter()
    elapsed = t_start - run_start
    if elapsed >= RUN_DURATION:
        break

    ret, frame = cap.read()
    if not ret:
        break
    H, W, _ = frame.shape

    # 1. Pre-processing latency (only on keyframes)
    t1 = time.perf_counter()
    if frame_idx % MEDIAPIPE_EVERY_N == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            last_landmarks = results.multi_hand_landmarks[0]
        else:
            last_landmarks = None
    stats['pre_processing'].append(time.perf_counter() - t1)
    frame_idx += 1

    # 2. Inference latency
    t2 = time.perf_counter()
    if last_landmarks is not None:
        hand_landmarks = last_landmarks
        features       = extract_features(hand_landmarks)
        label, confidence = predict(features)
        stats['inference'].append(time.perf_counter() - t2)

        # 3. Drawing latency
        t3 = time.perf_counter()
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"{label} {confidence:.2f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stats['drawing'].append(time.perf_counter() - t3)
    else:
        stats['inference'].append(0)
        stats['drawing'].append(0)

    loop_time   = time.perf_counter() - t_start
    current_fps = 1.0 / loop_time if loop_time > 0 else 0
    stats['total_loop'].append(loop_time)
    stats['fps_timeline'].append(current_fps)
    stats['time_stamps'].append(elapsed)

    cv2.imshow('Performance Benchmark', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# ══════════════════════════════════════════════
# ANALYSIS & PLOTTING
# ══════════════════════════════════════════════
print("\nGenerating Performance Report...")

avg_pre  = np.mean(stats['pre_processing']) * 1000
avg_inf  = np.mean([x for x in stats['inference'] if x > 0]) * 1000
avg_draw = np.mean([x for x in stats['drawing']   if x > 0]) * 1000
avg_fps  = np.mean(stats['fps_timeline'])
std_fps  = np.std(stats['fps_timeline'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(stats['time_stamps'], stats['fps_timeline'],
         color='tab:blue', alpha=0.6, label='Instantaneous FPS')
ax1.axhline(avg_fps, color='red', linestyle='--', label=f'Avg: {avg_fps:.1f}')
ax1.set_title(f"FPS Stability (Jitter σ: {std_fps:.2f})")
ax1.set_xlabel("Elapsed Time (seconds)")
ax1.set_ylabel("Frames Per Second")
ax1.legend()

categories = ['Pre-processing', 'Inference', 'Drawing']
values     = [avg_pre, avg_inf, avg_draw]
ax2.bar(categories, values, color=['#3498db', '#e74c3c', '#2ecc71'])
ax2.set_title("Latency Breakdown (Average ms per Frame)")
ax2.set_ylabel("Milliseconds (ms)")

plt.tight_layout()
plot_filename = "device_performance_results.png"
plt.savefig(plot_filename)
print(f"Results saved to: {os.path.abspath(plot_filename)}")

print(f"""
--- Summary Statistics ---
Avg FPS:       {avg_fps:.2f}
FPS Stability: {std_fps:.2f} (Lower is better)
Inference:     {avg_inf:.2f} ms
Total Latency: {np.mean(stats['total_loop'])*1000:.2f} ms
""")
