"""
inference_fps_tracking.py
─────────────────────────────────────────────────────────────────────────────
Updated with Performance Metrics & Matplotlib Plotting
─────────────────────────────────────────────────────────────────────────────
"""
import pickle
import time
import collections
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt  # Added for plotting
import os                         # Added for file paths

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
checkpoint = pickle.load(open('./model_mlp.p', 'rb'))
cfg        = checkpoint['model_config']

model = GestureMLP(
    input_size   = cfg['input_size'],
    hidden_sizes = cfg['hidden_sizes'],
    num_classes  = cfg['num_classes'],
    dropout      = 0.0,   # dropout OFF at inference
)
model.load_state_dict(checkpoint['model_state'])
model.eval()              # puts BatchNorm into eval mode (uses running stats)

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
    static_image_mode=False,   # tracking mode — faster on video streams
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ══════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.45   # show orange '?' below this

def extract_features(hand_landmarks):
    """
    Normalise all 21 landmark (x, y) coordinates relative to the hand's
    bounding box minimum.
    """
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]
    min_x, min_y = min(x_), min(y_)
    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x - min_x)
        features.append(lm.y - min_y)
    return np.asarray(features, dtype=np.float32)


def predict(features):
    """
    Run MLP inference on a 42-feature landmark vector.
    """
    tensor = torch.tensor(features).unsqueeze(0)   # (1, 42)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    idx        = probs.argmax(1).item()
    confidence = probs[0, idx].item()
    label      = le.inverse_transform([idx])[0]
    return label, confidence


# ══════════════════════════════════════════════
# FPS TRACKER & TIMER (Enhanced)
# ══════════════════════════════════════════════
FPS_WINDOW        = 30          
RUN_DURATION      = 30          
FPS_LOG_INTERVAL  = 5           
frame_times       = collections.deque(maxlen=FPS_WINDOW)
fps_log           = []          
last_log_time     = None        

# New metric containers for device comparison
metrics = {
    'time_stamps': [],
    'fps_values': [],
    'latency_pre': [],   # MediaPipe + Image conversion
    'latency_inf': [],   # PyTorch Model Prediction
    'latency_draw': []   # OpenCV Drawing
}


# ══════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════
cap = cv2.VideoCapture(0)   
if not cap.isOpened():
    print("Error: could not open camera.")
    exit(1)

print(f"Starting webcam — running for {RUN_DURATION}s. Press Q to quit early.")

run_start     = time.perf_counter()
last_log_time = run_start

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

    H, W, _   = frame.shape

    # --- METRIC: Pre-processing Latency ---
    t_pre = time.perf_counter()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)
    metrics['latency_pre'].append(time.perf_counter() - t_pre)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # --- METRIC: Inference Latency ---
        t_inf = time.perf_counter()
        features          = extract_features(hand_landmarks)
        label, confidence = predict(features)
        metrics['latency_inf'].append(time.perf_counter() - t_inf)

        # --- METRIC: Drawing Latency ---
        t_draw = time.perf_counter()
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

        if confidence >= CONFIDENCE_THRESHOLD:
            colour = (0, 220, 0)                       
            text   = f"{label}  {confidence*100:.0f}%"
        else:
            colour = (0, 165, 255)                     
            text   = f"?  {confidence*100:.0f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(frame, text, (x1, max(y1 - 12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2, cv2.LINE_AA)
        metrics['latency_draw'].append(time.perf_counter() - t_draw)

    # ── FPS calculation ────────────────────────────────────────────────────
    frame_times.append(time.perf_counter() - loop_start)
    fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
    
    # Store for plotting
    metrics['time_stamps'].append(elapsed)
    metrics['fps_values'].append(fps)

    if loop_start - last_log_time >= FPS_LOG_INTERVAL:
        snapshot = (elapsed, fps)
        fps_log.append(snapshot)
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

# ── Final Summaries & Plotting ─────────────────────────────────────────────
if fps_log:
    print("\n── FPS Summary ──────────────────────")
    for t, f in fps_log:
        print(f"  t={t:5.1f}s  →  {f:.1f} FPS")
    avg = sum(f for _, f in fps_log) / len(fps_log)
    print(f"  Average across snapshots: {avg:.1f} FPS")
    print("─────────────────────────────────────")

# Generate Comparison Plot
print("\nSaving performance plots...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 1. FPS Over Time (Stability Check)
ax1.plot(metrics['time_stamps'], metrics['fps_values'], color='green', label='Live FPS')
ax1.axhline(y=np.mean(metrics['fps_values']), color='red', linestyle='--', label='Mean FPS')
ax1.set_title("Device FPS Stability")
ax1.set_xlabel("Seconds")
ax1.set_ylabel("FPS")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Latency Distribution
avg_pre = np.mean(metrics['latency_pre']) * 1000
avg_inf = np.mean(metrics['latency_inf']) * 1000 if metrics['latency_inf'] else 0
avg_draw = np.mean(metrics['latency_draw']) * 1000 if metrics['latency_draw'] else 0

labels = ['Preprocessing', 'MLP Inference', 'OpenCV Drawing']
values = [avg_pre, avg_inf, avg_draw]
ax2.bar(labels, values, color=['#3498db', '#e74c3c', '#2ecc71'])
ax2.set_title("Average Latency per Component (Lower is Better)")
ax2.set_ylabel("Milliseconds (ms)")

plt.tight_layout()
plt.savefig('performance_comparison.png')
print(f"Done! Plot saved as 'performance_comparison.png' in {os.getcwd()}")