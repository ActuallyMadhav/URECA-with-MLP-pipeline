import pickle
import time
import collections
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

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
# LOAD MODEL & CONFIG
# ══════════════════════════════════════════════
checkpoint = pickle.load(open('./model_mlp.p', 'rb'))
cfg = checkpoint['model_config']
model = GestureMLP(cfg['input_size'], cfg['hidden_sizes'], cfg['num_classes'])
model.load_state_dict(checkpoint['model_state'])
model.eval()
le = checkpoint['label_encoder']

# ══════════════════════════════════════════════
# MEDIAPIPE & SETTINGS
# ══════════════════════════════════════════════
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

CONFIDENCE_THRESHOLD = 0.45
RUN_DURATION = 30  

# ══════════════════════════════════════════════
# PERFORMANCE TRACKING DATA
# ══════════════════════════════════════════════
stats = {
    'pre_processing': [],
    'inference': [],
    'drawing': [],
    'total_loop': [],
    'fps_timeline': []
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
# MAIN LOOP
# ══════════════════════════════════════════════
cap = cv2.VideoCapture(0)
print(f"Benchmarking started for {RUN_DURATION}s...")
run_start = time.perf_counter()

while True:
    t_start = time.perf_counter()
    elapsed = t_start - run_start
    if elapsed >= RUN_DURATION: break

    ret, frame = cap.read()
    if not ret: break
    H, W, _ = frame.shape

    # 1. Pre-processing Latency
    t1 = time.perf_counter()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    stats['pre_processing'].append(time.perf_counter() - t1)

    # 2. Inference Latency
    t2 = time.perf_counter()
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        features = extract_features(hand_landmarks)
        label, confidence = predict(features)
        stats['inference'].append(time.perf_counter() - t2)

        # 3. Drawing Latency
        t3 = time.perf_counter()
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"{label} {confidence:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stats['drawing'].append(time.perf_counter() - t3)
    else:
        stats['inference'].append(0)
        stats['drawing'].append(0)

    # Total Loop & FPS
    loop_time = time.perf_counter() - t_start
    stats['total_loop'].append(loop_time)
    current_fps = 1.0 / loop_time if loop_time > 0 else 0
    stats['fps_timeline'].append(current_fps)

    cv2.imshow('Performance Benchmark', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# ══════════════════════════════════════════════
# DATA ANALYSIS & PLOTTING
# ══════════════════════════════════════════════
print("\nGenerating Performance Report...")

# Calculate Averages (ms)
avg_pre = np.mean(stats['pre_processing']) * 1000
avg_inf = np.mean([x for x in stats['inference'] if x > 0]) * 1000
avg_draw = np.mean([x for x in stats['drawing'] if x > 0]) * 1000
avg_fps = np.mean(stats['fps_timeline'])
std_fps = np.std(stats['fps_timeline'])

# Create Plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: FPS Stability
ax1.plot(stats['fps_timeline'], color='tab:blue', alpha=0.6, label='Instantaneous FPS')
ax1.axhline(avg_fps, color='red', linestyle='--', label=f'Avg: {avg_fps:.1f}')
ax1.set_title(f"FPS Stability (Jitter σ: {std_fps:.2f})")
ax1.set_ylabel("Frames Per Second")
ax1.legend()

# Plot 2: Latency Breakdown (Stacked Bar)
categories = ['Pre-processing', 'Inference', 'Drawing']
values = [avg_pre, avg_inf, avg_draw]
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