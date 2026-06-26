"""
benchmark_ablation.py
─────────────────────────────────────────────────────────────────────────────
Benchmarks each optimisation configuration on the webcam for RUN_DURATION
seconds and records FPS and per-stage latency.

Configurations tested (cumulative — each builds on the previous):
  0  baseline      Original code — no optimisations
  1  +resolution   Cap camera to 640x480
  2  +complexity   Also model_complexity=0
  3  +threads      Also OMP_NUM_THREADS=4
  4  +jit          Also TorchScript JIT
  5  +frameskip    Also frame skip N=3 (fully optimised)

Usage:
    python3 benchmark_ablation.py

Output:
    benchmark_ablation_results.json

Note: Each config runs for RUN_DURATION seconds. Total runtime:
      6 configs × 30s = ~3 minutes. Press Q to skip a config early.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import json
import time
import collections
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

RUN_DURATION = 30   # seconds per config

# ── Model ──────────────────────────────────────────────────────────────────
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

checkpoint = pickle.load(open('./model_mlp.p', 'rb'))
cfg        = checkpoint['model_config']
le         = checkpoint['label_encoder']

def load_model(jit=False):
    m = GestureMLP(cfg['input_size'], cfg['hidden_sizes'], cfg['num_classes'], dropout=0.0)
    m.load_state_dict(checkpoint['model_state'])
    m.eval()
    if jit:
        m = torch.jit.trace(m, torch.zeros(1, cfg['input_size']))
    return m

def extract_features(hand_landmarks):
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]
    mx, my = min(x_), min(y_)
    return np.asarray([v for lm in hand_landmarks.landmark
                       for v in (lm.x - mx, lm.y - my)], dtype=np.float32)

def predict(model, features):
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(features).unsqueeze(0)), dim=1)
    idx = probs.argmax(1).item()
    return le.inverse_transform([idx])[0], probs[0, idx].item()

# ── Configurations ─────────────────────────────────────────────────────────
configs = [
    {
        'label':            'Baseline',
        'note':             'No optimisations',
        'resolution':       False,
        'model_complexity': 1,
        'threads':          False,
        'jit':              False,
        'frameskip':        1,    # 1 = every frame (no skip)
    },
    {
        'label':            '+Resolution',
        'note':             'Cap camera to 640x480',
        'resolution':       True,
        'model_complexity': 1,
        'threads':          False,
        'jit':              False,
        'frameskip':        1,
    },
    {
        'label':            '+Complexity',
        'note':             'Also model_complexity=0',
        'resolution':       True,
        'model_complexity': 0,
        'threads':          False,
        'jit':              False,
        'frameskip':        1,
    },
    {
        'label':            '+Threads',
        'note':             'Also OMP_NUM_THREADS=4',
        'resolution':       True,
        'model_complexity': 0,
        'threads':          True,
        'jit':              False,
        'frameskip':        1,
    },
    {
        'label':            '+JIT',
        'note':             'Also TorchScript JIT',
        'resolution':       True,
        'model_complexity': 0,
        'threads':          True,
        'jit':              True,
        'frameskip':        1,
    },
    {
        'label':            '+Frameskip',
        'note':             'Also frame skip N=3 (fully optimised)',
        'resolution':       True,
        'model_complexity': 0,
        'threads':          True,
        'jit':              True,
        'frameskip':        3,
    },
]

# ── Run each config ────────────────────────────────────────────────────────
all_results = []

for ci, cfg_run in enumerate(configs):
    print(f"\n{'='*55}")
    print(f"  [{ci+1}/{len(configs)}]  {cfg_run['label']}  —  {cfg_run['note']}")
    print(f"  Running for {RUN_DURATION}s. Press Q to skip to next config.")
    print(f"{'='*55}")

    # Apply thread settings before MediaPipe init
    if cfg_run['threads']:
        os.environ["OMP_NUM_THREADS"]       = "4"
        os.environ["OPENBLAS_NUM_THREADS"]  = "4"
        torch.set_num_threads(1)
    else:
        os.environ["OMP_NUM_THREADS"]       = "1"
        os.environ["OPENBLAS_NUM_THREADS"]  = "1"
        torch.set_num_threads(4)

    # MediaPipe
    hands = mp.solutions.hands.Hands(
        static_image_mode       = False,
        max_num_hands           = 1,
        min_detection_confidence= 0.5,
        min_tracking_confidence = 0.5,
        model_complexity        = cfg_run['model_complexity'],
    )

    # Model
    model = load_model(jit=cfg_run['jit'])

    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: Could not open camera. Skipping.")
        hands.close()
        continue

    if cfg_run['resolution']:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Stats
    stats = {
        'latency_pre':  [],
        'latency_inf':  [],
        'latency_draw': [],
        'total_loop':   [],
        'fps_timeline': [],
        'time_stamps':  [],
    }
    frame_times    = collections.deque(maxlen=30)
    frame_idx      = 0
    last_landmarks = None
    N              = cfg_run['frameskip']
    run_start      = time.perf_counter()

    while True:
        t0      = time.perf_counter()
        elapsed = t0 - run_start
        if elapsed >= RUN_DURATION:
            break

        ret, frame = cap.read()
        if not ret:
            break
        H, W, _ = frame.shape

        # Pre-processing
        t1 = time.perf_counter()
        if frame_idx % N == 0:
            frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_mp = hands.process(frame_rgb)
            last_landmarks = (results_mp.multi_hand_landmarks[0]
                              if results_mp.multi_hand_landmarks else None)
        stats['latency_pre'].append(time.perf_counter() - t1)
        frame_idx += 1

        # Inference
        t2 = time.perf_counter()
        if last_landmarks is not None:
            feat, conf = predict(model, extract_features(last_landmarks))
            stats['latency_inf'].append(time.perf_counter() - t2)

            # Drawing
            t3 = time.perf_counter()
            x_ = [lm.x for lm in last_landmarks.landmark]
            y_ = [lm.y for lm in last_landmarks.landmark]
            x1 = max(int(min(x_) * W) - 20, 0)
            y1 = max(int(min(y_) * H) - 20, 0)
            x2 = min(int(max(x_) * W) + 20, W)
            y2 = min(int(max(y_) * H) + 20, H)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, f"{feat} {conf:.2f}", (x1, max(y1-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
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

        # FPS overlay
        fps_txt = f"{cfg_run['label']}  |  FPS: {fps:.1f}"
        cv2.putText(frame, fps_txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        # Progress bar
        bar_w  = W - 20
        filled = int(bar_w * (elapsed / RUN_DURATION))
        cv2.rectangle(frame, (10, H-20), (10+bar_w, H-10), (60,60,60), -1)
        cv2.rectangle(frame, (10, H-20), (10+filled, H-10), (0,200,0), -1)

        cv2.imshow(f'Ablation [{ci+1}/{len(configs)}]  Q=skip', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("  Skipped early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Summarise
    inf_vals  = [x for x in stats['latency_inf']  if x > 0]
    draw_vals = [x for x in stats['latency_draw'] if x > 0]

    summary = {
        'label':          cfg_run['label'],
        'note':           cfg_run['note'],
        'avg_fps':        float(np.mean(stats['fps_timeline'])),
        'std_fps':        float(np.std(stats['fps_timeline'])),
        'avg_pre_ms':     float(np.mean(stats['latency_pre'])  * 1000),
        'avg_inf_ms':     float(np.mean(inf_vals)  * 1000) if inf_vals  else 0.0,
        'avg_draw_ms':    float(np.mean(draw_vals) * 1000) if draw_vals else 0.0,
        'avg_total_ms':   float(np.mean(stats['total_loop']) * 1000),
        'fps_timeline':   stats['fps_timeline'],
        'time_stamps':    stats['time_stamps'],
    }

    print(f"  Avg FPS      : {summary['avg_fps']:.1f}")
    print(f"  Pre-process  : {summary['avg_pre_ms']:.2f} ms")
    print(f"  Inference    : {summary['avg_inf_ms']:.2f} ms")
    print(f"  Total        : {summary['avg_total_ms']:.2f} ms")

    all_results.append(summary)

# ── Save ───────────────────────────────────────────────────────────────────
with open('benchmark_ablation_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n── FPS Summary ──────────────────────────────────────────────")
print(f"{'Config':<18} {'Avg FPS':>10} {'Pre (ms)':>12} {'Total (ms)':>12}")
print("─" * 56)
for r in all_results:
    print(f"{r['label']:<18} {r['avg_fps']:>10.1f} {r['avg_pre_ms']:>12.2f} {r['avg_total_ms']:>12.2f}")
print("─────────────────────────────────────────────────────────────")
print("\nSaved → benchmark_ablation_results.json")
print("Run ablation_compare.py to generate the combined plot.")
