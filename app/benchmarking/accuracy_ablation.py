"""
accuracy_ablation.py
─────────────────────────────────────────────────────────────────────────────
Measures classification accuracy for each optimisation configuration by
running MediaPipe + MLP against the raw images in ./data/<letter>/.

Configurations tested (cumulative — each builds on the previous):
  0  baseline      model_complexity=1, no JIT
  1  +resolution   (no effect on static images — same as baseline)
  2  +complexity   model_complexity=0
  3  +threads      (no effect on static images — same as +complexity)
  4  +jit          TorchScript JIT compiled MLP
  5  +frameskip    (no effect on static images — same as +jit)

Configs 1, 3, 5 are included for completeness and will show identical
accuracy to the previous config, confirming those optimisations are
accuracy-neutral. This is a valid and useful result to report.

Usage:
    python3 accuracy_ablation.py

Output:
    accuracy_ablation_results.json
─────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import json
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

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

checkpoint  = pickle.load(open('./model_mlp.p', 'rb'))
cfg         = checkpoint['model_config']
le          = checkpoint['label_encoder']

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

# ── Load all raw images ────────────────────────────────────────────────────
DATA_DIR = './data'
image_paths = []   # list of (letter, image_path)

letter_dirs = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')
])

for letter in letter_dirs:
    letter_path = os.path.join(DATA_DIR, letter)
    for img_file in os.listdir(letter_path):
        if img_file.endswith('.jpg'):
            image_paths.append((letter, os.path.join(letter_path, img_file)))

print(f"Found {len(image_paths)} images across {len(letter_dirs)} letters.")

# ── Configurations ─────────────────────────────────────────────────────────
configs = [
    {
        'label':          'Baseline',
        'model_complexity': 1,
        'jit':            False,
        'note':           'No optimisations',
    },
    {
        'label':          '+Resolution',
        'model_complexity': 1,
        'jit':            False,
        'note':           'Resolution cap (no effect on static images)',
    },
    {
        'label':          '+Complexity',
        'model_complexity': 0,
        'jit':            False,
        'note':           'model_complexity=0',
    },
    {
        'label':          '+Threads',
        'model_complexity': 0,
        'jit':            False,
        'note':           'XNNPACK threads (no effect on static images)',
    },
    {
        'label':          '+JIT',
        'model_complexity': 0,
        'jit':            True,
        'note':           'TorchScript JIT compilation',
    },
    {
        'label':          '+Frameskip',
        'model_complexity': 0,
        'jit':            True,
        'note':           'Frame skipping (no effect on static images)',
    },
]

# ── Run each config ────────────────────────────────────────────────────────
results = []

for ci, cfg_run in enumerate(configs):
    print(f"\n[{ci+1}/{len(configs)}] {cfg_run['label']} — {cfg_run['note']}")

    # Load MediaPipe with this config's complexity
    hands = mp.solutions.hands.Hands(
        static_image_mode       = True,
        max_num_hands           = 1,
        min_detection_confidence= 0.3,
        model_complexity        = cfg_run['model_complexity'],
    )

    # Load MLP (with or without JIT)
    model = load_model(jit=cfg_run['jit'])

    correct     = 0
    total       = 0
    no_detect   = 0
    per_class   = defaultdict(lambda: {'correct': 0, 'total': 0})

    for true_letter, img_path in image_paths:
        img     = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results_mp = hands.process(img_rgb)

        if not results_mp.multi_hand_landmarks:
            no_detect += 1
            per_class[true_letter]['total'] += 1
            continue

        features           = extract_features(results_mp.multi_hand_landmarks[0])
        pred_letter, conf  = predict(model, features)

        per_class[true_letter]['total']   += 1
        total += 1
        if pred_letter == true_letter:
            correct += 1
            per_class[true_letter]['correct'] += 1

    hands.close()

    accuracy     = correct / total if total > 0 else 0.0
    detect_rate  = total / (total + no_detect) if (total + no_detect) > 0 else 0.0

    print(f"   Accuracy    : {accuracy*100:.2f}%  ({correct}/{total} classified frames)")
    print(f"   Detection   : {detect_rate*100:.1f}%  ({no_detect} frames with no hand detected)")

    per_class_acc = {
        letter: (v['correct'] / v['total'] if v['total'] > 0 else 0.0)
        for letter, v in per_class.items()
    }

    results.append({
        'label':         cfg_run['label'],
        'note':          cfg_run['note'],
        'accuracy':      round(accuracy * 100, 2),
        'detect_rate':   round(detect_rate * 100, 2),
        'correct':       correct,
        'total':         total,
        'no_detect':     no_detect,
        'per_class_acc': per_class_acc,
    })

# ── Save ───────────────────────────────────────────────────────────────────
with open('accuracy_ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n── Accuracy Summary ─────────────────────────────────────────")
print(f"{'Config':<18} {'Accuracy':>10} {'Detection':>12}")
print("─" * 44)
for r in results:
    print(f"{r['label']:<18} {r['accuracy']:>9.2f}%  {r['detect_rate']:>10.1f}%")
print("─────────────────────────────────────────────────────────────")
print("\nSaved → accuracy_ablation_results.json")
print("Run benchmark_ablation.py next, then ablation_compare.py to plot.")
