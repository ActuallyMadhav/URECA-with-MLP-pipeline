"""
create_dataset.py
─────────────────────────────────────────────────────────────────────────────
Reads raw webcam frames from ./data/<letter>/, extracts 21 MediaPipe hand
landmarks per frame, normalises them to a 42-feature vector, and saves
everything to data.pickle for training.

Frames where MediaPipe cannot detect a hand are skipped with a warning.
If detection rate is below 80% for any letter, consider re-collecting that
letter with better lighting or a plainer background.
─────────────────────────────────────────────────────────────────────────────
"""
import os
import pickle
import numpy as np
import cv2
import mediapipe as mp
from collections import Counter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR  = './data'
SAVE_PATH = './data.pickle'
# ─────────────────────────────────────────────

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3,
)

data, labels = [], []
skip_counts  = Counter()
total_counts = Counter()

letter_dirs = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
    and not d.startswith('.')
])

if not letter_dirs:
    print(f"No letter folders found in {DATA_DIR}. Run collect_data.py first.")
    exit(1)

print(f"Processing {len(letter_dirs)} letter(s): {letter_dirs}\n")

for letter in letter_dirs:
    letter_path = os.path.join(DATA_DIR, letter)
    img_files   = [f for f in os.listdir(letter_path) if f.endswith('.jpg')]

    if not img_files:
        print(f"  {letter} — no images found, skipping")
        continue

    for img_file in img_files:
        total_counts[letter] += 1

        img     = cv2.imread(os.path.join(letter_path, img_file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            skip_counts[letter] += 1
            continue

        # Use only first detected hand
        hand = results.multi_hand_landmarks[0]
        x_   = [lm.x for lm in hand.landmark]
        y_   = [lm.y for lm in hand.landmark]
        min_x, min_y = min(x_), min(y_)

        # Normalise landmarks relative to bounding box minimum
        # This makes features position-invariant
        features = []
        for lm in hand.landmark:
            features.append(lm.x - min_x)
            features.append(lm.y - min_y)

        data.append(features)
        labels.append(letter)

    kept     = total_counts[letter] - skip_counts[letter]
    det_rate = kept / total_counts[letter] * 100
    status   = "✓" if det_rate >= 80 else "⚠ low detection"
    print(f"  {letter} : {kept}/{total_counts[letter]} frames kept  "
          f"({det_rate:.0f}% detection)  {status}")

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\nTotal samples saved : {len(data)}")
print(f"Total skipped       : {sum(skip_counts.values())} "
      f"(no hand detected)")

low_detection = [l for l in letter_dirs
                 if total_counts[l] > 0 and
                 (total_counts[l] - skip_counts[l]) / total_counts[l] < 0.8]
if low_detection:
    print(f"\nWarning — low detection rate for: {low_detection}")
    print("Consider re-collecting these letters with better lighting "
          "or a plainer background.")

if len(data) == 0:
    print("\nError: no samples saved. Check that MediaPipe can see your hand "
          "in the collected images.")
    exit(1)

# ── Save ──────────────────────────────────────────────────────────────────
with open(SAVE_PATH, 'wb') as f:
    pickle.dump({'data': data, 'labels': np.array(labels)}, f)

print(f"\nSaved {SAVE_PATH}")
print("Run train_classifier_mlp.py next.")
