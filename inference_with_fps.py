"""
inference_classifier_mlp.py
─────────────────────────────────────────────────────────────────────────────
Real-time ASL classification pipeline:

  Webcam frame
    → MediaPipe  (extracts 21 hand landmarks)
    → Normalise  (42 features, position-invariant)
    → MLP        (trained on your own webcam data)
    → Letter + confidence displayed on screen

Controls:  Q = quit
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
    bounding box minimum. Identical to create_dataset.py — this is what
    keeps training and inference distributions aligned.
    Returns float32 array of shape (42,).
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
    Returns (predicted_letter, confidence).
    """
    tensor = torch.tensor(features).unsqueeze(0)   # (1, 42)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    idx        = probs.argmax(1).item()
    confidence = probs[0, idx].item()
    label      = le.inverse_transform([idx])[0]
    return label, confidence


# ══════════════════════════════════════════════
# FPS TRACKER
# ══════════════════════════════════════════════
FPS_WINDOW        = 30          # rolling average over this many frames
FPS_PRINT_EVERY   = 15          # print to terminal every N frames
frame_times       = collections.deque(maxlen=FPS_WINDOW)
frame_count       = 0


# ══════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════
cap = cv2.VideoCapture(0)   # 0 - for integrated webcam. on linux run "ls /dev/video* and select appropriate number"
if not cap.isOpened():
    print("Error: could not open camera.")
    exit(1)

print("Starting webcam — press Q to quit.")

while True:
    loop_start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        print("Camera error — exiting.")
        break

    H, W, _   = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        # Bounding box
        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]
        x1 = max(int(min(x_) * W) - 20, 0)
        y1 = max(int(min(y_) * H) - 20, 0)
        x2 = min(int(max(x_) * W) + 20, W)
        y2 = min(int(max(y_) * H) + 20, H)

        # Classify
        features          = extract_features(hand_landmarks)
        label, confidence = predict(features)

        if confidence >= CONFIDENCE_THRESHOLD:
            colour = (0, 220, 0)                       # green — confident
            text   = f"{label}  {confidence*100:.0f}%"
        else:
            colour = (0, 165, 255)                     # orange — uncertain
            text   = f"?  {confidence*100:.0f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(frame, text, (x1, max(y1 - 12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2, cv2.LINE_AA)

    # ── FPS calculation ────────────────────────────────────────────────────
    frame_times.append(time.perf_counter() - loop_start)
    frame_count += 1
    fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

    # Terminal output (throttled so it doesn't flood the console)
    if frame_count % FPS_PRINT_EVERY == 0:
        print(f"FPS: {fps:.1f}", end="\r", flush=True)

    # Overlay — top-right corner
    fps_text    = f"FPS: {fps:.1f}"
    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = 0.75
    thickness   = 2
    (tw, th), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
    tx = W - tw - 12          # 12 px padding from right edge
    ty = th + 12              # 12 px padding from top edge
    # Dark drop-shadow for legibility over any background
    cv2.putText(frame, fps_text, (tx + 1, ty + 1),
                font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (tx, ty),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    # ──────────────────────────────────────────────────────────────────────

    cv2.imshow('ASL Classifier  [Q = quit]', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print()   # newline after the carriage-return FPS line