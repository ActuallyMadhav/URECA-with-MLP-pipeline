"""
collect_data.py
─────────────────────────────────────────────────────────────────────────────
Captures 100 webcam frames per ASL letter (24 letters, no J or Z).
Saves raw JPG frames to ./data/<letter>/ for processing by create_dataset.py.

Controls:
  SPACE — start collecting the current letter
  S     — skip the current letter (useful if already collected)
  Q     — quit early (progress is saved — re-running resumes safely)
─────────────────────────────────────────────────────────────────────────────
"""
import os
import cv2

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR     = './data'
DATASET_SIZE = 100
LETTERS      = list('ABCDEFGHIKLMNOPQRSTUVWXY')  # 24 letters — no J or Z
# ─────────────────────────────────────────────

os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open camera.")
    exit(1)

total   = len(LETTERS)
done_count = 0

for idx, letter in enumerate(LETTERS):
    letter_dir = os.path.join(DATA_DIR, letter)
    os.makedirs(letter_dir, exist_ok=True)

    # Resume safely — skip letters already fully collected
    existing = len([f for f in os.listdir(letter_dir) if f.endswith('.jpg')])
    if existing >= DATASET_SIZE:
        print(f"[{idx+1}/{total}]  {letter} — already have {existing} frames, skipping")
        done_count += 1
        continue

    print(f"\n[{idx+1}/{total}]  Get ready to sign: {letter}  "
          f"({DATASET_SIZE - existing} frames needed)")
    print("  SPACE = start  |  S = skip  |  Q = quit")

    # ── Wait for SPACE or S ───────────────────────────────────────────────
    skip = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Progress bar across all letters
        progress     = done_count / total
        bar_w        = frame.shape[1] - 40
        filled       = int(bar_w * progress)
        cv2.rectangle(frame, (20, frame.shape[0]-30),
                      (20 + bar_w, frame.shape[0]-10), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, frame.shape[0]-30),
                      (20 + filled, frame.shape[0]-10), (0, 200, 0), -1)

        cv2.putText(frame,
                    f"Sign: {letter}  [{idx+1}/{total}]  SPACE=start  S=skip  Q=quit",
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.imshow('Collect ASL Data', frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord(' '):
            break
        if key == ord('s'):
            skip = True
            break
        if key == ord('q'):
            print("\nQuitting early — progress saved.")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    if skip:
        print(f"  Skipped {letter}")
        continue

    # ── Collect frames ────────────────────────────────────────────────────
    counter = existing   # resume from where we left off
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            print("  Camera error — stopping.")
            break

        cv2.imwrite(os.path.join(letter_dir, f'{counter}.jpg'), frame)
        counter += 1

        # Overlay counter on frame
        display = frame.copy()
        cv2.putText(display,
                    f"{letter}  {counter}/{DATASET_SIZE}",
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.imshow('Collect ASL Data', display)
        cv2.waitKey(25)

    print(f"  Saved {counter} frames for {letter}")
    done_count += 1

cap.release()
cv2.destroyAllWindows()
print(f"\nCollection complete — {done_count}/{total} letters collected.")
print("Run create_dataset.py next.")
