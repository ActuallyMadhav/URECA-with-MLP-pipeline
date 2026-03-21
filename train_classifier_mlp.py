"""
train_classifier_mlp.py
─────────────────────────────────────────────────────────────────────────────
Trains a lightweight MLP on 42 MediaPipe landmark features extracted from
real webcam frames. Because training data comes from the same camera and
preprocessing pipeline as inference, there is no domain mismatch.

Expected accuracy: 90–99% on held-out webcam test frames.
─────────────────────────────────────────────────────────────────────────────
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIG  (edit these freely)
# ─────────────────────────────────────────────
HIDDEN_SIZES = [256, 128, 64]   # neurons per hidden layer
DROPOUT      = 0.3              # regularisation — reduce if underfitting
EPOCHS       = 150
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 20               # early stopping patience
SAVE_PATH    = './model_mlp.p'
# ─────────────────────────────────────────────

device = torch.device('cpu')


# ══════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════
print("Loading data...")
data_dict = pickle.load(open('./data.pickle', 'rb'))
data      = np.asarray(data_dict['data'],   dtype=np.float32)
labels    = np.asarray(data_dict['labels'])

le          = LabelEncoder()
labels_enc  = le.fit_transform(labels).astype(np.int64)
num_classes = len(le.classes_)
n_features  = data.shape[1]

print(f"Total samples  : {len(data)}")
print(f"Features       : {n_features}  (21 landmarks × x,y = 42)")
print(f"Classes        : {list(le.classes_)}  ({num_classes} total)")
print(f"Samples/class  : min={np.bincount(labels_enc).min()}  "
      f"max={np.bincount(labels_enc).max()}")

expected = set('ABCDEFGHIKLMNOPQRSTUVWXY')
missing  = expected - set(le.classes_)
if missing:
    print(f"Warning: {sorted(missing)} have no collected data yet")


# ══════════════════════════════════════════════
# 2.  TRAIN / VAL / TEST SPLIT  (60 / 20 / 20)
# ══════════════════════════════════════════════
X_temp, X_test, y_temp, y_test = train_test_split(
    data, labels_enc, test_size=0.2, stratify=labels_enc, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,   test_size=0.25, stratify=y_temp,    random_state=42)

print(f"\nSplit →  train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")


def make_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val)
test_loader  = make_loader(X_test,  y_test)


# ══════════════════════════════════════════════
# 3.  MODEL
# ══════════════════════════════════════════════
class GestureMLP(nn.Module):
    """
    Lightweight MLP for real-time ASL gesture classification.

    Input  : 42 normalised landmark coordinates
    Hidden : [256 → 128 → 64]  each with BatchNorm + ReLU + Dropout
    Output : num_classes logits  (no activation — CrossEntropyLoss expects logits)

    BatchNorm stabilises training on small datasets.
    Dropout prevents overfitting when samples per class is low.
    Inference time on CPU: < 1ms — bottleneck is MediaPipe, not this model.
    """
    def __init__(self, input_size, hidden_sizes, num_classes, dropout):
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


model     = GestureMLP(n_features, HIDDEN_SIZES, num_classes, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5)

print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
print(model)


# ══════════════════════════════════════════════
# 4.  TRAINING LOOP
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING")
print("="*60)

history        = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_loss  = float('inf')
best_state     = None
patience_count = 0


def evaluate(loader):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits      = model(xb)
            total_loss += criterion(logits, yb).item() * len(xb)
            correct    += (logits.argmax(1) == yb).sum().item()
            n          += len(xb)
    return total_loss / n, correct / n


for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()

    train_loss, train_acc = evaluate(train_loader)
    val_loss,   val_acc   = evaluate(val_loader)
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train loss {train_loss:.4f}  acc {train_acc*100:.1f}%  |  "
              f"val loss {val_loss:.4f}  acc {val_acc*100:.1f}%")

    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
print("Restored best model weights.")


# ══════════════════════════════════════════════
# 5.  FINAL EVALUATION
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL EVALUATION ON TEST SET")
print("="*60)

model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_preds.extend(model(xb).argmax(1).numpy())
        all_true.extend(yb.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)

test_accuracy  = accuracy_score(all_true, all_preds)
test_precision = precision_score(all_true, all_preds, average='weighted', zero_division=0)
test_recall    = recall_score(all_true, all_preds, average='weighted',    zero_division=0)
test_f1        = f1_score(all_true, all_preds, average='weighted',        zero_division=0)

print(f"\nTest Set Performance:")
print(f"  Accuracy : {test_accuracy  * 100:.2f}%")
print(f"  Precision: {test_precision * 100:.2f}%")
print(f"  Recall   : {test_recall    * 100:.2f}%")
print(f"  F1-Score : {test_f1        * 100:.2f}%")

class_names   = [str(c) for c in le.classes_]
active_labels = sorted(set(all_true) | set(all_preds))
active_names  = [class_names[i] for i in active_labels]

print("\nDetailed Classification Report:")
print(classification_report(all_true, all_preds,
                            labels=active_labels,
                            target_names=active_names,
                            zero_division=0))


# ══════════════════════════════════════════════
# 6.  VISUALISATIONS
# ══════════════════════════════════════════════
print("Generating visualisations...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs_ran = range(1, len(history['train_loss']) + 1)
ax1.plot(epochs_ran, history['train_loss'], label='Train')
ax1.plot(epochs_ran, history['val_loss'],   label='Val')
ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
ax2.plot(epochs_ran, [a*100 for a in history['train_acc']], label='Train')
ax2.plot(epochs_ran, [a*100 for a in history['val_acc']],   label='Val')
ax2.set_title('Accuracy (%)'); ax2.set_xlabel('Epoch'); ax2.legend()
plt.tight_layout()
plt.savefig('./training_curves.png', dpi=150, bbox_inches='tight')
print("Saved training_curves.png")

cm = confusion_matrix(all_true, all_preds, labels=active_labels)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=active_names, yticklabels=active_names)
plt.title('Confusion Matrix — Webcam Test Set')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('./confusion_matrix_mlp.png', dpi=150, bbox_inches='tight')
print("Saved confusion_matrix_mlp.png")


# ══════════════════════════════════════════════
# 7.  SAVE MODEL
# ══════════════════════════════════════════════
print("\nSaving model...")
with open(SAVE_PATH, 'wb') as f:
    pickle.dump({
        'model_state':   model.state_dict(),
        'model_config':  {
            'input_size':   n_features,
            'hidden_sizes': HIDDEN_SIZES,
            'num_classes':  num_classes,
            'dropout':      DROPOUT,
        },
        'label_encoder': le,
        'test_accuracy': test_accuracy,
        'test_metrics':  {
            'precision': test_precision,
            'recall':    test_recall,
            'f1_score':  test_f1,
        },
    }, f)
print(f"Saved {SAVE_PATH}")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)