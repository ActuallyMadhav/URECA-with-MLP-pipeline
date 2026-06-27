"""
accuracy_compare.py
─────────────────────────────────────────────────────────────────────────────
Reads accuracy_ablation_results.json and produces a clean before/after
accuracy comparison plot suitable for a research paper.

Compares:
  BEFORE  →  Baseline config  (model_complexity=1, no JIT, no frameskip)
  AFTER   →  +Frameskip config (all optimisations applied)

Usage:
    python3 accuracy_compare.py

Output:
    accuracy_comparison.png   — bar chart + per-class breakdown
    accuracy_comparison.txt   — summary table for copy-paste into paper
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load ───────────────────────────────────────────────────────────────────
if not os.path.exists('app/benchmarking/accuracy_ablation_results.json'):
    print("Missing: accuracy_ablation_results.json")
    print("Run accuracy_ablation.py first.")
    exit(1)

with open('app/benchmarking/accuracy_ablation_results.json') as f:
    data = json.load(f)

# First config = before, last config = after
before = data[0]   # Baseline
after  = data[-1]  # +Frameskip (all optimisations)

# ── Terminal summary ───────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════════╗
║         ACCURACY COMPARISON:  BEFORE  →  AFTER              ║
╠══════════════════════════════════════════════════════════════╣
║  Metric               BEFORE          AFTER                 ║
╠══════════════════════════════════════════════════════════════╣
║  Accuracy             {before['accuracy']:<15.2f} {after['accuracy']:<15.2f} ║
║  Detection Rate       {before['detect_rate']:<15.2f} {after['detect_rate']:<15.2f} ║
║  Correct / Total      {before['correct']}/{before['total']:<10} {after['correct']}/{after['total']:<10} ║
║  No Detection         {before['no_detect']:<15} {after['no_detect']:<15} ║
╚══════════════════════════════════════════════════════════════╝
""")

# Per-class comparison
letters = sorted(before['per_class_acc'].keys())
b_acc   = [before['per_class_acc'][l] * 100 for l in letters]
a_acc   = [after['per_class_acc'][l]  * 100 for l in letters]
delta   = [a - b for a, b in zip(a_acc, b_acc)]

print(f"{'Letter':<8} {'Before':>8} {'After':>8} {'Delta':>8}")
print("─" * 36)
for i, l in enumerate(letters):
    d = delta[i]
    sign = '+' if d >= 0 else ''
    flag = '  ✗' if d < 0 else ''
    print(f"  {l:<6} {b_acc[i]:>7.1f}%  {a_acc[i]:>7.1f}%  {sign}{d:>5.1f}%{flag}")
print("─" * 36)
print(f"  {'MEAN':<6} {np.mean(b_acc):>7.1f}%  {np.mean(a_acc):>7.1f}%  "
      f"{np.mean(delta):>+6.1f}%")

# ── Save summary text ──────────────────────────────────────────────────────
lines = [
    "ACCURACY COMPARISON: BEFORE vs AFTER OPTIMISATION",
    "=" * 50,
    f"{'Metric':<25} {'Before':>10} {'After':>10}",
    "-" * 50,
    f"{'Overall Accuracy (%)':<25} {before['accuracy']:>10.2f} {after['accuracy']:>10.2f}",
    f"{'Detection Rate (%)':<25} {before['detect_rate']:>10.2f} {after['detect_rate']:>10.2f}",
    f"{'Correct Frames':<25} {before['correct']:>10} {after['correct']:>10}",
    f"{'Total Frames':<25} {before['total']:>10} {after['total']:>10}",
    f"{'Undetected Frames':<25} {before['no_detect']:>10} {after['no_detect']:>10}",
    "",
    "PER-CLASS ACCURACY",
    "-" * 50,
    f"{'Letter':<10} {'Before':>10} {'After':>10} {'Delta':>10}",
    "-" * 50,
]
for i, l in enumerate(letters):
    d = delta[i]
    lines.append(f"  {l:<8} {b_acc[i]:>9.1f}%  {a_acc[i]:>9.1f}%  {d:>+9.1f}%")
lines += [
    "-" * 50,
    f"  {'MEAN':<8} {np.mean(b_acc):>9.1f}%  {np.mean(a_acc):>9.1f}%  {np.mean(delta):>+9.1f}%",
]
with open('accuracy_comparison.txt', 'w') as f:
    f.write('\n'.join(lines))

# ── Plot ───────────────────────────────────────────────────────────────────
RED   = '#e74c3c'
GREEN = '#2ecc71'
BLUE  = '#3498db'
GREY  = '#95a5a6'

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Accuracy Comparison: Before vs After Optimisation",
             fontsize=13, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1: Overall accuracy bar ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(['BEFORE', 'AFTER'], [before['accuracy'], after['accuracy']],
               color=[RED, GREEN], width=0.5)
for bar, val in zip(bars, [before['accuracy'], after['accuracy']]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.15,
             f'{val:.2f}%', ha='center', va='top', fontsize=11, fontweight='bold',
             color='white')
ax1.set_title("Overall Accuracy (%)", fontsize=10)
ax1.set_ylim(99, 100.1)
ax1.set_ylabel("Accuracy (%)")
ax1.grid(axis='y', alpha=0.3)

# ── Panel 2: Detection rate bar ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(['BEFORE', 'AFTER'], [before['detect_rate'], after['detect_rate']],
                color=[RED, GREEN], width=0.5)
for bar, val in zip(bars2, [before['detect_rate'], after['detect_rate']]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.15,
             f'{val:.2f}%', ha='center', va='top', fontsize=11, fontweight='bold',
             color='white')
ax2.set_title("Hand Detection Rate (%)", fontsize=10)
ax2.set_ylim(97, 100.1)
ax2.set_ylabel("Detection Rate (%)")
ax2.grid(axis='y', alpha=0.3)

# ── Panel 3: Frames summary ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
categories = ['Correct', 'No Detection']
b_vals = [before['correct'], before['no_detect']]
a_vals = [after['correct'],  after['no_detect']]
x = np.arange(len(categories))
w = 0.35
b3 = ax3.bar(x - w/2, b_vals, w, label='BEFORE', color=RED,   alpha=0.85)
a3 = ax3.bar(x + w/2, a_vals, w, label='AFTER',  color=GREEN, alpha=0.85)
for bar, val in list(zip(b3, b_vals)) + list(zip(a3, a_vals)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(val), ha='center', va='bottom', fontsize=9)
ax3.set_title("Frame Breakdown\n(out of 2400 total)", fontsize=10)
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# ── Panel 4 (bottom, full width): Per-class accuracy ─────────────────────
ax4 = fig.add_subplot(gs[1, :])
x2  = np.arange(len(letters))
w2  = 0.35
ax4.bar(x2 - w2/2, b_acc, w2, label='BEFORE', color=RED,   alpha=0.85, zorder=3)
ax4.bar(x2 + w2/2, a_acc, w2, label='AFTER',  color=GREEN, alpha=0.85, zorder=3)

# Highlight letters where accuracy changed
for i, l in enumerate(letters):
    if delta[i] != 0:
        ax4.annotate(f'{delta[i]:+.0f}%',
                     xy=(x2[i] + w2/2, a_acc[i]),
                     xytext=(x2[i] + w2/2, a_acc[i] - 3),
                     ha='center', fontsize=7, color='#27ae60' if delta[i] > 0 else RED,
                     fontweight='bold')

ax4.set_xticks(x2)
ax4.set_xticklabels(letters, fontsize=10)
ax4.set_title("Per-Class Accuracy: Before vs After Optimisation", fontsize=10)
ax4.set_ylabel("Accuracy (%)")
ax4.set_ylim(80, 102)
ax4.axhline(100, color='grey', linestyle='--', alpha=0.4, linewidth=0.8)
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3, zorder=0)

plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved → accuracy_comparison.png")
print("Saved → accuracy_comparison.txt")
