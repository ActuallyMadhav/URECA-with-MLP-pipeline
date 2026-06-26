"""
ablation_compare.py
─────────────────────────────────────────────────────────────────────────────
Reads benchmark_ablation_results.json and accuracy_ablation_results.json
and produces a combined ablation study plot suitable for a research paper.

Usage:
    python3 ablation_compare.py

Output:
    ablation_study.png
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load ───────────────────────────────────────────────────────────────────
for f in ['benchmark_ablation_results.json', 'accuracy_ablation_results.json']:
    if not os.path.exists(f):
        print(f"Missing: {f}")
        print("Run accuracy_ablation.py and benchmark_ablation.py first.")
        exit(1)

with open('benchmark_ablation_results.json') as f:
    bench = json.load(f)
with open('accuracy_ablation_results.json') as f:
    acc   = json.load(f)

labels     = [r['label']       for r in bench]
fps        = [r['avg_fps']     for r in bench]
pre_ms     = [r['avg_pre_ms']  for r in bench]
inf_ms     = [r['avg_inf_ms']  for r in bench]
draw_ms    = [r['avg_draw_ms'] for r in bench]
total_ms   = [r['avg_total_ms']for r in bench]
accuracy   = [r['accuracy']    for r in acc]

x = np.arange(len(labels))

BLUE    = '#3498db'
GREEN   = '#2ecc71'
RED     = '#e74c3c'
ORANGE  = '#f39c12'
PURPLE  = '#9b59b6'
GREY    = '#95a5a6'

# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 12))
fig.suptitle("Ablation Study: Impact of Each Optimisation on FPS and Accuracy",
             fontsize=13, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

# ── Panel 1 (top, full width): FPS per config ──────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
bars = ax1.bar(x, fps, color=[GREY, BLUE, GREEN, ORANGE, PURPLE, RED], width=0.6, zorder=3)
for bar, val in zip(bars, fps):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_title("Average FPS per Configuration (higher is better)", fontsize=10)
ax1.set_ylabel("FPS")
ax1.set_ylim(0, max(fps) * 1.2)
ax1.grid(axis='y', alpha=0.3, zorder=0)
# Annotate % gain vs baseline
for i in range(1, len(fps)):
    pct = (fps[i] - fps[0]) / fps[0] * 100
    ax1.annotate(f'+{pct:.0f}%\nvs baseline', xy=(x[i], fps[i]),
                 xytext=(x[i], fps[i] + max(fps)*0.06),
                 ha='center', fontsize=7, color='#27ae60')

# ── Panel 2 (middle left): Accuracy per config ────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
bars2 = ax2.bar(x, accuracy, color=[GREY, BLUE, GREEN, ORANGE, PURPLE, RED],
                width=0.6, zorder=3)
for bar, val in zip(bars2, accuracy):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
ax2.set_title("Classification Accuracy per Configuration\n(measured on raw image dataset)", fontsize=9)
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(max(0, min(accuracy) - 5), 101)
ax2.grid(axis='y', alpha=0.3, zorder=0)

# ── Panel 3 (middle right): Total latency per config ─────────────────────
ax3 = fig.add_subplot(gs[1, 1])
bars3 = ax3.bar(x, total_ms, color=[GREY, BLUE, GREEN, ORANGE, PURPLE, RED],
                width=0.6, zorder=3)
for bar, val in zip(bars3, total_ms):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
ax3.set_title("Total Latency per Configuration (ms)\n(lower is better)", fontsize=9)
ax3.set_ylabel("Milliseconds (ms)")
ax3.set_ylim(0, max(total_ms) * 1.2)
ax3.grid(axis='y', alpha=0.3, zorder=0)

# ── Panel 4 (bottom, full width): Stacked latency breakdown ───────────────
ax4 = fig.add_subplot(gs[2, :])
ax4.bar(x, pre_ms,  label='Pre-processing', color='#3498db', width=0.6, zorder=3)
ax4.bar(x, inf_ms,  label='MLP Inference',  color='#e74c3c', width=0.6,
        bottom=pre_ms, zorder=3)
ax4.bar(x, draw_ms, label='Drawing',        color='#2ecc71', width=0.6,
        bottom=[p+i for p,i in zip(pre_ms, inf_ms)], zorder=3)

# Value labels on each stack segment
for i in range(len(x)):
    if pre_ms[i]  > 2: ax4.text(x[i], pre_ms[i]/2,  f'{pre_ms[i]:.1f}',  ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    if inf_ms[i]  > 2: ax4.text(x[i], pre_ms[i] + inf_ms[i]/2, f'{inf_ms[i]:.1f}',  ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    if draw_ms[i] > 0.5: ax4.text(x[i], pre_ms[i] + inf_ms[i] + draw_ms[i]/2, f'{draw_ms[i]:.1f}', ha='center', va='center', fontsize=7, color='white', fontweight='bold')

ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=9)
ax4.set_title("Latency Breakdown per Stage per Configuration (ms)", fontsize=10)
ax4.set_ylabel("Milliseconds (ms)")
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(axis='y', alpha=0.3, zorder=0)

plt.savefig('ablation_study.png', dpi=150, bbox_inches='tight')
print("Saved → ablation_study.png")

# ── Terminal table ─────────────────────────────────────────────────────────
print(f"\n{'Config':<18} {'FPS':>7} {'Δ FPS':>8} {'Acc%':>7} {'Pre(ms)':>10} {'Inf(ms)':>9} {'Total(ms)':>11}")
print("─" * 76)
for i, label in enumerate(labels):
    d_fps = fps[i] - fps[0]
    sign  = '+' if d_fps >= 0 else ''
    print(f"{label:<18} {fps[i]:>7.1f} {sign+f'{d_fps:.1f}':>8} {accuracy[i]:>7.1f} "
          f"{pre_ms[i]:>10.2f} {inf_ms[i]:>9.2f} {total_ms[i]:>11.2f}")
print("─" * 76)
