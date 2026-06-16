"""
compare.py
─────────────────────────────────────────────────────────────────────────────
Reads results_before.json and results_after.json produced by benchmark.py
and generates a side-by-side comparison plot + prints a summary table.

Usage:
    python3 compare.py
    python3 compare.py --before results_before.json --after results_after.json
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument('--before', type=str, default='results_before.json')
parser.add_argument('--after',  type=str, default='results_after.json')
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────────────────
for path in [args.before, args.after]:
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        print("Run:  python3 benchmark.py --label before")
        print("      python3 benchmark.py --label after")
        exit(1)

with open(args.before) as f: before = json.load(f)
with open(args.after)  as f: after  = json.load(f)

b_label = before['label'].upper()
a_label = after['label'].upper()

# ── Terminal summary table ─────────────────────────────────────────────────
def delta(b, a, higher_better=True):
    d = a - b
    pct = (d / b * 100) if b != 0 else 0
    sign = '+' if d >= 0 else ''
    arrow = ('▲' if d >= 0 else '▼') if higher_better else ('▼' if d >= 0 else '▲')
    good  = (d >= 0) if higher_better else (d <= 0)
    tag   = '✓' if good else '✗'
    return f"{sign}{d:.2f}  ({sign}{pct:.1f}%)  {arrow} {tag}"

print(f"""
╔══════════════════════════════════════════════════════════════╗
║          PERFORMANCE COMPARISON:  {b_label:<10} →  {a_label:<10}    ║
╠══════════════════════════════════════════════════════════════╣
║  Metric            {b_label:<12} {a_label:<12} Change              ║
╠══════════════════════════════════════════════════════════════╣
║  Avg FPS           {before['avg_fps']:<12.1f} {after['avg_fps']:<12.1f} {delta(before['avg_fps'],  after['avg_fps'],  True):<20} ║
║  FPS std dev       {before['std_fps']:<12.2f} {after['std_fps']:<12.2f} {delta(before['std_fps'],  after['std_fps'],  False):<20} ║
║  Min FPS           {before['min_fps']:<12.1f} {after['min_fps']:<12.1f} {delta(before['min_fps'],  after['min_fps'],  True):<20} ║
║  Pre-process (ms)  {before['avg_pre_ms']:<12.2f} {after['avg_pre_ms']:<12.2f} {delta(before['avg_pre_ms'], after['avg_pre_ms'], False):<20} ║
║  Inference (ms)    {before['avg_inf_ms']:<12.2f} {after['avg_inf_ms']:<12.2f} {delta(before['avg_inf_ms'], after['avg_inf_ms'], False):<20} ║
║  Drawing (ms)      {before['avg_draw_ms']:<12.2f} {after['avg_draw_ms']:<12.2f} {delta(before['avg_draw_ms'],after['avg_draw_ms'],False):<20} ║
║  Total latency(ms) {before['avg_total_ms']:<12.2f} {after['avg_total_ms']:<12.2f} {delta(before['avg_total_ms'],after['avg_total_ms'],False):<20} ║
╚══════════════════════════════════════════════════════════════╝
""")

# ── Plot ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle(f"Performance Comparison: {b_label} vs {a_label}", fontsize=14, fontweight='bold')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

BLUE   = '#3498db'
GREEN  = '#2ecc71'
RED    = '#e74c3c'
ORANGE = '#f39c12'

# ── Panel 1: FPS over time ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])   # full width
ax1.plot(before['time_stamps'], before['fps_timeline'],
         color=RED,   alpha=0.7, linewidth=1.2, label=f'{b_label}  (avg {before["avg_fps"]:.1f})')
ax1.plot(after['time_stamps'],  after['fps_timeline'],
         color=GREEN, alpha=0.7, linewidth=1.2, label=f'{a_label}  (avg {after["avg_fps"]:.1f})')
ax1.axhline(before['avg_fps'], color=RED,   linestyle='--', alpha=0.5, linewidth=1)
ax1.axhline(after['avg_fps'],  color=GREEN, linestyle='--', alpha=0.5, linewidth=1)
ax1.set_title("FPS Over Time")
ax1.set_xlabel("Elapsed (s)")
ax1.set_ylabel("FPS")
ax1.legend()
ax1.grid(True, alpha=0.25)

# ── Panel 2: Avg FPS bar ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
bars = ax2.bar([b_label, a_label],
               [before['avg_fps'], after['avg_fps']],
               color=[RED, GREEN], width=0.5)
for bar, val in zip(bars, [before['avg_fps'], after['avg_fps']]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
ax2.set_title("Average FPS")
ax2.set_ylabel("FPS")
ax2.set_ylim(0, max(before['avg_fps'], after['avg_fps']) * 1.25)
ax2.grid(axis='y', alpha=0.25)

# ── Panel 3: Latency breakdown ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
components = ['Pre-process', 'Inference', 'Drawing', 'Total']
b_vals = [before['avg_pre_ms'], before['avg_inf_ms'],
          before['avg_draw_ms'], before['avg_total_ms']]
a_vals = [after['avg_pre_ms'],  after['avg_inf_ms'],
          after['avg_draw_ms'],  after['avg_total_ms']]

x      = np.arange(len(components))
width  = 0.35
bars_b = ax3.bar(x - width/2, b_vals, width, label=b_label, color=RED,   alpha=0.8)
bars_a = ax3.bar(x + width/2, a_vals, width, label=a_label, color=GREEN, alpha=0.8)

for bar, val in list(zip(bars_b, b_vals)) + list(zip(bars_a, a_vals)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}', ha='center', va='bottom', fontsize=7)

ax3.set_title("Latency Breakdown (ms)")
ax3.set_ylabel("Milliseconds")
ax3.set_xticks(x)
ax3.set_xticklabels(components, fontsize=8)
ax3.legend()
ax3.grid(axis='y', alpha=0.25)

plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved → comparison.png")
