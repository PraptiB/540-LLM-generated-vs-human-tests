"""
COSC 540 - Group 8 - Prapti's Analysis Script
Generates all charts and tables for slides 10-14
Run: python analysis.py
Output: PNG files in ./output_charts/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
import os

os.makedirs("output_charts", exist_ok=True)

# ============================================================
# COLORS
# ============================================================
HUMAN = "#6366f1"
LLM = "#f97316"
GREEN = "#22c55e"
RED = "#ef4444"
BG = "#0f1117"
CARD = "#1a1d27"
TEXT = "#e2e4eb"
MUTED = "#8b8fa3"

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': CARD,
    'text.color': TEXT,
    'axes.labelcolor': TEXT,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'axes.edgecolor': '#2d3348',
    'grid.color': '#2d3348',
    'font.family': 'sans-serif',
    'font.size': 10,
})

# ============================================================
# DATA - BUGSINPY (21 comparisons)
# ============================================================
bugsinpy = [
    {"id":"ansible_1",     "h_a":3, "l_a":1,  "h_loc":4,  "l_loc":14, "h_ec":3,  "l_ec":1,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"simple",      "l_pat":"advanced"},
    {"id":"ansible_2",     "h_a":5, "l_a":6,  "h_loc":10, "l_loc":30, "h_ec":5,  "l_ec":4,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"simple",      "l_pat":"advanced"},
    {"id":"ansible_3",     "h_a":1, "l_a":7,  "h_loc":12, "l_loc":18, "h_ec":1,  "l_ec":7,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":0,"h_pat":"simple",      "l_pat":"parametrize"},
    {"id":"ansible_4",     "h_a":5, "l_a":1,  "h_loc":12, "l_loc":18, "h_ec":3,  "l_ec":1,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"mock",        "l_pat":"mock"},
    {"id":"ansible_5",     "h_a":2, "l_a":5,  "h_loc":10, "l_loc":22, "h_ec":1,  "l_ec":3,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"try/except",  "l_pat":"pytest.raises"},
    {"id":"pandas_1",      "h_a":5, "l_a":1,  "h_loc":10, "l_loc":5,  "h_ec":5,  "l_ec":1,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":0,"h_pat":"simple",      "l_pat":"simple"},
    {"id":"pandas_2",      "h_a":1, "l_a":5,  "h_loc":5,  "l_loc":18, "h_ec":1,  "l_ec":5,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"simple",      "l_pat":"parametrize"},
    {"id":"pandas_3",      "h_a":1, "l_a":4,  "h_loc":4,  "l_loc":14, "h_ec":1,  "l_ec":4,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":0,"h_pat":"simple",      "l_pat":"advanced"},
    {"id":"pandas_4",      "h_a":2, "l_a":18, "h_loc":3,  "l_loc":22, "h_ec":1,  "l_ec":9,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"simple",      "l_pat":"parametrize"},
    {"id":"pandas_6",      "h_a":3, "l_a":8,  "h_loc":12, "l_loc":24, "h_ec":3,  "l_ec":8,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":0,"h_pat":"simple",      "l_pat":"advanced"},
    {"id":"pandas_7",      "h_a":1, "l_a":1,  "h_loc":5,  "l_loc":12, "h_ec":1,  "l_ec":1,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"simple",      "l_pat":"advanced"},
    {"id":"pandas_8",      "h_a":3, "l_a":4,  "h_loc":10, "l_loc":28, "h_ec":2,  "l_ec":2,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"simple",      "l_pat":"mock"},
    {"id":"pandas_9",      "h_a":3, "l_a":12, "h_loc":6,  "l_loc":30, "h_ec":3,  "l_ec":8,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"simple",      "l_pat":"advanced"},
    {"id":"ytdl_1",        "h_a":5, "l_a":12, "h_loc":10, "l_loc":20, "h_ec":5,  "l_ec":10, "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"simple",      "l_pat":"unittest"},
    {"id":"ytdl_2",        "h_a":3, "l_a":3,  "h_loc":14, "l_loc":20, "h_ec":2,  "l_ec":3,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"simple",      "l_pat":"advanced"},
    {"id":"ytdl_3",        "h_a":4, "l_a":9,  "h_loc":8,  "l_loc":35, "h_ec":4,  "l_ec":9,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"simple",      "l_pat":"advanced"},
    {"id":"ytdl_5",        "h_a":4, "l_a":1,  "h_loc":8,  "l_loc":16, "h_ec":4,  "l_ec":1,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"simple",      "l_pat":"advanced"},
    {"id":"ytdl_6",        "h_a":2, "l_a":6,  "h_loc":10, "l_loc":30, "h_ec":2,  "l_ec":6,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"mock",        "l_pat":"parametrize+mock"},
    {"id":"ytdl_7",        "h_a":5, "l_a":14, "h_loc":10, "l_loc":24, "h_ec":5,  "l_ec":14, "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":0,"h_pat":"simple",      "l_pat":"parametrize"},
    {"id":"ytdl_9",        "h_a":1, "l_a":8,  "h_loc":8,  "l_loc":40, "h_ec":1,  "l_ec":8,  "h_fd":0,"l_fd":1,"h_doc":0,"l_doc":1,"h_pat":"simple",      "l_pat":"parametrize"},
    {"id":"ytdl_10",       "h_a":4, "l_a":1,  "h_loc":8,  "l_loc":4,  "h_ec":4,  "l_ec":1,  "h_fd":0,"l_fd":0,"h_doc":0,"l_doc":0,"h_pat":"simple",      "l_pat":"simple"},
]

ids = [b["id"] for b in bugsinpy]
h_a = [b["h_a"] for b in bugsinpy]
l_a = [b["l_a"] for b in bugsinpy]
h_loc = [b["h_loc"] for b in bugsinpy]
l_loc = [b["l_loc"] for b in bugsinpy]
h_ec = [b["h_ec"] for b in bugsinpy]
l_ec = [b["l_ec"] for b in bugsinpy]
h_fd = [b["h_fd"] for b in bugsinpy]
l_fd = [b["l_fd"] for b in bugsinpy]

n = len(bugsinpy)

# ============================================================
# CHART 1: Assertions per Bug (Bar Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(n)
w = 0.35
ax.bar(x - w/2, h_a, w, label='Human', color=HUMAN, edgecolor='none', zorder=3)
ax.bar(x + w/2, l_a, w, label='LLM', color=LLM, edgecolor='none', zorder=3)
ax.set_xlabel('Bug ID')
ax.set_ylabel('Number of Assertions')
ax.set_title('Assertions per Bug — Human vs LLM', fontsize=14, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("output_charts/01_assertions_per_bug.png", dpi=200)
plt.close()
print("✓ 01_assertions_per_bug.png")

# ============================================================
# CHART 2: Lines of Code per Bug (Bar Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - w/2, h_loc, w, label='Human', color=HUMAN, edgecolor='none', zorder=3)
ax.bar(x + w/2, l_loc, w, label='LLM', color=LLM, edgecolor='none', zorder=3)
ax.set_xlabel('Bug ID')
ax.set_ylabel('Lines of Code')
ax.set_title('Lines of Test Code — Human vs LLM', fontsize=14, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("output_charts/02_loc_per_bug.png", dpi=200)
plt.close()
print("✓ 02_loc_per_bug.png")

# ============================================================
# CHART 3: Edge Cases per Bug (Bar Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - w/2, h_ec, w, label='Human', color=HUMAN, edgecolor='none', zorder=3)
ax.bar(x + w/2, l_ec, w, label='LLM', color=LLM, edgecolor='none', zorder=3)
ax.set_xlabel('Bug ID')
ax.set_ylabel('Edge Cases Covered')
ax.set_title('Edge Cases Covered — Human vs LLM', fontsize=14, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("output_charts/03_edge_cases_per_bug.png", dpi=200)
plt.close()
print("✓ 03_edge_cases_per_bug.png")

# ============================================================
# CHART 4: Fault Detection (Heatmap-style)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 3))
for i in range(n):
    ax.add_patch(plt.Rectangle((i-0.4, 0.55), 0.38, 0.35,
        facecolor=GREEN if h_fd[i] else RED, alpha=0.8))
    ax.add_patch(plt.Rectangle((i+0.02, 0.55), 0.38, 0.35,
        facecolor=GREEN if l_fd[i] else RED, alpha=0.8))
    ax.text(i-0.21, 0.725, '✓' if h_fd[i] else '✗', ha='center', va='center',
        fontsize=10, color='white', fontweight='bold')
    ax.text(i+0.21, 0.725, '✓' if l_fd[i] else '✗', ha='center', va='center',
        fontsize=10, color='white', fontweight='bold')

ax.set_xlim(-0.6, n-0.4)
ax.set_ylim(0.4, 1.1)
ax.set_xticks(range(n))
ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=7)
ax.set_yticks([])
ax.set_title('Fault Detection — Human (left) vs LLM (right) per Bug', fontsize=14, fontweight='bold', pad=12)
h_patch = mpatches.Patch(color=HUMAN, label='Human')
l_patch = mpatches.Patch(color=LLM, label='LLM')
g_patch = mpatches.Patch(color=GREEN, label='Detected')
r_patch = mpatches.Patch(color=RED, label='Missed')
ax.legend(handles=[h_patch, l_patch, g_patch, r_patch], loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig("output_charts/04_fault_detection.png", dpi=200)
plt.close()
print("✓ 04_fault_detection.png")

# ============================================================
# CHART 5: Fault Detection Pie Charts
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
h_det = sum(h_fd)
l_det = sum(l_fd)

ax1.pie([h_det, n-h_det], labels=[f'Detected ({h_det})', f'Missed ({n-h_det})'],
    colors=[GREEN, RED], autopct='%1.0f%%', textprops={'color':TEXT, 'fontsize':11},
    startangle=90, wedgeprops={'edgecolor':BG, 'linewidth':2})
ax1.set_title(f'Human Fault Detection\n{h_det}/{n}', fontsize=13, fontweight='bold', color=HUMAN)

ax2.pie([l_det, n-l_det], labels=[f'Detected ({l_det})', f'Missed ({n-l_det})'],
    colors=[GREEN, RED], autopct='%1.0f%%', textprops={'color':TEXT, 'fontsize':11},
    startangle=90, wedgeprops={'edgecolor':BG, 'linewidth':2})
ax2.set_title(f'LLM Fault Detection\n{l_det}/{n}', fontsize=13, fontweight='bold', color=LLM)

plt.tight_layout()
plt.savefig("output_charts/05_fault_detection_pies.png", dpi=200)
plt.close()
print("✓ 05_fault_detection_pies.png")

# ============================================================
# CHART 6: Summary Comparison Bar
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
metrics = ['Avg\nAssertions', 'Avg\nLOC', 'Avg Edge\nCases', 'Fault\nDetection %', 'Docstring\n%']
h_vals = [
    np.mean(h_a),
    np.mean(h_loc),
    np.mean(h_ec),
    sum(h_fd)/n*100,
    sum(b["h_doc"] for b in bugsinpy)/n*100
]
l_vals = [
    np.mean(l_a),
    np.mean(l_loc),
    np.mean(l_ec),
    sum(l_fd)/n*100,
    sum(b["l_doc"] for b in bugsinpy)/n*100
]

x2 = np.arange(len(metrics))
ax.bar(x2 - w/2, h_vals, w, label='Human', color=HUMAN, edgecolor='none', zorder=3)
ax.bar(x2 + w/2, l_vals, w, label='LLM', color=LLM, edgecolor='none', zorder=3)

for i, (hv, lv) in enumerate(zip(h_vals, l_vals)):
    ax.text(i - w/2, hv + 0.5, f'{hv:.1f}', ha='center', va='bottom', fontsize=9, color=HUMAN, fontweight='bold')
    ax.text(i + w/2, lv + 0.5, f'{lv:.1f}', ha='center', va='bottom', fontsize=9, color=LLM, fontweight='bold')

ax.set_xticks(x2)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_title('Overall Summary — Human vs LLM (BugsInPy)', fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("output_charts/06_summary_comparison.png", dpi=200)
plt.close()
print("✓ 06_summary_comparison.png")

# ============================================================
# CHART 7: Radar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
categories = ['Assertions', 'LOC', 'Fault Detection', 'Edge Cases', 'Documentation']
N = len(categories)

# Normalize values to 0-10 scale for radar
h_radar = [np.mean(h_a)/2, np.mean(h_loc)/4, sum(h_fd)/n*10,
           np.mean(h_ec)/2, sum(b["h_doc"] for b in bugsinpy)/n*10]
l_radar = [np.mean(l_a)/2, np.mean(l_loc)/4, sum(l_fd)/n*10,
           np.mean(l_ec)/2, sum(b["l_doc"] for b in bugsinpy)/n*10]

angles = [i / float(N) * 2 * np.pi for i in range(N)]
angles += angles[:1]
h_radar += h_radar[:1]
l_radar += l_radar[:1]

ax.plot(angles, h_radar, 'o-', linewidth=2, label='Human', color=HUMAN)
ax.fill(angles, h_radar, alpha=0.15, color=HUMAN)
ax.plot(angles, l_radar, 'o-', linewidth=2, label='LLM', color=LLM)
ax.fill(angles, l_radar, alpha=0.15, color=LLM)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_title('Overall Quality Profile', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(color='#2d3348')
plt.tight_layout()
plt.savefig("output_charts/07_radar_chart.png", dpi=200)
plt.close()
print("✓ 07_radar_chart.png")

# ============================================================
# CHART 8: Testing Pattern Distribution
# ============================================================
h_pats = {}
l_pats = {}
for b in bugsinpy:
    h_pats[b["h_pat"]] = h_pats.get(b["h_pat"], 0) + 1
    l_pats[b["l_pat"]] = l_pats.get(b["l_pat"], 0) + 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Human patterns
h_labels = sorted(h_pats.keys(), key=lambda k: h_pats[k], reverse=True)
h_counts = [h_pats[k] for k in h_labels]
ax1.barh(h_labels, h_counts, color=HUMAN, edgecolor='none')
ax1.set_title('Human Testing Patterns', fontsize=13, fontweight='bold', color=HUMAN)
ax1.set_xlabel('Count')
for i, v in enumerate(h_counts):
    ax1.text(v + 0.2, i, str(v), va='center', fontsize=11, fontweight='bold', color=TEXT)

# LLM patterns
l_labels = sorted(l_pats.keys(), key=lambda k: l_pats[k], reverse=True)
l_counts = [l_pats[k] for k in l_labels]
ax2.barh(l_labels, l_counts, color=LLM, edgecolor='none')
ax2.set_title('LLM Testing Patterns', fontsize=13, fontweight='bold', color=LLM)
ax2.set_xlabel('Count')
for i, v in enumerate(l_counts):
    ax2.text(v + 0.2, i, str(v), va='center', fontsize=11, fontweight='bold', color=TEXT)

plt.tight_layout()
plt.savefig("output_charts/08_testing_patterns.png", dpi=200)
plt.close()
print("✓ 08_testing_patterns.png")

# ============================================================
# STATISTICAL TESTS
# ============================================================
print("\n" + "="*60)
print("STATISTICAL RESULTS")
print("="*60)

# Paired t-tests
t_a, p_a = stats.ttest_rel(l_a, h_a)
t_loc, p_loc = stats.ttest_rel(l_loc, h_loc)
t_ec, p_ec = stats.ttest_rel(l_ec, h_ec)

# Cohen's d
def cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    return np.mean(diff) / np.std(diff, ddof=1)

d_a = cohens_d(l_a, h_a)
d_loc = cohens_d(l_loc, h_loc)
d_ec = cohens_d(l_ec, h_ec)

def effect_label(d):
    d = abs(d)
    if d >= 0.8: return "Large"
    elif d >= 0.5: return "Medium"
    elif d >= 0.2: return "Small"
    else: return "Negligible"

print(f"\nAssertions:  t={t_a:.3f}, p={p_a:.4f}, Cohen's d={d_a:.2f} ({effect_label(d_a)})")
print(f"LOC:         t={t_loc:.3f}, p={p_loc:.4f}, Cohen's d={d_loc:.2f} ({effect_label(d_loc)})")
print(f"Edge Cases:  t={t_ec:.3f}, p={p_ec:.4f}, Cohen's d={d_ec:.2f} ({effect_label(d_ec)})")
print(f"Fault Det:   Human={sum(h_fd)}/{n} ({sum(h_fd)/n*100:.0f}%), LLM={sum(l_fd)}/{n} ({sum(l_fd)/n*100:.0f}%)")

print(f"\nMeans:")
print(f"  Assertions: Human={np.mean(h_a):.1f}, LLM={np.mean(l_a):.1f}")
print(f"  LOC:        Human={np.mean(h_loc):.1f}, LLM={np.mean(l_loc):.1f}")
print(f"  Edge Cases: Human={np.mean(h_ec):.1f}, LLM={np.mean(l_ec):.1f}")

# ============================================================
# CHART 9: Statistical Results Table
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

table_data = [
    ['Metric', 'Human Avg', 'LLM Avg', 't-statistic', 'p-value', "Cohen's d", 'Effect Size'],
    ['Assertions', f'{np.mean(h_a):.1f}', f'{np.mean(l_a):.1f}', f'{t_a:.3f}', f'{p_a:.4f}', f'{d_a:.2f}', effect_label(d_a)],
    ['Lines of Code', f'{np.mean(h_loc):.1f}', f'{np.mean(l_loc):.1f}', f'{t_loc:.3f}', f'{p_loc:.4f}', f'{d_loc:.2f}', effect_label(d_loc)],
    ['Edge Cases', f'{np.mean(h_ec):.1f}', f'{np.mean(l_ec):.1f}', f'{t_ec:.3f}', f'{p_ec:.4f}', f'{d_ec:.2f}', effect_label(d_ec)],
    ['Fault Detection', f'{sum(h_fd)/n*100:.0f}%', f'{sum(l_fd)/n*100:.0f}%', '—', 'p<0.001', '—', 'McNemar test'],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[i, j]
        if i == 0:
            cell.set_facecolor('#2d3348')
            cell.set_text_props(color=TEXT, fontweight='bold')
        else:
            cell.set_facecolor(CARD)
            cell.set_text_props(color=TEXT)
        cell.set_edgecolor('#2d3348')

ax.set_title('Statistical Analysis — Paired t-test & Cohen\'s d (n=21)', fontsize=14, fontweight='bold', pad=20, color=TEXT)
plt.tight_layout()
plt.savefig("output_charts/09_statistical_table.png", dpi=200)
plt.close()
print("✓ 09_statistical_table.png")

# ============================================================
# CHART 10: Metrics Definition Table
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

metrics_table = [
    ['Metric', 'What It Measures', 'How We Calculate It'],
    ['Assertion Count', 'Number of assert statements in test', 'Count all assert/assertEqual lines'],
    ['Lines of Code', 'Length and verbosity of test', 'Count total lines of test function'],
    ['Edge Cases', 'Different inputs/scenarios tested', 'Count unique test inputs'],
    ['Fault Detection', 'Does test catch the actual bug?', 'Yes/No — does test fail on buggy code?'],
    ['Documentation', 'Does test explain what it tests?', 'Has docstring? Yes/No'],
    ['Testing Patterns', 'Complexity of test approach', 'Simple assert vs parametrize/mock/etc.'],
    ['Line Coverage*', '% of source code lines executed', 'pytest --cov (pending from Rishi)'],
    ['Branch Coverage*', '% of decision paths taken', 'pytest --cov-branch (pending from Rishi)'],
]

table = ax.table(cellText=metrics_table, loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.7)

for i in range(len(metrics_table)):
    for j in range(len(metrics_table[0])):
        cell = table[i, j]
        if i == 0:
            cell.set_facecolor('#2d3348')
            cell.set_text_props(color=TEXT, fontweight='bold')
        elif i >= 7:
            cell.set_facecolor('#1a1a2e')
            cell.set_text_props(color=MUTED)
        else:
            cell.set_facecolor(CARD)
            cell.set_text_props(color=TEXT)
        cell.set_edgecolor('#2d3348')

ax.set_title('Evaluation Metrics (* = pending execution data)', fontsize=14, fontweight='bold', pad=20, color=TEXT)
plt.tight_layout()
plt.savefig("output_charts/10_metrics_table.png", dpi=200)
plt.close()
print("✓ 10_metrics_table.png")

# ============================================================
# DONE
# ============================================================
print("\n" + "="*60)
print(f"All charts saved to ./output_charts/")
print(f"Total: 10 PNG files ready for your presentation")
print("="*60)