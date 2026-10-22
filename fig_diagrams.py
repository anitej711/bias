"""Patent-style black & white drawings: Fig.1 system block diagram, Fig.2 method flowchart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import os
os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

def box(ax, x, y, w, h, text, ref=None, rounded=True, lw=1.4, fs=9.5):
    st = "round,pad=0.02,rounding_size=0.08" if rounded else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=st, fc="white", ec="black", lw=lw))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, wrap=True)
    if ref: ax.text(x - w / 2 - 0.12, y, ref, ha="right", va="center", fontsize=9.5, fontweight="bold")

def arrow(ax, x1, y1, x2, y2, text=None, side="right"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black", shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1 + x2) / 2 + (0.08 if side == "right" else -0.08), (y1 + y2) / 2, text, fontsize=8.5,
                ha="left" if side == "right" else "right", va="center", style="italic")

# ------------------------------------------------------------ Fig 1: system block diagram
fig, ax = plt.subplots(figsize=(9.6, 10.2))
ax.set_xlim(0, 12.4); ax.set_ylim(0, 13); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.35, 0.35), 11.7, 12.0, boxstyle="square,pad=0", fc="none", ec="black", lw=1.0, ls=(0, (6, 4))))
ax.text(0.55, 12.15, "100  Proxy-leakage audit and repair system", fontsize=10, fontweight="bold", va="top")
W, H = 6.4, 0.85; X = 5.0
mods = [
    (11.2, "Data ingestion & cleaning module\n(missing-value tokens, weights/IDs excluded)", "102"),
    (10.0, "Sensitive-attribute identification module\n(name + value-signature matching; user override)", "104"),
    (8.8, "Pairwise association engine\n(Cramér's V / correlation ratio η)", "106"),
    (7.6, "Multivariate leakage estimator\n(auxiliary predictor of S → cross-validated AUC)", "108"),
    (6.4, "Leakage attribution & proxy-ranking module\n(leave-one-variable-out ΔAUC, NMI with target)", "110"),
    (5.2, "Targeted repair module\n(utility-aware category merging / quantile repair)", "112"),
    (4.0, "Verification gate\n(accept if leakage ↓ and accuracy loss ≤ budget)", "114"),
    (2.8, "Explanation & report generator\n(per-proxy evidence, actions taken)", "116"),
    (1.5, "Output: repaired dataset + audit report\n(CSV / JSON / PDF)", "118"),
]
for y, t, r in mods: box(ax, X, y, W, H, t, r)
for (y1, _, _), (y2, _, _) in zip(mods, mods[1:]): arrow(ax, X, y1 - H / 2, X, y2 + H / 2)
# loop from gate back to repair
ax.annotate("", xy=(X + W / 2, 5.2), xytext=(X + W / 2, 4.0), arrowprops=dict(arrowstyle="-|>", lw=1.2, connectionstyle="arc,angleA=0,angleB=0,armA=40,armB=40,rad=0"))
ax.text(X + W / 2 + 0.75, 4.6, "next ranked\nproxy", fontsize=8.5, style="italic", va="center")
# side inputs
box(ax, 10.6, 10.0, 2.2, 0.75, "User config\n(S, Y, τ, budget)", None, fs=8.5)
ax.text(10.6, 10.0 + 0.5, "120", ha="center", fontsize=9.5, fontweight="bold")
arrow(ax, 9.5, 10.0, X + W / 2, 10.0)
box(ax, 10.6, 11.2, 2.2, 0.75, "Tabular dataset\n(CSV / Excel / DB)", None, fs=8.5)
ax.text(10.6, 11.2 + 0.5, "122", ha="center", fontsize=9.5, fontweight="bold")
arrow(ax, 9.5, 11.2, X + W / 2, 11.2)
ax.text(6.2, 0.05, "FIG. 1", ha="center", fontsize=12, fontweight="bold")
fig.savefig("figs/fig1_system.png", dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ------------------------------------------------------------ Fig 2: method flowchart
fig, ax = plt.subplots(figsize=(9.4, 11.6))
ax.set_xlim(0, 11.6); ax.set_ylim(-0.6, 15.4); ax.axis("off")
X = 5.0; W = 6.6; H = 0.72
def term(y, t):
    ax.add_patch(FancyBboxPatch((X - 1.3, y - 0.3), 2.6, 0.6, boxstyle="round,pad=0.02,rounding_size=0.3", fc="white", ec="black", lw=1.4))
    ax.text(X, y, t, ha="center", va="center", fontsize=10, fontweight="bold")
def diamond(y, t, ref):
    w, h = 5.0, 1.25
    ax.add_patch(Polygon([(X, y + h / 2), (X + w / 2, y), (X, y - h / 2), (X - w / 2, y)], closed=True, fc="white", ec="black", lw=1.4))
    ax.text(X, y, t, ha="center", va="center", fontsize=8.8)
    ax.text(X + w / 2 + 0.05, y + 0.35, ref, fontsize=9, fontweight="bold")
term(15.0, "START")
steps = [
    (14.0, "S202  Load dataset; treat '?'/blank as missing;\nexclude sampling weights and row IDs"),
    (12.95, "S204  Identify sensitive attribute(s) S and target Y"),
    (11.9, "S206  Compute pairwise association of each feature with S\n(Cramér's V or correlation ratio η)"),
    (10.85, "S208  Train auxiliary predictor of S on all non-sensitive\nfeatures → baseline leakage L₀ (cross-validated AUC)"),
    (9.8, "S210  For each feature j: Lⱼ = leakage without j;\nattribution ΔLⱼ = L₀ − Lⱼ; rank proxies by ΔLⱼ, then V"),
    (8.75, "S212  Take next ranked proxy j"),
    (7.7, "S214  Repair j: categorical → utility-aware merging;\nnumeric → rank-preserving quantile repair"),
]
for y, t in steps: box(ax, X, y, W, H, t, fs=8.6)
arrow(ax, X, 14.7, X, 14.0 + H / 2)
for (y1, _), (y2, _) in zip(steps, steps[1:]): arrow(ax, X, y1 - H / 2, X, y2 + H / 2)
diamond(6.45, "S216  New leakage < current\nAND accuracy loss ≤ budget?", "")
arrow(ax, X, 7.7 - H / 2, X, 6.45 + 0.625)
box(ax, 10.3, 6.45, 2.0, 0.62, "S218  Discard\nrepair", fs=8.6)
arrow(ax, X + 2.5, 6.45, 10.3 - 1.0, 6.45, "No", "right")
box(ax, X, 5.15, W, 0.62, "S220  Keep repair; update dataset and leakage", fs=8.6)
arrow(ax, X, 6.45 - 0.625, X, 5.15 + 0.31, "Yes")
diamond(3.85, "S222  Leakage ≤ τ  OR\nno proxies left?", "")
arrow(ax, X, 5.15 - 0.31, X, 3.85 + 0.625)
# No branch back to S212
ax.plot([X - 2.5, 0.8], [3.85, 3.85], color="black", lw=1.2); ax.plot([0.8, 0.8], [3.85, 8.75], color="black", lw=1.2)
ax.annotate("", xy=(X - W / 2, 8.75), xytext=(0.8, 8.75), arrowprops=dict(arrowstyle="-|>", lw=1.2))
ax.text(0.9, 6.2, "No", fontsize=8.5, style="italic")
# discard rejoins
ax.plot([10.3, 10.3], [6.45 - 0.31, 3.85], color="black", lw=1.2)
ax.annotate("", xy=(X + 2.5, 3.85), xytext=(10.3, 3.85), arrowprops=dict(arrowstyle="-|>", lw=1.2))
box(ax, X, 2.55, W, 0.72, "S224  Generate explanation report: evidence per proxy,\nrepairs applied, leakage & fairness before/after", fs=8.6)
arrow(ax, X, 3.85 - 0.625, X, 2.55 + 0.36, "Yes")
box(ax, X, 1.35, W, 0.62, "S226  Output repaired dataset + audit report", fs=8.6)
arrow(ax, X, 2.55 - 0.36, X, 1.35 + 0.31)
term(0.3, "END"); arrow(ax, X, 1.35 - 0.31, X, 0.6)
ax.text(5.8, -0.55, "FIG. 2", ha="center", fontsize=12, fontweight="bold")
fig.savefig("figs/fig2_flowchart.png", dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("ok")
