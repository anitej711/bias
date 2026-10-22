import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, ".")
from leakaudit import association

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open("results.json"))["settings"]
get = lambda d, s: next(x for x in R if x["dataset"] == d and x["sensitive"] == s)

# ---------------- FIG. 3 association heatmap (Adult)
df = pd.read_csv("adult.csv", na_values="?", skipinitialspace=True).dropna().drop(columns=["fnlwgt", "educational-num", "income"])
feats = [c for c in df.columns if c not in ("gender", "race")]
M = np.array([[association(df[f], df[s].astype(str)) for f in feats] for s in ["gender", "race"]])
order = np.argsort(-M.max(0)); feats = [feats[i] for i in order]; M = M[:, order]
fig, ax = plt.subplots(figsize=(9.5, 2.6))
im = ax.imshow(M, cmap="Greys", vmin=0, vmax=0.7, aspect="auto")
ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=35, ha="right")
ax.set_yticks([0, 1]); ax.set_yticklabels(["gender (S)", "race (S)"])
for i in range(2):
    for j in range(len(feats)):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9, color="white" if M[i, j] > 0.38 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("Cramér's V / η", fontsize=9)
ax.set_title("Pairwise association of each feature with the sensitive attribute (Adult)", fontsize=10.5, loc="left")
for s in ax.spines.values(): s.set_visible(False)
fig.text(0.5, -0.42, "FIG. 3", ha="center", fontsize=12, fontweight="bold")
fig.savefig("figs/fig3_heatmap.png", dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------- FIG. 4 leakage attribution (Adult-gender)
st = get("Adult", "gender"); rk = pd.DataFrame(st["ranking"]).sort_values("leak_drop")
fig, ax = plt.subplots(figsize=(8.5, 4.0))
yy = np.arange(len(rk))
ax.barh(yy, rk.leak_drop, color="0.3", height=0.6)
ax.set_yticks(yy); ax.set_yticklabels([f"{f}  (V = {v:.2f})" for f, v in zip(rk.feature, rk.assoc_S)])
ax.set_xlabel("Leakage attribution: drop in gender-leakage AUC when the feature is removed (ΔAUC)")
for yi, v in zip(yy, rk.leak_drop): ax.text(v + 0.0008, yi, f"{v:.3f}", va="center", fontsize=8.5)
ax.set_xlim(0, rk.leak_drop.max() * 1.15)
ax.set_title(f"Adult, S = gender. Baseline leakage AUC = {st['base_leak']:.3f}. V = pairwise Cramér's V / η", fontsize=10, loc="left")
fig.text(0.5, -0.06, "FIG. 4", ha="center", fontsize=12, fontweight="bold")
fig.savefig("figs/fig4_attribution.png", dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------- FIG. 5 repair trajectory (Adult-gender and Adult-race)
fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex="col", gridspec_kw=dict(height_ratios=[1.6, 1]))
for col, s_ in enumerate(["gender", "race"]):
    tr = pd.DataFrame(get("Adult", s_)["trail"]); tr = tr[tr.kept]
    x = np.arange(len(tr)); a1, a2 = axes[0, col], axes[1, col]
    a1.plot(x, tr.leakage, "-o", color="black")
    a1.axhline(0.65, ls=":", color="0.4"); a1.text(-0.1, 0.635, "target τ = 0.65", fontsize=8.5, color="0.3", va="top")
    a1.axhline(0.5, ls="--", color="0.7", lw=0.8); a1.text(-0.1, 0.51, "no leakage = 0.5", fontsize=8, color="0.45")
    for xi, l in zip(x, tr.leakage): a1.text(xi + 0.08, l + 0.012, f"{l:.3f}", fontsize=8.5)
    a1.set_ylim(0.48, 0.98); a1.set_title(f"Adult, S = {s_}", fontsize=10.5, loc="left")
    a2.plot(x, tr.acc * 100, "-s", color="0.45", mfc="white")
    for xi, v in zip(x, tr.acc * 100): a2.text(xi, v + 0.25, f"{v:.1f}", ha="center", fontsize=8)
    a2.set_ylim(84, 88.5)
    a2.set_xticks(x); a2.set_xticklabels(["original"] + list(tr.feature[1:]), rotation=28, ha="right", fontsize=8.5)
axes[0, 0].set_ylabel("Leakage AUC of S\n(lower is better)"); axes[1, 0].set_ylabel("Accuracy (%)")
fig.tight_layout()
fig.text(0.5, -0.03, "FIG. 5", ha="center", fontsize=12, fontweight="bold")
fig.savefig("figs/fig5_trajectory.png", dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------- FIG. 6 method comparison (leakage + DI, HGB)
methods = ["Unawareness (drop S)", "Pairwise drop (V>=0.3)", "Reweighing", "CorrelationRemover", "Proposed: targeted repair", "Proposed + reweighing"]
short = ["Drop S", "Pairwise drop", "Reweighing", "CorrelationRemover", "Proposed\nrepair", "Proposed +\nreweighing"]
hatches = ["", "///", "...", "xxx", "", "\\\\\\"]
colors = ["white", "white", "white", "white", "0.2", "0.55"]
sets = [("Adult", "gender"), ("Adult", "race"), ("COMPAS", "race"), ("German", "sex")]
fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
w = 0.13
for k, (m, sh, ht, c) in enumerate(zip(methods, short, hatches, colors)):
    L, D = [], []
    for d, s in sets:
        row = next((r for r in get(d, s)["methods"] if r["method"] == m and r["model"] == "hgb"), None)
        L.append(row["leak"][0] if row else np.nan); D.append(row["di"][0] if row else np.nan)
    xs = np.arange(len(sets)) + (k - 2.5) * w
    axes[0].bar(xs, L, w, color=c, edgecolor="black", hatch=ht, label=sh.replace("\n", " "))
    axes[1].bar(xs, D, w, color=c, edgecolor="black", hatch=ht)
axes[0].axhline(0.5, ls="--", color="0.5", lw=0.8); axes[0].set_ylim(0.45, 1.03)
axes[0].set_ylabel("Leakage AUC of S\n(lower is better; 0.5 = none)")
axes[1].axhline(0.8, ls=":", color="black", lw=1); axes[1].text(3.55, 0.81, "80% rule", fontsize=8.5, ha="right")
axes[1].set_ylabel("Disparate impact ratio\n(higher is better; 1 = parity)"); axes[1].set_ylim(0, 1.1)
axes[1].set_xticks(range(len(sets))); axes[1].set_xticklabels([f"{d}\nS = {s}" for d, s in sets])
axes[0].legend(ncol=3, frameon=False, fontsize=8.8, loc="upper center", bbox_to_anchor=(0.5, 1.28))
fig.tight_layout()
fig.text(0.5, -0.02, "FIG. 6", ha="center", fontsize=12, fontweight="bold")
fig.savefig("figs/fig6_comparison.png", dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("ok")
