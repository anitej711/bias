"""
Multivariate proxy-leakage audit and targeted repair for tabular data.

Pipeline:
  1. ingest + clean            -> load(), clean
  2. sensitive attribute ID    -> detect_sensitive()
  3. pairwise association      -> association()  (Cramer's V / correlation ratio)
  4. multivariate leakage      -> leakage_auc()   (auxiliary predictor of S)
  5. leakage attribution       -> attribute()     (leave-one-variable-out drop in AUC)
  6. proxy risk score          -> risk = attribution x NMI(X, Y)
  7. targeted repair loop      -> repair()        (utility-aware category merging /
                                                   quantile repair, iterated until
                                                   leakage < tau or utility budget hit)
  8. report                    -> explain()
"""
import numpy as np, pandas as pd, warnings
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score, mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")

SENSITIVE_NAME_HINTS = ["gender", "sex", "race", "ethnic", "religion", "caste", "nationality", "marital_status_sex"]
SENSITIVE_VALUE_SIGNATURES = [
    {"male", "female"}, {"m", "f"}, {"man", "woman"},
    {"white", "black"}, {"caucasian", "african-american"}, {"asian-pac-islander"}, {"hispanic"},
]

def is_num(x): return pd.api.types.is_numeric_dtype(x)

# ---------------------------------------------------------------- 2. sensitive attributes
def detect_sensitive(df, target):
    found = []
    for c in df.columns:
        if c == target: continue
        name_hit = any(h in c.lower() for h in SENSITIVE_NAME_HINTS)
        vals = set(str(v).strip().lower() for v in df[c].dropna().unique()[:200]) if not is_num(df[c]) else set()
        # single-letter codes ({m,f}) are ambiguous (e.g. Felony/Misdemeanour), so they
        # only count when the column name also hints at a sensitive attribute
        value_hit = any(len(sig & vals) >= min(2, len(sig)) for sig in SENSITIVE_VALUE_SIGNATURES
                        if not all(len(v) == 1 for v in sig))
        if name_hit or value_hit: found.append(c)
    return found

# ---------------------------------------------------------------- 3. pairwise association
def cramers_v(x, s):
    t = pd.crosstab(x, s)
    if min(t.shape) < 2: return 0.0
    chi2 = chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))))

def corr_ratio(num, cat):
    num = num.astype(float); m = num.mean()
    ssb = sum(len(g) * (g.mean() - m) ** 2 for _, g in num.groupby(cat))
    sst = ((num - m) ** 2).sum()
    return float(np.sqrt(ssb / sst)) if sst > 0 else 0.0

def association(x, s):
    return corr_ratio(x, s) if is_num(x) else cramers_v(x, s)

def binned(x, q=10):
    return pd.qcut(x.rank(method="first"), q, labels=False).astype(str) if is_num(x) else x.astype(str)

# ---------------------------------------------------------------- 4. multivariate leakage
def _X(df):
    return pd.get_dummies(df, dtype=float)

def leakage_auc(df_feats, s, seed=0, folds=3):
    X = _X(df_feats)
    if X.shape[1] == 0: return 0.5
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
    p = cross_val_predict(clf, X, s, cv=StratifiedKFold(folds, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(s, p))

# ---------------------------------------------------------------- 5-6. attribution + risk
def attribute(df_feats, s, y, base=None, seed=0):
    base = base if base is not None else leakage_auc(df_feats, s, seed)
    rows = []
    for c in df_feats.columns:
        drop = base - leakage_auc(df_feats.drop(columns=[c]), s, seed)
        rows.append(dict(feature=c, assoc_S=association(df_feats[c], s.map(str)),
                         leak_drop=max(drop, 0.0),
                         nmi_Y=normalized_mutual_info_score(binned(df_feats[c]), y)))
    r = pd.DataFrame(rows)
    r["risk"] = r.leak_drop.rank(pct=True) * 0.5 + r.assoc_S.rank(pct=True) * 0.5
    return r.sort_values(["leak_drop", "assoc_S"], ascending=False).reset_index(drop=True), base

# ---------------------------------------------------------------- 7. targeted repair
def merge_categories(x, s, y, beta=1.0, target_v=0.10, min_cats=2):
    """Greedy utility-aware merging: repeatedly merge the category pair that most
    reduces I(X;S) per unit of I(X;Y) lost, until Cramer's V(X,S) <= target_v."""
    x = x.astype(str).copy(); mapping = {c: c for c in x.unique()}
    while x.nunique() > min_cats and cramers_v(x, s) > target_v:
        cats = x.value_counts().index[:25]  # cap search for very high-cardinality columns
        best, best_score = None, -np.inf
        i_s, i_y = mutual_info_score(x, s), mutual_info_score(x, y)
        for i, a in enumerate(cats):
            for b in cats[i + 1:]:
                xm = x.where(~x.isin([a, b]), f"{a}|{b}")
                ds = i_s - mutual_info_score(xm, s)
                dy = i_y - mutual_info_score(xm, y)
                score = ds - beta * dy
                if score > best_score: best, best_score = (a, b), score
        a, b = best; new = f"{a}|{b}"
        x = x.where(~x.isin([a, b]), new)
        for k, v in mapping.items():
            if v in (a, b): mapping[k] = new
    return x, mapping

def quantile_repair(x, s, lam=1.0, seed=0):
    """Rank-preserving repair of a numeric feature toward the pooled distribution
    (Feldman et al. style), with repair strength lam in [0,1]."""
    x = x.astype(float); out = x.copy(); pooled = np.sort(x.values)
    rng = np.random.default_rng(seed)
    for g in s.unique():
        m = (s == g).values
        # random tie-breaking: otherwise a tied value (e.g. 40 h/week) maps to a
        # different repaired value in each group and *creates* leakage
        xs = x[m].values + rng.uniform(0, 1e-6, m.sum())
        r = (pd.Series(xs).rank(method="first").values - 0.5) / m.sum()
        target = np.quantile(pooled, np.clip(r, 0, 1))
        out[m] = (1 - lam) * x[m].values + lam * target
    return out

def utility(df_feats, y, seed=0):
    X = _X(df_feats)
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(((p > 0.5) == y).mean())

def repair(df_feats, s, y, tau=0.65, max_acc_drop=0.02, seed=0, ranking=None, log=print):
    """Audit once, then walk the ranked proxy list applying a targeted repair to each.
    A repair is kept only if it lowers leakage AND keeps accuracy within budget.
    Stops when leakage <= tau or the ranked list is exhausted."""
    cur = df_feats.copy(); s_str = s.map(str)
    if ranking is None: ranking, _ = attribute(cur, s, y, seed=seed)
    leak = leakage_auc(cur, s, seed); acc0 = utility(cur, y, seed)
    trail = [dict(step=0, feature="-", action="original", leakage=leak, acc=acc0, kept=True)]
    cands = ranking[(ranking.leak_drop > 0) | (ranking.assoc_S >= 0.1)].feature.tolist()
    for step, f in enumerate(cands, 1):
        if leak <= tau: break
        trial = cur.copy()
        if is_num(cur[f]):
            trial[f] = quantile_repair(cur[f], s_str, seed=seed); desc = "rank-preserving quantile repair"
        else:
            trial[f], mp = merge_categories(cur[f], s_str, y)
            groups = {}
            for k, v in mp.items(): groups.setdefault(v, []).append(k)
            merged = [" + ".join(sorted(g)) for g in groups.values() if len(g) > 1]
            desc = "merge: " + ("; ".join(merged) if merged else "none")
        new_leak = leakage_auc(trial, s, seed); new_acc = utility(trial, y, seed)
        kept = (new_leak < leak - 0.002) and (acc0 - new_acc <= max_acc_drop)
        trail.append(dict(step=step, feature=f, action=desc, leakage=new_leak, acc=new_acc, kept=kept))
        log(f"  {f:22s} {'KEEP' if kept else 'skip'}  leak {new_leak:.3f} acc {new_acc:.3f}  {desc[:90]}")
        if kept: cur, leak = trial, new_leak
    return cur, pd.DataFrame(trail)

# ---------------------------------------------------------------- 8. explanation
def explain(ranking, df_feats, s, s_name, labels=None, top=3, min_support=0.01):
    """Human-readable evidence per ranked proxy. labels maps S codes to names,
    e.g. {1: 'Male', 0: 'Female'}. Categories below min_support are ignored."""
    labels = labels or {}
    lines = []
    for _, r in ranking.head(top).iterrows():
        f = r.feature; x = df_feats[f]
        if not is_num(x):
            keep = x.value_counts(normalize=True); keep = keep[keep >= min_support].index
            ct = pd.crosstab(x[x.isin(keep)], s[x.isin(keep)], normalize="index")
            g = ct.columns.max(); col = ct[g].sort_values()
            name = labels.get(g, f"{s_name}={g}")
            evidence = (f"'{col.index[-1]}' is {col.iloc[-1]*100:.1f}% {name}; "
                        f"'{col.index[0]}' is {col.iloc[0]*100:.1f}% {name} (overall {s.mean()*100:.1f}%)")
        else:
            means = x.groupby(s).mean()
            evidence = "mean " + ", ".join(f"{labels.get(k, k)} = {v:.1f}" for k, v in means.items())
        lines.append(dict(feature=f, strength=("strong" if r.assoc_S >= .3 else "moderate" if r.assoc_S >= .1 else "weak"),
                          assoc=round(r.assoc_S, 3), leak_drop=round(r.leak_drop, 3), nmi_Y=round(r.nmi_Y, 3), evidence=evidence))
    return lines
