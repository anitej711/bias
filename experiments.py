import sys, json, time; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from fairlearn.preprocessing import CorrelationRemover
from leakaudit import *

SEEDS = [0, 1, 2, 3, 4]

# ------------------------------------------------------------------ datasets
def load_adult():
    df = pd.read_csv("adult.csv", na_values="?", skipinitialspace=True).dropna()
    df = df.drop(columns=["fnlwgt", "educational-num"])
    y = (df.income == ">50K").astype(int).values  # favourable = >50K
    return df.drop(columns=["income"]), y, "income"

def load_compas():
    c = pd.read_csv("data/compas.csv")
    c = c[(c.days_b_screening_arrest.abs() <= 30) & (c.is_recid != -1) & (c.c_charge_degree != "O") & (c.score_text != "N/A")]
    c = c[c.race.isin(["African-American", "Caucasian"])]
    df = c[["sex", "race", "age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count", "c_charge_degree"]].copy()
    y = (c.two_year_recid == 0).astype(int).values  # favourable = no recidivism
    return df.reset_index(drop=True), y, "two_year_recid"

def load_german():
    g = pd.read_csv("data/german.csv")
    g["sex"] = g.personal_status_sex.str.startswith("female").map({True: "female", False: "male"})
    g["marital"] = g.personal_status_sex.str.split(" : ").str[1]  # a realistic hidden proxy column
    y = g.credit_risk.astype(int).values  # favourable = good credit
    return g.drop(columns=["credit_risk", "personal_status_sex"]), y, "credit_risk"

SETTINGS = [
    ("Adult", load_adult, "gender", lambda d: (d.gender == "Male").astype(int), ["gender", "race"]),
    ("Adult", load_adult, "race", lambda d: (d.race == "White").astype(int), ["gender", "race"]),
    ("COMPAS", load_compas, "race", lambda d: (d.race == "Caucasian").astype(int), ["sex", "race"]),
    ("German", load_german, "sex", lambda d: (d.sex == "male").astype(int), ["sex"]),
]

# ------------------------------------------------------------------ original (submitted) method
def original_method_hits(df, target_series, thr):
    enc = df.copy()
    for c in enc.columns:
        if not is_num(enc[c]): enc[c] = LabelEncoder().fit_transform(enc[c].astype(str))
    kw = ['gender', 'sex', 'race', 'caste', 'religion', 'age', 'dob', 'birth', 'income', 'salary', 'zipcode', 'zip', 'location', 'region', 'school', 'college', 'education']
    sens = [c for c in enc.columns if any(k in c.lower() for k in kw)]
    hits = set()
    for s in sens:
        for c in enc.columns:
            if c != s and abs(pearsonr(enc[s], enc[c])[0]) >= thr: hits.add(tuple(sorted((s, c))))
    return sorted(hits)

# ------------------------------------------------------------------ downstream evaluation
def downstream(Xdf, y, s, seed, weights=None, model="hgb"):
    X = pd.get_dummies(Xdf, dtype=float) if isinstance(Xdf, pd.DataFrame) else Xdf
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed) if model == "hgb" else LogisticRegression(max_iter=2000)
    Xa = X.values if hasattr(X, "values") else X
    if model == "lr":
        mu, sd = Xa[tr].mean(0), Xa[tr].std(0) + 1e-9; Xa = (Xa - mu) / sd
    clf.fit(Xa[tr], y[tr], sample_weight=None if weights is None else weights[tr])
    p = clf.predict(Xa[te]); yt, st = y[te], s.values[te]
    priv, unpriv = p[st == 1].mean(), p[st == 0].mean()
    tpr = lambda m: p[(yt == 1) & m].mean(); fpr = lambda m: p[(yt == 0) & m].mean()
    return dict(acc=(p == yt).mean(), dp=priv - unpriv, di=unpriv / priv if priv > 0 else np.nan,
                eod=max(abs(tpr(st == 1) - tpr(st == 0)), abs(fpr(st == 1) - fpr(st == 0))))

def reweigh(s, y):
    s, y = np.asarray(s), np.asarray(y); w = np.zeros(len(y))
    for a in (0, 1):
        for b in (0, 1):
            m = (s == a) & (y == b)
            if m.sum(): w[m] = (s == a).mean() * (y == b).mean() / m.mean()
    return w

def corr_remover(F, s):
    X = pd.get_dummies(F, dtype=float); X["__s"] = s.values
    Xt = CorrelationRemover(sensitive_feature_ids=["__s"], alpha=1.0).fit_transform(X)
    return pd.DataFrame(Xt, columns=[c for c in X.columns if c != "__s"])

def leak_on_matrix(X, s, seed):
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=100 + seed)
    p = cross_val_predict(clf, X, s, cv=StratifiedKFold(5, shuffle=True, random_state=100 + seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(s, p))

# ------------------------------------------------------------------ run
results = {"settings": []}
for dname, loader, sname, smap, all_sens in SETTINGS:
    t0 = time.time()
    df, y, tname = loader(); s = smap(df).reset_index(drop=True); df = df.reset_index(drop=True)
    detected = detect_sensitive(pd.concat([df, pd.Series(y, name=tname)], axis=1), tname)
    F = df.drop(columns=all_sens)                       # non-sensitive features
    print(f"\n=== {dname} / {sname}  n={len(df)}  detected={detected}")
    orig07 = original_method_hits(pd.concat([df, pd.Series(y, name=tname)], axis=1), y, 0.7)
    orig03 = original_method_hits(pd.concat([df, pd.Series(y, name=tname)], axis=1), y, 0.3)
    ranking, base = attribute(F, s, y, seed=0)
    print(ranking.round(3).head(6).to_string())
    pairwise_proxies = ranking[ranking.assoc_S >= 0.3].feature.tolist()
    repaired, trail = repair(F, s, y, ranking=ranking, seed=0)
    expl = explain(ranking, F, s, sname)

    methods = {
        "No mitigation (S used)": dict(X=df.drop(columns=[c for c in all_sens if c != sname]), Xleak=F),
        "Unawareness (drop S)": dict(X=F, Xleak=F),
        "Pairwise drop (V>=0.3)": dict(X=F.drop(columns=pairwise_proxies), Xleak=F.drop(columns=pairwise_proxies)),
        "Reweighing": dict(X=F, Xleak=F, w=reweigh(s, y)),
        "CorrelationRemover": dict(X=corr_remover(F, s), Xleak=corr_remover(F, s)),
        "Proposed: targeted repair": dict(X=repaired, Xleak=repaired),
    }
    rows = []
    for m, spec in methods.items():
        Xl = pd.get_dummies(spec["Xleak"], dtype=float) if isinstance(spec["Xleak"], pd.DataFrame) and any(not is_num(spec["Xleak"][c]) for c in spec["Xleak"].columns) else spec["Xleak"]
        leaks = [leak_on_matrix(Xl, s, sd) for sd in SEEDS[:3]]
        for model in ["hgb", "lr"]:
            ev = [downstream(spec["X"], y, s, sd, spec.get("w"), model) for sd in SEEDS]
            agg = {k: (float(np.nanmean([e[k] for e in ev])), float(np.nanstd([e[k] for e in ev]))) for k in ev[0]}
            rows.append(dict(method=m, model=model, leak=(float(np.mean(leaks)), float(np.std(leaks))), **agg))
            print(f"  {m:28s} {model}  leak {np.mean(leaks):.3f}  acc {agg['acc'][0]:.3f}  DP {agg['dp'][0]:.3f}  DI {agg['di'][0]:.3f}  EOD {agg['eod'][0]:.3f}")
    results["settings"].append(dict(
        dataset=dname, sensitive=sname, n=int(len(df)), n_features=int(F.shape[1]), detected=detected,
        original_hits_07=orig07, original_hits_03=orig03, base_leak=base,
        ranking=ranking.to_dict("records"), pairwise_proxies=pairwise_proxies,
        trail=trail.to_dict("records"), explanations=expl, methods=rows,
        base_rate_priv=float(y[s.values == 1].mean()), base_rate_unpriv=float(y[s.values == 0].mean()),
    ))
    print(f"  ({time.time()-t0:.0f}s)")
    json.dump(results, open("results.json", "w"), indent=1, default=str)
print("done")
