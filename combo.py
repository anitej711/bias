import json, os
src = open("experiments.py").read().split("# ------------------------------------------------------------------ run")[0]
exec(src)
os.makedirs("repaired", exist_ok=True)
res = json.load(open("results.json"))
for st, (dname, loader, sname, smap, all_sens) in zip(res["settings"], SETTINGS):
    df, y, tname = loader(); s = smap(df).reset_index(drop=True); df = df.reset_index(drop=True)
    st["detected"] = detect_sensitive(pd.concat([df, pd.Series(y, name=tname)], axis=1), tname)
    F = df.drop(columns=all_sens)
    ranking = pd.DataFrame(st["ranking"])
    repaired, trail = repair(F, s, y, ranking=ranking, seed=0, log=lambda *a: None)
    out = repaired.copy(); out[tname] = y; out.to_csv(f"repaired/{dname}_{sname}_repaired.csv", index=False)
    w = reweigh(s, y)
    leaks = [leak_on_matrix(pd.get_dummies(repaired, dtype=float), s, sd) for sd in SEEDS[:3]]
    for model in ["hgb", "lr"]:
        ev = [downstream(repaired, y, s, sd, w, model) for sd in SEEDS]
        agg = {k: (float(np.nanmean([e[k] for e in ev])), float(np.nanstd([e[k] for e in ev]))) for k in ev[0]}
        row = dict(method="Proposed + reweighing", model=model, leak=(float(np.mean(leaks)), float(np.std(leaks))), **agg)
        st["methods"] = [m for m in st["methods"] if not (m["method"] == row["method"] and m["model"] == model)] + [row]
        print(dname, sname, model, {k: round(v[0], 3) for k, v in row.items() if isinstance(v, tuple)})
    print(dname, sname, "detected:", st["detected"])
json.dump(res, open("results.json", "w"), indent=1, default=str)
print("done")
