# Proxy-leakage audit and repair: prototype

Prototype code cited in the revised IDF documents (modules 102–116, FIG. 1).

| File | What it is |
|---|---|
| `leakaudit.py` | The method: sensitive-attribute detection, Cramér's V / η, leakage AUC, leave-one-out attribution, utility-aware category merging, tie-safe quantile repair, verified repair loop, explanation report |
| `experiments.py` | Evaluation on Adult (gender, race), COMPAS (race), German Credit (sex) against 5 alternatives, 5 seeds |
| `combo.py` | Adds "repair + reweighing" and writes the repaired datasets |
| `fig_diagrams.py`, `fig_results.py` | Regenerate FIG. 1–6 |
| `results.json` | All numbers used in the IDF tables and text |

## Run
```
pip install pandas numpy scipy scikit-learn fairlearn matplotlib
# place adult.csv here, and compas.csv + german.csv in data/
python experiments.py && python combo.py && python fig_diagrams.py && python fig_results.py
```
Adult: the 48,842-row version with columns age, workclass, fnlwgt, education, educational-num, ...
COMPAS: ProPublica compas-scores-two-years.csv. German Credit: the 1,000-row version with a personal_status_sex column.
