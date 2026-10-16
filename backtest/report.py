"""Turn results.json + team_games_scored.csv into backtest.html (run after backtest.py and rules.py)."""
import json
from pathlib import Path
import pandas as pd
from backtest import auc, board_score, KEYS

HERE = Path(__file__).parent
r = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
model = json.loads((HERE.parent / "model.json").read_text(encoding="utf-8"))
hist = model.get("history", [])
r["model"] = model
r["prev_weights"] = hist[-2]["weights"] if len(hist) > 1 else hist[0]["weights"]
learn = HERE / "learn_status.json"
r["learn"] = json.loads(learn.read_text(encoding="utf-8")) if learn.exists() else None
f = pd.read_csv(HERE / "team_games_scored.csv")

# the current model's score on every season (in case model.json changed since backtest.py ran)
W = {k: model["weights"].get(k, 0) for k in KEYS}
f["score_new"] = board_score(f, W)
for row in r["by_season"]:
    y = int(row["season"][:4])
    s = f[f.season == y]
    row["new"] = round(auc(s.score_new, s.over_lg), 4)
bands = pd.cut(f.score_new, [-1, 30, 40, 50, 60, 70, 101], labels=["Under 30", "30–40", "40–50", "50–60", "60–70", "70+"])
r["bands_new"] = [dict(band=str(i), pts=round(x.pts, 2), resid=round(x.resid, 2), under=round(x.under, 4), n=int(x.n)) for i, x in
                  f.groupby(bands, observed=False).agg(pts=("pts", "mean"), resid=("resid", "mean"), under=("under", "mean"), n=("pts", "size")).iterrows()]
r["auc_new_all"] = round(auc(f.score_new, f.over_lg), 4)
r["auc_new_mkt"] = round(auc(f.score_new, f.over), 4)
r["situational"] = r.get("spots", [])

tpl = (HERE / "report_template.html").read_text(encoding="utf-8")
head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>')
html = tpl.replace("/*__DATA__*/null", json.dumps(r, default=float))
(HERE / "report.html").write_text(html, encoding="utf-8")
(HERE.parent / "backtest.html").write_text(head + html + "</body></html>", encoding="utf-8")
print("wrote backtest.html", r["auc_new_all"], r["bands_new"])
