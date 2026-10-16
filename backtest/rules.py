"""Score-threshold rules: what happened when the board's score, or the edge against the
book, was very low or very high.

Uses team_games_scored.csv from backtest.py (scores there use the same 0-100 scales as the
live board). Reports, for team totals and for game totals, how often the under hit against
the closing line, with a 95% floor, on the training seasons and the last complete season.

Run after backtest.py:  python backtest/rules.py   -> adds "rules" to results.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from backtest import wilson_lo, P

HERE = Path(__file__).parent


def summarize(d, label, kind, side, test):
    y = d.under if kind == "team" else d.gunder
    out = dict(label=label, kind=kind, side=side, n=int(len(d)))
    if not len(d):
        return out
    p = float(y.mean())
    out.update(under=round(p, 4), under_lo=round(float(wilson_lo(p if side != "over" else 1 - p, len(d))), 4),
               resid=round(float((d.resid if kind == "team" else d.gresid).mean()), 2))
    t = d[d.season == test]
    out["test_n"] = int(len(t))
    out["test_under"] = round(float((t.under if kind == "team" else t.gunder).mean()), 4) if len(t) else None
    return out


def main():
    f = pd.read_csv(HERE / "team_games_scored.csv", parse_dates=["date"])
    seasons = sorted(f.season.unique())
    current = max(seasons)
    complete = [s for s in seasons if s < current] if (f.season == current).sum() < 2000 else seasons
    f = f[f.season.isin(complete)].copy()
    test = max(complete)
    T = f
    gm = f[f.home == 1].merge(f[f.home == 0][["gameId", "proj", "score_default"]].rename(columns={"proj": "proj_a", "score_default": "score_a"}), on="gameId")
    gm["proj_total"] = 2 * gm.lg_pts + P["gameShrink"] * (gm.proj + gm.proj_a - 2 * gm.lg_pts)
    gm["gedge"] = gm.total - gm.proj_total
    gm["gunder"] = ((gm.pts + gm.pa) < gm.total).astype(int)
    gm["gresid"] = gm.pts + gm.pa - gm.total
    gm["combined"] = gm.score_default + gm.score_a
    G = gm
    specs = [
        ("All team totals", "team", lambda d: d.score_default >= -1, "all"),
        ("Team score under 25", "team", lambda d: d.score_default < 25, "under"),
        ("Team score under 30", "team", lambda d: d.score_default < 30, "under"),
        ("Team score under 35", "team", lambda d: d.score_default < 35, "under"),
        ("Team score 65 or higher", "team", lambda d: d.score_default >= 65, "over"),
        ("Team score 70 or higher", "team", lambda d: d.score_default >= 70, "over"),
        ("Team score 75 or higher", "team", lambda d: d.score_default >= 75, "over"),
        ("Edge 2+ toward the under", "team", lambda d: d.edge >= 2, "under"),
        ("Edge 3+ toward the under", "team", lambda d: d.edge >= 3, "under"),
        ("Edge 4+ toward the under", "team", lambda d: d.edge >= 4, "under"),
        ("Edge 6+ toward the under", "team", lambda d: d.edge >= 6, "under"),
        ("Edge 3+ and score under 40", "team", lambda d: (d.edge >= 3) & (d.score_default < 40), "under"),
        ("Edge 3+ on a back-to-back", "team", lambda d: (d.edge >= 3) & (d.b2b == 1), "under"),
        ("Edge 3+ and fatigue 30+", "team", lambda d: (d.edge >= 3) & (d.fatigue >= 30), "under"),
        ("Edge 3+ toward the over", "team", lambda d: d.edge <= -3, "over"),
        ("All games", "game", lambda d: d.combined >= -1, "all"),
        ("Combined under 60", "game", lambda d: d.combined < 60, "under"),
        ("Combined under 70", "game", lambda d: d.combined < 70, "under"),
        ("Combined under 80", "game", lambda d: d.combined < 80, "under"),
        ("Combined 120 or higher", "game", lambda d: d.combined >= 120, "over"),
        ("Combined 130 or higher", "game", lambda d: d.combined >= 130, "over"),
        ("Game edge 3+ toward the under", "game", lambda d: d.gedge >= 3, "under"),
        ("Game edge 5+ toward the under", "game", lambda d: d.gedge >= 5, "under"),
        ("Game edge 8+ toward the under", "game", lambda d: d.gedge >= 8, "under"),
        ("Game edge 5+ toward the over", "game", lambda d: d.gedge <= -5, "over"),
    ]
    rules = []
    for label, kind, mask, side in specs:
        D = T if kind == "team" else G
        r = summarize(D[mask(D)], label, kind, side, test)
        r["share"] = round(len(D[mask(D)]) / len(D), 4)
        rules.append(r)
    res_path = HERE / "results.json"
    res = json.loads(res_path.read_text(encoding="utf-8"))
    res["rules"] = dict(seasons=[f"{s}-{str(s + 1)[2:]}" for s in complete], test=f"{test}-{str(test + 1)[2:]}",
                        team_n=int(len(T)), game_n=int(len(G)), rows=rules,
                        dist=dict(team=[round(float(x), 1) for x in np.percentile(T.score_default.dropna(), [10, 25, 50, 75, 90])],
                                  game=[round(float(x), 1) for x in np.percentile(G.combined.dropna(), [10, 25, 50, 75, 90])]))
    res_path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    pct = lambda v: "—" if v is None else f"{v * 100:.1f}%"
    print(f"team-games {len(T)}, games {len(G)}; score pctiles {res['rules']['dist']['team']}, combined {res['rules']['dist']['game']}")
    for r in rules:
        print(f"{r['label']:<34} n={r['n']:>5} ({r['share'] * 100:4.1f}%)  under {pct(r.get('under'))} floor {pct(r.get('under_lo'))}  miss {r.get('resid')}  | {res['rules']['test']}: n={r.get('test_n')} under {pct(r.get('test_under'))}")


if __name__ == "__main__":
    main()
