"""Walk-forward backtest of the NBA Unders Board scoring.

For every regular-season team-game from 2022-23 on, rebuild each input using only games
played before that date (plus last season, counted as priorGames games, exactly like the
live board), then check which inputs predict that team's points, and, more importantly,
which predict points *relative to the closing line* (the market already knows the obvious).

Run:  python backtest/backtest.py    (downloads any missing season into backtest/data/)
Writes results.json and team_games_scored.csv next to this file. The weekly GitHub
Action runs it, then rules.py, learn.py and report.py.
"""
import json, sys
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import nbadata as nd

FIRST_SEASON = 2022
CURRENT = nd.season_of(date.today())
SEASONS = list(range(FIRST_SEASON, CURRENT + 1))         # trimmed to seasons with games in load()
MODEL = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
PRIOR_GAMES = MODEL.get("priorGames", 15)
P, FP = MODEL["projection"], MODEL["fatigue"]
KEYS = ["off", "def", "pace", "fatigue", "oppFatigue", "avail", "l5"]
DEFAULT_W = {k: MODEL["weights"].get(k, 0) for k in KEYS}
MIN_PPG = 6           # a player counts as "missing" only if he had been scoring at least this much


# ---------------------------------------------------------------- load
def load():
    frames, players = [], []
    for y in range(FIRST_SEASON - 1, CURRENT + 1):
        rows = nd.team_games(y, refresh=(y == CURRENT) or not (nd.DATA / f"games_{y}.csv").exists())
        if rows:
            frames.append(pd.DataFrame(rows))
            players.append(pd.DataFrame(nd._read_csv(nd.DATA / f"players_{y}.csv")).assign(season=y))
    g = pd.concat(frames, ignore_index=True)
    for c in ["season", "type", "home", "neutral", "pts", "pa", "ot", "fga", "fta", "tpa", "orb", "tov"]:
        g[c] = pd.to_numeric(g[c])
    for c in ["poss", "total", "spread"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g[g.type == 2].copy()
    g["date"] = pd.to_datetime(g["date"])
    g["gameId"] = g["gameId"].astype(str)
    p = pd.concat(players, ignore_index=True)
    for c in ["starter", "dnp", "min", "pts"]:
        p[c] = pd.to_numeric(p[c])
    p["gameId"] = p["gameId"].astype(str)
    SEASONS[:] = [y for y in SEASONS if (g.season == y).sum() > 0]
    return g.sort_values(["date", "gameId", "team"]).reset_index(drop=True), p


# ---------------------------------------------------------------- walk-forward features
def team_features(g):
    stat = ["pts", "pa", "poss"]
    tot = g.groupby(["team", "season"])[stat].sum()
    gp = g.groupby(["team", "season"]).size()
    league = {y: (g[g.season == y][stat].sum() / (g.season == y).sum()).to_dict() for y in g.season.unique()}
    out = []
    for (team, season), tg in g[g.season.isin(SEASONS)].groupby(["team", "season"]):
        tg = tg.sort_values("date").copy()
        prior = tot.loc[(team, season - 1)] / gp.loc[(team, season - 1)] if (team, season - 1) in gp.index else pd.Series(league[season - 1])
        cum = tg[stat].cumsum().shift(1).fillna(0)
        n = np.arange(len(tg))
        for c in stat:
            tg["pre_" + c] = (prior[c] * PRIOR_GAMES + cum[c].values) / (PRIOR_GAMES + n)
        tg["pre_ortg"] = 100 * tg.pre_pts / tg.pre_poss
        tg["pre_drtg"] = 100 * tg.pre_pa / tg.pre_poss
        tg["pre_pace"] = tg.pre_poss
        tg["pre_l5"] = tg["pts"].shift(1).rolling(5, min_periods=3).mean()
        tg["n_cur"] = n
        out.append(tg)
    f = pd.concat(out)
    # schedule features from the schedule alone (same function as the live board)
    feats = {}
    for y in SEASONS:
        gs = g[(g.season == y) & (g.home == 1)]
        sched = [dict(id=r.gameId, date=r.date.strftime("%Y-%m-%d"), away=r.opp, home=r.team, neutral=bool(r.neutral), city=r.city) for r in gs.itertuples()]
        feats.update(nd.schedule_features(sched))
    fe = pd.DataFrame([dict(gameId=k[0], team=k[1], **v) for k, v in feats.items()])
    fe["fatigue"] = [nd.fatigue_points(v, FP) for v in feats.values()]
    f = f.merge(fe.drop(columns="ha"), on=["gameId", "team"], how="left")
    return f, league


def availability(f, p):
    """Pregame scoring share of the regulars who did not play: a proxy for the injury report.
    A regular = played for this team in the last 21 days with a blended ppg of MIN_PPG+."""
    p = p.drop(columns=["season"]).merge(f[["gameId", "team", "date", "season", "pre_pts"]].drop_duplicates(["gameId", "team"]), on=["gameId", "team"], how="inner")
    p = p.sort_values(["pid", "date"])
    played = p[(p.dnp == 0) & (p["min"] > 0)]
    # blended ppg before this game: prior season (scaled to PRIOR_GAMES) + current season so far
    rows = []
    for pid, px in played.groupby("pid"):
        px = px.sort_values("date")
        for y in px.season.unique():
            cur = px[px.season == y]
            pr = px[px.season == y - 1]
            ppg_prior = pr.pts.sum() / len(pr) if len(pr) else None
            cpts = cur.pts.cumsum().shift(1).fillna(0).values
            n = np.arange(len(cur))
            if ppg_prior is not None:
                ppg = (ppg_prior * PRIOR_GAMES + cpts) / (PRIOR_GAMES + n)
            else:
                ppg = np.where(n >= 3, cpts / np.maximum(n, 1), np.nan)
            rows.append(pd.DataFrame(dict(pid=pid, gameId=cur.gameId.values, team=cur.team.values, date=cur.date.values, ppg=ppg)))
    ppg = pd.concat(rows)
    # roster snapshot: for each team-game, regulars seen in the previous 21 days and their latest pregame ppg
    out = {}
    for team, tg in f.groupby("team"):
        tg = tg.sort_values("date")
        seen = ppg[ppg.team == team].sort_values("date")
        appear = played[played.team == team][["gameId", "pid"]]
        in_game = appear.groupby("gameId")["pid"].apply(set).to_dict()
        for r in tg.itertuples():
            win = seen[(seen.date < r.date) & (seen.date >= r.date - pd.Timedelta(days=21))]
            if win.empty:
                out[(r.gameId, team)] = (0.0, "")
                continue
            last = win.groupby("pid").tail(1)
            regs = last[last.ppg >= MIN_PPG]
            there = in_game.get(r.gameId, set())
            miss = regs[~regs.pid.isin(there)]
            out[(r.gameId, team)] = (float(miss.ppg.sum() / r.pre_pts) if r.pre_pts else 0.0, ";".join(miss.pid.astype(str)))
    f["lost"] = [out.get((a, b), (0.0, ""))[0] for a, b in zip(f.gameId, f.team)]
    f["missing"] = [out.get((a, b), (0.0, ""))[1] for a, b in zip(f.gameId, f.team)]
    return f


def league_ranks(f):
    """Each team's league rank (1 = best offense / best defense / fastest) as of each game date,
    from the 30 teams' latest pregame values, for the top-10 vs bottom-10 matchup flags."""
    out = []
    for y, fs in f.groupby("season"):
        snap = fs[["date", "team", "pre_ortg", "pre_drtg", "pre_pace"]].sort_values("date")
        dates = pd.DataFrame({"date": sorted(fs.date.unique())})
        rows = []
        for t, tg in snap.groupby("team"):
            m = pd.merge_asof(dates, tg.drop(columns="team"), on="date", direction="forward")
            m["team"] = t
            rows.append(m)
        s = pd.concat(rows)
        s["rk_off"] = s.groupby("date").pre_ortg.rank(ascending=False, method="first")
        s["rk_def"] = s.groupby("date").pre_drtg.rank(ascending=True, method="first")
        s["rk_pace"] = s.groupby("date").pre_pace.rank(ascending=False, method="first")
        out.append(fs.merge(s[["date", "team", "rk_off", "rk_def", "rk_pace"]], on=["date", "team"], how="left"))
    return pd.concat(out)


def add_model(f, league):
    lg = pd.DataFrame({y: league[y - 1] for y in SEASONS}).T     # last season's league averages (known pregame)
    lg["ortg"] = 100 * lg.pts / lg.poss
    L = lg.loc[f.season].reset_index(drop=True)
    f = f.reset_index(drop=True)
    opp = f[["gameId", "team", "pre_ortg", "pre_drtg", "pre_pace", "pre_pa", "fatigue", "lost", "b2b", "g4", "rest", "rk_def", "rk_pace"]].rename(
        columns={"team": "opp", "pre_ortg": "opp_ortg", "pre_drtg": "opp_drtg", "pre_pace": "opp_pace", "pre_pa": "opp_pa", "fatigue": "opp_fatigue",
                 "rk_def": "rk_oppdef", "rk_pace": "rk_opppace",
                 "lost": "opp_lost", "b2b": "opp_b2b", "g4": "opp_g4", "rest": "opp_rest"})
    f = f.merge(opp, on=["gameId", "opp"], how="inner")
    L = lg.loc[f.season].reset_index(drop=True)
    f["offR"] = f.pre_ortg / L.ortg
    f["defR"] = f.opp_drtg / L.ortg
    f["exp_poss"] = L.poss * (f.pre_pace / L.poss) * (f.opp_pace / L.poss)
    venue = np.where(f.home == 1, P["home"], P["away"])
    raw = f.exp_poss / 100 * L.ortg * f.offR * f.defR * venue * (1 - P["fatigueK"] * f.fatigue / 100) * (1 + P["oppFatigueK"] * f.opp_fatigue / 100) * (1 - P["availK"] * (f.lost.clip(upper=1) - P["availBase"]))
    f["proj"] = L.pts + P["shrink"] * (raw - L.pts)
    f["implied"] = np.where(f.home == 1, (f.total - f.spread) / 2, (f.total + f.spread) / 2)
    f["edge"] = f.implied - f.proj                      # positive = model sees fewer points than the book
    f["resid"] = f.pts - f.implied                      # what the market missed
    f["under"] = (f.pts < f.implied).astype(int)
    f["over"] = (f.pts > f.implied).astype(int)
    f["lg_pts"] = L.pts
    f["over_lg"] = (f.pts > L.pts).astype(int)            # the board's job: rank spots by points vs league average
    top, bot = lambda r: r <= 10, lambda r: r >= 21
    f["matchup"] = np.select([top(f.rk_off) & bot(f.rk_oppdef), bot(f.rk_off) & top(f.rk_oppdef), top(f.rk_off) & top(f.rk_oppdef), bot(f.rk_off) & bot(f.rk_oppdef)],
                             ["OVER", "UNDER", "STRENGTH", "WEAK"], "")
    f["pace_tag"] = np.select([top(f.rk_pace) | top(f.rk_opppace), bot(f.rk_pace) & bot(f.rk_opppace)], ["fast", "slow"], "")
    f["proj_slope"] = 0.0
    return f


# ---------------------------------------------------------------- scoring like the board
SUB = dict(off=("pre_ortg", 1), def_=("opp_drtg", 1), pace=("exp_poss", 1), fatigue=("fatigue", -1), oppFatigue=("opp_fatigue", 1),
           avail=("lost", -1), l5=("pre_l5", 1))
SUB = {k.rstrip("_"): v for k, v in SUB.items()}


def subscores(f):
    for k, (col, sign) in SUB.items():
        out = pd.Series(np.nan, index=f.index)
        for y, idx in f.groupby("season").groups.items():
            v = f.loc[idx, col]
            if k in ("fatigue", "oppFatigue"):
                lo, hi = 0, 60
            elif k == "avail":
                lo, hi = 0, 0.4
            else:
                lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
            s = np.clip((v - lo) / (hi - lo), 0, 1) * 100
            out.loc[idx] = s if sign > 0 else 100 - s
        f["s_" + k] = out
    return f


def board_score(f, W):
    num = sum(f["s_" + k].fillna(0) * w for k, w in W.items())
    den = sum(f["s_" + k].notna() * w for k, w in W.items())
    return num / den.replace(0, np.nan)


# ---------------------------------------------------------------- stats helpers
def auc(score, y):
    m = score.notna() & y.notna()
    s, y = score[m].values, y[m].values
    r = pd.Series(s).rank().values
    pos = y == 1
    if not pos.sum() or not (~pos).sum():
        return None
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def spearman(a, b):
    m = a.notna() & b.notna()
    return float(np.corrcoef(a[m].rank(), b[m].rank())[0, 1])


def quintiles(score, f, y="over"):
    m = score.notna() & f[y].notna()
    q = pd.qcut(score[m].rank(method="first"), 5, labels=False)
    d = f[m].assign(q=q.values)
    t = d.groupby("q").agg(pts=("pts", "mean"), resid=("resid", "mean"), over=(y, "mean"), n=("pts", "size"))
    return [dict(q=int(i) + 1, pts=round(r.pts, 2), resid=round(r.resid, 2), over=round(r.over, 4), n=int(r.n)) for i, r in t.iterrows()]


def logistic(X, y, l2=1e-3, iters=50):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (y - p) - l2 * np.r_[0, b[1:]]
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(b) - 1)])
        b += np.linalg.solve(H, g)
    return b


def wilson_lo(p, n, z=1.96):
    if not n:
        return None
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / d


# ---------------------------------------------------------------- main
def main():
    g, p = load()
    f, league = team_features(g)
    f = availability(f, p)
    f = league_ranks(f)
    f = add_model(f, league)
    f = subscores(f)
    f["score_default"] = board_score(f, DEFAULT_W)
    f = f[f.implied.notna()].copy()
    label = lambda y: nd.season_label(y) + (" (to date)" if y == CURRENT else "")
    res = {"n": int(len(f)), "seasons": [label(y) for y in SEASONS], "current": label(CURRENT),
           "current_n": int((f.season == CURRENT).sum()), "data_through": str(f.date.max().date()),
           "base_under": round(float(f.under.mean()), 4), "base_pts": round(float(f.pts.mean()), 2),
           "base_implied": round(float(f.implied.mean()), 2), "sd_resid": round(float(f.resid.std()), 2),
           "ot_share": round(float((f.ot > 0).mean()), 4), "ot_extra": round(float(f[f.ot > 0].pts.mean() - f[f.ot == 0].pts.mean()), 2)}

    # 1. single inputs: vs points, and vs the market's miss (resid). AUC is for "went over the implied line".
    inputs = [
        ("Offense rating (ORtg)", "pre_ortg", 1, "team"), ("Team Pts/G", "pre_pts", 1, "team"), ("Pts L5", "pre_l5", 1, "team"),
        ("Opp DRtg", "opp_drtg", 1, "opp"), ("Opp pts allowed/G", "opp_pa", 1, "opp"),
        ("Expected possessions", "exp_poss", 1, "matchup"), ("Team pace", "pre_pace", 1, "team"), ("Opp pace", "opp_pace", 1, "opp"),
        ("Fatigue index", "fatigue", -1, "spot"), ("Opp fatigue index", "opp_fatigue", 1, "spot"),
        ("Scoring share missing", "lost", -1, "spot"), ("Opp scoring share missing", "opp_lost", 1, "spot"),
        ("Home court", "home", 1, "spot"), ("Back-to-back", "b2b", -1, "spot"), ("Opp back-to-back", "opp_b2b", 1, "spot"),
        ("3 games in 4 days", "g4", -1, "spot"), ("Rest days", "rest", 1, "spot"), ("Road trip game #", "trip", -1, "spot"),
        ("Travel km", "km", -1, "spot"), ("Time zones crossed", "tz", -1, "spot"), ("Visiting altitude", "alt", -1, "spot"),
        ("Projected points", "proj", 1, "model"), ("Edge (book − proj)", "edge", -1, "model"),
    ]
    rows = []
    for lab, col, sign, grp in inputs:
        v = f[col] * sign
        r = dict(label=lab, group=grp, rho=round(spearman(v, f.pts), 4), rho_resid=round(spearman(v, f.resid), 4), auc=round(auc(v, f.over), 4))
        if f[col].nunique() <= 2:
            hi = f[col] == f[col].max()
            a, b = f[hi], f[~hi]
            r.update(binary=True, yes_pts=round(a.pts.mean(), 2), no_pts=round(b.pts.mean(), 2), yes_resid=round(a.resid.mean(), 2),
                     no_resid=round(b.resid.mean(), 2), yes_under=round(a.under.mean(), 4), no_under=round(b.under.mean(), 4), yes_n=int(len(a)))
        else:
            qs = quintiles(v, f)
            r.update(binary=False, low_under=round(1 - qs[0]["over"], 4), high_under=round(1 - qs[-1]["over"], 4),
                     low_pts=qs[0]["pts"], high_pts=qs[-1]["pts"], low_resid=qs[0]["resid"], high_resid=qs[-1]["resid"])
        rows.append(r)
    res["inputs"] = rows

    # 2. schedule spots in detail: points, market miss, under rate (this is the fatigue agent's report card)
    spots = [("All team-games", f.index == f.index), ("Back-to-back", f.b2b == 1), ("Back-to-back, away", (f.b2b == 1) & (f.home == 0)),
             ("Back-to-back, home", (f.b2b == 1) & (f.home == 1)), ("3rd game in 4 days", f.g4 >= 3), ("4th+ game in 7 days", f.g7 >= 4),
             ("Road game 3+ of a trip", f.trip >= 3), ("Road game 4+ of a trip", f.trip >= 4), ("Travelled 2,000+ km", f.km >= 2000),
             ("Crossed 2+ time zones", f.tz >= 2), ("Visiting Denver / Utah", f.alt == 1), ("3+ days rest", f.rest >= 3),
             ("Rested (2+ days) vs tired opp (B2B)", (f.rest >= 2) & (f.opp_b2b == 1)), ("Tired (B2B) vs rested opp (2+ days)", (f.b2b == 1) & (f.opp_rest >= 2)),
             ("Fatigue index 30+", f.fatigue >= 30), ("Fatigue index 50+", f.fatigue >= 50), ("Fatigue 30+ and opp rested", (f.fatigue >= 30) & (f.opp_fatigue < 10)),
             ("Missing 15%+ of scoring", f.lost >= 0.15), ("Missing 25%+ of scoring", f.lost >= 0.25), ("Opp missing 15%+ of scoring", f.opp_lost >= 0.15),
             ("Went to overtime", f.ot > 0),
             ("Matchup OVER: top-10 O vs bottom-10 D", (f.matchup == "OVER") & (f.n_cur >= 10)),
             ("Matchup OVER + fast pace", (f.matchup == "OVER") & (f.pace_tag == "fast") & (f.n_cur >= 10)),
             ("Matchup UNDER: bottom-10 O vs top-10 D", (f.matchup == "UNDER") & (f.n_cur >= 10)),
             ("Matchup UNDER + slow pace", (f.matchup == "UNDER") & (f.pace_tag == "slow") & (f.n_cur >= 10)),
             ("Matchup STRENGTH: top-10 O vs top-10 D", (f.matchup == "STRENGTH") & (f.n_cur >= 10)),
             ("Matchup WEAK: bottom-10 O vs bottom-10 D", (f.matchup == "WEAK") & (f.n_cur >= 10))]
    res["spots"] = []
    for lab, m in spots:
        d = f[m]
        if len(d) < 30:
            continue
        res["spots"].append(dict(label=lab, n=int(len(d)), share=round(len(d) / len(f), 4), pts=round(d.pts.mean(), 2), implied=round(d.implied.mean(), 2),
                                 resid=round(d.resid.mean(), 2), under=round(d.under.mean(), 4), under_lo=round(float(wilson_lo(d.under.mean(), len(d))), 4),
                                 game_under=round(float(((d.pts + d.pa) < d.total).mean()), 4), pace=round(d.poss.mean(), 2)))

    # 3. composites vs the market
    comps = [("Board score (current weights)", "score_default"), ("Projected points", "proj")]
    res["composites"] = [dict(label=lab, auc=round(auc(f[col], f.over), 4), auc_lg=round(auc(f[col], f.over_lg), 4), rho=round(spearman(f[col], f.pts), 4),
                              rho_resid=round(spearman(f[col], f.resid), 4), quintiles=quintiles(f[col], f)) for lab, col in comps]
    q = pd.qcut(f.score_default.rank(method="first"), 10, labels=False)
    d = f.assign(q=q).groupby("q").agg(pts=("pts", "mean"), resid=("resid", "mean"), under=("under", "mean"), lo=("score_default", "min"), hi=("score_default", "max"))
    res["deciles"] = [dict(d=int(i) + 1, pts=round(r.pts, 2), resid=round(r.resid, 2), under=round(r.under, 4), lo=round(r.lo, 1), hi=round(r.hi, 1)) for i, r in d.iterrows()]
    bands = pd.cut(f.score_default, [-1, 30, 40, 50, 60, 70, 101], labels=["<30", "30-40", "40-50", "50-60", "60-70", "70+"])
    b = f.groupby(bands, observed=False).agg(pts=("pts", "mean"), resid=("resid", "mean"), under=("under", "mean"), n=("pts", "size"))
    res["bands"] = [dict(band=str(i), pts=round(r.pts, 2), resid=round(r.resid, 2), under=round(r.under, 4), n=int(r.n)) for i, r in b.iterrows()]

    # 4. the edge table: when the projection sits below the book line, does the under hit?
    eb = pd.cut(f.edge, [-99, -6, -4, -2, 0, 2, 4, 6, 99], labels=["≤−6", "−6…−4", "−4…−2", "−2…0", "0…2", "2…4", "4…6", "6+"])
    e = f.groupby(eb, observed=False).agg(n=("pts", "size"), under=("under", "mean"), resid=("resid", "mean"), pts=("pts", "mean"), proj=("proj", "mean"), implied=("implied", "mean"))
    res["edge_table"] = [dict(edge=str(i), n=int(r.n), under=round(r.under, 4), under_lo=round(float(wilson_lo(r.under, r.n)), 4), resid=round(r.resid, 2),
                              pts=round(r.pts, 2), proj=round(r.proj, 2), implied=round(r.implied, 2)) for i, r in e.iterrows() if r.n >= 30]
    # game totals: both teams' projections vs the closing total
    gm = f[f.home == 1].merge(f[f.home == 0][["gameId", "proj", "score_default", "fatigue", "lost"]].rename(columns={"proj": "proj_a", "score_default": "score_a", "fatigue": "fat_a", "lost": "lost_a"}), on="gameId")
    gm["proj_total"] = 2 * gm.lg_pts + P["gameShrink"] * (gm.proj + gm.proj_a - 2 * gm.lg_pts)
    gm["gedge"] = gm.total - gm.proj_total
    gm["gunder"] = ((gm.pts + gm.pa) < gm.total).astype(int)
    gm["combined"] = gm.score_default + gm.score_a
    gb = pd.cut(gm.gedge, [-99, -8, -5, -2.5, 0, 2.5, 5, 8, 99], labels=["≤−8", "−8…−5", "−5…−2.5", "−2.5…0", "0…2.5", "2.5…5", "5…8", "8+"])
    e2 = gm.groupby(gb, observed=False).agg(n=("pts", "size"), under=("gunder", "mean"), total=("total", "mean"), proj=("proj_total", "mean"), actual=("pts", lambda s: 0))
    e2["actual"] = gm.groupby(gb, observed=False).apply(lambda d: (d.pts + d.pa).mean(), include_groups=False)
    res["game_edge_table"] = [dict(edge=str(i), n=int(r.n), under=round(r.under, 4), under_lo=round(float(wilson_lo(r.under, r.n)), 4), total=round(r.total, 2),
                                   proj=round(r.proj, 2), actual=round(r.actual, 2)) for i, r in e2.iterrows() if r.n >= 30]
    res["game_n"] = int(len(gm))
    res["game_base_under"] = round(float(gm.gunder.mean()), 4)
    res["game_sd"] = round(float((gm.pts + gm.pa - gm.proj_total).std()), 2)
    res["team_sd"] = round(float((f.pts - f.proj).std()), 2)
    res["proj_bias"] = round(float((f.pts - f.proj).mean()), 2)
    res["game_rho_resid"] = round(spearman(-gm.gedge, (gm.pts + gm.pa) - gm.total), 4)

    # 5. measured effects to feed model.json: home, fatigue, availability (points relative to a no-spot projection)
    base = f.proj / np.where(f.home == 1, P["home"], P["away"]) / (1 - P["fatigueK"] * f.fatigue / 100) / (1 + P["oppFatigueK"] * f.opp_fatigue / 100) / (1 - P["availK"] * (f.lost.clip(upper=1) - P["availBase"]))
    ratio = f.pts / base
    def slope(x, y):
        m = x.notna() & y.notna()
        if m.sum() < 100 or x[m].std() == 0:
            return None
        return float(np.polyfit(x[m], y[m], 1)[0])
    res["measured"] = dict(home=round(float(ratio[f.home == 1].mean()), 4), away=round(float(ratio[f.home == 0].mean()), 4),
                           fatigueK=round(-slope(f.fatigue / 100, ratio), 4), oppFatigueK=round(slope(f.opp_fatigue / 100, ratio), 4),
                           availK=round(-slope(f.lost, ratio), 4), sdTeam=res["team_sd"], sdGame=res["game_sd"])

    # 6. fitted weights: train on all but the last complete season, test on it (target: over the implied line)
    holdout = max(y for y in SEASONS if y < CURRENT) if any(y < CURRENT for y in SEASONS) else SEASONS[-1]
    train, test = f[f.season < holdout], f[f.season == holdout]
    if len(train) < 1000:
        train = f[f.season <= holdout]

    def fit(df):
        X = df[["s_" + k for k in KEYS]].copy()
        X = X.fillna(X.mean()) / 100
        b = logistic(X.values, df.over_lg.values)
        coef = dict(zip(KEYS, b[1:]))
        pos = {k: max(v, 0) for k, v in coef.items()}
        tot = sum(pos.values()) or 1
        return coef, {k: v / tot for k, v in pos.items()}
    coef, fitted = fit(train)
    res["coef"] = {k: round(v, 3) for k, v in coef.items()}
    res["fitted_w"] = {k: round(v, 3) for k, v in fitted.items()}
    res["per_season_w"] = {label(y): {k: round(v, 3) for k, v in fit(f[f.season == y])[1].items()} for y in SEASONS if (f.season == y).sum() >= 1000}
    rounded = {k: round(v * 20) / 20 for k, v in fitted.items()}
    res["suggested_w"] = rounded
    test = test.copy()
    test["score_fitted"] = board_score(test, rounded)
    res["test"] = dict(season=label(holdout), n=int(len(test)), default_auc=round(auc(test.score_default, test.over_lg), 4),
                       fitted_auc=round(auc(test.score_fitted, test.over_lg), 4), proj_auc=round(auc(test.proj, test.over_lg), 4),
                       default_mkt=round(auc(test.score_default, test.over), 4), fitted_mkt=round(auc(test.score_fitted, test.over), 4),
                       edge_mkt=round(auc(-test.edge, test.over), 4),
                       default_q=quintiles(test.score_default, test), fitted_q=quintiles(test.score_fitted, test))
    res["by_season"] = [dict(season=label(y), n=int(len(s)), board=round(auc(s.score_default, s.over_lg), 4), board_mkt=round(auc(s.score_default, s.over), 4),
                             edge=round(auc(-s.edge, s.over), 4), under=round(s.under.mean(), 4), pts=round(s.pts.mean(), 2), resid=round(s.resid.mean(), 2))
                        for y, s in f.groupby("season") if len(s) >= 50]
    # how much of the projection's spread is real: slope of actual points on projected (1.0 = perfectly scaled)
    m = f.proj.notna()
    res["proj_slope"] = round(float(np.polyfit(f.proj[m], f.pts[m], 1)[0]), 3)
    res["auc_lg_all"] = round(auc(f.score_default, f.over_lg), 4)
    res["auc_mkt_all"] = round(auc(f.score_default, f.over), 4)
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    f.to_csv(HERE / "team_games_scored.csv", index=False)
    print(json.dumps({k: res[k] for k in ["n", "base_under", "base_pts", "base_implied", "sd_resid", "proj_bias", "proj_slope", "team_sd", "game_sd", "measured", "fitted_w", "suggested_w", "auc_lg_all", "auc_mkt_all", "test", "by_season"]}, indent=1, default=float))
    for r in rows:
        print(f'{r["label"]:<30} rho {r["rho"]:+.3f}  vs market {r["rho_resid"]:+.3f}  auc {r["auc"]:.3f}')
    print("\nSpots:")
    for s in res["spots"]:
        print(f'{s["label"]:<38} n={s["n"]:>5}  pts {s["pts"]:6.1f} vs book {s["implied"]:6.1f} (miss {s["resid"]:+.2f})  under {s["under"] * 100:.1f}%  game under {s["game_under"] * 100:.1f}%')
    print("\nEdge table (team totals):")
    for e in res["edge_table"]:
        print(f'  {e["edge"]:<8} n={e["n"]:>5}  under {e["under"] * 100:.1f}% (floor {e["under_lo"] * 100:.1f}%)  proj {e["proj"]:.1f} book {e["implied"]:.1f} actual {e["pts"]:.1f}')
    print("\nEdge table (game totals):")
    for e in res["game_edge_table"]:
        print(f'  {e["edge"]:<8} n={e["n"]:>5}  under {e["under"] * 100:.1f}% (floor {e["under_lo"] * 100:.1f}%)  proj {e["proj"]:.1f} book {e["total"]:.1f} actual {e["actual"]:.1f}')


if __name__ == "__main__":
    main()
