"""Daily accuracy scanner.

Reads the graded log, measures how the board's pregame scores and projections are doing
this season against the book lines, compares that with the backtest, runs health checks
on today's data, and writes:
  monitor/status.json   metrics + any alerts (alerts.py turns alerts into GitHub issues)
  monitor.html          the Accuracy Monitor page
Run:  python monitor/accuracy.py
"""
import csv, json, math
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
MODEL = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
EXP = MODEL["expect"]
SUBS = [("off", "Offense", "Offense rating (ORtg)"), ("def", "Opp defense", "Opp DRtg"), ("pace", "Pace", "Expected possessions"),
        ("fatigue", "Rest", "Fatigue index"), ("oppFatigue", "Opp rest", "Opp fatigue index"), ("avail", "Availability", "Scoring share missing"),
        ("l5", "Pts L5", "Pts L5")]
MIN_RANK_N = 500      # graded team-games before ranking alerts can fire (~early December)
MIN_INPUT_N = 800


# ---------------------------------------------------------------- stats
def auc(scores, ys):
    pairs = sorted(zip(scores, ys))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    npos = sum(y for _, y in pairs)
    nneg = len(pairs) - npos
    if not npos or not nneg:
        return None, None, None
    a = (sum(r for r, (_, y) in zip(ranks, pairs) if y) - npos * (npos + 1) / 2) / (npos * nneg)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)                                   # Hanley-McNeil
    se = math.sqrt((a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg))
    return a, a - 1.96 * se, a + 1.96 * se


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 30:
        return None, None
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx); syy = sum((b - my) ** 2 for b in ry)
    rho = sxy / math.sqrt(sxx * syy) if sxx and syy else 0
    return rho, 1.96 / math.sqrt(len(x) - 3)


def mean(v):
    return sum(v) / len(v) if v else None


def wilson_lo(p, n, z=1.96):
    if not n:
        return None
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / d


# ---------------------------------------------------------------- main
def main():
    now = datetime.now()
    today = date.today().isoformat()
    logs = sorted(HERE.glob("log_*.csv"))
    rows = []
    if logs:
        with open(logs[-1], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    season = logs[-1].stem[4:] if logs else "—"
    num = lambda v: float(v) if v not in ("", None) else None
    g = [r for r in rows if r["actual"] != ""]
    lined = [r for r in g if r["line"] != ""]              # rows that had a posted line at tip-off
    for r in g:
        r["pts"] = int(float(r["actual"]))
        r["y"] = (1 if r["pts"] > float(r["line"]) else 0) if r["line"] != "" else None     # over the book line
    alerts = []

    def alert(key, level, title, body):
        alerts.append(dict(key=key, level=level, title=title, body=body))

    st = dict(season=season, logged=len(rows), graded=len(g), lined=len(lined), today=today, built=now.strftime("%Y-%m-%d %H:%M"),
              model_version=MODEL["version"], weights=MODEL["weights"], expect=EXP)
    if g:
        st["avg_pts"] = mean([r["pts"] for r in g])
        st["avg_proj"] = mean([float(r["proj"]) for r in g])
        st["ot_share"] = mean([1 if r.get("ot") not in ("", "0", None) else 0 for r in g])
    if lined:
        st["under_rate"] = mean([1 - r["y"] for r in lined])
        st["avg_line"] = mean([float(r["line"]) for r in lined])
        a, lo, hi = auc([float(r["score"]) for r in lined], [r["y"] for r in lined])
        st["auc"], st["auc_lo"], st["auc_hi"] = a, lo, hi
        st["auc_edge"] = auc([float(r["proj"]) - float(r["line"]) for r in lined], [r["y"] for r in lined])[0]
        bands = [("Under 30", 0, 30), ("30–40", 30, 40), ("40–50", 40, 50), ("50–60", 50, 60), ("60–70", 60, 70), ("70+", 70, 101)]
        st["bands"] = []
        for lab, a0, a1 in bands:
            b = [r for r in lined if a0 <= float(r["score"]) < a1]
            st["bands"].append(dict(band=lab, n=len(b), under=mean([1 - r["y"] for r in b]), pts=mean([r["pts"] for r in b]),
                                    resid=mean([r["pts"] - float(r["line"]) for r in b])))
        edges = [("≤−4", -99, -4), ("−4…−2", -4, -2), ("−2…0", -2, 0), ("0…2", 0, 2), ("2…4", 2, 4), ("4+", 4, 99)]
        st["edges"] = []
        for lab, a0, a1 in edges:
            b = [r for r in lined if a0 <= float(r["line"]) - float(r["proj"]) < a1]
            st["edges"].append(dict(edge=lab, n=len(b), under=mean([1 - r["y"] for r in b]), under_lo=wilson_lo(mean([1 - r["y"] for r in b]), len(b)) if b else None,
                                    pts=mean([r["pts"] for r in b]), line=mean([float(r["line"]) for r in b])))
        # spots this season
        spots = [("Back-to-back", lambda r: r["b2b"] == "1"), ("3rd game in 4 days", lambda r: r["g4"] not in ("", None) and int(float(r["g4"])) >= 3),
                 ("Fatigue 30+", lambda r: float(r["fatigue"]) >= 30), ("Missing 15%+ of scoring", lambda r: r["lost_share"] != "" and float(r["lost_share"]) >= 0.15),
                 ("Opp fatigue 30+", lambda r: float(r["opp_fatigue"]) >= 30),
                 ("Rested vs opp on back-to-back", lambda r: r["rest"] not in ("", None) and int(float(r["rest"])) >= 2 and r.get("opp_b2b", "") == "1"),
                 ("Pick-em game (spread within 2.5)", lambda r: r["spread"] not in ("", None) and abs(float(r["spread"])) <= 2.5),
                 ("Blowout spread (10+)", lambda r: r["spread"] not in ("", None) and abs(float(r["spread"])) >= 10),
                 ("Matchup OVER (top-10 O vs bottom-10 D)", lambda r: r.get("matchup") == "OVER"),
                 ("Matchup UNDER (bottom-10 O vs top-10 D)", lambda r: r.get("matchup") == "UNDER"),
                 ("Matchup STRENGTH (top-10 O vs top-10 D)", lambda r: r.get("matchup") == "STRENGTH"),
                 ("Matchup WEAK (bottom-10 O vs bottom-10 D)", lambda r: r.get("matchup") == "WEAK")]
        st["spots"] = []
        for lab, fn in spots:
            b = [r for r in lined if fn(r)]
            st["spots"].append(dict(label=lab, n=len(b), under=mean([1 - r["y"] for r in b]), resid=mean([r["pts"] - float(r["line"]) for r in b])))
        bt = {}
        res = ROOT / "backtest" / "results.json"
        if res.exists():
            bt = {i["label"]: i["rho_resid"] for i in json.loads(res.read_text(encoding="utf-8")).get("inputs", [])}
        st["inputs"] = []
        for k, lab, btlab in SUBS:
            pts = [(float(r["s_" + k]), r["pts"] - float(r["line"])) for r in lined if r["s_" + k] != ""]
            rho, ci = spearman([p[0] for p in pts], [p[1] for p in pts])
            st["inputs"].append(dict(key=k, label=lab, n=len(pts), rho=rho, ci=ci, backtest=bt.get(btlab), weight=MODEL["weights"].get(k, 0)))
            if rho is not None and len(pts) >= MIN_INPUT_N and MODEL["weights"].get(k, 0) > 0 and rho + ci < 0:
                alert(f"input-{k}", "action", f"{lab} has stopped predicting points",
                      f"Season to date, {lab} points the wrong way against the book line (correlation {rho:+.3f}, n={len(pts)}) "
                      f"while it carries {MODEL['weights'][k]:.0%} of the score. The backtest had it at {bt.get(btlab, 0):+.3f}. "
                      f"The weekly learner will lower it if the drop holds; check the data source too.")
        red = [r for r in lined if float(r["score"]) < 35]
        green = [r for r in lined if float(r["score"]) >= 65]
        st["red"] = dict(n=len(red), under=mean([1 - r["y"] for r in red]))
        st["green"] = dict(n=len(green), over=mean([r["y"] for r in green]))
        if len(lined) >= MIN_RANK_N and hi is not None and hi < EXP["auc_low"]:
            alert("ranking-drop", "action", "The score is ranking games worse than the backtest",
                  f"Season-to-date ranking accuracy against the book line is {a:.1%} (95% range {lo:.1%}–{hi:.1%}, n={len(lined)}). "
                  f"The backtest's worst season was {EXP['auc_low']:.1%}. Something changed: league scoring, a broken input, "
                  f"or weights that no longer fit. See the Inputs table on the monitor page.")
        recent = sorted(g, key=lambda r: r["date"])[-300:]
        if len(recent) >= 300:
            act = [r["pts"] for r in recent]; pr = [float(r["proj"]) for r in recent]
            diff = mean(act) - mean(pr)
            sd = math.sqrt(sum((x - mean(act)) ** 2 for x in act) / (len(act) - 1))
            z = diff / (sd / math.sqrt(len(act)))
            st["recent_bias"] = dict(n=len(recent), diff=diff, z=z)
            if abs(diff) > 2 and abs(z) > 3:
                alert("projection-bias", "watch", f"Projections are running {'low' if diff > 0 else 'high'} by {abs(diff):.1f} points",
                      f"Over the last {len(recent)} graded team-games, teams averaged {mean(act):.1f} points against {mean(pr):.1f} projected. "
                      f"League scoring may have shifted. The board's league average updates as games are played, so this often fixes itself; "
                      f"if it persists two weeks, the projection shrink in model.json should be refit.")

    # -- daily table, last 14 days
    days = sorted({r["date"] for r in rows})[-14:]
    st["days"] = []
    for d in reversed(days):
        dr = [r for r in rows if r["date"] == d]; dg = [r for r in dr if r["actual"] != ""]; dl = [r for r in dg if r["line"] != ""]
        red = [r for r in dl if float(r["score"]) < 35]
        edge = [r for r in dl if float(r["line"]) - float(r["proj"]) >= 3]
        st["days"].append(dict(date=d, rows=len(dr), graded=len(dg), proj=mean([float(r["proj"]) for r in dr]), pts=mean([r["pts"] for r in dg]),
                               red_n=len(red), red_under=sum(1 for r in red if r["pts"] < float(r["line"])),
                               edge_n=len(edge), edge_under=sum(1 for r in edge if r["pts"] < float(r["line"])),
                               no_line=sum(1 for r in dr if r["line"] == "")))

    # -- health checks on today's data
    dj = ROOT / "data.json"
    if dj.exists():
        D = json.loads(dj.read_text(encoding="utf-8"))
        games_today = len(D.get("schedule", {}).get(today, []))
        st["games_today"] = games_today
        if games_today and D.get("today") != today and now.hour >= 12:
            alert("stale-board", "action", "The board didn't refresh today",
                  f"data.json is from {D.get('today')} but there are {games_today} games today. Check the latest run under the Actions tab.")
        todays = [r for r in rows if r["date"] == today]
        if games_today and not todays and now.hour >= 15:
            alert("log-missing", "action", "Today's games weren't logged",
                  "The board built but no pregame rows were saved, so today can't be graded. Check the 'Build board' step log.")
        if todays and now.hour >= 17:
            share = sum(1 for r in todays if r["line"] == "") / len(todays)
            if share > 0.5:
                alert("lines-missing", "watch", "Posted lines aren't coming through",
                      f"{share:.0%} of today's rows have no book line, so Edge and Under % are blank. ESPN may have changed its odds feed. "
                      f"The Score still works; the projection-vs-book columns don't.")
        if not D.get("injuries") and now.hour >= 12 and games_today:
            alert("injuries-missing", "watch", "The injury report is empty",
                  "No injuries came through from ESPN, so every team is scored at full strength. Check the injuries() step in nbadata.py.")
    stale = [r for r in rows if r["actual"] == "" and r["date"] <= (date.today() - timedelta(days=2)).isoformat()]
    if stale:
        alert("grading-stuck", "action", f"{len(stale)} finished games haven't been graded",
              f"Rows from {min(r['date'] for r in stale)} on still have no final score. The ESPN scoreboard may have changed; check the 'Grade' step.")

    st["alerts"] = alerts
    old = HERE / "status.json"
    if old.exists():                        # keep the old timestamp if nothing else changed
        o = json.loads(old.read_text(encoding="utf-8"))
        if {k: v for k, v in o.items() if k != "built"} == {k: v for k, v in json.loads(json.dumps(st, default=str)).items() if k != "built"}:
            st["built"] = o["built"]
    (HERE / "status.json").write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    tpl = (HERE / "monitor_template.html").read_text(encoding="utf-8")
    learn = ROOT / "backtest" / "learn_status.json"
    st["learn"] = json.loads(learn.read_text(encoding="utf-8")) if learn.exists() else None
    st["history"] = MODEL.get("history", [])
    (ROOT / "monitor.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(st, default=str)), encoding="utf-8")
    print(f"{season}: {len(rows)} logged, {len(g)} graded, auc={st.get('auc')}, alerts={[a['key'] for a in alerts]}")


if __name__ == "__main__":
    main()
