"""Build the NBA Unders Board and log today's pregame rows.

Pulls this season's and last season's game logs (ESPN box scores), the schedule with
rest/travel features, the injury report and the posted lines, blends last season with
the current one, and writes:
  index.html                 the board (template.html + engine.js + data)
  data.json                  the same data, for the monitor
  monitor/log_<season>.csv   today's rows, frozen once each game starts

Run:  python build.py [YYYY-MM-DD]   (the GitHub Action runs it hourly on game days)
"""
import csv, json, subprocess, sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import nbadata as nd

HERE = Path(__file__).parent
MODEL = json.loads((HERE / "model.json").read_text(encoding="utf-8"))
PRIOR_GAMES = MODEL.get("priorGames", 15)   # last season counts like this many current-season games
WINDOW = 7                                   # days of schedule on the board


def season_block(rows):
    """Per-team season totals from team-game rows."""
    out = {}
    for r in rows:
        if int(r["type"]) != 2:
            continue
        t = out.setdefault(r["team"], dict(gp=0, pts=0, pa=0, poss=0, fga=0, fta=0, tpa=0, last=[]))
        t["gp"] += 1
        for k in ("pts", "pa", "poss", "fga", "fta", "tpa"):
            t[k] += float(r[k])
        t["last"].append((r["date"], int(r["pts"])))
    for t in out.values():
        t["last"] = [p for _, p in sorted(t["last"])[-5:]]
    return out


def blend_teams(prior, cur):
    keys = ["pts", "pa", "poss", "fga", "fta", "tpa"]
    teams = {}
    for ab in nd.TEAMS:
        p, c = prior.get(ab, {}), cur.get(ab, {})
        pgp, cgp = p.get("gp", 0), c.get("gp", 0)
        wp = PRIOR_GAMES / pgp if pgp else 0
        g = wp * pgp + cgp
        t = {"ab": ab, "name": nd.TEAM_NAMES[ab], "gpCur": cgp, "gpPrior": pgp}
        for k in keys:
            t[k] = (p.get(k, 0) * wp + c.get(k, 0)) / g if g else 0
        t["pace"] = t["poss"]
        t["ortg"] = 100 * t["pts"] / t["poss"] if t["poss"] else 0
        t["drtg"] = 100 * t["pa"] / t["poss"] if t["poss"] else 0
        t["tpaRate"] = t["tpa"] / t["fga"] if t["fga"] else 0
        t["ftaRate"] = t["fta"] / t["fga"] if t["fga"] else 0
        last = c.get("last", [])
        t["l5"] = sum(last) / len(last) if len(last) >= 3 else None
        t["cur"] = {"gp": cgp, "pts": c.get("pts", 0) / cgp if cgp else None, "pa": c.get("pa", 0) / cgp if cgp else None,
                    "pace": c.get("poss", 0) / cgp if cgp else None}
        t["prior"] = {"gp": pgp, "pts": p.get("pts", 0) / pgp if pgp else None, "pa": p.get("pa", 0) / pgp if pgp else None,
                      "pace": p.get("poss", 0) / pgp if pgp else None}
        teams[ab] = t
    return teams


def player_scoring(prior_p, cur_p, inj, teams):
    """Blended points per game for every player, grouped by current team, for the availability input."""
    agg = {}
    for rows, prior in ((prior_p, True), (cur_p, False)):
        for r in rows:
            if int(r["dnp"]):
                continue
            a = agg.setdefault(r["pid"], dict(name=r["name"], pp=0, pg=0, cp=0, cg=0, team=None, last=""))
            if prior:
                a["pp"] += int(r["pts"]); a["pg"] += 1
            else:
                a["cp"] += int(r["pts"]); a["cg"] += 1
            if r["gameId"] >= a["last"] and not prior:
                a["last"], a["team"] = r["gameId"], r["team"]
            elif prior and a["team"] is None:
                a["team"] = r["team"]
    for ab, lst in inj.items():                     # the injury report says who is on which team today
        for i in lst:
            if i["pid"] in agg:
                agg[i["pid"]]["team"] = ab
    out = defaultdict(list)
    for pid, a in agg.items():
        if not a["team"] or a["team"] not in teams:
            continue
        wp = PRIOR_GAMES / a["pg"] if a["pg"] else 0
        g = wp * a["pg"] + a["cg"]
        ppg = (a["pp"] * wp + a["cp"]) / g if g else 0
        if ppg >= 2:
            out[a["team"]].append(dict(pid=pid, name=a["name"], ppg=round(ppg, 1), gp=a["pg"] + a["cg"],
                                       pts=round(ppg, 1)))
    return out


def schedule_window(sched, today):
    """Board schedule: yesterday through the next WINDOW days, keyed by date."""
    lo, hi = (today - timedelta(days=1)).isoformat(), (today + timedelta(days=WINDOW)).isoformat()
    days = defaultdict(list)
    for g in sched:
        if lo <= g["date"] <= hi:
            days[g["date"]].append({k: g.get(k) for k in ("id", "date", "start", "type", "away", "home", "awayScore", "homeScore",
                                                            "state", "ot", "neutral", "city", "total", "spread", "book")})
    return dict(days)


def main():
    today = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    cur_y = nd.season_of(today)
    prior_y = cur_y - 1
    prior_rows = nd.team_games(prior_y, refresh=not (nd.DATA / f"games_{prior_y}.csv").exists())
    cur_rows = nd.team_games(cur_y, refresh=True)
    teams = blend_teams(season_block(prior_rows), season_block(cur_rows))
    sched = nd.schedule(cur_y)
    inj = nd.injuries()
    players = player_scoring(nd._read_csv(nd.DATA / f"players_{prior_y}.csv"), nd._read_csv(nd.DATA / f"players_{cur_y}.csv"), inj, teams)
    feats = nd.schedule_features(sched)
    window = schedule_window(sched, today)
    features = {}
    for d, games in window.items():
        for g in games:
            features[g["id"]] = {ab: feats.get((g["id"], ab)) for ab in (g["away"], g["home"])}
    data = dict(built=datetime.now().strftime("%Y-%m-%d %H:%M"), today=today.isoformat(),
                seasons=dict(prior=nd.season_label(prior_y), cur=nd.season_label(cur_y)),
                priorGames=PRIOR_GAMES, teams=teams, schedule=window, features=features,
                injuries=inj, players=players, model=MODEL)
    status = HERE / "monitor" / "status.json"
    if status.exists():
        data["monitor"] = {k: v for k, v in json.loads(status.read_text(encoding="utf-8")).items() if k in ("auc", "graded", "under_implied")}
    prev = HERE / "data.json"
    if prev.exists():                       # keep the old timestamp if nothing else changed (no empty commits)
        old = json.loads(prev.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "built"} == {k: v for k, v in json.loads(json.dumps(data)).items() if k != "built"}:
            data["built"] = old["built"]
    if sum(1 for t in teams.values() if t["poss"] > 0) < 30 or not sched:
        sys.exit(f"Refusing to publish: {sum(1 for t in teams.values() if t['poss'] > 0)} teams with data, {len(sched)} scheduled games")
    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    engine = (HERE / "engine.js").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"))
    html = tpl.replace("/*__ENGINE__*/", engine).replace("/*__DATA__*/null", payload)
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>')
    (HERE / "index.html").write_text(head + html + "</body></html>", encoding="utf-8")
    (HERE / "data.json").write_text(payload, encoding="utf-8")
    n_games = sum(len(v) for v in window.values())
    print(f"Built index.html: {len(teams)} teams ({sum(t['gpCur'] for t in teams.values()) // 2} games this season), "
          f"{n_games} games on the board, injuries for {len(inj)} teams")
    log_today(data)


LOG_FIELDS = ["date", "gameId", "start", "team", "opp", "ha", "s_off", "s_def", "s_pace", "s_fatigue", "s_oppFatigue", "s_avail", "s_l5",
              "score", "proj", "line", "game_total", "spread", "p_under", "fatigue", "opp_fatigue", "rest", "b2b", "g4", "trip", "km",
              "lost_share", "out", "opp_b2b", "rk_off", "rk_oppdef", "rk_pace", "rk_opppace", "matchup", "pace_tag",
              "model_version", "logged_at", "actual", "ot", "graded_at"]


def log_path(data):
    return HERE / "monitor" / f"log_{data['seasons']['cur']}.csv"


def log_today(data):
    """Upsert today's rows. A game's row keeps updating until tip-off, then never changes."""
    try:
        out = subprocess.run(["node", str(HERE / "monitor" / "snapshot.js"), str(HERE / "data.json")],
                             capture_output=True, text=True, check=True).stdout
    except Exception as e:
        print(f"Snapshot skipped: {e}")
        return
    rows = json.loads(out)
    path = log_path(data)
    existing = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    by_key = {(r["gameId"], r["team"]): r for r in existing}
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    changed = 0
    for r in rows:
        key = (str(r["gameId"]), r["team"])
        if r.pop("state") != "FUT":
            continue                                   # started or finished: frozen (or never logged)
        r.update(gameId=str(r["gameId"]), logged_at=stamp, actual="", ot="", graded_at="")
        by_key[key] = {k: r.get(k, "") for k in LOG_FIELDS}
        changed += 1
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        for r in sorted(by_key.values(), key=lambda r: (r["date"], r["gameId"], r["team"])):
            w.writerow({k: r.get(k, "") for k in LOG_FIELDS})
    print(f"Logged {changed} pregame rows to {path.name} ({len(by_key)} total)")


if __name__ == "__main__":
    main()
