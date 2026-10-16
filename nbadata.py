"""Everything that talks to ESPN, plus the schedule math the fatigue score is built on.

Used by build.py (the live board), the backtest, and the monitor, so every part of the
system sees the same numbers. All data is free and public:

  scoreboard  one call per month: schedule, results, quarters/OT, venue
  summary     one call per finished game: team box score (possessions) and player minutes
  odds        one call per game: the closing total and spread (ESPN BET / DraftKings)
  injuries    one call: every team's current injury report

Team-games are cached in backtest/data/games_<season>.csv and players_<season>.csv so a
season is only downloaded once; the current season refreshes incrementally.
"""
import csv, gzip, json, math, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "backtest" / "data"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36",
      "Accept-Encoding": "gzip"}
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
BOOKS = ("DraftKings", "ESPN BET", "Caesars Sportsbook", "consensus", "MGM", "PointsBet", "Unibet", "BetfairSportsbook")

# Home arenas: ESPN abbreviation -> (lat, lon, UTC offset in winter, altitude in feet)
ARENAS = {
    "ATL": (33.757, -84.396, -5, 1050), "BOS": (42.366, -71.062, -5, 20), "BKN": (40.683, -73.975, -5, 30),
    "CHA": (35.225, -80.839, -5, 750), "CHI": (41.881, -87.674, -6, 600), "CLE": (41.496, -81.688, -5, 650),
    "DAL": (32.790, -96.810, -6, 430), "DEN": (39.749, -105.008, -7, 5280), "DET": (42.341, -83.055, -5, 600),
    "GS": (37.768, -122.388, -8, 10), "HOU": (29.751, -95.362, -6, 50), "IND": (39.764, -86.156, -5, 715),
    "LAC": (33.945, -118.342, -8, 100), "LAL": (34.043, -118.267, -8, 250), "MEM": (35.138, -90.051, -6, 340),
    "MIA": (25.781, -80.187, -5, 10), "MIL": (43.045, -87.917, -6, 600), "MIN": (44.979, -93.276, -6, 830),
    "NO": (29.949, -90.082, -6, 5), "NY": (40.751, -73.994, -5, 30), "OKC": (35.463, -97.515, -6, 1200),
    "ORL": (28.539, -81.384, -5, 100), "PHI": (39.901, -75.172, -5, 40), "PHX": (33.446, -112.071, -7, 1100),
    "POR": (45.532, -122.667, -8, 50), "SAC": (38.580, -121.500, -8, 30), "SA": (29.427, -98.438, -6, 650),
    "TOR": (43.643, -79.379, -5, 250), "UTAH": (40.768, -111.901, -7, 4300), "WSH": (38.898, -77.021, -5, 30),
}
NEUTRAL = {"Mexico City": (19.43, -99.13, -6, 7350), "Paris": (48.86, 2.35, 1, 115), "London": (51.50, -0.12, 0, 35),
           "Las Vegas": (36.09, -115.18, -8, 2000), "Abu Dhabi": (24.45, 54.65, 4, 20), "Seattle": (47.62, -122.35, -8, 50)}
ALTITUDE_FT = 4000      # Denver and Utah: thin air counts against the visitor
TEAMS = sorted(ARENAS)


# ---------------------------------------------------------------- fetch
def get(url, tries=3, timeout=30):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
            return json.loads(body.decode("utf-8"))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def season_of(d):
    """Season start year: 2026 for the 2026-27 season (opens in October)."""
    return d.year if d.month >= 9 else d.year - 1


def season_label(y):
    return f"{y}-{str(y + 1)[2:]}"


def _month_events(yyyymm):
    try:
        return get(f"{SITE}/scoreboard?dates={yyyymm}&limit=1000").get("events", [])
    except Exception:
        return []


def _event_row(e):
    """One schedule row per game from a scoreboard event (regular season and playoffs)."""
    c = e["competitions"][0]
    comp = {t["homeAway"]: t for t in c["competitors"]}
    if "home" not in comp or "away" not in comp:
        return None
    h, a = comp["home"], comp["away"]
    if a["team"].get("abbreviation") not in ARENAS or h["team"].get("abbreviation") not in ARENAS:
        return None                                    # All-Star, Rising Stars, exhibition teams
    st = e["status"]["type"]
    utc = datetime.strptime(e["date"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    local = utc.astimezone(timezone(timedelta(hours=-5)))        # game date = Eastern date
    periods = e["status"].get("period") or 0
    venue = (c.get("venue") or {}).get("address", {}) or {}
    row = dict(id=e["id"], date=local.strftime("%Y-%m-%d"), start=e["date"], type=e["season"]["type"],
               season=season_of(local.date()), away=a["team"]["abbreviation"], home=h["team"]["abbreviation"],
               awayScore=int(a["score"]) if st["completed"] and a.get("score") else None,
               homeScore=int(h["score"]) if st["completed"] and h.get("score") else None,
               state="FINAL" if st["completed"] else ("LIVE" if st["state"] == "in" else "FUT"),
               ot=max(0, periods - 4) if st["completed"] else 0,
               neutral=bool(c.get("neutralSite")), city=venue.get("city", ""))
    odds = (c.get("odds") or [{}])[0]
    if odds.get("overUnder") is not None:
        row["total"] = odds["overUnder"]
        row["spread"] = odds.get("spread")
        row["book"] = (odds.get("provider") or {}).get("name", "")
    return row


def schedule(season):
    """Every game of a season (Oct-Jun), one row each, from monthly scoreboard calls."""
    months = [f"{season}{m:02d}" for m in range(10, 13)] + [f"{season + 1}{m:02d}" for m in range(1, 7)]
    rows = {}
    for m in months:
        for e in _month_events(m):
            r = _event_row(e)
            if r and r["season"] == season and r["type"] in (2, 3):
                rows[r["id"]] = r
    return sorted(rows.values(), key=lambda r: (r["start"], r["id"]))


def box_score(game_id):
    """Team totals and player minutes for one finished game, or None."""
    try:
        d = get(f"{SITE}/summary?event={game_id}")
    except Exception:
        return None
    bs = d.get("boxscore", {})
    if not bs.get("teams") or len(bs["teams"]) != 2:
        return None
    teams = {}
    for t in bs["teams"]:
        s = {x["name"]: x["displayValue"] for x in t.get("statistics", [])}
        try:
            fgm, fga = map(int, s["fieldGoalsMade-fieldGoalsAttempted"].split("-"))
            ftm, fta = map(int, s["freeThrowsMade-freeThrowsAttempted"].split("-"))
            tpm, tpa = map(int, s["threePointFieldGoalsMade-threePointFieldGoalsAttempted"].split("-"))
            teams[t["team"]["abbreviation"]] = dict(fga=fga, fgm=fgm, fta=fta, ftm=ftm, tpa=tpa, tpm=tpm,
                                                    orb=int(s.get("offensiveRebounds", 0)), tov=int(s.get("totalTurnovers", s.get("turnovers", 0))))
        except (KeyError, ValueError):
            return None
    players = []
    for p in bs.get("players", []):
        ab = p["team"]["abbreviation"]
        for grp in p.get("statistics", []):
            names = grp.get("names", [])
            for a in grp.get("athletes", []):
                st = dict(zip(names, a.get("stats", [])))
                mins = st.get("MIN")
                ath = a.get("athlete") or {}
                if not ath.get("id"):
                    continue
                players.append(dict(team=ab, pid=ath["id"], name=ath.get("displayName", "?"),
                                    starter=int(bool(a.get("starter"))), dnp=int(bool(a.get("didNotPlay"))),
                                    min=int(mins) if mins and mins.isdigit() else 0,
                                    pts=int(st["PTS"]) if st.get("PTS", "").isdigit() else 0))
    return dict(teams=teams, players=players)


def closing_line(game_id):
    """Closing total and spread (home negative when favored) from the best available book."""
    try:
        items = get(f"{CORE}/events/{game_id}/competitions/{game_id}/odds").get("items", [])
    except Exception:
        return None
    items = [i for i in items if i.get("overUnder") and "Live" not in i["provider"]["name"]]
    if not items:
        return None
    rank = {b: n for n, b in enumerate(BOOKS)}
    items.sort(key=lambda i: rank.get(i["provider"]["name"], 99))
    i = items[0]
    close = ((i.get("close") or {}).get("total") or {}).get("american")
    try:
        total = float(close) if close else float(i["overUnder"])
    except (TypeError, ValueError):
        total = float(i["overUnder"])
    return dict(total=total, spread=i.get("spread"), book=i["provider"]["name"])


def injuries():
    """Current injury report: {team: [{name, status, detail, date}]} (Out / Doubtful / Questionable / Day-To-Day)."""
    out = {}
    try:
        d = get(f"{SITE}/injuries")
    except Exception:
        return out
    for t in d.get("injuries", []):
        ab = (t.get("abbreviation") or "").upper()
        if not ab:
            for team in TEAMS:
                if t.get("displayName", "").endswith(TEAM_NAMES.get(team, "?")):
                    ab = team
        rows = []
        for i in t.get("injuries", []):
            det = i.get("details") or {}
            ath = i.get("athlete") or {}
            pid = ath.get("id") or next((l["href"].split("/id/")[1].split("/")[0] for l in ath.get("links", []) if "/id/" in l.get("href", "")), "")
            rows.append(dict(pid=str(pid), name=ath.get("displayName", "?"), status=i.get("status", ""),
                             detail=det.get("type", ""), ret=det.get("returnDate", ""), date=(i.get("date") or "")[:10],
                             note=(i.get("shortComment") or "")[:160]))
        out[ab] = rows
    return out


TEAM_NAMES = {"ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets",
              "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
              "DET": "Detroit Pistons", "GS": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
              "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
              "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NO": "New Orleans Pelicans", "NY": "New York Knicks",
              "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
              "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SA": "San Antonio Spurs", "TOR": "Toronto Raptors",
              "UTAH": "Utah Jazz", "WSH": "Washington Wizards"}


# ---------------------------------------------------------------- team-game cache
GAME_FIELDS = ["gameId", "season", "date", "start", "type", "team", "opp", "home", "neutral", "city", "pts", "pa", "ot",
               "fga", "fgm", "fta", "ftm", "tpa", "tpm", "orb", "tov", "ofga", "ofta", "oorb", "otov", "poss", "total", "spread", "book"]
PLAYER_FIELDS = ["gameId", "team", "pid", "name", "starter", "dnp", "min", "pts"]


def possessions(t, o):
    """Average of both teams' possession estimates (the standard box-score formula)."""
    mine = t["fga"] - t["orb"] + t["tov"] + 0.44 * t["fta"]
    theirs = o["fga"] - o["orb"] + o["tov"] + 0.44 * o["fta"]
    return round((mine + theirs) / 2, 2)


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def team_games(season, refresh=True, workers=8, log=print):
    """Two rows (one per team) for every finished game of a season, with box score and closing line.

    Cached in backtest/data. With refresh=True the schedule is re-pulled and any finished game
    not yet cached is fetched, so the current season grows a day at a time. Returns the rows.
    """
    gpath, ppath = DATA / f"games_{season}.csv", DATA / f"players_{season}.csv"
    rows, players = _read_csv(gpath), _read_csv(ppath)
    have = {r["gameId"] for r in rows}
    if not refresh and rows:
        return rows
    sched = schedule(season)
    done = [g for g in sched if g["state"] == "FINAL" and g["id"] not in have
            and g["awayScore"] is not None and g["type"] == 2]
    if not done:
        log(f"{season_label(season)}: {len(rows) // 2} games cached, nothing new")
        return rows

    def fetch(g):
        return g, box_score(g["id"]), closing_line(g["id"])
    added = 0
    with ThreadPoolExecutor(workers) as ex:
        for g, bs, line in ex.map(fetch, done):
            if not bs or g["away"] not in bs["teams"] or g["home"] not in bs["teams"]:
                continue
            line = line or {}
            a, h = bs["teams"][g["away"]], bs["teams"][g["home"]]
            poss = possessions(a, h)
            for side, me, them, pts, pa in (("away", a, h, g["awayScore"], g["homeScore"]), ("home", h, a, g["homeScore"], g["awayScore"])):
                ab, opp = (g["away"], g["home"]) if side == "away" else (g["home"], g["away"])
                rows.append(dict(gameId=g["id"], season=season, date=g["date"], start=g["start"], type=g["type"], team=ab, opp=opp,
                                 home=int(side == "home"), neutral=int(g["neutral"]), city=g["city"], pts=pts, pa=pa, ot=g["ot"],
                                 fga=me["fga"], fgm=me["fgm"], fta=me["fta"], ftm=me["ftm"], tpa=me["tpa"], tpm=me["tpm"], orb=me["orb"], tov=me["tov"],
                                 ofga=them["fga"], ofta=them["fta"], oorb=them["orb"], otov=them["tov"], poss=poss,
                                 total=line.get("total") if line else g.get("total"),
                                 spread=line.get("spread") if line else g.get("spread"), book=line.get("book") if line else g.get("book", "")))
            for p in bs["players"]:
                players.append(dict(gameId=g["id"], **p))
            added += 1
    rows.sort(key=lambda r: (r["date"], r["gameId"], -int(r["home"])))
    _write_csv(gpath, rows, GAME_FIELDS)
    _write_csv(ppath, players, PLAYER_FIELDS)
    log(f"{season_label(season)}: added {added} games ({len(rows) // 2} cached)")
    return rows


# ---------------------------------------------------------------- schedule features (the fatigue inputs)
def _km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def venue_of(game):
    """(lat, lon, utc offset, altitude) for where a game is played."""
    if game.get("neutral") and game.get("city") in NEUTRAL:
        return NEUTRAL[game["city"]]
    return ARENAS.get(game["home"], ARENAS["NY"])


def schedule_features(games):
    """Rest and travel for every team in every game, from the schedule alone (no look-ahead).

    games: schedule rows (dicts with id, date, away, home, neutral, city), any order, one season.
    Returns {(gameId, team): features}:
      rest        days since the team's previous game (7 at season start, capped at 7)
      b2b         1 if it played yesterday
      g4          games in the last 4 days including this one (3 = a 3-in-4)
      g7          games in the last 7 days including this one
      trip        consecutive road games including this one (0 at home)
      km          distance from the previous game's arena
      tz          time zones crossed since the previous game (absolute)
      alt         1 if the visitor plays at altitude (Denver, Utah, Mexico City)
      ha          'Home' or 'Away'
    """
    by_team = {}
    for g in games:
        for ab in (g["away"], g["home"]):
            by_team.setdefault(ab, []).append(g)
    out = {}
    for ab, gs in by_team.items():
        gs.sort(key=lambda g: (g["date"], g["id"]))
        dates, prev_venue, trip = [], None, 0
        for g in gs:
            d = date.fromisoformat(g["date"])
            home = g["home"] == ab
            v = venue_of(g)
            rest = min(7, (d - dates[-1]).days) if dates else 7
            g4 = 1 + sum(1 for x in dates if (d - x).days <= 3)
            g7 = 1 + sum(1 for x in dates if (d - x).days <= 6)
            trip = 0 if (home and not g.get("neutral")) else trip + 1
            km = _km(prev_venue, v) if prev_venue else 0
            tz = abs(v[2] - prev_venue[2]) if prev_venue else 0
            out[(g["id"], ab)] = dict(rest=rest, b2b=int(rest == 1), g4=g4, g7=g7, trip=trip, km=round(km),
                                      tz=tz, alt=int(not home and v[3] >= ALTITUDE_FT), ha="Home" if home else "Away")
            dates.append(d)
            prev_venue = v
    return out


def fatigue_points(f, P):
    """Fatigue index 0-100 from the schedule features, using the point values in model.json.
    Higher = more tired. Exactly mirrored by engine.js (fatigueIndex)."""
    pts = 0
    if f["b2b"]:
        pts += P["b2b"]
    if f["g4"] >= 3:
        pts += P["threeInFour"]
    if f["g7"] >= 4:
        pts += P["fourInSeven"]
    if f["trip"] >= 3:
        pts += P["tripGame"] * min(f["trip"] - 2, 3)
    pts += P["travel"] * min(f["km"], 3000) / 1000
    pts += P["tz"] * min(f["tz"], 3)
    if f["alt"]:
        pts += P["altitude"]
    if f["rest"] >= 3 and f.get("ha") == "Home":
        pts -= P["restBonus"]
    return max(0, min(100, pts))


if __name__ == "__main__":
    import sys
    y = int(sys.argv[1]) if len(sys.argv) > 1 else season_of(date.today())
    rows = team_games(y)
    print(len(rows), "team-game rows for", season_label(y))
