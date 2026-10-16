# NBA Unders Board

Scores every NBA team total on the slate: one team's offense against tonight's opponent,
the pace both teams play at, who is missing, and, the part the other boards don't have,
how tired the team is. Green = good spot to score, red = under spot. Each row also carries a
points projection, the book's implied team total, the edge between them, and a fair price.
The board logs its own pregame scores, grades them against the closing line, checks its
accuracy every day, re-runs a multi-season backtest every week, and adjusts its weights
when the data says so.

Same skeleton as the NHL Unders Board and the MLB Matchup Scores board.

| Page | What it shows |
|---|---|
| `index.html` — the board | Today's slate, sortable, with spot tags, edge vs the book, weight sliders, and tabs for game totals, team ratings, the schedule/fatigue table and the injury report |
| `monitor.html` — accuracy monitor | How this season's pregame scores and edges are doing against the book, data health, model history |
| `backtest.html` — backtest | Four seasons replayed game by game: what the schedule does to scoring, whether the projection beats the closing line, which inputs matter |

## Where the edge is supposed to come from

A full-game NBA total is the sharpest number in sports: season ratings alone will match the
book, not beat it. So the board is built around the two things the number moves on late:

1. **The schedule.** Rest, back-to-backs, 3-in-4s, road-trip length, travel distance, time
   zones and altitude are known weeks in advance and fully backtestable with no look-ahead.
   A tired team scores less *and* plays slower, which is why the under side gets structural
   help. The `fatigue index` (0–100) adds these up; its point values live in `model.json`.
2. **Who is missing.** The injury report is refreshed hourly. Each listed player's share of
   his team's scoring comes off the projection (Out in full, Doubtful ¾, Questionable ⅖,
   Day-To-Day ¼). On the Injuries tab you can untick a player you know will play.

Team totals rather than game totals because the fatigue signal attaches to a team, not a
game; the Game totals tab simply adds the two projections and compares them to the posted
total.

## What runs on its own

**Hourly, 9am–midnight ET** (`.github/workflows/daily.yml`)
1. `build.py` pulls the latest box scores, injury report and lines and rebuilds the board.
2. It saves each of today's games to `monitor/log_<season>.csv`. A row keeps updating
   until tip-off (so it picks up late scratches and line moves), then never changes.
3. `monitor/grade.py` fills in final points for finished games (overtime included, like
   the books).
4. `monitor/accuracy.py` measures the season so far against the book lines and runs
   health checks.
5. `monitor/alerts.py` opens or closes GitHub issues.

**Every Monday ~7am ET** (`.github/workflows/weekly.yml`)
1. `backtest/backtest.py` replays every game from 2022-23 through yesterday using only
   data available before each game.
2. `backtest/rules.py` reports hit rates for score and edge thresholds.
3. `backtest/learn.py` refits the weights and changes `model.json` only if the
   guardrails below pass.
4. `backtest/report.py` rebuilds `backtest.html`, and the board is rebuilt with the
   current model.

## Where the data comes from
| Data | Source | Updated |
|---|---|---|
| Schedule, scores, quarters/OT, venue | ESPN scoreboard (`site.api.espn.com`, one call per month) | Live |
| Possessions, player minutes and points | ESPN game summary (box score, one call per finished game) | After each game |
| Posted game total and spread, closing line for the backtest | ESPN odds (DraftKings / ESPN BET) | Through the day |
| Injury report | ESPN injuries | Through the day |

All free and public, no keys. Finished games are cached in `backtest/data/games_<season>.csv`
and `players_<season>.csv` (committed, so a season is only downloaded once). Arena
coordinates, time zones and altitude are a table in `nbadata.py`.

## How the score works
Each input is scaled 0–100 across the league (high = favors scoring for the team in the row)
and combined with the weights in `model.json`:

| Sub-score | Input | Direction |
|---|---|---|
| Offense | Points per 100 possessions, last season blended with this one (last season counts as `priorGames` games) | High = good |
| Opp D | Points per 100 the opponent allows | Leaky = good |
| Pace | Expected possessions: both teams' pace against the league average | Fast = good |
| Rest | 100 − this team's fatigue index | Rested = good |
| Opp rest | The opponent's fatigue index | Tired opponent = good |
| Availability | Share of this team's scoring on the injury report | Healthy = good |
| Pts L5 | Points per game, last 5 | Shown when weighted |

**Projection**: expected possessions × offense × opponent defense × home/away × fatigue ×
opponent fatigue × availability, pulled toward the league average. **Book line** is the
team total implied by the posted game total and spread. **Edge** = book line − projection.
**Under %** treats points as normal around the projection (spread `sdTeam` in `model.json`,
fit in the backtest) and shows the fair American price. Break-even at −110 is 52.4%.

## How the model learns (guardrails)
The learner refits the weights weekly on the last three seasons plus this one (this season
counts double), then tests on the last 6 weeks of games it didn't fit on. It changes the
board only when **all** of these hold:
- 500+ team-games played this season (about early December)
- 28+ days since the last change
- the new weights rank games against the book better (by 0.3+ points of ranking accuracy)
- and do no worse on last season
- each input moves at most 10 points per change, in 5% steps

Every change is a commit to `model.json` with the reason in its `history`, and opens an
issue. To undo one, revert that commit or edit the weights back.

## Alerts
Issues labeled `nba-alert` (GitHub emails the repo owner; you're @mentioned):

| Alert | When |
|---|---|
| Board didn't refresh / games not logged / games not graded | Data pipeline broke |
| Lines aren't coming through | >50% of today's rows have no book line by 5pm |
| Injury report empty | Nothing came back from ESPN on a game day |
| Backtest data stale | The cached game log is 4+ days behind in season |
| Score ranking worse than backtest | 500+ graded games and the 95% range sits below the worst backtest season |
| An input stopped predicting | 800+ games and a weighted input points the wrong way against the book |
| Projections running high/low | Last 300 games off by 2+ points, beyond noise |
| Home/rest/availability effects drifted, or a large weight shift | From the weekly learner — suggestions for you to review |
| Model updated | The learner changed the weights (stays open until you close it) |

## Known limits
- The backtest's availability input uses who actually sat (regulars who didn't play) as a
  stand-in for the pregame injury report, so it is a little more informed than the live
  board can be. The live board only knows what ESPN lists.
- Lines are ESPN's posted game total and spread; the implied team total is a derivation,
  not a book's team-total price. Compare against your own book before betting.
- The model doesn't know about blowouts, garbage time, or who the backup rotation is.

## Run by hand
```
python build.py                 # board + today's log (fetches new box scores first)
python monitor/grade.py         # grade finished games
python monitor/accuracy.py      # monitor page + alerts (status.json)
python backtest/backtest.py     # full backtest (first run downloads ~5 seasons, ~15 min)
python backtest/rules.py        # threshold rules
python backtest/learn.py        # learner (may edit model.json)
python backtest/report.py       # backtest page
```
Or use **Actions → Run workflow** on GitHub. The backtest needs `pip install pandas numpy`;
everything else is the Python standard library plus Node for the daily log.

## Files
- `model.json` — the live weights, fatigue points, projection settings, and change history
- `engine.js` — the scoring engine, shared by the board and the daily log
- `nbadata.py` — ESPN fetches, the season cache, arena table, and the rest/travel features
- `template.html` — board page; `build.py` fills it in
- `monitor/` — log, grading, accuracy scan, alerts
- `backtest/` — backtest, rules, learner, report, cached seasons
