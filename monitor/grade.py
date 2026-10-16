"""Fill in final points for every logged row whose game has finished.

Points come from the ESPN scoreboard and include overtime, which is how books settle
team totals. Run:  python monitor/grade.py
"""
import csv, sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import nbadata as nd


def main():
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    for path in sorted(HERE.glob("log_*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields, rows = list(reader.fieldnames), list(reader)
        todo = sorted({r["date"][:7].replace("-", "") for r in rows if r["actual"] == ""})
        graded = 0
        for ym in todo:
            try:
                games = {}
                for e in nd._month_events(ym):
                    r = nd._event_row(e)
                    if r:
                        games[r["id"]] = r
            except Exception as e:
                print(f"{ym}: scoreboard failed ({e}); will retry next run")
                continue
            if not games:
                continue
            for r in rows:
                if r["actual"] != "" or r["date"][:7].replace("-", "") != ym:
                    continue
                g = games.get(r["gameId"])
                if not g or g["state"] != "FINAL":
                    continue
                r["actual"] = g["homeScore"] if r["ha"] == "Home" else g["awayScore"]
                r["ot"] = g["ot"]
                r["graded_at"] = stamp
                graded += 1
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        left = sum(1 for r in rows if r["actual"] == "")
        print(f"{path.name}: graded {graded}, {left} waiting on results")


if __name__ == "__main__":
    sys.exit(main())
