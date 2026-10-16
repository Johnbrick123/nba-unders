"""Turn alerts from the monitor and the learner into GitHub issues.

- A new alert opens an issue labeled nba-alert (GitHub emails the repo owner).
- An alert that is still active gets a reminder comment at most once a week.
- An alert that cleared gets a closing comment and is closed.
- Model-change notices (sticky) stay open until you close them.
Only runs inside GitHub Actions; locally it just prints what it would do.
Run:  python monitor/alerts.py
"""
import json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
LIVE = os.environ.get("GITHUB_ACTIONS") == "true"
OWNER = os.environ.get("GITHUB_REPOSITORY_OWNER", "")
LEVEL = {"action": "🔴 Action needed", "watch": "🟡 Watch", "info": "🔵 Model change"}


def gh(*args):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def load_alerts():
    out = []
    for p in (ROOT / "monitor" / "status.json", ROOT / "backtest" / "learn_status.json"):
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            src = "monitor" if p.parent.name == "monitor" else "learner"
            out += [dict(a, source=src) for a in d.get("alerts", [])]
    return out


def main(scope):
    alerts = [a for a in load_alerts() if a["source"] in scope]
    if not LIVE:
        for a in alerts:
            print(f"[dry run] would raise {a['key']}: {a['title']}")
        return
    for lab, color in (("nba-alert", "d73a4a"), ("model-change", "1d76db")):
        gh("label", "create", lab, "--color", color, "--force")
    open_issues = json.loads(gh("issue", "list", "--label", "nba-alert", "--state", "open", "--limit", "100",
                                "--json", "number,title,body,updatedAt,labels"))
    by_key = {}
    for i in open_issues:
        for line in i["body"].splitlines():
            if line.startswith("<!-- key:"):
                by_key[line[9:-4].strip()] = i
    now = datetime.now(timezone.utc)
    active = {a["key"] for a in alerts}
    for a in alerts:
        body = (f"**{LEVEL.get(a['level'], a['level'])}**\n\n{a['body']}\n\n"
                f"Monitor: see `monitor.html` on the site · source: {a['source']}\n\n@{OWNER}\n\n<!-- key: {a['key']} -->")
        if a["key"] in by_key:
            i = by_key[a["key"]]
            age = (now - datetime.fromisoformat(i["updatedAt"].replace("Z", "+00:00"))).days
            if age >= 7 and not a.get("sticky"):
                gh("issue", "comment", str(i["number"]), "--body", f"Still active as of {now:%Y-%m-%d}.\n\n{a['body']}")
            continue
        labels = "nba-alert,model-change" if a.get("sticky") else "nba-alert"
        gh("issue", "create", "--title", a["title"], "--body", body, "--label", labels)
        print(f"opened: {a['title']}")
    for key, i in by_key.items():
        sticky = any(l["name"] == "model-change" for l in i["labels"])
        src_ok = ("learner" in scope) if key in ("data-stale", "projection-effects", "big-shift") else ("monitor" in scope)
        if key not in active and not sticky and src_ok:
            gh("issue", "close", str(i["number"]), "--comment", f"Cleared on {now:%Y-%m-%d}: the check passed again.")
            print(f"closed: {i['title']}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or ["monitor", "learner"])
