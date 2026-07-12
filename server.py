#!/usr/bin/env python3
"""
Backend that links the betting model + browser agent to the Tandem UI.

  python server.py [port]      (default 8080)

Zero extra deps (stdlib http.server). Serves platform/index.html and exposes:
  GET  /api/stats     → performance summary + recent bets from SQLite
  GET  /api/model     → live GRPO adapter status
  GET  /api/markets   → market snapshot parsed from the last bet run's log
  POST /api/fund      → kick off add-funds browser session   → watch URL
  POST /api/kickoff   → kick off check/scrape browser session → watch URL
  POST /api/bet       → run the full bet pipeline in background

Browser sessions are created synchronously (fast) and their watch URL returned.
The bet pipeline is slow (model load) so it runs as a detached subprocess; the
UI polls /api/stats to see the resulting bet.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

UI_FILE = ROOT / "platform" / "control.html"      # simple control panel (default)
DECK_FILE = ROOT / "platform" / "index.html"      # full Tandem deck
ARCH_FILE = ROOT / "platform" / "architecture.html"  # architecture explainer


def _kalshi():
    return (os.getenv("KALSHI_URL", "https://demo.kalshi.co"),
            os.environ["KALSHI_EMAIL"], os.environ["KALSHI_PASSWORD"])


def start_browser_session(kind: str, amount: float = 100.0) -> dict:
    """Create an H Company browser session and return its watch URL."""
    from hai_agents import Client
    from agent.tasks import add_funds_task, check_results_task, scrape_markets_task
    from agent.runner import AGENT_VIEW_HOST, EXECUTE_AGENT

    url, email, password = _kalshi()
    task = {
        "fund": lambda: add_funds_task(url, email, password, amount),
        "check": lambda: check_results_task(url, email, password),
        "scrape": lambda: scrape_markets_task(url, email, password),
    }[kind]()

    client = Client(api_key=os.environ["HAI_API_KEY"])
    session = client.sessions.create_session(agent=EXECUTE_AGENT, messages=task)
    return {"session_id": session.id,
            "watch_url": f"{AGENT_VIEW_HOST}/agent-view/{session.id}",
            "kind": kind}


def stats() -> dict:
    from agent import memory
    memory.init_db()
    perf = memory.get_performance_summary()
    wins, losses = perf.get("wins") or 0, perf.get("losses") or 0
    bets = [{
        "date": b["created_at"][:19], "market": b["market"][:60],
        "direction": b["direction"], "amount": b["amount"],
        "status": b["status"], "pl": b.get("profit_loss"),
    } for b in memory.get_all_bets(limit=10)]
    return {
        "total_bets": perf.get("total_bets") or 0,
        "wins": wins, "losses": losses,
        "win_rate": wins / max(wins + losses, 1) if (wins + losses) else None,
        "wagered": perf.get("total_wagered") or 0,
        "pnl": perf.get("total_profit_loss") or 0,
        "bets": bets,
    }


def model_status() -> dict:
    adapter = ROOT / "data" / "grpo_weights" / "adapter"
    trained = adapter.exists() and any(adapter.iterdir())
    return {"trained": trained,
            "base": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter_present": trained}


BET_LOG = ROOT / "data" / "last_bet.log"
IMPROVE_LOG = ROOT / "data" / "last_improve.log"


def run_improve_async() -> dict:
    """Self-improvement cycle (dataset refresh + tabular research loop),
    detached; UI polls /api/improve-watch."""
    with open(IMPROVE_LOG, "w") as fh:
        subprocess.Popen([sys.executable, str(ROOT / "main.py"), "improve"],
                         cwd=str(ROOT), env=dict(os.environ),
                         stdout=fh, stderr=subprocess.STDOUT)
    return {"started": True}


AUTO_IMPROVE = os.getenv("AUTO_IMPROVE", "1") != "0"
IMPROVE_EVERY_H = float(os.getenv("IMPROVE_EVERY_HOURS", "12"))


def _auto_improve_scheduler():
    """No-click self-improvement: while the platform runs, kick off an
    improve cycle whenever the last one is older than IMPROVE_EVERY_HOURS.

    Staleness is judged by the improve log's mtime, so server restarts
    don't retrigger and a run in progress (fresh mtime) is never doubled.
    Random jitter keeps two server instances from firing simultaneously.
    """
    import random
    import time
    time.sleep(60 + random.uniform(0, 120))   # let the server settle
    while True:
        try:
            last = IMPROVE_LOG.stat().st_mtime if IMPROVE_LOG.exists() else 0
            if time.time() - last > IMPROVE_EVERY_H * 3600:
                print(f"[auto-improve] last cycle >{IMPROVE_EVERY_H:.0f}h ago "
                      "— starting a new one")
                run_improve_async()
        except Exception as e:
            print(f"[auto-improve] error: {e}")
        time.sleep(600 + random.uniform(0, 180))


def improve_watch() -> dict:
    if not IMPROVE_LOG.exists():
        return {"phase": "idle", "champion": None, "log_tail": ""}
    text = IMPROVE_LOG.read_text(errors="replace")
    out = {"phase": "refreshing dataset", "champion": None,
           "log_tail": "\n".join(text.splitlines()[-3:])}
    if "tabular:" in text:
        out["phase"] = "running research loop"
    import re
    m = re.search(r"New champion: (\S+)", text)
    if m:
        out["phase"] = "done"
        out["champion"] = m.group(1)
    return out


def run_bet_async() -> dict:
    """Launch the full bet pipeline detached; UI polls /api/bet-watch for progress."""
    env = dict(os.environ, GRPO_DEVICE="cpu")
    with open(BET_LOG, "w") as fh:
        subprocess.Popen([sys.executable, str(ROOT / "main.py"), "bet"],
                         cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    return {"started": True}


def bet_watch() -> dict:
    """Parse the running bet log for phase, watch URL, decision and result."""
    import re
    if not BET_LOG.exists():
        return {"phase": "idle", "watch_url": None, "decision": None,
                "result": None, "engine": None}
    text = BET_LOG.read_text(errors="replace")
    out = {"phase": "starting", "watch_url": None, "decision": None,
           "result": None, "engine": None}
    if "Phase 2" in text or "GRPO]" in text:
        out["phase"] = "deciding"
    m = re.search(r"agent-view/([0-9a-f-]+)", text)
    if m:
        out["phase"] = "executing"
        out["watch_url"] = f"https://platform.eu.hcompany.ai/agent-view/{m.group(1)}"
    m = re.search(r"\[(GRPO|XGB)\] → (\w+) \| (\S+) \| \$([\d.]+) \| conf=(\d+)%", text)
    if m:
        out["engine"] = {"XGB": "XGBoost", "GRPO": "GRPO policy"}[m.group(1)]
        out["decision"] = {"direction": m.group(2), "ticker": m.group(3),
                           "amount": float(m.group(4)), "confidence": int(m.group(5))}
    if "placed successfully" in text.lower():
        out["phase"] = "done"; out["result"] = "placed"
    elif "Insufficient funds" in text:
        out["phase"] = "done"; out["result"] = "insufficient_funds"
    elif "Execution failed" in text:
        out["phase"] = "done"; out["result"] = "failed"
    elif "Skipped —" in text:
        out["phase"] = "done"; out["result"] = "skipped"
    elif "Traceback (most recent call last)" in text:
        out["phase"] = "done"; out["result"] = "failed"
    return out


def learning_status() -> dict:
    """Offline-learning artifacts for the architecture page."""
    def _load(name):
        p = ROOT / "data" / name
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except json.JSONDecodeError:
            return None
    train = _load("dataset_train.json") or []
    test = _load("dataset_test.json") or []
    log = _load("research_log.json") or []
    simple_eval = _load("simple_model_eval.json") or {}
    return {
        "dataset": {"train": len(train), "test": len(test),
                    "events": len({e.get("event") for e in train + test})},
        "offline_eval": _load("offline_eval.json"),
        "tabular_champion": _load("tabular_champion.json"),
        "baseline_full": simple_eval.get("implied_baseline"),
        "trend_rules": _load("trend_rules.json") or [],
        "research_log": log[-8:],
    }


def markets_snapshot() -> dict:
    """Parse the last bet run's log for the markets that were sent to the model."""
    import re
    if not BET_LOG.exists():
        return {"total": 0, "sent": 0, "markets": []}
    text = BET_LOG.read_text(errors="replace")
    m = re.search(r"(\d+) WC markets total, (\d+) sent to model", text)
    total, sent = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    markets = [{"type": t, "ticker": tk, "yes": float(y), "volume": int(v.replace(",", ""))}
               for t, tk, y, v in
               re.findall(r"\[(\w+)\] (\S+) \| Yes=([\d.]+) \| Vol=\$([\d,]+)", text)]
    return {"total": total, "sent": sent, "markets": markets}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/control", "/control.html"):
                self._html(UI_FILE.read_text())
            elif path in ("/deck", "/index.html"):
                self._html(DECK_FILE.read_text())
            elif path in ("/arch", "/architecture", "/architecture.html"):
                self._html(ARCH_FILE.read_text())
            elif path == "/api/learning":
                self._json(learning_status())
            elif path == "/api/improve-watch":
                self._json(improve_watch())
            elif path == "/api/stats":
                self._json(stats())
            elif path == "/api/model":
                self._json(model_status())
            elif path == "/api/bet-watch":
                self._json(bet_watch())
            elif path == "/api/markets":
                self._json(markets_snapshot())
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        qs = parse_qs(urlparse(self.path).query)
        try:
            if path == "/api/fund":
                amt = float(qs.get("amount", ["100"])[0])
                self._json(start_browser_session("fund", amt))
            elif path == "/api/kickoff":
                mode = qs.get("mode", ["check"])[0]
                self._json(start_browser_session(mode))
            elif path == "/api/bet":
                self._json(run_bet_async())
            elif path == "/api/improve":
                self._json(run_improve_async())
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)


if __name__ == "__main__":
    PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"Tandem platform → http://localhost:{PORT}")
    print(f"  UI: {UI_FILE}")
    if AUTO_IMPROVE:
        import threading
        threading.Thread(target=_auto_improve_scheduler, daemon=True).start()
        print(f"  auto-improve: every {IMPROVE_EVERY_H:.0f}h "
              "(AUTO_IMPROVE=0 to disable)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
