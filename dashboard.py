#!/usr/bin/env python3
"""
Live GRPO training dashboard — zero dependencies (stdlib http.server).

  python dashboard.py <training_log_path> [port]

Parses tqdm progress lines from the training log and trajectory stats from
data/trajectories.json, serves a live-updating page at http://localhost:<port>.
"""

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else None
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8787
TRAJ_PATH = Path(__file__).parent / "data" / "trajectories.json"
ADAPTER_PATH = Path(__file__).parent / "data" / "grpo_weights" / "adapter"

TQDM_RE = re.compile(
    r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+),\s*([\d.]+)(s/it|it/s)\]"
)


def parse_log():
    state = {
        "phase": "starting", "step": 0, "total": 0, "pct": 0,
        "elapsed": "", "eta": "", "rate": "", "log_tail": [], "done": False,
    }
    if not LOG_PATH or not LOG_PATH.exists():
        return state
    text = LOG_PATH.read_text(errors="replace").replace("\r", "\n")
    lines = [l for l in text.splitlines() if l.strip()]
    state["log_tail"] = [l for l in lines if not TQDM_RE.search(l)][-12:]

    for l in lines:
        if "Fetching live Kalshi" in l:
            state["phase"] = "fetching markets"
        elif "Generating" in l and "trajectories" in l:
            state["phase"] = "simulating"
        elif "Fine-tuning on" in l:
            state["phase"] = "training"
        elif "Training Complete" in l or "training complete" in l.lower():
            state["phase"] = "complete"
            state["done"] = True

    for m in TQDM_RE.finditer(text):
        state["pct"] = int(m.group(1))
        state["step"] = int(m.group(2))
        state["total"] = int(m.group(3))
        state["elapsed"] = m.group(4)
        state["eta"] = m.group(5)
        state["rate"] = f"{m.group(6)} {m.group(7)}"

    if ADAPTER_PATH.exists() and any(ADAPTER_PATH.iterdir()):
        state["phase"] = "complete"
        state["done"] = True
    return state


def traj_stats():
    if not TRAJ_PATH.exists():
        return None
    try:
        trajs = json.loads(TRAJ_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    rewards = [t.get("reward", 0) for t in trajs]
    wins = sum(1 for t in trajs if t.get("won"))
    return {
        "count": len(trajs),
        "wins": wins,
        "win_rate": wins / max(len(trajs), 1),
        "avg_reward": sum(rewards) / max(len(rewards), 1),
        "rewards": rewards,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        if self.path == "/status":
            self._send(json.dumps({"train": parse_log(), "traj": traj_stats()}),
                       "application/json")
        else:
            self._send(PAGE, "text/html")


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>GRPO Training — Kalshi FIFA Agent</title>
<style>
  :root { --bg:#0d1117; --card:#161b22; --border:#30363d; --fg:#e6edf3;
          --dim:#8b949e; --accent:#3fb950; --amber:#d29922; --blue:#58a6ff; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system,'SF Pro',Segoe UI,sans-serif; padding:2rem; max-width:960px; margin:auto; }
  h1 { font-size:1.3rem; font-weight:600; margin-bottom:.25rem; }
  .sub { color:var(--dim); margin-bottom:1.5rem; font-size:.9rem; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:.75rem; margin-bottom:1rem; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:.9rem 1rem; }
  .card .label { color:var(--dim); font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; }
  .card .value { font-size:1.4rem; font-weight:600; font-variant-numeric:tabular-nums; }
  .bar-wrap { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:1rem; margin-bottom:1rem; }
  .bar-bg { background:#21262d; border-radius:6px; height:22px; overflow:hidden; }
  .bar-fill { background:linear-gradient(90deg,#238636,#3fb950); height:100%; width:0; border-radius:6px; transition:width .8s ease; }
  .bar-meta { display:flex; justify-content:space-between; color:var(--dim); font-size:.85rem; margin-top:.5rem; font-variant-numeric:tabular-nums; }
  .phase { display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.8rem; font-weight:600; }
  .phase.training { background:#1f3a5f; color:var(--blue); }
  .phase.complete { background:#12351d; color:var(--accent); }
  .phase.other { background:#3a2d10; color:var(--amber); }
  canvas { width:100%; height:120px; }
  pre { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:1rem;
        color:var(--dim); font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace; overflow-x:auto; white-space:pre-wrap; }
  h2 { font-size:.9rem; color:var(--dim); text-transform:uppercase; letter-spacing:.06em; margin:1.25rem 0 .5rem; }
  .flow { display:flex; flex-wrap:wrap; gap:.5rem; align-items:stretch; margin-bottom:.75rem; }
  .stage { flex:1 1 150px; background:var(--card); border:1px solid var(--border); border-radius:8px;
           padding:.75rem .85rem; position:relative; }
  .stage .num { display:inline-flex; align-items:center; justify-content:center; width:20px; height:20px;
                border-radius:50%; background:#1f3a5f; color:var(--blue); font-size:.75rem; font-weight:700; margin-bottom:.35rem; }
  .stage .name { font-weight:600; font-size:.9rem; margin-bottom:.2rem; }
  .stage .desc { color:var(--dim); font-size:.78rem; line-height:1.45; }
  .stage .tech { display:inline-block; margin-top:.4rem; font:11px ui-monospace,Menlo,monospace;
                 color:var(--blue); background:#0d2137; border-radius:4px; padding:.1rem .4rem; }
  .arrow { align-self:center; color:var(--dim); font-size:1.1rem; flex:0 0 auto; }
  .loop { background:var(--card); border:1px dashed var(--accent); border-radius:8px; padding:.75rem 1rem;
          color:var(--dim); font-size:.85rem; margin-bottom:1rem; }
  .loop b { color:var(--accent); }
  .grpo { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:1rem; margin-bottom:1rem; }
  .grpo ol { margin:.25rem 0 0 1.25rem; color:var(--dim); font-size:.85rem; }
  .grpo li { margin:.3rem 0; }
  .grpo li b { color:var(--fg); font-weight:600; }
  .topnav { display:flex; align-items:center; gap:1rem; margin-bottom:1rem; }
  .topnav a { color:var(--blue); text-decoration:none; font-size:.85rem; font-weight:600;
              border:1px solid var(--border); border-radius:6px; padding:.35rem .7rem; }
  .topnav a:hover { border-color:var(--blue); background:#0d2137; }
</style></head><body>
<div class="topnav"><a href="http://localhost:4321/">← Full command center</a></div>
<h1>GRPO Fine-tuning <span id="phase" class="phase other">…</span></h1>
<div class="sub">Qwen2.5-0.5B · LoRA · Kalshi FIFA World Cup betting agent · auto-refreshes every 2s</div>

<div class="bar-wrap">
  <div class="bar-bg"><div id="bar" class="bar-fill"></div></div>
  <div class="bar-meta"><span id="steps">step 0 / 0</span><span id="timing"></span></div>
</div>

<div class="grid">
  <div class="card"><div class="label">Progress</div><div class="value" id="pct">0%</div></div>
  <div class="card"><div class="label">ETA</div><div class="value" id="eta">–</div></div>
  <div class="card"><div class="label">Trajectories</div><div class="value" id="count">–</div></div>
  <div class="card"><div class="label">Sim win rate</div><div class="value" id="wr">–</div></div>
  <div class="card"><div class="label">Avg reward</div><div class="value" id="avgr">–</div></div>
</div>

<h2>System design — how the agent works</h2>
<div class="flow">
  <div class="stage"><span class="num">1</span>
    <div class="name">📈 Market feed</div>
    <div class="desc">Pulls live FIFA World Cup prediction markets (prices, volume) from Kalshi's REST API. Illiquid markets are filtered out.</div>
    <span class="tech">kalshi_api.py</span>
  </div>
  <div class="arrow">→</div>
  <div class="stage"><span class="num">2</span>
    <div class="name">🧠 Bet decision</div>
    <div class="desc">Our fine-tuned model reads the market state and decides: which side to bet, how much (Kelly sizing), and its confidence.</div>
    <span class="tech">Qwen2.5 + LoRA · grpo_model.py</span>
  </div>
  <div class="arrow">→</div>
  <div class="stage"><span class="num">3</span>
    <div class="name">✅ Sanity check</div>
    <div class="desc">A second, larger model (Holo) reviews the bet. If it looks bad, the bet is vetoed or replaced with a safe fallback.</div>
    <span class="tech">holo3-1-35b · decision.py</span>
  </div>
  <div class="arrow">→</div>
  <div class="stage"><span class="num">4</span>
    <div class="name">🖱️ Execution</div>
    <div class="desc">An H Company browser agent logs into demo.kalshi.co and actually places the bet — demo money only.</div>
    <span class="tech">web-surfer-flash · runner.py</span>
  </div>
</div>
<div class="loop">↻ <b>Self-improvement loop:</b> after each market settles, the profit/loss becomes a <b>reward signal</b>
  (P&amp;L + calibration bonus + sizing score). Rewards feed a <b>GRPO retrain</b> of the model in step 2 and a
  <b>reflection step</b> that writes new lessons to memory — so every bet makes the next one smarter.
  <span style="font:11px ui-monospace,Menlo,monospace">reward.py · reflection.py · memory.py</span></div>

<h2>How GRPO training works (what the progress bar above is doing)</h2>
<div class="grpo"><ol>
  <li><b>Sample:</b> for each market state, the model generates <b>4 different bet decisions</b> instead of one.</li>
  <li><b>Score:</b> each decision gets a reward — simulated P&amp;L plus bonuses for good calibration and Kelly-optimal sizing.</li>
  <li><b>Compare to the group:</b> the average reward of the 4 acts as the baseline — no separate critic network needed.</li>
  <li><b>Update:</b> decisions that beat the group average become more likely; below-average ones less likely. Repeat over all trajectories → a better betting policy.</li>
</ol></div>

<h2>Reward distribution (100 simulated trajectories)</h2>
<div class="bar-wrap"><canvas id="hist" width="880" height="120"></canvas></div>

<h2>Log tail</h2>
<pre id="log">waiting for data…</pre>

<script>
function drawHist(rewards) {
  const c = document.getElementById('hist'), x = c.getContext('2d');
  x.clearRect(0,0,c.width,c.height);
  if (!rewards || !rewards.length) return;
  const bins = 24, lo = Math.min(...rewards), hi = Math.max(...rewards);
  const span = (hi-lo)||1, counts = new Array(bins).fill(0);
  rewards.forEach(r => counts[Math.min(bins-1, Math.floor((r-lo)/span*bins))]++);
  const max = Math.max(...counts), w = c.width/bins;
  counts.forEach((n,i) => {
    const mid = lo + (i+.5)/bins*span;
    x.fillStyle = mid >= 0 ? '#3fb950' : '#f85149';
    const h = n/max*(c.height-14);
    x.fillRect(i*w+1, c.height-h, w-2, h);
  });
  x.fillStyle = '#8b949e'; x.font = '11px ui-monospace';
  x.fillText(lo.toFixed(2), 2, 11); x.fillText(hi.toFixed(2), c.width-38, 11);
}
async function tick() {
  try {
    const {train:t, traj} = await (await fetch('/status')).json();
    const done = t.done;
    const pct = done ? 100 : t.pct;
    document.getElementById('bar').style.width = pct + '%';
    document.getElementById('pct').textContent = pct + '%';
    document.getElementById('steps').textContent = `step ${done ? t.total : t.step} / ${t.total}`;
    document.getElementById('timing').textContent = t.rate ? `${t.elapsed} elapsed · ${t.rate}` : '';
    document.getElementById('eta').textContent = done ? 'done' : (t.eta || '–');
    const ph = document.getElementById('phase');
    ph.textContent = t.phase;
    ph.className = 'phase ' + (t.phase === 'training' ? 'training' : done ? 'complete' : 'other');
    document.getElementById('log').textContent = t.log_tail.join('\\n') || 'no log yet';
    if (traj) {
      document.getElementById('count').textContent = traj.count;
      document.getElementById('wr').textContent = Math.round(traj.win_rate*100) + '%';
      document.getElementById('avgr').textContent = traj.avg_reward.toFixed(3);
      drawHist(traj.rewards);
    }
  } catch(e) {}
}
tick(); setInterval(tick, 2000);
</script></body></html>"""


if __name__ == "__main__":
    print(f"Dashboard → http://localhost:{PORT}  (log: {LOG_PATH})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
