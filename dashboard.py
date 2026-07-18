#!/usr/bin/env python3
"""
dashboard.py — Live web dashboard for the wearable streaming pipeline.

One self-contained Flask app that:
  * consumes the raw Kafka topic (`wearables.raw`) in a background thread and keeps
    a rolling in-memory buffer of recent heart-rate readings per user (the LIVE view);
  * receives anomaly alerts POSTed by spark_processor.py at /anomaly
    (so it replaces agent_stub.py — same port, same endpoint);
  * reads Spark's finalized 15-min rolling-average windows back from the Delta table
    (best-effort, via deltalake) for the historical panel;
  * serves an auto-refreshing dashboard at http://localhost:8000/.

Run (with Kafka up and simulator + spark_processor running):
    python3 dashboard.py
    # then open http://localhost:8000

No external JS/CSS — the page is fully offline-capable and theme-aware.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import defaultdict, deque

from flask import Flask, Response, jsonify, request
from kafka import KafkaConsumer

try:
    from deltalake import DeltaTable
except Exception:  # deltalake optional; the windows panel just stays empty without it
    DeltaTable = None

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "wearables.raw")
DELTA_PATH = os.getenv("DELTA_PATH", "./lakehouse/wearable_windows")
PORT = int(os.getenv("DASHBOARD_PORT", "8000"))

WINDOW_MS = 10 * 60 * 1000        # keep ~10 min of raw readings for the live chart
MAX_POINTS_PER_USER = 800
MAX_ANOMALIES = 40

app = Flask(__name__)

# ---- shared state (guarded by _lock) -------------------------------------- #
_lock = threading.Lock()
_readings: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_POINTS_PER_USER))
_state: dict[str, dict] = {}
_anomalies: deque = deque(maxlen=MAX_ANOMALIES)
_user_order: list[str] = []       # first-seen order → stable color assignment
_msg_times: deque = deque(maxlen=2000)
_total_msgs = 0


def _epoch_ms(iso_ts: str) -> int:
    """Parse the simulator's ISO-8601 UTC ts to epoch ms (no tz libs needed)."""
    import datetime as dt
    return int(dt.datetime.fromisoformat(iso_ts).timestamp() * 1000)


def kafka_loop() -> None:
    """Background consumer → rolling in-memory buffers."""
    global _total_msgs
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=1000,
    )
    while True:
        try:
            for msg in consumer:
                v = msg.value
                user = v.get("user_id")
                # Apple sends hr_bpm; Oura sends avg_hr — unify like the Spark job.
                hr = v.get("hr_bpm", v.get("avg_hr"))
                if user is None or hr is None:
                    continue
                try:
                    t = _epoch_ms(v["ts"])
                except Exception:
                    t = int(time.time() * 1000)
                with _lock:
                    if user not in _user_order:
                        _user_order.append(user)
                    _readings[user].append((t, float(hr)))
                    _state[user] = {
                        "device": v.get("device"),
                        "event": v.get("event", "—"),
                        "hr": float(hr),
                        "t": t,
                    }
                    _msg_times.append(time.time())
                    _total_msgs += 1
        except Exception:
            time.sleep(1)  # consumer hiccup — back off and keep going


def read_windows() -> list[dict]:
    """Read Spark's finalized rolling-average windows from Delta (best-effort)."""
    if DeltaTable is None:
        return []
    try:
        df = DeltaTable(DELTA_PATH).to_pandas()
    except Exception:
        return []
    if df.empty:
        return []
    df = df.sort_values("window_start").tail(30)
    out = []
    for _, r in df.iterrows():
        avg = float(r["avg_hr"])
        workout = int(r["workout_samples"])
        out.append({
            "window_start": str(r["window_start"]),
            "window_end": str(r["window_end"]),
            "user_id": r["user_id"],
            "avg_hr": round(avg, 1),
            "max_hr": float(r["max_hr"]),
            "min_hr": float(r["min_hr"]),
            "sample_count": int(r["sample_count"]),
            "workout_samples": workout,
            "sources": list(r["sources"]) if r["sources"] is not None else [],
            "anomaly": bool(avg > 140 and workout == 0),
        })
    out.reverse()  # newest first
    return out


@app.route("/anomaly", methods=["POST"])
def anomaly():
    payload = request.get_json(force=True, silent=True) or {}
    payload["received_ms"] = int(time.time() * 1000)
    with _lock:
        _anomalies.appendleft(payload)
    print(f"ANOMALY received: user={payload.get('user_id')} avg_hr={payload.get('avg_hr')}")
    return jsonify({"status": "received"})


@app.route("/data")
def data():
    now = int(time.time() * 1000)
    cutoff = now - WINDOW_MS
    with _lock:
        series = {
            u: [[t, hr] for (t, hr) in pts if t >= cutoff]
            for u, pts in _readings.items()
        }
        state = dict(_state)
        users = list(_user_order)
        anomalies = list(_anomalies)
        recent = [ts for ts in _msg_times if ts >= time.time() - 5]
        total = _total_msgs
    windows = read_windows()
    active = sum(1 for u in users if state.get(u, {}).get("t", 0) >= now - 30000)
    return jsonify({
        "now": now,
        "users": users,
        "series": series,
        "state": state,
        "windows": windows,
        "anomalies": anomalies,
        "stats": {
            "msg_rate": round(len(recent) / 5.0, 1),
            "active_users": active,
            "windows": len(windows),
            "anomalies": len(anomalies),
            "total_msgs": total,
        },
    })


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


# --------------------------------------------------------------------------- #
# Frontend — inline HTML/CSS/JS, no external assets. Also usable as an artifact
# snapshot: if window.__SNAPSHOT__ is defined it renders that once and never polls.
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Wearable Pipeline — Live</title>
<style>
  :root{
    color-scheme: light;
    --plane:#f9f9f7; --surface:#fcfcfb; --text:#0b0b0b; --text2:#52514e; --muted:#898781;
    --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
    --s1:#2a78d6; --s2:#1baf7a; --s3:#4a3aa7;
    --critical:#d03b3b; --good:#0ca30c; --warn:#fab219;
  }
  @media (prefers-color-scheme: dark){ :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19; --text:#fff; --text2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --s1:#3987e5; --s2:#199e70; --s3:#9085e9; --critical:#e05656;
  }}
  :root[data-theme="dark"]{
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19; --text:#fff; --text2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --s1:#3987e5; --s2:#199e70; --s3:#9085e9; --critical:#e05656;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--plane);color:var(--text);
       font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14px;line-height:1.4}
  .wrap{max-width:1160px;margin:0 auto;padding:20px}
  header{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:4px}
  h1{font-size:19px;margin:0;font-weight:650}
  .sub{color:var(--text2);font-size:13px}
  .live{display:inline-flex;align-items:center;gap:6px;color:var(--text2);font-size:12px;margin-left:auto}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--good);
       box-shadow:0 0 0 0 rgba(12,163,12,.5);animation:pulse 1.6s infinite}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(12,163,12,.5)}70%{box-shadow:0 0 0 7px rgba(12,163,12,0)}100%{box-shadow:0 0 0 0 rgba(12,163,12,0)}}
  button.theme{margin-left:8px;background:var(--surface);color:var(--text2);border:1px solid var(--border);
       border-radius:7px;padding:3px 9px;font:inherit;font-size:12px;cursor:pointer}
  .tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}
  .tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
  .tile .k{color:var(--muted);font-size:12px;letter-spacing:.02em}
  .tile .v{font-size:26px;font-weight:650;margin-top:3px;font-variant-numeric:tabular-nums}
  .tile .u{color:var(--muted);font-size:12px;font-weight:400}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
  .grid{display:grid;grid-template-columns:2fr 1fr;gap:14px;align-items:start}
  .ttl{font-size:13px;font-weight:600;margin:0 0 2px}
  .cap{color:var(--muted);font-size:12px;margin:0 0 10px}
  svg{width:100%;height:auto;display:block;touch-action:none}
  .legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px}
  .lg{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--text2)}
  .sw{width:12px;height:3px;border-radius:2px}
  .feed{display:flex;flex-direction:column;gap:8px;max-height:340px;overflow:auto}
  .alert{display:flex;gap:9px;align-items:flex-start;border:1px solid var(--border);
         border-left:3px solid var(--critical);border-radius:8px;padding:9px 11px;background:var(--surface)}
  .alert .ic{color:var(--critical);font-weight:700;line-height:1.2}
  .alert .who{font-weight:600}
  .alert .det{color:var(--text2);font-size:12px;margin-top:2px}
  .empty{color:var(--muted);font-size:13px;padding:14px;text-align:center}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{text-align:right;padding:7px 8px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums;white-space:nowrap}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left;font-variant-numeric:normal}
  th{color:var(--muted);font-weight:500;font-size:12px}
  .chip{display:inline-flex;align-items:center;gap:4px;font-size:11px;font-weight:600;
        padding:2px 7px;border-radius:20px}
  .chip.crit{color:#fff;background:var(--critical)}
  .chip.ok{color:var(--text2);border:1px solid var(--border)}
  .mt{margin-top:14px}
  @media (max-width:760px){.tiles{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Wearable Streaming Pipeline</h1>
    <span class="sub">Kafka → Spark Structured Streaming → Delta</span>
    <span class="live"><span class="dot" id="livedot"></span><span id="livetxt">live</span>
      <button class="theme" onclick="toggleTheme()">◐ theme</button></span>
  </header>

  <div class="tiles" id="tiles"></div>

  <div class="grid">
    <div class="card">
      <p class="ttl">Heart rate — live raw stream</p>
      <p class="cap">Per-user readings from Kafka, last 10 min. Dashed line = 140 bpm anomaly threshold.</p>
      <svg id="chart" viewBox="0 0 720 320" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="legend" id="legend"></div>
    </div>
    <div class="card">
      <p class="ttl">Anomaly alerts</p>
      <p class="cap">POSTed by Spark: avg HR &gt; 140 with no workout.</p>
      <div class="feed" id="feed"></div>
    </div>
  </div>

  <div class="card mt">
    <p class="ttl">Rolling 15-min windows <span style="color:var(--muted);font-weight:400">— Spark output, read back from Delta</span></p>
    <div id="wtable"></div>
  </div>
</div>

<script>
const PAL = ["--s1","--s2","--s3","--s1","--s2","--s3"];
function cvar(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}
function colorFor(users,u){return cvar(PAL[users.indexOf(u)%PAL.length]);}
function toggleTheme(){const r=document.documentElement;
  const cur=r.getAttribute("data-theme")|| (matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");
  r.setAttribute("data-theme", cur==="dark"?"light":"dark"); if(LAST)render(LAST);}
function fmtTime(ms){const d=new Date(ms);return d.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"});}

let LAST=null;
function render(d){
  LAST=d;
  // ---- tiles ----
  const s=d.stats;
  const anomClass = s.anomalies>0 ? 'style="color:var(--critical)"' : '';
  document.getElementById("tiles").innerHTML = [
    tile("Message rate", s.msg_rate, "msg/s"),
    tile("Active users", s.active_users, "streaming"),
    tile("Windows persisted", s.windows, "in Delta"),
    tile(anomClass?"⚠ Anomalies":"Anomalies", s.anomalies, "alerts", anomClass),
  ].join("");
  drawChart(d);
  drawFeed(d);
  drawWindows(d);
}
function tile(k,v,u,attr=""){return `<div class="tile"><div class="k">${k}</div>
  <div class="v" ${attr}>${v}<span class="u"> ${u}</span></div></div>`;}

function drawChart(d){
  const W=720,H=320,L=42,R=64,T=14,B=26, x0=L,x1=W-R,y0=T,y1=H-B;
  const HRmin=40,HRmax=200;
  const now=d.now, tmin=now-10*60*1000, tmax=now;
  const users=d.users;
  const sx=t=>x0+(x1-x0)*Math.max(0,Math.min(1,(t-tmin)/(tmax-tmin)));
  const sy=h=>y1-(y1-y0)*((h-HRmin)/(HRmax-HRmin));
  let s=`<g font-family="system-ui" font-size="11">`;
  // gridlines + y labels
  for(let h=40;h<=200;h+=40){const y=sy(h);
    s+=`<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="${cvar('--grid')}"/>`;
    s+=`<text x="${x0-7}" y="${y+3}" text-anchor="end" fill="${cvar('--muted')}">${h}</text>`;}
  // x labels (every 2 min)
  for(let k=0;k<=5;k++){const t=tmin+(tmax-tmin)*k/5;const x=sx(t);
    s+=`<text x="${x}" y="${y1+16}" text-anchor="middle" fill="${cvar('--muted')}">${fmtTime(t)}</text>`;}
  // 140 threshold
  const yT=sy(140);
  s+=`<line x1="${x0}" y1="${yT}" x2="${x1}" y2="${yT}" stroke="${cvar('--critical')}" stroke-dasharray="5 4" stroke-width="1.5" opacity="0.85"/>`;
  s+=`<text x="${x1+4}" y="${yT+3}" fill="${cvar('--critical')}" font-size="10">140 bpm</text>`;
  // axes
  s+=`<line x1="${x0}" y1="${y1}" x2="${x1}" y2="${y1}" stroke="${cvar('--axis')}"/>`;
  // series
  users.forEach(u=>{const pts=(d.series[u]||[]).filter(p=>p[0]>=tmin);
    if(!pts.length)return; const c=colorFor(users,u);
    let path=pts.map((p,i)=>(i?"L":"M")+sx(p[0]).toFixed(1)+" "+sy(p[1]).toFixed(1)).join(" ");
    s+=`<path d="${path}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    const last=pts[pts.length-1];
    s+=`<circle cx="${sx(last[0])}" cy="${sy(last[1])}" r="3.2" fill="${c}" stroke="${cvar('--surface')}" stroke-width="1.5"/>`;
    // direct label: colored mark + name in text ink
    const ly=Math.max(y0+8,Math.min(y1-2,sy(last[1])));
    s+=`<circle cx="${x1+8}" cy="${ly-3}" r="3" fill="${c}"/>`;
    s+=`<text x="${x1+14}" y="${ly}" fill="${cvar('--text2')}" font-size="10">${u.replace('user_','')}</text>`;
  });
  s+=`</g>`;
  document.getElementById("chart").innerHTML=s;
  // legend (identity never color-alone)
  document.getElementById("legend").innerHTML = users.map(u=>{
    const st=d.state[u]||{}; const c=colorFor(users,u);
    const ev = st.event && st.event!=="—" ? " · "+st.event : "";
    const hr = st.hr!=null ? " · "+Math.round(st.hr)+" bpm" : "";
    return `<span class="lg"><span class="sw" style="background:${c}"></span>${u}${ev}${hr}</span>`;
  }).join("");
}

function drawFeed(d){
  const f=document.getElementById("feed");
  if(!d.anomalies.length){f.innerHTML=`<div class="empty">No anomalies detected.</div>`;return;}
  f.innerHTML=d.anomalies.map(a=>`<div class="alert">
     <span class="ic">⚠</span>
     <div><div class="who">${a.user_id||"?"}</div>
       <div class="det">avg ${a.avg_hr} bpm · max ${a.max_hr} · ${a.sample_count||"?"} samples<br>
       ${(a.reason||"").replace(/</g,"&lt;")}</div></div></div>`).join("");
}

function drawWindows(d){
  const el=document.getElementById("wtable");
  if(!d.windows.length){el.innerHTML=`<div class="empty">No finalized windows yet — the first 15-min window commits once its watermark passes.</div>`;return;}
  const rows=d.windows.map(w=>`<tr>
    <td>${fmtTime(Date.parse(w.window_start))||w.window_start}</td>
    <td>${w.user_id}</td>
    <td>${w.avg_hr}</td><td>${w.max_hr}</td><td>${w.min_hr}</td>
    <td>${w.sample_count}</td><td>${w.workout_samples}</td>
    <td style="text-align:left">${(w.sources||[]).join(", ")}</td>
    <td style="text-align:center">${w.anomaly
        ? '<span class="chip crit">⚠ anomaly</span>'
        : '<span class="chip ok">normal</span>'}</td></tr>`).join("");
  el.innerHTML=`<div style="overflow-x:auto"><table>
    <thead><tr><th>window</th><th>user</th><th>avg hr</th><th>max</th><th>min</th>
    <th>samples</th><th>workout</th><th>sources</th><th style="text-align:center">status</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

// ---- data source: snapshot (artifact) or live poll (server) ----
if (window.__SNAPSHOT__){
  document.getElementById("livetxt").textContent="snapshot";
  document.getElementById("livedot").style.animation="none";
  document.getElementById("livedot").style.background=cvar('--muted');
  render(window.__SNAPSHOT__);
} else {
  async function poll(){
    try{const r=await fetch("/data");render(await r.json());
        document.getElementById("livedot").style.background=cvar('--good');}
    catch(e){document.getElementById("livedot").style.background=cvar('--critical');
             document.getElementById("livetxt").textContent="disconnected";}
  }
  poll(); setInterval(poll,2000);
}
</script>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Live dashboard for the wearable pipeline")
    ap.add_argument("--bootstrap", default=BOOTSTRAP)
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--delta-path", default=DELTA_PATH)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    globals().update(BOOTSTRAP=args.bootstrap, TOPIC=args.topic, DELTA_PATH=args.delta_path)

    threading.Thread(target=kafka_loop, daemon=True).start()
    print(f"Dashboard on http://localhost:{args.port}  (Kafka {args.bootstrap}, topic {args.topic})")
    print("Point spark_processor.py's --agent-url at http://localhost:%d/anomaly" % args.port)
    app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
