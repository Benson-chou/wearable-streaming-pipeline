# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A local streaming pipeline for wearable health telemetry: a Python simulator produces raw
device JSON to Kafka, and a PySpark Structured Streaming job windows/aggregates it into a
Delta Lake table and POSTs anomalies to a local AI agent.

## Architecture

Data flows in one direction through four processes:

```
simulator.py ──JSON──▶ Kafka topic `wearables.raw` ──▶ spark_processor.py ──┬─▶ Delta table (./lakehouse/wearable_windows)
                                                                            └─▶ POST anomaly ─▶ agent_stub.py OR dashboard.py (:8000)
                                              │
                              dashboard.py ◀──┘ (also consumes the raw topic directly for the live chart)
```

- **`simulator.py`** — spawns 4 daemon threads (2 users × 2 devices). `Apple_Watch` emits
  erratic per-reading HR (every 10s in a `workout` phase, every 5min in `rest`);
  `Oura_Ring` emits rigid 5-min aggregate chunks (`avg_hr`, `sleep_score`, …). All messages
  are keyed by `user_id`. The two device payloads have **different, disparate schemas**.
- **`spark_processor.py`** — reads the topic, parses each record with a single *permissive
  superset* `StructType` (fields absent for a given device parse as null), then normalizes
  both shapes into a common `(event_time, user_id, source, hr, activity)` row:
  `hr = coalesce(hr_bpm, avg_hr)`, `activity = 1` only for Apple `workout` readings. It then
  applies a **15-min tumbling event-time window** (5-min watermark) per user for rolling
  stats, and in `foreachBatch` does two side effects per micro-batch: (a) append finalized
  windows to Delta, partitioned by `user_id`; (b) POST any window with
  `avg_hr > 140 AND workout_samples == 0` to the agent endpoint.
- **`agent_stub.py`** — a stand-in AI agent on `:8000` that just logs POST bodies; swap for
  the real agent later.
- **`agent_server.py`** — the real clinical-triage **AI agent** (FastAPI on `:8000`). On each
  anomaly POST it runs a Claude (`claude-opus-4-8`) tool-use loop with two tools —
  `fetch_historical_trends(user_id)` (reads the Delta table for the user's 7-day HR baseline)
  and `convert_to_fhir(user_id, metrics)` (builds a FHIR R4 `Observation`, LOINC 8867-4) —
  then reasons about whether the reading is a dangerous outlier vs. that baseline and returns a
  **Clinician Action Report** (`RISK LEVEL: LOW|ELEVATED|CRITICAL`). A third alternative for the
  `:8000` anomaly sink (run exactly one of `agent_stub` / `dashboard` / `agent_server`).
  **The target architecture is a local-first, provider-portable agent** (Ollama placeholder →
  Claude at scale, with a fallback/compare mode and a medical model consulted as a tool) —
  see [`docs/agent-design.md`](docs/agent-design.md). `agent_server.py` is the current
  Anthropic-only slice that will fold into the `AnthropicBackend` case of that design.
- **`dashboard.py`** — a Flask live dashboard on `:8000` (a superset of `agent_stub.py` — run
  **one or the other**, not both). It consumes the raw topic directly for a live per-user HR
  chart, receives Spark's anomaly POSTs at `/anomaly`, and reads the Delta table back
  (via `deltalake`) for the rolling-window table. Serves an inline-HTML page (no external
  assets) that polls `/data` every 2s; also builds a static snapshot when `window.__SNAPSHOT__`
  is present.
- **`demo.sh`** — one-command orchestrator: starts Kafka, launches dashboard+simulator+processor
  (short windows), auto-injects an anomaly, keeps event-time advancing, opens the dashboard, and
  tears everything down on Ctrl-C.
- **`docker-compose.yml`** — the *intended* infra (Kafka+Zookeeper+Spark cluster). See the
  Docker caveat below; the verified path uses native Homebrew Kafka instead.

The three streams' contract lives in **two places that must stay in sync**: the message keys
in `simulator.py` and the `RAW_SCHEMA` superset in `spark_processor.py`. Changing a producer
field name requires updating `RAW_SCHEMA` (and the `normalize()` projection) or windows go null.

## Commands

Prerequisites: Homebrew Kafka, a JDK 8/11/17 (**not** newer — Spark 3.5 rejects Java 16+),
and Python deps: `pip install pyspark==3.5.1 delta-spark==3.2.0 requests kafka-python numpy pandas`.
For the dashboard also: `pip install flask deltalake` (the windows panel degrades gracefully
without `deltalake`). For the AI agent (`agent_server.py`): `pip install fastapi "uvicorn[standard]"
anthropic deltalake` **plus** Anthropic credentials — `export ANTHROPIC_API_KEY=...` or
`ant auth login` (the zero-arg client resolves either). Without credentials the `/anomaly`
endpoint returns a graceful JSON error instead of a report.

### One-command demo

```bash
./demo.sh          # starts Kafka + all processes, injects an anomaly, opens the dashboard
                   # Ctrl-C tears everything down (stops Kafka, removes ./lakehouse)
```

Then open <http://localhost:8000>. It uses short 1-min windows so results appear in ~75s, and
auto-injects a high-HR resting anomaly ~30s in. `demo.sh` runs a pacer that emits a steady
heartbeat — without it, when both simulated users hit `rest` phases the event-time watermark
stalls and 1-min windows stop finalizing (see gotcha below).

### Live dashboard (manual)

Run `dashboard.py` **instead of** `agent_stub.py` (both bind `:8000`); point the processor's
anomaly POST at it (its `--agent-url` already defaults to `http://localhost:8000/anomaly`):

```bash
python3 dashboard.py        # live chart + anomaly feed + windows table at :8000
python3 simulator.py
python3 spark_processor.py
```

### AI triage agent (manual)

```bash
export ANTHROPIC_API_KEY=...        # or `ant auth login`
python3 agent_server.py             # clinical agent on :8000 (drop-in for agent_stub)
python3 simulator.py
python3 spark_processor.py          # POSTs anomalies to :8000/anomaly by default
```

Each anomaly returns JSON with `report`, `risk_level`, `baseline` (tool output), `fhir`
(the Observation), and `model`. Note it needs Delta history to have a baseline — run the
processor a while first, or it reasons from the reading alone.

### Local quickstart (verified path — no Docker)

Docker Desktop does not run on this machine ("Incompatible CPU"), so the stack runs with
**native Kafka in KRaft mode** (no Zookeeper). Since the Spark job uses the local runner, the
only infra it needs is Kafka on `localhost:9092` — the Python scripts are identical either way.

```bash
# One-time: install + a topic with a '.' is fine (ignore the metric-name warning)
brew install kafka
brew services start kafka
kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists \
  --topic wearables.raw --partitions 2 --replication-factor 1

# Spark 3.5 needs Java 8/11/17. Corretto 11 works; point JAVA_HOME at a supported JDK:
export JAVA_HOME=/Users/bensonchou/Library/Java/JavaVirtualMachines/corretto-11.0.12/Contents/Home

# Run each in its own terminal (all connect to localhost:9092):
python3 agent_stub.py                                  # 1. anomaly sink on :8000
python3 simulator.py                                   # 2. produce telemetry
python3 spark_processor.py                             # 3. stream processor

brew services stop kafka                               # tear down when done
```

The Kafka + Delta **jars** are fetched from Maven on the processor's first run (via
`spark.jars.packages`, cached in `~/.ivy2`) — no `--packages` flag needed.

### Fast verification (short windows)

With production 15-min windows, append-mode watermarking means the first window won't commit
to Delta for ~20 min. To exercise the exact same path in ~75s, shrink the window:

```bash
python3 spark_processor.py --window "1 minute" --watermark "15 seconds" --trigger "20 seconds"
```

Force an anomaly (natural HR never exceeds 140) by injecting high-HR `rest` packets:

```python
# python3 - ; sends to wearables.raw
import json, time, pandas as pd
from kafka import KafkaProducer
p = KafkaProducer(bootstrap_servers="localhost:9092",
                  value_serializer=lambda v: json.dumps(v).encode(), key_serializer=lambda k: k.encode())
for i in range(8):
    p.send("wearables.raw", key="user_anomaly", value={
        "device":"Apple_Watch","user_id":"user_anomaly","event":"rest",
        "metric":"heart_rate","hr_bpm":170+i,"ts":pd.Timestamp.utcnow().isoformat()})
    time.sleep(0.3)
p.flush()
```

Then confirm: `agent_stub.py` logs `=== ANOMALY @ /anomaly ===`, and the Delta table is
queryable as a batch source:

```python
spark.read.format("delta").load("./lakehouse/wearable_windows").orderBy("window_start").show()
```

### Inspect Kafka directly

```bash
kafka-console-consumer --bootstrap-server localhost:9092 --topic wearables.raw \
  --from-beginning --max-messages 5 --timeout-ms 8000
```

## Conventions / gotchas

- **`docker-compose.yml` is aspirational here** — Docker won't start on this CPU. It targets a
  containerized Kafka+Spark cluster; to actually use it you'd switch the processor's bootstrap
  to `kafka:29092`, mount output/checkpoint volumes, and route the anomaly POST to
  `host.docker.internal:8000`. The Homebrew path above is the working alternative.
- **`kafka-python` 3.x removed `NoBrokersAvailable`** — `simulator.py` imports it with a
  try/except fallback to the base `KafkaError`. Keep that guard if editing the imports.
- **`agent_server.py` sends `output_config` via `extra_body`** — older `anthropic` SDKs (e.g.
  0.72.0) don't type the `output_config`/`effort` kwarg and would raise `TypeError`; passing it
  through `extra_body` forwards it as JSON so the call constructs on old SDKs and still sends the
  right fields to the current API. Adaptive `thinking` is passed as a plain dict for the same
  reason. FastAPI route/model annotations use `Optional[...]`/`Dict[...]` (not `X | None` / `dict[...]`)
  because the anaconda runtime is Python 3.8, which can't evaluate those at import time.
- **macOS lacks GNU `timeout`** — use `kafka-console-consumer --timeout-ms` instead of piping
  through `timeout`.
- **Watermark stalls on bursty input** — append-mode windows finalize only when *event-time*
  advances past `window_end + watermark`. When both simulated users are in `rest` (packets every
  5 min), event-time barely moves and windows stop committing. This is expected event-time
  semantics, not a bug; `demo.sh`'s pacer keeps event-time moving. In production, size the
  watermark/window to the slowest expected source cadence.
- **`./lakehouse/`** (Delta table + `_checkpoints/`) is generated output; safe to delete for a
  clean run. Deleting only the checkpoint while keeping the table causes duplicate reprocessing.
