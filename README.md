# Wearable Streaming Pipeline

A local, end-to-end streaming pipeline for wearable health telemetry:
**Kafka → Spark Structured Streaming → Delta Lake**, with real-time anomaly
detection and a live web dashboard.

Two simulated users each stream from two devices with **disparate schemas** — an
Apple Watch (erratic per-reading heart rate) and an Oura Ring (rigid 5-minute
aggregates). Spark aligns them into common 15-minute tumbling windows, computes
rolling averages, persists each window to Delta for historical batch queries, and
raises an alert whenever a window shows an elevated heart rate with no activity.

## Architecture

```
simulator.py ──JSON──▶ Kafka topic `wearables.raw` ──▶ spark_processor.py ──┬─▶ Delta table (./lakehouse/wearable_windows)
                                              │                             └─▶ POST anomaly ─▶ agent_stub.py OR dashboard.py (:8000)
                              dashboard.py ◀──┘ (also consumes the raw topic directly for the live chart)
```

| File | Role |
|------|------|
| `simulator.py` | Produces raw device JSON to Kafka — 2 users × 2 devices, 4 threads. Apple Watch sends HR every 10s in `workout` / every 5min in `rest`; Oura sends 5-min aggregates (`avg_hr`, `sleep_score`, …). |
| `spark_processor.py` | Reads the topic, normalizes both device shapes into `(event_time, user_id, hr, activity)`, applies 15-min tumbling windows per user, writes finalized windows to Delta (partitioned by user), and POSTs anomalies. |
| `dashboard.py` | Live Flask dashboard at `:8000` — per-user HR chart, anomaly feed, and the rolling-window table (read back from Delta). Superset of `agent_stub.py`. |
| `agent_server.py` | Clinical-triage **AI agent** (FastAPI, `:8000`). Runs a Claude tool-use loop over `fetch_historical_trends` (7-day Delta baseline) and `convert_to_fhir` (FHIR R4 Observation), then emits a Clinician Action Report with a risk level. Needs an Anthropic API key. |
| `agent_stub.py` | Minimal stand-in "AI agent" that logs the anomaly POSTs. |
| `demo.sh` | One-command demo: starts everything, injects an anomaly, opens the dashboard, tears down on Ctrl-C. |
| `docker-compose.yml` | Intended containerized infra (Kafka + Spark cluster) — see the caveat below. |

## Anomaly rule

Both streams are unified, then per 15-minute window:

> **`avg_hr > 140 bpm` AND `workout_samples == 0`** → POST an alert.

Apple `workout` readings count as activity (`1`); Oura carries no activity signal
(`0`). So a window with a high average heart rate but *no workout activity* — an
elevated resting heart rate — is flagged, while a hard workout is not.

## Quickstart

**Prerequisites:** [Homebrew](https://brew.sh) Kafka, a JDK 8/11/17 (**not** newer —
Spark 3.5 rejects Java 16+), and Python deps:

```bash
brew install kafka
pip install pyspark==3.5.1 delta-spark==3.2.0 requests kafka-python numpy pandas flask deltalake
# for the AI triage agent (agent_server.py):
pip install fastapi "uvicorn[standard]" anthropic        # plus: export ANTHROPIC_API_KEY=...
```

**One-command demo:**

```bash
./demo.sh          # starts Kafka + all processes, injects an anomaly, opens the dashboard
                   # Ctrl-C tears everything down
```

Then open <http://localhost:8000>. It uses short 1-minute windows so results appear
in ~75s and auto-injects a high-HR resting anomaly ~30s in.

**Run the pieces manually** (each in its own terminal):

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 11)
python3 dashboard.py         # or agent_stub.py — both bind :8000
python3 simulator.py
python3 spark_processor.py   # production 15-min windows; add --window "1 minute" for a fast demo
```

Query the persisted history as a batch source:

```python
spark.read.format("delta").load("./lakehouse/wearable_windows").orderBy("window_start").show()
```

## Notes

- **Docker**: `docker-compose.yml` targets a containerized Kafka + Spark cluster, but
  the verified path uses **native Homebrew Kafka in KRaft mode** (no Zookeeper). To use
  the compose stack you'd point the processor at `kafka:29092`, mount the output/checkpoint
  volumes, and route the anomaly POST to `host.docker.internal:8000`.
- **Event-time watermarking**: append-mode windows finalize only once event-time passes
  `window_end + watermark`. With bursty sources (both users resting → packets every 5 min)
  the watermark can stall; `demo.sh` runs a pacer to keep it advancing. Size the
  watermark/window to your slowest expected source cadence in production.

- **AI agent layer (in progress)**: `agent_server.py` is the current Anthropic-only slice. The
  target is a local-first, provider-portable agent — Ollama as a free local placeholder, Claude
  at scale, a fallback/compare mode, and a medical-tuned model consulted as a tool — with a real
  tool-calling loop. Full design: [`docs/agent-design.md`](./docs/agent-design.md).

See [`CLAUDE.md`](./CLAUDE.md) for deeper architecture notes and gotchas, and
[`docs/agent-design.md`](./docs/agent-design.md) for the agent-layer design.
