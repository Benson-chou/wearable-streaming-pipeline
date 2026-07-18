#!/usr/bin/env bash
#
# demo.sh — one-command live demo of the wearable streaming pipeline.
#
# Brings up native Kafka, launches the dashboard + simulator + Spark processor,
# auto-injects an anomaly, keeps event-time advancing so the 1-minute demo windows
# finalize promptly, opens the dashboard, and tears everything down on Ctrl-C.
#
#   ./demo.sh
#   open http://localhost:8000     # (also opened automatically on macOS)
#
# Uses short 1-min windows so results appear in ~75s. For the production 15-min
# cadence, run spark_processor.py directly (see CLAUDE.md).

set -uo pipefail
cd "$(dirname "$0")"

BOOTSTRAP="${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}"
TOPIC="${KAFKA_TOPIC:-wearables.raw}"

# --- Java: Spark 3.5 needs a JDK 8/11/17 -----------------------------------
export JAVA_HOME="${JAVA_HOME:-$(/usr/libexec/java_home -v 11 2>/dev/null || /usr/libexec/java_home -v 17 2>/dev/null || true)}"
if [ -z "${JAVA_HOME:-}" ]; then
  echo "ERROR: no JDK 8/11/17 found. Install one (e.g. 'brew install openjdk@17') and set JAVA_HOME." >&2
  exit 1
fi

KAFKA_BIN="$(brew --prefix 2>/dev/null)/opt/kafka/bin"
if [ ! -x "$KAFKA_BIN/kafka-topics" ]; then
  echo "ERROR: Homebrew Kafka not found. Run 'brew install kafka'." >&2
  exit 1
fi

LOGDIR="$(mktemp -d)"
PIDS=()

cleanup() {
  echo ""
  echo "Tearing down..."
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  pkill -f "dashboard.py"       2>/dev/null || true
  pkill -f "simulator.py"       2>/dev/null || true
  pkill -f "spark_processor.py" 2>/dev/null || true
  brew services stop kafka >/dev/null 2>&1 || true
  rm -rf lakehouse
  echo "Done. (logs were in $LOGDIR)"
}
trap cleanup EXIT INT TERM

echo "==> Starting Kafka (KRaft)..."
brew services start kafka >/dev/null 2>&1 || true
host="${BOOTSTRAP%%:*}"; port="${BOOTSTRAP##*:}"
# Fast, reliable readiness probe: wait for the broker's TCP port to accept (up to ~150s).
ready=0
for i in $(seq 1 150); do
  if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then exec 3>&- 3<&-; ready=1; break; fi
  sleep 1
done
[ "$ready" = 1 ] || { echo "ERROR: Kafka port $BOOTSTRAP never opened." >&2; exit 1; }
# Give the broker a moment to finish coming up, then ensure the topic exists.
for i in $(seq 1 15); do
  "$KAFKA_BIN/kafka-topics" --bootstrap-server "$BOOTSTRAP" --create --if-not-exists \
    --topic "$TOPIC" --partitions 2 --replication-factor 1 >/dev/null 2>&1 && break
  sleep 2
done
echo "    Kafka ready; topic '$TOPIC' present."

rm -rf lakehouse

echo "==> Launching dashboard, simulator, Spark processor..."
python3 -u dashboard.py > "$LOGDIR/dashboard.log" 2>&1 & PIDS+=($!)
sleep 2
python3 -u simulator.py > "$LOGDIR/sim.log" 2>&1 & PIDS+=($!)
sleep 3
python3 -u spark_processor.py \
  --window "1 minute" --watermark "15 seconds" --trigger "15 seconds" \
  --agent-url "http://localhost:8000/anomaly" > "$LOGDIR/spark.log" 2>&1 & PIDS+=($!)

# Pacer: emits a steady heartbeat so event-time keeps advancing (bursty rest phases
# would otherwise stall the watermark and delay window finalization), and injects one
# high-HR resting anomaly ~32s in so the alert path is demonstrated automatically.
python3 -u - > "$LOGDIR/pacer.log" 2>&1 <<'PY' & PIDS+=($!)
import json, time, numpy as np, pandas as pd
from kafka import KafkaProducer
p = KafkaProducer(bootstrap_servers="localhost:9092",
                  value_serializer=lambda v: json.dumps(v).encode(),
                  key_serializer=lambda k: k.encode())
n = 0
while True:
    ts = pd.Timestamp.utcnow().isoformat()
    p.send("wearables.raw", key="user_alice", value={"device":"Apple_Watch","user_id":"user_alice",
        "event":"workout","metric":"heart_rate","hr_bpm":int(122+np.random.normal(0,9)),"ts":ts})
    p.send("wearables.raw", key="user_bob", value={"device":"Apple_Watch","user_id":"user_bob",
        "event":"rest","metric":"heart_rate","hr_bpm":int(58+np.random.normal(0,4)),"ts":ts})
    if n == 8:  # ~32s in
        for i in range(10):
            p.send("wearables.raw", key="user_anomaly", value={"device":"Apple_Watch",
                "user_id":"user_anomaly","event":"rest","metric":"heart_rate",
                "hr_bpm":166+i,"ts":pd.Timestamp.utcnow().isoformat()})
        print("injected anomaly burst")
    p.flush(); n += 1; time.sleep(4)
PY

sleep 2
command -v open >/dev/null 2>&1 && open "http://localhost:8000" 2>/dev/null || true

cat <<EOF

======================================================================
  Live dashboard:  http://localhost:8000   (auto-refreshes every 2s)

  * heart-rate lines stream in immediately
  * an anomaly is injected ~30s in — watch the alert + red tile appear
  * rolling 15-min-style windows fill the table as they finalize

  Press Ctrl-C to stop everything and tear down.
======================================================================
EOF

wait
