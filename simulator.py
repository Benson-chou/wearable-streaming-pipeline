#!/usr/bin/env python3
"""
simulator.py — Wearable telemetry simulator.

Simulates two users, each emitting two independent streaming threads:

  * Apple_Watch : high-frequency, erratic heart-rate packets.
                  - during a 'workout' event  -> one packet every 10 seconds
                  - during a 'rest' event      -> one packet every 5 minutes
  * Oura_Ring   : rigid, aggregated 5-minute health chunks
                  (keys: avg_hr, sleep_score, ...).

Each stream runs in its own thread and publishes raw JSON to a Kafka topic
continuously until interrupted (Ctrl-C).

Requires: numpy, pandas, kafka-python
    pip install numpy pandas kafka-python
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import threading
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from kafka import KafkaProducer
try:
    # kafka-python 2.x
    from kafka.errors import NoBrokersAvailable
except ImportError:
    # kafka-python 3.x removed this name; fall back to the base error.
    from kafka.errors import KafkaError as NoBrokersAvailable

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DEFAULT_TOPIC = os.getenv("KAFKA_TOPIC", "wearables.raw")

USERS = ["user_alice", "user_bob"]

# Apple Watch cadence (seconds)
WORKOUT_INTERVAL = 10          # HR packet every 10s while working out
REST_INTERVAL = 5 * 60         # HR packet every 5 min while resting

# Oura Ring cadence (seconds)
OURA_INTERVAL = 5 * 60         # aggregated chunk every 5 min

# How long a simulated workout / rest phase lasts (seconds).
# Kept short-ish so you can observe both cadences quickly in a demo.
WORKOUT_PHASE_RANGE = (120, 300)   # 2–5 min of workout
REST_PHASE_RANGE = (300, 900)      # 5–15 min of rest

# Global stop flag, flipped by SIGINT/SIGTERM.
_stop = threading.Event()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def now_iso() -> str:
    """UTC timestamp in ISO-8601, via pandas for consistent formatting."""
    return pd.Timestamp.utcnow().isoformat()


def build_producer(bootstrap: str) -> KafkaProducer:
    """Create a Kafka producer that serializes dict -> raw JSON bytes."""
    return KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        linger_ms=50,
        retries=3,
    )


def interruptible_sleep(seconds: float) -> None:
    """Sleep that returns early if a shutdown was requested."""
    _stop.wait(timeout=seconds)


# --------------------------------------------------------------------------- #
# Stream: Apple Watch
# --------------------------------------------------------------------------- #

@dataclass
class HeartState:
    """Per-user drifting heart-rate baseline for realistic-looking noise."""
    baseline: float

    def sample(self, event: str, rng: np.random.Generator) -> int:
        """Return one HR reading, erratic and event-dependent."""
        if event == "workout":
            # Elevated, volatile: baseline pushed up, wide jitter, occasional spikes.
            hr = self.baseline + 55 + rng.normal(0, 8)
            if rng.random() < 0.10:            # sporadic spike / dropout
                hr += rng.normal(0, 20)
        else:  # rest
            hr = self.baseline + rng.normal(0, 4)
        # Random-walk the baseline a little so users diverge over time.
        self.baseline += rng.normal(0, 0.5)
        self.baseline = float(np.clip(self.baseline, 48, 75))
        return int(np.clip(round(hr), 40, 200))


def apple_watch_stream(user: str, producer: KafkaProducer, topic: str, seed: int) -> None:
    """High-frequency, erratic HR stream with alternating workout/rest phases."""
    rng = np.random.default_rng(seed)
    state = HeartState(baseline=float(rng.uniform(55, 68)))

    # Start each user in a random phase so the two users aren't in lockstep.
    event = rng.choice(["workout", "rest"])
    phase_ends = time.monotonic() + _phase_duration(event, rng)

    while not _stop.is_set():
        # Transition between workout and rest when the current phase elapses.
        if time.monotonic() >= phase_ends:
            event = "rest" if event == "workout" else "workout"
            phase_ends = time.monotonic() + _phase_duration(event, rng)

        hr = state.sample(event, rng)
        packet = {
            "device": "Apple_Watch",
            "user_id": user,
            "event": event,                    # 'workout' | 'rest'
            "metric": "heart_rate",
            "hr_bpm": hr,
            "ts": now_iso(),
        }
        producer.send(topic, key=user, value=packet)

        interval = WORKOUT_INTERVAL if event == "workout" else REST_INTERVAL
        interruptible_sleep(interval)


def _phase_duration(event: str, rng: np.random.Generator) -> float:
    lo, hi = WORKOUT_PHASE_RANGE if event == "workout" else REST_PHASE_RANGE
    return float(rng.uniform(lo, hi))


# --------------------------------------------------------------------------- #
# Stream: Oura Ring
# --------------------------------------------------------------------------- #

def oura_ring_stream(user: str, producer: KafkaProducer, topic: str, seed: int) -> None:
    """Rigid 5-minute aggregated health chunks."""
    rng = np.random.default_rng(seed)

    while not _stop.is_set():
        # Simulate the underlying per-second HR that Oura would internally
        # aggregate over the 5-minute window, then roll it up with pandas.
        n = OURA_INTERVAL  # one synthetic sample per second in the window
        base = rng.uniform(55, 70)
        samples = base + rng.normal(0, 6, size=n)
        hr_series = pd.Series(np.clip(samples, 40, 180))

        chunk = {
            "device": "Oura_Ring",
            "user_id": user,
            "window_seconds": OURA_INTERVAL,
            "avg_hr": round(float(hr_series.mean()), 1),
            "min_hr": int(hr_series.min()),
            "max_hr": int(hr_series.max()),
            "hrv_ms": round(float(rng.uniform(30, 90)), 1),
            "resp_rate": round(float(rng.uniform(12, 18)), 1),
            "sleep_score": int(np.clip(rng.normal(78, 10), 0, 100)),
            "readiness_score": int(np.clip(rng.normal(75, 12), 0, 100)),
            "ts": now_iso(),
        }
        producer.send(topic, key=user, value=chunk)

        interruptible_sleep(OURA_INTERVAL)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Wearable telemetry -> Kafka simulator")
    parser.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP,
                        help=f"Kafka bootstrap servers (default: {DEFAULT_BOOTSTRAP})")
    parser.add_argument("--topic", default=DEFAULT_TOPIC,
                        help=f"Kafka topic (default: {DEFAULT_TOPIC})")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    args = parser.parse_args()

    # Graceful shutdown on Ctrl-C / docker stop.
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())

    try:
        producer = build_producer(args.bootstrap)
    except NoBrokersAvailable:
        raise SystemExit(
            f"Could not reach Kafka at {args.bootstrap!r}. "
            "Is the broker up (docker compose up -d)?"
        )

    print(f"Streaming to topic {args.topic!r} on {args.bootstrap!r}. Ctrl-C to stop.")

    threads: list[threading.Thread] = []
    for i, user in enumerate(USERS):
        # Distinct seeds per (user, stream) so the four threads are independent.
        threads.append(threading.Thread(
            target=apple_watch_stream, args=(user, producer, args.topic, args.seed + i * 10 + 1),
            name=f"{user}:Apple_Watch", daemon=True,
        ))
        threads.append(threading.Thread(
            target=oura_ring_stream, args=(user, producer, args.topic, args.seed + i * 10 + 2),
            name=f"{user}:Oura_Ring", daemon=True,
        ))

    for t in threads:
        t.start()

    # Keep the main thread alive until a shutdown is requested.
    try:
        while not _stop.is_set():
            interruptible_sleep(1)
    finally:
        print("\nShutting down, flushing producer...")
        producer.flush(timeout=10)
        producer.close(timeout=10)
        print("Done.")


if __name__ == "__main__":
    main()
