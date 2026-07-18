#!/usr/bin/env python3
"""
spark_processor.py — Structured Streaming consumer for wearable telemetry.

Pipeline:
  1. Read raw JSON from the Kafka topic `wearables.raw` (produced by simulator.py).
  2. Normalize the two disparate device streams (Apple_Watch per-reading HR and
     Oura_Ring 5-min aggregates) into one common schema.
  3. Align them with 15-minute tumbling event-time windows and compute rolling
     averages per user.
  4. In each micro-batch:
       (a) append finalized windows to a local Delta Lake table, and
       (b) POST any anomalous window (avg HR > 140 bpm with no workout activity)
           to a local AI agent endpoint.

Run locally (Kafka on localhost:9092):
    pip install pyspark==3.5.1 delta-spark==3.2.0 requests
    python spark_processor.py

The Kafka + Delta jars are fetched automatically on first run via
`spark.jars.packages`, so no manual --packages flag is required.
"""

from __future__ import annotations

import argparse
import json
import os

import requests
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# --------------------------------------------------------------------------- #
# Configuration defaults (all overridable via CLI / env)
# --------------------------------------------------------------------------- #

DEFAULT_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DEFAULT_TOPIC = os.getenv("KAFKA_TOPIC", "wearables.raw")
DEFAULT_DELTA_PATH = os.getenv("DELTA_PATH", "./lakehouse/wearable_windows")
DEFAULT_CHECKPOINT = os.getenv(
    "CHECKPOINT_PATH", "./lakehouse/_checkpoints/wearable_windows"
)
DEFAULT_AGENT_URL = os.getenv("AI_AGENT_URL", "http://localhost:8000/anomaly")

HR_ANOMALY_THRESHOLD = float(os.getenv("HR_ANOMALY_THRESHOLD", "140"))

# Kafka + Delta packages compatible with Spark 3.5 / Scala 2.12.
SPARK_PACKAGES = ",".join(
    [
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
        "io.delta:delta-spark_2.12:3.2.0",
    ]
)

# Permissive superset schema covering every key from BOTH device payloads.
# Fields absent in a given message parse as null (e.g. Oura rows have no hr_bpm).
RAW_SCHEMA = StructType(
    [
        StructField("device", StringType()),
        StructField("user_id", StringType()),
        StructField("ts", StringType()),
        # Apple_Watch fields
        StructField("event", StringType()),        # 'workout' | 'rest'
        StructField("metric", StringType()),
        StructField("hr_bpm", IntegerType()),
        # Oura_Ring fields
        StructField("window_seconds", IntegerType()),
        StructField("avg_hr", DoubleType()),
        StructField("min_hr", IntegerType()),
        StructField("max_hr", IntegerType()),
        StructField("hrv_ms", DoubleType()),
        StructField("resp_rate", DoubleType()),
        StructField("sleep_score", IntegerType()),
        StructField("readiness_score", IntegerType()),
    ]
)


# --------------------------------------------------------------------------- #
# Spark session
# --------------------------------------------------------------------------- #

def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("wearable-stream-processor")
        .master("local[*]")
        .config("spark.jars.packages", SPARK_PACKAGES)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Keep the local shuffle small — this is a dev-scale workload.
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


# --------------------------------------------------------------------------- #
# Stream transformations
# --------------------------------------------------------------------------- #

def read_kafka(spark: SparkSession, bootstrap: str, topic: str) -> DataFrame:
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .load()
    )


def normalize(raw: DataFrame) -> DataFrame:
    """Parse JSON and unify both device shapes into a common schema."""
    parsed = raw.select(
        F.from_json(F.col("value").cast("string"), RAW_SCHEMA).alias("d")
    ).select("d.*")

    return parsed.select(
        F.to_timestamp("ts").alias("event_time"),
        F.col("user_id"),
        F.col("device").alias("source"),
        # Apple uses hr_bpm (int), Oura uses avg_hr (double); unify to one HR column.
        F.coalesce(F.col("hr_bpm").cast("double"), F.col("avg_hr")).alias("hr"),
        # Activity signal: only Apple 'workout' readings count. Oura -> 0.
        F.when(
            (F.col("device") == "Apple_Watch") & (F.col("event") == "workout"), 1
        )
        .otherwise(0)
        .alias("activity"),
    ).where(F.col("event_time").isNotNull() & F.col("hr").isNotNull())


def windowed_aggregate(df: DataFrame, window: str, watermark: str) -> DataFrame:
    """15-minute tumbling windows per user with rolling HR stats."""
    return (
        df.withWatermark("event_time", watermark)
        .groupBy(F.window("event_time", window), "user_id")
        .agg(
            F.round(F.avg("hr"), 1).alias("avg_hr"),
            F.max("hr").alias("max_hr"),
            F.min("hr").alias("min_hr"),
            F.count("hr").alias("sample_count"),
            F.sum("activity").alias("workout_samples"),
            F.collect_set("source").alias("sources"),
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "user_id",
            "avg_hr",
            "max_hr",
            "min_hr",
            "sample_count",
            "workout_samples",
            "sources",
        )
    )


# --------------------------------------------------------------------------- #
# Sink: Delta write + anomaly POST (runs per micro-batch on the driver)
# --------------------------------------------------------------------------- #

def make_batch_handler(delta_path: str, agent_url: str):
    def process_batch(batch_df: DataFrame, batch_id: int) -> None:
        # Cache: we scan it twice (Delta write + anomaly filter).
        batch_df.persist()
        try:
            n = batch_df.count()
            print(f"[batch {batch_id}] {n} finalized window(s)")
            if n == 0:
                return

            # (a) Commit to the Delta Lake historical table.
            (
                batch_df.write.format("delta")
                .mode("append")
                .partitionBy("user_id")
                .save(delta_path)
            )

            # (b) Detect anomalies: elevated HR with no workout activity.
            anomalies = batch_df.where(
                (F.col("avg_hr") > HR_ANOMALY_THRESHOLD)
                & (F.col("workout_samples") == 0)
            )
            for row in anomalies.collect():
                post_anomaly(agent_url, row)
        finally:
            batch_df.unpersist()

    return process_batch


def post_anomaly(agent_url: str, row) -> None:
    """Fire a single anomaly alert to the AI agent. Never raises."""
    payload = {
        "user_id": row["user_id"],
        "window_start": row["window_start"].isoformat(),
        "window_end": row["window_end"].isoformat(),
        "avg_hr": row["avg_hr"],
        "max_hr": row["max_hr"],
        "sample_count": row["sample_count"],
        "sources": list(row["sources"]),
        "reason": (
            f"avg_hr {row['avg_hr']} bpm exceeds {HR_ANOMALY_THRESHOLD:.0f} "
            f"with no workout activity"
        ),
    }
    try:
        resp = requests.post(agent_url, json=payload, timeout=5)
        print(
            f"  -> anomaly POST {agent_url} [{resp.status_code}] "
            f"user={payload['user_id']} avg_hr={payload['avg_hr']}"
        )
    except requests.RequestException as exc:
        # Log and continue — a flaky agent must not kill the stream.
        print(f"  !! anomaly POST failed ({exc}); payload={json.dumps(payload)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Wearable Kafka -> Delta stream processor")
    parser.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--delta-path", default=DEFAULT_DELTA_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--agent-url", default=DEFAULT_AGENT_URL)
    parser.add_argument("--window", default="15 minutes")
    parser.add_argument("--watermark", default="5 minutes")
    parser.add_argument("--trigger", default="30 seconds")
    args = parser.parse_args()

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    raw = read_kafka(spark, args.bootstrap, args.topic)
    unified = normalize(raw)
    windows = windowed_aggregate(unified, args.window, args.watermark)

    print(
        f"Streaming {args.topic!r} @ {args.bootstrap} -> Delta {args.delta_path!r}; "
        f"anomalies -> {args.agent_url}. Ctrl-C to stop."
    )

    query = (
        windows.writeStream.outputMode("append")
        .foreachBatch(make_batch_handler(args.delta_path, args.agent_url))
        .option("checkpointLocation", args.checkpoint)
        .trigger(processingTime=args.trigger)
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
