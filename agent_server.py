#!/usr/bin/env python3
"""
agent_server.py — Local-first, provider-portable clinical triage agent.

A FastAPI service that receives wearable anomaly alerts (POSTed by spark_processor.py),
runs a *real tool-calling loop* over a small tool set, applies a deterministic safety
floor, structures the reading as FHIR, and returns a Clinician Action Report plus a
`tool_trace` of the agent's decisions.

The LLM backend is pluggable (see docs/agent-design.md):
  * OllamaBackend   — local, free (default). The Phase-1 focus.
  * AnthropicBackend — Claude, for scale-up. Same tools, same loop.

Select with AGENT_BACKEND=ollama|anthropic (comma-list = fallback chain, tried in order).

Tools the agent can call:
  fetch_historical_trends(user_id, days)      — 7-day HR baseline from Delta Lake
  recall_patient_memory(user_id)              — prior assessments (cross-session memory)
  lookup_vitals_reference(metric)             — clinical normal ranges (deterministic)
  analyze_hr_trend(user_id, current_hr)       — EWMA trend + z-score outlier from Delta
  consult_medical_model(question, context)    — ask a medical-tuned Ollama model
  notify_clinician(user_id, risk_level, msg)  — real action: log an outbound alert
  submit_report(risk_level, ...)              — structured finish; ends the loop
convert_to_fhir is applied deterministically after the loop (never a valid-FHIR gamble).
Each assessment is persisted to ./agent_memory (recall_patient_memory reads it back) and the
FHIR Observation is written to ./fhir_store — both OUTSIDE ./lakehouse so they survive teardown.

Run:
    pip install fastapi "uvicorn[standard]" ollama anthropic deltalake pandas
    ollama serve &  &&  ollama pull llama3.1:8b  &&  ollama pull meditron
    python3 agent_server.py            # serves http://localhost:8000

Decision support for demonstration — not a medical device, not a diagnosis.
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

try:
    from deltalake import DeltaTable
except Exception:  # pragma: no cover
    DeltaTable = None

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BACKENDS = [b.strip() for b in os.getenv("AGENT_BACKEND", "ollama").split(",") if b.strip()]
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_MEDICAL_MODEL = os.getenv("OLLAMA_MEDICAL_MODEL", "meditron")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
DELTA_PATH = os.getenv("DELTA_PATH", "./lakehouse/wearable_windows")
PORT = int(os.getenv("AGENT_PORT", "8000"))
BASELINE_DAYS = int(os.getenv("BASELINE_DAYS", "7"))
MAX_TURNS = int(os.getenv("AGENT_MAX_TURNS", "16"))

# Cross-session memory + FHIR store live OUTSIDE ./lakehouse so tearing the pipeline
# down (which wipes lakehouse) does not erase the agent's learned history.
MEMORY_DIR = os.getenv("AGENT_MEMORY_DIR", "./agent_memory")
FHIR_STORE_DIR = os.getenv("FHIR_STORE_DIR", "./fhir_store")
MEMORY_RECALL_LIMIT = int(os.getenv("MEMORY_RECALL_LIMIT", "10"))

_SEVERITY = {"LOW": 0, "ELEVATED": 1, "CRITICAL": 2}


# --------------------------------------------------------------------------- #
# Persistence helpers (cross-session memory + FHIR store, both JSONL/file-based)
# --------------------------------------------------------------------------- #

def _assessments_path() -> str:
    return os.path.join(MEMORY_DIR, "assessments.jsonl")


def _notifications_path() -> str:
    return os.path.join(MEMORY_DIR, "notifications.jsonl")


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _persist_fhir(user_id: str, fhir: Dict[str, Any]) -> str:
    ts = fhir.get("effectiveDateTime") or dt.datetime.now(dt.timezone.utc).isoformat()
    safe = "".join(c if c.isalnum() else "_" for c in ts)
    d = os.path.join(FHIR_STORE_DIR, user_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{safe}.json")
    with open(path, "w") as f:
        json.dump(fhir, f, indent=2)
    return path


# --------------------------------------------------------------------------- #
# Tools (provider-neutral implementations)
# --------------------------------------------------------------------------- #

def fetch_historical_trends(user_id: str, days: int = BASELINE_DAYS) -> Dict[str, Any]:
    """Read the local Delta table and summarize the user's N-day HR baseline."""
    if DeltaTable is None:
        return {"available": False, "reason": "deltalake not installed"}
    try:
        df = DeltaTable(DELTA_PATH).to_pandas()
    except Exception as exc:
        return {"available": False, "reason": f"could not read Delta table: {exc}"}

    df = df[df["user_id"] == user_id]
    if df.empty:
        return {"available": False, "reason": f"no history for {user_id}"}

    import pandas as pd

    ws = pd.to_datetime(df["window_start"], errors="coerce")
    try:
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
        ws_cmp = ws.dt.tz_localize(None) if ws.dt.tz is not None else ws
        recent = df[ws_cmp >= cutoff]
        if not recent.empty:
            df = recent
    except Exception:
        pass

    resting = df[df["workout_samples"] == 0]
    out: Dict[str, Any] = {
        "available": True,
        "user_id": user_id,
        "window_days": days,
        "window_count": int(len(df)),
        "baseline_avg_hr": round(float(df["avg_hr"].mean()), 1),
        "observed_max_hr": float(df["max_hr"].max()),
        "observed_min_hr": float(df["min_hr"].min()),
        "resting_window_count": int(len(resting)),
    }
    if not resting.empty:
        out["resting_baseline_hr"] = round(float(resting["avg_hr"].mean()), 1)
        out["resting_max_hr"] = float(resting["max_hr"].max())
    return out


_VITALS_REFERENCE = {
    "heart_rate": {
        "unit": "bpm",
        "resting_normal_adult": "60-100",
        "bradycardia_below": 60,
        "tachycardia_above": 100,
        "concerning_resting_above": 150,
        "dangerous_above": 180,
        "note": "Resting HR sustained above ~150 bpm warrants attention; above 180 is dangerous.",
    },
    "resp_rate": {"unit": "breaths/min", "normal_adult": "12-20", "concerning_above": 24},
    "hrv": {"unit": "ms", "note": "Higher is generally better; low HRV can indicate stress/strain."},
    "spo2": {"unit": "%", "normal": "95-100", "concerning_below": 92},
}


def lookup_vitals_reference(metric: str) -> Dict[str, Any]:
    """Return clinical normal ranges for a vital sign (deterministic, no model)."""
    key = metric.lower().replace(" ", "_")
    if key in _VITALS_REFERENCE:
        return {"metric": key, **_VITALS_REFERENCE[key]}
    return {"metric": metric, "available": False,
            "known_metrics": list(_VITALS_REFERENCE.keys())}


def consult_medical_model(question: str, context: str = "") -> Dict[str, Any]:
    """Ask a medical-tuned Ollama model for a focused specialist opinion."""
    try:
        import ollama
        cli = ollama.Client(host=OLLAMA_HOST)
        resp = cli.chat(
            model=OLLAMA_MEDICAL_MODEL,
            messages=[
                {"role": "system", "content": "You are a clinical specialist. Give a brief, "
                 "focused opinion (2-4 sentences). You are decision support, not a diagnosis."},
                {"role": "user", "content": f"{question}\n\nContext:\n{context}"},
            ],
            options={"temperature": 0.2},
        )
        return {"model": OLLAMA_MEDICAL_MODEL, "opinion": resp["message"]["content"].strip()}
    except Exception as exc:
        return {"error": f"medical model unavailable: {exc}"}


def recall_patient_memory(user_id: str, limit: int = MEMORY_RECALL_LIMIT) -> Dict[str, Any]:
    """Recall this user's prior anomaly assessments from persistent cross-session memory."""
    records = [r for r in _read_jsonl(_assessments_path()) if r.get("user_id") == user_id]
    if not records:
        return {"available": False, "user_id": user_id, "prior_count": 0,
                "reason": f"no prior assessments recorded for {user_id} (first-time event)"}
    records = records[-limit:]
    levels = [r.get("risk_level") for r in records]
    return {
        "available": True,
        "user_id": user_id,
        "prior_count": len(records),
        "last_assessment_at": records[-1].get("ts"),
        "risk_levels_seen": levels,
        "critical_count": sum(1 for lvl in levels if lvl == "CRITICAL"),
        "recent": [
            {"ts": r.get("ts"), "avg_hr": r.get("avg_hr"), "risk_level": r.get("risk_level"),
             "summary": r.get("summary"), "recommended_action": r.get("recommended_action")}
            for r in records[-5:]
        ],
    }


def analyze_hr_trend(user_id: str, days: int = BASELINE_DAYS,
                     current_hr: Optional[float] = None) -> Dict[str, Any]:
    """EWMA trend of the user's recent resting HR + how the current reading compares."""
    if DeltaTable is None:
        return {"available": False, "reason": "deltalake not installed"}
    try:
        import numpy as np  # noqa: F401
        import pandas as pd
        df = DeltaTable(DELTA_PATH).to_pandas()
    except Exception as exc:
        return {"available": False, "reason": f"could not read Delta table: {exc}"}

    df = df[df["user_id"] == user_id].copy()
    if df.empty:
        return {"available": False, "reason": f"no history for {user_id}"}

    # Prefer resting windows for a stable trend; fall back to all if too few.
    resting = df[df["workout_samples"] == 0].copy()
    use = resting if len(resting) >= 3 else df.copy()
    use["ws"] = pd.to_datetime(use["window_start"], errors="coerce", utc=True)
    use = use.sort_values("ws")
    series = use["avg_hr"].astype(float).reset_index(drop=True)

    span = min(10, max(2, len(series)))
    ewma = series.ewm(span=span).mean()
    recent = series.tail(20)
    mean = float(recent.mean())
    std = float(recent.std(ddof=0)) or 1.0
    direction = ("rising" if ewma.iloc[-1] > ewma.iloc[0] + 1
                 else "falling" if ewma.iloc[-1] < ewma.iloc[0] - 1 else "stable")

    out: Dict[str, Any] = {
        "available": True, "user_id": user_id, "window_count": int(len(use)),
        "ewma_resting_hr": round(float(ewma.iloc[-1]), 1),
        "recent_mean_hr": round(mean, 1), "recent_std_hr": round(std, 1),
        "trend": direction,
    }
    if current_hr is not None:
        try:
            z = (float(current_hr) - mean) / std
            out["current_hr"] = float(current_hr)
            out["z_score"] = round(z, 1)
            out["is_statistical_outlier"] = bool(abs(z) >= 3)
        except Exception:
            pass
    return out


def notify_clinician(user_id: str, risk_level: str, message: str) -> Dict[str, Any]:
    """Record an outbound clinician alert — a real action, logged to persistent memory."""
    record = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "user_id": user_id, "risk_level": str(risk_level).upper(),
        "message": message, "channel": "local-log",
    }
    _append_jsonl(_notifications_path(), record)
    return {"delivered": True, "channel": "local-log", "user_id": user_id,
            "risk_level": record["risk_level"],
            "note": "notification recorded to agent_memory/notifications.jsonl"}


def convert_to_fhir(user_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Structure a heart-rate reading into a FHIR R4 Observation (deterministic)."""
    hr = metrics.get("heart_rate_bpm", metrics.get("avg_hr"))
    effective = metrics.get("effective_time") or metrics.get("window_end") \
        or dt.datetime.now(dt.timezone.utc).isoformat()
    baseline = metrics.get("baseline_hr")

    interp_code, interp_display = "N", "Normal"
    if hr is not None:
        if hr > 140:
            interp_code, interp_display = "HH", "Critically high"
        elif baseline is not None and hr > float(baseline) * 1.5:
            interp_code, interp_display = "H", "High"

    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
            "code": "vital-signs", "display": "Vital Signs"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}],
                 "text": "Heart rate"},
        "subject": {"reference": f"Patient/{user_id}", "display": user_id},
        "effectiveDateTime": effective,
        "valueQuantity": {"value": hr, "unit": "beats/minute",
                          "system": "http://unitsofmeasure.org", "code": "/min"},
        "interpretation": [{"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
            "code": interp_code, "display": interp_display}]}],
    }
    components = []
    if metrics.get("max_hr") is not None:
        components.append(_hr_component("8873-2", "Heart rate --maximum", metrics["max_hr"]))
    if metrics.get("min_hr") is not None:
        components.append(_hr_component("8872-4", "Heart rate --minimum", metrics["min_hr"]))
    if components:
        obs["component"] = components
    notes = []
    if baseline is not None:
        notes.append(f"User {BASELINE_DAYS}-day baseline avg HR: {baseline} bpm.")
    if metrics.get("activity") is not None:
        notes.append(f"Activity samples in window: {metrics['activity']}.")
    if notes:
        obs["note"] = [{"text": " ".join(notes)}]
    return obs


def _hr_component(loinc: str, display: str, value: Any) -> Dict[str, Any]:
    return {"code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": display}]},
            "valueQuantity": {"value": value, "unit": "beats/minute",
                              "system": "http://unitsofmeasure.org", "code": "/min"}}


# submit_report is a terminal "tool" — its execution just echoes the structured args;
# the loop stops once the agent calls it. The args ARE the agent's verdict.
def submit_report(risk_level: str, summary: str, reasoning: str,
                  recommended_action: str) -> Dict[str, Any]:
    return {"accepted": True, "risk_level": risk_level}


# --------------------------------------------------------------------------- #
# Provider-neutral tool specs + dispatch
# --------------------------------------------------------------------------- #

TOOL_SPECS = [
    {
        "name": "fetch_historical_trends",
        "description": "Read the Delta Lake table for a wearable user's recent heart-rate "
                       "baseline (overall + resting avg, observed min/max, window counts). "
                       "Call this FIRST to learn the user's personal normal.",
        "parameters": {"type": "object", "properties": {
            "user_id": {"type": "string"},
            "days": {"type": "integer", "description": f"Look-back days (default {BASELINE_DAYS})."},
        }, "required": ["user_id"]},
    },
    {
        "name": "recall_patient_memory",
        "description": "Recall this user's PRIOR anomaly assessments from the agent's persistent "
                       "memory (past risk levels, actions taken, timestamps). Call this early to "
                       "learn whether this anomaly is recurrent or a first-time event.",
        "parameters": {"type": "object", "properties": {
            "user_id": {"type": "string"}}, "required": ["user_id"]},
    },
    {
        "name": "lookup_vitals_reference",
        "description": "Get clinical normal ranges for a vital sign "
                       "(heart_rate, resp_rate, hrv, spo2). Use to ground your thresholds.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string"}}, "required": ["metric"]},
    },
    {
        "name": "analyze_hr_trend",
        "description": "Compute an EWMA trend of the user's recent resting heart rate and how the "
                       "current reading compares (z-score, statistical-outlier flag). Use this for "
                       "a smarter signal than a single fixed threshold. Pass current_hr = the "
                       "alert's avg_hr to get the z-score.",
        "parameters": {"type": "object", "properties": {
            "user_id": {"type": "string"},
            "days": {"type": "integer", "description": f"Look-back days (default {BASELINE_DAYS})."},
            "current_hr": {"type": "number", "description": "The current window's avg HR to compare."},
        }, "required": ["user_id"]},
    },
    {
        "name": "notify_clinician",
        "description": "Send an alert to the on-call clinician. This is a REAL ACTION — call it "
                       "only for ELEVATED or CRITICAL findings that warrant human attention, "
                       "before submit_report. Provide a concise, specific message.",
        "parameters": {"type": "object", "properties": {
            "user_id": {"type": "string"},
            "risk_level": {"type": "string", "enum": ["ELEVATED", "CRITICAL"]},
            "message": {"type": "string"}}, "required": ["user_id", "risk_level", "message"]},
    },
    {
        "name": "consult_medical_model",
        "description": "Ask a medical specialist model for a focused clinical opinion when you "
                       "are uncertain whether a reading is dangerous. Provide the question and "
                       "the relevant context (reading + baseline).",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"},
            "context": {"type": "string"}}, "required": ["question"]},
    },
    {
        "name": "submit_report",
        "description": "Submit your final Clinician Action Report. Call this exactly once when "
                       "you have reached a verdict; it ends the assessment.",
        "parameters": {"type": "object", "properties": {
            "risk_level": {"type": "string", "enum": ["LOW", "ELEVATED", "CRITICAL"]},
            "summary": {"type": "string", "description": "One-line summary."},
            "reasoning": {"type": "string", "description": "Baseline comparison + clinical reasoning."},
            "recommended_action": {"type": "string",
                                   "description": "e.g. continue monitoring / contact patient / escalate."},
        }, "required": ["risk_level", "summary", "reasoning", "recommended_action"]},
    },
]

TOOL_FUNCS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "fetch_historical_trends": fetch_historical_trends,
    "recall_patient_memory": recall_patient_memory,
    "lookup_vitals_reference": lookup_vitals_reference,
    "analyze_hr_trend": analyze_hr_trend,
    "consult_medical_model": consult_medical_model,
    "notify_clinician": notify_clinician,
    "submit_report": submit_report,
}

SYSTEM = (
    "You are a clinical triage decision-support agent for a remote patient-monitoring platform. "
    "You receive an automated wearable anomaly alert. Assess it by CALLING TOOLS, then finish by "
    "calling submit_report.\n\n"
    "CRITICAL RULES:\n"
    "- Use the real tool-calling mechanism. NEVER write a tool call as text or JSON in your reply.\n"
    "- After every tool result, either call ANOTHER tool or call submit_report. Do not stop with a "
    "plain text message — the assessment is only complete once you call submit_report.\n"
    "- Call submit_report EXACTLY ONCE, as your final action.\n\n"
    "Suggested flow:\n"
    "1. fetch_historical_trends — establish the user's own recent baseline.\n"
    "2. recall_patient_memory — check whether this user has had prior anomalies (recurrent vs new).\n"
    "3. lookup_vitals_reference('heart_rate') — ground your thresholds in clinical norms.\n"
    "4. analyze_hr_trend(current_hr=<alert avg_hr>) — get an EWMA trend + z-score outlier signal.\n"
    "5. (optional) consult_medical_model — if you are still unsure whether the reading is dangerous.\n"
    "6. notify_clinician — if your finding is ELEVATED or CRITICAL, alert the on-call clinician "
    "BEFORE submitting (this is a real action; skip it for LOW).\n"
    "7. submit_report — risk_level (LOW/ELEVATED/CRITICAL), a one-line summary, your reasoning "
    "(compare the reading to the baseline, trend, and any prior history), and a recommended action.\n\n"
    "Judge DANGER relative to the user's baseline, trend, and vital-sign norms, not just the raw "
    "number. A sustained resting heart rate far above the user's baseline is dangerous; a recurrent "
    "critical pattern raises concern further. You are decision support, not a diagnosis; rely on "
    "tool output and default to caution."
)


# --------------------------------------------------------------------------- #
# Backend interface + implementations
# --------------------------------------------------------------------------- #

class AgentRun:
    def __init__(self, trace: List[Dict[str, Any]], report: Optional[Dict[str, Any]],
                 final_text: str, model: str):
        self.trace = trace
        self.report = report          # submit_report args, or None
        self.final_text = final_text
        self.model = model


class LLMBackend(ABC):
    name: str
    model: str

    @abstractmethod
    def run_agent(self, system: str, user_prompt: str,
                  dispatch: Callable[[str, Dict[str, Any]], Dict[str, Any]]) -> AgentRun:
        ...


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        import ollama
        self.model = model
        self.client = ollama.Client(host=host)
        # provider tool format: OpenAI-style function specs
        self.tools = [{"type": "function", "function": s} for s in TOOL_SPECS]

    def run_agent(self, system, user_prompt, dispatch) -> AgentRun:
        messages: List[Any] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        trace: List[Dict[str, Any]] = []
        report = None
        final_text = ""
        for _ in range(MAX_TURNS):
            resp = self.client.chat(model=self.model, messages=messages,
                                    tools=self.tools, options={"temperature": 0})
            msg = resp.message
            messages.append(msg)
            tool_calls = msg.tool_calls or []
            if not tool_calls:
                final_text = (msg.content or "").strip()
                break
            for tc in tool_calls:
                name = tc.function.name
                args = tc.function.arguments   # ollama returns a parsed dict
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                args = dict(args)
                result = dispatch(name, args)
                trace.append({"tool": name, "args": args, "result": result})
                messages.append({"role": "tool", "content": json.dumps(result)})
                if name == "submit_report":
                    report = args
            if report is not None:
                break
        return AgentRun(trace, report, final_text, self.model)


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_MODEL):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic()
        self.tools = [{"name": s["name"], "description": s["description"],
                       "input_schema": s["parameters"]} for s in TOOL_SPECS]

    def run_agent(self, system, user_prompt, dispatch) -> AgentRun:
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        trace: List[Dict[str, Any]] = []
        report = None
        final_text = ""
        for _ in range(MAX_TURNS):
            resp = self.client.messages.create(
                model=self.model, max_tokens=4000, system=system,
                thinking={"type": "adaptive"}, tools=self.tools, messages=messages,
                extra_body={"output_config": {"effort": "medium"}},
            )
            if resp.stop_reason != "tool_use":
                final_text = "".join(b.text for b in resp.content if b.type == "text").strip()
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type != "tool_use":
                    continue
                result = dispatch(b.name, dict(b.input))
                trace.append({"tool": b.name, "args": dict(b.input), "result": result})
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(result)})
                if b.name == "submit_report":
                    report = dict(b.input)
            messages.append({"role": "user", "content": results})
            if report is not None:
                break
        return AgentRun(trace, report, final_text, self.model)


def build_backend(name: str) -> LLMBackend:
    if name == "ollama":
        return OllamaBackend()
    if name == "anthropic":
        return AnthropicBackend()
    raise ValueError(f"unknown backend {name!r}")


# --------------------------------------------------------------------------- #
# Orchestration: run the loop, apply safety floor, build FHIR
# --------------------------------------------------------------------------- #

def _safety_floor(anomaly: Dict[str, Any]) -> str:
    """Deterministic minimum risk level from hard vitals thresholds."""
    hr = anomaly.get("avg_hr") or 0
    resting = "no workout" in (anomaly.get("reason") or "").lower()
    if hr > 180:
        return "CRITICAL"
    if hr > 140 and resting:
        return "ELEVATED"
    return "LOW"


def run_agent(anomaly: Dict[str, Any]) -> Dict[str, Any]:
    """Drive the tool loop over the backend chain, then guarantee correctness."""
    captured: Dict[str, Any] = {"baseline": None, "memory": None, "trend": None}
    success_counts: Dict[str, int] = {}
    REPEAT_LIMIT = 2  # small models loop on read-only tools; steer them to finish

    def _coerce_args(func: Callable[..., Any], name: str, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Make small-model tool calls robust: drop hallucinated kwargs the function does not
        accept, and inject known context (user_id, current_hr) the server already has when the
        model omits it. This turns most 'bad arguments' failures into successful calls."""
        params = set(inspect.signature(func).parameters)
        clean = {k: v for k, v in raw.items() if k in params}
        if "user_id" in params and not clean.get("user_id"):
            clean["user_id"] = anomaly.get("user_id")
        if name == "analyze_hr_trend" and "current_hr" in params and "current_hr" not in clean:
            clean["current_hr"] = anomaly.get("avg_hr")
        return clean

    def dispatch(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        func = TOOL_FUNCS.get(name)
        if func is None:
            return {"error": f"unknown tool {name}"}
        # Loop guard: a confused small model re-calls the same tool forever. Once it has
        # SUCCEEDED enough times, stop recomputing and nudge it to finish. Only *successful*
        # calls count — a malformed call that errored must be retryable, not blocked.
        if name != "submit_report" and success_counts.get(name, 0) >= REPEAT_LIMIT:
            return {"note": f"You have already called {name} successfully and have its result. "
                            "Do NOT call it again. You have enough information: if the finding is "
                            "ELEVATED or CRITICAL call notify_clinician (once), then call "
                            "submit_report NOW with your verdict."}
        try:
            result = func(**_coerce_args(func, name, args))
        except TypeError as exc:
            # Still malformed after coercion (missing a genuine required arg): let it retry.
            return {"error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:
            return {"error": str(exc)}
        success_counts[name] = success_counts.get(name, 0) + 1
        if name == "fetch_historical_trends":
            captured["baseline"] = result
        elif name == "recall_patient_memory":
            captured["memory"] = result
        elif name == "analyze_hr_trend":
            captured["trend"] = result
        return result

    user_prompt = ("A wearable anomaly alert fired. Assess it and submit a Clinician Action "
                   f"Report.\n\nALERT (raw):\n{json.dumps(anomaly, indent=2)}")

    last_error = None
    run: Optional[AgentRun] = None
    used_backend = None
    for name in BACKENDS:
        try:
            backend = build_backend(name)
            run = backend.run_agent(SYSTEM, user_prompt, dispatch)
            used_backend = backend.name
            break
        except Exception as exc:  # backend unreachable / not installed -> try next
            last_error = f"{name}: {exc}"
            continue
    if run is None:
        return {"status": "error", "detail": f"all backends failed ({last_error})"}

    # LLM verdict (structured if submit_report was called, else parse text).
    if run.report:
        llm_level = str(run.report.get("risk_level", "LOW")).upper()
        summary = run.report.get("summary", "")
        reasoning = run.report.get("reasoning", "")
        action = run.report.get("recommended_action", "")
    else:
        llm_level = _parse_risk(run.final_text)
        summary, reasoning, action = run.final_text, "", ""

    # Deterministic safety floor: the model may raise risk, never lower it.
    floor = _safety_floor(anomaly)
    final_level = llm_level if _SEVERITY.get(llm_level, 0) >= _SEVERITY[floor] else floor
    floor_applied = _SEVERITY[floor] > _SEVERITY.get(llm_level, 0)

    # Deterministic FHIR from the reading + fetched baseline.
    baseline = captured["baseline"] or {}
    baseline_hr = baseline.get("resting_baseline_hr") or baseline.get("baseline_avg_hr")
    fhir = convert_to_fhir(anomaly.get("user_id", "unknown"), {
        "heart_rate_bpm": anomaly.get("avg_hr"),
        "max_hr": anomaly.get("max_hr"),
        "baseline_hr": baseline_hr,
        "activity": 0,
        "effective_time": anomaly.get("window_end"),
    })

    report_text = _format_report(final_level, summary, reasoning, action,
                                 floor_applied, floor, run)

    # Persist to cross-session memory + FHIR store (deterministic, code-guaranteed).
    user_id = anomaly.get("user_id", "unknown")
    assessment = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "user_id": user_id,
        "avg_hr": anomaly.get("avg_hr"),
        "max_hr": anomaly.get("max_hr"),
        "risk_level": final_level,
        "summary": summary,
        "recommended_action": action,
        "tools_used": [t["tool"] for t in run.trace],
        "backend": used_backend,
        "model": run.model,
        "safety_floor_applied": floor_applied,
    }
    _append_jsonl(_assessments_path(), assessment)
    fhir_path = _persist_fhir(user_id, fhir)

    return {
        "status": "ok",
        "backend": used_backend,
        "model": run.model,
        "risk_level": final_level,
        "is_dangerous": final_level != "LOW",
        "report": report_text,
        "baseline": captured["baseline"],
        "memory": captured["memory"],
        "trend": captured["trend"],
        "fhir": fhir,
        "tool_trace": run.trace,
        "safety_floor_applied": floor_applied,
        "clinician_notified": any(t["tool"] == "notify_clinician" for t in run.trace),
        "persisted": {"assessment_log": _assessments_path(), "fhir": fhir_path},
    }


def _parse_risk(text: str) -> str:
    up = (text or "").upper()
    for lvl in ("CRITICAL", "ELEVATED", "LOW"):
        if lvl in up:
            return lvl
    return "LOW"


def _format_report(level, summary, reasoning, action, floor_applied, floor, run: AgentRun) -> str:
    lines = [f"RISK LEVEL: {level}"]
    if summary:
        lines.append(summary)
    if reasoning:
        lines.append("")
        lines.append(f"Reasoning: {reasoning}")
    if action:
        lines.append(f"Recommended action: {action}")
    if floor_applied:
        lines.append("")
        lines.append(f"[safety floor raised risk to {floor} regardless of model output]")
    if not run.report and run.final_text:
        lines.append("")
        lines.append(run.final_text)
    tools = " -> ".join(t["tool"] for t in run.trace) or "(none)"
    lines.append("")
    lines.append(f"Tools used: {tools}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

app = FastAPI(title="Wearable Clinical Triage Agent (local-first)")


class AnomalyAlert(BaseModel, extra="allow"):
    user_id: Optional[str] = None
    avg_hr: Optional[float] = None
    max_hr: Optional[float] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    sample_count: Optional[int] = None
    sources: Optional[List[str]] = None
    reason: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "backends": BACKENDS, "ollama_model": OLLAMA_MODEL,
            "medical_model": OLLAMA_MEDICAL_MODEL, "delta_path": DELTA_PATH,
            "memory_dir": MEMORY_DIR, "fhir_store": FHIR_STORE_DIR,
            "tools": [s["name"] for s in TOOL_SPECS]}


@app.get("/memory")
def memory(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Inspect the agent's persistent memory (assessments + clinician notifications)."""
    assessments = _read_jsonl(_assessments_path())
    notifications = _read_jsonl(_notifications_path())
    if user_id:
        assessments = [a for a in assessments if a.get("user_id") == user_id]
        notifications = [n for n in notifications if n.get("user_id") == user_id]
    return {"user_id": user_id, "assessment_count": len(assessments),
            "notification_count": len(notifications),
            "assessments": assessments[-20:], "notifications": notifications[-20:]}


@app.post("/anomaly")
def anomaly(alert: AnomalyAlert) -> Dict[str, Any]:
    payload = alert.model_dump()
    print(f"[agent] anomaly user={payload.get('user_id')} avg_hr={payload.get('avg_hr')} "
          f"backends={BACKENDS}")
    result = run_agent(payload)
    print(f"[agent] -> backend={result.get('backend')} RISK={result.get('risk_level')} "
          f"floor={result.get('safety_floor_applied')}")
    return result


def main() -> None:
    import uvicorn
    print(f"Triage agent on http://localhost:{PORT}  backends={BACKENDS} "
          f"(ollama={OLLAMA_MODEL}, medical={OLLAMA_MEDICAL_MODEL})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
