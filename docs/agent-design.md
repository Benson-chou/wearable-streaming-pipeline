# AI Agent Layer — Design

Design for the agentic AI layer that consumes wearable anomaly alerts, reasons about
them with tool use, and emits a FHIR `Observation` + Clinician Action Report.

**Status:** Phase 1 (local Ollama tool-loop) **done**; Phase 4 (cross-session memory + richer
tools) **done locally**. Phases 2–3 (the Claude backend + compare/fallback) are intentionally
**deferred** — the local agentic surface is being built out first. This design supersedes the
single-backend `agent_server.py` (Anthropic-only) as the target architecture.

## Goals

- **Learn agentic tool-use** — a real tool-calling loop where the model decides which
  tools to call and when, not a hardcoded pipeline.
- **Local-first, scale later** — [Ollama](https://ollama.com) is the free local placeholder
  for now; switching to Claude at scale is a one-env-var change.
- **Fallback / comparison** — the backend interface supports an `ollama → claude` fallback
  chain and a side-by-side compare mode on the same anomaly.
- **Medical-tuned model in a sensible role** — medical fine-tunes are weak at tool-calling,
  so the general model orchestrates and consults the medical model *as a tool*.

## Guiding principles

1. **Provider-portable.** One `LLMBackend` interface; `OllamaBackend` and `AnthropicBackend`
   are interchangeable. The tool definitions and their Python implementations are identical
   across providers — only the LLM transport swaps. That portability *is* the lesson.
2. **The agent reasons; the code guarantees.** The LLM makes clinical *judgments*; deterministic
   code guarantees *correctness* (always-valid FHIR, a safety floor the model can't undershoot).
3. **Observable.** Every run returns a `tool_trace` so the agent's decisions are visible.

## Architecture

```
                          ┌────────────────────────────────────────────┐
 Spark ──POST /anomaly──▶ │  FastAPI shell (unchanged contract)         │
                          │                                             │
                          │   run_agent(anomaly)                        │
                          │     │                                       │
                          │     ▼   ┌──────────── LLMBackend ─────────┐ │
                          │  tool-  │ OllamaBackend  |  AnthropicBack. │ │
                          │  loop ◀─┤ (default)      |  (scale-up)     │ │
                          │     │   └─────────────────────────────────┘ │
                          │     ▼                                       │
                          │  tools:  fetch_historical_trends            │
                          │          lookup_vitals_reference            │
                          │          consult_medical_model  ──▶ Ollama med model
                          │          submit_report (structured finish)  │
                          │     │                                       │
                          │     ▼  deterministic safety floor           │
                          │     ▼  convert_to_fhir (validated builder)  │
                          │  return {backend, model, report, risk_level,│
                          │          is_dangerous, baseline, fhir,      │
                          │          tool_trace}                        │
                          └────────────────────────────────────────────┘
```

## The tool-calling loop (centerpiece)

The loop is provider-agnostic; only the `backend.chat(...)` line is provider-specific:

```
messages = [ user: "assess this anomaly …" ]
while turns < MAX:
    reply = backend.chat(system, messages, tools)     # ← only provider-specific line
    if reply.tool_calls:
        append(assistant, reply)
        for call in reply.tool_calls:
            result = DISPATCH[call.name](**call.args)   # your Python runs
            append(tool_result, call.id, result)
        continue
    else:
        break   # model answered without a tool → done
```

**Universal vs provider-specific** — the contrast to internalize:

| | Ollama (`/api/chat`) | Anthropic (Messages) |
|---|---|---|
| Send tools | `tools=[{type:function, function:{name,description,parameters}}]` | `tools=[{name,description,input_schema}]` |
| Model asks for a tool | `message.tool_calls[].function.{name,arguments}` | `content` block `{type:"tool_use", name, input, id}` |
| You return the result | message `{role:"tool", content:"…"}` | `{type:"tool_result", tool_use_id, content}` |
| "Keep going" signal | presence of `tool_calls` | `stop_reason == "tool_use"` |

The tool definitions and dispatch functions are byte-identical across both. The adapter
(schema translation + tool-call parsing + result formatting) is ~30 lines per provider and
is the whole portability story.

## Tools

Designed so the agent must actually *choose* — which to call, in what order, whether to
consult the specialist:

| Tool | Kind | What it does | Why it teaches agency |
|---|---|---|---|
| `fetch_historical_trends(user_id, days)` | data (Delta) | 7-day baseline (overall + resting) | Learn *this* user's normal before judging |
| `recall_patient_memory(user_id)` | memory (read) | Prior assessments for this user (risk levels, actions, timestamps) | Is this recurrent or first-time? State across sessions |
| `lookup_vitals_reference(metric)` | retrieval (static) | Clinical normal ranges (deterministic dict) | Ground thresholds in facts, not hallucination |
| `analyze_hr_trend(user_id, current_hr)` | data (Delta) | EWMA of recent resting HR + z-score / outlier flag for the current reading | A smarter-than-fixed-threshold signal the agent opts into |
| `consult_medical_model(question, context)` | sub-model | Ask the **medical-tuned** Ollama model for a specialist opinion | Agent decides *when* it needs an expert (router→specialist) |
| `notify_clinician(user_id, risk_level, message)` | action (write) | Log an outbound clinician alert (real side effect) | The agent *acts*, not just reports — only for ELEVATED/CRITICAL |
| `submit_report(risk_level, summary, reasoning, recommended_action)` | structured finish | Final action; ends the loop with structured output | "The last tool call is your typed output" — reliable on any model |

`convert_to_fhir` stays **out** of the agent's hands — it runs deterministically after
`submit_report`, so the FHIR resource is always valid. (Optional: let the agent supply the
`note` narrative text.)

**Persistence (deterministic, after the loop):** every assessment is appended to
`./agent_memory/assessments.jsonl` (which `recall_patient_memory` reads back — that's the
cross-session memory), and each FHIR `Observation` is written to `./fhir_store/{user_id}/`.
Both live **outside** `./lakehouse` so tearing the pipeline down does not erase learned history.

## Medical model as a specialist tool

- **Orchestrator model:** general, strong tool-caller — `llama3.1:8b` or `qwen2.5:7b`.
- **Specialist model:** medical fine-tune — candidates `meditron:7b` or `med42` (confirm what's
  pullable at build time). Never does tool-calling; answers a focused clinical question when consulted.
- Honest about local limits *and* a real production pattern (cheap router delegating to a
  specialized model).

## Deterministic safety floor

After `submit_report`, code applies hard rules the LLM can raise but never lower:

```
avg_hr > 180              → CRITICAL
avg_hr > 140 and resting  → at least ELEVATED
```

Guarantees the system never *under*-alerts on a clearly dangerous vital, regardless of what a
7B model says. `tool_trace` records both the LLM's level and the enforced floor.

## Config & selection (scale-up path)

```
AGENT_BACKEND=ollama                 # default; free, local
AGENT_BACKEND=anthropic              # scale up — same tools, same loop
AGENT_BACKEND=ollama,anthropic       # fallback chain: try local, fall back to Claude on failure
POST /anomaly?compare=1              # run BOTH, return both results + tool_traces side by side
OLLAMA_MODEL=llama3.1:8b   OLLAMA_MEDICAL_MODEL=meditron   OLLAMA_HOST=localhost:11434
```

Switching to Claude is one env var because nothing above the backend adapter knows which
provider ran.

## Response contract (adds observability)

Same request as `agent_server.py` today; response gains `backend`, `model`, `tool_trace`:

```json
{ "status":"ok", "backend":"ollama", "model":"llama3.1:8b",
  "risk_level":"CRITICAL", "is_dangerous":true,
  "report":"…", "baseline":{…}, "memory":{…}, "trend":{…}, "fhir":{…},
  "tool_trace":[
     {"tool":"fetch_historical_trends","args":{…},"result":{…}},
     {"tool":"recall_patient_memory","args":{…},"result":{…}},
     {"tool":"analyze_hr_trend","args":{…},"result":{…}},
     {"tool":"notify_clinician","args":{…},"result":{…}},
     {"tool":"submit_report","args":{…}} ],
  "safety_floor_applied": true, "clinician_notified": true,
  "persisted": {"assessment_log":"./agent_memory/assessments.jsonl", "fhir":"./fhir_store/…"} }
```

Inspect accumulated state at any time: `GET /memory?user_id=…` returns the stored assessments
and clinician notifications.

## Failure handling (also a lesson)

- **Tool errors** → returned to the model as error results so it can recover mid-loop.
- **Malformed tool args** from a small model → the dispatcher **coerces** before failing: it
  drops hallucinated kwargs the function doesn't accept (e.g. `ewma_weight`, `z_score_threshold`)
  and injects known context (`user_id`, `current_hr`) from the anomaly the server already holds.
  This alone turned a 14×-repeated failing `analyze_hr_trend` loop into a single clean call.
  Genuinely-missing required args still return an error result so the model can retry.
- **Repeat-call loop guard** → only *successful* calls count toward a per-tool repeat limit, so
  once the agent has an answer it's steered to `notify_clinician`/`submit_report` — but a call it
  fumbled and then self-corrects is never wrongly blocked.
- **Ollama unreachable** → clear error, or next backend in the chain — never a silent jump to
  the paid API.
- **Max-turns guard** so a confused local model can't loop forever.

## Roadmap

- **Phase 1 (DONE):** `OllamaBackend` + 4 tools + safety floor + `tool_trace`, same `/anomaly`.
  Fully local, $0. Verified end-to-end with `llama3.1:8b` (orchestrator) + `meditron` (specialist).
  Findings from the first runs:
  - The full designed flow works: `fetch_historical_trends → lookup_vitals_reference →
    consult_medical_model → submit_report`, producing a structured CRITICAL verdict with sound
    reasoning ("171.5 bpm ≈ 3× the resting baseline of 61").
  - `llama3.1:8b` frequently guesses **wrong tool-argument names** (`vital_name` vs `metric`) and
    **self-corrects from the error result** — so it needs headroom: `AGENT_MAX_TURNS` default
    raised **8 → 12** so it can converge to `submit_report` after fumbling a couple of turns.
  - When it *doesn't* converge, the **safety floor is the backstop** — it reliably raises an
    unresolved verdict to the rule-based minimum (a dangerous reading never slips through as LOW).
  - `meditron:latest` is a weak base model (echoes prompts); fine as a demo "specialist consult,"
    but a better medical fine-tune (or the Claude path) improves quality.
  - Latency ~2 min/anomaly warm on an M4 Pro (8B multi-turn + a second model load for the consult).
    Motivates the Phase-2 Claude backend for cleaner, faster completion.
- **Phase 4 (DONE, local):** richer tools + cross-session memory, built out before the Claude
  backend (Phases 2–3 deferred by choice). Added `recall_patient_memory` (memory read),
  `analyze_hr_trend` (EWMA + z-score), and `notify_clinician` (action). Each assessment persists
  to `./agent_memory` and each Observation to `./fhir_store`, both outside `./lakehouse`.
  Verified end-to-end with `llama3.1:8b`:
  - Two anomalies for the same user, run in sequence — the **second recalled the first** from
    memory (`prior_count: 1, ['CRITICAL']`) and cited it in the report ("prior history of
    critical readings, which raises further concern").
  - With 7 tools the model fumbles more (hallucinated arg names, omitted `user_id`, looped on one
    tool). The **arg-coercion + success-based loop guard** (see Failure handling) were needed to
    get clean convergence — runs dropped from ~290s (14 failed loops) to ~120–150s reaching
    `submit_report`. You can still see the fumble-then-retry in the `tool_trace`; it's now bounded.
  - Still-open: `notify_clinician` is a local log, not a real channel; the medical consult and a
    `write-to-FHIR-store` *as a tool* (vs. deterministic) remain future work.
- **Phase 2 (deferred):** `AnthropicBackend` behind the same interface + `?compare=1`.
- **Phase 3 (deferred):** fallback chain; prompt caching + structured outputs on the Claude path.
- **Phase 5:** the same tool defs lift into Anthropic **Managed Agents** (hosted loop) with
  near-zero rewrite — the portable-tool design is what makes that cheap.

## Caveats

- 7B tool-calling is **flaky** — expect occasional skipped/malformed calls. Seeing where local
  models struggle vs Claude is part of the learning; the guards contain it.
- Medical fine-tunes are **not clinically validated** — this is a demo of the *pattern*, labeled
  decision-support, never diagnosis.

## Relationship to existing code

- Reuses `convert_to_fhir` and `fetch_historical_trends` from `agent_server.py` (already built +
  tested) as two of the tools.
- The existing `agent_server.py` (Anthropic-only, hardcoded loop) becomes the `AnthropicBackend`
  case under this design — or is refactored into it.
- Keeps the `POST /anomaly` contract so `spark_processor.py` and the rest of the pipeline are
  untouched.
