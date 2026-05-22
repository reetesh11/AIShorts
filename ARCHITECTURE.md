# Agentic Architecture — AI Shorts Agent

## Overview

The system is a **sequential multi-agent pipeline** built on LangGraph. Each agent is a pure function that reads from a shared state dict and returns a partial update. No agent modifies state directly — updates are merged by the graph runtime after each node completes.

A single topic string enters the pipeline. A production-ready `.mp4` exits.

---

## Pipeline

```
┌─────────┐   ┌────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐
│research │──▶│ script │──▶│ quality_gate │──▶│ screenplay │──▶│ image_prompt │
└─────────┘   └────────┘   └──────────────┘   └────────────┘   └──────────────┘
                                                                        │
                                                                        ▼
┌────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌───────────────┐
│ output │◀──│  stitch  │◀──│ metadata │◀──│ voiceover │◀──│   image_gen   │
└────────┘   └──────────┘   └──────────┘   └───────────┘   └───────────────┘
```

> **Why sequential and not parallel?** LangGraph's fan-in mechanism triggers a downstream node once per incoming edge. If parallel branches have unequal depths, the downstream node fires multiple times. Sequential sidesteps this entirely with no performance cost — the real bottleneck is image generation (network I/O), which is parallelised internally within the `image_gen` node using `ThreadPoolExecutor`.

---

## Agent Catalogue

### 1. `research_agent`
**Role:** Grounds the pipeline in facts. Prevents hallucinated scripts.

| | |
|---|---|
| **Model** | `llama-3.3-70b-versatile` (Groq) |
| **Inputs** | `topic`, `max_words` (from config) |
| **Outputs** | `research_summary`, `key_facts[]`, `target_audience`, `unique_angle` |
| **Technique** | Structured output via Pydantic (`ResearchResponse`) |

The agent is explicitly prompted to surface *surprising* facts and flag common misconceptions — both are strong predictors of retention in short-form video.

---

### 2. `script_agent`
**Role:** Writes the spoken script from research material.

| | |
|---|---|
| **Model** | `llama-3.3-70b-versatile` (Groq) |
| **Inputs** | `topic`, `target_audience`, `unique_angle`, `key_facts`, `research_summary` |
| **Outputs** | `hook`, `full_script`, `cta` |
| **Technique** | Structured output (`ScriptResponse`) |

Target is 120–150 words (45–60 seconds spoken). The hook, body, and CTA are returned as separate fields so downstream agents can reference them independently.

---

### 3. `quality_gate_agent`
**Role:** Scores the script before expensive downstream work begins.

| | |
|---|---|
| **Model** | `llama-3.1-8b-instant` (Groq) — fast scorer, no creativity needed |
| **Inputs** | `topic`, `hook`, `full_script`, `cta` |
| **Outputs** | `quality_approved`, `quality_notes`, (optionally) patched `hook` + `full_script` |
| **Technique** | Structured output with numeric scores (hook/pacing/CTA each 1–10) |

Approves if all scores ≥ 6 and word count is 100–160. If `hook_score < 7`, the agent rewrites the hook and patches the first sentence of `full_script` automatically — no retry loop needed.

This is the only agent that **mutates** upstream state (`hook`, `full_script`).

---

### 4. `screenplay_agent`
**Role:** Translates the script into a cinematic scene-by-scene plan.

| | |
|---|---|
| **Model** | `llama-3.3-70b-versatile` (Groq) |
| **Inputs** | `research_summary`, `full_script` |
| **Outputs** | `art_style`, `scene_plans[]` |
| **Technique** | Structured output with nested Pydantic models (`ScreenplayResponse → ScenePlanModel → VoiceoverLineModel`) |

Produces 5–6 scenes. Each scene has:
- `image_description` — visual brief for the image generator
- `voiceover_lines[]` — text + emotional tone per line
- `transition_in` — dissolve / cut / fade / zoom_in / zoom_out
- `duration_hint` — target seconds

The `art_style` field (e.g. *"vibrant 3D cartoon"*) is locked here and enforced across all subsequent image prompts for visual consistency.

---

### 5. `image_prompt_agent`
**Role:** Converts scene descriptions into detailed AI image generation prompts.

| | |
|---|---|
| **Model** | `llama-3.3-70b-versatile` (Groq) |
| **Inputs** | `art_style`, `scene_plans[]` |
| **Outputs** | `image_prompts[]` — each with `scene_id`, `prompt`, `style_type`, `width`, `height` |
| **Technique** | Structured output (`ImagePromptsResponse`) |

Exists as a separate agent from `screenplay_agent` intentionally: crafting FLUX/Imagen prompts is a different cognitive task from scene planning. The model is instructed to start every prompt with the exact `art_style` string to enforce visual consistency.

---

### 6. `image_gen_agent`
**Role:** Generates one portrait image per scene.

| | |
|---|---|
| **Provider** | Pollinations.ai (FLUX, free, no key) — or Google Imagen 3 |
| **Inputs** | `image_prompts[]`, video `width`/`height` from config |
| **Outputs** | `generated_images[]` — ordered list of absolute file paths |
| **Concurrency** | `ThreadPoolExecutor(max_workers=3)` with 2 s stagger between starts |

**Pollinations URL flags used:**
- `enhance=true` — auto-improves prompts via an LLM before generation
- `negative=...` — excludes text, watermarks, distortion, blurry content
- `seed=hash(prompt)` — deterministic output for the same prompt

Retry logic: exponential backoff on HTTP 429, up to 5 attempts.

---

### 7. `voiceover_agent`
**Role:** Synthesises per-scene audio with emotional tone modulation.

| | |
|---|---|
| **Provider** | Microsoft edge-tts (Azure Neural voices, free, no key) |
| **Inputs** | `scene_plans[]` (voiceover lines + tones) |
| **Outputs** | `scene_audios[]` — each with `audio_path`, `word_timings[]`, `duration` |
| **Concurrency** | `asyncio.gather` — all scenes synthesised in parallel |

Each voiceover line has an assigned tone (`excited`, `mysterious`, `dramatic`, `energetic`, `calm`, `warm`, `inspiring`). Tones map to TTS `rate` and `pitch` adjustments via `TONE_PROSODY`:

```python
TONE_PROSODY = {
    "excited":    ("+28%", "+10Hz"),
    "mysterious": ("-22%", "-5Hz"),
    "dramatic":   ("-12%", "-3Hz"),
    ...
}
```

Word timings are derived from `SentenceBoundary` events (Azure Neural voices do not emit `WordBoundary`). Words within a sentence are time-distributed proportionally by character count + punctuation weight.

---

### 8. `metadata_agent`
**Role:** Generates YouTube upload metadata.

| | |
|---|---|
| **Model** | `llama-3.1-8b-instant` (Groq) — fast, structured task |
| **Inputs** | `topic`, `target_audience`, `key_facts`, `hook`, `unique_angle` |
| **Outputs** | `metadata` dict: `title`, `title_options[3]`, `description`, `hashtags[]`, `thumbnail_prompt` |
| **Technique** | Structured output (`MetadataResponse`) |

---

### 9. `stitch_agent`
**Role:** Assembles the final MP4 from images, audio, and word timings.

| | |
|---|---|
| **Library** | moviepy 1.x + Pillow + NumPy |
| **Inputs** | `generated_images[]`, `scene_audios[]`, `scene_plans[]` |
| **Outputs** | `video_path` |

Key behaviours:
- Images are resized/cropped to exact portrait dimensions (1080×1920 default) preserving aspect ratio with a centre crop
- Per-scene transitions applied: `dissolve`/`fade` use `crossfadein(0.4s)`; `cut` is an instant splice
- **Karaoke captions**: 4-word chunks rendered with Pillow at 76% height; the currently-spoken word is highlighted yellow, others white, all with black stroke for legibility
- Audio per scene is synced via `AudioFileClip` set on the matching `ImageClip`

---

### 10. `output_agent`
**Role:** Persists human-readable artefacts alongside the video.

| | |
|---|---|
| **Inputs** | Full pipeline state |
| **Outputs** | `output_dir`, writes `description.md` and `meta.json` to disk |

`description.md` includes the full script, screenplay breakdown, image prompts used, research summary, and YouTube copy — everything needed to post the video without re-running the pipeline.

---

## Shared State

All agents communicate through a single `ShortsState` TypedDict (defined in `graph/state.py`). LangGraph merges each agent's return dict into this shared state before the next node runs.

```
ShortsState
├── topic, cfg                          ← immutable input
├── research_summary, key_facts,        ← research_agent
│   target_audience, unique_angle
├── hook, full_script, cta              ← script_agent (may be patched by quality_gate)
├── quality_approved, quality_notes     ← quality_gate_agent
├── art_style, scene_plans[]            ← screenplay_agent
├── image_prompts[]                     ← image_prompt_agent
├── generated_images[]                  ← image_gen_agent
├── scene_audios[]                      ← voiceover_agent
├── metadata{}                          ← metadata_agent
├── video_path                          ← stitch_agent
├── output_dir                          ← output_agent
└── errors[]                            ← accumulated across all agents (operator.add)
```

The `errors` field uses `Annotated[list[str], operator.add]` so any agent can append errors without overwriting another agent's errors.

---

## LLM Layer

All LLM calls go through `providers/llm_factory.py` which:
1. Resolves the provider (`groq` / `anthropic` / `google`) from the agent's config entry
2. Attaches a module-level `_UsageTracker` callback that accumulates `prompt_tokens` + `completion_tokens` per call
3. Returns a LangChain `BaseChatModel` — all agents use `.with_structured_output(PydanticModel)` so responses are always type-safe

Token usage is printed at the end of every run:

```
── Token Usage ──────────────────────────────────────
  research         in:  380  out: 240  total:  620
  script           in:  540  out: 195  total:  735
  quality_gate     in:  640  out: 110  total:  750
  screenplay       in:  730  out: 680  total: 1410
  image_prompt     in:  620  out: 590  total: 1210
  metadata         in:  480  out: 390  total:  870
  TOTAL            in: 3390  out:2205  total: 5595
─────────────────────────────────────────────────────
```

---

## Performance Profile

| Stage | Time (before) | Time (after optimisation) | Notes |
|---|---|---|---|
| LLM calls (×6) | ~6s | ~6s | Not a bottleneck at Groq speeds |
| Image gen (5 imgs) | ~115s | ~40s | ThreadPoolExecutor, 3 workers |
| Voiceover (6 scenes) | ~18s | ~3s | asyncio.gather, all in parallel |
| Video stitch | ~45s | ~45s | CPU-bound moviepy, hard to parallelize |
| **Total** | **~3 min** | **~1.5 min** | |

---

## Tech Stack

| Concern | Library / Service |
|---|---|
| Agent orchestration | LangGraph 0.2+ |
| LLM abstraction | LangChain Core + provider SDKs |
| LLM provider (default) | Groq (llama-3.3-70b-versatile, llama-3.1-8b-instant) |
| Image generation | Pollinations.ai — FLUX (free) or Google Imagen 3 |
| Text-to-speech | edge-tts (Microsoft Azure Neural, free) |
| Video assembly | moviepy 1.x |
| Image processing | Pillow, NumPy |
| FFmpeg | imageio-ffmpeg (bundled, no system install needed) |
| Config | PyYAML + python-dotenv |
| Type safety | Pydantic v2 (structured LLM outputs + state types) |
