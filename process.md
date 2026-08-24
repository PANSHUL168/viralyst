# Viralyst Implementation Process

Last updated: 24 August 2026

## 1. Project Goal

Viralyst is a local-first synthetic-audience simulator for short-form video.
It extracts information from a video, creates varied artificial viewers,
connects them in a social network, and asks a local language model how each
viewer would react. Later phases will propagate those reactions through the
network, calculate engagement metrics, and recommend video improvements.

The output must be described as a **directional simulation**, not as a proven
prediction of whether a real video will go viral. `VIRALYST.md` remains the
authoritative specification for scope, architecture, and formulas.

## 2. Repository and Environment Setup

The project was initialized as a Python 3.11/3.12 repository with a local
virtual environment in `.venv/`. Dependencies are pinned in
`requirements.txt`, while `requirements-dev.txt` adds `pytest` for testing.
Major libraries include OpenCV, Faster Whisper, Ultralytics YOLO, Pydantic,
NumPy, pandas, NetworkX, Ollama, LangGraph, Streamlit, and Plotly.

The repository is divided by responsibility:

- `video/` contains media extraction and perception code.
- `personas/` generates the synthetic population.
- `simulation/` contains network and, eventually, propagation logic.
- `agents/` contains local-LLM integration and audience behavior.
- `workflow/` is reserved for LangGraph orchestration.
- `scripts/` contains runnable diagnostics and export utilities.
- `tests/` mirrors the implemented modules.
- `cache/` and `tmp/` hold generated data and are excluded from Git.

`config.py` centralizes every tunable value, including model names, sampling
limits, graph settings, propagation probabilities, score weights, timeouts,
and worker counts. `.env.example` documents the environment overrides.
Videos, model weights, `.env`, caches, temporary files, and generated example
outputs are excluded through `.gitignore`.

`AGENTS.md` was added as the contributor guide. It documents structure,
commands, style rules, testing expectations, commit conventions, reliability
requirements, and the rule that presentation code must not contain simulation
logic.

## 3. Shared Contracts and Utilities

`schemas.py` defines Pydantic models at the boundaries between modules:

- `ObjectStats` stores aggregated detections.
- `TranscriptSegment` and `TranscriptResult` store cleaned speech output.
- `VideoFeatures` is the complete multimodal description of a video.
- `Persona` stores demographics, interests, personality, and behavior traits.
- `Reaction` stores one persona's structured engagement response.
- `Wave`, `SegmentMetrics`, `EngagementMetrics`, and `Recommendations` define
  contracts required by the remaining phases.

Validation prevents bad data from moving through the pipeline. For example,
a skipped reaction is limited to 30% watch time and cannot also like, comment,
share, or follow. A follow action is repaired to include a like when necessary.

`utils.py` provides shared infrastructure:

- Idempotent logging configuration.
- Streaming SHA-256 hashing for cache keys and reproducibility.
- A timing context manager for measuring pipeline stages.
- Defensive JSON coercion that accepts a dictionary, bytes, plain JSON,
  Markdown-fenced JSON, or a JSON object embedded in surrounding text.
- A dedicated `JSONCoercionError` for controlled parsing failures.

These utilities remove repeated error-prone code and give later modules one
consistent way to log, time, cache, and parse results.

## 4. Phase 1: Video Extraction

`video/extractor.py` implements the OpenCV and FFmpeg foundation:

1. `probe()` validates a video and reads duration, FPS, frame count, aspect
   ratio, and orientation.
2. `sample_frames()` selects frames across the complete video, respects the
   configured minimum and maximum sample counts, and resizes them while
   preserving aspect ratio.
3. `temporal_stats()` calculates average brightness, brightness variance,
   motion, and scene changes.
4. `extract_audio()` invokes FFmpeg to produce mono 16 kHz WAV audio. A video
   without an audio stream returns `None` instead of crashing.

Custom `VideoReadError` and `AudioExtractionError` exceptions turn unreadable
videos, invalid metadata, missing FFmpeg, and failed conversions into useful
messages. The extractor also has a command-line entry point for direct checks.

## 5. Phase 2: Multimodal Information Extraction

### Object detection

`video/detector.py` loads one cached YOLOv8 nano model and runs it over the
sampled frames. It aggregates class counts, unique classes, average objects per
frame, and the dominant object. Empty frames and videos with no detections
produce valid empty statistics. Ties are resolved deterministically.

### Speech transcription

`video/transcriber.py` loads one cached Faster Whisper `base` model using
`int8` computation. It transcribes the extracted WAV, normalizes text, records
language and timestamps, removes high no-speech-probability segments, measures
speech duration and word count, and extracts speech overlapping the first
three seconds as the hook. Silent videos receive a canonical empty transcript.

### Feature fusion and caching

`video/features.py` combines metadata, temporal measurements, object
detections, and transcription into one validated `VideoFeatures` object. It
derives speech rate, silence ratio, hook text, pacing (`slow`, `moderate`, or
`fast`), and visual density (`sparse`, `balanced`, or `busy`).

Features are cached in `cache/<video-hash>.json`. A cache hit avoids OpenCV,
YOLO, FFmpeg, and Whisper work; corrupt or incompatible cache data is ignored
and regenerated. Temporary audio is removed in a `finally` block.

### Inspectable stage exporter

`scripts/export_video_analysis.py` runs every perception stage once and saves
inspectable artifacts. This was used on `example/index.mp4`, producing:

- `00_source.json`: input path, size, and full SHA-256.
- `01_metadata.json`: video metadata.
- `02_sampled_frames/` and its manifest: 20 JPEG samples.
- `03_temporal_stats.json`: brightness, motion, and cuts.
- `04_object_detections.json`: aggregated YOLO output.
- `05_audio.wav`: extracted audio.
- `06_transcript.json`: Whisper text and timestamped segments.
- `07_video_features.json`: fused contract used by later phases.
- `08_run_manifest.json`: artifact map and stage timings.

The real sample is 47.5 seconds long, vertical, 30 FPS, and 1,425 frames.
Twenty frames were analyzed. YOLO found `person`, `cell phone`, and `kite`,
with `person` dominant. Whisper produced 164 English words, detected speech in
46.16 seconds, and the fused pipeline labelled the video as fast-paced with
balanced visual density. The recorded cold run completed in 15.22 seconds.

## 6. Phase 3: Synthetic Population and Social Graph

### Persona generation

`personas/generator.py` creates reproducible synthetic viewers from one seeded
NumPy random generator. Seven weighted archetypes are represented:
`tech_enthusiast`, `student`, `developer`, `marketer`, `creator`,
`professional`, and `casual_viewer`.

Each persona has a stable ID and name, age, profession, interests, Big Five
personality traits, attention span, skepticism, share and comment propensity,
and daily scrolling time. Archetype-specific ranges and trait skews create
meaningful behavioral differences without using an LLM. `population_summary()`
returns a consistent pandas table of counts, percentages, and mean traits.

### Homophily graph

`simulation/social_graph.py` represents personas as NetworkX nodes. Every
unique pair is scored once during construction using:

- 50% shared-interest similarity.
- 20% age similarity.
- 15% matching archetype.
- 15% Big Five personality similarity.

Pairs above `HOMOPHILY_THRESHOLD` are added from most to least similar while
respecting `MAX_DEGREE = 12`. Seeded random weak ties preferentially connect
different archetypes, allowing information to escape a single social cluster.
A connectivity-repair step joins any remaining components without exceeding
the degree ceiling. The graph is frozen after construction so simulation code
cannot accidentally alter the audience network.

Seed users can be selected using `random`, `degree`, or `mixed` strategy. The
default mixed strategy chooses two highly connected users and fills the rest
with seeded random users.

The latest 100-person checkpoint produced 539 edges, average degree 10.78,
maximum degree 12, clustering coefficient 0.641, and exactly one connected
component containing all 100 personas. The deterministic seed users were
`[0, 1, 66, 10, 77]`.

## 7. Phase 4: Audience Reaction Agents

### Shared Ollama boundary

`agents/llm.py` is the only module that talks directly to Ollama. It provides:

- A thread-safe shared client with configurable host and request timeout.
- `health_check()` to distinguish an unreachable service from a missing model.
- `chat_json()` using Ollama JSON mode, fixed context/output limits, and
  configurable temperature.
- JSON coercion followed by one explicit repair prompt when output is invalid.
- Clear error types for unavailable service, missing model, request failure,
  and permanently malformed output.
- `run_batch()` using `ThreadPoolExecutor`, stable result ordering, bounded
  concurrency, and per-completion progress callbacks.

Audience temperature is 0.7 to permit behavioral variation. The code never
silently treats malformed LLM output as trustworthy structured data.

### Audience prompt and reaction logic

`agents/audience.py` builds a prompt containing video duration, orientation,
pacing, speech rate, scene changes, detected objects, brightness, motion, a
separate opening hook, a bounded transcript, and all relevant numeric persona
traits. The system prompt tells the model to act as that viewer, be decisive,
respect attention and skepticism, make sharing rare, and return JSON only.

`react()` validates the model response as `Reaction`, overrides any model-made
persona ID or wave number with application-owned values, records latency, and
applies the schema consistency rules. If inference, parsing, or validation
fails, it produces a visible deterministic fallback instead of terminating the
run.

`heuristic_reaction()` uses a SHA-256-derived seed plus interest overlap,
attention span, skepticism, video duration, hook presence, pacing, and motion.
It calculates watch percentage and probabilistic-but-reproducible skip, like,
comment, share, follow, and purchase actions. Fallback reactions are marked
with `fallback=True`, allowing the UI and metrics to disclose degraded runs.

`scripts/dry_run.py` exercises Phase 4 from the command line. It loads cached
video features, generates personas, runs reactions concurrently, prints a
reaction table, summarizes watch/skip/share/fallback rates, compares
archetypes, and optionally exports CSV. `--heuristic-only` tests the complete
path without Ollama.

The heuristic checkpoint ran on the real sample with 10 personas. It produced
32.0% average watch, 30.0% skips, and 0% shares. Its fallback rate was 100%
because heuristic-only mode was requested, not because parsing failed. The
rows are saved in `example/output/phase4_heuristic_reactions.csv`.

Ollama's Python integration is implemented, but the Windows Ollama application
and the configured 4.9 GB `llama3.1:8b-instruct-q4_K_M` model are not currently
installed. Therefore the real-model reaction and 20-person performance
checkpoints remain pending and are deliberately not marked complete.

## 8. Testing and Verification

The suite currently contains 77 passing tests and makes no real LLM calls or
model downloads. External inference is mocked so tests remain deterministic.
Coverage includes:

- File hashing, JSON recovery, logging, and timing.
- Reaction consistency repair and schema boundaries.
- Video metadata, sampling limits, resizing, temporal statistics, FFmpeg
  commands, silent video behavior, and error messages.
- YOLO aggregation, empty output, deterministic ties, failure wrapping, and
  singleton model loading.
- Whisper normalization, VAD filtering, hook selection, empty speech, and
  singleton model loading.
- Feature derivation, cache hits, corrupt cache recovery, and temporary cleanup.
- Persona reproducibility, valid ranges, expected archetype distribution,
  behavioral differences, and summaries.
- Graph determinism, connectivity, degree limits, clustering, edge metadata,
  connectivity repair, seed strategies, and CLI export.
- Ollama health checks, embedded JSON, repair retries, permanent failures,
  batch ordering, and progress reporting.
- Audience prompt content, application-owned identifiers, schema repair,
  deterministic fallback, wave handling, and concurrent order preservation.

Primary verification commands are:

```powershell
pytest -q
python -m simulation.social_graph --personas 100 --seed 42
python -m scripts.dry_run --video example\index.mp4 --n 10 --seed 42 --heuristic-only
```

## 9. Reliability Decisions

The implementation follows several rules intended to keep demos and future UI
runs recoverable:

- Missing audio, empty detections, malformed JSON, and failed LLM calls produce
  valid degraded results instead of stack traces where possible.
- Expensive models use cached singleton loaders.
- Video features are keyed by content hash rather than filename.
- Seeds make personas, graph structure, seed users, and fallback reactions
  reproducible.
- Structured Pydantic objects cross module boundaries instead of unchecked
  dictionaries.
- Generated outputs expose whether a fallback was used.
- Configuration values live in one module instead of being scattered through
  algorithms.

## 10. Current Status and Remaining Work

Completed implementation includes repository setup, contracts and utilities,
the complete multimodal video-information pipeline, deterministic persona
generation, connected homophily graph construction, the Ollama client layer,
audience reaction prompting, deterministic fallback behavior, batch execution,
diagnostic CLIs, real-video artifacts, and automated tests.

The next four major phases are:

1. **Propagation and metrics:** implement `simulation/propagation.py` and
   `simulation/metrics.py`, including waves, stopping rules, segment metrics,
   and the 0–100 simulated virality score.
2. **Workflow orchestration:** implement `workflow/state.py` and
   `workflow/graph.py` to connect analysis, population, graph, reactions,
   propagation, and finalization through LangGraph.
3. **Recommendations and UI:** implement `agents/recommender.py` and replace
   the placeholder `app.py` with the complete Streamlit upload and results UI.
4. **Hardening and release:** add visualizations, precomputed demo mode,
   user-facing errors, full README documentation, cold-start performance tests,
   and the final multi-video test matrix.

Before Phase 4 can be considered operationally complete, install Ollama, pull
the configured model, make `health_check()` return green, and run the real
10- and 20-person dry-run checkpoints. The code path itself is ready for those
checks.
