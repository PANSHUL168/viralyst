# Viralyst Implementation Process

Last updated: 28 August 2026

## 1. Project Goal

Viralyst is a local-first synthetic-audience simulator for short-form video.
It extracts information from a video, creates varied artificial viewers,
connects them in a social network, and asks a local language model how each
viewer would react. It propagates those reactions through the network,
calculates engagement metrics, and recommends concrete video improvements.

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

Ollama 0.32.15 and the configured 4.9 GB
`llama3.1:8b-instruct-q4_K_M` model are installed. Viralyst's health check is
green. A warm one-person run produced a valid non-fallback reaction in 15.44
seconds end to end, with 11.05 seconds spent on inference.

The live 20-person run also passed correctness and prompt-quality checks. All
20 responses validated without fallback, all reasons were distinct, nine
action profiles were produced, and watch percentage had a 14.05-point standard
deviation. Archetype averages ranged from 16.0% to 32.5%, demonstrating that
the prompt used persona differences. The batch took 188.2 seconds, however,
so it did not meet the aspirational under-40-second Checkpoint E. On this RTX
3050 Laptop GPU, Ollama loaded one inference slot and split the 8B model across
58% CPU and 42% GPU. The live rows are stored in the gitignored files
`example/output/phase4_live_1.csv` and `phase4_live_20.csv`.

The first cold request exposed an additional constraint: loading this model
took 44.2 seconds and pushed total request time beyond the original 60-second
timeout. `OLLAMA_REQUEST_TIMEOUT` was raised to 120 seconds so a valid cold
response is not unnecessarily replaced by a heuristic fallback.

## 8. Phase 5: Propagation and Metrics

`simulation/propagation.py` now turns reactions into new exposures. A
reaction's strongest action controls its base probability: share 80%, comment
40%, like 20%, watch-only 5%, and skip 0%. The probability is multiplied by
`0.7 + 0.3 × edge_weight`, so stronger relationships propagate somewhat more
effectively without making weak ties useless.

When several reacting users connect to the same neighbour, their independent
chances are combined with `1 - product(1 - probability)`. This gives the
neighbour multiple chances without allowing the result to exceed 100%.
Already-exposed users are excluded, every random decision uses the supplied
seeded generator, and the next wave is capped at 30 users. If too many users
pass their probability roll, candidates with the strongest combined exposure
probability are retained.

`summarize_wave()` validates that reactions match exactly the users exposed in
that wave and records shares, comments, likes, skips, average watch, new
exposures, and cumulative reach. `should_continue()` returns both a decision
and a UI-ready explanation. Propagation stops after four waves, below three
new exposures, below 5% sharing, or at 85% population saturation.

`simulation/metrics.py` calculates completion, skip, like, share, comment,
follow, purchase-intent, reach, and reach-percentage metrics over exposed
personas only. `segment_table()` groups reactions by archetype in stable order.
Segments with `n < 3` remain available but can be labelled insufficient sample
by the presentation layer.

The virality score is transparent rather than learned. It combines completion,
sharing, comments, and skipping with the configured weights, passes the raw
value through the documented monotonic calibration curve, adds up to ten
points for reach, and clips the result to 0–100. Calibration spreads realistic
simulation results across a useful display range; it is not fitted to real
platform data.

A deterministic 30-person integration fixture now creates the real homophily
graph, selects mixed seed users, runs up to four synthetic sharing waves,
prevents duplicate exposure, records wave summaries, and finalizes engagement
and segment metrics without Ollama. This verifies that the Phase 3 graph and
Phase 5 algorithms fit together before LangGraph orchestration is introduced.

## 9. Phase 6: LangGraph Workflow

`workflow/state.py` defines `SimulationConfig` and `SimulationState` as typed
dictionaries. `create_initial_state()` validates the video and run options,
clamps seed size to small populations, and initializes every collection and
output field. The reactions field uses LangGraph's additive reducer; all other
fields remain last-write-wins.

`workflow/graph.py` now compiles the complete stateful topology: analyze video,
generate personas, build the social graph, seed the audience, simulate agents,
aggregate the wave, propagate, check thresholds, finalize metrics, and pass
through the reserved Phase 7 recommendation node. A recursion limit of 50 is
set explicitly so a valid four-wave cascade cannot collide with LangGraph's
default safety boundary.

Each node reconstructs Pydantic models at module boundaries and serializes its
output back into state. Video analysis retains hash caching; persona and graph
construction reuse their deterministic seeds. `simulate_agents` can call the
real batched Ollama audience or deterministic heuristics. It returns only the
current wave's reactions, allowing LangGraph's reducer to append them to every
earlier wave.

Propagation writes candidates to `pending_users` first. Threshold checking
commits those IDs to `active_users` and cumulative reach only when every
continuation rule passes. This prevents the stopped workflow from counting
people who were selected probabilistically but never received a simulated
reaction. Per-wave random generators are derived from the base seed and wave
index with `SeedSequence`, so replaying a run reproduces its cascade without
storing a mutable RNG object in state.

Wave summaries are accumulated explicitly because only reactions have a
LangGraph reducer. The threshold node updates each wave's `continued` value and
stores a human-readable reason. The router then reads one Boolean and performs
no duplicate calculation. Finalization verifies that reaction IDs match
exposed IDs exactly before calculating Phase 5 metrics and the virality score.

`scripts/run_workflow.py` provides the headless entry point. It streams each
node update, prints wave progress and stopping decisions, reconstructs the
final state with reducer semantics, prints compact JSON, and optionally exports
the complete serializable state without the in-memory NetworkX graph. Its final
node now generates Phase 7 recommendations and reports whether they came from
Ollama or the deterministic fallback.

The real-video heuristic checkpoint used ten personas and completed every
node. Five seed users reacted, no one shared, propagation stopped after wave
zero, reach was 50%, and the final simulated virality score was 22.87. The
state was saved to `example/output/phase6_heuristic_state.json`.

Checkpoint G also passed with real Ollama inference. A five-person run used the
cached `index.mp4` features, generated five valid reactions with no fallbacks,
and finalized after one wave. All five personas skipped, producing 17% average
watch and a score of 10.00. The run took 63.23 seconds, of which 57.83 seconds
was audience inference. Its gitignored state is stored in
`example/output/phase6_live_state.json`.

## 10. Phase 7: Recommendations and Streamlit UI

`agents/recommender.py` now turns video evidence and simulation results into
creator advice. It selects up to ten representative audience reasons in a
stable order, prioritizing skips and low watch percentages while retaining
archetype variety. The compact prompt contains the opening hook, transcript,
pacing, duration, visual evidence, overall metrics, segment metrics, and wave
summaries. This grounds advice in observed simulation output instead of asking
the model for generic content tips.

The recommender uses one low-temperature Ollama call and validates the result
with the `Recommendations` Pydantic contract. It removes duplicates, sorts
high-priority edits first, limits output to five items, and rejects unknown
segment labels. A common local-model response containing one flattened item is
wrapped into the documented structure without a second call. If inference or
validation still fails, the workflow records the error and returns transparent
rules based on retention, skipping, sharing, pacing, duration, and the weakest
sufficiently sized segment.

`app.py` is now the complete Streamlit presentation layer. Uploads are checked
for type and size, stored in `tmp/uploads/` under their SHA-256 content hash,
and reused safely. The sidebar exposes the audience engine, population, seed
audience, seed strategy, random seed, and wave cap. Ollama health is shown and
cached briefly. The workflow streams node-by-node progress through `st.status`,
and its final state remains in session state across UI reruns.

The six result tabs show the directional score and engagement metrics, video
features and transcript, segment and individual reactions, wave behavior and a
colored NetworkX graph, prioritized recommendations, and raw exports. Plotly
factories and pandas transformations live under `ui/`, keeping business logic
out of the app. The Raw tab discloses fallback counts, errors, timings, CSV
reactions, and downloadable JSON without the runtime-only graph object.

`scripts/recommend_from_state.py` supports fast prompt iteration from a saved
workflow result. The real Ollama checkpoint ran against the saved Phase 6 live
state and returned a validated high-priority recommendation in about 40
seconds. The result is stored in the gitignored
`example/output/phase7_live_recommendations.json`. The full deterministic run
on `index.mp4` scored 22.87/100 and stored three rule-based recommendations in
`phase7_heuristic_state.json`. Streamlit started successfully on port 8502 and
its health endpoint returned `ok`.

## 11. Testing and Verification

The suite currently contains 128 passing tests and makes no real LLM calls or
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
- Dominant-action priority, probabilistic exposure, edge-weight modulation,
  combined exposure paths, wave caps, duplicate prevention, and stop reasons.
- Hand-computed overall metrics, archetype segments, score monotonicity,
  calibration bounds, reach bonus, empty cascades, and a multi-wave integration.
- Initial workflow validation, graph topology, streamed node order, reducer
  accumulation, pending-user admission, dead cascades, deterministic replay,
  final metrics, and recommendation-node fallback reporting.
- Recommendation evidence selection, prompt grounding, response normalization,
  deduplication, priority ordering, and deterministic fallback advice.
- Upload validation, content-addressed storage, UI tables, Plotly figures,
  serializable exports, and Streamlit startup without an upload.

Primary verification commands are:

```powershell
pytest -q
python -m simulation.social_graph --personas 100 --seed 42
python -m scripts.dry_run --video example\index.mp4 --n 10 --seed 42 --heuristic-only
pytest -q tests\test_propagation.py tests\test_metrics.py
pytest -q tests\test_workflow.py
python -m scripts.run_workflow --video example\index.mp4 --n 10 --seed 42 --heuristic-only
streamlit run app.py
python -m scripts.recommend_from_state --state example\output\phase6_live_state.json
```

## 12. Reliability Decisions

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

## 13. Current Status and Remaining Work

Completed implementation includes repository setup, contracts and utilities,
the complete multimodal video-information pipeline, deterministic persona
generation, connected homophily graph construction, the Ollama client layer,
audience reaction prompting, deterministic fallback behavior, batch execution,
diagnostic CLIs, real-video artifacts, probabilistic propagation, explainable
stopping rules, engagement segmentation, calibrated scoring, and automated
tests. LangGraph connects these layers into a streamed, bounded, multi-wave
workflow that produces final metrics and evidence-grounded advice from either
heuristic or real Ollama reactions. The Streamlit application now provides
upload, configuration, progress, all six result tabs, visualizations, fallback
disclosure, and exports.

One major phase remains: **hardening and release**. It includes precomputed
demo mode, exhaustive user-facing error banners, README screenshots and honest
claims, cold-start measurement, and a three-video acceptance matrix covering a
talking head, fast-cut montage, and silent clip.

Phase 4 is functionally complete: installation, health, real inference,
structured validation, prompt spread, fallback behavior, and batching all
work. Its only open acceptance item is Checkpoint E's under-40-second target.
Meeting that target on this hardware requires a smaller model or a faster
inference device; it is a performance limitation rather than a correctness
failure.
