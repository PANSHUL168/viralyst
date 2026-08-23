# Repository Guidelines

## Project Structure & Module Organization

`VIRALYST.md` is the single source of truth for scope, contracts, and architecture. Keep the Streamlit entry point in `app.py`; shared settings, Pydantic contracts, and utilities belong in `config.py`, `schemas.py`, and `utils.py`. Organize implementation by responsibility:

- `video/`: OpenCV extraction, Whisper transcription, YOLO detection, and feature fusion.
- `personas/`: deterministic synthetic population generation.
- `simulation/`: NetworkX graph construction, propagation, and metrics.
- `agents/`: Ollama client, audience reactions, and recommendations.
- `workflow/`: LangGraph state and orchestration.
- `tests/`: unit tests mirroring these modules.

Generated files belong in gitignored `cache/` and `tmp/`. Dependencies must flow downward as defined in `VIRALYST.md`; presentation code must not contain simulation or scoring logic.

## Build, Test, and Development Commands

Use Python 3.11 or 3.12 and create a local environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

Run `ollama serve` separately and pull the model configured in `config.py`. Use `python -m scripts.dry_run --video sample.mp4 --n 10 --seed 42` for prompt iteration once that utility exists.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation, type hints, and small single-purpose functions. Use `snake_case` for modules and functions, `PascalCase` for Pydantic models, and uppercase names for configuration constants. Put every tunable value in `config.py`. Never name modules `whisper.py`, `yolo.py`, `types.py`, `json.py`, `random.py`, or `logging.py`. Use structured models at module boundaries rather than unvalidated dictionaries.

## Testing Guidelines

Use `pytest`; name files `test_<module>.py` and tests `test_<behavior>()`. Tests must not call Ollama or download models—mock external inference. Prioritize deterministic persona generation, graph connectivity, propagation limits, metric edge cases, and score monotonicity. The complete suite should run in under five seconds.

## Commit & Pull Request Guidelines

No repository history exists yet. Use short, imperative Conventional Commit messages, for example `feat: add persona generator` or `test: cover empty cascades`. Keep commits focused. Pull requests should describe the change, verification commands, configuration or contract changes, and known limitations. Include screenshots for Streamlit UI changes and update `VIRALYST.md` whenever scope or architecture intentionally changes.

## Reliability & Configuration

Never commit videos, model weights, `.env`, caches, or temporary media. Fail gracefully for missing audio, empty detections, malformed LLM JSON, and dead cascades. Describe scores as simulated directional signals, never validated virality predictions.
