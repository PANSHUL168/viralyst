# Viralyst

Viralyst is a local-first synthetic audience and content-propagation simulator for short-form video. Its scores are directional simulation outputs, not validated predictions of real-world virality.

The complete MVP specification lives in [`VIRALYST.md`](VIRALYST.md).

## Local setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
streamlit run app.py
```

FFmpeg and Ollama are required outside Python. Start Ollama with `ollama serve` and pull the model named in `.env` before running a live simulation.

## Phase 4 audience dry run

After Ollama is ready, validate persona reactions against a real video:

```powershell
python -m scripts.dry_run --video example\index.mp4 --n 10 --seed 42
```

Use `--heuristic-only` to test the complete local pipeline without Ollama. Add
`--output example\output\reactions.csv` to retain the reaction-level results.
