# Viralyst

Viralyst is a local-first synthetic audience and content-propagation simulator
for short-form video. Its results are directional simulation signals, not
validated predictions of real-world virality.

The complete specification and implementation history live in
[`VIRALYST.md`](VIRALYST.md) and [`process.md`](process.md).

## Setup

Use Python 3.11 or 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

FFmpeg and Ollama are required outside Python. Pull the Ollama model configured
in `.env` before using the local-AI audience mode.

## Run Viralyst

Start Ollama in one terminal and Streamlit in another:

```powershell
ollama serve
streamlit run app.py
```

Upload an MP4, MOV, MKV, or WebM video, choose the simulation settings, and
press **Run simulation**. Select **Fast deterministic preview** to run without
Ollama. Results appear in six tabs: Overview, Video analysis, Audience,
Propagation, Recommendations, and Raw. The Raw tab exports reactions as CSV
and the serializable workflow result as JSON.

## Runtime Code

- `app.py` and `ui/`: Streamlit interface and visualizations.
- `video/`: audio, frame, object, transcript, and feature extraction.
- `personas/`: deterministic synthetic audience generation.
- `simulation/`: social graph, propagation, metrics, and scoring.
- `agents/`: Ollama reactions and recommendations with safe fallbacks.
- `workflow/`: complete LangGraph orchestration.
- `config.py`, `schemas.py`, and `utils.py`: shared runtime configuration,
  validated contracts, and infrastructure.
