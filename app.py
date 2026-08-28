"""Streamlit presentation layer for the Viralyst simulation workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import networkx as nx
import streamlit as st

from agents.llm import health_check
from config import (
    MAX_NEW_PER_WAVE,
    MAX_UPLOAD_MB,
    N_PERSONAS,
    OLLAMA_MODEL,
    RANDOM_SEED,
    SEED_SIZE,
    SEED_STRATEGY,
)
from ui.charts import (
    network_chart,
    score_gauge,
    segment_engagement_chart,
    wave_chart,
)
from ui.data import (
    fallback_counts,
    reaction_frame,
    score_label,
    segment_frame,
    wave_frame,
)
from ui.files import UploadValidationError, save_uploaded_video, serializable_state
from workflow.graph import merge_update, stream_workflow
from workflow.state import SimulationState, create_initial_state

st.set_page_config(page_title="Viralyst", page_icon="📈", layout="wide")

NODE_LABELS = {
    "analyze_video": "Extracting video, audio, and visual features",
    "generate_personas": "Generating the synthetic audience",
    "build_social_graph": "Building the audience social graph",
    "seed_audience": "Selecting the initial viewers",
    "simulate_agents": "Simulating audience reactions",
    "aggregate_wave": "Summarizing the current wave",
    "propagate": "Calculating possible new exposures",
    "check_threshold": "Checking whether propagation continues",
    "finalize_metrics": "Calculating engagement metrics and score",
    "generate_recommendations": "Generating evidence-based recommendations",
}


@st.cache_data(ttl=5, show_spinner=False)
def cached_health_check() -> tuple[bool, str]:
    """Avoid contacting Ollama repeatedly during normal Streamlit reruns."""

    return health_check()


def run_simulation(video_path: str, settings: dict[str, Any]) -> SimulationState:
    """Stream workflow updates into one final state and visible status box."""

    initial = create_initial_state(video_path, **settings)
    state = initial
    progress = st.progress(0, text="Starting the workflow")
    completed_events = 0
    with st.status("Running Viralyst…", expanded=True) as status:
        for event in stream_workflow(initial):
            for node_name, update in event.items():
                if not isinstance(update, Mapping):
                    continue
                state = merge_update(state, update)
                completed_events += 1
                label = NODE_LABELS.get(node_name, node_name.replace("_", " "))
                st.write(f"✓ {label}")
                progress.progress(
                    min(0.95, completed_events / 16),
                    text=label,
                )
        progress.progress(1.0, text="Simulation complete")
        status.update(label="Simulation complete", state="complete")
    return state


def render_overview(state: SimulationState) -> None:
    metrics = state["engagement_metrics"]
    score = float(state["virality_score"])
    gauge_column, metrics_column = st.columns((1.15, 1.85))
    with gauge_column:
        st.plotly_chart(score_gauge(score), use_container_width=True)
        st.caption(score_label(score))
    with metrics_column:
        first = st.columns(3)
        first[0].metric("Synthetic reach", f"{metrics['reach_pct']:.0%}")
        first[1].metric("Average watched", f"{metrics['completion_rate']:.0%}")
        first[2].metric("Skip rate", f"{metrics['skip_rate']:.0%}")
        second = st.columns(3)
        second[0].metric("Like rate", f"{metrics['like_rate']:.0%}")
        second[1].metric("Share rate", f"{metrics['share_rate']:.0%}")
        second[2].metric("Comment rate", f"{metrics['comment_rate']:.0%}")
        st.info(state.get("stop_reason", "Simulation finished."))
    st.warning(
        "This score is a simulated directional signal from synthetic personas. "
        "It is not a real-world virality prediction or guarantee."
    )


def render_video_analysis(state: SimulationState) -> None:
    features = state["video_features"]
    columns = st.columns(4)
    columns[0].metric("Duration", f"{features['duration_sec']:.1f}s")
    columns[1].metric("Pacing", str(features["pacing_label"]).title())
    columns[2].metric("Scene changes", features["scene_changes"])
    columns[3].metric("Speech rate", f"{features['speech_rate_wps']:.2f} w/s")
    st.subheader("Opening hook (0–3 seconds)")
    st.write(features.get("hook_transcript") or "No spoken hook was detected.")
    left, right = st.columns(2)
    with left:
        st.subheader("Visual evidence")
        st.json(
            {
                "orientation": (
                    "vertical" if features["is_vertical"] else "horizontal"
                ),
                "motion_score": features["motion_score"],
                "visual_density": features["density_label"],
                "average_brightness": features["avg_brightness"],
                "objects": features["objects"],
            },
            expanded=True,
        )
    with right:
        st.subheader("Transcript")
        st.text_area(
            "Extracted speech",
            value=features.get("transcript") or "No speech detected.",
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )


def render_audience(state: SimulationState) -> None:
    segments = segment_frame(state)
    reactions = reaction_frame(state)
    st.subheader("Segment response")
    if segments.empty:
        st.info("No segment results are available.")
        return
    st.plotly_chart(segment_engagement_chart(segments), use_container_width=True)
    st.dataframe(
        segments.rename(
            columns={
                "segment": "Segment",
                "n": "Viewers",
                "avg_watch": "Avg. watched (%)",
                "like_rate": "Likes (%)",
                "share_rate": "Shares (%)",
                "comment_rate": "Comments (%)",
                "skip_rate": "Skips (%)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Individual synthetic reactions")
    st.dataframe(
        reactions[
            [
                "name",
                "segment",
                "wave",
                "watch_percentage",
                "skipped",
                "liked",
                "commented",
                "shared",
                "reason",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_propagation(state: SimulationState) -> None:
    waves = wave_frame(state)
    if waves.empty:
        st.info("No propagation waves are available.")
        return
    st.plotly_chart(wave_chart(waves), use_container_width=True)
    graph = state.get("social_graph")
    if isinstance(graph, nx.Graph):
        st.subheader("Audience network")
        st.caption("Purple nodes were reached; grey nodes were not reached.")
        st.plotly_chart(
            network_chart(
                graph,
                set(state.get("exposed_ids", [])),
                int(state["config"]["seed"]),
            ),
            use_container_width=True,
        )
    st.subheader("Wave details")
    st.dataframe(waves, use_container_width=True, hide_index=True)
    st.info(state.get("stop_reason", "No stop reason was recorded."))


def render_recommendations(state: SimulationState) -> None:
    recommendations = state.get("recommendations", {})
    if state.get("recommendation_fallback"):
        st.warning(
            "The recommendation model was unavailable or disabled. These "
            "recommendations were generated by transparent metric rules."
        )
    st.subheader("What the simulation suggests")
    st.write(recommendations.get("summary", "No recommendations are available."))
    strengths = recommendations.get("strengths", [])
    if strengths:
        st.subheader("Strengths to preserve")
        for strength in strengths:
            st.success(strength)
    for index, item in enumerate(recommendations.get("items", []), start=1):
        priority = str(item["priority"]).upper()
        target = item.get("target_segment") or "All reached segments"
        with st.container(border=True):
            st.markdown(f"#### {index}. {priority} · {target}")
            st.write(f"**Problem:** {item['problem']}")
            st.write(f"**Likely cause:** {item['likely_cause']}")
            st.write(f"**Edit to test:** {item['recommendation']}")


def render_raw(state: SimulationState) -> None:
    reactions = reaction_frame(state)
    clean_state = serializable_state(dict(state))
    reaction_fallbacks, reaction_count = fallback_counts(state)
    columns = st.columns(3)
    columns[0].metric("Reaction fallbacks", f"{reaction_fallbacks}/{reaction_count}")
    columns[1].metric(
        "Recommendation source",
        "Rules" if state.get("recommendation_fallback") else "Ollama",
    )
    columns[2].metric("Recorded errors", len(state.get("errors", [])))
    if state.get("errors"):
        with st.expander("Controlled errors and fallback reasons"):
            for error in state["errors"]:
                st.write(f"• {error}")
    st.subheader("Runtime by stage")
    st.json(state.get("timings", {}), expanded=True)
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download reactions as CSV",
        data=reactions.to_csv(index=False).encode("utf-8"),
        file_name="viralyst_reactions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_columns[1].download_button(
        "Download full result as JSON",
        data=json.dumps(clean_state, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="viralyst_result.json",
        mime="application/json",
        use_container_width=True,
    )
    with st.expander("Complete serializable workflow state"):
        st.json(clean_state, expanded=False)


st.title("Viralyst")
st.caption("Understand how a synthetic audience might respond to a short video.")

if "result_state" not in st.session_state:
    st.session_state.result_state = None
if "upload_digest" not in st.session_state:
    st.session_state.upload_digest = None

with st.sidebar:
    st.header("Simulation setup")
    run_mode = st.radio(
        "Audience engine",
        ("Local AI (Ollama)", "Fast deterministic preview"),
        help=(
            "Ollama produces richer individual reasons. The preview mode is "
            "fast, repeatable, and does not call a language model."
        ),
    )
    heuristic_only = run_mode == "Fast deterministic preview"
    ollama_ready, ollama_message = cached_health_check()
    if ollama_ready:
        st.success(f"Ollama ready · {OLLAMA_MODEL}")
    else:
        st.warning(ollama_message)
    n_personas = st.slider("Synthetic personas", 5, 200, N_PERSONAS, 5)
    seed_size = st.slider(
        "Initial viewers",
        1,
        min(20, n_personas),
        min(SEED_SIZE, n_personas),
    )
    seed_strategy = st.selectbox(
        "Initial viewer strategy",
        ("mixed", "degree", "random"),
        index=("mixed", "degree", "random").index(SEED_STRATEGY),
    )
    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        value=RANDOM_SEED,
        step=1,
    )
    max_new_per_wave = st.slider(
        "Maximum new viewers per wave",
        3,
        100,
        MAX_NEW_PER_WAVE,
    )

uploaded = st.file_uploader(
    "Upload a video",
    type=("mp4", "mov", "mkv", "webm"),
    help=f"Maximum upload size: {MAX_UPLOAD_MB} MB.",
)

video_path: str | None = None
upload_error: str | None = None
if uploaded is not None:
    data = uploaded.getvalue()
    try:
        stored_path, digest = save_uploaded_video(uploaded.name, data)
        video_path = str(stored_path)
        st.video(data)
        if st.session_state.upload_digest != digest:
            st.session_state.upload_digest = digest
            st.session_state.result_state = None
    except UploadValidationError as exc:
        upload_error = str(exc)
        st.error(upload_error)

live_blocked = not heuristic_only and not ollama_ready
run_disabled = video_path is None or upload_error is not None or live_blocked
if live_blocked:
    st.info(
        "Start Ollama and install the configured model, or choose the fast "
        "deterministic preview."
    )

if st.button(
    "Run simulation",
    type="primary",
    disabled=run_disabled,
    use_container_width=True,
):
    settings = {
        "n_personas": int(n_personas),
        "seed": int(random_seed),
        "seed_size": int(seed_size),
        "seed_strategy": str(seed_strategy),
        "max_new_per_wave": int(max_new_per_wave),
        "heuristic_only": heuristic_only,
        "use_video_cache": True,
    }
    try:
        st.session_state.result_state = run_simulation(video_path, settings)
    except Exception as exc:
        st.session_state.result_state = None
        st.error(f"The simulation could not finish: {exc}")

result = st.session_state.result_state
if result:
    tabs = st.tabs(
        (
            "Overview",
            "Video analysis",
            "Audience",
            "Propagation",
            "Recommendations",
            "Raw",
        )
    )
    with tabs[0]:
        render_overview(result)
    with tabs[1]:
        render_video_analysis(result)
    with tabs[2]:
        render_audience(result)
    with tabs[3]:
        render_propagation(result)
    with tabs[4]:
        render_recommendations(result)
    with tabs[5]:
        render_raw(result)
else:
    st.info(
        "Upload a video, choose the audience settings, and run the simulation. "
        "Results from every stage will appear here."
    )
