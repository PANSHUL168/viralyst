"""Plotly chart factories used by the Streamlit presentation layer."""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def score_gauge(score: float) -> go.Figure:
    """Build a compact zero-to-one-hundred directional score gauge."""

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#7C3AED"},
                "steps": [
                    {"range": [0, 25], "color": "#FEE2E2"},
                    {"range": [25, 50], "color": "#FEF3C7"},
                    {"range": [50, 75], "color": "#DBEAFE"},
                    {"range": [75, 100], "color": "#DCFCE7"},
                ],
            },
        )
    )
    figure.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
    return figure


def segment_engagement_chart(frame: pd.DataFrame) -> go.Figure:
    """Compare the main segment-level engagement rates."""

    columns = ["like_rate", "share_rate", "comment_rate", "skip_rate"]
    available = [column for column in columns if column in frame]
    melted = frame.melt(
        id_vars="segment",
        value_vars=available,
        var_name="metric",
        value_name="percent",
    )
    figure = px.bar(
        melted,
        x="segment",
        y="percent",
        color="metric",
        barmode="group",
        labels={"segment": "Audience segment", "percent": "Rate (%)"},
    )
    figure.update_layout(legend_title_text="Metric", height=390)
    return figure


def wave_chart(frame: pd.DataFrame) -> go.Figure:
    """Show new and cumulative exposure across propagation waves."""

    figure = go.Figure()
    figure.add_bar(
        x=frame["index"],
        y=frame["new_exposures"],
        name="New exposures",
        marker_color="#A78BFA",
    )
    figure.add_scatter(
        x=frame["index"],
        y=frame["cumulative_reached"],
        name="Cumulative reach",
        mode="lines+markers",
        line={"color": "#2563EB", "width": 3},
    )
    figure.update_layout(
        xaxis_title="Wave",
        yaxis_title="Synthetic personas",
        height=370,
        hovermode="x unified",
    )
    return figure


def network_chart(
    graph: nx.Graph,
    exposed_ids: set[int],
    seed: int,
) -> go.Figure:
    """Render the social graph, distinguishing reached and unreached nodes."""

    positions = nx.spring_layout(graph, seed=seed)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for left, right in graph.edges:
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        edge_x.extend((float(x0), float(x1), None))
        edge_y.extend((float(y0), float(y1), None))
    edges = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        hoverinfo="skip",
        line={"width": 0.5, "color": "#CBD5E1"},
    )
    nodes = sorted(graph.nodes)
    node_trace = go.Scatter(
        x=[float(positions[node][0]) for node in nodes],
        y=[float(positions[node][1]) for node in nodes],
        mode="markers",
        text=[
            f"Persona {node}<br>{'Reached' if node in exposed_ids else 'Unreached'}"
            for node in nodes
        ],
        hoverinfo="text",
        marker={
            "size": [11 if node in exposed_ids else 7 for node in nodes],
            "color": [
                "#7C3AED" if node in exposed_ids else "#CBD5E1"
                for node in nodes
            ],
            "line": {"width": 0.5, "color": "white"},
        },
    )
    figure = go.Figure((edges, node_trace))
    figure.update_layout(
        height=470,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


__all__ = [
    "network_chart",
    "score_gauge",
    "segment_engagement_chart",
    "wave_chart",
]
