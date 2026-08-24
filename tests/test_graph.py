"""Tests for deterministic homophily-based graph construction."""

from __future__ import annotations

import json

import networkx as nx
import pytest

from config import MAX_DEGREE
from personas.generator import generate_personas
from schemas import Persona
from simulation import social_graph
from simulation.social_graph import (
    build_graph,
    graph_stats,
    persona_similarity,
    select_seed_users,
)


@pytest.fixture(scope="module")
def personas_100() -> list[Persona]:
    return generate_personas(100, seed=42)


@pytest.fixture(scope="module")
def graph_100(personas_100) -> nx.Graph:
    return build_graph(personas_100, seed=42)


def _edge_signature(graph: nx.Graph) -> list[tuple[int, int, float, str]]:
    return sorted(
        (
            min(left_id, right_id),
            max(left_id, right_id),
            float(data["weight"]),
            str(data["kind"]),
        )
        for left_id, right_id, data in graph.edges(data=True)
    )


def test_persona_similarity_is_symmetric_and_bounded(personas_100) -> None:
    left, right = personas_100[:2]

    assert persona_similarity(left, right) == persona_similarity(right, left)
    assert 0.0 <= persona_similarity(left, right) <= 1.0
    assert persona_similarity(left, left) == 1.0


def test_graph_is_connected_frozen_and_degree_bounded(graph_100) -> None:
    stats = graph_stats(graph_100)

    assert nx.is_connected(graph_100)
    assert nx.is_frozen(graph_100)
    assert stats["n_components"] == 1
    assert stats["largest_component_size"] == 100
    assert stats["max_degree"] <= MAX_DEGREE
    assert 4.0 <= stats["avg_degree"] <= 12.0


def test_graph_nodes_retain_complete_persona_attributes(
    graph_100,
    personas_100,
) -> None:
    persona = personas_100[37]
    attributes = graph_100.nodes[persona.id]

    assert attributes["name"] == "P37"
    assert attributes["archetype"] == persona.archetype
    assert attributes["interests"] == persona.interests
    assert attributes["share_propensity"] == persona.share_propensity


def test_graph_edges_have_valid_weight_and_kind(graph_100) -> None:
    valid_kinds = {"homophily", "bridge", "repair"}

    for _, _, data in graph_100.edges(data=True):
        assert 0.0 <= data["weight"] <= 1.0
        assert data["kind"] in valid_kinds


def test_same_seed_produces_identical_edge_set(personas_100) -> None:
    first = build_graph(personas_100, seed=42)
    second = build_graph(personas_100, seed=42)

    assert _edge_signature(first) == _edge_signature(second)


def test_homophily_clusters_more_than_equal_density_random_graph(graph_100) -> None:
    random_graph = nx.gnp_random_graph(
        graph_100.number_of_nodes(),
        nx.density(graph_100),
        seed=42,
    )

    assert nx.average_clustering(graph_100) > nx.average_clustering(random_graph)


def test_connectivity_repair_joins_dissimilar_personas() -> None:
    base = generate_personas(3, seed=7)
    personas = [
        persona.model_copy(
            update={
                "archetype": f"isolated_{index}",
                "interests": [f"interest_{index}"],
                "age": 16 + 19 * index,
            }
        )
        for index, persona in enumerate(base)
    ]

    graph = build_graph(personas, seed=7)

    assert nx.is_connected(graph)
    assert graph.number_of_edges() == 2
    assert set(nx.get_edge_attributes(graph, "kind").values()) == {"repair"}


@pytest.mark.parametrize("strategy", ["random", "degree", "mixed"])
def test_seed_selection_is_deterministic(
    strategy,
    graph_100,
    personas_100,
) -> None:
    first = select_seed_users(
        graph_100,
        personas_100,
        k=5,
        seed=42,
        strategy=strategy,
    )
    second = select_seed_users(
        graph_100,
        personas_100,
        k=5,
        seed=42,
        strategy=strategy,
    )

    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5


def test_degree_seed_strategy_selects_highest_degree(
    graph_100,
    personas_100,
) -> None:
    expected = sorted(
        graph_100.nodes,
        key=lambda node: (-graph_100.degree(node), node),
    )[:5]

    assert select_seed_users(
        graph_100,
        personas_100,
        k=5,
        strategy="degree",
    ) == expected


def test_graph_stats_handle_empty_and_single_node() -> None:
    empty = build_graph([])
    single_persona = generate_personas(1, seed=42)
    single = build_graph(single_persona)

    assert graph_stats(empty)["n_components"] == 0
    assert graph_stats(empty)["avg_degree"] == 0.0
    assert graph_stats(single)["n_components"] == 1
    assert graph_stats(single)["avg_shortest_path_length"] == 0.0


def test_seed_selection_rejects_invalid_inputs(graph_100, personas_100) -> None:
    with pytest.raises(ValueError, match="between zero"):
        select_seed_users(graph_100, personas_100, k=101)
    with pytest.raises(ValueError, match="strategy"):
        select_seed_users(
            graph_100,
            personas_100,
            k=5,
            strategy="unknown",
        )


def test_cli_can_export_full_graph(tmp_path, capsys) -> None:
    output_path = tmp_path / "graph.json"

    assert social_graph.main(
        [
            "--personas",
            "20",
            "--seed",
            "42",
            "--output",
            str(output_path),
        ]
    ) == 0
    console = json.loads(capsys.readouterr().out)
    exported = json.loads(output_path.read_text("utf-8"))

    assert console["graph_stats"]["n_nodes"] == 20
    assert console["output"] == str(output_path)
    assert len(exported["personas"]) == 20
    assert len(exported["edges"]) == exported["graph_stats"]["n_edges"]
