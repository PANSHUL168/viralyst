"""Deterministic homophily-based social graph construction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from config import (
    HOMOPHILY_THRESHOLD,
    MAX_DEGREE,
    N_PERSONAS,
    RANDOM_EDGE_RATIO,
    RANDOM_SEED,
    SEED_SIZE,
    SEED_STRATEGY,
)
from personas.generator import generate_personas, population_summary
from schemas import Persona

BIG_FIVE_TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


def persona_similarity(left: Persona, right: Persona) -> float:
    """Return the documented weighted similarity between two personas."""

    left_interests = set(left.interests)
    right_interests = set(right.interests)
    union = left_interests | right_interests
    interest_similarity = (
        len(left_interests & right_interests) / len(union) if union else 1.0
    )
    age_similarity = 1.0 - min(abs(left.age - right.age) / 40.0, 1.0)
    archetype_similarity = float(left.archetype == right.archetype)
    personality_distance = np.mean(
        [abs(getattr(left, trait) - getattr(right, trait)) for trait in BIG_FIVE_TRAITS]
    )
    personality_similarity = 1.0 - float(personality_distance)

    similarity = (
        0.50 * interest_similarity
        + 0.20 * age_similarity
        + 0.15 * archetype_similarity
        + 0.15 * personality_similarity
    )
    return round(float(np.clip(similarity, 0.0, 1.0)), 6)


def build_graph(
    personas: Sequence[Persona],
    seed: int = RANDOM_SEED,
) -> nx.Graph:
    """Build a connected, degree-bounded graph from synthetic personas."""

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    persona_by_id = {persona.id: persona for persona in personas}
    if len(persona_by_id) != len(personas):
        raise ValueError("persona IDs must be unique")

    graph = nx.Graph()
    for persona in sorted(personas, key=lambda item: item.id):
        graph.add_node(persona.id, **persona.model_dump(mode="json"))
    if graph.number_of_nodes() <= 1:
        return nx.freeze(graph)

    candidates: list[tuple[float, int, int]] = []
    ordered_personas = sorted(personas, key=lambda item: item.id)
    for index, left in enumerate(ordered_personas):
        for right in ordered_personas[index + 1 :]:
            similarity = persona_similarity(left, right)
            if similarity > HOMOPHILY_THRESHOLD:
                candidates.append((similarity, left.id, right.id))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    for similarity, left_id, right_id in candidates:
        if graph.degree(left_id) >= MAX_DEGREE:
            continue
        if graph.degree(right_id) >= MAX_DEGREE:
            continue
        graph.add_edge(
            left_id,
            right_id,
            weight=similarity,
            kind="homophily",
        )

    _add_random_bridges(graph, persona_by_id, seed)
    _repair_connectivity(graph, persona_by_id)
    return nx.freeze(graph)


def _add_random_bridges(
    graph: nx.Graph,
    persona_by_id: dict[int, Persona],
    seed: int,
) -> None:
    """Add seeded weak ties, preferring cross-archetype pairs."""

    target_count = round(RANDOM_EDGE_RATIO * graph.number_of_nodes())
    if target_count <= 0:
        return

    nodes = sorted(graph.nodes)
    candidates: list[tuple[int, int]] = []
    for index, left_id in enumerate(nodes):
        for right_id in nodes[index + 1 :]:
            if graph.has_edge(left_id, right_id):
                continue
            if graph.degree(left_id) >= MAX_DEGREE:
                continue
            if graph.degree(right_id) >= MAX_DEGREE:
                continue
            candidates.append((left_id, right_id))

    rng = np.random.default_rng(seed)
    cross_archetype = [
        pair
        for pair in candidates
        if persona_by_id[pair[0]].archetype != persona_by_id[pair[1]].archetype
    ]
    cross_archetype_set = set(cross_archetype)
    same_archetype = [
        pair for pair in candidates if pair not in cross_archetype_set
    ]
    ordered_candidates = _shuffled(cross_archetype, rng) + _shuffled(
        same_archetype,
        rng,
    )

    added = 0
    for left_id, right_id in ordered_candidates:
        if added >= target_count:
            break
        if graph.has_edge(left_id, right_id):
            continue
        if graph.degree(left_id) >= MAX_DEGREE:
            continue
        if graph.degree(right_id) >= MAX_DEGREE:
            continue
        graph.add_edge(
            left_id,
            right_id,
            weight=persona_similarity(
                persona_by_id[left_id],
                persona_by_id[right_id],
            ),
            kind="bridge",
        )
        added += 1


def _shuffled(
    values: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if not values:
        return []
    return [values[index] for index in rng.permutation(len(values)).tolist()]


def _repair_connectivity(
    graph: nx.Graph,
    persona_by_id: dict[int, Persona],
) -> None:
    """Join components without violating the configured degree ceiling."""

    while nx.number_connected_components(graph) > 1:
        components = sorted(
            nx.connected_components(graph),
            key=lambda component: (-len(component), min(component)),
        )
        anchor = set(components[0])
        other = set(components[1])
        left_candidates = _nodes_with_capacity(graph, anchor)
        right_candidates = _nodes_with_capacity(graph, other)
        if not left_candidates:
            left_candidates = [_free_capacity(graph, anchor)]
        if not right_candidates:
            right_candidates = [_free_capacity(graph, other)]

        repair_candidates = [
            (
                persona_similarity(persona_by_id[left_id], persona_by_id[right_id]),
                graph.degree(left_id) + graph.degree(right_id),
                left_id,
                right_id,
            )
            for left_id in left_candidates
            for right_id in right_candidates
        ]
        similarity, _, left_id, right_id = min(
            repair_candidates,
            key=lambda item: (-item[0], -item[1], item[2], item[3]),
        )
        graph.add_edge(
            left_id,
            right_id,
            weight=similarity,
            kind="repair",
        )


def _nodes_with_capacity(graph: nx.Graph, component: set[int]) -> list[int]:
    return sorted(
        (node for node in component if graph.degree(node) < MAX_DEGREE),
        key=lambda node: (-graph.degree(node), node),
    )


def _free_capacity(graph: nx.Graph, component: set[int]) -> int:
    """Remove the weakest non-bridge edge from a saturated component."""

    subgraph = graph.subgraph(component)
    bridge_edges = {tuple(sorted(edge)) for edge in nx.bridges(subgraph)}
    removable = [
        (
            float(data.get("weight", 0.0)),
            min(left_id, right_id),
            max(left_id, right_id),
        )
        for left_id, right_id, data in subgraph.edges(data=True)
        if tuple(sorted((left_id, right_id))) not in bridge_edges
    ]
    if not removable:
        raise RuntimeError(
            "cannot repair graph connectivity without exceeding MAX_DEGREE"
        )
    _, left_id, right_id = min(removable)
    graph.remove_edge(left_id, right_id)
    return min(left_id, right_id)


def graph_stats(graph: nx.Graph) -> dict[str, int | float | None]:
    """Return safe structural statistics for display and validation."""

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    if node_count == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "avg_degree": 0.0,
            "max_degree": 0,
            "density": 0.0,
            "clustering_coefficient": 0.0,
            "n_components": 0,
            "largest_component_size": 0,
            "avg_shortest_path_length": 0.0,
        }

    components = list(nx.connected_components(graph))
    connected = len(components) == 1
    return {
        "n_nodes": node_count,
        "n_edges": edge_count,
        "avg_degree": 2.0 * edge_count / node_count,
        "max_degree": max(dict(graph.degree()).values(), default=0),
        "density": nx.density(graph),
        "clustering_coefficient": nx.average_clustering(graph),
        "n_components": len(components),
        "largest_component_size": max(map(len, components), default=0),
        "avg_shortest_path_length": (
            nx.average_shortest_path_length(graph) if connected else None
        ),
    }


def select_seed_users(
    graph: nx.Graph,
    personas: Sequence[Persona],
    k: int = SEED_SIZE,
    seed: int = RANDOM_SEED,
    strategy: str = SEED_STRATEGY,
) -> list[int]:
    """Select deterministic seed users using random, degree, or mixed policy."""

    if not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k < 0 or k > graph.number_of_nodes():
        raise ValueError("k must be between zero and the graph's node count")
    if strategy not in {"random", "degree", "mixed"}:
        raise ValueError("strategy must be 'random', 'degree', or 'mixed'")
    persona_ids = {persona.id for persona in personas}
    if not set(graph.nodes).issubset(persona_ids):
        raise ValueError("every graph node must have a matching persona")
    if k == 0:
        return []

    nodes = sorted(graph.nodes)
    ranked = sorted(nodes, key=lambda node: (-graph.degree(node), node))
    if strategy == "degree":
        return ranked[:k]

    rng = np.random.default_rng(seed)
    if strategy == "random":
        return [int(node) for node in rng.choice(nodes, size=k, replace=False)]

    influencer_count = min(2, k)
    selected = ranked[:influencer_count]
    remaining = [node for node in nodes if node not in selected]
    random_count = k - influencer_count
    if random_count:
        selected.extend(
            int(node)
            for node in rng.choice(remaining, size=random_count, replace=False)
        )
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--personas", type=int, default=N_PERSONAS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--strategy",
        choices=("random", "degree", "mixed"),
        default=SEED_STRATEGY,
    )
    parser.add_argument("--output", type=Path, help="Optional full graph JSON path")
    args = parser.parse_args(argv)

    personas = generate_personas(args.personas, args.seed)
    graph = build_graph(personas, args.seed)
    stats = graph_stats(graph)
    seeds = select_seed_users(
        graph,
        personas,
        k=min(SEED_SIZE, len(personas)),
        seed=args.seed,
        strategy=args.strategy,
    )
    payload: dict[str, Any] = {
        "graph_stats": stats,
        "seed_strategy": args.strategy,
        "seed_users": seeds,
        "archetype_counts": {
            str(row.archetype): int(row.count)
            for row in population_summary(personas).itertuples()
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export = {
            **payload,
            "personas": [persona.model_dump(mode="json") for persona in personas],
            "edges": [
                {
                    "source": left_id,
                    "target": right_id,
                    **data,
                }
                for left_id, right_id, data in graph.edges(data=True)
            ],
        }
        args.output.write_text(
            json.dumps(export, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        payload["output"] = str(args.output)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BIG_FIVE_TRAITS",
    "build_graph",
    "graph_stats",
    "persona_similarity",
    "select_seed_users",
]
