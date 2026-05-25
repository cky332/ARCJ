"""Multi-topology construction for TMCHT (paper §2.1, A.5, A.6, A.7).

Agents are 0-indexed (0 .. N-1). Each builder returns an *order list* ``O``:
a list of rounds, where each round is a list of ``(asker, answerer)`` pairs.
Only adjacent agents are ever paired. After a round the asker/answerer roles
are swapped, matching the paper's interaction protocol (A.7).
"""
from __future__ import annotations

import random
from collections import defaultdict

Pair = tuple[int, int]
Round = list[Pair]
OrderList = list[Round]


# --------------------------------------------------------------------------- #
# Graph topology (complete graph) -- paper Algorithm 1
# --------------------------------------------------------------------------- #
def build_graph(num_agents: int, num_rounds: int, seed: int = 0) -> OrderList:
    """Random matchings: shuffle, pair adjacents, then swap roles (Alg. 1)."""
    rng = random.Random(seed)
    agents = list(range(num_agents))
    order: OrderList = []
    # Each outer iteration produces two rounds (the pairing and its swap).
    while len(order) < num_rounds:
        rng.shuffle(agents)
        pairs: Round = []
        for i in range(0, num_agents - 1, 2):
            pairs.append((agents[i], agents[i + 1]))
        order.append(pairs)
        order.append([(b, a) for (a, b) in pairs])
    return order[:num_rounds]


# --------------------------------------------------------------------------- #
# Line topology (path graph) -- paper Algorithm 2
# --------------------------------------------------------------------------- #
def _line_even_pairs(num_agents: int) -> Round:
    """Pairs (0,1),(2,3),(4,5),... -- the 'offset 0' matching."""
    return [(i, i + 1) for i in range(0, num_agents - 1, 2)]


def _line_odd_pairs(num_agents: int) -> Round:
    """Pairs (1,2),(3,4),(5,6),... -- the 'offset 1' matching."""
    return [(i, i + 1) for i in range(1, num_agents - 1, 2)]


def build_line(num_agents: int, num_rounds: int, seed: int = 0) -> OrderList:
    """Alternate even / odd matchings of a path, swapping roles (Alg. 2).

    round %4 == 1 -> even pairs;          == 2 -> even pairs reversed
    round %4 == 3 -> odd pairs;           == 0 -> odd pairs reversed
    """
    order: OrderList = []
    for r in range(1, num_rounds + 1):
        m = r % 4
        if m == 1:
            pairs = _line_even_pairs(num_agents)
        elif m == 2:
            pairs = [(b, a) for (a, b) in _line_even_pairs(num_agents)]
        elif m == 3:
            pairs = _line_odd_pairs(num_agents)
        else:  # m == 0
            pairs = [(b, a) for (a, b) in _line_odd_pairs(num_agents)]
        order.append(pairs)
    return order


# --------------------------------------------------------------------------- #
# Star topology (center connected to several arms) -- paper A.6
# --------------------------------------------------------------------------- #
def star_arms(num_agents: int, branches: int) -> list[list[int]]:
    """Distribute agents 1..N-1 into ``branches`` arms (paths). Agent 0 is the
    center. Each arm is an ordered list of agent ids extending from the center."""
    branches = max(1, min(branches, num_agents - 1))
    arms: list[list[int]] = [[] for _ in range(branches)]
    for idx, agent in enumerate(range(1, num_agents)):
        arms[idx % branches].append(agent)
    return arms


def star_edges(num_agents: int, branches: int) -> list[Pair]:
    """Edge set of the star: center->arm head and consecutive arm nodes (A.6)."""
    center = 0
    edges: list[Pair] = []
    for arm in star_arms(num_agents, branches):
        prev = center
        for node in arm:
            edges.append((prev, node))
            prev = node
    return edges


def _edge_coloring(edges: list[Pair]) -> list[Round]:
    """Greedy proper edge coloring -> list of matchings (vertex-disjoint pairs)."""
    used: dict[int, set[int]] = defaultdict(set)
    colors: list[Round] = []
    for (u, v) in edges:
        c = 0
        while c in used[u] or c in used[v]:
            c += 1
        used[u].add(c)
        used[v].add(c)
        while len(colors) <= c:
            colors.append([])
        colors[c].append((u, v))
    return colors


def build_star(num_agents: int, num_rounds: int, branches: int = 4,
               seed: int = 0) -> OrderList:
    """Cycle through edge-coloring matchings, swapping roles on alternate passes.

    The paper does not give the exact star ordering (it says star is "similar"
    to line/graph, A.5); we use a matching scheduler that respects the star edge
    set, keeps each round a valid matching, and exercises every edge over time.
    """
    matchings = _edge_coloring(star_edges(num_agents, branches))
    if not matchings:
        return [[] for _ in range(num_rounds)]
    order: OrderList = []
    for r in range(num_rounds):
        base = matchings[(r // 2) % len(matchings)]
        if r % 2 == 0:
            order.append(list(base))
        else:
            order.append([(b, a) for (a, b) in base])
    return order


# --------------------------------------------------------------------------- #
# Dispatch + connectivity helpers
# --------------------------------------------------------------------------- #
def build_topology(name: str, num_agents: int, num_rounds: int, seed: int = 0,
                   star_branches: int = 4) -> OrderList:
    name = name.lower()
    if name == "graph":
        return build_graph(num_agents, num_rounds, seed)
    if name == "line":
        return build_line(num_agents, num_rounds, seed)
    if name == "star":
        return build_star(num_agents, num_rounds, star_branches, seed)
    raise ValueError(f"Unknown topology: {name!r} (expected graph/line/star)")


def neighbors(name: str, num_agents: int, star_branches: int = 4) -> dict[int, set[int]]:
    """Adjacency map (which agents may ever communicate)."""
    name = name.lower()
    adj: dict[int, set[int]] = defaultdict(set)
    if name == "graph":
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    adj[i].add(j)
    elif name == "line":
        for i in range(num_agents - 1):
            adj[i].add(i + 1)
            adj[i + 1].add(i)
    elif name == "star":
        for (u, v) in star_edges(num_agents, star_branches):
            adj[u].add(v)
            adj[v].add(u)
    else:
        raise ValueError(f"Unknown topology: {name!r}")
    return adj
