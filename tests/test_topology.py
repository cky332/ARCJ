from arcj import topology as T


def _is_matching(round_pairs, num_agents):
    seen = set()
    for a, b in round_pairs:
        assert 0 <= a < num_agents and 0 <= b < num_agents
        assert a != b
        if a in seen or b in seen:
            return False
        seen.add(a)
        seen.add(b)
    return True


def test_graph_rounds_are_valid_matchings():
    N, R = 20, 40
    order = T.build_graph(N, R, seed=1)
    assert len(order) == R
    for rnd in order:
        assert _is_matching(rnd, N)
        assert len(rnd) == N // 2  # even N -> everyone paired


def test_graph_swap_pairs():
    order = T.build_graph(6, 2, seed=3)
    first, second = order[0], order[1]
    assert [(b, a) for (a, b) in first] == second


def test_line_matches_paper_example():
    # Paper A.7 (1-indexed) maps to these 0-indexed rounds for 7 agents.
    order = T.build_line(7, 4, seed=0)
    assert order[0] == [(0, 1), (2, 3), (4, 5)]
    assert order[1] == [(1, 0), (3, 2), (5, 4)]
    assert order[2] == [(1, 2), (3, 4), (5, 6)]
    assert order[3] == [(2, 1), (4, 3), (6, 5)]


def test_line_only_adjacent():
    order = T.build_line(10, 20, seed=0)
    for rnd in order:
        assert _is_matching(rnd, 10)
        for a, b in rnd:
            assert abs(a - b) == 1


def test_star_pairs_are_edges_and_matchings():
    N, R, k = 20, 60, 4
    edges = set(T.star_edges(N, k))
    edges |= {(b, a) for (a, b) in edges}
    order = T.build_star(N, R, branches=k, seed=0)
    assert len(order) == R
    covered = set()
    for rnd in order:
        assert _is_matching(rnd, N)
        for pair in rnd:
            assert pair in edges
            covered.add(tuple(sorted(pair)))
    # Every star edge should be exercised over enough rounds.
    undirected = {tuple(sorted(e)) for e in T.star_edges(N, k)}
    assert covered == undirected


def test_neighbors():
    adj = T.neighbors("graph", 5)
    assert adj[0] == {1, 2, 3, 4}
    adj = T.neighbors("line", 5)
    assert adj[0] == {1}
    assert adj[4] == {3}
    assert adj[2] == {1, 3}
    adj = T.neighbors("star", 7, star_branches=3)
    assert 0 in adj and len(adj[0]) == 3  # center connects to 3 arm heads
