from arcj.memory import Memory


def test_independent_memory():
    a = Memory(["correct fact"])
    b = Memory(["misleading fact"])
    a.add_history("a heard something")
    assert "a heard something" not in b.items()
    assert b.items() == ["misleading fact"]


def test_items_order_and_len():
    m = Memory(["k1", "k2"])
    m.add_history("h1")
    assert m.items() == ["k1", "k2", "h1"]
    assert len(m) == 3
    assert not m.is_empty()


def test_clone_is_deep():
    m = Memory(["k1"])
    m.add_history("h1")
    c = m.clone()
    c.add_history("h2")
    assert "h2" not in m.items()
    assert "h2" in c.items()


def test_empty_memory():
    m = Memory()
    assert m.is_empty()
    m.add_knowledge("")
    assert m.is_empty()  # empty strings are ignored
