from arcj import metrics as M


def test_current_asr():
    assert M.current_asr([[1, 0], [1, 1]]) == 0.75
    assert M.current_asr([[0, 0], [0, 0]]) == 0.0
    assert M.current_asr([]) == 0.0


def test_asr_max():
    assert M.asr_max({0: 0.1, 5: 0.4, 10: 0.3}) == 0.4
    assert M.asr_max([0.2, 0.9, 0.5]) == 0.9


def test_attack_speed_rate():
    series = {10: 0.1, 20: 0.55, 30: 0.8}
    assert M.attack_speed_rate(series, 50, 150) == "20"
    assert M.attack_speed_rate(series, 75, 150) == "30"
    assert M.attack_speed_rate(series, 90, 150) == "150+"


def test_misleading_rate():
    assert M.misleading_rate("E", "E") == 1
    assert M.misleading_rate("d", "D") == 1
    assert M.misleading_rate("A", "E") == 0
    assert M.misleading_rate(None, "E") == 0


def test_option_letter():
    assert M.option_letter("E.Flavor Wheels") == "E"
    assert M.option_letter("  c. lower") == "C"


def test_parse_choice():
    opts = ["A.Tasty Trails", "B.Bite Delight", "C.Rollin Spice",
            "D.Taco Town", "E.Flavor Wheels"]
    assert M.parse_choice("My choice is <E>", opts) == "E"
    assert M.parse_choice("My choice is D.", opts) == "D"
    assert M.parse_choice("I think it is Flavor Wheels for sure", opts) == "E"
    assert M.parse_choice("the answer is B", opts) == "B"
    assert M.parse_choice("no idea here", opts) is None
