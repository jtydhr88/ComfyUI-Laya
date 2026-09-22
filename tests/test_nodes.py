"""Execute-level tests. They load a real checkpoint on CPU and skip if none is available."""
import pytest
import torch


def _options(*pairs):
    from ComfyUI_Laya.nodes import LayaOption
    return {f"option{i}": LayaOption.execute(l, c).result[0] for i, (l, c) in enumerate(pairs)}


ROUTES = (
    ("t2v", "a moving shot: camera movement, action over time, or an explicit clip or animation"),
    ("t2i", "a single still photograph or illustration with no motion"),
    ("upscale", "an existing image that should be enlarged, restored, or sharpened"),
    ("inpaint", "an existing image where one region should be replaced or removed"),
)
INSTRUCTIONS = "Which generation pipeline should this prompt be routed to?"


def test_option_node_emits_label_and_criteria(pack):
    from ComfyUI_Laya.nodes import LayaOption
    assert LayaOption.execute(" t2v ", " a moving shot ").result[0] == {
        "label": "t2v", "criteria": "a moving shot"}


def test_choose_reports_every_option(pack, agent):
    from ComfyUI_Laya.nodes import LayaChoose
    with torch.inference_mode():
        choice, index, confidence, probs = LayaChoose.execute(
            agent, "a neon alley, slow dolly in", INSTRUCTIONS, _options(*ROUTES)).result
    import json
    probs = json.loads(probs)
    assert set(probs) == {label for label, _ in ROUTES}
    assert probs[choice] == max(probs.values())
    assert [label for label, _ in ROUTES][index] == choice
    assert 0.0 <= confidence <= 1.0


def test_choose_index_follows_socket_order_not_dict_order(pack, agent):
    """Autogrow hands over an unordered dict; a wrong sort silently routes to the wrong branch."""
    from ComfyUI_Laya.nodes import LayaChoose
    forward = _options(*ROUTES)
    scrambled = dict(reversed(list(forward.items())))
    with torch.inference_mode():
        a = LayaChoose.execute(agent, "make this scan 4x bigger", INSTRUCTIONS, forward).result
        b = LayaChoose.execute(agent, "make this scan 4x bigger", INSTRUCTIONS, scrambled).result
    assert a[0] == b[0] and a[1] == b[1]


def test_choose_skips_options_with_no_label(pack, agent):
    from ComfyUI_Laya.nodes import LayaChoose
    opts = _options(*ROUTES) | {"option4": {"label": "", "criteria": "never selected"}}
    import json
    with torch.inference_mode():
        probs = json.loads(LayaChoose.execute(agent, "a still photo", INSTRUCTIONS, opts).result[3])
    assert set(probs) == {label for label, _ in ROUTES}


def test_choose_needs_two_labelled_options(pack, agent):
    from ComfyUI_Laya.nodes import LayaChoose
    with pytest.raises(ValueError):
        LayaChoose.execute(agent, "x", INSTRUCTIONS, _options(("only", "one")))


def test_rate_returns_an_expected_value_over_the_levels(pack, agent):
    from ComfyUI_Laya.nodes import LayaRate
    levels = {"levels": "3 levels", "level_0": "minimal", "level_1": "moderate", "level_2": "very high"}
    with torch.inference_mode():
        score, level, _ = LayaRate.execute(agent, "a plain grey square", "How detailed?", levels).result
    assert 0.0 <= score <= 2.0
    assert level == round(score)


def test_rate_needs_two_levels(pack, agent):
    from ComfyUI_Laya.nodes import LayaRate
    with pytest.raises(ValueError):
        LayaRate.execute(agent, "x", "y", {"levels": "2 levels", "level_0": "only", "level_1": ""})


def test_ask_probability_and_threshold_agree(pack, agent):
    from ComfyUI_Laya.nodes import LayaAsk
    with torch.inference_mode():
        answer, probability, _ = LayaAsk.execute(
            agent, "a neon alley, slow dolly in", "Does the prompt describe motion?", 0.5).result
    assert answer == (probability >= 0.5)


def test_switch_passes_the_indexed_branch(pack):
    from ComfyUI_Laya.switch import LayaSwitch
    branches = {f"branch{i}": v for i, v in enumerate(["A", "B", "C"])}
    assert LayaSwitch.execute(1, branches).result[0] == "B"


def test_switch_asks_only_for_the_selected_branch(pack):
    from ComfyUI_Laya.switch import LayaSwitch
    branches = {"branch0": ("A", "branches.branch0"), "branch1": (None, "branches.branch1")}
    assert LayaSwitch.check_lazy_status(1, branches) == ["branches.branch1"]
    assert LayaSwitch.check_lazy_status(0, branches) is None


def test_switch_reports_an_unconnected_index(pack):
    from ComfyUI_Laya.switch import LayaSwitch
    with pytest.raises(ValueError, match="index 3"):
        LayaSwitch.execute(3, {"branch0": "A", "branch1": "B"})
