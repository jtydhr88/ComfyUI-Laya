"""Schema shape. These run without any checkpoint on disk."""


def test_every_node_is_registered(pack):
    import asyncio
    names = [n.define_schema().node_id for n in asyncio.run(pack.LayaExtension().get_node_list())]
    assert names == ["LayaLoader", "LayaOption", "LayaChoose", "LayaSwitch", "LayaAsk", "LayaRate"]


def test_choose_takes_option_nodes_through_an_autogrow(pack):
    from comfy_api.latest import _io
    from ComfyUI_Laya.nodes import LayaChoose

    options = next(i for i in LayaChoose.define_schema().inputs if i.id == "options")
    assert isinstance(options, _io.Autogrow.Input)
    assert (options.template.prefix, options.template.min) == ("option", 2)
    assert options.template.input.io_type == "LAYA_OPTION"


def test_switch_branches_are_lazy(pack):
    """Without lazy every branch executes, which makes the node worse than no router at all."""
    from comfy_api.latest import _io
    from ComfyUI_Laya.switch import LayaSwitch

    branches = next(i for i in LayaSwitch.define_schema().inputs if i.id == "branches")
    assert isinstance(branches, _io.Autogrow.Input)
    assert branches.template.input.lazy is True


def test_switch_output_matches_the_branch_type(pack):
    from ComfyUI_Laya.switch import LayaSwitch

    schema = LayaSwitch.define_schema()
    branches = next(i for i in schema.inputs if i.id == "branches")
    assert branches.template.input.template is schema.outputs[0].template


def test_rate_levels_expand_per_count(pack):
    from ComfyUI_Laya.nodes import LayaRate

    levels = next(i for i in LayaRate.define_schema().inputs if i.id == "levels")
    keys = [o.key for o in levels.options]
    assert keys == [f"{n} levels" for n in range(2, 8)]
    three = next(o for o in levels.options if o.key == "3 levels")
    assert [i.id for i in three.inputs] == ["level_0", "level_1", "level_2"]


def test_a_new_rate_node_runs_without_being_filled_in(pack):
    """DynamicCombo has no default key, so option zero is what a fresh node ships with."""
    from ComfyUI_Laya.nodes import LayaRate

    levels = next(i for i in LayaRate.define_schema().inputs if i.id == "levels")
    assert all(i.default for i in levels.options[0].inputs)
