from comfy_api.latest import _io, io

MAX_BRANCHES = 16


def _unwrap(entry):
    """check_lazy_status sees (value, flat_key); execute sees the bare value."""
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], str):
        return entry
    return entry, None


class LayaSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("laya_switch")
        return io.Schema(
            node_id="LayaSwitch",
            display_name="Laya Switch (route)",
            search_aliases=["switch", "route", "multi switch", "select", "branch", "index"],
            category="utilities/logic",
            is_experimental=True,
            inputs=[
                io.Int.Input("index", default=0, min=0, max=MAX_BRANCHES - 1,
                             tooltip="Which branch to pass through. Wire Laya Choose's `index` output here; it follows the order of the Laya Option sockets."),
                _io.Autogrow.Input("branches", tooltip="Connect one branch per route. Only the selected branch is evaluated - the graph behind the others never runs.",
                                   template=_io.Autogrow.TemplatePrefix(
                                       input=io.MatchType.Input("branch", template=template, lazy=True),
                                       prefix="branch", min=2, max=MAX_BRANCHES)),
            ],
            outputs=[io.MatchType.Output(template=template, display_name="output")],
        )

    @classmethod
    def check_lazy_status(cls, index, branches):
        value, key = _unwrap(branches.get(f"branch{index}"))
        if value is None and key is not None:
            return [key]

    @classmethod
    def execute(cls, index, branches) -> io.NodeOutput:
        name = f"branch{index}"
        if name not in branches:
            raise ValueError(
                f"Laya Switch got index {index}, but only {len(branches)} branches are connected. "
                f"Connect another branch, or check the Laya Choose it is wired to."
            )
        return io.NodeOutput(_unwrap(branches[name])[0])
