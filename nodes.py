import json

from comfy_api.latest import _io, io

from .model import ALIASES, list_checkpoints, load_agent

LayaModelType = io.Custom("LAYA_MODEL")
LayaOptionType = io.Custom("LAYA_OPTION")

STATE_TIP = "The text Laya judges: a prompt, a caption, a user message, or any JSON blob."

# DynamicCombo has no default key, so the first option is what a new node ships with: keep it filled.
LEVEL_DEFAULTS = {2: ["low", "high"], 3: ["minimal", "moderate", "very high"]}


def _ask(model, state, question):
    return model.predict(state, {"q": question})["answers"]["q"]


def _ordered(autogrow, prefix="option"):
    """Autogrow hands back a name-keyed dict; socket order must be restored from the suffix."""
    def ordinal(name):
        tail = name[len(prefix):]
        return int(tail) if tail.isdigit() else 0
    return [autogrow[k] for k in sorted(autogrow, key=ordinal) if autogrow.get(k)]


def _level_option(n):
    inputs = []
    defaults = LEVEL_DEFAULTS.get(n, [])
    for i in range(n):
        default = defaults[i] if i < len(defaults) else ""
        inputs.append(io.String.Input(f"level_{i}", default=default,
                                      tooltip="Describe this level. Lowest first; score is the expected value over them."))
    return io.DynamicCombo.Option(key=f"{n} levels", inputs=inputs)


def _collect(values, prefix):
    out = []
    i = 0
    while f"{prefix}{i}" in values:
        out.append(values[f"{prefix}{i}"].strip())
        i += 1
    return out


class LayaLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LayaLoader",
            display_name="Load Laya Decision Model",
            category="utilities/logic",
            inputs=[
                io.Combo.Input("model_name", options=list_checkpoints(),
                               tooltip="Local dirs under models/laya, then the Hugging Face checkpoints (downloaded on first use)."),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"], default="auto",
                               tooltip="cuda holds ~1.7 GB for as long as ComfyUI runs; cpu frees that at the cost of latency."),
            ],
            outputs=[LayaModelType.Output(display_name="laya")],
        )

    @classmethod
    def validate_inputs(cls, model_name, device):
        if model_name in ALIASES or model_name in list_checkpoints():
            return True
        return f"Unknown Laya checkpoint: {model_name}"

    @classmethod
    def execute(cls, model_name, device) -> io.NodeOutput:
        return io.NodeOutput(load_agent(model_name, device))


class LayaAsk(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LayaAsk",
            display_name="Laya Ask (yes/no)",
            search_aliases=["gate", "guard", "boolean", "decide"],
            category="utilities/logic",
            inputs=[
                LayaModelType.Input("laya"),
                io.String.Input("state", multiline=True, tooltip=STATE_TIP),
                io.String.Input("question", multiline=True,
                                default="Does the prompt describe a moving shot rather than a still image?",
                                tooltip="A statement Laya answers true or false."),
                io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="probability >= threshold becomes true. The public checkpoints skew heavily towards false on unseen domains, so calibrate this against your own examples rather than trusting 0.5 - or use Laya Choose, whose argmax needs no threshold."),
            ],
            outputs=[
                io.Boolean.Output(display_name="answer"),
                io.Float.Output(display_name="probability"),
                io.Float.Output(display_name="confidence"),
            ],
        )

    @classmethod
    def execute(cls, laya, state, question, threshold) -> io.NodeOutput:
        a = _ask(laya, state, {"type": "noul", "instructions": question})
        return io.NodeOutput(a["noul"] >= threshold, a["noul"], a["confidence"])


class LayaOption(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LayaOption",
            display_name="Laya Option",
            search_aliases=["option", "route", "label", "criteria", "rubric"],
            category="utilities/logic",
            inputs=[
                io.String.Input("label", default="",
                                tooltip="Short name. This is what the `choice` output returns when Laya picks this option."),
                io.String.Input("criteria", multiline=True, default="",
                                tooltip="The rubric Laya actually scores. Describe what belongs in this option, not just its name."),
            ],
            outputs=[LayaOptionType.Output(display_name="option")],
        )

    @classmethod
    def execute(cls, label, criteria) -> io.NodeOutput:
        return io.NodeOutput({"label": label.strip(), "criteria": criteria.strip()})


class LayaChoose(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LayaChoose",
            display_name="Laya Choose (route)",
            search_aliases=["route", "router", "classify", "switch"],
            category="utilities/logic",
            inputs=[
                LayaModelType.Input("laya"),
                io.String.Input("state", multiline=True, tooltip=STATE_TIP),
                io.String.Input("instructions", multiline=True,
                                default="Which kind of render does this prompt call for?",
                                tooltip="What the options are being chosen between."),
                _io.Autogrow.Input("options", tooltip="Connect one Laya Option node per route. A new socket appears as you fill the last one.",
                                   template=_io.Autogrow.TemplatePrefix(
                                       input=LayaOptionType.Input("option"), prefix="option", min=2, max=16)),
            ],
            outputs=[
                io.String.Output(display_name="choice"),
                io.Int.Output(display_name="index"),
                io.Float.Output(display_name="confidence"),
                io.String.Output(display_name="probabilities"),
            ],
        )

    @classmethod
    def execute(cls, laya, state, instructions, options) -> io.NodeOutput:
        criteria = {}
        for opt in _ordered(options):
            label = opt["label"]
            if label:
                criteria[label] = opt["criteria"] or None
        if len(criteria) < 2:
            raise ValueError("Laya Choose needs at least two Laya Option nodes with a label filled in.")
        a = _ask(laya, state, {"type": "choice", "instructions": instructions, "criteria": criteria})
        index = list(criteria).index(a["choice"])
        return io.NodeOutput(a["choice"], index, a["confidence"], json.dumps(a["probabilities"]))


class LayaRate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LayaRate",
            display_name="Laya Rate (score)",
            search_aliases=["score", "rank", "grade", "urgency"],
            category="utilities/logic",
            inputs=[
                LayaModelType.Input("laya"),
                io.String.Input("state", multiline=True, tooltip=STATE_TIP),
                io.String.Input("instructions", multiline=True,
                                default="How much fine detail does this prompt demand?",
                                tooltip="What is being rated."),
                io.DynamicCombo.Input("levels", display_name="levels",
                                      options=[_level_option(n) for n in range(2, 8)],
                                      tooltip="How many levels to rate across, lowest first."),
            ],
            outputs=[
                io.Float.Output(display_name="score"),
                io.Int.Output(display_name="level"),
                io.Float.Output(display_name="confidence"),
            ],
        )

    @classmethod
    def execute(cls, laya, state, instructions, levels) -> io.NodeOutput:
        criteria = [lv for lv in _collect(levels, "level_") if lv]
        if len(criteria) < 2:
            raise ValueError("Laya Rate needs at least two levels filled in.")
        a = _ask(laya, state, {"type": "score", "instructions": instructions, "criteria": criteria})
        return io.NodeOutput(a["score"], round(a["score"]), a["confidence"])
