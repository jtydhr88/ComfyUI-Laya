English | [简体中文](README.zh-CN.md)

# ComfyUI-Laya

Route a ComfyUI workflow with [Laya](https://huggingface.co/convaiinnovations/laya), a 421M
System-1 decision model — so the expensive branch is the only one that runs.

Laya does not generate anything. You hand it a piece of text and typed questions; it returns
calibrated probabilities in one forward pass, in ~15 ms on GPU or ~75 ms on CPU.

## Why this exists

ComfyUI has had the pieces of a router for a while — `Generate Text`, the API LLM nodes,
`If/Else Switch` — but not a router anybody could actually use on a consumer GPU. Once a video
model is resident there is no VRAM left to host a 7B model whose whole job is to answer *"is this
prompt asking for a video or a still?"*. So that decision stayed manual: you picked the workflow,
you set the steps, you dragged the wire.

Laya fits where an LLM never could. The whole checkpoint is 808 MB on disk, and on `device: cpu` it
costs **zero VRAM** — it can decide while the GPU is busy sampling. That, not the 15 ms, is the
point.

**This does not speed up sampling.** Nothing here touches the UNet, the sampler or the VAE. The
saving comes from the branch that never runs.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/jtydhr88/ComfyUI-Laya.git
pip install laya
```

Checkpoints download on first use into `ComfyUI/models/laya/`. Drop your own fine-tune in that
directory and it appears in the loader:

| checkpoint | backbone | params | context | notes |
|---|---|---|---|---|
| `laya-english` | ModernBERT-large | 421M | 512 | English |
| `laya-multilingual` | mmBERT-base | 322M | 1024 | 100+ languages |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 | fine-tuned on Laya's own decision suite |

## Nodes

All under `utilities/logic`.

| node | output |
|---|---|
| **Load Laya Decision Model** | `LAYA_MODEL`. `device` auto / cuda / cpu |
| **Laya Option** | one route: a `label` and the `criteria` rubric Laya scores |
| **Laya Choose (route)** | `choice`, `index`, `confidence`, `probabilities` (JSON) |
| **Laya Switch (route)** | passes through branch `index`; **only that branch is evaluated** |
| **Laya Ask (yes/no)** | `answer` BOOLEAN, `probability`, `confidence` |
| **Laya Rate (score)** | `score` (expected value over the levels), `level`, `confidence` |

## Wiring

```
Laya Option ×N ──► Laya Choose ──index──► Laya Switch ──► the selected pipeline
                              └─choice──► (a label, for previewing or comparing)
```

![Four routes wired through Laya Choose into Laya Switch](docs/img.png)

Above: the four routes as `Laya Option` nodes, a drone prompt landing on `t2v` at 0.95, and
`Laya Switch` passing that branch through. The other three pipelines were never evaluated. This is
`example_workflows/laya-router-demo.json`.

`Laya Choose`'s `options` and `Laya Switch`'s `branches` are both autogrow inputs: a fresh socket
appears each time you fill the last one, up to 16. `index` follows socket order, so option *n* and
branch *n* line up.

Two-way gating also works with core nodes alone: `choice` → `Compare Text` (Equal) →
`If/Else Switch`.

## Writing options

The **criteria** text is what Laya actually scores; the label is just what comes back out. A bare
label with no rubric is much weaker. Compare, on the same prompt:

```
video: the prompt asks for motion, a camera move, or a clip   → 0.96
video                                                          → far less separation
```

Read `probabilities`, not `confidence` — the public checkpoints ship temperatures outside the sane
range and the library warns about it on load. The signal worth acting on is the gap between the top
two.

Prefer `Laya Choose` over `Laya Ask`: the public checkpoints skew heavily towards *false* on domains
they were not trained for, so a 0.5 threshold is not meaningful until you calibrate it on your own
examples. `Laya Choose`'s argmax needs no threshold.

For non-English prompts use `laya-multilingual`. The English checkpoint on Chinese input returns
near-uniform probabilities — it is not making a decision. Criteria may stay in English even when the
prompts are not; the multilingual checkpoint aligns across languages.

## Requires

`laya>=0.3.5` (which pulls `torch>=2.0.0`, `transformers>=4.48.0`, `huggingface_hub>=0.20.0`).

`Laya Switch` needs a ComfyUI that includes
[Comfy-Org/ComfyUI#16377](https://github.com/Comfy-Org/ComfyUI/pull/16377). Before that fix,
`TopologicalSort.get_input_info` resolves against the **unexpanded** `INPUT_TYPES()`, so an autogrow
input name like `branches.branch0` matches no spec, reads as non-lazy, and **every branch executes**
— which makes a router slower than no router at all. Measured here on a 4-branch graph: before the
fix all four branch sources ran; after it, exactly one did.

## Tests

```
pytest tests
```

Schema tests run anywhere. The execute-level tests load a checkpoint on CPU and skip if none is
available; `LAYA_TEST_CHECKPOINT` picks which one.

## Licence

GPL-3.0, see `LICENSE`. Laya's own weights are Apache-2.0 and are not redistributed here.
