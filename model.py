import os

import folder_paths

# Keyed by download directory, so a saved workflow's value survives the checkpoint landing on disk.
HF_CHECKPOINTS = {
    "laya-english": ("convaiinnovations/laya", None),
    "laya-multilingual": ("convaiinnovations/laya", "multilingual"),
    "laya-typed-decisions": ("convaiinnovations/laya", "typed-decisions"),
}

ALIASES = {
    "convaiinnovations/laya": "laya-english",
    "convaiinnovations/laya (multilingual)": "laya-multilingual",
    "convaiinnovations/laya (typed-decisions)": "laya-typed-decisions",
}

_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
_agents = {}


def register_model_folder():
    folder_paths.add_model_folder_path("laya", os.path.join(folder_paths.models_dir, "laya"))


def _model_dir(path):
    """A checkpoint dir, or the subdir holding it: snapshot_download keeps the repo's subfolder prefix."""
    if os.path.isfile(os.path.join(path, "rl_agent_config.json")):
        return path
    for entry in sorted(os.listdir(path)):
        inner = os.path.join(path, entry)
        if os.path.isfile(os.path.join(inner, "rl_agent_config.json")):
            return inner
    return None


def list_checkpoints():
    local = []
    for root in folder_paths.get_folder_paths("laya"):
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            path = os.path.join(root, entry)
            if entry not in local and os.path.isdir(path) and _model_dir(path):
                local.append(entry)
    return local + [k for k in HF_CHECKPOINTS if k not in local]


def resolve(name):
    name = ALIASES.get(name, name)
    for root in folder_paths.get_folder_paths("laya"):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            found = _model_dir(path)
            if found:
                return found
    if name not in HF_CHECKPOINTS:
        raise FileNotFoundError(
            f"Laya checkpoint {name!r} not found under models/laya. "
            f"Put a checkpoint directory there or pick one of: {', '.join(HF_CHECKPOINTS)}"
        )
    from huggingface_hub import snapshot_download

    repo, subfolder = HF_CHECKPOINTS[name]
    prefix = f"{subfolder}/" if subfolder else ""
    root = folder_paths.get_folder_paths("laya")[0]
    # local_dir keeps this off the symlinked HF cache, which needs privileges Windows withholds
    path = snapshot_download(
        repo,
        allow_patterns=[prefix + f for f in _FILES],
        local_dir=os.path.join(root, name),
    )
    found = _model_dir(path)
    if not found:
        raise FileNotFoundError(f"Downloaded {name!r} to {path} but found no rl_agent_config.json")
    return found


def load_agent(name, device):
    try:
        import laya
    except ImportError as e:
        raise ImportError("ComfyUI-Laya needs the 'laya' package: pip install laya") from e

    if device == "auto":
        import comfy.model_management
        device = str(comfy.model_management.get_torch_device())

    key = (name, device)
    if key not in _agents:
        _agents[key] = laya.load(resolve(name), device=device)
    return _agents[key]
