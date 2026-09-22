import importlib.util
import os
import sys

import pytest

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_DIR = os.path.dirname(os.path.dirname(PACK_DIR))

if COMFY_DIR not in sys.path:
    sys.path.insert(0, COMFY_DIR)


def _import_pack():
    """Import the pack the way ComfyUI does. Its directory name is not a valid module name, and
    pytest reaches the __init__.py first if this is left until fixture time."""
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_Laya", os.path.join(PACK_DIR, "__init__.py"),
        submodule_search_locations=[PACK_DIR])
    module = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_Laya"] = module
    spec.loader.exec_module(module)
    return module


PACK = _import_pack()


@pytest.fixture(scope="session")
def pack():
    return PACK


@pytest.fixture(scope="session")
def agent(pack):
    from ComfyUI_Laya.nodes import LayaLoader
    name = os.environ.get("LAYA_TEST_CHECKPOINT", "laya-english")
    try:
        return LayaLoader.execute(name, "cpu").result[0]
    except Exception as e:
        pytest.skip(f"no usable Laya checkpoint ({name}): {e}")
