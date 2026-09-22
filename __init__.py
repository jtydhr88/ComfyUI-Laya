from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .model import register_model_folder
from .nodes import LayaAsk, LayaChoose, LayaLoader, LayaOption, LayaRate
from .switch import LayaSwitch

register_model_folder()


class LayaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LayaLoader, LayaOption, LayaChoose, LayaSwitch, LayaAsk, LayaRate]


async def comfy_entrypoint() -> LayaExtension:
    return LayaExtension()
