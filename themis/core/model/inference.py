import gc
import logging
from typing import Any, Type

import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.models.vllm_causallms import VLLM

from themis.definitions.config import InterfaceConfig

__all__ = ["inference", "backends"]

logger = logging.getLogger(__name__)

BACKEND_REGISTRY = {
    "lm_eval_hf": HFLM,
}

backends = list(BACKEND_REGISTRY.keys())


def register_backend(name):
    def decorate(cls):
        BACKEND_REGISTRY[name] = cls
        return cls

    return decorate


def get_model(model_name):
    try:
        return BACKEND_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Supported backend names: {', '.join(BACKEND_REGISTRY.keys())}")


class __Inference:

    backend: Type[Any] = None
    config: InterfaceConfig = None

    @classmethod
    def setup(cls, interface: InterfaceConfig) -> None:
        if cls.backend and cls.config == interface:  # is initialized
            logger.info("Using already initialized inference instance")
            logger.info(interface)
            return

        backend_class = get_model(interface.name)
        cls.backend = backend_class(**interface.args)
        cls.config = interface

    @classmethod
    def collect(cls):
        del cls.backend
        cls.backend = None
        cls.config = None


@register_backend("lm_eval_vllm")
class ThVLLM(VLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_chat = True if self.tokenizer.chat_template else False

    def apply_chat_template(self, chat_history):
        return self.tokenizer.apply_chat_template(
            conversation=chat_history, tokenize=False, continue_final_message=False
        )

    def __del__(self):
        logger.info("Deleting VLLM instance")
        super().__del__()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("VLLM instance deleted")


inference = __Inference()
