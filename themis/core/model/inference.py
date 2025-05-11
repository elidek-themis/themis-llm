import gc
import logging

from typing import Any

import torch

from lm_eval.models.huggingface import HFLM
from lm_eval.models.vllm_causallms import VLLM

from themis.definitions.config import InterfaceConfig

__all__ = ["inference", "backends"]

logger = logging.getLogger(__name__)

BACKEND_REGISTRY = {
    "lm_eval_hf": HFLM,
    "lm_eval_vllm": VLLM,
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
    except KeyError as e:
        raise ValueError(f"Supported backend names: {', '.join(BACKEND_REGISTRY.keys())}") from e


class __Inference:
    backend: type[Any] = None
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
    def collect(cls) -> None:
        logger.info("Deleting inference instance")
        cls.backend = None
        cls.config = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Inference instance deleted")


inference = __Inference()
