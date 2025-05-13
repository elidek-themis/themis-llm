import gc
import logging

import torch

from lm_eval.api.model import TemplateLM

from themis.definitions.config import InterfaceConfig
from themis.definitions.registry import get_backend

__all__ = ["inference"]

logger = logging.getLogger(__name__)


class __Inference:
    backend: TemplateLM | None = None
    config: InterfaceConfig | None = None

    @classmethod
    def setup(cls, interface: InterfaceConfig) -> None:
        if cls.backend and cls.config == interface:  # is initialized
            logger.info("Using already initialized inference instance")
            logger.info(interface)
            return

        backend_class = get_backend(interface.name)
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
