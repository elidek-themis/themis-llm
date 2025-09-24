import json
import logging

from typing import TypeVar

import torch

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import get_model, register_model, register_aggregation

T = TypeVar("T", bound="TemplateLM")

logger = logging.getLogger(__name__)


@register_model("themis-singleton")
class ThemisLM(TemplateLM):
    """
    A wrapper for LM classes that persists a single instance across reruns.
    Should be instantiated with `create_from_arg_obj` or `create_from_arg_string`.

    The wrapped model must be a field of model_args.
    .: e.g.
        model_args = {
            "model": "hf",
            "model_args": {
                "pretrained": "facebook/opt-125m",
            }
        }
    """

    _singleton_instance = None
    _singleton_key = None

    def __init__(self, wrapped):
        self._wrapped = wrapped

    @classmethod
    def _make_key(cls, model: str, model_args: dict) -> str:
        """Creates a hashable key."""
        return json.dumps(
            {
                "model": model,
                "args": model_args,
            },
            sort_keys=True,
        )

    @classmethod
    def _cleanup(cls):
        """Clean up resources, if any, before replacing singleton."""
        if cls._singleton_instance is not None:
            logger.info("Cleaning up singleton instance.")

            del cls._singleton_instance
            cls._singleton_instance = None
            cls._singleton_key = None

            import gc  # noqa: PLC0415

            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Called torch.cuda.empty_cache()")

    @classmethod
    def _create_or_reuse(cls, model: str, model_args: dict, extra_args: dict) -> T:
        key = cls._make_key(model, model_args)

        if cls._singleton_instance is not None and cls._singleton_key == key:
            logger.info(f"Reusing cached model instance: {model}")
            return cls._singleton_instance

        cls._cleanup()
        model_cls = get_model(model)

        logger.info("Populating singleton instance")
        logger.info(f"Backend: {model_cls.__name__}")
        logger.info(f"model_args {model_args}")

        wrapped = model_cls.create_from_arg_obj(model_args, extra_args)

        instance = cls(wrapped)
        cls._singleton_instance = instance
        cls._singleton_key = key
        return instance

    @classmethod
    def create_from_arg_obj(cls: type[T], arg_dict: dict, additional_config: dict | None = None) -> T:
        additional_config = {k: v for k, v in (additional_config or {}).items() if v is not None}

        model = arg_dict["model"]
        model_args = arg_dict.get("model_args", {})
        return cls._create_or_reuse(model, model_args, additional_config)

    @classmethod
    def create_from_arg_string(cls: type[T], arg_string: str, additional_config: dict | None = None) -> T:
        additional_config = {k: v for k, v in (additional_config or {}).items() if v is not None}

        arg_dict = utils.simple_parse_args_string(arg_string)
        model = arg_dict["model"]
        model_args = arg_dict.get("model_args", {})
        return cls._create_or_reuse(model, model_args, additional_config)

    def _loglikelihood_tokens(self, requests):
        return self._wrapped._loglikelihood_tokens(requests)

    # Delegate to the wrapped model
    @property
    def eot_token_id(self):
        return self._wrapped.eot_token_id

    @property
    def tokenizer_name(self):
        return self._wrapped.tokenizer_name

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self._wrapped.apply_chat_template(chat_history, add_generation_prompt)

    def generate_until(self, requests):
        return self._wrapped.generate_until(requests)

    def loglikelihood_rolling(self, requests):
        return self._wrapped.loglikelihood_rolling(requests)

    def _loglikelihood_tokens(self, requests, **kwargs):
        return self._wrapped._loglikelihood_tokens(requests, **kwargs)

    def tok_encode(self, string: str):
        return self._wrapped.tok_encode(string)

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)


# custom metrics/aggregations
@register_aggregation("pass")
def pass_agg(arr):  # type: ignore[no-untyped-def]
    return arr
