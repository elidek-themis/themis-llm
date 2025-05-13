from lm_eval.models.huggingface import HFLM
from lm_eval.models.vllm_causallms import VLLM

EVAL_REGISTRY = {}


def register_eval(name):
    def decorate(cls):
        EVAL_REGISTRY[name] = cls
        return cls

    return decorate


def get_eval(eval_name):
    """Eval factory function"""
    try:
        return EVAL_REGISTRY[eval_name]
    except KeyError as e:
        raise NotImplementedError(f"Evaluation {eval_name} not found") from e


BACKEND_REGISTRY = {
    "lm_eval_hf": HFLM,
    "lm_eval_vllm": VLLM,
}


def register_backend(name):
    def decorate(cls):
        BACKEND_REGISTRY[name] = cls
        return cls

    return decorate


def get_backend(model_name):
    """Backend factory function"""
    try:
        return BACKEND_REGISTRY[model_name]
    except KeyError as e:
        raise ValueError(f"Supported backend names: {', '.join(BACKEND_REGISTRY.keys())}") from e
