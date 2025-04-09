# from langchain_community.llms import VLLM as LCVLLM
from lm_eval.api.model import TemplateLM
from lm_eval.models.vllm_causallms import VLLM

from themis.utils.config import Config, GenerationConfig, MultipleChoiceConfig


def get_model(config: Config) -> TemplateLM:

    if config.interface == "huggingface":
        pass

    if config.interface == "vllm":
        if isinstance(config, MultipleChoiceConfig):
            return VLLM(pretrained=config.model, seed=config.seed)
        if isinstance(config, GenerationConfig):
            return VLLM(pretrained=config.model, max_gen_toks=config.max_tokens, seed=config.seed)

    # if config.interface == "langchain_vllm":
    #     return LCVLLM(
    #         model=config.model,
    #         max_new_tokens=config.max_tokens,
    #         top_k=config.top_k,
    #         top_p=config.top_p,
    #         temperature=config.temperature
    #     )

    if config.interface == "api":
        pass

    raise NotImplementedError(f"{config.interface} not implemented")


def get_all_seed(config: Config) -> dict:
    """Seeds for lm_eval.simple_evaluate"""

    return {
        "random_seed": config.seed,
        "numpy_random_seed": config.seed,
        "torch_random_seed": config.seed,
        "fewshot_random_seed": config.seed,
    }


def get_generation_args(config: Config) -> str:
    """
    String of comma separated argument assignments
    Used by lm_eval.simple_evaluate for generation tasks
    """

    gen_kwargs = {
        "max_gen_tokens": config.max_tokens,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "temperature": config.temperature,
    }

    return ",".join([f"{arg}={value}" for arg, value in gen_kwargs.items()])
