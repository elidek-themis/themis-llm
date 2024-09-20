# from langchain_community.llms import VLLM as LCVLLM
from lm_eval.api.model import TemplateLM
from lm_eval.models.vllm_causallms import VLLM

from themis.utils.config import Config


def get_model(cfg: Config) -> TemplateLM:

    if cfg.interface == "huggingface":
        pass

    if cfg.interface == "vllm":
        return VLLM(pretrained=cfg.model, max_gen_toks=cfg.max_tokens, seed=cfg.seed, trust_remote_code=True)

    # if cfg.interface == "langchain_vllm":
    #     return LCVLLM(
    #         model=cfg.model,
    #         max_new_tokens=cfg.max_tokens,
    #         top_k=cfg.top_k,
    #         top_p=cfg.top_p,
    #         temperature=cfg.temperature
    #     )

    if cfg.interface == "api":
        pass

    raise NotImplementedError(f"{cfg.interface} not implemented")


def get_all_seed(cfg: Config) -> dict:
    """Seeds for lm_eval.simple_evaluate"""

    return {
        "random_seed": cfg.seed,
        "numpy_random_seed": cfg.seed,
        "torch_random_seed": cfg.seed,
        "fewshot_random_seed": cfg.seed,
    }


def get_generation_args(cfg: Config) -> str:
    """
    String of comma separated argument assignments
    Used by lm_eval.simple_evaluate for generation tasks
    """

    gen_kwargs = {
        "top_k": cfg.top_k,
        "top_p": cfg.top_p,
        "temperature": cfg.temperature,
    }

    return ",".join([f"{arg}={value}" for arg, value in gen_kwargs.items()])
