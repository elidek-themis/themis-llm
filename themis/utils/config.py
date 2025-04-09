import os.path as osp
import pprint

from yaml import dump

from themis.utils.tools import load_yaml_config


class Config:

    _DEFAULT_INTERFACE = "vllm"
    _DEFAULT_LIMIT = None
    _DEFAULT_SEED = 2024
    _DEFAULT_PROMPT_TEMPLATE = "empty_template"

    def __init__(self, **config):
        self.model = config.pop("model")
        self.interface = config.pop("interface", self._DEFAULT_INTERFACE)
        self.task = config.pop("task")
        self.prompt_template = config.pop("prompt_template", self._DEFAULT_PROMPT_TEMPLATE)
        self.limit = config["eval_args"].pop("limit", self._DEFAULT_LIMIT)
        self.seed = config["model_args"].pop("seed", self._DEFAULT_SEED)

    def signature(self):
        return {
            "model": self.model,
            "task": self.task,
            "prompt_template": self.prompt_template,
            "seed": self.seed,
            "limit": self.limit,
        }

    def dump(self, path):
        save_path = osp.join(path, "config.yaml")

        with open(save_path, "w") as yaml_fh:
            dump(self.signature(), stream=yaml_fh)

    def __eq__(self, config_description):
        return self.signature() == config_description

    def __repr__(self):
        return pprint.pformat(self.__dict__)


class MultipleChoiceConfig(Config):
    def __init__(self, **config) -> None:
        super().__init__(**config)


class GenerationConfig(Config):

    _DEFAULT_TEMP = 0.0
    _DEFAULT_TOP_P = 1.0
    _DEFAULT_TOP_K = -1
    _DEFAULT_MAX_TOKENS = 128

    def __init__(self, **config) -> None:
        super().__init__(**config)

        self.temperature = config["model_args"].pop("temperature", self._DEFAULT_TEMP)
        self.top_p = config["model_args"].pop("top_p", self._DEFAULT_TOP_P)
        self.top_k = config["model_args"].pop("top_k", self._DEFAULT_TOP_K)
        self.max_tokens = config["model_args"].pop("max_tokens", self._DEFAULT_MAX_TOKENS)

    def signature(self):
        sig = {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k, "max_tokens": self.max_tokens}
        return dict(super().signature(), **sig)


def config_factory(config_path: str) -> Config:
    config = load_yaml_config(config_path=config_path)

    if config["task"] in ["crows_pairs_mc"]:
        return MultipleChoiceConfig(**config)

    if config["task"] in ["crows_pairs_gen"]:
        return GenerationConfig(**config)
