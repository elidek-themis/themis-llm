import pprint

from themis.utils.tools import load_yaml_config


# https://www.hackerearth.com/practice/notes/samarthbhargav/a-design-pattern-for-configuration-management-in-python/
class Config:
    def __init__(self, **kwargs):
        self.model = kwargs.pop("model")
        self.interface = kwargs.pop("interface", "vllm")
        self.task = kwargs.pop("task")

        self.temperature = kwargs["model_args"].pop("temperature", 1.0)
        self.top_p = kwargs["model_args"].pop("top_p", 1.0)
        self.top_k = kwargs["model_args"].pop("top_k", -1)
        self.max_tokens = kwargs["model_args"].pop("max_tokens", 256)
        self.seed = kwargs["model_args"].pop("seed", 2024)

        self.limit = kwargs["eval_args"].pop("limit", None)

    def __eq__(self, other):
        if isinstance(other, Config):
            return self.__dict__ == other.__dict__
        return False

    def __repr__(self):
        return pprint.pformat(self.__dict__)


def load_config(config_path: str) -> Config:
    config = load_yaml_config(config_path=config_path)
    return Config(**config)
