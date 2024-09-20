"""
.env files are generally used to store information related to the particular deployment environment,
while config files might be used to store data particular to the application as a whole.
"""

import json
import logging
import pprint

import yaml


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
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    config = Config(**config)

    return config


def load_log_config(config_path: str) -> None:
    with open(config_path) as config:
        logging.config.dictConfig(json.load(config))
