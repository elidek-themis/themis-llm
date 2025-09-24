import logging

from typing import Any
from argparse import Namespace

import hydra

from dotenv import load_dotenv
from lm_eval.__main__ import cli_evaluate
from hydra.core.hydra_config import OmegaConf, DictConfig

import themis.evaluation  # noqa: F401

from themis.utils.tools import slug, to_string, sanitize_task_name, sanitize_model_name
from themis.definitions.constants import CONFIG_PATH

_ = load_dotenv()
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("slug", slug)
OmegaConf.register_new_resolver("to_string", to_string)
OmegaConf.register_new_resolver("sanitize_model", sanitize_model_name)
OmegaConf.register_new_resolver("sanitize_tasks", sanitize_task_name)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def experiment_entry(config: DictConfig) -> dict[str, Any]:
    config = OmegaConf.to_container(cfg=config, resolve=True)
    cli_evaluate(Namespace(**config["evaluation"]))


if __name__ == "__main__":
    experiment_entry()
