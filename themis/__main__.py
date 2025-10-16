import logging

from typing import Any

import hydra

from dotenv import load_dotenv
from hydra.utils import call
from hydra.core.hydra_config import OmegaConf, DictConfig

import themis.evaluation  # noqa: F401

from themis.utils.tools import (
    to_string,
    sanitize_task_name,
    sanitize_model_name,
)
from themis.definitions.constants import CONFIG_PATH

_ = load_dotenv()
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("to_string", to_string)
OmegaConf.register_new_resolver("sanitize_model", sanitize_model_name)
OmegaConf.register_new_resolver("sanitize_tasks", sanitize_task_name)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def experiment_entry(config: DictConfig, **kwargs) -> dict[str, Any]:
    from lm_eval.evaluator import simple_evaluate  # noqa: PLC0415

    eval_config = call(config.evaluation)

    logger.info(f"Running evaluation: {eval_config.model}")
    return simple_evaluate(**eval_config.get_simple_evaluate_kwargs())


# if __name__ == "__main__":
#     experiment_entry()
