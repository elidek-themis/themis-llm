import logging

from typing import Any

import lm_eval

from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.group import ConfigurableGroup

from themis.data.repository import Repository
from themis.definitions.config import ExperimentConfig
from themis.definitions.constants import TASK_PATH
from themis.definitions.exceptions import ExperimentExists

logger = logging.getLogger(__name__)


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def init_task(self, repo: Repository) -> None:
        logger.info("Initializing tasks directory")
        self.task_manager = TaskManager(include_path=TASK_PATH, include_defaults=False)
        task_dict = get_task_dict(self.config.task, task_manager=self.task_manager)
        task_dict = self._sanitize_task_dict(task_dict=task_dict, repo=repo)
        self.config.task, self.tasks = zip(*task_dict.items())

    def _sanitize_task_dict(self, task_dict: dict[Any, Any], repo: Repository) -> dict[str, ConfigurableTask]:
        task = next(iter(task_dict))  # first entry of task_dict
        # combine sub task dicts if this is a group
        if isinstance(task, ConfigurableGroup):
            task_dict = {k: v for task in task_dict.values() for k, v in task.items()}

        tasks = list(task_dict.keys())
        for task_name in tasks:
            # delete sub tasks if they exist in the repository
            if repo.contains(config=self.config, task=task_name):
                logger.info(f"Task {task_name} already in the repository for {self.config.model}")
                logger.info(f"Skipping sub task {task_name}")
                del task_dict[task_name]

        if not task_dict:
            msg = "Experiment already run for given configuration"
            raise ExperimentExists(msg)

        return task_dict

    def evaluate(self, lm):
        logger.info("Running evaluation task")

        results = lm_eval.simple_evaluate(
            model=lm, tasks=self.config.task, task_manager=self.task_manager, **self.config.eval_kwargs.model_dump()
        )
        # update with the (modified) experiment config
        results["config"] = self.config
        return results
