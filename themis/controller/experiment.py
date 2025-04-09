import logging
import os
import os.path as osp
import pickle
import random

import lm_eval
import yaml
from coolname import generate_slug
from lm_eval.tasks import TaskManager

from themis.definitions.constants import EXPERIMENTS_PATH, TASK_PATH
from themis.model.model import get_generation_args, get_model
from themis.utils.config import Config, GenerationConfig, MultipleChoiceConfig

logger = logging.getLogger(__name__)


class Experiment:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.save_path = self._make_experiment_dir()
        self._init_task()
        self._instantiate_model()

    def run(self) -> None:
        self.evaluate()
        self._log_results()
    
    def init_task(self):
        logger.info("Initializing tasks")
        self.task_manager = TaskManager(include_path=TASK_PATH, include_defaults=False)

    def instantiate_model(self):
        logger.info("Instantiating model")
        self.lm = get_model(config=self.config)

    def _make_experiment_dir(self) -> str:
        random.seed(os.urandom(128))
        save_path = osp.join(EXPERIMENTS_PATH, generate_slug())

        logger.info(f"Creating experiment directory: {save_path}")
        os.makedirs(save_path)

        return save_path

    def _log_results(self) -> None:
        logger.info("Saving evaluation results and configuration")
        self._dump_results()  # save results dictionary
        self._dump_config()  # save experiment config

    def _dump_results(self) -> None:
        path = osp.join(self.save_path, "results.pickle")
        with open(path, "wb") as handle:
            pickle.dump(self.results, handle)

    def _dump_config(self) -> None:
        path = osp.join(self.save_path, "config.yaml")
        with open(path, "w") as f:
            yaml.dump(self.config.signature(), stream=f)
    
    def _get_all_seed(self) -> dict:
        return {
            "random_seed": self.config.seed,
            "numpy_random_seed": self.config.seed,
            "torch_random_seed": self.config.seed,
            "fewshot_random_seed": self.config.seed,
        }



class CrowsPairsMC(Experiment):
    def __init__(self, config: MultipleChoiceConfig) -> None:
        super().__init__(config=config)

    def evaluate(self):
        logger.info("Running evaluation task")
        self.results = lm_eval.simple_evaluate(
            model=self.lm,
            task=self.config.task,
            task_manager=self.task_manager,
            limit=self.config.limit,
            **self._get_all_seed()
        )


class CrowsPairsGen(Experiment):
    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config=config)

        self.system_instruction = "You are an AI assistant tasked to answer multiple choice questions."
        self.apply_chat_template = True

        self.gen_kwargs = get_generation_args(config=config)

    def evaluate(self):
        logger.info("Running evaluation task")
        self.results = lm_eval.simple_evaluate(
            model=self.lm,
            task=self.config.task,
            task_manager=self.task_manager,
            limit=self.config.limit,
            system_instruction=self.system_instruction,
            apply_chat_template=self.apply_chat_template,
            gen_kwargs=self._get_generation_args()
            **self._get_all_seed()
        )

    def _get_generation_args(self) -> str:
        """
        String of comma separated argument assignments
        Used by lm_eval.simple_evaluate for generation tasks
        """

        gen_kwargs = {
            "max_gen_tokens": self.config.max_tokens,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "temperature": self.config.temperature,
        }

        return ",".join([f"{arg}={value}" for arg, value in gen_kwargs.items()])

class CrowsPairsReflection(Experiment):
    pass

def experiment_factory(config: Config) -> Experiment:
    if config.task == "crows_pairs_mc":
        return CrowsPairsMC(config=config)

    if config.task == "crows_pairs_gen":
        return CrowsPairsGen(config=config)

    raise NotImplementedError(f"{config.task} not implemented")

