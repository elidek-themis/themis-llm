from themis.definitions.constants import CONFIG_PATH, EXPERIMENTS_PATH, TEMPLATES_PATH, PROJECT, LOGS_PATH
from themis.utils.config import config_factory
import os, os.path as osp, pickle
from yaml import dump, load
from themis.utils.tools import load_yaml_config, make_active_prompt_template
from themis.controller.experiment import experiment_factory, Experiment
import logging
from lm_eval.loggers import WandbLogger

logger = logging.getLogger(__name__)

class Controller:

    config = config_factory(config_path=CONFIG_PATH)

    def __init__(self) -> None:
        pass
    
    def run_experiment(self) -> None:
        results_path = self._experiments_exists()

        if results_path:
            logger.info(f"Experiment already run in {results_path}")
        else:
            self.experiment = experiment_factory(config=self.config)
            make_active_prompt_template(self.config.prompt_template)
            self.experiment.run()
    
    @staticmethod
    def log_to_wandb(self) -> None:
        results_path = self.experiment_exists()

        if results_path:
            results = self._load_results(results_path=results_path)

            name = osp.basename(results_path)

            wandb_logger = WandbLogger(
                name=name,
                project=PROJECT,
                job_type="eval",
                group=self.config.task,
                config=self.config.signature(),
                dir=LOGS_PATH
            )
            logger.info("Logging evaluation results to Weights & Biases")
            # wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            wandb_logger.log_eval_samples(results["samples"])
        else:
            logger.info("No results found for given evaluation")

    @staticmethod
    def load_task_results(self, task: str) -> dict:
        """Returns the results of all models for a given task"""
        experiments = {}
        experiment_path = osp.join(EXPERIMENTS_PATH)

        for root, _, files in os.walk(experiment_path):
            if files:
                results_path, config_path = files
                config = load_yaml_config(osp.join(root, config_path))

                if config.task == task:
                    experiments[config.model] = self._load_results(osp.join(root, results_path))

        return experiments

    def _load_results(self, results_path: str) -> dict:
        with open(results_path, "rb") as handle:
            return pickle.load(handle)
        
    
    def _experiments_exists(self):
        for root, _, files in os.walk(EXPERIMENTS_PATH):
            if files:
                _, config_path = files
                config_description = load_yaml_config(osp.join(root, config_path))

                if self.config == config_description:
                    return osp.join(root)
        
        return False