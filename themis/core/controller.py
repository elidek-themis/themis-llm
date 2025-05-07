import logging

from hydra.core.hydra_config import OmegaConf, DictConfig

import themis.core.evaluation  # noqa: F401

from themis.data.repository import Repository
from themis.definitions.config import Config, table_print
from themis.core.model.inference import inference
from themis.definitions.registry import get_eval

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.repo = Repository()

    def validate(self, verbose: bool = True) -> None:
        self.config = self._validate_config(config=self.config, verbose=verbose)

        experiment_config = self.config.experiment
        eval_cls = get_eval(experiment_config.name)
        self.experiment = eval_cls(config=experiment_config)
        self.experiment.validate(repo=self.repo)

    def _validate_config(self, config: DictConfig, verbose: bool = False) -> Config:
        config = OmegaConf.to_container(cfg=config, resolve=True)  # to dict while resolving
        config = Config(**config)  # pass config for validations

        logger.info("Valid input configuration")
        if verbose:
            table_print(config)

        return config

    def run_experiment(self) -> dict | None:
        inference.setup(interface=self.config.interface)  # initiliaze model
        results = self.experiment.evaluate(lm=inference.lm)  # run evaluation

        return results
