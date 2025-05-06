import logging

from hydra.core.hydra_config import OmegaConf, DictConfig

from themis.core.experiment import Experiment
from themis.data.repository import Repository
from themis.definitions.config import Config, table_print
from themis.core.model.inference import inference

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, config: DictConfig) -> None:
        self.config = self._validate_config(config=config, verbose=True)
        self.repo = Repository()
        self.experiment = Experiment(config=self.config.experiment)

        self.experiment.init_task(repo=self.repo)

    def _validate_config(self, config: DictConfig, verbose: bool = False) -> Config:
        config = OmegaConf.to_container(cfg=config, resolve=True)  # to dict while resolving
        config = Config(**config)  # pass config for validations

        logger.info("Valid input configuration")
        if verbose:
            table_print(config)

        return config

    def run_experiment_job(self) -> dict | None:
        inference.setup(interface=self.config.interface)  # initiliaze model
        results = self.experiment.evaluate(lm=inference.backend)  # run evaluation

        return results
