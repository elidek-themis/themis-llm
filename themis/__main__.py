import logging

import hydra

from dotenv import load_dotenv
from hydra.core.hydra_config import OmegaConf, DictConfig

from themis.utils.tools import slug
from themis.core.controller import Controller
from themis.definitions.constants import CONFIG_PATH

_ = load_dotenv()
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def experiment_entry(config: DictConfig) -> dict:
    controller = Controller(config=config)
    controller.validate(verbose=True)
    return controller.run_experiment()


if __name__ == "__main__":
    OmegaConf.register_new_resolver("slug", slug)
    experiment_entry()
