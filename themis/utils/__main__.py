import logging

import click
from dotenv import load_dotenv

import os.path as osp


from themis.data.repository import Repository
from themis.definitions.config import Config, table_print
from themis.definitions.constants import CONFIG_PATH, TASK_PATH
from themis.utils.tools import recompose_config, slug

from hydra.core.hydra_config import DictConfig, OmegaConf
from lm_eval.tasks import TaskManager

_ = load_dotenv()
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command(name="read_config")
def read_config() -> None:
    OmegaConf.register_new_resolver("slug", slug)
    
    config = recompose_config(config_dir=CONFIG_PATH)
    OmegaConf.resolve(config) # interpolations
    config = OmegaConf.to_container(cfg=config)  # to dict
    config = Config(**config)  # pass config for validations
    table_print(config)
    
    print (config.experiment)
    
    
    
@cli.command(name="list_tasks")
def list_tasks() -> None:
    task_manager = TaskManager(include_path=TASK_PATH, include_defaults=False)
    print(task_manager.list_all_tasks())


@cli.command(name="print_repo")
def print_repo() -> None:
    repo = Repository()
    print(repo)


if __name__ == "__main__":
    cli()
