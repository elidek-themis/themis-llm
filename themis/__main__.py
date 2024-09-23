import click
from dotenv import load_dotenv

from themis.controller.evaluator import make_experiment
from themis.controller.logger import log_to_wandb
from themis.definitions.constants import CONFIG_PATH, LOG_CONFIG_PATH
from themis.utils.config import Config, load_config
from themis.utils.tools import initialize_logging


@click.group()
def cli():
    pass


@cli.command(name="experiment")
def experiment_entrypoint() -> None:
    cfg = init()
    make_experiment(cfg=cfg)


@cli.command()
def sync_wandb() -> None:
    cfg = init()
    log_to_wandb(cfg=cfg)


def init() -> Config:
    _ = load_dotenv()
    initialize_logging(config_path=LOG_CONFIG_PATH)
    cfg = load_config(config_path=CONFIG_PATH)

    return cfg


if __name__ == "__main__":
    cli()
