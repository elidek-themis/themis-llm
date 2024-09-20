import click

from themis.controller.evaluator import make_experiment
from themis.controller.logger import log_to_wandb
from themis.utils.tools import init


@click.group()
def cli():
    pass


@cli.command(name="experiment")
def experiment_entrypoint() -> None:
    cfg = init()
    make_experiment(cfg=cfg)


@cli.command(name="sync_wandb")
def sync_wandb() -> None:
    cfg = init()
    log_to_wandb(cfg=cfg)


if __name__ == "__main__":
    cli()
