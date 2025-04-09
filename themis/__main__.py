import click

from themis.controller.controller import Controller

controller = Controller()


@click.group()
def cli():
    pass


@cli.command(name="experiment")
def experiment_entrypoint() -> None:
    controller.run_experiment()


@cli.command()
def sync_wandb() -> None:
    Controller.log_to_wandb(controller)


if __name__ == "__main__":
    cli()
