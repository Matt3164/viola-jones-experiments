import click

from pypurr.cli.build_dataset import _build_dataset
from pypurr.cli.compute_scales import _compute_scales
from pypurr.cli.extract_negatives import _extract_negatives
from pypurr.cli.extract_positives import _extract_positives
from pypurr.cli.extract_raw_negatives import _extract_raw_negatives
from pypurr.cli.train_cli import _train
from pypurr.tools.preparation.preparation import _prepare


@click.group()
def cli():
    pass


@cli.command()
@click.argument("run_id", default=0)
def build_dataset(run_id: int):
    return _build_dataset(run_id)


@cli.command()
def compute_scales():
    _compute_scales()

@cli.command()
@click.argument("run_id", default=0)
def extract_negatives(run_id: int):
    _extract_negatives(run_id)

@cli.command()
def extract_positives():
    _extract_positives()

@cli.command()
@click.argument("run_id", default=0)
def extract_raw_negatives(run_id: int):
    _extract_raw_negatives(run_id)

@cli.command()
def prepare():
    _prepare()

@cli.command()
@click.argument("run_id", default=0)
def train(run_id: int):
    _train(run_id)

if __name__ == '__main__':
    cli()

