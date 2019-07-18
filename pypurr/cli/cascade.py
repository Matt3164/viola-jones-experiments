from pypurr.cli.build_dataset import _build_dataset
from pypurr.cli.extract_negatives import _extract_negatives
from pypurr.cli.train_cli import _train

if __name__ == '__main__':
    for run_id in range(15, 25):
        _extract_negatives(run_id)
        _build_dataset(run_id)
        _train(run_id)