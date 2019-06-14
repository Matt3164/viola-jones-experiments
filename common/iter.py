from typing import Iterator


def batch(iter: Iterator, batch_size: int):

    batch = list()

    for e in iter:

        batch.append(e)

        if len(batch)>=batch_size:
            yield batch
            batch=list()

    yield batch
