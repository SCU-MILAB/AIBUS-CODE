"""Exact validation shards and host-side synchronization for evaluation/IO."""

from datetime import timedelta

import torch.distributed as dist
from torch.utils.data import Sampler


class ValidationSampler(Sampler):
    """Contiguous, non-padding shards of the first N validation images.

    Concatenating rank outputs preserves dataset order, including confidence
    ties in AP calculation. Empty shards are allowed when N < world size.
    """

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.max_images = 0

    def _indices(self):
        size = len(self.dataset)
        if self.max_images > 0:
            size = min(size, self.max_images)
        return range(size * self.rank // self.world_size,
                     size * (self.rank + 1) // self.world_size)

    def __iter__(self):
        return iter(self._indices())

    def __len__(self):
        return len(self._indices())


def create_rank_zero_group(timeout_seconds):
    """Call on every rank after initializing the training process group."""
    return dist.new_group(backend='gloo', timeout=timedelta(seconds=timeout_seconds))


def wait_for_rank_zero(group):
    """Keep idle ranks off NCCL while rank zero evaluates the unwrapped model.

    NCCL implements barrier as a GPU all-reduce, which times out if validation
    exceeds the training collective timeout. This CPU group has its own timeout
    and leaves the training group's watchdog unchanged. None is single-GPU mode.
    """
    if group is not None:
        dist.barrier(group=group)
