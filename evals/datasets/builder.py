"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os

import torch
from hydra.utils import instantiate
from PIL import ImageFile
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# avoid open file error
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy("file_system")


def build_loader(cfg, split, batch_size, num_gpus=1, **kwargs):
    """
    Build a PyTorch dataloader and the underlying dataset (using config).
    """
    # Build a dataset from the provided dataset config.
    dataset = instantiate(cfg, split=split, **kwargs)

    use_ddp = num_gpus > 1
    sampler = DistributedSampler(dataset) if use_ddp else None
    shuffle = (split == "train") and not use_ddp
    n_workers = min(len(os.sched_getaffinity(0)), 2)

    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=n_workers,
        drop_last=False,
        pin_memory=True,
        shuffle=shuffle,
        sampler=sampler,
    )

    return loader
