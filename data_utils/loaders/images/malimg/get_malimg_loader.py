from collections import OrderedDict
from random import shuffle
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch as T
from data_utils import constants
from data_utils.utils.files import ifnot_create
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import (Compose, Grayscale, Normalize, Resize,
                                    ToTensor)

DEFAULT_TRANSFORM = Compose([Grayscale(), ToTensor()])
DATASET_PATH = ifnot_create(constants.DATASETS_DIR + "images/malimg/")


class BucketBatchSampler(Sampler):
    # want inputs to be an array
    @T.no_grad()
    def __init__(self, batch_size, inputs, dataset_idx):
        # # Throw an error if the number of inputs and targets don't match
        # if targets is not None:
        #     if len(inputs) != len(targets):
        #         raise Exception(
        #             "[BucketBatchSampler] inputs and targets have different sizes"
        #         )
        # Remember batch size
        self.batch_size = batch_size
        # For each data item (it's index), keep track of combination of input and target lengths
        self.ind_n_len = []
        for i in dataset_idx:
            self.ind_n_len.append((i, (inputs[i][0], inputs[i][1])))

        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    @T.no_grad()
    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[(5,8)] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [
                    indices[i:(i + self.batch_size)]
                    for i in range(0, len(indices), self.batch_size)
            ]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map(
        )  # <-- Could be a waste of performance
        # shuffle all the batches so they aren't ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i


def get_bucket_batch_sampler(dataset, train_sampler, eval_sampler, batch_size,
                             train_idx, test_idx):
    temp_dl = DataLoader(dataset, batch_size=1)

    dataset_len = []
    for data, _ in temp_dl:
        dataset_len.append(data.size()[2:4])

    train_batch_sampler = BucketBatchSampler(batch_size=batch_size,
                                             inputs=dataset_len,
                                             dataset_idx=train_idx)

    eval_batch_sampler = BucketBatchSampler(batch_size=batch_size,
                                            inputs=dataset_len,
                                            dataset_idx=test_idx)

    del temp_dl, dataset_len
    return train_batch_sampler, eval_batch_sampler


class MalimgDataset(VisionDataset):
    def __init__(self, image_folder_ds: ImageFolder, sampler):
        self.dataset = image_folder_ds
        self.sampler = sampler
        self.target = image_folder_ds.targets

    def __getitem__(self, index: int) -> Any:
        return self.dataset.__getitem__(index)

    def __len__(self) -> int:
        return len(self.sampler)

    def __repr__(self) -> str:
        return self.dataset.__repr__()

    def _format_transform_repr(self, transform: Callable,
                               head: str) -> List[str]:
        return self._format_transform_repr(transform, head)


def get_malimg_dl(
    batch_size=256,
    transforms=DEFAULT_TRANSFORM,
    use_bucket_batch_sampler=False,
    manual_seed=None,
    num_workers=0,
):

    dataset = ImageFolder(DATASET_PATH, transform=transforms)

    _, counts = T.unique(T.tensor(dataset.targets), return_counts=True)

    weights = (1.0 / counts).cuda()

    train_idx, eval_idx = train_test_split(np.arange(len(dataset.targets)),
                                           test_size=0.1,
                                           shuffle=True,
                                           stratify=dataset.targets)

    train_sampler = SubsetRandomSampler(train_idx)
    eval_sampler = SubsetRandomSampler(eval_idx)

    worker_init_fn = None
    if manual_seed is not None:
        worker_init_fn = np.random.seed(manual_seed)
        num_workers = 0

    if use_bucket_batch_sampler:
        train_sampler, eval_sampler = get_bucket_batch_sampler(
            dataset, train_sampler, eval_sampler, batch_size, train_idx,
            eval_idx)
        train_ds = MalimgDataset(dataset, train_sampler)
        eval_ds = MalimgDataset(dataset, eval_sampler)

        train_dl = DataLoader(train_ds,
                              batch_sampler=train_sampler,
                              num_workers=num_workers,
                              worker_init_fn=worker_init_fn)

        eval_dl = DataLoader(eval_ds,
                             batch_sampler=eval_sampler,
                             num_workers=num_workers,
                             worker_init_fn=worker_init_fn)

    else:
        train_ds = MalimgDataset(dataset, train_sampler)
        eval_ds = MalimgDataset(dataset, eval_sampler)

        train_dl = DataLoader(train_ds,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True)

        eval_dl = DataLoader(eval_ds,
                             sampler=eval_sampler,
                             batch_size=batch_size,
                             worker_init_fn=worker_init_fn,
                             pin_memory=True)
    return train_dl, eval_dl, weights


# train_dl, eval_dl = get_malimg_dl(batch_size=32)

# train_total, eval_total = 0, 0
# for i, (inputs, target) in enumerate(train_dl):
#     train_total += inputs.size(0)

# for i, (inputs, target) in enumerate(eval_dl):
#     eval_total += inputs.size(0)

# print("Total train_set", train_total)
# print("Total eval_set", eval_total)
