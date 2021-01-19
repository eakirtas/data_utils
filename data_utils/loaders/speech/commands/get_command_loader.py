## This code in taken by https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio_tutorial.html
import os
import sys
from typing import Any, Callable, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from data_utils import constants
from data_utils.utils.files import ifnot_create
from torch.cuda.random import manual_seed
from torch.utils.data import DataLoader
from torchaudio import transforms
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS

F = TypeVar('F', bound=Callable[..., Any])

DEFAULT_TRANFORM_LAMDA = lambda origin_freq, resampled_rate: torchaudio.transforms.Resample(
    orig_freq=origin_freq, new_freq=resampled_rate)
DATASET_DIR = ifnot_create(constants.DATASETS_DIR + "speech/commands/")
DEFAULT_RESAMPLED_RATE = 8000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SubsetSC(SPEECHCOMMANDS):

    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [
                    os.path.normpath(os.path.join(self._path, line.strip()))
                    for line in fileobj
                ]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list(
                "testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

    def index_to_label(self, index: int) -> str:
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return self.labels[index]

    def label_to_index(self, word: str) -> torch.Tensor:
        # Return the position of the word in labels
        return torch.tensor(self.labels.index(word))

    def get_categories_len(self):
        return len(self.labels)

    def get_original_item(self,
                          n: int) -> Tuple[torch.Tensor, int, str, str, int]:
        return super().__getitem__(n)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        waveform, sample_rate, label, speaker_id, utterance_number = super(
        ).__getitem__(n)

        return waveform, self.label_to_index(label)


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch,
                                            batch_first=True,
                                            padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, label in batch:
        tensors += [waveform]
        targets += [label]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


def get_speechcommand_loader(
        batch_size,
        resampled_rate=DEFAULT_RESAMPLED_RATE,
        lambda_transform=DEFAULT_TRANFORM_LAMDA,
        manual_seed=None,
        num_workers=4,
        pin_memory=False) -> Tuple[DataLoader, DataLoader, int, int, F]:

    train_ds = SubsetSC("training")
    test_ds = SubsetSC("testing")

    waveform, sample_rate, label, speaker_id, utterance_number = train_ds.get_original_item(
        0)
    transform = lambda_transform(sample_rate, resampled_rate)

    tranformed_waveform = transform(waveform)

    n_input, n_output = tranformed_waveform.size(
        0), train_ds.get_categories_len()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             collate_fn=collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, test_loader, n_input, n_output, transform
