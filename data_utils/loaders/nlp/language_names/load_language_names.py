from __future__ import division, print_function, unicode_literals

import glob
import os
import random
import string
import unicodedata
from io import open

import numpy as np
import torch as T
from data_utils.constants import DATASETS_DIR
from torch.utils.data import DataLoader

LANGUAGE_DS = os.path.join(DATASETS_DIR, 'language_names/data/names/')
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)


def findFiles(path):
    return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS)


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def read_files():
    category_lines = {}
    all_categories = []

    for filename in findFiles(LANGUAGE_DS + '*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    return all_categories, category_lines


def from_tensor_to_letter(letter):
    array = letter.detach().cpu().numpy()
    return ALL_LETTERS[np.argmax(array).item()]


def print_names(x):
    names_in_batch = ''
    for i in range(x.size(1)):
        x_i = x[:, i, :]
        for letter in x_i:
            names_in_batch += from_tensor_to_letter(letter)
        names_in_batch += '\n'

    print("Names:", names_in_batch)


def print_names_batched(x):
    names_in_batch = ''
    for i in range(x.size(0)):
        x_i = x[i]
        for letter in x_i:
            names_in_batch += from_tensor_to_letter(letter)
        names_in_batch += '\n'

    print("Names:", names_in_batch)


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return ALL_LETTERS.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = T.zeros(1, N_LETTERS)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    tensor = T.zeros(len(line), N_LETTERS)
    for li, letter in enumerate(line):
        tensor[li][letterToIndex(letter)] = 1
    return tensor


def get_random_category(all_categories):
    return all_categories[random.randint(0, len(all_categories) - 1)]


def get_random_line(all_lines, min_per, max_per):
    start = int((len(all_lines) - 1) * min_per)
    stop = int((len(all_lines) - 1) * max_per)

    return all_lines[random.randint(start, stop)]


def randomTrainingExample(all_categories, category_lines, min_per, max_per):

    category = get_random_category(all_categories)
    line = get_random_line(category_lines[category], min_per, max_per)

    print("Line:", line)
    category_tensor = T.tensor([all_categories.index(category)], dtype=T.long)
    line_tensor = lineToTensor(line)

    return (line_tensor, line), (category_tensor, category)


def get_input_targets(all_categories,
                      category_lines,
                      min_per=0.0,
                      max_per=1.0):
    inputs, targets = [], []
    for c, category in enumerate(all_categories):

        start = int(min_per * len(category_lines[category]))
        stop = int(max_per * len(category_lines[category]))

        for i in range(start, stop):
            line = category_lines[category][i]
            inputs.append(lineToTensor(line))
            targets.append(c)

    return inputs, T.tensor(targets).view(-1, 1)


class LanguageDataset():
    def __init__(self, all_categories, category_lines, min_per, max_per, size):
        self.all_categories, self.category_lines = all_categories, category_lines
        self.size = size
        self.min_per = min_per
        self.max_per = max_per

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        print("Index", index)
        example = randomTrainingExample(self.all_categories,
                                        self.category_lines,
                                        min_per=self.min_per,
                                        max_per=self.max_per)

        print("Line tensor:", example[0][0].size())
        # print("Line:", example[1].size())

        print("Category Tensor:", example[1][0].size())
        # print("Category:", example[3].size())
        return


def get_language_names_dl(train_size=1000, test_size=3000, batch_size=32):
    print("Datasets path:", LANGUAGE_DS)
    print("Files:", findFiles(LANGUAGE_DS + '*.txt'))

    all_categories, category_lines = read_files()

    n_categories = len(all_categories)

    train_ds = LanguageDataset(all_categories, category_lines, 0.0, 0.9,
                               train_size)
    test_ds = LanguageDataset(all_categories, category_lines, 0.9, 1.0,
                              test_size)

    train_dataloader = DataLoader(train_ds,
                                  batch_size=batch_size,
                                  num_workers=4)

    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=4)

    return train_dataloader, test_dataloader, N_LETTERS, n_categories
