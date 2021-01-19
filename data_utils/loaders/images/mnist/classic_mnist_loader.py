import math

import matplotlib.pyplot as plt
import numpy as np
import torch as T
from data_utils import constants
from data_utils.datasets.simple_dataset import SimpleDataset
from data_utils.utils.files import ifnot_create
from numpy.random import _generator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor

ROOT_DIR = ifnot_create(constants.DATASETS_DIR + "images/mnist/")

DEFAULT_TRANSFORMS = Compose([
    ToTensor(),
])


def get_mnist(batch_size,
              transforms=DEFAULT_TRANSFORMS,
              manual_seed=None,
              generator=None,
              num_workers=4):

    worker_init_fn = None
    if manual_seed is not None:
        worker_init_fn = np.random.seed(manual_seed)
        num_workers = 0

    train_mnist = MNIST(ROOT_DIR + "train",
                        transform=transforms,
                        train=True,
                        download=True)
    test_mnist = MNIST(ROOT_DIR + "test",
                       transform=transforms,
                       train=False,
                       download=True)

    train_mnist = DataLoader(train_mnist,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             worker_init_fn=worker_init_fn,
                             generator=generator)

    test_mnist = DataLoader(test_mnist,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            worker_init_fn=worker_init_fn,
                            generator=generator)

    # plot_mnist(dataloader_dict['train']['dataloader'], plot_path, tune_path)
    return train_mnist, test_mnist


def get_mnist_per_target(size=100, batch_size=32, transforms=None):
    mnist = MNIST(ROOT_DIR + "train",
                  transform=transforms,
                  train=True,
                  download=True)

    mnist_per_digit = []
    for digit in range(10):

        digit_indices = T.where(mnist.targets == digit)

        x_digit = mnist.data[digit_indices]
        y_digit = mnist.targets[digit_indices]

        mnist_per_digit.append(
            SimpleDataset(T.ByteTensor(x_digit), T.LongTensor(y_digit)))

    # plot_average_per_target(mnist_per_digit)
    return mnist_per_digit
