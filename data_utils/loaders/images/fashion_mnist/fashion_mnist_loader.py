import numpy as np
import torch as T
import torchvision.transforms as transforms
from data_utils import constants
from data_utils.utils.files import ifnot_create
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets.mnist import FashionMNIST

FASHION_MNIST_DIR = ifnot_create(constants.DATASETS_DIR +
                                 "images/fashion_mnist/")
DEFAULT_TRANFORM = transforms.Compose([transforms.ToTensor()])


def get_fashion_mnist(batch_size,
                      transforms=DEFAULT_TRANFORM,
                      manual_seed=None,
                      num_workers=4):

    worker_init_fn = None
    if manual_seed is not None:
        worker_init_fn = np.random.seed(manual_seed)
        num_workers = 0

    train_mnist = FashionMNIST(
        FASHION_MNIST_DIR,
        transform=transforms,
        train=True,
        download=True,
    )

    test_mnist = FashionMNIST(
        FASHION_MNIST_DIR,
        transform=transforms,
        train=False,
        download=True,
    )

    train_dl = DataLoader(train_mnist,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          worker_init_fn=worker_init_fn)

    test_dl = DataLoader(test_mnist,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         worker_init_fn=worker_init_fn)

    return train_dl, test_dl


def get_fmnist_selection(batch_size,
                         transforms=DEFAULT_TRANFORM,
                         manual_seed=None,
                         num_workers=4,
                         selection=[1, 2]):

    worker_init_fn = None
    if manual_seed is not None:
        worker_init_fn = np.random.seed(manual_seed)
        num_workers = 0

    train_mnist = FashionMNIST(
        FASHION_MNIST_DIR,
        transform=transforms,
        train=True,
        download=True,
    )

    test_mnist = FashionMNIST(
        FASHION_MNIST_DIR,
        transform=transforms,
        train=False,
        download=True,
    )

    train_idx, test_idx = train_mnist.targets == selection[
        0], test_mnist.targets == selection[0]

    for id_class in selection[1:]:
        train_idx = T.logical_or(train_idx, train_mnist.targets == id_class)
        test_idx = T.logical_or(test_idx, test_mnist.targets == id_class)

    train_mnist.targets, train_mnist.data = train_mnist.targets[
        train_idx], train_mnist.data[train_idx]

    test_mnist.targets, test_mnist.data = test_mnist.targets[
        test_idx], test_mnist.data[test_idx]

    train_dl = DataLoader(train_mnist,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          worker_init_fn=worker_init_fn)

    test_dl = DataLoader(test_mnist,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         worker_init_fn=worker_init_fn)

    return train_dl, test_dl
