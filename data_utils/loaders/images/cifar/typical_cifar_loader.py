import numpy as np
import torch as T
import torchvision
import torchvision.transforms as transforms
from data_utils import constants
from data_utils.utils.files import ifnot_create
from torch.utils.data import DataLoader

CIFAR10_DIR = ifnot_create(constants.DATASETS_DIR + "images/cifar10/")
CIFAR100_DIR = ifnot_create(constants.DATASETS_DIR + "images/cifar100/")
DEFAULT_TRANFORM = transforms.Compose([transforms.ToTensor()])

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR10_DEFAULT_TRAIN_TRANFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

CIFAR10_DEFAULT_TEST_TRANFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CIFAR100_DEFAULT_TRAIN_TRANFORM = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
])

CIFAR100_DEFAULT_TEST_TRANFORM = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])


def get_cifar10_dl(batch_size=32,
                   train_transforms=CIFAR10_DEFAULT_TRAIN_TRANFORM,
                   test_transforms=CIFAR100_DEFAULT_TEST_TRANFORM,
                   manual_seed=None,
                   num_workers=4,
                   download=True):

    worker_init_fn = None
    if manual_seed is not None:
        worker_init_fn = np.random.seed(manual_seed)
        num_workers = 0

    trainset = torchvision.datasets.CIFAR10(root=CIFAR10_DIR,
                                            train=True,
                                            download=download,
                                            transform=train_transforms)

    trainloader = T.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers,
                                          worker_init_fn=worker_init_fn)

    if test_transforms is None:
        test_transforms = train_transforms

    testset = torchvision.datasets.CIFAR10(root=CIFAR10_DIR,
                                           train=False,
                                           download=download,
                                           transform=test_transforms)

    testloader = T.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=num_workers,
                                         worker_init_fn=worker_init_fn)

    return trainloader, testloader


def get_cifar100_dl(batch_size=32,
                    train_transforms=CIFAR100_DEFAULT_TRAIN_TRANFORM,
                    test_transforms=CIFAR100_DEFAULT_TEST_TRANFORM,
                    manual_seed=None,
                    num_workers=4):

    worker_init_fn = None
    if manual_seed is not None:
        worker_init_fn = np.random.seed(manual_seed)
        num_workers = 0

    trainset = torchvision.datasets.CIFAR100(root=CIFAR100_DIR,
                                             train=True,
                                             download=True,
                                             transform=train_transforms)

    trainloader = T.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers,
                                          worker_init_fn=worker_init_fn)
    if test_transforms is None:
        test_transforms = train_transforms

    testset = torchvision.datasets.CIFAR100(root=CIFAR100_DIR,
                                            train=False,
                                            download=True,
                                            transform=test_transforms)
    testloader = T.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=num_workers,
                                         worker_init_fn=worker_init_fn)

    return trainloader, testloader
