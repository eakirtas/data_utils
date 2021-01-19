import torch as T
import torchvision.transforms as transforms
from data_utils import constants
from data_utils.utils.files import ifnot_create
from datasets import load_dataset

TINY_IMAGENET_DIR = ifnot_create(constants.DATASETS_DIR +
                                 "images/tiny_imagenet/")

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.4802, 0.4481, 0.3975],
        [0.2302, 0.2265, 0.2262],
    )
])

VAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
])


def get_tiny_imagenet_dl(batch_size=256,
                         train_transform=TRAIN_TRANSFORM,
                         test_transform=VAL_TRANSFORM,
                         number_workers=100):

    train_ds = load_dataset('Maysee/tiny-imagenet',
                            split='train').with_format("torch")

    val_ds = load_dataset(
        'Maysee/tiny-imagenet',
        split='valid',
    ).with_format("torch")

    train_dl = T.utils.data.DataLoader(train_ds,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       number_workers=number_workers,
                                       transform=train_transform)

    val_dl = T.utils.data.DataLoader(val_ds,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     number_workers=number_workers,
                                     transform=test_transform)

    return train_dl, val_dl
