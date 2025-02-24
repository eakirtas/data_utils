from __future__ import division, print_function

import os

import h5py
import numpy as np
import torch
from data_utils import constants
from torch.utils.data import DataLoader, Dataset

FI2020_DIR = os.path.join(constants.DATASETS_DIR,
                          'financial/fi2020/lob_raw.h5')


class LOB_WF(Dataset):
    def __init__(self,
                 h5_path=FI2020_DIR,
                 split=1,
                 train=True,
                 n_window=1,
                 normalization='std',
                 epsilon=1e-15,
                 horizon=0,
                 shift=None):
        """
        Loads the LoB dataset and prepares it to perform an anchored walk-forwared ecaluation

        :param h5_path:  Path to the h5 file containing the dataset
        :param split: split to use (0 to 8)
        :param train: whether to load the train or the test split
        :param n_window: window of features before the current time stamp to load
        :param normalization: None or 'std' (z-score normalization)
        :param epsilon: epsilon to be used to ensure the stability of the normalization
        :param horizon: the prediction horizon (0 -> next (10), 1-> next 5 (50), 2-> next 10 (100))
        """

        self.window = n_window

        assert 0 <= split <= 8
        assert 0 <= horizon <= 2

        self.split = split
        self.horizon = horizon

        # Translate the prediction to horizon to the data
        if horizon == 1:
            horizon = 3
        elif horizon == 2:
            horizon = 4

        # Load the data
        file = h5py.File(
            h5_path,
            'r',
        )
        features = np.float32(file['features'])
        targets = np.int32(file['targets'])
        day_train_split_idx = file['day_train_split_idx'][:].astype('bool')
        day_test_split_idx = file['day_test_split_idx'][:].astype('bool')
        stock_idx = file['stock_idx'][:].astype('bool')
        file.close()

        # Get the data for the specific split and setup (train/test)
        if train:
            idx = day_train_split_idx[split]

            # Get the statistics needed for normalization
            if normalization == 'std':
                self.mean = np.mean(features[idx], axis=0)
                self.std = np.std(features[idx], axis=0)
                features = (features - self.mean) / (self.std + epsilon)
        else:
            idx = day_test_split_idx[split]

            if shift is not None:
                print("testing shift = ", shift)
                # Shift the testing features only
                features[idx] += shift[0] * np.mean(features, axis=0)
                features[idx] *= shift[1]

            # Also get the train data to normalize the test data accordingly (if needed)
            if normalization == 'std':
                train_idx = day_train_split_idx[split]
                self.mean = np.mean(features[train_idx], axis=0)
                self.std = np.std(features[train_idx], axis=0)
                features = (features - self.mean) / (self.std + epsilon)
                del train_idx

        # Get the data per stock
        self.features_per_stock = []
        self.labels = []
        for i in range(len(stock_idx)):
            cur_idx = np.logical_and(idx, stock_idx[i])
            self.features_per_stock.append(features[cur_idx])
            self.labels.append(targets[cur_idx, horizon])

        # Create a lookup table to find the correct stock
        self.look_up_margins = []
        current_sum = 0
        for i in range(len(self.features_per_stock)):
            # Remove n_window since they are used to ensure that we are always operate on a full window
            cur_limit = self.features_per_stock[i].shape[0] - n_window - 1
            current_sum += cur_limit
            self.look_up_margins.append(current_sum)

        # Get the total number of samples
        self.n = self.look_up_margins[-1]
        self.n_stocks = len(self.look_up_margins)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):

        # Get the stock id
        stock_id = self.n_stocks - 1
        for i in range(self.n_stocks - 1):
            if idx < self.look_up_margins[i]:
                stock_id = i
                break

        # Get the in-split idx for the corresponding stock
        if stock_id > 0:
            in_split_idx = idx - self.look_up_margins[stock_id - 1]
        else:
            in_split_idx = idx

        # Get the actual data with a window of the previous window values
        cur_idx = in_split_idx + self.window

        data = self.features_per_stock[stock_id][cur_idx - self.window:cur_idx]
        label = self.labels[stock_id][cur_idx]

        return torch.Tensor(np.copy(data)), torch.Tensor([label]).long()


def get_wf_lob_loaders(h5_path=FI2020_DIR,
                       window=50,
                       split=0,
                       horizon=0,
                       batch_size=128,
                       class_resample=False,
                       normalization='std',
                       shift=None):
    """
    Prepare PyTorch loaders for training and evaluating a model
    :param h5_path:  Path to the h5 file containing the dataset
    :param n_window: window of features before the current time stamp to load    :param split:
    :param split: split to use (0 to 8)
    :param horizon: the prediction horizon (0 -> next, 1-> next 5, 2-> next 10)
    :param batch_size: the batch size to be used
    :param n_workers: number of workers to use for loading the data
    :return: the train and test loaders
    """

    train_dataset = LOB_WF(h5_path=h5_path,
                           split=split,
                           train=True,
                           n_window=window,
                           normalization=normalization,
                           epsilon=1e-15,
                           horizon=horizon)

    test_dataset = LOB_WF(h5_path=h5_path,
                          split=split,
                          train=False,
                          n_window=window,
                          normalization=normalization,
                          epsilon=1e-15,
                          horizon=horizon,
                          shift=shift)

    if class_resample:
        from data_utils.utils.samplers.imbalance_dataset_sampler import \
            ImbalancedDatasetSampler
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=1,
            sampler=ImbalancedDatasetSampler(train_dataset))
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=1,
                                  shuffle=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=1)

    return train_loader, test_loader
