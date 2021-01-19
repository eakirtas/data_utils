import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch as T
from sklearn import datasets
from torch.utils.data import DataLoader, Dataset


def normalize(min_x, max_x, a, b, x):
    x_new = a + (((x - min_x) * (b - a)) / (max_x - min_x))
    return x_new


def transformation_list(t_list, x):
    for tranformation in t_list:
        x = tranformation(x)

    return x


def moon_transformation(x, x_tr=1, y_tr=0.5):
    x[:, 0] = x[:, 0] + x_tr
    x[:, 1] = x[:, 1] + y_tr
    return x


def mooon_minmax_normalizations(x_limits, y_limits, a, b, data):
    data[:, 0] = normalize(x_limits[0], x_limits[1], a, b, data[:, 0])
    data[:, 1] = normalize(y_limits[0], y_limits[1], a, b, data[:, 1])
    return data


# def moon_align_origin(x):
#     x[:, 0] = x[:, 0] - 0.5
#     x[:, 1] = x[:, 1] - 0.25
#     return x


def moon_degrees_transformation(x, theta=np.pi / 2):

    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    for i in range(len(x)):
        x[i] = np.matmul(rotation_matrix, x[i].T)

    return x


def plot_moons(x_data, y_data, plot_path=None):
    colors = ['red', 'green', 'blue', 'purple']
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.scatter(x_data[:, 0],
               x_data[:, 1],
               c=y_data,
               cmap=matplotlib.colors.ListedColormap(colors))
    ax.set_title("Moons Dataset")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    if plot_path is not None:
        fig.savefig(plot_path)
    else:
        fig.show()
    return fig


class MoonDataset(Dataset):
    def __init__(self, x_data, y_data, plot_path=None):

        self.x_data = T.FloatTensor(x_data)
        self.y_data = T.LongTensor(y_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


def get_moons(train_size=300,
              test_size=100,
              batch_size=32,
              transformation=None):

    x_train, y_train = datasets.make_moons(n_samples=train_size,
                                           shuffle=True,
                                           noise=None)

    x_test, y_test = datasets.make_moons(n_samples=test_size,
                                         shuffle=True,
                                         noise=None)
    if transformation is not None:
        x_train = transformation(x_train)
        x_test = transformation(x_test)

    train_moon = MoonDataset(x_train, y_train)
    test_moon = MoonDataset(x_test, y_test)

    fig = plot_moons(x_train, y_train)

    train_dl = DataLoader(train_moon, batch_size=batch_size, num_workers=4)

    test_dl = DataLoader(test_moon, batch_size=batch_size, num_workers=4)

    return train_dl, test_dl
