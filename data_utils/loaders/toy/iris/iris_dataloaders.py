import torch as T
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class IrisDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = T.FloatTensor(x_data)
        self.y_data = T.LongTensor(y_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


def get_iris_loader(batch_size=32, scaler=None):
    iris = load_iris()

    data = iris['data']
    target = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']

    if scaler is not None:
        data = scaler.fit_transform(data)

    train_x, eval_x, train_y, eval_y = train_test_split(data,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=2)

    train_ds = IrisDataset(train_x, train_y)
    eval_ds = IrisDataset(eval_x, eval_y)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=True)

    return train_dl, eval_dl
