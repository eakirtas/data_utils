import pdb

import torch as T
import torchtext as TT
from data_utils import constants
from torch.utils.data import DataLoader

ROOT_DIR = constants.DATASETS_DIR + "ag_news/"

# def generate_batch(batch):
#     label = T.tensor([entry[0] for entry in batch])
#     text = [entry[1] for entry in batch]
#     offsets = [0] + [len(entry) for entry in text]
#     # torch.Tensor.cumsum returns the cumulative sum
#     # of elements in the dimension dim.
#     # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

#     # offsets = T.tensor(offsets[:-1]).cumsum(dim=0)
#     # text = T.cat(text)

#     pdb.set_trace()
#     return text, offsets, label


class BagOfWords(T.nn.Module):
    def __init__(self, train_ds):
        "docstring"
        super(BagOfWords, self).__init__()
        self.train_ds = train_ds

        print("Data", self.train_ds._data[-10])
        print("Label", self.train_ds._labels)
        print("Vocab Len", len(self.train_ds.get_vocab()))
        # print("Freq Lenght:", self.train_ds.get_vocab().freqs)

    def forward(self, x):
        data, counts = T.unique(x, return_counts=True)

        print("Size of x", x.size())
        # for i in x.squeeze():
        # print(self.train_ds.get_vocab().itos[int(i)], end=" ")
        #     # print("Lookup:", train_ds.get_vocab().lookup_indices(input))
        #     print("\nTarget:",
        #           TT.datasets.text_classification.LABELS['AG_NEWS'][target.item()])


# fixed_size = 100

# def generate_batch(batch):
#     print("Batch:", batch)
#     label = batch[-1][0]
#     text = batch[-1][1]
#     print("Label", label)
#     print("Text", text)

#     # label = T.tensor([entry[0] for entry in batch])
#     # text = [entry[1] for entry in batch]
#     # offsets = [0] + [len(entry) for entry in text]
#     # # torch.Tensor.cumsum returns the cumulative sum
#     # # of elements in the dimension dim.
#     # # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

#     # offsets = T.tensor(offsets[:-1]).cumsum(dim=0)
#     # text = T.cat(text)
#     return (label, text)

# def pad_data(data):
#     # Find max length of the mini-batch
#     max_len = max(list(zip(*data))[0])
#     label_list = list(zip(*data))[2]
#     txt_list = list(zip(*data))[3]
#     padded_tensors = torch.stack([torch.cat((txt, \
#             torch.tensor([pad_id] * (max_len - len(txt))).long())) \
#             for txt in txt_list])
#     return padded_tensors, label_list


def get_dataloader(batch_size=32):
    train_dataset, test_dataset = TT.datasets.AG_NEWS(root=ROOT_DIR, ngrams=1)

    train_vocab = TT.vocab.Vocab(train_dataset.get_vocab().freqs,
                                 min_freq=10,
                                 max_size=None)
    print("Vocab Len", len(train_vocab))
    # print("Vocab size", train_dataset.get_vocab().freqs)
    # # print("Vocab size", train_dataset.get_vocab().__dict__.values())
    # print("Vocab vectors", train_dataset.get_vocab().data)
    # print("Vocab size ", train_dataset.get_vocab()['freqs'])
    # print("Vocab size", train_dataset.get_labels())

    train_news = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_news = DataLoader(test_dataset)

    return train_dataset, test_dataset, train_news, test_news


train_ds, test_ds, train_news, test_news = get_dataloader()

print("Input", input)
bag_of_words = BagOfWords(train_ds)

for j in range(10):
    target, input = iter(train_news).next()
    print("Input size:", input.size())

    bag_of_words(input)
    # for i in input.squeeze():
    #     print(train_ds.get_vocab().itos[int(i)], end=" ")
    # # print("Lookup:", train_ds.get_vocab().lookup_indices(input))
    # print("\nTarget:",
    #       TT.datasets.text_classification.LABELS['AG_NEWS'][target.item()])
# print("Input shape", input.size())
# print("Input", input)
# print("Target", target)
