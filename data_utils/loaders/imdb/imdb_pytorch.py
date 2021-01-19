import numpy as np
import torch
from data_utils import constants
from torch.nn import functional as F
from torchtext import data, datasets
from torchtext.vocab import GloVe, Vectors

ROOT_DIR = constants.DATASETS_DIR + "imdb/imdb_pytorch/"


def get_dataloader(test_sen=None):
    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.

    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.

    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.

    """

    tokenize = lambda x: x.split()
    in_text = data.Field(sequential=True,
                         tokenize=tokenize,
                         lower=True,
                         include_lengths=True,
                         batch_first=True,
                         fix_length=200)
    label = data.LabelField()
    imdb_ds = datasets.IMDB(ROOT_DIR, in_text, label)
    train_data, test_data = imdb_ds.splits(in_text, label)
    in_text.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    label.build_vocab(train_data)

    word_embeddings = in_text.vocab.vectors

    print("Length of Text Vocabulary: " + str(len(in_text.vocab)))
    # print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print("Label Length: " + str(len(label.vocab)))
    print("Train Data", train_data[0].__dict__.keys())
    print("Train Data", train_data[0].__dict__.values())
    print("Word embeddings", word_embeddings)

    train_iter, test_iter = imdb_ds.iters(batch_size=32)
    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    # vocab_size = len(TEXT.vocab)

    return in_text, label, word_embeddings, train_iter, test_iter


TEXT, LABEL, word_embeddings, train_iter, test_iter = get_dataloader()
batch = next(iter(train_iter))
# print("Shape in_text", in_text.size())
print("in_text", batch.text[:, 5].size())

# print("Target", target)
# print("Word embeddings")

# input, target = train_iter[1]

print("Word embeddings", word_embeddings)
