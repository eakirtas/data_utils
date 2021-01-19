import numpy as np
import scipy.io as sio
from data_utils import constants

TRAIN_DIR = constants.DATASETS_DIR + "reuters/train.mat"
TEST_DIR = constants.DATASETS_DIR + "reuters/test.mat"
DICT_DIR = constants.DATASETS_DIR + "reuters/dictionary.mat"


def get_dataloader():
    '''Reuters21578 is composed of documents that appeared in the Reuters newswire in 1987. We used the
    ModApte split limited to ten most frequent categories. We have used a processed (stemming,
    stop-word removal) version in bag-of-words format obtained from:
    http://people.kyb.tuebingen.mpg.de/pgehler/rap/. The dataset contains 11413 documents with
    12317 words/ dimensions. Two techniques were used to reduce the dimensionality of each document
    to contain the most informative and less correlated words. First, the words were sorted based
    on their frequency of occurrence in the data set. Then, the words with frequency below 4 and
    above 70 were removed. After that the information gain with the class attribute [43] was used
    to select the most informative words which do not occur in every topic. The remaining words in
    the data set were sorted using this method, and the less important words were removed based on
    the desired dimension of documents.
    '''

    train_reuters = sio.loadmat(TRAIN_DIR)
    test_reuters = sio.loadmat(TEST_DIR)
    dict_reuters = sio.loadmat(DICT_DIR)
    print("Labels:", np.unique(train_reuters['labels']))
    word = dict_reuters['word']
    print("train", train_reuters.keys())
    train_counts = train_reuters['counts']
    train_labels = train_reuters['labels']


# print("Labels", np.unique(train_labels.squeeze(), return_counts=True))
# print("Dict keys:", dict_reuters)
# print("Labels:", train_labels)

get_dataloader()
