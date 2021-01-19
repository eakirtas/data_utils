import re

import torch as T
from data_utils import constants
from data_utils.datasets.simple_dataset import SimpleDataset
from data_utils.utils.files import ifnot_create
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")

ALL_LABELS = [
    'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
    'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
    'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
    'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
    'talk.politics.misc', 'talk.religion.misc'
]

ROOT_DIR = ifnot_create(constants.DATASETS_DIR +
                        "newsgroup/newsgroup_pytorch/")


def remove_digits(text):
    text = text.lower()
    text = re.sub(r'\d*', '', text)
    text = re.sub('![\W\_]', '', text)
    return text


def get_dataloader(categories=None, batch_size=32, device=DEVICE):

    news = fetch_20newsgroups(subset='all',
                              categories=categories,
                              remove=('footer'),
                              shuffle=True,
                              data_home=ROOT_DIR)

    preprocess = Pipeline([
        ('vect',
         CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(
             ['com', 'edu', 'org', 'gov', 'net']),
                         max_df=0.9,
                         min_df=0.01,
                         token_pattern=r'\b[a-zA-Z]{3,15}\b',
                         lowercase=True,
                         preprocessor=remove_digits)),
        ('tfidf', TfidfTransformer())
    ])

    preprocess = preprocess.fit(news.data, news.target)

    vocab = preprocess['vect'].get_feature_names()

    x_all, y_all = preprocess.transform(news.data), news.target

    x_train, x_test, y_train, y_test = train_test_split(x_all,
                                                        y_all,
                                                        test_size=0.25)

    train_dataloader = DataLoader(SimpleDataset(
        T.tensor(x_train.todense(), dtype=T.float),
        T.tensor(y_train).long()),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=0)

    test_dataloader = DataLoader(SimpleDataset(
        T.tensor(x_test.todense(), dtype=T.float),
        T.tensor(y_test).long()),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=0)

    return train_dataloader, test_dataloader, vocab


def get_per_target():
    train_dataloader, _, vocab = get_dataloader()
    dataset = train_dataloader.dataset
    per_class = []

    for tclass in range(20):
        class_indices = T.where(dataset.targets == tclass)

        x_class = dataset.data[class_indices]
        y_class = dataset.targets[class_indices]

        per_class.append(SimpleDataset(x_class, y_class))

    return per_class, vocab
