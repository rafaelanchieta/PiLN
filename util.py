from typing import Tuple, Generator

import gensim
import pandas as pd
from pandas import DataFrame


def read_corpus() -> Tuple[DataFrame, DataFrame]:
    ironia = pd.read_csv('data/ironia.csv', sep=';')
    nao_ironico = pd.read_csv('data/nao-ironico.csv', sep=';')
    return ironia, nao_ironico


def prepare_corpus(tweets: list) -> Generator:
    for i, line in enumerate(tweets):
        tokens = gensim.utils.simple_preprocess(line)
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])