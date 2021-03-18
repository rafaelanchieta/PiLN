from typing import Tuple, Generator

import gensim
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def read_corpus() -> Tuple[DataFrame, DataFrame]:
    
    ironia = pd.read_csv('data/ironia.csv', sep=';')
    nao_ironico = pd.read_csv('data/nao-ironico.csv', sep=';')
    
    X = []
    X.extend(ironia['text'].values)
    ironia_lenght = len(X)
    X.extend(nao_ironico['text'].values)
    
    y = []
    for i in range(len(ironia['text'].values)):
        y.append(0)
    for i in range(len(nao_ironico['text'].values)):
        y.append(1)
   
    return X, y

def split_data_set(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y, test_size = 0.1, random_state = 42)

    return X_train, X_test, y_train, y_test


def prepare_corpus(tweets: list) -> Generator:
    for i, line in enumerate(tweets):
        tokens = gensim.utils.simple_preprocess(line)
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
        
        
