import os

import gensim
import joblib
from gensim.models import Doc2Vec


def load_classifier_model(filepath):
    cls_model = joblib.load(filepath)
    return cls_model


def get_d2v_model(filepath):
    model = Doc2Vec.load(filepath)
    return model


def Ironia(corpus, sentence):
    if corpus == 'twitter':
        cls_model = load_classifier_model(os.getcwd() + '/trained-models/twitter_classifier.model')
        d2v_model = get_d2v_model(os.getcwd() + '/embeddings/twitter_doc2vec.model')
        vector_representation = d2v_model.infer_vector(gensim.utils.simple_preprocess(sentence))
        pred = cls_model.predict([vector_representation])[0]
        if pred:
            return 'Ironia'
        else:
            return 'Não irônico'
    else:
        cls_model = load_classifier_model(os.getcwd() + '/trained-models/news_classifier.model')
        d2v_model = get_d2v_model(os.getcwd() + '/embeddings/news_doc2vec.model')
        vector_representation = d2v_model.infer_vector(gensim.utils.simple_preprocess(sentence))
        pred = cls_model.predict([vector_representation])[0]
        if pred:
            return 'Ironia'
        else:
            return 'Não irônico'