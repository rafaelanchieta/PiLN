from gensim.models import Doc2Vec
from nltk import NaiveBayesClassifier

import util


def train_and_save():

    ironia, nao_ironico = util.read_corpus()
    tweets = []
    tweets.extend(ironia['text'].values)
    tweets.extend(nao_ironico['text'].values)

    labels = []
    for i in range(len(ironia['text'].values)):
        labels.append(1)
    for i in range(len(nao_ironico['text'].values)):
        labels.append(0)
    documents = list(util.prepare_corpus(tweets))
    model = Doc2Vec(vector_size=100, window=4, min_count=1, epochs=300, sample=1e-4, workers=5)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    # model.save('../saved_models/doc2vec.model')
    # print(model.wv.most_similar('irônia'))
    train_tweets = []
    train_labels = labels[:14000]
    for i in range(14000):
        train_tweets.append(model[i])

    classifier = NaiveBayesClassifier()
    classifier.fit(train_tweets, train_labels)


if __name__ == '__main__':
    train_and_save()