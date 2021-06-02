from os import pipe
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer, recall_score, f1_score,  precision_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import operator

pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words = stopwords.words("english")+stopwords.words("portuguese"))),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams, bigrams or trigrams
    'clf__max_iter': (20, 50, 100),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
}

def saveModel(corpus, arq):
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    clf = grid_search.fit(corpus.text, corpus.prediction)

    print("\n\n----Best parameters set:----")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    with open(arq, 'wb') as f:
        pickle.dump(clf, f)
        f.close() 


def evaluateModel(X_train,y_train, X_test, y_test):
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    clf = grid_search.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    bacc = balanced_accuracy_score(df_test.prediction, y_pred)

    print(classification_report(y_test, y_pred, target_names=['not irony','irony']))

    print('\n------- Test Score -------')
    print(" ACCURACY: %0.4f" % acc)
    print("PRECISION: %0.4f" % prec)
    print("   RECALL: %0.4f" % rec)
    print("       F1: %0.4f" % f1)
    print("     BACC: %0.4f" % bacc)

    print("\n\n----Best parameters set:----")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == '__main__':
    print('\n\n------------ Tweets ----------------')
    #carrega corpus de tweets dividido em splits de 90% para treino e 10% para teste
    df_train = pd.read_csv('data/90-10/train_tweets.csv', encoding='utf-8', sep=',')
    df_test = pd.read_csv('data/90-10/test_tweets.csv', encoding='utf-8', sep=',')
    evaluateModel(df_train.text,df_train.prediction, df_test.text, df_test.prediction)
    

    # Os splits de treino e teste são concatenados para treinar o modelo em todo o corpus de tweets e salvá-lo
    # corpus = pd.concat([df_train,df_test])
    # saveModel(corpus, 'models/ModeloTweetsTF-IDF.pickle')

    #repete mesmo processo para notícias

    print('\n\n------------ News ----------------')
    df_train = pd.read_csv('data/90-10/train_news.csv', encoding='utf-8', sep=',')
    df_test = pd.read_csv('data/90-10/test_news.csv', encoding='utf-8', sep=',')
    evaluateModel(df_train.text,df_train.prediction, df_test.text, df_test.prediction)

    # corpus = pd.concat([df_train,df_test])
    # saveModel(corpus, 'models/ModeloNewsTF-IDF.pickle')