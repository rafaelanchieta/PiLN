
from gensim.models import Doc2Vec
import util
import gensim
from gensim.models import KeyedVectors
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics._classification import accuracy_score
from sklearn.metrics import balanced_accuracy_score

def train_and_save():

    X, y = util.read_corpus() 
    documents = list(util.prepare_corpus(X))
    model = Doc2Vec(vector_size=100, window=4, min_count=1, epochs=300, sample=1e-4, workers=5)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('saved_models/doc2vec.model')
    #model.save_word2vec_format('saved_models/doc2vec.bin', binary=True)
    print(model.wv.most_similar('ironia'))
    
def get_d2v_model(filepath):
    model = Doc2Vec.load(filepath)
    #model = Doc2Vec.load_word2vec_format('saved_models/doc2vec.bin', binary=True)
    return model

def get_embedding_model(filepath):
    model = KeyedVectors.load_word2vec_format(filepath)
    return model

def generate_d2v_vectors_from_splitted_data(d2v_model,X,y):
    
    X_train, X_test, y_train, y_test = util.split_data_set(X,y)
    
    for i in range(len(X_train)):
        model_vector = d2v_model.infer_vector(gensim.utils.simple_preprocess(X_train[i]))
        X_train[i] = model_vector
    
    for i in range(len(X_test)):
        model_vector = d2v_model.infer_vector(gensim.utils.simple_preprocess(X_test[i]))
        X_test[i] = model_vector
        
    return X_train, X_test, y_train, y_test

def generate_d2v_vectors_from_all_data(d2v_model,X):
    
    train_data = []
    
    for i in range(len(X)):
        model_vector = d2v_model.infer_vector(gensim.utils.simple_preprocess(X[i]))
        train_data.append(model_vector)
        
    return train_data

def run_classifier(classifier, X_train, X_test, y_train, y_test):
    
    classifier.fit(X_train,y_train)
    
    pred = classifier.predict(X_test)
    
    acc_score = accuracy_score(y_test,pred)
    ballanced_acc_score = balanced_accuracy_score(y_test, pred)
    
    return acc_score, ballanced_acc_score
  

#train_and_save()

X, y = util.read_corpus()  
#get_d2v_model('saved_models/doc2vec.model')
X = generate_d2v_vectors_from_all_data(get_d2v_model('saved_models/doc2vec.model'), X)
X_train, X_test, y_train, y_test = util.split_data_set(X,y)

'''
X, y = util.read_corpus()  
X_train, X_test, y_train, y_test = generate_d2v_vectors_from_splitted_data(get_d2v_model('saved_models/doc2vec.model'), X,y)
#X_train, X_test, y_train, y_test = generate_d2v_vectors_from_splitted_data(get_embedding_model('saved_models/skip_s100.txt'), X,y)
'''
classifier = LinearSVC()

acc_score, ballanced_acc_scores = run_classifier(classifier, X_train, X_test, y_train, y_test)

print(acc_score)
print(ballanced_acc_scores)


