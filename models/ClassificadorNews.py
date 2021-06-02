import pickle


class ClassificadorNews:
    def __init__(self):
        with open('models/ModeloTweetsTF-IDF.pickle', 'rb') as file_pickle:
            self.clf = pickle.loads(file_pickle.read())
            file_pickle.close()
    
    def predict(self, texto):
        if self.clf.predict([texto]):
            return 'Ironia'
        else:
            return 'Não irônico'
