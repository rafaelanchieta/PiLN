import argparse

from models.ClassificadorNews import ClassificadorNews
from models.ClassificadorTweet import ClassificadorTweet
from models.doc2vec import Ironia


def main(data):
    if data.model == 'superficial' and data.corpus == 'twitter':
        print(ClassificadorTweet().predict(data.sentence))
    elif data.model == 'superficial' and data.corpus == 'news':
        print(ClassificadorNews().predict(data.sentence))
    elif data.model == 'embeddings' and data.corpus == 'twitter':
        print(Ironia(data.corpus, data.sentence))
    elif data.model == 'embeddings' and data.corpus == 'news':
        print(Ironia(data.corpus, data.sentence))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Irony Detection for Portuguese',
                                   epilog='Usage: python irony_detection.py -m model -t text')
    args.add_argument('-m', '--model', help='Trained model', required=True)
    args.add_argument('-c', '--corpus', help='Text type', required=True)
    args.add_argument('-t', '--sentence', help='Text to be classified', required=True)
    main(args.parse_args())
