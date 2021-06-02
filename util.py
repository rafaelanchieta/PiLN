import codecs
import shutil
from typing import Tuple, Generator

import gensim
import numpy
import pandas as pd
import rouge
from nltk import tokenize
from nltk.translate.bleu_score import sentence_bleu
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def read_corpus() -> Tuple[DataFrame, DataFrame]:
    ironia = pd.read_csv('data/ironia.csv', sep=';')
    nao_ironico = pd.read_csv('data/nao-ironico.csv', sep=';')
    return ironia, nao_ironico


def read_augumented(pt_file, label_file):
    texts, labels = [], []
    with codecs.open(pt_file, 'r', 'utf-8') as f:
        for line in f.readlines():
            texts.append(line)
    with open(label_file, 'r') as f:
        for line in f.readlines():
            labels.append(line.strip())
    return texts, labels


def get_snts(ref_file, hyp_file):
    ref_snts, hyp_snts = [], []
    with codecs.open(ref_file, 'r', 'utf-8') as ref_f, codecs.open(hyp_file, 'r', 'utf-8') as hyp_f:
        for ref, hyp in zip(ref_f, hyp_f):
            ref_snts.append(ref.split('\t')[2])
            # ref_snts.append(ref)
            hyp_snts.append(hyp)
    return ref_snts, hyp_snts


def join_files():
    ironia, nao_ironico = read_corpus()
    tweets = pd.concat([ironia, nao_ironico])
    labels = [1] * len(ironia['text'])
    labels.extend([0] * len(nao_ironico['text']))
    tweets['label'] = labels
    return tweets


def prepare_corpus(tweets: list) -> Generator:
    for i, line in enumerate(tweets):
        tokens = gensim.utils.simple_preprocess(line)
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def tokenizer(text):
    tokens = tokenize.word_tokenize(text, language='portuguese')
    tokens = tokens[:512-2]
    return tokens


def create_augumented_splits(in_file):
    lines = []
    with codecs.open(in_file, 'r', 'utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            idx += 1
            lines.append(line.split('\t')[2])
            if idx % 200 == 0:
                with codecs.open('augumented/' + str(idx) + '.txt', 'w', 'utf-8') as out:
                    out.writelines(lines)
                    out.write('\n')
                lines = []


def calculate_bleu(snt_refs, snt_hyps):
    bleu_scores = []
    for ref, hyp in zip(snt_refs, snt_hyps):
        bleu_scores.append(sentence_bleu([ref], hyp))
    return bleu_scores


def calculate_rouge(snt_refs, snt_hyps):
    rouge_scores = []
    for ref, hyp in zip(snt_refs, snt_hyps):
        evaluator = rouge.Rouge(metrics=['rouge-l'])
        scores = evaluator.get_scores(hyp, ref)
        # rouge_scores.append(scores[0]['rouge-l']['f'])
        rouge_scores.append(scores['rouge-l']['f'])
    return rouge_scores


def calculate_fscore(snt_refs, snt_hyps):
    f_scores = []
    for bleu, rouge in zip(calculate_bleu(snt_refs, snt_hyps), calculate_rouge(snt_refs, snt_hyps)):
        if bleu + rouge > 0.0:
            f_score = 2 * bleu * rouge / (bleu + rouge)
            f_scores.append(f_score)
        else:
            f_scores.append(0.0)
    return f_scores


def concat_files():
    with codecs.open('augumented/hyp-en.txt', 'w', 'utf-8') as out:
        for i in range(1, 23):
            with codecs.open('augumented/en-' + str(i) + '.txt', 'r', 'utf-8') as f:
                shutil.copyfileobj(f, out)
                out.write('\n')


def get_labels():
    labels = []
    with codecs.open('augumented/ref-en.txt', 'r', 'utf-8') as f:
        for line in f.readlines():
            labels.append(line.split('\t')[1])
    with open('augumented/labels.txt', 'w') as out:
        for line in labels:
            out.write(line)
            out.write('\n')


def read_corpus_data(corpus_file: str, sep: str) -> DataFrame:
    return pd.read_csv(corpus_file, sep=sep)


def create_splits_data(corpus_file: str, corpus_name: str) -> None:
    test_ratio = 0.1
    corpus = read_corpus_data(corpus_file, ',')
    x_train, x_test, y_train, y_test = train_test_split(corpus['text'].values, corpus['prediction'].values,
                                                        test_size=test_ratio, random_state=203)
    train = {'text': x_train, 'prediction': y_train}
    test = {'text': x_test, 'prediction': y_test}
    df_train = pd.DataFrame(train, columns=train.keys())
    df_train.to_csv('data/balanceado/90-10/train_media_' + corpus_name + '.csv', index=False, header=True)

    df_test = pd.DataFrame(test, columns=test.keys())
    df_test.to_csv('data/balanceado/90-10/test_media_' + corpus_name + '.csv', index=False, header=True)


def balancing_data(corpus_file: str, balancing_file: str):
    corpus = read_corpus_data(corpus_file, '\t')
    balanc = read_corpus_data(balancing_file, ',')
    print(corpus.groupby('prediction').count())
    print(balanc.f_score.describe())
    df = balanc.loc[(balanc['prediction'] == 0) & (balanc['f_score'] >= 0.79)][['text', 'prediction']]
    new = pd.concat([corpus, df], ignore_index=True)
    new.to_csv('data/balanceado/tweets_balanceado_75.csv', index=False, header=True)
    print(new.groupby('prediction').count())


def create_splits():
    # 80% 10% 10%
    # train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    corpus = join_files()
    x_train, x_test, y_train, y_test = train_test_split(corpus['text'].values, corpus['label'].values,
                                                        test_size=test_ratio, random_state=2021)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_ratio / (1 - test_ratio),
                                                          random_state=2021)

    train = {'text': x_train, 'label': y_train}
    valid = {'text': x_valid, 'label': y_valid}
    test = {'text': x_test, 'label': y_test}

    df = pd.DataFrame(train, columns=train.keys())
    df.to_csv('data/train.csv', index=False, header=True)
    df = pd.DataFrame(valid, columns=valid.keys())
    df.to_csv('data/valid.csv', index=False, header=True)
    df = pd.DataFrame(test, columns=test.keys())
    df.to_csv('data/test.csv', index=False, header=True)

    # train, valid, test = numpy.split(corpus.sample(frac=1), [int(.6 * len(corpus)), int(.8 * len(corpus))])
    # print(len(train))
    # print(len(valid))
    # print(len(test))


if __name__ == '__main__':
    # corpus = read_corpus_data('data/70-30/test_news.csv', ',')
    # corpus = read_corpus_data('data/test_news.csv', '\t')
    # corpus = read_corpus_data('data/balanceado/tweets_balanceado_75.csv', ',')
    # print(corpus.groupby('prediction').count())
    balanc = read_corpus_data('data/semeval-pt.csv', ',')
    print(balanc.f_score.describe())
    print(balanc.groupby('prediction').count())
    # df = balanc.loc[(balanc['prediction'] == 0) & (balanc['f_score'] >= 0.79)][['text', 'prediction']]
    # print(df)
    # print(corpus.groupby('prediction').count())
    # ref_snts, hyp_snts = get_snts('augumented/ref-en.txt', 'augumented/hyp-en.txt')
    # f_scores = calculate_fscore(ref_snts, hyp_snts)
    # print(numpy.mean(f_scores))
    # print(numpy.min(f_scores))
    # print(numpy.std(f_scores))
    # print(numpy.quantile(f_scores, 0.75))
    # print(f_scores)
    # texts, labels = read_augumented('augumented/semeval-pt.txt', 'augumented/labels.txt')
    # df = DataFrame({'text': texts, 'label': labels, 'f_score': f_scores})
    # df.to_csv('data/semeval-pt.csv', index=False, header=True)
    # create_splits_data('data/balanceado/tweets_balanceado_media.csv', 'tweets')
    # create_splits_data('data/training_news.csv', 'news')
    # read_corpus_data('data/90-10/test_tweets.csv')
    # balancing_data('data/training_tweets.csv', 'data/semeval-pt.csv')

    # count = len([i for i in f_scores if i >= 0.89])
    # print(f'Maior que: 89 {count}')
    # tweets = pd.read_csv('data/training_tweets.csv', sep='\t')
    # news = pd.read_csv('data/training_news.csv', sep='\t')
    # print(tweets.groupby('prediction').count())
    # print(news.groupby('prediction').count())
    # pass
    # concat_files()
    # get_labels()
    # create_augumented_splits('augumented/ref-en.txt')
