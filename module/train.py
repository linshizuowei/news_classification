import os
import sys
import yaml
import json
import time

from sklearn import naive_bayes as nb
from sklearn.metrics import multilabel_confusion_matrix, classification_report

from .data_preprocess import Tokenizer
from .data_preprocess import StopWordDeleter
from .data_preprocess import TextPresenter


def build_dataset(config, train=True):
    data_file = config.get('data_config').get('train_data') if train else config.get('data_config').get('test_data')
    if not data_file:
        raise Exception('data file not exists: %s' % data_file)
    label = []
    data = []
    with open(data_file) as fd:
        for line in fd:
            llist = line.strip().split('\t')
            label.append(llist[0])
            data.append('\t'.join(llist[1:]))
    # tokenize
    print('=====>>> start tokenize')
    tokenizer = Tokenizer(config)
    data = tokenizer.tokenize(data)

    # filter stop words
    print('=====>>> start filter stop words')
    sw_filter = StopWordDeleter(config.get('stop_word_file'))
    data = [sw_filter.run(item) for item in data]

    # text representation
    print('=====>>> start text represent')
    text_presenter = TextPresenter(config)
    data = text_presenter.run(data)

    return label, data


def model_evaluate(model, config):
    test_label, test_data = build_dataset(config, train=False)
    predict_ret = model.predict(test_data)
    predict_ret = predict_ret.tolist()
    cmat = multilabel_confusion_matrix(test_label, predict_ret)
    report = classification_report(test_label, predict_ret)
    print('=====>> confusion matrix:')
    print(cmat)
    print('=====>> cls report:')
    print(report)


def train_model(config, dataset):
    label, data = dataset
    model_name = config['model_config']['model_name']
    if model_name == 'Native_Bayes_Bernoulli':
        model = nb.BernoulliNB()
        model.fit(data, label)
    elif model_name == 'Native_Bayes_Categorical':
        param = config['model_config'].get('model_parameter')
        model = nb.CategoricalNB(min_categories=param.get('min_categories'))
        model.fit(data, label)
    elif model_name == 'Native_Bayes_Complement':
        model = nb.ComplementNB()
        model.fit(data, label)
    elif model_name == 'Native_Bayes_Gaussian':
        model = nb.GaussianNB()
        model.fit(data, label)
    elif model_name == 'Native_Bayes_Multinomial':
        model = nb.MultinomialNB()
        model.fit(data, label)

    model_evaluate(model, config)

    print('===>>>doneeeeee')

def main(conf_file):
    # load config
    print('=====>> load confog: %s' % conf_file)
    with open(conf_file, encoding='utf-8') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    # build train and evaluate data
    print('=====>> build dataset')
    dataset = build_dataset(config)

    # train model
    print('=====>> train model')
    train_model(config, dataset)


if __name__ == '__main__':
    main('/Users/dianxiaonao/Work/NLP_work/news_classification/config.yaml')
