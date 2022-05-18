import os
import sys
import json
import random
import yaml

from .data_preprocess import Tokenizer


def build_vocabulary(vocab_file):
    conf_file = '/Users/dianxiaonao/Work/NLP_work/news_classification/config.yaml'
    with open(conf_file, encoding='utf-8') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    tokenizer = Tokenizer(config)

    train_data = []
    with open(config['data_config']['train_data']) as fd:
        for line in fd:
            train_data.append('\t'.join(line.strip().split('\t')[1:]))
    train_data = tokenizer.tokenize(train_data)

    test_data = []
    with open(config['data_config']['test_data']) as fd:
        for line in fd:
            test_data.append('\t'.join(line.strip().split('\t')[1:]))
    test_data = tokenizer.tokenize(test_data)

    visited = set()
    with open(vocab_file, 'w') as out:
        out.write('CLS\nSEP\nUNK\n')
        cnt = 3
        for dset in [train_data, test_data]:
            for item in dset:
                for w in item:
                    w = w.strip()
                    if w and w not in visited:
                        cnt += 1
                        out.write(w + '\n')
                        visited.add(w)
    print('====>>> done, word count: ', cnt)
