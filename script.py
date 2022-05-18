import os
import sys
import json
import random
import yaml

from module.train import main

def split_and_reformat():
    """
    two jobs will be finish in this function,
    1. split train data file and test data file
    2. reformat data file to standard format: label\tdata\n
    """
    test_cnt = 100
    sample_extract = 1000

    # tou tiao dataset
    # dataset format:
    # text id_!_class id_!_class name_!_origin text_!_keyword
    # delimiter: _!_
    # file_path = '/Users/dianxiaonao/Work/NLP_work/news_classification/dataset/toutiao_cat_data.txt'
    file_path = '/Users/dianxiaonao/Work/NLP_work/news_classification/dataset/toutiao_test_cat_data.txt'
    train_file = '/Users/dianxiaonao/Work/NLP_work/news_classification/dataset/toutiao_train_data.txt'
    test_file = '/Users/dianxiaonao/Work/NLP_work/news_classification/dataset/toutiao_test_data.txt'
    with open(file_path) as fd:
        dataset = fd.readlines()[:sample_extract]
        if test_cnt >= len(dataset):
            test_cnt = int(len(dataset) * 0.1)
        random.shuffle(dataset)
        test_dataset = dataset[:test_cnt]
        with open(test_file, 'w') as out:
            for line in test_dataset:
                llist = line.strip().split('_!_')
                out.write('%s\t%s\n' % (llist[2], llist[3]))
        print('=====> %s test data saved done' % test_cnt)
        del test_dataset
        with open(train_file, 'w') as out:
            for line in dataset[test_cnt:]:
                llist = line.strip().split('_!_')
                out.write('%s\t%s\n' % (llist[2], llist[3]))
        print('=====> %s train data saved done' % (len(dataset)-test_cnt))


def labelmap():
    data_file = '/Users/dianxiaonao/Work/NLP_work/news_classification/dataset/toutiao_cat_data.txt'
    lm_file = '/Users/dianxiaonao/Work/NLP_work/news_classification/dataset/toutiao_label2id.txt'
    with open(lm_file, 'w') as out:
        mdict = {}
        with open(data_file) as fd:
            for line in fd:
                llist = line.strip().split('_!_')
                if llist[2] not in mdict:
                    mdict[llist[2]] = llist[1]
        for k, v in mdict.items():
            out.write('%s\t%s\n' % (k, v))



if __name__ == '__main__':
    # labelmap()
    # split_and_reformat()
    main('/Users/dianxiaonao/Work/NLP_work/news_classification/config.yaml')
