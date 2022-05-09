import sys
import os
import jieba
import math
from collections import defaultdict


class JiebaCut(object):
    def __init__(self, config):
        self.cut_all = config.get('cut_all', True)
        self.cut_for_search = config.get('cut_for_search', False)
        self.user_dict = config.get('user_dict', None)
        if self.user_dict is not None:
            jieba.load_userdict(self.user_dict)

    def run(self, sentence):
        if self.cut_for_search:
            seg_list = jieba.cut_for_search(sentence)
        else:
            seg_list = jieba.lcut(sentence, cut_all=self.cut_all)

        return seg_list


class NgramToken(object):
    def __init__(self, granularity, stop_words_file=None):
        self.granularity = granularity
        if stop_words_file:
            with open(stop_words_file) as fd:
                self.stop_words = {line.strip() for line in fd}

    def run(self, data):
        # char level
        if self.granularity == '1-gram':
            return list(data)
        # # binary gram
        # elif self.granularity == '2-gram':
        #     for item in data:
        #         # delete punctuation
        #         for i in range(len(item)-1):
        #
        # tri-gram


class Tokenizer(object):
    def __init__(self, config):
        self.config = config

    def tokenize(self, data):
        result = []
        token_granularity = self.config.get('data_config').get('granularity')
        # 1-gram
        if token_granularity == '1-gram':
            tokenizer = NgramToken(token_granularity)
        # 2-gram
        elif token_granularity == '2-gram':
            tokenizer = NgramToken(token_granularity, self.config.get('stop_word_file'))
        # 3-gram
        elif token_granularity == '3-gram':
            tokenizer = NgramToken(token_granularity, self.config.get('stop_word_file'))
        # word
        elif token_granularity == 'jieba_segment':
            tokenizer = JiebaCut(config=self.config)

        for item in data:
            result.append(tokenizer.run(item))
        return result


class StopWordDeleter(object):
    def __init__(self, stop_word_file):
        self.stop_word_set = set()
        with open(stop_word_file) as fd:
            for line in fd:
                self.stop_word_set.add(line.strip())

    def run(self, word_list):
        filtered_word_list = []
        for word in word_list:
            if word not in self.stop_word_set:
                filtered_word_list.append(word)
        return filtered_word_list


class TextPresenter(object):
    def __init__(self, config):
        self.present_type = config.get('data_config').get('text_representation')
        vocabulary_file = config.get('data_config').get('vocabulary')
        self.vocab_dict = {}
        # word id start from 3, for 0,1,2 reprenting start token, end token and unknown token respectively
        with open(vocabulary_file) as fd:
            for idx, line in enumerate(fd):
                self.vocab_dict[line.strip()] = idx

    def run(self, dataset):
        result = []
        if self.present_type == 'one-hot':
            for data in dataset:
                item = [0] * len(self.vocab_dict)
                for w in data:
                    idx = self.vocab_dict.get(w, 2)
                    item[idx] = 1
                result.append(item)
        elif self.present_type == 'bow':
            for data in dataset:
                item = [0] * len(self.vocab_dict)
                for w in data:
                    idx = self.vocab_dict.get(w, 2)
                    item[idx] += 1
                result.append(item)
        elif self.present_type == 'tf-idf':
            # calculate idf
            idf_dict = defaultdict(int)
            for data in dataset:
                for w in set(data):
                    idf_dict[w] += 1
            # calculate tf-idf
            text_cnt = len(dataset)
            for data in dataset:
                itemdic = {w: 0 for w in data}
                for w in data:
                    itemdic[w] += 1
                item = [0] * len(self.vocab_dict)
                for w, cnt in itemdic.items():
                    idx = self.vocab_dict.get(w, 2)
                    tf = itemdic[w] / len(data)
                    idf = math.log(text_cnt / (idf_dict[w] + 1))
                    item[idx] = tf * idf
                result.append(item)
        elif self.present_type == 'w2v':
            pass

        return result
