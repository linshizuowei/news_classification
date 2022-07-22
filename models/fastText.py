import os
import sys
import json


class FastTextModel(object):
    """
    fastText的关键点在于Word embedding和subword embedding，subword embedding是在
    单词内的n-gram，需要注意的是fastText并没有利用单词间的n-gram，这在英语中是适用
    的，但在汉语中，句子都是字符组成的，就只存在字符间的n-gram，不存在字符内的n-gram了。

    """
    def __init__(self):
        pass

    def fit(self, data, label):
        pass

    def predict(self, data):
        pass


