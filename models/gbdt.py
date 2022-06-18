import numpy as np
import os
import json
import math
import sys
from collections import Counter

from .decision_tree import DecisionTreeCartReg
from .decision_tree import CartTreeNode


class GradientBoostingDecisionTree(object):
    """
    """

    def __init__(self, config):
        self.tree_nums = config.get('tree_nums', 5)
        self.deepth = config.get('deepth', 3)
        self.leaf_nodes = config.get('leaf_nodes', 8)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.tree_list = []
        self.Fout = 0

    def fit(self, data, label):
        # build tree
        self.build_trees(data, label)

    def predict(self):
        pass

    def build_trees(self, data, label):
        pass


class GradientBoostingDecisionTreeRegressor(GradientBoostingDecisionTree):
    def build_trees(self, data, label):
        # build first tree
        mean = np.mean(label)
        self.Fout += mean
        residual = label - self.Fout
        tree = CartTreeNode(mean)
        self.tree_list.append(tree)

        # build left trees
        for i in range(1, self.tree_nums):
            cartree = DecisionTreeCartReg()
            cartree.fit(data, residual)
            pred = cartree.predict(data)
            self.Fout += pred * self.learning_rate
            residual = label - self.Fout
            self.tree_list.append(cartree)


class GradientBoostingDecisionTreeClassifier(GradientBoostingDecisionTree):
    def build_trees(self, data, label):
        # build first tree
        cnt = Counter(label)
        log_odds = math.log2(cnt[1]/cnt[0])
        pred = 1 / (1 + math.e ** (-1 * log_odds))
        self.Fout += pred
        residual = label - self.Fout
        tree = CartTreeNode(pred)
        self.tree_list.append(tree)

        # build left trees
        for i in range(1, self.tree_nums):
            cartree = DecisionTreeCartReg()
            cartree.fit(data, residual)
            pred = cartree.predict(data)


