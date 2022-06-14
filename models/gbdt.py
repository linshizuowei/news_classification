import numpy as np
import os
import json
import sys

from .decision_tree import DecisionTreeCartReg
from ,decision_tree import CartTreeNode


class GradientBoostingDecisionTree(object):
    """
    """

    def __init__(self, config):
        self.tree_nums = config.get('tree_nums', 5)
        self.deepth = config.get('deepth', 3)
        self.leaf_nodes = config.get('leaf_nodes', 8)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.tree_list = []

    def fit(self, data, label):
        # build first tree
        self.build_first_tree(label)

        # build other trees
        self.build_trees(data, label)

    def predict(self):
        pass

    def build_first_tree(self, label):
        mean = np.mean(label)
        tree = CartTreeNode(mean)
        self.tree_list.append(tree)

    def build_trees(self, data, label):
        pass
