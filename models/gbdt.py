import numpy as np
import os
import json
import sys

from .decision_tree import DecisionTreeCartReg


class GradientBoostingDecisionTree(object):
    """
    """

    def __init__(self, config):
        self.tree_cnt = config.get('trees', 5)
        self.deepth = config.get('deepth', 3)
        self.leaf_nodes = config.get('leaf_nodes', 8)
        self.learning_rate = config.get('learning_rate', 0.1)

    def fit(self, data, label):
        # build first tree
        pass

        # build other trees

    def predict(self):
        pass

