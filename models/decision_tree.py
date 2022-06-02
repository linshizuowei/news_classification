import os
import sys
import numpy as np
import math

from collections import Counter


class TreeNode(object):
    def __init__(self, val=None):
        self.val = val
        self.children = {}
        self.cls = None


class DecisionTree(object):
    def __init__(self):
        self.feature_map = {}
        self.root = TreeNode()

    def fit(self, data, label):
        """

        Args:
            data: np.ndarray, shape: [samples, features], train data set.
            label: np.ndarray, shape: [samples,], label of train data.

        Returns:

        """
        self.initialize_feature_map(data)
        self.build_tree(self.root, data, label)

    def build_tree(self, parent_node, data, label):
        """

        Args:
            parent_node: TreeNode
            data: np.ndarray, shape: [samples, features], train data set.
            label: np.ndarray, shape: [samples,], label of train data.

        Returns:

        """

        # stop split node or not
        if len(set(label)) == 1:
            parent_node.cls = set(label).pop()
            return

        # find split feature
        feature_name = self.search_split_feature(data, label)

        # initialize tree node of feature values
        feature_vals = self.feature_map[feature_name]
        for val in feature_vals:
            node = TreeNode(val)
            parent_node.children[val] = node

        # build tree recursively
        for child in parent_node.children:
            child_node = parent_node.children[child]
            child_index = np.squeeze(np.argwhere(data[:, feature_name] == child))
            child_data = data[child_index]
            child_label = label[child_index]
            self.build_tree(child_node, child_data, child_label)

    def search_split_feature(self, data, label):
        """
        1. calculate entropy
        2. calculate conditional entropy of features
        3. select feature of max Gain(D, a)
        Args:
            data:
            label:

        Returns:

        """
        raise NotImplementedError()

    def cal_entropy(self, samples):
        """

        Args:
            samples:

        Returns:

        """

        assert len(samples.shape) == 1
        cnt_dict = Counter(samples)
        total = samples.shape[0]
        entropy = 0.
        for k in cnt_dict:
            prob_k = cnt_dict[k] / total
            entropy += (prob_k * math.log2(prob_k))
        return entropy * -1

    def cal_con_entropy(self, feature, label):
        """

        Args:
            feature:
            label:

        Returns:

        """

        assert len(feature.shape) == 1
        sample_cnt = feature.shape[0]
        value_cnt = Counter(feature)
        conditional_ent = 0
        for val, cnt in value_cnt.items():
            ind = np.argwhere(feature == val)
            sub_label = label[ind]
            sub_entropy = self.cal_entropy(sub_label)
            conditional_ent += cnt / sample_cnt * sub_entropy
        return conditional_ent

    def cal_IV(self, feature, label):
        """

        Args:
            feature:
            label:

        Returns:

        """

        assert len(feature.shape) == 1
        sample_cnt = feature.shape[0]
        value_cnt = Counter(feature)
        feature_iv = 0.
        for val, cnt in value_cnt.items():
            feature_iv += cnt / sample_cnt * math.log2(cnt/sample_cnt)
        return feature_iv * -1

    def initialize_feature_map(self, data):
        """

        Args:
            data: np.ndarray, shape: [samples, features]

        Returns:

        """
        assert len(data.shpae) == 2
        fcnt = data.shape[1]
        for col in range(fcnt):
            feature_set = set(data[:, col])
            self.feature_map[col] = feature_set

    def predict(self, data):
        """

        Args:
            data: np.ndarray, shape: [samples, features], data to be predicted.

        Returns:
            result: np.ndarray, shape: [samples, ], prediction result.

        """

        raise NotImplementedError()


class DecisionTreeID3(DecisionTree):
    """
    implementation of algorithm of ID3 of decision tree
    Notes:
        1、ID3算法只能应用于类别型或枚举型数据，用于连续型数据也可计算，效果应该会较差
        2、由于要划分属性并构建树，所以需要知道各属性(即特征)的名字，因此输入数据的格式可以dict或者array传入；
            2.1 dict格式：
                直接以key值作为属性名称
                data: dict, {feature1: [1, ..., samples], feature2: []}
                label: np.ndarray, shape: [samples,]
            2.2 array格式：
                以列序号作为属性名称
                data: np.ndarray, shape: [samples, features]
                label: np.ndarray, shape: [samples,]
        3、目前使用array格式，故测试集特征顺序要与训练集特征顺序一致。
    step:
        统计各属性的值，给属性命名
        初始化根节点
        寻找划分属性
        构建子节点
        为每个子节点迭代属性划分
    """

    def search_split_feature(self, data, label):
        """ID3 split feature
        1. calculate entropy
        2. calculate conditional entropy of features
        3. select feature of max Gain(D, a)
        Args:
            data:
            label:

        Returns:

        """

        # calculate entropy of samples
        entropy = self.cal_entropy(label)

        # calculate conditional entropy of features
        features_con_entropy = {}
        for fea in self.feature_map:
            con_entropy = self.cal_con_entropy(data[:, fea], label)
            features_con_entropy[fea] = con_entropy

        # select feature of max Gain(D, a)
        max_gain = float('-inf')
        selected_feature = None
        for fea in self.feature_map:
            gain = entropy - features_con_entropy[fea]
            if gain > max_gain:
                selected_feature = fea

        return selected_feature

class DecisionTreeC45(DecisionTree):
    """

    """

    def search_split_feature(self, data, label):
        """C4.5 split feature
        1. calculate entropy
        2. calculate conditional entropy of features
        3. select feature of max Gain(D, a)
        Args:
            data:
            label:

        Returns:

        """

        # calculate entropy of samples
        entropy = self.cal_entropy(label)

        # calculate conditional entropy of features
        features_con_entropy = {}
        for fea in self.feature_map:
            con_entropy = self.cal_con_entropy(data[:, fea], label)
            features_con_entropy[fea] = con_entropy

        # select feature of max Gain(D, a)
        feature_rates = {}
        for fea in self.feature_map:
            gain = entropy - features_con_entropy[fea]
            iv = self.cal_IV(data[:, fea], label)
            gain_ratio = gain / iv
            feature_rates[fea] = [gain, gain_ratio]

        max_gain_ratio = float('-inf')
        selected_feature = None
        gain_list = [it[0] for it in feature_rates.values()]
        avg_gain = sum(gain_list) / len(gain_list)
        for fea in self.feature_map:
            gain, gain_ratio = feature_rates[fea]
            # select feature whose gain is larger than average gain
            if gain <= avg_gain:
                continue
            # select feature whose gain ratio is largest than others
            if gain_ratio > max_gain_ratio:
                max_gain_ratio = gain_ratio
                selected_feature = fea

        return selected_feature
