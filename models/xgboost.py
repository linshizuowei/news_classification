import time
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Xgboost(object):
    def __init__(self, config):
        if config['is_classification']:
            label = []
            ori_file = config['data_config']['origin_data']
            with open(ori_file) as fd:
                for line in fd:
                    llist = line.strip().split('_!_')
                    label.append(llist[2])
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(label)

        self.labelmap_file = config['label2id_file']
        self.param = config['model_config'].get('model_parameter')
        self.epoch = config['train_config'].get('epoch')
        self.early_stopping_rounds = config['train_config'].get('early_stopping_rounds')
        self.bst = None

    def fit(self, dataset, label):
        start = time.time()
        print('=====>>>start build xgboost dataset')
        data = np.asarray(dataset)
        # with open(self.labelmap_file) as fd:
        #     l2id = {line.split()[0]: line.split()[1] for line in fd}
        #     label = [l2id[it] for it in label]
        # label = np.asarray(label)
        label_encoded = self.label_encoder.transform(label)
        dtrain = xgb.DMatrix(data, label=label_encoded)
        print('=====>>>xgboost dataset build done, cost: %.2fs' % (time.time()-start))

        self.bst = xgb.train(self.param, dtrain, self.epoch, early_stopping_rounds=self.early_stopping_rounds)

    def predict(self, testdata):
        dtest = xgb.DMatrix(testdata)
        ypred = self.bst.predict(dtest, iteration_range=(0, self.bst.best_iteration + 1))
        return ypred
