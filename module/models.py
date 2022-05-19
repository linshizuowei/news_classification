import time
import xgboost as xgb
import numpy as np

class Xgboost(object):
    def __init__(self, config):
        self.labelmap_file = config['label2id_file']
        self.param = config['model_config'].get('model_parameter')
        self.epoch = config['train_config'].get('epoch')
        self.early_stopping_rounds = config['train_config'].get('early_stopping_rounds')
        self.bst = None

    def fit(self, dataset, label):
        start = time.time()
        print('=====>>>start build xgboost dataset')
        data = np.asarray(dataset)
        with open(self.labelmap_file) as fd:
            l2id = {line.split()[0]: line.split()[1] for line in fd}
            label = [l2id[it] for it in label]
        label = np.asarray(label)
        dtrain = xgb.DMatrix(data, label=label)
        print('=====>>>xgboost dataset build done, cost: %.2fs' % (time.time()-start))

        self.bst = xgb.train(self.param, dtrain, self.epoch, early_stopping_rounds=self.early_stopping_rounds)

    def predict(self, testdata):
        dtest = xgb.DMatrix(testdata)
        ypred = self.bst.predict(dtest, iteration_range=(0, self.bst.best_iteration + 1))
        return ypred
