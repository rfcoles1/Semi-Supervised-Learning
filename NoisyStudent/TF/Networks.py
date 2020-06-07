import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pickle
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as layers
from tensorflow.keras import backend as K

class Network():
    def __init__(self):
        self.batch_size = 4
        self.lr = 0.0001

        self.hist = defaultdict(list)
        self.curr_epoch = 0

    def predict(self, x_test):
        preds = self.Net.predict(x_test,
                        batch_size=self.batch_size,
                        verbose=0) 
        return preds
   
    def evaluate(self, x_test, y_test):
        return self.Net.evaluate(x_test,y_test)

    def history(self):
        return self.hist

    def save(self, path):
        f = open(self.dirpath + path + '.pickle', 'wb')
        pickle.dump(self.hist,f)
        f.close()
        self.Net.save_weights(self.dirpath + path + '.h5')

    def load(self, path):
        f = open(self.dirpath + path + '.pickle', 'rb')
        self.hist = pickle.load(f)
        f.close()
        self.Net.load_weights(self.dirpath + path + '.h5')

        self.curr_epoch = self.hist['iterations'][0][-1]
